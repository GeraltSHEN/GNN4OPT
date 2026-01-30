#!/usr/bin/env python3
"""Find samples that break candidate cleaning and write filename lists."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import load_gzip  # noqa: E402

DATASET_NAMES = ("set_cover", "cauctions", "facilities", "indset")
VARIABLE_REQUIRED = [
    "type_0",
    "type_1",
    "type_2",
    "type_3",
    "has_lb",
    "has_ub",
    "sol_is_at_lb",
    "sol_is_at_ub",
    "sol_frac",
    "coef_normalized",
    "sol_val",
]


def _read_cfg(path: Path) -> Dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _select_cfg(config_root: Path, dataset: str, cfg_idx: Optional[int]) -> Optional[Path]:
    if cfg_idx is not None:
        cfg_path = config_root / f"{dataset}_{cfg_idx}"
        return cfg_path if cfg_path.exists() else None

    matches = sorted(config_root.glob(f"{dataset}_*"))
    if not matches:
        return None

    def _sort_key(p: Path) -> Tuple[int, str]:
        suffix = p.name[len(dataset) + 1 :]
        try:
            return (int(suffix), p.name)
        except ValueError:
            return (10**9, p.name)

    return sorted(matches, key=_sort_key)[0]


def _build_variable_features(variable_dict: Dict, use_default_features: bool) -> Tuple[torch.Tensor, Dict[str, int]]:
    variable_names = variable_dict["names"]
    variable_default_features = torch.as_tensor(variable_dict["values"], dtype=torch.float32)
    if use_default_features:
        variable_features = variable_default_features
        variable_feature_indices = {name: variable_names.index(name) for name in variable_names}
    else:
        missing = [name for name in VARIABLE_REQUIRED if name not in variable_names]
        if missing:
            raise KeyError(f"Missing required variable features: {missing}")
        variable_features = torch.stack(
            [variable_default_features[:, variable_names.index(name)] for name in VARIABLE_REQUIRED],
            axis=-1,
        )
        variable_feature_indices = {
            name: new_index for name, new_index in zip(VARIABLE_REQUIRED, range(len(VARIABLE_REQUIRED)))
        }
    return variable_features, variable_feature_indices


def _check_sample(sample_path: Path, use_default_features: bool) -> Optional[str]:
    sample = load_gzip(sample_path)
    sample_state, _, sample_action, sample_action_set, sample_scores = sample["data"]

    _, _, variable_dict = sample_state
    variable_features, variable_feature_indices = _build_variable_features(
        variable_dict, use_default_features
    )

    candidates = torch.as_tensor(sample_action_set, dtype=torch.int64)
    candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)
    _ = candidate_scores  # kept for parity with utils.py, not needed for checks
    candidate_choice_node_id = int(sample_action)

    sol_is_not_at_lb = (
        variable_features[candidates, variable_feature_indices["sol_is_at_lb"]] == 0
    )
    sol_is_not_at_ub = (
        variable_features[candidates, variable_feature_indices["sol_is_at_ub"]] == 0
    )
    sol_is_not_at_lub = sol_is_not_at_lb & sol_is_not_at_ub
    cleaned_candidates = candidates[sol_is_not_at_lub]

    if cleaned_candidates.numel() < 1:
        return "all_cand_soln_at_lub"
    if not torch.any(cleaned_candidates == candidate_choice_node_id):
        return "all_cand_equally_bad"
    return None


def _iter_samples(dataset_root: Path, split_name: str, file_pattern: str) -> Iterable[Path]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(file_pattern))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find samples that break candidate cleaning and write filename lists.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("./cfg"),
        help="Directory containing configuration files.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_NAMES),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--cfg-idx",
        type=int,
        default=None,
        help="Optional cfg index to select per dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("bad_samples"),
        help="Output root for bad sample lists.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    config_root = (repo_root / args.config_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        print("No datasets provided.")
        return 1

    for dataset in datasets:
        cfg_path = _select_cfg(config_root, dataset, args.cfg_idx)
        if cfg_path is None:
            print(f"[skip] no config found for dataset '{dataset}' in {config_root}")
            continue

        cfg = _read_cfg(cfg_path)
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            print(f"[skip] dataset_path missing for '{dataset}' in {cfg_path}")
            continue

        dataset_root = (repo_root / dataset_path).resolve()
        if not dataset_root.exists():
            print(f"[skip] dataset root does not exist: {dataset_root}")
            continue

        file_pattern = cfg.get("file_pattern", "sample_*.pkl")
        train_split = cfg.get("train_split", "train")
        val_split = cfg.get("val_split", "valid")
        test_split = cfg.get("test_split", "test")
        use_default_features = bool(cfg.get("use_default_features", True))

        bad: Dict[str, List[str]] = {
            "all_cand_soln_at_lub": [],
            "all_cand_equally_bad": [],
        }

        split_names = [train_split, val_split, test_split]
        for split_name in split_names:
            sample_files = _iter_samples(dataset_root, split_name, file_pattern)
            for sample_path in sample_files:
                try:
                    reason = _check_sample(sample_path, use_default_features)
                except Exception as exc:
                    print(f"[error] {dataset} {split_name} {sample_path}: {exc}")
                    continue
                if reason is None:
                    continue
                try:
                    rel_path = sample_path.relative_to(dataset_root)
                    bad[reason].append(rel_path.as_posix())
                except ValueError:
                    bad[reason].append(str(sample_path))

        for reason, entries in bad.items():
            out_dir = output_root / dataset / reason
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "bad_file_name_list"
            with out_path.open("w") as fh:
                for name in sorted(entries):
                    fh.write(f"{name}\n")
            print(f"[{dataset}] {reason}: {len(entries)} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
