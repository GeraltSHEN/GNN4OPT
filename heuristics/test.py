"""Executable test script for anchor strong branching."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml

try:
    from .anchor_strong_branching import run_anchor_strong_branching
except Exception:  # pragma: no cover - script execution fallback
    from anchor_strong_branching import run_anchor_strong_branching


def _load_parent_utils_module(project_root: Path):
    utils_path = project_root / "utils.py"
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    spec = importlib.util.spec_from_file_location("project_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load project utils module from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_split_subdir(cfg: dict, split: str) -> str:
    key_map = {
        "train": "train_split",
        "valid": "val_split",
        "test": "test_split",
    }
    default_map = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }
    return str(cfg.get(key_map[split], default_map[split]))


def _resolve_sample_files(cfg: dict, split: str, max_samples: int | None) -> list[Path]:
    dataset_root = Path(cfg["dataset_path"])
    split_subdir = _resolve_split_subdir(cfg, split)
    split_dir = dataset_root / split_subdir
    file_pattern = cfg.get("file_pattern", "sample_*.pkl")

    sample_files: list[Path] | None = None
    subsamples = cfg.get("subsamples", None)
    if subsamples is not None and str(subsamples).strip().lower() not in {"", "none", "null"}:
        subsample_root = Path(str(subsamples))
        if not subsample_root.is_absolute():
            subsample_root = dataset_root.parent / subsample_root
        manifest = subsample_root / split_subdir / "sample_files.txt"
        if manifest.exists():
            sample_files = []
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = split_dir / p
                sample_files.append(p)

    if sample_files is None:
        sample_files = sorted(split_dir.glob(file_pattern))

    if max_samples is not None:
        sample_files = sample_files[: int(max_samples)]

    return sample_files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate anchor strong branching heuristic.")
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/set_cover_50",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of anchor candidates sampled per instance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample cap (after subsample manifest if used).",
    )
    parser.add_argument(
        "--verbose_every",
        type=int,
        default=20,
        help="Print progress every N samples.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "dataset_path" not in cfg:
        raise KeyError(f"'dataset_path' is missing in {cfg_path}")

    sample_files = _resolve_sample_files(cfg, split=args.split, max_samples=args.max_samples)
    if not sample_files:
        raise RuntimeError("No sample files found for the requested split.")

    project_root = Path(__file__).resolve().parents[1]
    project_utils = _load_parent_utils_module(project_root)
    GraphDataset = project_utils.GraphDataset

    # No pairwise features are needed for this heuristic evaluation.
    cfg_for_dataset = dict(cfg)
    cfg_for_dataset["two_fwl"] = False
    dataset_args = argparse.Namespace(**cfg_for_dataset)

    dataset = GraphDataset(
        sample_files,
        edge_nfeats=int(cfg_for_dataset.get("edge_nfeats", 1)),
        two_fwl=False,
        args=dataset_args,
    )

    rng = np.random.default_rng(args.seed)
    tol = 1e-8

    evaluated = 0
    matched = 0
    failures = 0
    total_gap = 0.0
    total_candidates = 0
    anchor_contains_best = 0

    use_default_features = bool(cfg_for_dataset.get("use_default_features", True))

    for i in range(len(dataset)):
        graph = dataset.get(i)

        try:
            result = run_anchor_strong_branching(
                graph,
                k=args.k,
                rng=rng,
                use_default_features=use_default_features,
            )
        except Exception as exc:
            failures += 1
            if args.verbose_every > 0 and (i + 1) % args.verbose_every == 0:
                print(f"[{i + 1}/{len(dataset)}] failed: {exc}")
            continue

        truth_scores = graph.candidate_scores.detach().cpu().numpy()
        best_score = float(np.max(truth_scores))
        best_positions = np.flatnonzero(truth_scores >= best_score - tol)

        pred_pos = int(result.selected_candidate_pos)
        pred_score = float(truth_scores[pred_pos])
        is_match = bool(pred_pos in best_positions)

        if is_match:
            matched += 1
        total_gap += (best_score - pred_score)
        total_candidates += int(graph.nb_candidates)

        anchor_globals = set(result.anchor_candidate_global_indices.tolist())
        best_globals = set(graph.candidates[best_positions].detach().cpu().numpy().tolist())
        if bool(anchor_globals & best_globals):
            anchor_contains_best += 1

        evaluated += 1

        if args.verbose_every > 0 and (evaluated % args.verbose_every == 0):
            acc = matched / evaluated
            avg_gap = total_gap / evaluated
            print(
                f"[{evaluated}/{len(dataset)}] "
                f"acc={acc:.4f} avg_gap={avg_gap:.4f} failures={failures}"
            )

    if evaluated == 0:
        raise RuntimeError("No samples were successfully evaluated.")

    print("\nAnchor Strong Branching Summary")
    print(f"config: {cfg_path}")
    print(f"split: {args.split}")
    print(f"samples_requested: {len(dataset)}")
    print(f"samples_evaluated: {evaluated}")
    print(f"failures: {failures}")
    print(f"k_anchors: {args.k}")
    print(f"top1_match: {matched} / {evaluated} = {matched / evaluated:.6f}")
    print(f"avg_truth_score_gap: {total_gap / evaluated:.6f}")
    print(f"avg_num_candidates: {total_candidates / evaluated:.2f}")
    print(
        "anchor_contains_best: "
        f"{anchor_contains_best} / {evaluated} = {anchor_contains_best / evaluated:.6f}"
    )


if __name__ == "__main__":
    main()
