"""Executable test script for anchor strong branching."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    else:
        # Enforce deterministic non-shuffled traversal even with manifest inputs.
        sample_files = sorted(sample_files)

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
        choices=["train", "valid", "test", "all"],
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
        default=1000,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="heuristics/results/anchor_sb_summary.json",
        help="Path to save JSON summary.",
    )
    return parser.parse_args(argv)


def _build_summary(
    *,
    cfg_path: Path,
    split: str,
    samples_requested: int,
    evaluated: int,
    failures: int,
    k_anchors: int,
    matched: int,
    total_gap: float,
    total_candidates: int,
    anchor_contains_best: int,
    run_status: str = "ok",
    error: str | None = None,
) -> dict:
    top1_ratio = (matched / evaluated) if evaluated > 0 else 0.0
    avg_gap = (total_gap / evaluated) if evaluated > 0 else None
    avg_candidates = (total_candidates / evaluated) if evaluated > 0 else None
    anchor_contains_best_ratio = (anchor_contains_best / evaluated) if evaluated > 0 else 0.0
    return {
        "run_status": run_status,
        "error": error,
        "config": str(cfg_path),
        "split": split,
        "samples_requested": int(samples_requested),
        "samples_evaluated": int(evaluated),
        "failures": int(failures),
        "k_anchors": int(k_anchors),
        "top1_match": {
            "count": int(matched),
            "denom": int(evaluated),
            "ratio": float(top1_ratio),
        },
        "avg_truth_score_gap": None if avg_gap is None else float(avg_gap),
        "avg_num_candidates": None if avg_candidates is None else float(avg_candidates),
        "anchor_contains_best": {
            "count": int(anchor_contains_best),
            "denom": int(evaluated),
            "ratio": float(anchor_contains_best_ratio),
        },
    }


def _save_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _print_summary_block(summary: dict) -> None:
    evaluated = int(summary["samples_evaluated"])
    matched = int(summary["top1_match"]["count"])
    anchor_contains = int(summary["anchor_contains_best"]["count"])

    print("\nAnchor Strong Branching Summary")
    print(f"config: {summary['config']}")
    print(f"split: {summary['split']}")
    print(f"samples_requested: {summary['samples_requested']}")
    print(f"samples_evaluated: {summary['samples_evaluated']}")
    print(f"failures: {summary['failures']}")
    print(f"k_anchors: {summary['k_anchors']}")
    print(f"top1_match: {matched} / {evaluated} = {summary['top1_match']['ratio']:.6f}")
    if summary["avg_truth_score_gap"] is not None:
        print(f"avg_truth_score_gap: {summary['avg_truth_score_gap']:.6f}")
    else:
        print("avg_truth_score_gap: null")
    if summary["avg_num_candidates"] is not None:
        print(f"avg_num_candidates: {summary['avg_num_candidates']:.2f}")
    else:
        print("avg_num_candidates: null")
    print(
        "anchor_contains_best: "
        f"{anchor_contains} / {evaluated} = {summary['anchor_contains_best']['ratio']:.6f}"
    )
    if summary["run_status"] != "ok":
        print(f"run_status: {summary['run_status']}")
        print(f"error: {summary['error']}")


def _evaluate_split(
    *,
    cfg: dict,
    cfg_path: Path,
    split: str,
    args,
    GraphDataset,
    rng_seed: int,
) -> tuple[dict, Exception | None]:
    sample_files = _resolve_sample_files(cfg, split=split, max_samples=args.max_samples)
    if not sample_files:
        summary = _build_summary(
            cfg_path=cfg_path,
            split=split,
            samples_requested=0,
            evaluated=0,
            failures=0,
            k_anchors=args.k,
            matched=0,
            total_gap=0.0,
            total_candidates=0,
            anchor_contains_best=0,
            run_status="error",
            error="RuntimeError: No sample files found for the requested split.",
        )
        return summary, RuntimeError("No sample files found for the requested split.")

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
    # This evaluator is intentionally sequential (effective batch size = 1).

    rng = np.random.default_rng(rng_seed)
    tol = 1e-8
    use_default_features = bool(cfg_for_dataset.get("use_default_features", True))

    evaluated = 0
    matched = 0
    failures = 0
    total_gap = 0.0
    total_candidates = 0
    anchor_contains_best = 0

    try:
        for i in range(len(dataset)):
            processed = i + 1
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
                if args.verbose_every > 0 and processed % args.verbose_every == 0:
                    print(f"[{split} {processed}/{len(dataset)}] failed: {exc}")
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

            if args.verbose_every > 0 and processed % args.verbose_every == 0:
                acc = (matched / evaluated) if evaluated > 0 else 0.0
                avg_gap = (total_gap / evaluated) if evaluated > 0 else float("nan")
                print(
                    f"[{split} {processed}/{len(dataset)}] "
                    f"evaluated={evaluated} acc={acc:.4f} "
                    f"avg_gap={avg_gap:.4f} failures={failures}"
                )

        if evaluated == 0:
            raise RuntimeError("No samples were successfully evaluated.")

        summary = _build_summary(
            cfg_path=cfg_path,
            split=split,
            samples_requested=len(dataset),
            evaluated=evaluated,
            failures=failures,
            k_anchors=args.k,
            matched=matched,
            total_gap=total_gap,
            total_candidates=total_candidates,
            anchor_contains_best=anchor_contains_best,
            run_status="ok",
        )
        return summary, None
    except Exception as exc:
        summary = _build_summary(
            cfg_path=cfg_path,
            split=split,
            samples_requested=len(dataset),
            evaluated=evaluated,
            failures=failures,
            k_anchors=args.k,
            matched=matched,
            total_gap=total_gap,
            total_candidates=total_candidates,
            anchor_contains_best=anchor_contains_best,
            run_status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        return summary, exc


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "dataset_path" not in cfg:
        raise KeyError(f"'dataset_path' is missing in {cfg_path}")

    project_root = Path(__file__).resolve().parents[1]
    project_utils = _load_parent_utils_module(project_root)
    GraphDataset = project_utils.GraphDataset
    results_path = Path(args.results_json)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    per_split: dict[str, dict] = {}
    first_error: Exception | None = None

    for split_idx, split in enumerate(splits):
        summary, err = _evaluate_split(
            cfg=cfg,
            cfg_path=cfg_path,
            split=split,
            args=args,
            GraphDataset=GraphDataset,
            rng_seed=int(args.seed) + split_idx,
        )
        per_split[split] = summary
        _print_summary_block(summary)
        if err is not None:
            first_error = err
            break

    if args.split == "all":
        run_status = "error" if first_error is not None else "ok"
        output = {
            "run_status": run_status,
            "config": str(cfg_path),
            "split": "all",
            "k_anchors": int(args.k),
            "splits": per_split,
        }
        if first_error is not None:
            output["error"] = f"{type(first_error).__name__}: {first_error}"
    else:
        output = per_split[splits[0]]

    _save_summary_json(results_path, output)
    if first_error is not None:
        print(f"saved_partial_json_on_error: {results_path}")
        raise first_error
    print(f"saved_json: {results_path}")


if __name__ == "__main__":
    main()
