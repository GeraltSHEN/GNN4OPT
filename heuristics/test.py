"""Executable test script for anchor strong branching."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
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
    parser.add_argument(
        "--use_trained_model",
        action="store_true",
        help="Use trained model logits to pick top-k anchor candidates (otherwise random).",
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
    anchor_mode: str,
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
        "anchor_mode": anchor_mode,
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


def _parse_dataset_and_cfg_idx(cfg_path: Path) -> tuple[str, int]:
    match = re.match(r"^(.*)_(\d+)$", cfg_path.name)
    if match is None:
        raise ValueError(
            f"Cannot infer dataset/cfg_idx from config filename '{cfg_path.name}'. "
            "Expected pattern like 'set_cover_51'."
        )
    dataset = match.group(1)
    cfg_idx = int(match.group(2))
    return dataset, cfg_idx


def _infer_feature_dimensions_from_sample(GraphDataset, cfg_for_dataset: dict, sample_path: Path):
    dataset_args = argparse.Namespace(**cfg_for_dataset)
    dataset = GraphDataset(
        [sample_path],
        edge_nfeats=int(cfg_for_dataset.get("edge_nfeats", 1)),
        two_fwl=bool(cfg_for_dataset.get("two_fwl", False)),
        args=dataset_args,
    )
    sample = dataset.get(0)
    cons_nfeats = sample.constraint_features.shape[-1]
    edge_nfeats = sample.edge_attr.shape[-1]
    var_nfeats = sample.variable_features.shape[-1]
    return cons_nfeats, edge_nfeats, var_nfeats


def _load_trained_model_for_cfg(
    *,
    project_utils,
    GraphDataset,
    cfg: dict,
    cfg_path: Path,
    splits: list[str],
    max_samples: int | None,
):
    dataset_name, cfg_idx = _parse_dataset_and_cfg_idx(cfg_path)

    first_sample: Path | None = None
    for split in splits:
        files = _resolve_sample_files(cfg, split=split, max_samples=max_samples)
        if files:
            first_sample = files[0]
            break
    if first_sample is None:
        raise RuntimeError("Could not find any sample file to infer model feature dimensions.")

    cfg_for_dataset = dict(cfg)
    # Use the feature pipeline expected by the trained model config.
    cfg_for_dataset["two_fwl"] = bool(cfg.get("two_fwl", False))
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions_from_sample(
        GraphDataset,
        cfg_for_dataset,
        first_sample,
    )

    model_args = argparse.Namespace(**cfg)
    model_args.dataset = dataset_name
    model_args.cfg_idx = cfg_idx
    model_args.model_id = f"{dataset_name}_cfg{cfg_idx}"
    model_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = project_utils.load_model(model_args, cons_nfeats, edge_nfeats, var_nfeats)

    base_model_dir = Path(getattr(model_args, "model_dir", "./models"))
    if getattr(model_args, "model", None):
        base_model_dir = base_model_dir / model_args.model
    if getattr(model_args, "model_id", None):
        base_model_dir = base_model_dir / model_args.model_id
    model_suffix = getattr(model_args, "model_suffix", "")
    if model_suffix:
        base_model_dir = Path(f"{base_model_dir}_{model_suffix}")
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {base_model_dir}")

    step = project_utils.load_checkpoint(
        policy,
        None,
        step="max",
        save_dir=str(base_model_dir),
        device=model_args.device,
    )
    if int(step) == 0:
        raise RuntimeError(f"No checkpoint found in model directory: {base_model_dir}")
    policy.eval()
    return {
        "policy": policy,
        "device": model_args.device,
        "model_dir": str(base_model_dir),
        "model_step": int(step),
        "two_fwl": bool(cfg.get("two_fwl", False)),
    }


def _select_anchor_positions_with_model(graph, model_bundle, k: int) -> np.ndarray:
    policy = model_bundle["policy"]
    device = model_bundle["device"]

    graph = graph.to(device)
    n_constraints = torch.as_tensor([int(graph.n_constraints_per_graph)], device=device)
    n_variables = torch.as_tensor([int(graph.n_variables_per_graph)], device=device)
    con_var_features = graph.con_var_features if hasattr(graph, "con_var_features") else None
    var_var_features = graph.var_var_features if hasattr(graph, "var_var_features") else None
    if con_var_features is not None and con_var_features.dim() == 3:
        con_var_features = con_var_features.unsqueeze(0)
    if var_var_features is not None and var_var_features.dim() == 3:
        var_var_features = var_var_features.unsqueeze(0)

    with torch.no_grad():
        logits = policy(
            graph.constraint_features,
            graph.edge_index,
            graph.edge_attr,
            graph.variable_features,
            con_var_features,
            var_var_features,
            candidates=graph.candidates,
            n_constraints_per_graph=n_constraints,
            n_variables_per_graph=n_variables,
        )
        candidate_logits = logits[graph.candidates]
        k_eff = min(int(k), int(candidate_logits.numel()))
        if k_eff < 1:
            raise ValueError("No candidates available for model-based anchor selection.")
        top_pos = torch.topk(candidate_logits, k=k_eff).indices.detach().cpu().numpy().astype(np.int64)
    return np.sort(top_pos)


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
    print(f"anchor_mode: {summary['anchor_mode']}")
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
    trained_model_bundle=None,
) -> tuple[dict, Exception | None]:
    anchor_mode = "trained_model" if trained_model_bundle is not None else "random"
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
            anchor_mode=anchor_mode,
            run_status="error",
            error="RuntimeError: No sample files found for the requested split.",
        )
        return summary, RuntimeError("No sample files found for the requested split.")

    cfg_for_dataset = dict(cfg)
    if trained_model_bundle is None:
        cfg_for_dataset["two_fwl"] = False
    else:
        cfg_for_dataset["two_fwl"] = bool(trained_model_bundle.get("two_fwl", False))
    dataset_args = argparse.Namespace(**cfg_for_dataset)

    dataset = GraphDataset(
        sample_files,
        edge_nfeats=int(cfg_for_dataset.get("edge_nfeats", 1)),
        two_fwl=bool(cfg_for_dataset.get("two_fwl", False)),
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
                anchor_positions = None
                if trained_model_bundle is not None:
                    anchor_positions = _select_anchor_positions_with_model(
                        graph,
                        trained_model_bundle,
                        k=args.k,
                    )
                result = run_anchor_strong_branching(
                    graph,
                    k=args.k,
                    rng=rng,
                    use_default_features=use_default_features,
                    anchor_positions=anchor_positions,
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
            anchor_mode=anchor_mode,
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
            anchor_mode=anchor_mode,
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
    trained_model_bundle = None
    if args.use_trained_model:
        trained_model_bundle = _load_trained_model_for_cfg(
            project_utils=project_utils,
            GraphDataset=GraphDataset,
            cfg=cfg,
            cfg_path=cfg_path,
            splits=splits,
            max_samples=args.max_samples,
        )
        print(
            f"Loaded trained model from {trained_model_bundle['model_dir']} "
            f"(step {trained_model_bundle['model_step']})."
        )

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
            trained_model_bundle=trained_model_bundle,
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
            "anchor_mode": "trained_model" if args.use_trained_model else "random",
            "splits": per_split,
        }
        if trained_model_bundle is not None:
            output["model_dir"] = trained_model_bundle["model_dir"]
            output["model_step"] = trained_model_bundle["model_step"]
        if first_error is not None:
            output["error"] = f"{type(first_error).__name__}: {first_error}"
    else:
        output = per_split[splits[0]]
        if trained_model_bundle is not None:
            output["model_dir"] = trained_model_bundle["model_dir"]
            output["model_step"] = trained_model_bundle["model_step"]

    _save_summary_json(results_path, output)
    if first_error is not None:
        print(f"saved_partial_json_on_error: {results_path}")
        raise first_error
    print(f"saved_json: {results_path}")


if __name__ == "__main__":
    main()
