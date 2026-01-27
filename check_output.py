import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import torch

from eval import _infer_feature_dimensions, _load_config, _merge_args_with_config
from utils import load_data, load_checkpoint, load_model, set_seed


TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _parse_cfgs(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_model_dir(args: argparse.Namespace) -> Path:
    base_dir = Path(getattr(args, "model_dir", "./models"))
    model_name = getattr(args, "model", None)
    if model_name:
        base_dir = base_dir / model_name
    model_id = getattr(args, "model_id", None)
    if model_id:
        base_dir = base_dir / model_id
    model_suffix = getattr(args, "model_suffix", "")
    if model_suffix:
        base_dir = Path(f"{base_dir}_{model_suffix}")
    return base_dir


def _iter_splits(data, splits: Sequence[str]):
    for split in splits:
        loader = data.get(split)
        if loader is not None:
            yield split, loader


def _infer_any_feature_dimensions(data) -> tuple:
    for split in ("train", "val", "test"):
        loader = data.get(split)
        if loader is not None:
            return _infer_feature_dimensions(loader)
    raise ValueError("No data loaders available to infer feature dimensions.")


def _write_split_rows(
    writer: csv.writer,
    model: torch.nn.Module,
    loader,
    device: str,
):
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            split_sizes = batch.nb_candidates.detach().cpu().tolist()
            candidate_scores = batch.candidate_scores

            logits = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )
            candidate_logits = logits[batch.candidates]

            score_splits = torch.split(candidate_scores, split_sizes)
            logit_splits = torch.split(candidate_logits, split_sizes)

            for scores, outputs in zip(score_splits, logit_splits):
                best_score = scores.max()
                tied_best = int((scores == best_score).sum().item())

                top_logit = outputs.max()
                diffs = (outputs - top_logit).abs()
                counts = [int((diffs < tol).sum().item()) for tol in TOLERANCES]
                eq_zero = int((diffs == 0).sum().item())
                writer.writerow([tied_best, *counts, eq_zero])


def _run_cfg(args: argparse.Namespace, cfg_idx: int) -> None:
    cfg = _load_config(Path(args.config_root), args.dataset, cfg_idx)
    init_args = argparse.Namespace(
        dataset=args.dataset,
        cfg_idx=cfg_idx,
        config_root=args.config_root,
        model_suffix=args.model_suffix,
        eval_batch_size=args.eval_batch_size,
    )
    merged = _merge_args_with_config(init_args, cfg)
    if getattr(merged, "eval_batch_size", None) is None:
        merged.eval_batch_size = getattr(merged, "batch_size", 32)
    if args.max_samples_per_split is not None:
        merged.max_samples_per_split = args.max_samples_per_split

    set_seed(merged.seed)
    data = load_data(merged, for_training=False)
    cons_nfeats, edge_nfeats, var_nfeats = _infer_any_feature_dimensions(data)
    model = load_model(merged, cons_nfeats, edge_nfeats, var_nfeats)

    model_dir = _resolve_model_dir(merged)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    load_checkpoint(model, None, step="max", save_dir=str(model_dir), device=merged.device)
    model.eval()

    output_root = Path(args.output_dir) / model_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    header = [
        "tied_best_scores",
        "diff_lt_1e-1",
        "diff_lt_1e-2",
        "diff_lt_1e-3",
        "diff_lt_1e-4",
        "diff_lt_1e-5",
        "diff_lt_1e-6",
        "diff_eq_0",
    ]

    for split, loader in _iter_splits(data, args.splits):
        output_path = output_root / f"{split}_samples.csv"
        with output_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            _write_split_rows(writer, model, loader, merged.device)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check tie counts and top-score output similarities for set cover models."
    )
    parser.add_argument("--dataset", type=str, default="set_cover")
    parser.add_argument("--cfgs", type=_parse_cfgs, default=_parse_cfgs("21,22"))
    parser.add_argument("--config-root", type=str, default="./cfg")
    parser.add_argument("--model-suffix", type=str, default="")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        choices=("train", "val", "test"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results_summary/check_output",
    )
    args = parser.parse_args(argv)

    for cfg_idx in args.cfgs:
        _run_cfg(args, cfg_idx)


if __name__ == "__main__":
    main()
