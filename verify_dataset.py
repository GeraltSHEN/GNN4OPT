import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import time
import torch

from default_args import DATASET_DEFAULTS
from eval import _infer_feature_dimensions, _load_config, _merge_args_with_config, pad_tensor
from utils import load_checkpoint, load_data, load_model, print_dash_str, set_seed


AnnotationRow = Tuple[int, int, float, int]


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dataset statistics with a single model configuration.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_DEFAULTS.keys()),
        default="set_cover",
        help="Dataset to verify.",
    )
    parser.add_argument(
        "--cfg_idx",
        type=int,
        default=0,
        help="Configuration index to load.",
    )
    parser.add_argument(
        "--config_root",
        type=str,
        default="./cfg",
        help="Root directory containing configuration files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("raw", "holo"),
        default=None,
        help="Override model type defined in the configuration.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Dataset split(s) to evaluate.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size (defaults to batch_size in config).",
    )
    parser.add_argument(
        "--eval_train_batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size for train split.",
    )
    parser.add_argument(
        "--eval_val_batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size for val split.",
    )
    parser.add_argument(
        "--eval_test_batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size for test split.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Maximum number of samples per split.",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="",
        help="Optional suffix appended to model directories.",
    )
    parser.add_argument(
        "--parent_test_stats_dir",
        type=str,
        default=None,
        help="Directory to store verification stats.",
    )
    return parser.parse_args()


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


def _build_args():
    cli_args = _parse_cli_args()
    config_root = Path(cli_args.config_root)
    stats_dir = (
        Path(cli_args.parent_test_stats_dir)
        if cli_args.parent_test_stats_dir
        else Path("data/data_dist") / cli_args.dataset
    )
    cfg = _load_config(config_root, cli_args.dataset, cli_args.cfg_idx)
    eval_bs = cli_args.eval_batch_size or cfg.get("batch_size", 1)
    init_args = argparse.Namespace(
        dataset=cli_args.dataset,
        cfg_idx=cli_args.cfg_idx,
        config_root=str(config_root),
        model_suffix=cli_args.model_suffix,
        parent_test_stats_dir=str(stats_dir),
        eval_split=cli_args.eval_split,
        eval_batch_size=eval_bs,
        eval_train_batch_size=cli_args.eval_train_batch_size,
        eval_val_batch_size=cli_args.eval_val_batch_size,
        eval_test_batch_size=cli_args.eval_test_batch_size,
        max_samples_per_split=cli_args.max_samples_per_split,
    )
    args = _merge_args_with_config(init_args, cfg)
    if cli_args.model:
        args.model = cli_args.model
    return args



def _write_split_csv(split: str, samples: List[AnnotationRow], output_dir: Path) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split}_samples.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_idx", "sample_type", "tied_best_scores", "first_margin"])
        writer.writerows(samples)


def _load_model(args: argparse.Namespace, cons_nfeats: int, edge_nfeats: int, var_nfeats: int):
    device = args.device
    model = load_model(args, cons_nfeats, edge_nfeats, var_nfeats)
    model_dir = _resolve_model_dir(args)
    load_checkpoint(model, None, step="max", save_dir=str(model_dir), device=device)
    model.eval()
    return model


def evaluate_split(
    data_loader, model, device, record_samples: bool = False, k_max: int = 8
) -> Tuple[Dict[str, object], List[AnnotationRow]]:
    model.eval()
    sample_records: List[AnnotationRow] = []
    sample_type_counts = {1: 0, 2: 0, 3: 0}
    total_graphs = 0
    sample_offset = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )

            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)

            num_candidates = batch.nb_candidates
            candidate_mask = (
                torch.arange(true_scores.size(1), device=device).unsqueeze(0) < num_candidates.unsqueeze(1)
            )
            masked_true_scores = true_scores.masked_fill(~candidate_mask, -1e9)

            best_scores = masked_true_scores.max(dim=-1).values
            best_scores_unsqueezed = best_scores.unsqueeze(-1)
            k_top = min(k_max, true_scores.size(1))

            topk_idx = logits.topk(k_top, dim=-1).indices
            top1_idx = topk_idx[:, 0]

            top1_scores = masked_true_scores.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)
            topk_scores = masked_true_scores.gather(-1, topk_idx)

            top1_match = top1_scores == best_scores
            topk_contains_best = (topk_scores == best_scores_unsqueezed).any(dim=-1)

            best_norm = best_scores.clamp_min(1e-8)
            first_margin_values = torch.zeros_like(best_scores)
            if masked_true_scores.size(1) > 1:
                sorted_true_scores, _ = masked_true_scores.sort(dim=1, descending=True)
                full_score_diffs = (sorted_true_scores[:, :-1] - sorted_true_scores[:, 1:]) / best_norm.unsqueeze(-1)
                valid_full_margin_mask = (
                    torch.arange(full_score_diffs.size(1), device=device).unsqueeze(0)
                    < (num_candidates - 1).unsqueeze(1)
                )
                positive_margin_mask = (full_score_diffs > 0) & valid_full_margin_mask
                margin_indices = (
                    torch.arange(full_score_diffs.size(1), device=device)
                    .unsqueeze(0)
                    .expand_as(full_score_diffs)
                )
                first_margin_idx = torch.where(
                    positive_margin_mask,
                    margin_indices,
                    torch.full_like(margin_indices, full_score_diffs.size(1)),
                ).min(dim=1).values
                has_first_margin = first_margin_idx < full_score_diffs.size(1)
                if has_first_margin.any():
                    margin_rows = has_first_margin.nonzero(as_tuple=False).squeeze(-1)
                    chosen_idx = first_margin_idx[margin_rows].long()
                    first_margin_values[margin_rows] = full_score_diffs[margin_rows, chosen_idx]

            total_graphs += batch.num_graphs
            if record_samples:
                sample_indices = torch.arange(
                    sample_offset, sample_offset + batch.num_graphs, device=device, dtype=torch.int64
                )
                sample_offset += batch.num_graphs
                tied_best = ((masked_true_scores == best_scores_unsqueezed) & candidate_mask).sum(dim=1)
                sample_type = torch.ones_like(best_scores, dtype=torch.int64)
                sample_type = sample_type + ((~top1_match) & topk_contains_best).to(torch.int64)
                sample_type = sample_type + (~topk_contains_best).to(torch.int64) * 2

                for key in sample_type_counts:
                    sample_type_counts[key] += int((sample_type == key).sum().item())

                sample_records.extend(
                    zip(
                        sample_indices.cpu().tolist(),
                        sample_type.cpu().tolist(),
                        tied_best.cpu().tolist(),
                        first_margin_values.cpu().tolist(),
                    )
                )

    metrics = {
        "samples": total_graphs,
        "sample_type_counts": sample_type_counts,
    }
    return metrics, sample_records


def _print_results(split: str, metrics: Dict[str, object], duration: float) -> None:
    print_dash_str(f"{split.upper()} results")
    total = metrics.get("samples", 0)
    counts = metrics.get("sample_type_counts", {})
    print(f"Samples: {total}")
    print(
        "Sample type counts -> "
        f"type 1: {counts.get(1, 0)}, type 2: {counts.get(2, 0)}, type 3: {counts.get(3, 0)}"
    )
    print(f"Split time: {duration:.2f}s")


if __name__ == "__main__":
    args = _build_args()
    device = args.device
    print(f"Using device: {device}")
    set_seed(args.seed)
    data = load_data(args, for_training=False)
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions(data.get("train"))

    model = _load_model(args, cons_nfeats, edge_nfeats, var_nfeats)

    run_root = Path(args.parent_test_stats_dir) / _resolve_model_dir(args).name
    run_root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test") if args.eval_split == "all" else (args.eval_split,)
    for split in splits:
        loader = data.get(split)
        if loader is None:
            continue
        start = time.time()
        metrics, sample_records = evaluate_split(loader, model, device, record_samples=True)
        elapsed = time.time() - start
        _print_results(split, metrics, elapsed)
        _write_split_csv(split, sample_records, run_root)
