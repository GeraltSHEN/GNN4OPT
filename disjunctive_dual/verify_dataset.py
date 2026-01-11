import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import time
import torch

from eval import _infer_feature_dimensions, _load_config, _merge_args_with_config, pad_tensor
from utils import load_checkpoint, load_data, load_model, print_dash_str, set_seed

AnnotationRow = Tuple[int, int, float]


def _write_split_csv(split: str, samples: List[AnnotationRow], output_dir: Path) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split}_samples.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_type", "tied_best_scores", "first_margin"])
        writer.writerows(samples)


def _format_rate(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _format_scores(values: List[float]) -> List[str]:
    return [f"{value:.4f}" for value in values]


def _build_args() -> argparse.Namespace:
    dataset = "set_cover"
    cfg_idx = 1
    config_root = Path("./disjunctive_dual/cfg")
    cfg = _load_config(config_root, dataset, cfg_idx)
    init_args = argparse.Namespace(
        dataset=dataset,
        cfg_idx=cfg_idx,
        config_root=str(config_root),
        model_suffix="",
        parent_test_stats_dir="data/data_dist/set_cover",
        eval_split="all",
        eval_batch_size=cfg.get("batch_size", 8),
        eval_train_batch_size=None,
        eval_val_batch_size=None,
        eval_test_batch_size=None,
        max_samples_per_split=None,
    )
    return _merge_args_with_config(init_args, cfg)


def _load_models(args: argparse.Namespace, cons_nfeats: int, edge_nfeats: int, var_nfeats: int):
    device = args.device
    holo_model_dir = Path("./disjunctive_dual/models/holo/set_cover_cfg1")
    selector_model_dir = Path("./disjunctive_dual/models/raw/set_cover_cfg0")

    holo_model = load_model(args, cons_nfeats, edge_nfeats, var_nfeats)
    load_checkpoint(holo_model, None, step="max", save_dir=str(holo_model_dir), device=device)
    holo_model.eval()

    selector_args = argparse.Namespace(**vars(args))
    selector_args.model = "raw"
    selector_model = load_model(selector_args, cons_nfeats, edge_nfeats, var_nfeats)
    load_checkpoint(selector_model, None, step="max", save_dir=str(selector_model_dir), device=device)
    selector_model.eval()
    return holo_model, selector_model


def _update_norm_scores(
    scores: torch.Tensor,
    nb_candidates: torch.Tensor,
    accum_sum: List[float],
    accum_count: List[int],
) -> None:
    max_k = min(5, scores.size(1))
    device = scores.device
    position_mask = torch.arange(max_k, device=device).unsqueeze(0) < nb_candidates.unsqueeze(1)
    for idx in range(max_k):
        valid_mask = position_mask[:, idx]
        if valid_mask.any():
            accum_sum[idx] += scores[:, idx][valid_mask].sum().item()
            accum_count[idx] += int(valid_mask.sum().item())


def evaluate_split(
    data_loader, holo_model, selector_model, device, record_samples: bool = False
) -> Tuple[Dict[str, object], List[AnnotationRow]]:
    holo_model.eval()
    selector_model.eval()

    total_graphs = 0
    k_max = 8
    sample_records: List[AnnotationRow] = []

    consistency_count = 0.0
    consistency_correct = 0.0
    correction_count = 0.0
    correction_correct = 0.0
    exploration_count = 0.0
    exploration_correct = 0.0

    top1_match_diff_sum = [0.0 for _ in range(k_max - 1)]
    top1_match_diff_count = [0 for _ in range(k_max - 1)]
    top1_mismatch_diff_sum = [0.0 for _ in range(k_max - 1)]
    top1_mismatch_diff_count = [0 for _ in range(k_max - 1)]
    topk_contains_diff_sum = [0.0 for _ in range(k_max - 1)]
    topk_contains_diff_count = [0 for _ in range(k_max - 1)]
    topk_miss_diff_sum = [0.0 for _ in range(k_max - 1)]
    topk_miss_diff_count = [0 for _ in range(k_max - 1)]
    small_top12_gap_count = 0
    first_margin_thresholds = (("1e-4", 1e-4), ("1e-3", 1e-3))
    first_margin_stats = {
        label: {
            "total": 0,
            "top1_match": 0,
            "top1_mismatch": 0,
            "top8_contains": 0,
            "top8_miss": 0,
        }
        for label, _ in first_margin_thresholds
    }
    large_first_margin_outside_top8_count = 0
    large_first_margin_outside_top8_contains = 0
    large_first_margin_outside_top8_miss = 0
    large_first_margin_outside_top8_top1_match = 0
    large_first_margin_outside_top8_top1_mismatch = 0
    top12_gap_counts = {
        "selector_top1_match": 0,
        "selector_top1_mismatch": 0,
        "selector_top8_contains": 0,
        "selector_top8_misses": 0,
    }

    def _accumulate_difficulty(
        diffs: torch.Tensor,
        valid_mask: torch.Tensor,
        group_mask: torch.Tensor,
        accum_sum: List[float],
        accum_count: List[int],
    ) -> None:
        for idx in range(diffs.size(1)):
            pos_mask = group_mask & valid_mask[:, idx]
            if pos_mask.any():
                accum_sum[idx] += diffs[pos_mask, idx].sum().item()
                accum_count[idx] += int(pos_mask.sum().item())

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            holo_logits = holo_model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )
            selector_logits = selector_model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )

            holo_logits = pad_tensor(holo_logits[batch.candidates], batch.nb_candidates)
            selector_logits = pad_tensor(selector_logits[batch.candidates], batch.nb_candidates)
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)

            num_candidates = batch.nb_candidates
            candidate_mask = (
                torch.arange(true_scores.size(1), device=device).unsqueeze(0) < num_candidates.unsqueeze(1)
            )
            masked_true_scores = true_scores.masked_fill(~candidate_mask, -1e9)

            best_scores = masked_true_scores.max(dim=-1).values
            best_scores_unsqueezed = best_scores.unsqueeze(-1)
            k_top = min(k_max, true_scores.size(1))

            holo_top1_idx = holo_logits.argmax(dim=-1)
            selector_topk_idx = selector_logits.topk(k_top, dim=-1).indices
            selector_top1_idx = selector_topk_idx[:, 0]

            selector_top1_scores = masked_true_scores.gather(-1, selector_top1_idx.unsqueeze(-1)).squeeze(-1)
            selector_topk_scores = masked_true_scores.gather(-1, selector_topk_idx)
            holo_top1_scores = masked_true_scores.gather(-1, holo_top1_idx.unsqueeze(-1)).squeeze(-1)

            selector_top1_match = selector_top1_scores == best_scores
            selector_topk_contains_best = (selector_topk_scores == best_scores_unsqueezed).any(dim=-1)
            holo_top1_match = holo_top1_scores == best_scores

            consistency_count += selector_top1_match.float().sum().item()
            consistency_correct += (selector_top1_match & holo_top1_match).float().sum().item()

            correction_mask = (~selector_top1_match) & selector_topk_contains_best
            correction_count += correction_mask.float().sum().item()
            correction_correct += (correction_mask & holo_top1_match).float().sum().item()

            exploration_mask = ~selector_topk_contains_best
            exploration_count += exploration_mask.float().sum().item()
            exploration_correct += (exploration_mask & holo_top1_match).float().sum().item()

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

                for label, threshold in first_margin_thresholds:
                    margin_mask = has_first_margin & (first_margin_values < threshold)
                    if margin_mask.any():
                        stats = first_margin_stats[label]
                        stats["total"] += int(margin_mask.sum().item())
                        stats["top1_match"] += int((margin_mask & selector_top1_match).sum().item())
                        stats["top1_mismatch"] += int((margin_mask & ~selector_top1_match).sum().item())
                        stats["top8_contains"] += int(
                            (margin_mask & selector_topk_contains_best).sum().item()
                        )
                        stats["top8_miss"] += int((margin_mask & ~selector_topk_contains_best).sum().item())

                first_margin_outside_top8_mask = has_first_margin & (first_margin_idx >= (k_max - 1))
                if first_margin_outside_top8_mask.any():
                    margin_mask = first_margin_outside_top8_mask
                    margin_mask_count = int(margin_mask.sum().item())
                    large_first_margin_outside_top8_count += margin_mask_count
                    large_first_margin_outside_top8_contains += int(
                        (margin_mask & selector_topk_contains_best).sum().item()
                    )
                    large_first_margin_outside_top8_miss += int(
                        (margin_mask & ~selector_topk_contains_best).sum().item()
                    )
                    large_first_margin_outside_top8_top1_match += int(
                        (margin_mask & selector_top1_match).sum().item()
                    )
                    large_first_margin_outside_top8_top1_mismatch += int(
                        (margin_mask & ~selector_top1_match).sum().item()
                    )

            if k_top > 1:
                gt_top_values = masked_true_scores.topk(k_top, dim=-1).values
                valid_diff_mask = num_candidates.unsqueeze(1) > (
                    torch.arange(k_top - 1, device=device) + 1
                )
                gt_diffs = (gt_top_values[:, :-1] - gt_top_values[:, 1:]) / best_norm.unsqueeze(-1)

                _accumulate_difficulty(
                    gt_diffs,
                    valid_diff_mask,
                    selector_top1_match,
                    top1_match_diff_sum,
                    top1_match_diff_count,
                )
                _accumulate_difficulty(
                    gt_diffs,
                    valid_diff_mask,
                    ~selector_top1_match,
                    top1_mismatch_diff_sum,
                    top1_mismatch_diff_count,
                )
                _accumulate_difficulty(
                    gt_diffs,
                    valid_diff_mask,
                    selector_topk_contains_best,
                    topk_contains_diff_sum,
                    topk_contains_diff_count,
                )
                _accumulate_difficulty(
                    gt_diffs,
                    valid_diff_mask,
                    ~selector_topk_contains_best,
                    topk_miss_diff_sum,
                    topk_miss_diff_count,
                )

                top12_valid = valid_diff_mask[:, 0]
                if top12_valid.any():
                    top12_diffs = gt_diffs[:, 0]
                    small_gap_mask = top12_valid & (top12_diffs < 1e-4)
                    if small_gap_mask.any():
                        small_top12_gap_count += int(small_gap_mask.sum().item())

                    def _update_gap_counts(group_mask, key: str) -> None:
                        valid_group = top12_valid & group_mask
                        if valid_group.any():
                            top12_gap_counts[key] += int((top12_diffs[valid_group] < 1e-4).sum().item())

                    _update_gap_counts(selector_top1_match, "selector_top1_match")
                    _update_gap_counts(~selector_top1_match, "selector_top1_mismatch")
                    _update_gap_counts(selector_topk_contains_best, "selector_top8_contains")
                    _update_gap_counts(~selector_topk_contains_best, "selector_top8_misses")

            total_graphs += batch.num_graphs
            if record_samples:
                tied_best = ((masked_true_scores == best_scores_unsqueezed) & candidate_mask).sum(dim=1)
                sample_type = torch.ones_like(best_scores, dtype=torch.int64)
                sample_type = sample_type + ((~selector_top1_match) & selector_topk_contains_best).to(torch.int64)
                sample_type = sample_type + (~selector_topk_contains_best).to(torch.int64) * 2
                sample_records.extend(
                    zip(
                        sample_type.cpu().tolist(),
                        tied_best.cpu().tolist(),
                        first_margin_values.cpu().tolist(),
                    )
                )

    def _compute_rates(correct: float, count: float) -> float:
        return (correct / count) if count > 0 else 0.0

    def _compute_avg(accum_sum: List[float], accum_count: List[int]) -> List[float]:
        return [
            (accum_sum[i] / accum_count[i]) if accum_count[i] > 0 else 0.0 for i in range(k_max - 1)
        ]

    def _build_first_margin_metrics(stats: Dict[str, int]) -> Dict[str, object]:
        top1_total = stats["top1_match"] + stats["top1_mismatch"]
        top8_total = stats["top8_contains"] + stats["top8_miss"]
        return {
            "total": stats["total"],
            "selector_top1": {
                "match": {
                    "count": stats["top1_match"],
                    "rate": _compute_rates(stats["top1_match"], top1_total),
                },
                "mismatch": {
                    "count": stats["top1_mismatch"],
                    "rate": _compute_rates(stats["top1_mismatch"], top1_total),
                },
            },
            "selector_top8": {
                "contains": {
                    "count": stats["top8_contains"],
                    "rate": _compute_rates(stats["top8_contains"], top8_total),
                },
                "misses": {
                    "count": stats["top8_miss"],
                    "rate": _compute_rates(stats["top8_miss"], top8_total),
                },
            },
        }

    outside_top8_first_margin_total = (
        large_first_margin_outside_top8_contains + large_first_margin_outside_top8_miss
    )
    outside_top8_first_margin_top1_total = (
        large_first_margin_outside_top8_top1_match + large_first_margin_outside_top8_top1_mismatch
    )
    first_margin_metrics = {
        "thresholds": {
            label: _build_first_margin_metrics(stats) for label, stats in first_margin_stats.items()
        },
        "threshold_order": [label for label, _ in first_margin_thresholds],
        "outside_top8": {
            "total": large_first_margin_outside_top8_count,
            "selector_top1": {
                "match": {
                    "count": large_first_margin_outside_top8_top1_match,
                    "rate": _compute_rates(
                        large_first_margin_outside_top8_top1_match, outside_top8_first_margin_top1_total
                    ),
                },
                "mismatch": {
                    "count": large_first_margin_outside_top8_top1_mismatch,
                    "rate": _compute_rates(
                        large_first_margin_outside_top8_top1_mismatch, outside_top8_first_margin_top1_total
                    ),
                },
            },
            "selector_top8": {
                "contains": {
                    "count": large_first_margin_outside_top8_contains,
                    "rate": _compute_rates(
                        large_first_margin_outside_top8_contains, outside_top8_first_margin_total
                    ),
                },
                "misses": {
                    "count": large_first_margin_outside_top8_miss,
                    "rate": _compute_rates(
                        large_first_margin_outside_top8_miss, outside_top8_first_margin_total
                    ),
                },
            },
        },
    }

    metrics = {
        "holo_consistency": {
            "rate": _compute_rates(consistency_correct, consistency_count),
            "count": consistency_count,
        },
        "holo_correction": {
            "rate": _compute_rates(correction_correct, correction_count),
            "count": correction_count,
        },
        "holo_exploration": {
            "rate": _compute_rates(exploration_correct, exploration_count),
            "count": exploration_count,
        },
        "difficulty": {
            "selector_top1_match": {
                "match": _compute_avg(top1_match_diff_sum, top1_match_diff_count),
                "mismatch": _compute_avg(top1_mismatch_diff_sum, top1_mismatch_diff_count),
            },
            "selector_top8_contains_gt": {
                "contains": _compute_avg(topk_contains_diff_sum, topk_contains_diff_count),
                "misses": _compute_avg(topk_miss_diff_sum, topk_miss_diff_count),
            },
        },
        "small_top12_gap_count": small_top12_gap_count,
        "small_first_margin_count": first_margin_stats["1e-4"]["total"],
        "large_first_margin_outside_top8_count": large_first_margin_outside_top8_count,
        "top12_gap_counts": top12_gap_counts,
        "first_margin": first_margin_metrics,
        "samples": total_graphs,
    }
    return metrics, sample_records


def _print_results(split: str, metrics: Dict[str, object], duration: float) -> None:
    consistency = metrics["holo_consistency"]
    correction = metrics["holo_correction"]
    exploration = metrics["holo_exploration"]
    difficulty = metrics["difficulty"]
    small_gap_count = int(metrics["small_top12_gap_count"])
    top12_gap_counts = metrics["top12_gap_counts"]
    first_margin = metrics["first_margin"]
    first_margin_by_threshold = first_margin["thresholds"]
    first_margin_order = first_margin.get("threshold_order", tuple(first_margin_by_threshold.keys()))
    outside_top8_first_margin = first_margin["outside_top8"]
    outside_top8_first_margin_top1 = outside_top8_first_margin["selector_top1"]

    def _fmt_gap(key: str) -> str:
        return f"<1e-4: {top12_gap_counts[key]}"

    def _print_first_margin_stats(label: str, stats: Dict[str, object]) -> None:
        print(
            f"First margin normalized diff < {label} -> "
            f"total: {stats['total']}, "
            f"top1 match: {stats['selector_top1']['match']['count']} "
            f"({_format_rate(stats['selector_top1']['match']['rate'])}), "
            f"top1 mismatch: {stats['selector_top1']['mismatch']['count']} "
            f"({_format_rate(stats['selector_top1']['mismatch']['rate'])})"
        )
        print(
            f"First margin < {label} by selector top8 coverage -> "
            f"contains: {stats['selector_top8']['contains']['count']} "
            f"({_format_rate(stats['selector_top8']['contains']['rate'])}), "
            f"misses: {stats['selector_top8']['misses']['count']} "
            f"({_format_rate(stats['selector_top8']['misses']['rate'])})"
        )

    print_dash_str(f"{split.upper()} results")
    print(
        f"Holo consistency (selector top1 matches GT): {_format_rate(consistency['rate'])} "
        f"over {int(consistency['count'])} samples"
    )
    print(
        f"Holo correction (selector top8 includes GT, top1 wrong): {_format_rate(correction['rate'])} "
        f"over {int(correction['count'])} samples"
    )
    print(
        f"Holo exploration (selector top8 misses GT): {_format_rate(exploration['rate'])} "
        f"over {int(exploration['count'])} samples"
    )
    print(
        f"Difficulty by selector top1 match -> \nmatch: {_format_scores(difficulty['selector_top1_match']['match'])}, "
        f"\nmismatch: {_format_scores(difficulty['selector_top1_match']['mismatch'])}"
    )
    print(
        f"Difficulty by selector top8 coverage -> \ncontains: {_format_scores(difficulty['selector_top8_contains_gt']['contains'])}, "
        f"\nmisses: {_format_scores(difficulty['selector_top8_contains_gt']['misses'])}"
    )
    print(
        "Top1 vs top2 normalized diff counts -> "
        f"top1 match: {_fmt_gap('selector_top1_match')}, "
        f"top1 mismatch: {_fmt_gap('selector_top1_mismatch')}, "
        f"top8 contains: {_fmt_gap('selector_top8_contains')}, "
        f"top8 misses: {_fmt_gap('selector_top8_misses')}"
    )
    for label in first_margin_order:
        _print_first_margin_stats(label, first_margin_by_threshold[label])
    print(
        "First margin outside top8 -> "
        f"{outside_top8_first_margin['total']} samples | "
        f"top1 match: {outside_top8_first_margin_top1['match']['count']} "
        f"({_format_rate(outside_top8_first_margin_top1['match']['rate'])}), "
        f"top1 mismatch: {outside_top8_first_margin_top1['mismatch']['count']} "
        f"({_format_rate(outside_top8_first_margin_top1['mismatch']['rate'])}) | "
        f"top8 contains: {outside_top8_first_margin['selector_top8']['contains']['count']} "
        f"({_format_rate(outside_top8_first_margin['selector_top8']['contains']['rate'])}), "
        f"top8 misses: {outside_top8_first_margin['selector_top8']['misses']['count']} "
        f"({_format_rate(outside_top8_first_margin['selector_top8']['misses']['rate'])})"
    )
    print(f"Top1 vs top2 normalized diff < 1e-4: {small_gap_count} samples")
    print(f"Split time: {duration:.2f}s")


if __name__ == "__main__":
    args = _build_args()
    print(f"Using device: {args.device}")
    set_seed(args.seed)
    data = load_data(args, for_training=False)
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions(data.get("train"))

    device = args.device
    holo_model, selector_model = _load_models(args, cons_nfeats, edge_nfeats, var_nfeats)

    output_dir = Path(args.parent_test_stats_dir)
    for split in ("train", "val", "test"):
        loader = data.get(split)
        if loader is None:
            continue
        start = time.time()
        metrics, sample_records = evaluate_split(loader, holo_model, selector_model, device, record_samples=True)
        elapsed = time.time() - start
        _print_results(split, metrics, elapsed)
        _write_split_csv(split, sample_records, output_dir)
