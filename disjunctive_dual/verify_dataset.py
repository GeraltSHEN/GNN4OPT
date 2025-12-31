import argparse
from pathlib import Path
from typing import Dict, List
import time
import torch

from eval import _infer_feature_dimensions, _load_config, _merge_args_with_config, pad_tensor
from utils import load_checkpoint, load_data, load_model, print_dash_str, set_seed

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
        parent_test_stats_dir=".",
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


def evaluate_split(data_loader, holo_model, selector_model, device) -> Dict[str, object]:
    holo_model.eval()
    selector_model.eval()

    total_graphs = 0
    k_max = 8

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

            best_scores, best_indices = masked_true_scores.max(dim=-1)
            k_top = min(k_max, true_scores.size(1))

            holo_top1_idx = holo_logits.argmax(dim=-1)
            selector_topk_idx = selector_logits.topk(k_top, dim=-1).indices
            selector_top1_idx = selector_topk_idx[:, 0]

            selector_top1_match = selector_top1_idx == best_indices
            selector_topk_contains_best = (selector_topk_idx == best_indices.unsqueeze(-1)).any(dim=-1)
            holo_top1_match = holo_top1_idx == best_indices

            consistency_count += selector_top1_match.float().sum().item()
            consistency_correct += (selector_top1_match & holo_top1_match).float().sum().item()

            correction_mask = (~selector_top1_match) & selector_topk_contains_best
            correction_count += correction_mask.float().sum().item()
            correction_correct += (correction_mask & holo_top1_match).float().sum().item()

            exploration_mask = ~selector_topk_contains_best
            exploration_count += exploration_mask.float().sum().item()
            exploration_correct += (exploration_mask & holo_top1_match).float().sum().item()

            if k_top > 1:
                gt_top_values = masked_true_scores.topk(k_top, dim=-1).values
                valid_diff_mask = num_candidates.unsqueeze(1) > (
                    torch.arange(k_top - 1, device=device) + 1
                )
                best_norm = best_scores.clamp_min(1e-8).unsqueeze(-1)
                gt_diffs = (gt_top_values[:, :-1] - gt_top_values[:, 1:]) / best_norm

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
                    small_gap_mask = top12_valid & (gt_diffs[:, 0] < 1e-4)
                    if small_gap_mask.any():
                        small_top12_gap_count += int(small_gap_mask.sum().item())

            total_graphs += batch.num_graphs

    def _compute_rates(correct: float, count: float) -> float:
        return (correct / count) if count > 0 else 0.0

    def _compute_avg(accum_sum: List[float], accum_count: List[int]) -> List[float]:
        return [
            (accum_sum[i] / accum_count[i]) if accum_count[i] > 0 else 0.0 for i in range(k_max - 1)
        ]

    return {
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
        "samples": total_graphs,
    }


def _print_results(split: str, metrics: Dict[str, object], duration: float) -> None:
    consistency = metrics["holo_consistency"]
    correction = metrics["holo_correction"]
    exploration = metrics["holo_exploration"]
    difficulty = metrics["difficulty"]
    small_gap_count = int(metrics["small_top12_gap_count"])

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

    for split in ("train", "val", "test"):
        loader = data.get(split)
        if loader is None:
            continue
        start = time.time()
        metrics = evaluate_split(loader, holo_model, selector_model, device)
        elapsed = time.time() - start
        _print_results(split, metrics, elapsed)
