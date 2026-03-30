import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch_geometric.loader import DataLoader

from tmp_utils import (
    load_data,
    load_model,
    load_checkpoint,
    set_seed,
    save_json,
    print_dash_str,
)
from heuristics.postprocess_interface import (
    HeuristicPostProcessInterface,
    IndexedGraphDataset,
)


def _infer_feature_dimensions(train_loader):
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise ValueError("Training dataset is empty; cannot infer feature dimensions.")
    sample = dataset[0]
    cons_nfeats = sample.constraint_features.shape[-1]
    edge_nfeats = sample.edge_attr.shape[-1]
    var_nfeats = sample.variable_features.shape[-1]
    return cons_nfeats, edge_nfeats, var_nfeats


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output


def _wrap_loader_for_postprocess(
    loader,
    *,
    dual_option: int,
    universal_cutoffbound: float,
):
    indexed_dataset = IndexedGraphDataset(loader.dataset)
    wrapped_loader = DataLoader(
        indexed_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )
    postprocess_interface = HeuristicPostProcessInterface(
        indexed_dataset.sample_files,
        dual_option=int(dual_option),
        universal_cutoffbound=float(universal_cutoffbound),
    )
    return wrapped_loader, postprocess_interface


def evaluate_topk(policy, data_loader, device, postprocess_interface, ranking_csv_path: Path):
    mean_loss = 0.0
    mean_acc = 0.0
    mean_top5_acc = 0.0
    mean_score_diff = 0.0
    mean_normalized_score_diff = 0.0
    mean_topk_candidate_score = 0.0
    mean_topk_true_score = 0.0
    mean_all_cand_loss = 0.0
    mean_all_cand_acc = 0.0
    mean_all_cand_top5_acc = 0.0
    mean_all_cand_score_diff = 0.0
    mean_all_cand_normalized_score_diff = 0.0

    n_samples_processed = 0

    ranking_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ranking_csv_path, "w", newline="", encoding="utf-8") as ranking_fh:
        writer = csv.writer(ranking_fh)
        writer.writerow(
            [
                "graph_id",
                "predicted_topk_global",
                "true_topk_global",
                "top1_correct",
            ]
        )

        policy.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, disable=True):
                batch = batch.to(device)
                post_data = postprocess_interface.make_batch_data(
                    batch.graph_id,
                    device=batch.constraint_features.device,
                    dtype=batch.constraint_features.dtype,
                )
                _, padded_logits, aux = policy(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
                    candidates=batch.candidates,
                    nb_candidates=batch.nb_candidates,
                    n_constraints_per_graph=batch.n_constraints_per_graph,
                    n_variables_per_graph=batch.n_variables_per_graph,
                    data=post_data,
                    return_aux=True,
                )

                logits = padded_logits
                loss = F.cross_entropy(logits, batch.candidate_choices)

                top_local = aux["top_local"].to(device=device, dtype=torch.long)
                top_local = top_local.clamp(min=0, max=logits.size(-1) - 1)

                topk_logits = aux["topk_pseudo_scores"].to(device=device, dtype=logits.dtype)
                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
                topk_true_scores = true_scores.gather(1, top_local)
                all_cand_bestscore = true_scores.max(dim=-1, keepdims=True).values
                all_cand_predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
                all_cand_accuracy = (
                    true_scores.gather(-1, all_cand_predicted_bestindex) == all_cand_bestscore
                ).float().mean().item()
                all_cand_top5_acc = (
                    true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == all_cand_bestscore
                ).float().max(dim=-1).values.mean().item()
                all_cand_score_diff = (
                    all_cand_bestscore - true_scores.gather(-1, all_cand_predicted_bestindex)
                ).abs().mean().item()
                all_cand_normalized_score_diff = (
                    (all_cand_bestscore - true_scores.gather(-1, all_cand_predicted_bestindex)) / all_cand_bestscore
                ).mean().item()

                topk_bestscore = topk_true_scores.max(dim=-1, keepdims=True).values
                predicted_topk_index = topk_logits.max(dim=-1, keepdims=True).indices
                predicted_score = topk_true_scores.gather(-1, predicted_topk_index)

                accuracy = (predicted_score == topk_bestscore).float().mean().item()
                top5_acc = (
                    topk_true_scores.gather(
                        -1,
                        topk_logits.topk(min(5, topk_logits.size(-1))).indices,
                    )
                    == topk_bestscore
                ).float().max(dim=-1).values.mean().item()

                score_diff = (topk_bestscore - predicted_score).abs().mean().item()
                normalized_score_diff = (
                    (topk_bestscore - predicted_score) / topk_bestscore.clamp_min(1e-9)
                ).mean().item()

                padded_candidates = pad_tensor(batch.candidates, batch.nb_candidates, pad_value=-1)
                topk_candidate_global = padded_candidates.gather(1, top_local)
                predicted_rank_local = topk_logits.argsort(dim=-1, descending=True)
                true_rank_local = topk_true_scores.argsort(dim=-1, descending=True)
                predicted_rank_global = topk_candidate_global.gather(1, predicted_rank_local)
                true_rank_global = topk_candidate_global.gather(1, true_rank_local)
                top1_correct = (
                    topk_true_scores.gather(-1, predicted_rank_local[:, :1]) == topk_bestscore
                ).to(dtype=torch.long)
                graph_ids = batch.graph_id.reshape(-1).detach().cpu().tolist()
                for row_idx, graph_id in enumerate(graph_ids):
                    pred_list = predicted_rank_global[row_idx].detach().cpu().tolist()
                    true_list = true_rank_global[row_idx].detach().cpu().tolist()
                    writer.writerow(
                        [
                            int(graph_id),
                            " ".join(str(x) for x in pred_list),
                            " ".join(str(x) for x in true_list),
                            int(top1_correct[row_idx].item()),
                        ]
                    )

                mean_loss += loss.item() * batch.num_graphs
                mean_acc += accuracy * batch.num_graphs
                mean_top5_acc += top5_acc * batch.num_graphs
                mean_score_diff += score_diff * batch.num_graphs
                mean_normalized_score_diff += normalized_score_diff * batch.num_graphs
                mean_topk_candidate_score += topk_logits.mean().item() * batch.num_graphs
                mean_topk_true_score += topk_true_scores.mean().item() * batch.num_graphs
                mean_all_cand_loss += loss.item() * batch.num_graphs
                mean_all_cand_acc += all_cand_accuracy * batch.num_graphs
                mean_all_cand_top5_acc += all_cand_top5_acc * batch.num_graphs
                mean_all_cand_score_diff += all_cand_score_diff * batch.num_graphs
                mean_all_cand_normalized_score_diff += all_cand_normalized_score_diff * batch.num_graphs
                n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_top5_acc /= n_samples_processed
    mean_score_diff /= n_samples_processed
    mean_normalized_score_diff /= n_samples_processed
    mean_topk_candidate_score /= n_samples_processed
    mean_topk_true_score /= n_samples_processed
    mean_all_cand_loss /= n_samples_processed
    mean_all_cand_acc /= n_samples_processed
    mean_all_cand_top5_acc /= n_samples_processed
    mean_all_cand_score_diff /= n_samples_processed
    mean_all_cand_normalized_score_diff /= n_samples_processed

    return {
        "Loss": mean_loss,
        "Accuracy": mean_acc,
        "Top5_Accuracy": mean_top5_acc,
        "Score_diff": mean_score_diff,
        "Normalized_score_diff": mean_normalized_score_diff,
        "TopK_CandidateScoreMean": mean_topk_candidate_score,
        "TopK_TrueScoreMean": mean_topk_true_score,
        "all_cand_Loss": mean_all_cand_loss,
        "all_cand_Accuracy": mean_all_cand_acc,
        "all_cand_Top5_Accuracy": mean_all_cand_top5_acc,
        "all_cand_Score_diff": mean_all_cand_score_diff,
        "all_cand_Normalized_score_diff": mean_all_cand_normalized_score_diff,
        "n_samples": int(n_samples_processed),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate HeuristicPolicy on top-k candidates only.")
    parser.add_argument("--dataset", type=str, default="set_cover", help="Dataset key.")
    parser.add_argument("--cfg_idx", type=int, default=0, help="Configuration index.")
    parser.add_argument("--config_root", type=str, default="./cfg", help="Directory containing configuration files.")
    parser.add_argument("--model_suffix", type=str, default="", help="Optional suffix appended to model directory.")
    parser.add_argument("--parent_test_stats_dir", type=str, default="data/results_summary/", help="Directory to store evaluation statistics.")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=("train", "val", "test", "all"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Evaluation batch size for all splits.",
    )
    parser.add_argument(
        "--dual_option",
        type=int,
        default=argparse.SUPPRESS,
        choices=[1, 2],
        help="Post-process dual option used by HeuristicPolicy.",
    )
    parser.add_argument(
        "--universal_cutoffbound",
        type=float,
        default=argparse.SUPPRESS,
        help="Universal cutoffbound used when dual_option=2.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=str,
        default="max",
        help="Checkpoint step to load (e.g., 'max' or an integer step).",
    )
    return parser.parse_args(argv)


def _load_config(config_root: Path, dataset: str, cfg_idx: int) -> Dict[str, Any]:
    cfg_path = config_root / f"{dataset}_{cfg_idx}"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _merge_args_with_config(init_args, cfg: Dict[str, Any]):
    args_dict = {**cfg, **vars(init_args)}
    args = argparse.Namespace(**args_dict)
    args.model_id = f"{args.dataset}_cfg{args.cfg_idx}"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model = "heuristics"
    if not hasattr(args, "eval_batch_size"):
        args.eval_batch_size = int(getattr(args, "batch_size", 1))
    if not hasattr(args, "dual_option"):
        args.dual_option = 1
    if not hasattr(args, "universal_cutoffbound"):
        args.universal_cutoffbound = 1e6
    return args


def _resolve_model_dir(args):
    base_model_dir = Path(getattr(args, "model_dir", "./models"))
    if getattr(args, "model", None):
        base_model_dir = base_model_dir / args.model
    model_id = getattr(args, "model_id", None)
    if model_id:
        base_model_dir = base_model_dir / model_id
    model_suffix = getattr(args, "model_suffix", "")
    if model_suffix:
        base_model_dir = Path(f"{base_model_dir}_{model_suffix}")
    return base_model_dir


def _checkpoint_step_value(checkpoint_step: str):
    if checkpoint_step == "max":
        return "max"
    return int(checkpoint_step)


def main(argv=None):
    init_args = parse_args(argv)
    cfg = _load_config(Path(init_args.config_root), init_args.dataset, init_args.cfg_idx)
    args = _merge_args_with_config(init_args, cfg)

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    set_seed(args.seed)
    data = load_data(args, for_training=False)
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions(data.get("train"))

    policy = load_model(args, cons_nfeats, edge_nfeats, var_nfeats)
    model_dir = _resolve_model_dir(args)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    loaded_step = load_checkpoint(
        policy,
        None,
        step=_checkpoint_step_value(str(args.checkpoint_step)),
        save_dir=str(model_dir),
        device=args.device,
    )
    if int(loaded_step) == 0:
        raise RuntimeError(f"No checkpoint found in model directory: {model_dir}")
    print(f"Loaded checkpoint step: {loaded_step}")

    model_name = model_dir.name
    stats_root = Path(args.parent_test_stats_dir)
    stats_root.mkdir(parents=True, exist_ok=True)
    model_stats_root = stats_root / model_name
    model_stats_root.mkdir(parents=True, exist_ok=True)

    available_splits = ("train", "val", "test")
    if args.eval_split == "all":
        splits_to_evaluate = [split for split in available_splits if data.get(split) is not None]
    else:
        splits_to_evaluate = [args.eval_split]

    results = {}
    for split in splits_to_evaluate:
        data_loader = data.get(split)
        if data_loader is None:
            print_dash_str(f"No data loader available for split '{split}', skipping.")
            continue

        wrapped_loader, postprocess_interface = _wrap_loader_for_postprocess(
            data_loader,
            dual_option=int(args.dual_option),
            universal_cutoffbound=float(args.universal_cutoffbound),
        )

        split_stats_dir = model_stats_root / split
        split_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_filename = split_stats_dir / "eval_topk_acc.json"
        print_dash_str(f"[{split}] Save stats to: {split_stats_dir}")

        ranking_csv_path = split_stats_dir / "topk_ranking_compare.csv"
        metrics = evaluate_topk(
            policy,
            wrapped_loader,
            args.device,
            postprocess_interface,
            ranking_csv_path=ranking_csv_path,
        )
        save_json(str(stats_filename), metrics)
        results[split] = {
            "metrics": metrics,
            "stats_path": stats_filename,
            "ranking_csv_path": ranking_csv_path,
        }

        print(
            f"[{split}] loss {metrics['Loss']:.4f}, acc {metrics['Accuracy']:.4f}, top5 {metrics['Top5_Accuracy']:.4f}, "
            f"score_diff {metrics['Score_diff']:.4f}, normalized {metrics['Normalized_score_diff']:.4f}, "
            f"topk_pred_mean {metrics['TopK_CandidateScoreMean']:.4f}, topk_true_mean {metrics['TopK_TrueScoreMean']:.4f}, "
            f"all_cand_acc {metrics['all_cand_Accuracy']:.4f}, all_cand_top5 {metrics['all_cand_Top5_Accuracy']:.4f}, "
            f"ranking_csv {ranking_csv_path}"
        )

    if args.eval_split == "all" and results:
        aggregate_dir = model_stats_root / "all"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = aggregate_dir / "eval_topk_acc.json"

        aggregate_totals = {
            "Loss": 0.0,
            "Accuracy": 0.0,
            "Top5_Accuracy": 0.0,
            "Score_diff": 0.0,
            "Normalized_score_diff": 0.0,
            "TopK_CandidateScoreMean": 0.0,
            "TopK_TrueScoreMean": 0.0,
            "all_cand_Loss": 0.0,
            "all_cand_Accuracy": 0.0,
            "all_cand_Top5_Accuracy": 0.0,
            "all_cand_Score_diff": 0.0,
            "all_cand_Normalized_score_diff": 0.0,
        }
        total_samples = 0
        for split_result in results.values():
            split_stats = split_result["metrics"]
            n_samples = int(split_stats.get("n_samples", 0))
            if n_samples == 0:
                continue
            total_samples += n_samples
            for key in aggregate_totals.keys():
                aggregate_totals[key] += float(split_stats[key]) * n_samples

        if total_samples > 0:
            for key in aggregate_totals.keys():
                aggregate_totals[key] /= total_samples

        aggregate_stats = {
            **aggregate_totals,
            "n_samples": total_samples,
        }
        save_json(str(aggregate_path), aggregate_stats)
        print_dash_str("Aggregated results across all splits")
        print(
            f"[all] loss {aggregate_stats['Loss']:.4f}, acc {aggregate_stats['Accuracy']:.4f}, "
            f"top5 {aggregate_stats['Top5_Accuracy']:.4f}, score_diff {aggregate_stats['Score_diff']:.4f}, "
            f"normalized {aggregate_stats['Normalized_score_diff']:.4f}, "
            f"topk_pred_mean {aggregate_stats['TopK_CandidateScoreMean']:.4f}, "
            f"topk_true_mean {aggregate_stats['TopK_TrueScoreMean']:.4f}, "
            f"all_cand_acc {aggregate_stats['all_cand_Accuracy']:.4f}, "
            f"all_cand_top5 {aggregate_stats['all_cand_Top5_Accuracy']:.4f}, samples {total_samples}"
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total eval time: {(time.time() - start_time) / 60:.2f} minutes")
