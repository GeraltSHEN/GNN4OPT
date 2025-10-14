import os
import time
import argparse
from pathlib import Path
import pdb
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from utils import (load_data, load_model,
                   load_json, save_json, set_seed, load_checkpoint, print_dash_str)
import tqdm
import yaml


def _infer_feature_dimensions(train_loader):
    """Infer feature dimensions from a single sample in the training dataset."""
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise ValueError("Training dataset is empty; cannot infer feature dimensions.")
    sample = dataset[0]
    cons_nfeats = sample.constraint_features.shape[-1]
    edge_nfeats = sample.edge_attr.shape[-1]
    var_nfeats = sample.variable_features.shape[-1]
    return cons_nfeats, edge_nfeats, var_nfeats


def evaluate(policy, data_loader, device, stats_filename):
    mean_loss = 0
    mean_acc = 0
    mean_top5_acc = 0
    mean_score_diff = 0
    mean_normalized_score_diff = 0

    policy.eval()

    n_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, disable=True):
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)
            # if isnan: pdb
            if torch.isnan(loss):
                pdb.set_trace()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()
            top5_acc = (true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == true_bestscore).float().max(dim=-1).values.mean().item()

            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex)).abs().mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex)) / (true_bestscore + 1e-5)).mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs

            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_top5_acc /= n_samples_processed
    mean_score_diff /= n_samples_processed
    mean_normalized_score_diff /= n_samples_processed

    instance_dir_results = {
        'Loss': mean_loss,
        'Accuracy': mean_acc,
        'Top5_Accuracy': mean_top5_acc,
        'Score_diff': mean_score_diff,
        'Normalized_score_diff': mean_normalized_score_diff,
        "n_samples": n_samples_processed
    }

    save_json(stats_filename, instance_dir_results)
    return mean_loss, mean_acc, mean_top5_acc, mean_score_diff, mean_normalized_score_diff


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Eval the MILP branching policy.")
    parser.add_argument("--dataset", type=str, default="set_cover", help="Dataset key.")
    parser.add_argument("--cfg_idx", type=int, default=0, help="Configuration index.")
    parser.add_argument("--config_root", type=str, default="./cfg", help="Directory containing configuration files.")
    parser.add_argument("--model_suffix", type=str, default="", help="Optional suffix appended to model/log directories.")
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
        default=32, 
        help="Base evaluation batch size (used if split-specific batch size is not provided).",
    )
    parser.add_argument(
        "--eval_train_batch_size",
        type=int,
        required=False,
        help="Evaluation batch size for the training split.",
    )
    parser.add_argument(
        "--eval_val_batch_size",
        type=int,
        required=False,
        help="Evaluation batch size for the validation split.",
    )
    parser.add_argument(
        "--eval_test_batch_size",
        type=int,
        required=False,
        help="Evaluation batch size for the test split.",
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
    return args


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
    print(f"Number of parameters: {sum(p.numel() for p in policy.parameters())}")
    
    base_model_dir = Path(getattr(args, "model_dir", "./models"))
    model_id = getattr(args, "model_id", None)
    if model_id:
        base_model_dir = base_model_dir / model_id
    model_suffix = getattr(args, "model_suffix", "")
    if model_suffix:
        base_model_dir = Path(f"{base_model_dir}_{model_suffix}")
    assert os.path.exists(base_model_dir), f"Model directory does not exist: {base_model_dir}"

    load_checkpoint(policy, None, step="max", save_dir=str(base_model_dir), device=args.device)

    policy.eval()

    model_name = base_model_dir.name
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
        split_stats_dir = model_stats_root / split
        split_stats_dir.mkdir(parents=True, exist_ok=True)
        stats_filename = split_stats_dir / "eval_acc.json"
        print_dash_str(f"[{split}] Save stats to: {split_stats_dir}")
        metrics = evaluate(policy, data_loader, args.device, str(stats_filename))
        results[split] = {"metrics": metrics, "stats_path": stats_filename}
        print(
            f"[{split}] loss {metrics[0]:.4f}, acc {metrics[1]:.4f}, top5 {metrics[2]:.4f}, "
            f"score_diff {metrics[3]:.4f}, normalized {metrics[4]:.4f}"
        )

    if args.eval_split == "all" and results:
        aggregate_dir = model_stats_root / "all"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = aggregate_dir / "eval_acc.json"

        aggregate_totals = {
            "Loss": 0.0,
            "Accuracy": 0.0,
            "Top5_Accuracy": 0.0,
            "Score_diff": 0.0,
            "Normalized_score_diff": 0.0,
        }
        total_samples = 0
        for split_result in results.values():
            split_stats = load_json(str(split_result["stats_path"]))
            n_samples = split_stats.get("n_samples", 0)
            if n_samples == 0:
                continue
            total_samples += n_samples
            for key in aggregate_totals.keys():
                aggregate_totals[key] += split_stats[key] * n_samples

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
            f"normalized {aggregate_stats['Normalized_score_diff']:.4f}, samples {total_samples}"
        )



if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"Total eval time: {(time.time() - start_time) / 60:.2f} minutes")
