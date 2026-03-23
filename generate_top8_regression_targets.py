import argparse
import gzip
import pickle
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heuristics.utils import SCIPBranchingContext, load_sample, unpack_sample_data
from tmp_utils import GraphDataset, load_model


TARGET_KEY = "top8_regression_targets"


def resolve_sample_files(cfg: dict, split: str, max_samples: Optional[int]):
    dataset_root = Path(cfg["dataset_path"])
    file_pattern = cfg.get("file_pattern", "sample_*.pkl")
    if split == "all":
        split_dirs = [
            dataset_root / cfg.get("train_split", "train"),
            dataset_root / cfg.get("val_split", "valid"),
            dataset_root / cfg.get("test_split", "test"),
        ]
    elif split == "train":
        split_dirs = [dataset_root / cfg.get("train_split", "train")]
    elif split == "valid":
        split_dirs = [dataset_root / cfg.get("val_split", "valid")]
    elif split == "test":
        split_dirs = [dataset_root / cfg.get("test_split", "test")]
    else:
        raise ValueError(f"unsupported split: {split}")

    sample_files = []
    for split_dir in split_dirs:
        sample_files.extend(sorted(split_dir.glob(file_pattern)))
    if max_samples is not None:
        sample_files = sample_files[: int(max_samples)]
    return sample_files


def select_topk_local_positions(all_var_scores: torch.Tensor, candidates: torch.Tensor, k: int):
    candidate_scores = all_var_scores[candidates]
    k_eff = min(int(k), int(candidate_scores.numel()))
    top_local = torch.topk(candidate_scores, k=k_eff).indices
    if k_eff < k:
        top_local = torch.cat([top_local, top_local[:1].expand(k - k_eff)], dim=0)
    return top_local


def main():
    parser = argparse.ArgumentParser(description="Generate top-8 regression targets and save into sample files.")
    parser.add_argument("--cfg", type=str, default="cfg/set_cover_60")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/raw/set_cover_cfg60/1250000.pth",
        help="Exact checkpoint path to load for top-k selection.",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "valid", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    sample_files = resolve_sample_files(cfg, args.split, args.max_samples)
    if len(sample_files) == 0:
        raise RuntimeError("no sample files found")

    model_args = argparse.Namespace(**cfg)
    model_args.device = args.device
    model_args.model = "raw"

    feature_dataset = GraphDataset(
        [sample_files[0]],
        edge_nfeats=int(cfg.get("edge_nfeats", 1)),
        args=model_args,
    )
    first_graph = feature_dataset.get(0)
    cons_nfeats = int(first_graph.constraint_features.shape[-1])
    edge_nfeats = int(first_graph.edge_attr.shape[-1])
    var_nfeats = int(first_graph.variable_features.shape[-1])

    model = load_model(model_args, cons_nfeats, edge_nfeats, var_nfeats)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = GraphDataset(
        sample_files,
        edge_nfeats=int(cfg.get("edge_nfeats", 1)),
        args=model_args,
    )

    for idx in range(len(sample_files)):
        graph = dataset.get(idx)
        graph_device = graph.to(args.device)
        n_constraints = torch.as_tensor([int(graph.n_constraints_per_graph)], device=args.device)
        n_variables = torch.as_tensor([int(graph.n_variables_per_graph)], device=args.device)

        with torch.no_grad():
            all_var_scores = model(
                graph_device.constraint_features,
                graph_device.edge_index,
                graph_device.edge_attr,
                graph_device.variable_features,
                candidates=graph_device.candidates,
                n_constraints_per_graph=n_constraints,
                n_variables_per_graph=n_variables,
            )

        top_local = select_topk_local_positions(all_var_scores, graph_device.candidates, int(args.k))
        top_local_cpu = top_local.detach().cpu()
        top_candidates = graph.candidates[top_local_cpu].detach().cpu().numpy().astype(np.int64)
        top_scores = graph.candidate_scores[top_local_cpu].detach().cpu().numpy().astype(np.float32)

        raw_sample = load_sample(sample_files[idx])
        sample_record = unpack_sample_data(raw_sample["data"])
        context = SCIPBranchingContext.from_sample_state(sample_record.sample_state)
        parent_primal = context.solve_parent_primal()
        cutoffbound = float(sample_record.cutoffbound)

        n_cons = int(graph.n_constraints_per_graph)
        n_vars = int(graph.n_variables_per_graph)
        k = int(args.k)

        true_y = np.full((k, 2, n_cons), np.nan, dtype=np.float32)
        true_alpha = np.full((k, 2, n_vars), np.nan, dtype=np.float32)
        true_beta = np.full((k, 2, n_vars), np.nan, dtype=np.float32)
        true_obj = np.full((k, 2), np.nan, dtype=np.float32)

        for j, var_idx in enumerate(top_candidates.tolist()):
            down = context.solve_child_dual(int(var_idx), direction="down", cutoffbound=cutoffbound)
            up = context.solve_child_dual(int(var_idx), direction="up", cutoffbound=cutoffbound)

            if down.success:
                true_y[j, 0, :] = down.y.astype(np.float32, copy=False)
                true_alpha[j, 0, :] = down.alpha.astype(np.float32, copy=False)
                true_beta[j, 0, :] = down.beta.astype(np.float32, copy=False)
                true_obj[j, 0] = np.float32(down.objective_value)
            if up.success:
                true_y[j, 1, :] = up.y.astype(np.float32, copy=False)
                true_alpha[j, 1, :] = up.alpha.astype(np.float32, copy=False)
                true_beta[j, 1, :] = up.beta.astype(np.float32, copy=False)
                true_obj[j, 1] = np.float32(up.objective_value)

        raw_sample[TARGET_KEY] = {
            "candidate_positions": top_local_cpu.numpy().astype(np.int64),
            "candidate_indices": top_candidates,
            "scores": top_scores,
            "y": true_y,
            "alpha": true_alpha,
            "beta": true_beta,
            "obj": true_obj,
            "parent_obj": np.float32(parent_primal.objective_value),
            "cutoffbound": np.float32(cutoffbound),
        }

        with gzip.open(sample_files[idx], "wb") as f:
            pickle.dump(raw_sample, f)

    print(f"saved {TARGET_KEY} for {len(sample_files)} samples.")


if __name__ == "__main__":
    main()
