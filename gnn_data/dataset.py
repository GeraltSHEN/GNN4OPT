from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_gzip_sample(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _split_dir(dataset_root: Path, split: str) -> Path:
    split_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    key = str(split).lower()
    if key not in split_map:
        raise ValueError(f"Unsupported split '{split}'. Use one of: train, valid/val, test.")
    split_dir = dataset_root / split_map[key]
    if split_dir.exists() and split_dir.is_dir():
        return split_dir
    if dataset_root.exists() and dataset_root.is_dir():
        return dataset_root
    raise FileNotFoundError(f"Dataset directory not found: {split_dir}")


def _name_to_index(names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def _resolve_target_keys(dual_option: int) -> List[str]:
    if int(dual_option) == 1:
        return ["top8_regression_targets_option1", "top8_regression_targets", "top8_regression_targets_option2"]
    if int(dual_option) == 2:
        return ["top8_regression_targets_option2", "top8_regression_targets_option1", "top8_regression_targets"]
    return ["top8_regression_targets", "top8_regression_targets_option1", "top8_regression_targets_option2"]


def _select_topk_target(sample: Dict[str, Any], dual_option: int) -> Dict[str, Any]:
    for key in _resolve_target_keys(dual_option):
        if key in sample:
            return sample[key]
    raise KeyError(f"Missing top-k target keys for dual_option={dual_option}.")


def _target_tensor_or_nan(
    target: Dict[str, Any],
    key: str,
    *,
    k: int,
    dim: int,
    sample_path: Path,
) -> np.ndarray:
    raw = target.get(key, None)
    if raw is None:
        return np.full((int(k), 2, int(dim)), np.nan, dtype=np.float32)

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 2:
        raise ValueError(f"Expected {key} shape (k,2,d), got {arr.shape} in {sample_path}")
    if arr.shape[0] < int(k):
        raise ValueError(f"{key} has too few candidates: {arr.shape[0]} < {k} in {sample_path}")
    if arr.shape[2] < int(dim):
        raise ValueError(f"{key} has too few features: {arr.shape[2]} < {dim} in {sample_path}")
    return arr[: int(k), :, : int(dim)].astype(np.float32, copy=False)


def _clean_candidates(
    candidates: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_choice_node_id: int,
    variable_features: torch.Tensor,
    variable_feature_indices: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    sol_is_not_at_lb = variable_features[candidates, variable_feature_indices["sol_is_at_lb"]] == 0
    sol_is_not_at_ub = variable_features[candidates, variable_feature_indices["sol_is_at_ub"]] == 0
    sol_is_not_at_lub = sol_is_not_at_lb & sol_is_not_at_ub
    cleaned_candidates = candidates[sol_is_not_at_lub]
    cleaned_candidate_scores = candidate_scores[sol_is_not_at_lub]
    if cleaned_candidates.numel() < 1:
        raise ValueError("No candidate exists after cleaning.")
    if candidate_choice_node_id not in cleaned_candidates:
        new_candidate_choice = cleaned_candidate_scores.argmax().item()
        candidate_choice_node_id = int(cleaned_candidates[new_candidate_choice].item())
    return cleaned_candidates, cleaned_candidate_scores, candidate_choice_node_id


def _build_milp_graph_from_sample(
    sample: Dict[str, Any],
    *,
    sample_path: Path,
    remove_bad_candidates: bool,
) -> Dict[str, Any]:
    sample_data = sample["data"]
    if not isinstance(sample_data, (list, tuple)) or len(sample_data) < 5:
        raise ValueError(f"Expected sample['data'] length >= 5. Got {type(sample_data)}")

    sample_state = sample_data[0]
    sample_action = int(sample_data[2])
    sample_action_set = sample_data[3]
    sample_scores = sample_data[4]
    sample_cutoffbound = float(sample_data[5]) if len(sample_data) > 5 else float("nan")

    constraint_dict, edge_dict, variable_dict = sample_state
    edge_index = torch.as_tensor(edge_dict["indices"], dtype=torch.long)
    edge_attr = torch.as_tensor(edge_dict["values"], dtype=torch.float32)

    variable_names = list(variable_dict["names"])
    variable_default_features = torch.as_tensor(variable_dict["values"], dtype=torch.float32)
    variable_feature_indices = _name_to_index(variable_names)

    constraint_names = list(constraint_dict["names"])
    constraint_default_features = torch.as_tensor(constraint_dict["values"], dtype=torch.float32)
    constraint_feature_indices = _name_to_index(constraint_names)

    cutoff_feature_name = "cutoffbound_normalized"
    variable_required = [
            "type_0",
            "type_1",
            "type_2",
            "type_3",
            "has_lb",
            "has_ub",
            "sol_is_at_lb",
            "sol_is_at_ub",
            "sol_frac",
            "coef_normalized",
            "sol_val",
            cutoff_feature_name,
        ]
    constraint_required = ["bias", "dualsol_val_normalized", cutoff_feature_name]

    missing_variable = [name for name in variable_required if name not in variable_feature_indices]
    missing_constraint = [name for name in constraint_required if name not in constraint_feature_indices]
    if missing_variable:
        raise KeyError(f"Missing variable features {missing_variable} in {sample_path}")
    if missing_constraint:
        raise KeyError(f"Missing constraint features {missing_constraint} in {sample_path}")

    variable_features = torch.stack(
            [variable_default_features[:, variable_feature_indices[name]] for name in variable_required],
            dim=-1,
        )
    constraint_features = torch.stack(
            [constraint_default_features[:, constraint_feature_indices[name]] for name in constraint_required],
            dim=-1,
        )
    variable_feature_indices = {name: i for i, name in enumerate(variable_required)}

    candidates = torch.as_tensor(sample_action_set, dtype=torch.long)
    candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)
    candidate_choice_node_id = sample_action

    if remove_bad_candidates:
        if "sol_is_at_lb" not in variable_feature_indices or "sol_is_at_ub" not in variable_feature_indices:
            raise KeyError("remove_bad_candidates=True requires sol_is_at_lb and sol_is_at_ub features.")
        candidates, candidate_scores, candidate_choice_node_id = _clean_candidates(
            candidates,
            candidate_scores,
            candidate_choice_node_id,
            variable_features,
            variable_feature_indices,
        )

    choice_tensor = torch.where(candidates == torch.as_tensor(candidate_choice_node_id, dtype=torch.long))[0]
    if choice_tensor.numel() == 0:
        raise ValueError(f"Chosen candidate id {candidate_choice_node_id} not present in candidates.")
    candidate_choice_local = int(choice_tensor[0].item())

    is_not_fixed_feature = torch.zeros(variable_features.size(0), dtype=torch.float32)
    is_not_fixed_feature[torch.as_tensor(sample_action_set, dtype=torch.long)] = 1.0
    candidates_feature = torch.zeros(variable_features.size(0), dtype=torch.float32)
    candidates_feature[candidates] = 1.0
    variable_features = torch.cat(
        [variable_features, is_not_fixed_feature.unsqueeze(-1), candidates_feature.unsqueeze(-1)],
        dim=-1,
    )

    return {
        "constraint_features": constraint_features.contiguous(),
        "edge_index": edge_index.contiguous(),
        "edge_attr": edge_attr.contiguous(),
        "variable_features": variable_features.contiguous(),
        "n_constraints": int(constraint_features.size(0)),
        "n_variables": int(variable_features.size(0)),
        "candidates": candidates.contiguous(),
        "candidate_scores": candidate_scores.contiguous(),
        "candidate_choice_local": candidate_choice_local,
        "candidate_choice_node_id": int(candidate_choice_node_id),
        "sample_cutoffbound": sample_cutoffbound,
    }


class MILPDataset(Dataset):
    """Dataset of MILP parent graphs from raw sample files.

    This class does not touch top-k targets. It only prepares parent MILP graph tensors.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        *,
        file_pattern: str = "sample_*.pkl",
        remove_bad_candidates: bool = True,
        transform=None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.split_dir = _split_dir(self.dataset_root, split)
        self.file_pattern = str(file_pattern)
        self.sample_files = sorted(self.split_dir.glob(self.file_pattern))
        if not self.sample_files:
            raise RuntimeError(f"No files matched pattern '{self.file_pattern}' in {self.split_dir}")

        self.remove_bad_candidates = bool(remove_bad_candidates)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_files)

    def _get_one(self, index: int) -> Dict[str, Any]:
        path = self.sample_files[int(index)]
        sample = load_gzip_sample(path)
        graph = _build_milp_graph_from_sample(
            sample,
            sample_path=path,
            remove_bad_candidates=self.remove_bad_candidates,
        )
        graph.update(
            {
                "sample_index": int(index),
                "sample_path": str(path),
            }
        )
        if self.transform is not None:
            graph = self.transform(graph)
        return graph

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._get_one(i) for i in range(start, stop, step)]
        return self._get_one(int(index))


class LPDataset(Dataset):
    """Dataset that expands each MILP sample into up to 2 * top_k LP sub-problem graphs.

    - Uses saved top-k targets from sample files (no live get_top_k).
    - Does NOT include parent graph in expansion by design.
    - Stores parent_obj and cutoffbound on each LP graph record.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        *,
        top_k: int = 8,
        dual_option: int = 1,
        file_pattern: str = "sample_*.pkl",
        remove_bad_candidates: bool = True,
        transform=None,
    ):
        super().__init__()
        if int(top_k) <= 0:
            raise ValueError("top_k must be >= 1.")
        self.top_k = int(top_k)
        self.dual_option = int(dual_option)
        self.transform = transform
        self.base = MILPDataset(
            dataset_root=dataset_root,
            split=split,
            file_pattern=file_pattern,
            remove_bad_candidates=remove_bad_candidates,
            transform=None,
        )

    @property
    def sample_files(self) -> List[Path]:
        return self.base.sample_files

    def __len__(self) -> int:
        return len(self.base)

    def _expand_one(self, index: int) -> Dict[str, Any]:
        milp_graph = self.base._get_one(index)
        sample_path = Path(milp_graph["sample_path"])
        sample = load_gzip_sample(sample_path)
        target = _select_topk_target(sample, self.dual_option)

        candidate_indices = np.asarray(target["candidate_indices"], dtype=np.int64).reshape(-1)
        scores = np.asarray(target["scores"], dtype=np.float32).reshape(-1)
        obj = np.asarray(target["obj"], dtype=np.float32)
        parent_obj = float(target["parent_obj"])
        cutoffbound = float(target["cutoffbound"])

        if candidate_indices.size == 0:
            raise ValueError(f"No candidate_indices in target for sample {sample_path}")
        if scores.shape[0] < candidate_indices.shape[0]:
            raise ValueError(
                f"Expected scores length >= {candidate_indices.shape[0]}, "
                f"got {scores.shape[0]} in {sample_path}"
            )
        if obj.ndim != 2 or obj.shape[1] != 2:
            raise ValueError(f"Expected obj shape (k,2), got {obj.shape} in {sample_path}")
        if obj.shape[0] < candidate_indices.shape[0]:
            raise ValueError(
                f"Expected obj first dim >= {candidate_indices.shape[0]}, "
                f"got {obj.shape[0]} in {sample_path}"
            )

        n_constraints = int(milp_graph["n_constraints"])
        n_variables = int(milp_graph["n_variables"])
        y = _target_tensor_or_nan(
            target,
            "y",
            k=int(candidate_indices.shape[0]),
            dim=n_constraints,
            sample_path=sample_path,
        )
        alpha = _target_tensor_or_nan(
            target,
            "alpha",
            k=int(candidate_indices.shape[0]),
            dim=n_variables,
            sample_path=sample_path,
        )
        beta = _target_tensor_or_nan(
            target,
            "beta",
            k=int(candidate_indices.shape[0]),
            dim=n_variables,
            sample_path=sample_path,
        )

        k_available = int(candidate_indices.shape[0])
        k_eff = min(self.top_k, k_available)
        candidate_indices = candidate_indices[:k_eff]
        scores = scores[:k_eff]
        obj = obj[:k_eff]
        y = y[:k_eff]
        alpha = alpha[:k_eff]
        beta = beta[:k_eff]

        base_variable_features = milp_graph["variable_features"]

        lp_graphs: List[Dict[str, Any]] = []
        for rank in range(k_eff):
            branch_var = int(candidate_indices[rank])
            if branch_var < 0 or branch_var >= n_variables:
                raise ValueError(f"branch_var {branch_var} out of range [0, {n_variables}) in {sample_path}")

            for branch_dir in (0, 1):  # 0=down, 1=up
                branch_onehot = torch.zeros((n_variables, 2), dtype=base_variable_features.dtype)
                branch_onehot[branch_var, int(branch_dir)] = 1.0
                variable_features = torch.cat([base_variable_features, branch_onehot], dim=-1).contiguous()
                lp_graphs.append(
                    {
                        "constraint_features": milp_graph["constraint_features"],
                        "edge_index": milp_graph["edge_index"],
                        "edge_attr": milp_graph["edge_attr"],
                        "variable_features": variable_features,
                        "n_constraints": n_constraints,
                        "n_variables": n_variables,
                        "branch_var_index": branch_var,
                        "branch_dir": int(branch_dir),
                        "topk_rank": int(rank),
                        "target_y": torch.as_tensor(y[rank, branch_dir, :n_constraints], dtype=torch.float32).contiguous(),
                        "target_alpha": torch.as_tensor(alpha[rank, branch_dir, :n_variables], dtype=torch.float32).contiguous(),
                        "target_beta": torch.as_tensor(beta[rank, branch_dir, :n_variables], dtype=torch.float32).contiguous(),
                        "target_obj": float(obj[rank, branch_dir]),
                        "target_score": float(scores[rank]),
                        "parent_obj": parent_obj,
                        "cutoffbound": cutoffbound,
                        "sample_index": int(index),
                        "sample_path": str(sample_path),
                    }
                )

        item = {
            "sample_index": int(index),
            "sample_path": str(sample_path),
            "n_lp_graphs": len(lp_graphs),
            "lp_graphs": lp_graphs,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._expand_one(i) for i in range(start, stop, step)]
        return self._expand_one(int(index))


class LPGraphDataset(Dataset):
    """Dataset of individual LP graphs sampled from a global LP pool.

    This is intended for training with highly diverse LP batches where LP graphs
    from the same MILP are unlikely to co-occur in a mini-batch.

    Uses fixed slots per MILP sample: `2 * top_k`.
    If a sample contains fewer than `2 * top_k` LP graphs, 
    slot selection wraps modulo available LP graphs for that sample.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        *,
        top_k: int = 8,
        dual_option: int = 1,
        file_pattern: str = "sample_*.pkl",
        remove_bad_candidates: bool = True,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.grouped = LPDataset(
            dataset_root=dataset_root,
            split=split,
            top_k=top_k,
            dual_option=dual_option,
            file_pattern=file_pattern,
            remove_bad_candidates=remove_bad_candidates,
            transform=None,
        )
        self.top_k = int(self.grouped.top_k)
        self.lp_graphs_per_milp = int(2 * self.top_k)

    @property
    def sample_files(self) -> List[Path]:
        return self.grouped.sample_files

    def __len__(self) -> int:
        return len(self.grouped) * self.lp_graphs_per_milp

    def _get_one(self, index: int) -> Dict[str, Any]:
        total = len(self)
        idx = int(index)
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"Index out of range: {index} for dataset of size {total}.")

        sample_index = idx // self.lp_graphs_per_milp
        local_index = idx % self.lp_graphs_per_milp

        item = self.grouped._expand_one(sample_index)
        lp_graphs = item["lp_graphs"]
        if len(lp_graphs) == 0:
            raise ValueError(f"No LP graphs found for sample index {sample_index}.")
        lp_graph = lp_graphs[local_index % len(lp_graphs)]

        if self.transform is not None:
            lp_graph = self.transform(lp_graph)
        return lp_graph

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._get_one(i) for i in range(start, stop, step)]
        return self._get_one(int(index))
