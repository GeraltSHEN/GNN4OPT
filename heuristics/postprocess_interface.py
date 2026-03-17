from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .utils import load_sample


class IndexedGraphDataset:
    """
    Lightweight wrapper that adds `graph_index` to each sample so batched code can
    recover which raw sample file each graph came from.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.sample_files = [str(p) for p in getattr(base_dataset, "sample_files")]

    def __len__(self):
        return len(self.base_dataset)

    def len(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        graph = self.base_dataset[index]
        graph.graph_index = torch.tensor([int(index)], dtype=torch.long)
        return graph

    def get(self, index):
        return self.__getitem__(index)


class HeuristicPostProcessInterface:
    """
    Build batched post-process tensors for HeuristicPolicy from sample files.

    Output dict keys:
      - rhs: flat (sum_i m_i,)
      - parent_lbs: flat (sum_i n_i,)
      - parent_ubs: flat (sum_i n_i,)
      - lp_solution: flat (sum_i n_i,)
      - parent_obj: (B,)
      - cutoffbound: (B,)
      - objective_offset: (B,)
    """

    def __init__(self, sample_files: Sequence[str | Path]):
        self.sample_files = [Path(p) for p in sample_files]
        self._cache: Dict[int, Dict[str, np.ndarray | float]] = {}

    @staticmethod
    def _col(values: np.ndarray, names: Sequence[str], name: str, default=None):
        if name not in names:
            if default is None:
                raise KeyError(f"Feature '{name}' not found.")
            return np.full(values.shape[0], float(default), dtype=np.float32)
        idx = names.index(name)
        return values[:, idx].astype(np.float32, copy=False)

    @staticmethod
    def _infer_parent_bounds(var_values: np.ndarray, var_names: Sequence[str]):
        n = var_values.shape[0]
        lbs = np.zeros(n, dtype=np.float32)
        ubs = np.ones(n, dtype=np.float32)

        # If explicit bound vectors are not stored in raw sample metadata, this
        # default matches set-cover style binaries and keeps post-process finite.
        if "type_0" in var_names:
            binary = HeuristicPostProcessInterface._col(var_values, var_names, "type_0", 0.0) > 0.5
            lbs[binary] = 0.0
            ubs[binary] = 1.0

        return lbs, ubs

    def _load_one(self, graph_idx: int):
        cached = self._cache.get(graph_idx)
        if cached is not None:
            return cached

        sample = load_sample(self.sample_files[int(graph_idx)])
        sample_data = sample["data"]
        sample_state = sample_data[0]
        cutoffbound = float(sample_data[5]) if len(sample_data) >= 6 and sample_data[5] is not None else float("inf")

        constraint_dict, _, variable_dict = sample_state
        constraint_values = np.asarray(constraint_dict["values"], dtype=np.float32)
        variable_values = np.asarray(variable_dict["values"], dtype=np.float32)
        constraint_names = list(constraint_dict["names"])
        variable_names = list(variable_dict["names"])

        bias = self._col(constraint_values, constraint_names, "bias", 0.0)
        rhs = -bias

        lp_solution = self._col(variable_values, variable_names, "sol_val", 0.0)
        obj_coeffs = self._col(variable_values, variable_names, "coef_normalized", 0.0)

        parent_lbs, parent_ubs = self._infer_parent_bounds(variable_values, variable_names)

        reconstruction = variable_dict.get("reconstruction", {})
        if "lbs" in reconstruction:
            rec_lbs = np.asarray(reconstruction["lbs"], dtype=np.float32).reshape(-1)
            if rec_lbs.shape[0] == lp_solution.shape[0]:
                parent_lbs = rec_lbs
        if "ubs" in reconstruction:
            rec_ubs = np.asarray(reconstruction["ubs"], dtype=np.float32).reshape(-1)
            if rec_ubs.shape[0] == lp_solution.shape[0]:
                parent_ubs = rec_ubs

        objective_offset = float(reconstruction.get("objective_offset", 0.0))
        parent_obj = float(np.dot(obj_coeffs.astype(np.float64), lp_solution.astype(np.float64)) + objective_offset)

        out = {
            "rhs": rhs,
            "parent_lbs": parent_lbs.astype(np.float32, copy=False),
            "parent_ubs": parent_ubs.astype(np.float32, copy=False),
            "lp_solution": lp_solution.astype(np.float32, copy=False),
            "parent_obj": parent_obj,
            "cutoffbound": cutoffbound,
            "objective_offset": objective_offset,
        }
        self._cache[graph_idx] = out
        return out

    def make_batch_data(self, graph_index: torch.Tensor, device: torch.device, dtype: torch.dtype = torch.float32):
        graph_ids = graph_index.reshape(-1).detach().cpu().tolist()
        batch = [self._load_one(int(i)) for i in graph_ids]

        rhs = np.concatenate([x["rhs"] for x in batch], axis=0)
        parent_lbs = np.concatenate([x["parent_lbs"] for x in batch], axis=0)
        parent_ubs = np.concatenate([x["parent_ubs"] for x in batch], axis=0)
        lp_solution = np.concatenate([x["lp_solution"] for x in batch], axis=0)

        parent_obj = np.asarray([x["parent_obj"] for x in batch], dtype=np.float32)
        cutoffbound = np.asarray([x["cutoffbound"] for x in batch], dtype=np.float32)
        objective_offset = np.asarray([x["objective_offset"] for x in batch], dtype=np.float32)

        return {
            "rhs": torch.as_tensor(rhs, device=device, dtype=dtype),
            "parent_lbs": torch.as_tensor(parent_lbs, device=device, dtype=dtype),
            "parent_ubs": torch.as_tensor(parent_ubs, device=device, dtype=dtype),
            "lp_solution": torch.as_tensor(lp_solution, device=device, dtype=dtype),
            "parent_obj": torch.as_tensor(parent_obj, device=device, dtype=dtype),
            "cutoffbound": torch.as_tensor(cutoffbound, device=device, dtype=dtype),
            "objective_offset": torch.as_tensor(objective_offset, device=device, dtype=dtype),
        }
