from __future__ import annotations

from typing import Any, Dict, List

import torch


def _cat_with_offsets(
    lp_graphs: List[Dict[str, Any]],
    n_constraints_per_graph: torch.Tensor,
    n_variables_per_graph: torch.Tensor,
):
    constraint_features = torch.cat([g["constraint_features"] for g in lp_graphs], dim=0)
    variable_features = torch.cat([g["variable_features"] for g in lp_graphs], dim=0)

    constraint_offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), n_constraints_per_graph[:-1]]), dim=0
    )
    variable_offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), n_variables_per_graph[:-1]]), dim=0
    )

    edge_index_list = []
    edge_attr_list = []
    for i, g in enumerate(lp_graphs):
        edge_index = g["edge_index"].clone()
        edge_index[0] += int(constraint_offsets[i].item())
        edge_index[1] += int(variable_offsets[i].item())
        edge_index_list.append(edge_index)
        edge_attr_list.append(g["edge_attr"])

    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_attr = torch.cat(edge_attr_list, dim=0)
    return constraint_features, variable_features, batched_edge_index, batched_edge_attr


def _split_down_up(item_lp_graphs: List[Dict[str, Any]]):
    down = [g for g in item_lp_graphs if int(g["branch_dir"]) == 0]
    up = [g for g in item_lp_graphs if int(g["branch_dir"]) == 1]
    down.sort(key=lambda g: int(g["topk_rank"]))
    up.sort(key=lambda g: int(g["topk_rank"]))
    if len(down) == 0 or len(up) == 0 or len(down) != len(up):
        raise ValueError("Each sample must contain matching down/up LP graphs.")
    return down, up


def collate_fn_lp_base(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of LPDataset items into one flattened LP-graph batch.

    Input:
      - list of dataset items, each item contains `lp_graphs` with length 2*k_i.

    Output:
      - PyG-style concatenated graph tensors plus per-LP metadata and targets
        for variable-k batching without LP padding.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch provided to collate_fn_lp_base.")

    lp_graphs: List[Dict[str, Any]] = []
    topk_per_milp: List[int] = []
    for item in batch:
        down, up = _split_down_up(item["lp_graphs"])
        k = len(down)
        topk_per_milp.append(k)
        for i in range(k):
            lp_graphs.append(down[i])
            lp_graphs.append(up[i])

    if len(lp_graphs) == 0:
        raise ValueError("No LP graphs found in batch.")

    n_constraints_per_graph = torch.as_tensor(
        [int(g["n_constraints"]) for g in lp_graphs], dtype=torch.long
    )
    n_variables_per_graph = torch.as_tensor(
        [int(g["n_variables"]) for g in lp_graphs], dtype=torch.long
    )

    (
        constraint_features,
        variable_features,
        edge_index,
        edge_attr,
    ) = _cat_with_offsets(lp_graphs, n_constraints_per_graph, n_variables_per_graph)

    target_y = torch.cat([g["target_y"] for g in lp_graphs], dim=0)
    target_alpha = torch.cat([g["target_alpha"] for g in lp_graphs], dim=0)
    target_beta = torch.cat([g["target_beta"] for g in lp_graphs], dim=0)
    target_obj = torch.as_tensor([float(g["target_obj"]) for g in lp_graphs], dtype=torch.float32)
    target_score = torch.as_tensor([float(g["target_score"]) for g in lp_graphs], dtype=torch.float32)

    branch_var_index = torch.as_tensor([int(g["branch_var_index"]) for g in lp_graphs], dtype=torch.long)
    branch_dir = torch.as_tensor([int(g["branch_dir"]) for g in lp_graphs], dtype=torch.long)
    topk_rank = torch.as_tensor([int(g["topk_rank"]) for g in lp_graphs], dtype=torch.long)
    parent_obj = torch.as_tensor([float(g["parent_obj"]) for g in lp_graphs], dtype=torch.float32)
    cutoffbound = torch.as_tensor([float(g["cutoffbound"]) for g in lp_graphs], dtype=torch.float32)

    source_sample_index = torch.as_tensor([int(g["sample_index"]) for g in lp_graphs], dtype=torch.long)
    source_sample_path = [str(g["sample_path"]) for g in lp_graphs]
    topk_per_milp = torch.as_tensor(topk_per_milp, dtype=torch.long)
    top_k_max = int(topk_per_milp.max().item()) if topk_per_milp.numel() > 0 else 0

    return {
        "num_milp_graphs": int(len(batch)),
        "num_lp_graphs": int(len(lp_graphs)),
        "topk_per_milp": topk_per_milp.contiguous(),
        "top_k_max": top_k_max,
        "constraint_features": constraint_features.contiguous(),
        "variable_features": variable_features.contiguous(),
        "edge_index": edge_index.contiguous(),
        "edge_attr": edge_attr.contiguous(),
        "n_constraints_per_graph": n_constraints_per_graph.contiguous(),
        "n_variables_per_graph": n_variables_per_graph.contiguous(),
        "branch_var_index": branch_var_index.contiguous(),
        "branch_dir": branch_dir.contiguous(),
        "topk_rank": topk_rank.contiguous(),
        "parent_obj": parent_obj.contiguous(),
        "cutoffbound": cutoffbound.contiguous(),
        "target_y": target_y.contiguous(),
        "target_alpha": target_alpha.contiguous(),
        "target_beta": target_beta.contiguous(),
        "target_obj": target_obj.contiguous(),
        "target_score": target_score.contiguous(),
        "source_sample_index": source_sample_index.contiguous(),
        "source_sample_path": source_sample_path,
    }
