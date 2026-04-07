import gzip
import json
import os
import random
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from models import GNNPolicy, HeuristicPolicy, StackedBipartiteGNN


def _ensure_sequence(sample_files: Union[str, Path, Sequence[Union[str, Path]]]) -> Sequence[str]:
    if isinstance(sample_files, (str, Path)):
        return [str(sample_files)]
    if isinstance(sample_files, np.ndarray):
        return [str(x) for x in sample_files.tolist()]
    return [str(x) for x in sample_files]


class BipartiteNodeData(Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
        candidates: torch.Tensor,
        nb_candidates: int,
        candidate_choices: int,
        candidate_scores: torch.Tensor,
        candidate_relevance: torch.Tensor,
        n_variables_per_graph: int,
        n_constraints_per_graph: int
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choices
        self.candidate_scores = candidate_scores
        self.candidate_relevance = candidate_relevance
        self.n_variables_per_graph = n_variables_per_graph
        self.n_constraints_per_graph = n_constraints_per_graph

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(
        self,
        sample_files: Sequence[Union[str, Path]],
        edge_nfeats: int = 1,
        two_fwl: bool = False,
        args=None,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = [str(path) for path in _ensure_sequence(sample_files)]
        self.edge_nfeats = edge_nfeats
        self.two_fwl = two_fwl
        self.args = args
        self._feature_index_cache: Dict[Tuple[str, ...], Dict[str, int]] = {}
        model_name = str(getattr(args, "model", "")).lower() if args is not None else ""
        self._attach_postprocess_data = bool(
            getattr(args, "attach_postprocess_data", model_name == "heuristics")
        ) if args is not None else False
        loss_option = str(getattr(args, "loss_option", "")).lower() if args is not None else ""
        self._attach_topk_targets = bool(
            getattr(args, "attach_topk_targets", loss_option == "regression")
        ) if args is not None else False
        self._dual_option = int(getattr(args, "dual_option", 1)) if args is not None else 1
        self._universal_cutoffbound = float(
            getattr(args, "universal_cutoffbound", 1e6)
        ) if args is not None else 1e6
        self._topk_target_keys = self._resolve_topk_target_keys()

    def len(self):
        return len(self.sample_files)

    def _name_to_index(self, names: Sequence[str]) -> Dict[str, int]:
        key = tuple(names)
        cached = self._feature_index_cache.get(key)
        if cached is None:
            cached = {name: idx for idx, name in enumerate(names)}
            self._feature_index_cache[key] = cached
        return dict(cached)

    def _resolve_topk_target_keys(self) -> Sequence[str]:
        if self._dual_option == 1:
            return (
                "top8_regression_targets_option1",
                "top8_regression_targets",
                "top8_regression_targets_option2",
            )
        if self._dual_option == 2:
            return (
                "top8_regression_targets_option2",
                "top8_regression_targets_option1",
                "top8_regression_targets",
            )
        return (
            "top8_regression_targets",
            "top8_regression_targets_option1",
            "top8_regression_targets_option2",
        )

    def _effective_cutoffbound(self, sample_cutoffbound: float) -> float:
        if self._dual_option in (1, 3):
            return float(sample_cutoffbound)
        if self._dual_option in (2, 4):
            return float(self._universal_cutoffbound)
        raise ValueError(
            f"Unsupported dual_option '{self._dual_option}'. Use one of: 1, 2, 3, 4."
        )

    def _attach_postprocess_fields(
        self,
        graph: BipartiteNodeData,
        sample_data: Sequence[Any],
        constraint_default_features: torch.Tensor,
        constraint_default_feature_indices: Dict[str, int],
        variable_default_features: torch.Tensor,
        variable_default_feature_indices: Dict[str, int],
        variable_dict: Dict[str, Any],
    ) -> None:
        if len(sample_data) < 6:
            return
        bias_idx = constraint_default_feature_indices.get("bias")
        sol_val_idx = variable_default_feature_indices.get("sol_val")
        coef_idx = variable_default_feature_indices.get("coef_normalized")
        if bias_idx is None or sol_val_idx is None or coef_idx is None:
            return

        rhs = -constraint_default_features[:, bias_idx].to(torch.float32).contiguous()
        lp_solution = variable_default_features[:, sol_val_idx].to(torch.float32).contiguous()
        obj_coeffs = variable_default_features[:, coef_idx].to(torch.float32).contiguous()

        n_vars = int(variable_default_features.size(0))
        parent_lbs = torch.zeros(n_vars, dtype=torch.float32)
        parent_ubs = torch.ones(n_vars, dtype=torch.float32)
        type0_idx = variable_default_feature_indices.get("type_0")
        if type0_idx is not None:
            binary = variable_default_features[:, type0_idx] > 0.5
            parent_lbs[binary] = 0.0
            parent_ubs[binary] = 1.0

        reconstruction = variable_dict.get("reconstruction", {})
        if "lbs" in reconstruction:
            rec_lbs = torch.as_tensor(reconstruction["lbs"], dtype=torch.float32).reshape(-1)
            if rec_lbs.numel() == n_vars:
                parent_lbs = rec_lbs.contiguous()
        if "ubs" in reconstruction:
            rec_ubs = torch.as_tensor(reconstruction["ubs"], dtype=torch.float32).reshape(-1)
            if rec_ubs.numel() == n_vars:
                parent_ubs = rec_ubs.contiguous()

        objective_offset = float(reconstruction.get("objective_offset", 0.0))
        parent_obj = (
            torch.dot(obj_coeffs.to(torch.float64), lp_solution.to(torch.float64)).to(torch.float32)
            + objective_offset
        )
        sample_cutoffbound = float(sample_data[5])
        cutoffbound = self._effective_cutoffbound(sample_cutoffbound)

        graph.post_rhs = rhs
        graph.post_parent_lbs = parent_lbs
        graph.post_parent_ubs = parent_ubs
        graph.post_lp_solution = lp_solution
        graph.post_parent_obj = torch.tensor([float(parent_obj)], dtype=torch.float32)
        graph.post_cutoffbound = torch.tensor([cutoffbound], dtype=torch.float32)
        graph.post_objective_offset = torch.tensor([objective_offset], dtype=torch.float32)

    def _attach_topk_target_fields(self, graph: BipartiteNodeData, sample: Dict[str, Any]) -> None:
        selected = None
        for key in self._topk_target_keys:
            if key in sample:
                selected = sample[key]
                break
        if selected is None:
            return
        graph.topk_targets = selected

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        sample = load_gzip(self.sample_files[index])
        sample_data = sample["data"]
        if not isinstance(sample_data, (list, tuple)) or len(sample_data) < 5:
            raise ValueError(f"Expected sample['data'] with at least 5 entries, got {type(sample_data)} len={len(sample_data) if hasattr(sample_data, '__len__') else 'n/a'}")
        sample_state = sample_data[0]
        sample_action = sample_data[2]
        sample_action_set = sample_data[3]
        sample_scores = sample_data[4]

        constraint_dict, edge_dict, variable_dict = sample_state

        edge_indices = torch.as_tensor(edge_dict["indices"], dtype=torch.int64)
        edge_features = torch.as_tensor(edge_dict["values"], dtype=torch.float32)

        variable_names = list(variable_dict["names"])
        variable_default_feature_indices = self._name_to_index(variable_names)
        variable_feature_indices = dict(variable_default_feature_indices)
        variable_default_features = torch.as_tensor(variable_dict["values"], dtype=torch.float32)

        constraint_names = list(constraint_dict["names"])
        constraint_default_feature_indices = self._name_to_index(constraint_names)
        constraint_feature_indices = dict(constraint_default_feature_indices)
        constraint_default_features = torch.as_tensor(constraint_dict["values"], dtype=torch.float32)

        # pick problem data features
        use_default_features = (
            bool(getattr(self.args, "use_default_features", True))
            if self.args is not None
            else True
        )
        use_cutoffbound_feature = (
            bool(getattr(self.args, "use_cutoffbound_feature", False))
            if self.args is not None
            else True
        )
        cutoff_feature_name = "cutoffbound_normalized"
        if use_default_features:
            if use_cutoffbound_feature:
                # if cutoff_feature_name not in variable_feature_indices:
                #     raise KeyError(f"Missing variable feature '{cutoff_feature_name}' in sample {self.sample_files[index]}")
                # if cutoff_feature_name not in constraint_feature_indices:
                #     raise KeyError(f"Missing constraint feature '{cutoff_feature_name}' in sample {self.sample_files[index]}")
                variable_features = variable_default_features
                constraint_features = constraint_default_features
            else:
                variable_keep_names = [name for name in variable_names if name != cutoff_feature_name]
                constraint_keep_names = [name for name in constraint_names if name != cutoff_feature_name]
                variable_keep_idxs = [variable_feature_indices[name] for name in variable_keep_names]
                constraint_keep_idxs = [constraint_feature_indices[name] for name in constraint_keep_names]

                variable_features = variable_default_features[:, variable_keep_idxs]
                constraint_features = constraint_default_features[:, constraint_keep_idxs]
                variable_feature_indices = {
                    name: new_index for new_index, name in enumerate(variable_keep_names)
                }
                constraint_feature_indices = {
                    name: new_index for new_index, name in enumerate(constraint_keep_names)
                }
        else:
            variable_required = ["type_0", "type_1", "type_2", "type_3", 
                                 "has_lb", "has_ub", "sol_is_at_lb", "sol_is_at_ub", "sol_frac",
                                 "coef_normalized", "sol_val"]
            constraint_required = ["bias", "dualsol_val_normalized"]
            if use_cutoffbound_feature:
                variable_required.append(cutoff_feature_name)
                constraint_required.append(cutoff_feature_name)

            missing_variable = [name for name in variable_required if name not in variable_feature_indices]
            missing_constraint = [name for name in constraint_required if name not in constraint_feature_indices]
            # if missing_variable:
            #     raise KeyError(
            #         f"Missing variable features {missing_variable} in sample {self.sample_files[index]}"
            #     )
            # if missing_constraint:
            #     raise KeyError(
            #         f"Missing constraint features {missing_constraint} in sample {self.sample_files[index]}"
            #     )

            variable_features = torch.stack([variable_default_features[:, variable_feature_indices[name]]
                                    for name in variable_required], 
                                    axis=-1)
            constraint_features = torch.stack([constraint_default_features[:, constraint_feature_indices[name]]
                                    for name in constraint_required], 
                                    axis=-1)
            variable_feature_indices = {name: new_index for name, new_index in zip(variable_required, range(len(variable_required)))}
            constraint_feature_indices = {name: new_index for name, new_index in zip(constraint_required, range(len(constraint_required)))}
    
        if self.edge_nfeats == 2:
            raise ValueError("edge_nfeats == 2 disabled")
        
        candidate_choice_node_id = sample_action
        candidates = torch.as_tensor(sample_action_set, dtype=torch.int64)
        candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)

        # 02_generate_samples' candidates seem to include some variables whose LP solution is at 0 or 1
        # it may consider all integer variables that haven't been fixed as candidates
        # it seems to assign a very small candidate score (close to 0) to those "candidates" whose LP solution is at 0 or 1
        # NOTE: perhaps clean up the candidates and candidate_scores to include true candidates only. 
        remove_bad_candidates = (
            bool(getattr(self.args, "remove_bad_candidates", False))
            if self.args is not None
            else False
        )
        if remove_bad_candidates:
            candidates, candidate_scores, candidate_choice_node_id = self.clean_candidates(candidates, candidate_scores, candidate_choice_node_id, 
                                                                 variable_features, variable_feature_indices, index)
        candidate_choices = torch.where(candidates == torch.as_tensor(candidate_choice_node_id, dtype=torch.int64))[0][0]
        
        # add is_fixed (variables that were considered as candidates by 02_generate_samples) indicator to variable features
        # as a variable can be integer or continuous, to make it simple, we add is_not_fixed indicator instead of is_fixed
        is_not_fixed_feature = torch.zeros(variable_features.size(0), dtype=torch.float32)
        is_not_fixed_feature[torch.as_tensor(sample_action_set, dtype=torch.int64)] = 1.0
        # add candidate (variables that will be considered in ranking) indicator to variable features
        candidates_feature = torch.zeros(variable_features.size(0), dtype=torch.float32)
        candidates_feature[candidates] = 1.0
        # add the two additional features to variable features
        variable_features = torch.cat([variable_features,
                                        is_not_fixed_feature.unsqueeze(-1), 
                                       candidates_feature.unsqueeze(-1)], dim=-1)

        candidate_relevance = self.assign_candidate_relevance(candidate_scores)
        graph = BipartiteNodeData(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            candidates,
            len(candidates),
            candidate_choices,
            candidate_scores,
            candidate_relevance,
            variable_features.size(0),
            constraint_features.size(0),
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        if self._attach_postprocess_data:
            self._attach_postprocess_fields(
                graph,
                sample_data,
                constraint_default_features,
                constraint_default_feature_indices,
                variable_default_features,
                variable_default_feature_indices,
                variable_dict,
            )
        if self._attach_topk_targets:
            self._attach_topk_target_fields(graph, sample)
        return graph
    
    def clean_candidates(self, candidates: torch.Tensor, candidate_scores: torch.Tensor, 
                         candidate_choice_node_id: int, 
                         variable_features: torch.Tensor, 
                         variable_feature_indices: Dict[str, int], index):
        sol_is_not_at_lb = variable_features[candidates, variable_feature_indices["sol_is_at_lb"]] == 0
        sol_is_not_at_ub = variable_features[candidates, variable_feature_indices["sol_is_at_ub"]] == 0
        sol_is_not_at_lub = sol_is_not_at_lb & sol_is_not_at_ub
        cleaned_candidates = candidates[sol_is_not_at_lub]
        cleaned_candidate_scores = candidate_scores[sol_is_not_at_lub]
        if cleaned_candidates.numel() < 1:
            raise ValueError("no candidate exists after cleaning")
        if candidate_choice_node_id not in cleaned_candidates:
            # NOTE: Rare case: all candidates are equally bad (as bad as nodes whose LP solution is at LB/UB.) 
            # This usually happens when branching on 'real candidate' produces the same optimal obj value.
            # Replace the candidate choice with a new one in the cleaned candidates.
            new_candidate_choice = cleaned_candidate_scores.argmax().item()
            new_candidate_choice_node_id = cleaned_candidates[new_candidate_choice].item()
            candidate_choice_node_id = new_candidate_choice_node_id
        return cleaned_candidates, cleaned_candidate_scores, candidate_choice_node_id
    
    def assign_candidate_relevance(self, candidate_scores: torch.Tensor) -> torch.Tensor:
        args = getattr(self, "args", None)
        loss_option = getattr(args, "loss_option", None) if args is not None else None
        tier1_ub = float(getattr(args, "tier1_ub", 0.0)) if args is not None else 0.0
        relevance_type = getattr(args, "relevance_type", "linear") if args is not None else "linear"

        if relevance_type == "linear":
            max_score = candidate_scores.max()
            score_gap = max_score - candidate_scores
            tier1_mask = score_gap <= tier1_ub
            return tier1_mask.to(torch.long)

        elif relevance_type == "true_score":
            true_scores = candidate_scores.clamp(min=0)
            # denom = true_scores.max().clamp(min=1e-8)
            denom = true_scores.max()
            normalized_scores = true_scores / denom
            return normalized_scores
        
        else:
            raise ValueError(f"Unknown relevance_type: {relevance_type}")


def load_data(args, for_training: bool = True) -> Dict[str, Union[torch.utils.data.DataLoader, Sequence[Path]]]:
    """Load train/val/test splits as torch_geometric DataLoaders based on CLI args."""
    dataset_root = Path(f"{args.dataset_path}")
    print(f'load dataset from {dataset_root}')
    file_pattern = getattr(args, "file_pattern", "sample_*.pkl")
    edge_nfeats = getattr(args, "edge_nfeats", 1)
    subsamples = getattr(args, "subsamples", None)
    subsample_manifest_name = "sample_files.txt"
    subsample_root = None
    if subsamples is not None:
        subsamples_str = str(subsamples).strip().rstrip(",").strip()
        if subsamples_str and subsamples_str.lower() not in {"none", "null"}:
            subsample_root = Path(subsamples_str)
            if not subsample_root.is_absolute():
                subsample_root = dataset_root.parent / subsample_root

    if for_training:
        splits = {
            "train": {
                "subdir": getattr(args, "train_split", "train"),
                "batch_size": getattr(args, "train_batch_size", getattr(args, "batch_size", 32)),
                "shuffle": getattr(args, "train_shuffle", True),
            },
            "val": {
                "subdir": getattr(args, "val_split", "valid"),
                "batch_size": getattr(args, "val_batch_size", getattr(args, "batch_size", 32)),
                "shuffle": getattr(args, "val_shuffle", False),
            },
            "test": {
                "subdir": getattr(args, "test_split", "test"),
                "batch_size": getattr(args, "test_batch_size", getattr(args, "batch_size", 32)),
                "shuffle": getattr(args, "test_shuffle", False),
            },
        }
    else:
        splits = {
            "train": {
                "subdir": getattr(args, "train_split", "train"),
                "batch_size": getattr(args, "eval_batch_size"),
                "shuffle": False,
            },
            "val": {
                "subdir": getattr(args, "val_split", "valid"),
                "batch_size": getattr(args, "eval_batch_size"),
                "shuffle": False,
            },
            "test": {
                "subdir": getattr(args, "test_split", "test"),
                "batch_size": getattr(args, "eval_batch_size"),
                "shuffle": False,
            },
        }


    data: Dict[str, Union[torch.utils.data.DataLoader, Sequence[Path]]] = {}
    metadata: Dict[str, Sequence[Path]] = {}
    num_workers = int(getattr(args, "num_workers", 8))
    pin_memory = bool(getattr(args, "pin_memory", True))
    persistent_workers = bool(getattr(args, "persistent_workers", num_workers > 0))
    prefetch_factor = getattr(args, "prefetch_factor", None)
    for split_name, cfg in splits.items():
        split_dir = dataset_root / cfg["subdir"]
        sample_files = None
        if subsample_root is not None:
            manifest_path = subsample_root / cfg["subdir"] / subsample_manifest_name
            if manifest_path.exists():
                manifest_entries = manifest_path.read_text(encoding="utf-8").splitlines()
                sample_files = []
                for entry in manifest_entries:
                    entry = entry.strip()
                    if not entry:
                        continue
                    sample_path = Path(entry)
                    if not sample_path.is_absolute():
                        sample_path = split_dir / sample_path
                    sample_files.append(sample_path)
                print(f"using subsample manifest for {split_name}: {manifest_path}")
        if sample_files is None:
            sample_files = sorted(split_dir.glob(file_pattern))
        max_split_samples = getattr(
            args,
            f"max_{split_name}_samples",
            getattr(args, "max_samples_per_split", None),
        )
        if max_split_samples is not None:
            max_split_samples = int(max_split_samples)
            sample_files = sample_files[:max_split_samples]
        metadata[f"{split_name}_files"] = sample_files
        if sample_files:
            dataset = GraphDataset(
                sample_files,
                edge_nfeats=edge_nfeats,
                args=args,
            )
            loader_kwargs = dict(
                batch_size=cfg["batch_size"],
                shuffle=cfg["shuffle"],
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = persistent_workers
                if prefetch_factor is not None:
                    loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            data[split_name] = DataLoader(dataset, **loader_kwargs)
        else:
            data[split_name] = None
    data.update(metadata)
    return data


def get_optimizer(args, model):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def load_model(args, cons_nfeats, edge_nfeats, var_nfeats) -> torch.nn.Module:
    emb_size = args.hidden_channels
    n_layers = args.num_layers
    num_heads = args.num_heads
    isab_num_inds = args.isab_num_inds
    use_set_transformer = getattr(args, "use_set_transformer", True)
    gnn_backbone = getattr(args, "gnn_backbone", "bipartite")
    sage_mlp_layers = int(getattr(args, "sage_mlp_layers", 2))
    output_size = 1

    def _load_model_state_exact(model: torch.nn.Module, checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=args.device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        print(f"load exact checkpoint {checkpoint_path}.")
        return model

    def _load_breaking_selector_model(model, selector_path: Union[str, Path]):
        if not selector_path:
            raise ValueError("`breaking_selector_model_path` must be provided when using the holo model.")
        selector_path = Path(selector_path)
        if not selector_path.exists():
            raise FileNotFoundError(f"breaking selector checkpoint not found: {selector_path}")
        checkpoints = [x for x in os.listdir(selector_path) 
                       if not x.startswith('events') and not x.endswith('.json') and not x.endswith('.pkl')]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
            selector_path = selector_path / last_checkpoint

        state = torch.load(selector_path, map_location=args.device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        print(f"load {selector_path} for breaking selector model.")
        return model

    if args.model == "raw":
        model = GNNPolicy(
            emb_size,
            cons_nfeats,
            edge_nfeats,
            var_nfeats,
            output_size,
            n_layers,
            holo=None,
            gnn_backbone=gnn_backbone,
            sage_mlp_layers=sage_mlp_layers,
        )
    elif args.model == "holo":
        selector_layers = n_layers
        selector_path = getattr(args, "breaking_selector_model_path", None)
        breaking_selector_model = GNNPolicy(emb_size, 
                                            cons_nfeats, 
                                            edge_nfeats, 
                                            var_nfeats,
                                            output_size,
                                            selector_layers,
                                            holo=None,
                                            gnn_backbone=gnn_backbone,
                                            sage_mlp_layers=sage_mlp_layers)
        breaking_selector_model = _load_breaking_selector_model(breaking_selector_model, 
                                                                selector_path)
        breaking_selector_model = breaking_selector_model.to(args.device)
        breaking_selector_model.eval()
        for param in breaking_selector_model.parameters():
            param.requires_grad_(False)

        sym_break_layers = getattr(args, "sym_break_layers", 2)
        symmetry_breaking_model = StackedBipartiteGNN(
            hidden_channels=emb_size + 2,
            edge_nfeats=edge_nfeats,
            n_layers=sym_break_layers,
        )

        # holo = SetCoverHolo(
        #     n_breakings=args.n_breakings,
        #     breaking_selector_model=breaking_selector_model,
        #     symmetry_breaking_model=symmetry_breaking_model,
        #     num_heads=num_heads,
        #     isab_num_inds=isab_num_inds,
        #     mp_layers=getattr(args, "mp_layers", 2),
        #     edge_nfeats=edge_nfeats,
        #     use_set_transformer=use_set_transformer,
        # )
        holo = None
        model = GNNPolicy(emb_size, 
                          cons_nfeats, 
                          edge_nfeats, 
                          var_nfeats,
                          output_size,
                          n_layers,
                          holo=holo,
                          gnn_backbone=gnn_backbone,
                          sage_mlp_layers=sage_mlp_layers)
    elif args.model == "heuristics":
        branching_model_path = getattr(args, "branching_model_path", None)
        if not branching_model_path:
            raise ValueError("`branching_model_path` must be provided when model='heuristics'.")

        branching_model = GNNPolicy(
            emb_size,
            cons_nfeats,
            edge_nfeats,
            var_nfeats,
            output_size,
            n_layers,
            holo=None,
            gnn_backbone=gnn_backbone,
            sage_mlp_layers=sage_mlp_layers,
        )
        branching_model = _load_model_state_exact(branching_model, branching_model_path)
        branching_model = branching_model.to(args.device)
        branching_model.eval()
        for param in branching_model.parameters():
            param.requires_grad_(False)

        model = HeuristicPolicy(
            emb_size,
            cons_nfeats,
            edge_nfeats,
            var_nfeats,
            output_size,
            n_layers,
            branching_model=branching_model,
            gnn_backbone=gnn_backbone,
            sage_mlp_layers=sage_mlp_layers,
            n_branching=int(getattr(args, "n_branching", 1)),
            use_cutoff_minimum=bool(getattr(args, "use_cutoff_minimum", True)),
        )
    else:
        raise NotImplementedError(f"Unknown model type '{args.model}'.")

    return model.to(args.device)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

"""
Utility functions to load and save files
"""
def load_gzip(path: Union[str, Path]):
    """Load a gzip-compressed pickle file"""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)

def save_json(path, d):
    dir_path, file_name = os.path.split(path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(path, 'w') as file:
        json.dump(d, file, indent=4)

def load_json(path, default=[]):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        d = json.load(f)
    return d


"""
Utility functions to load and save torch model checkpoints 
"""
def load_checkpoint(model, optimizer=None, step='max', save_dir='checkpoints', device='cpu', exclude_keys=[]):
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events') and not x.endswith('.json') and not x.endswith('.pkl')]

    save_path = None
    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
            save_path = os.path.join(save_dir, last_checkpoint)
    else:
        step = int(step)
        last_checkpoint = str(step) + '.pth'
        save_path = os.path.join(save_dir, last_checkpoint)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

    if save_path is not None:
        state = torch.load(save_path, map_location=device)

        if len(exclude_keys) > 0:
            model_state = state['model'] if 'model' in state else state
            model_state = {k: v for k, v in model_state.items() if not any(k.startswith(exclude_key) for exclude_key in exclude_keys)}
            model.load_state_dict(model_state, strict=False)
            
            if optimizer and 'optimizer' in state:
                optimizer_state_dict = state['optimizer']
                excluded_param_ids = {
                    id(param) for name, param in model.named_parameters() if any(name.startswith(exclude_key) for exclude_key in exclude_keys)
                }
                optimizer_state_dict['state'] = {k: v for k, v in optimizer_state_dict['state'].items() if k not in excluded_param_ids}
                optimizer.load_state_dict(optimizer_state_dict)
        else:
            model_state = state['model'] if 'model' in state else state
            model.load_state_dict(model_state)
            if optimizer and 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
        
        print('Loaded checkpoint %s' % save_path)
    
    return step

def save_checkpoint(model, step, optimizer=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + '.pth')

    if optimizer is None:
        torch.save(dict(model=model.state_dict()), save_path)
    else:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print('Saved checkpoint %s' % save_path)


def print_dash_str(message: str = "", width: int = 120) -> None:
    if not message:
        print("-" * width)
        return
    if len(message) + 2 >= width:
        print(message)
        return
    pad_total = width - len(message) - 2
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    print(f"{'-' * left_pad} {message} {'-' * right_pad}")
