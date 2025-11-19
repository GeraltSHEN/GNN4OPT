import gzip
import json
import os
import random
import pickle
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from models import (
    GNNPolicy,
    Holo,
    PowerMethod,
    ProductTupleEncoder,
    SymmetryBreakingGNN,
)


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
        n_variables_per_graph: int,
        n_constraints_per_graph: int,
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
        binarize_edge_features: bool = True,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = [str(path) for path in _ensure_sequence(sample_files)]
        self.edge_nfeats = edge_nfeats
        self.binarize_edge_features = binarize_edge_features

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        sample = load_gzip(self.sample_files[index])
        sample_state, _, sample_action, sample_action_set, sample_scores = sample["data"]

        constraint_dict, edge_dict, variable_dict = sample_state
        constraint_features = torch.as_tensor(constraint_dict["values"], dtype=torch.float32)
        edge_indices = torch.as_tensor(edge_dict["indices"], dtype=torch.int64)
        edge_features = torch.as_tensor(edge_dict["values"], dtype=torch.float32)
        variable_names = variable_dict["names"]
        variable_values = np.asarray(variable_dict["values"], dtype=np.float32)
        # pick the relevant variable features
        feature_indices = {
            name: variable_names.index(name)
            for name in ("coef_normalized", "sol_is_at_lb", "sol_is_at_ub", "sol_val")
        }
        coef_normalized = variable_values[:, feature_indices["coef_normalized"]]
        sol_is_at_lb = variable_values[:, feature_indices["sol_is_at_lb"]]
        sol_is_at_ub = variable_values[:, feature_indices["sol_is_at_ub"]]
        sol_val = variable_values[:, feature_indices["sol_val"]]
        fixed_mask = (sol_is_at_lb == 1) & (sol_is_at_ub == 1)
        is_fixed_to_1 = (fixed_mask & (sol_val == 1)).astype(np.float32)
        is_fixed_to_0 = (fixed_mask & (sol_val == 0)).astype(np.float32)
        is_not_fixed = 1.0 - is_fixed_to_1 - is_fixed_to_0
        variable_features = torch.as_tensor(
            np.stack(
                [coef_normalized, is_fixed_to_1, is_fixed_to_0, is_not_fixed],
                axis=-1,
            ),
            dtype=torch.float32,
        )

        if self.binarize_edge_features:
            non_zero_mask = edge_features != 0
            edge_features = torch.where(
                non_zero_mask,
                torch.ones_like(edge_features),
                torch.zeros_like(edge_features),
            )

        if self.edge_nfeats == 2:
            norm = torch.linalg.norm(edge_features)
            ef_norm = torch.where(norm > 0, edge_features / norm, torch.zeros_like(edge_features))
            edge_features = torch.cat((edge_features, ef_norm), dim=-1)
        
        candidates = torch.as_tensor(sample_action_set, dtype=torch.int64)
        candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)
        candidate_choices = torch.where(candidates == torch.as_tensor(sample_action, dtype=torch.int64))[0][0]

        graph = BipartiteNodeData(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            candidates,
            len(candidates),
            candidate_choices,
            candidate_scores,
            variable_features.size(0),
            constraint_features.size(0),
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


def load_data(args, for_training: bool = True) -> Dict[str, Union[torch.utils.data.DataLoader, Sequence[Path]]]:
    """Load train/val/test splits as torch_geometric DataLoaders based on CLI args."""
    dataset_root = Path(f"{args.dataset_path}")
    file_pattern = getattr(args, "file_pattern", "sample_*.pkl")
    edge_nfeats = getattr(args, "edge_nfeats", 1)
    binarize_edge_features = getattr(args, "binarize_edge_features", True)

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
    for split_name, cfg in splits.items():
        split_dir = dataset_root / cfg["subdir"]
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
                binarize_edge_features=binarize_edge_features,
            )
            data[split_name] = DataLoader(dataset, 
                                          batch_size=cfg["batch_size"], 
                                          shuffle=cfg["shuffle"], 
                                          num_workers=0, pin_memory=False)
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
    r = args.r
    num_heads = getattr(args, "num_heads", 0)
    isab_num_inds = getattr(args, "isab_num_inds", None)
    if args.r == 1:
         output_size = 1
    else:
        raise NotImplementedError("Only r=1 is supported for now.")

    if args.model == "raw":
        tuple_encoder = ProductTupleEncoder(emb_size)
    elif args.model == "holo":
        if args.symmetry_breaking_model == "power_method":
            symmetry_breaking_model = PowerMethod(args.power, emb_size + 1)
        elif args.symmetry_breaking_model == "gnn":
            symmetry_breaking_model = SymmetryBreakingGNN(emb_size + 1, emb_size + 1)
        else:
            raise ValueError(
                f"Unkown symmetry breaking model {args.symmetry_breaking_model}"
            )
        tuple_encoder = Holo(
            n_breakings=args.n_breakings,
            symmetry_breaking_model=symmetry_breaking_model,
            num_heads=num_heads,
            isab_num_inds=isab_num_inds,
        )
    else:
        raise NotImplementedError()

    model = GNNPolicy(emb_size, cons_nfeats, edge_nfeats, var_nfeats, output_size,
                      n_layers, tuple_encoder, r=r)

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

    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
    else:
        last_checkpoint = str(step) + '.pth'
    
    if step:
        save_path = os.path.join(save_dir, last_checkpoint)
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
