import gzip
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


def load_gzip(path: Union[str, Path]):
    """Load a gzip-compressed pickle file"""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


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
        candidate_choice: int,
        candidate_scores: torch.Tensor,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choice = candidate_choice
        self.candidate_scores = candidate_scores

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
        edge_nfeats: int = 2,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = [str(path) for path in _ensure_sequence(sample_files)]
        self.edge_nfeats = edge_nfeats

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
        variable_features = torch.as_tensor(variable_dict["values"], dtype=torch.float32)

        if self.edge_nfeats == 2:
            norm = torch.linalg.norm(edge_features)
            ef_norm = torch.where(norm > 0, edge_features / norm, torch.zeros_like(edge_features))
            edge_features = torch.cat((edge_features, ef_norm), dim=-1)
        
        candidates = torch.as_tensor(sample_action_set, dtype=torch.int64)
        candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)
        candidate_choice = torch.where(candidates == torch.as_tensor(sample_action, dtype=torch.int64))[0][0]

        graph = BipartiteNodeData(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            candidates,
            len(candidates),
            candidate_choice,
            candidate_scores,
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


def load_data(args) -> Dict[str, Union[torch.utils.data.DataLoader, Sequence[Path]]]:
    """Load train/val/test splits as torch_geometric DataLoaders based on CLI args."""
    dataset_root = Path(f"{args.dataset_path}")
    file_pattern = getattr(args, "file_pattern", "sample_*.pkl")
    edge_nfeats = getattr(args, "edge_nfeats", 1)

    splits = {
        "train": {
            "subdir": getattr(args, "train_split", "train"),
            "batch_size": getattr(args, "train_batch_size", getattr(args, "batch_size", 32)),
            "shuffle": getattr(args, "train_shuffle", True),
        },
        "val": {
            "subdir": getattr(args, "val_split", "valid"),
            "batch_size": getattr(args, "val_batch_size", getattr(args, "eval_batch_size", getattr(args, "batch_size", 32))),
            "shuffle": getattr(args, "val_shuffle", False),
        },
        "test": {
            "subdir": getattr(args, "test_split", "test"),
            "batch_size": getattr(args, "test_batch_size", getattr(args, "eval_batch_size", getattr(args, "batch_size", 32))),
            "shuffle": getattr(args, "test_shuffle", False),
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
            dataset = GraphDataset(sample_files, edge_nfeats=edge_nfeats)
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


def load_model(args, example_input: Data) -> torch.nn.Module:
    if example_input is None:
        raise ValueError("load_model requires a sample graph to infer feature dimensions.")

    emb_size = args.hidden_channels
    n_layers = args.num_layers
    r = args.r
    if args.r == 1:
         output_size = 1
    else:
        raise NotImplementedError("Only r=1 is supported for now.")

    cons_nfeats = example_input.constraint_features.size(-1)
    edge_nfeats = example_input.edge_attr.size(-1)
    var_nfeats = example_input.variable_features.size(-1)

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
        )
    else:
        raise NotImplementedError()

    model = GNNPolicy(emb_size, cons_nfeats, edge_nfeats, var_nfeats, output_size,
                      n_layers, tuple_encoder, r=r)

    return model.to(args.device)
