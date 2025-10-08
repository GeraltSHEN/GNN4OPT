import gzip
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

from models import (
    Classifier,
    Holo,
    GCNDataEncoder,
    PowerMethod,
    ProductTupleEncoder,
    SymmetryBreakingGNN,
)

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader


TensorTuple = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _ensure_sequence(sample_files: Union[str, Path, Sequence[Union[str, Path]]]) -> Sequence[str]:
    if isinstance(sample_files, (str, Path)):
        return [str(sample_files)]
    if isinstance(sample_files, np.ndarray):
        return [str(x) for x in sample_files.tolist()]
    return [str(x) for x in sample_files]


def _load_single_pyg_sample(sample_file: Union[str, Path]):
    """Load a single sample file and convert it into a torch_geometric HeteroData object."""
    with gzip.open(sample_file, "rb") as f:
        sample = pickle.load(f)

    sample_state, _, sample_action, sample_cands, cand_scores = sample["data"]

    sample_cands = np.asarray(sample_cands)
    cand_choice = int(np.where(sample_cands == sample_action)[0][0])

    c, e, v = sample_state

    data = HeteroData()
    data["constraint"].x = torch.as_tensor(c["values"], dtype=torch.float32)
    data["variable"].x = torch.as_tensor(v["values"], dtype=torch.float32)

    edge_index = torch.as_tensor(e["indices"], dtype=torch.int64)
    if edge_index.numel() > 0:
        data["constraint", "to", "variable"].edge_index = edge_index
        data["constraint", "to", "variable"].edge_attr = torch.as_tensor(e["values"], dtype=torch.float32)

    data.num_constraints = torch.tensor(c["values"].shape[0], dtype=torch.int32)
    data.num_variables = torch.tensor(v["values"].shape[0], dtype=torch.int32)
    data.candidate_indices = torch.as_tensor(sample_cands, dtype=torch.int64)
    data.candidate_scores = torch.as_tensor(cand_scores, dtype=torch.float32)
    data.candidate_choice = torch.tensor(cand_choice, dtype=torch.int64)

    return data


def load_data_pyg(sample_files: Union[str, Path, Sequence[Union[str, Path]]]):
    """Load sample files and convert them into a list of torch_geometric HeteroData objects."""
    sample_files = _ensure_sequence(sample_files)
    return [_load_single_pyg_sample(sample_file) for sample_file in sample_files]


class PyGSampleDataset(torch.utils.data.Dataset):
    """Dataset wrapper yielding torch_geometric data objects per sample file."""

    def __init__(self, sample_files: Union[str, Path, Sequence[Union[str, Path]]]):
        self.sample_files = _ensure_sequence(sample_files)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index: int):
        return _load_single_pyg_sample(self.sample_files[index])


def make_pyg_dataloader(
    sample_files: Union[str, Path, Sequence[Union[str, Path]]],
    batch_size: int,
    shuffle: bool = False,
    **kwargs,
):
    """Construct a torch_geometric DataLoader over the provided sample files."""

    dataset = PyGSampleDataset(sample_files)
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def load_data(args) -> Dict[str, Union[torch.utils.data.DataLoader, Sequence[Path]]]:
    """Load train/val/test splits as torch_geometric DataLoaders based on CLI args."""
    dataset_root = Path(f"{args.dataset_path}")
    file_pattern = getattr(args, "file_pattern", "sample_*.pkl")

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
        metadata[f"{split_name}_files"] = sample_files
        if sample_files:
            data[split_name] = make_pyg_dataloader(
                sample_files, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"]
            )
        else:
            data[split_name] = None

    data.update(metadata)
    return data


def load_model(args):
    data_encoder = GCNDataEncoder(
        in_channels=args.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    if args.model == "gcn":
        tuple_encoder = ProductTupleEncoder()
        in_dim = args.hidden_channels
    elif args.model == "holo":
        in_dim = args.hidden_channels + 1
        if args.symmetry_breaking_model == "power_method":
            symmetry_breaking_model = PowerMethod(args.power, in_dim)
        elif args.symmetry_breaking_model == "gnn":
            symmetry_breaking_model = SymmetryBreakingGNN(in_dim, in_dim)
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
    out_dim = args.out_dim
    model = Classifier(
        data_encoder,
        tuple_encoder,
        in_dim=in_dim,
        out_dim=out_dim,
        linear_classifier=args.linear_classifier,
        train_head_only=args.pretrained_path is not None,
    )

    print(f"Model Architecture: \n{model}")
    model = model.to(args.device)
    return model


class CooSampler:
    def __init__(self, coos, values, batch_size: int, shuffle: bool = False):
        assert coos.size(1) == values.size(0)
        self.coos = coos
        self.values = values
        self.shuffle = shuffle
        n_groups, rem = divmod(self.values.size(0), batch_size)

        self.batch_sizes = [batch_size] * n_groups
        if rem > 0:
            self.batch_sizes.append(rem)

    def __len__(self):
        return len(self.batch_sizes)

    def __iter__(self):
        size = self.values.size(0)
        perm_coos = self.coos
        perm_values = self.values
        if self.shuffle:
            perm = torch.randperm(size)
            perm_coos = self.coos[:, perm]
            perm_values = self.values[perm]

        return iter(
            [
                (coos, values)
                for (coos, values) in zip(
                    torch.split(perm_coos, self.batch_sizes, dim=1),
                    torch.split(perm_values, self.batch_sizes, dim=0),
                )
            ]
        )


def get_optimizer(args, model):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
