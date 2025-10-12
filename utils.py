import gzip
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from models import (
    GNNPolicy,
    Holo,
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
            data[split_name] = make_pyg_dataloader(
                sample_files, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"]
            )
        else:
            data[split_name] = None

    data.update(metadata)
    return data


def get_optimizer(args, model):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def _infer_feature_dims(sample: HeteroData) -> Tuple[int, int, int]:
    cons_dim = sample["constraint"].x.size(-1)
    var_dim = sample["variable"].x.size(-1)
    edge_attr = getattr(sample["constraint", "to", "variable"], "edge_attr", None)
    if edge_attr is None:
        edge_dim = 1
    else:
        edge_dim = edge_attr.size(-1)
    return cons_dim, var_dim, edge_dim


def _get_reference_sample(args) -> Optional[HeteroData]:
    dataset_root = Path(f"{args.dataset_path}")
    file_pattern = getattr(args, "file_pattern", "sample_*.pkl")
    splits = [
        getattr(args, "train_split", "train"),
        getattr(args, "val_split", "valid"),
        getattr(args, "test_split", "test"),
    ]
    for split in splits:
        split_dir = dataset_root / split
        sample_files = sorted(split_dir.glob(file_pattern))
        if sample_files:
            return _load_single_pyg_sample(sample_files[0])
    return None


def load_model(args) -> torch.nn.Module:
    torch.manual_seed(getattr(args, "seed", 42))
    device = getattr(args, "device", "cpu")

    reference_sample = _get_reference_sample(args)
    if reference_sample is None:
        raise FileNotFoundError(
            "Unable to locate any samples to infer feature dimensions. "
            "Please check dataset_path and file_pattern."
        )

    cons_dim, var_dim, edge_dim = _infer_feature_dims(reference_sample)

    emb_size = getattr(args, "hidden_channels", 64)
    encoder_kwargs = dict(
        emb_size=emb_size,
        cons_nfeats=cons_dim,
        var_nfeats=var_dim,
        edge_nfeats=edge_dim,
        num_layers=getattr(args, "num_layers", 2),
        conv_type=getattr(args, "conv_type", "sage"),
    )

    data_encoder = MILPEncoder(**encoder_kwargs)

    model_name = getattr(args, "model", "gnn_policy").lower()
    holo_aliases = {"holo", "bipartite_holo", "holo_tuple", "holo_power"}
    baseline_aliases = {"variable", "baseline", "gnn_policy"}

    if model_name in holo_aliases:
        breaker_choice = getattr(args, "holo_breaker", "gnn").lower()
        if model_name == "holo_power":
            breaker_choice = "power"
        if breaker_choice == "power":
            breaker = PowerMethod(
                k=getattr(args, "power_iterations", 2),
                out_dim=emb_size + 1,
            )
        else:
            breaker = SymmetryBreakingGNN(
                in_channels=emb_size + 1,
                hidden_channels=getattr(args, "holo_hidden_channels", emb_size),
            )
        tuple_encoder = Holo(
            n_breakings=getattr(args, "n_breakings", 8),
            symmetry_breaking_model=breaker,
        )
        head_in_dim = tuple_encoder.symmetry_breaking_model.out_dim
    elif model_name in baseline_aliases:
        tuple_encoder = ProductTupleEncoder()
        head_in_dim = emb_size
    else:
        raise ValueError(f"Unknown model '{model_name}'.")

    linear_classifier = getattr(args, "linear_classifier", False)
    train_head_only = getattr(args, "train_head_only", False)

    model = Classifier(
        data_encoder=data_encoder,
        tuple_encoder=tuple_encoder,
        in_dim=head_in_dim,
        out_dim=1,
        linear_classifier=linear_classifier,
        train_head_only=train_head_only,
    )
    model.to(device)
    return model
