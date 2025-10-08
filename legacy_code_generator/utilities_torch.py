import gzip
import pickle
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch


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
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError("torch_geometric must be installed to use load_data_pyg utilities.") from exc

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
    """Minimal dataset wrapper yielding torch_geometric data objects per sample file."""

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
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise ImportError("torch_geometric must be installed to build a PyG DataLoader.") from exc

    dataset = PyGSampleDataset(sample_files)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
