import pickle
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class PreNormException(RuntimeError):
    """Raised to signal that a PreNormLayer updated its statistics."""


class PreNormLayer(nn.Module):
    """
    Normalize an input tensor to zero mean / unit variance using pre-computed
    statistics gathered during a dedicated pre-training pass.
    """

    def __init__(self, n_units: int, shift: bool = True, scale: bool = True):
        super().__init__()
        if not (shift or scale):
            raise ValueError("Either shift or scale must be enabled.")

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

        if shift:
            self.shift = nn.Parameter(torch.zeros(n_units), requires_grad=False)
        else:
            self.shift = None

        if scale:
            self.scale = nn.Parameter(torch.ones(n_units), requires_grad=False)
        else:
            self.scale = None

        self._reset_stats()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        output = input
        if self.shift is not None:
            output = output + self.shift
        if self.scale is not None:
            output = output * self.scale
        return output

    def start_updates(self):
        """Begin statistics accumulation."""
        self._reset_stats()
        self.waiting_updates = True
        self.received_updates = False

    def stop_updates(self):
        """Finalize statistics and freeze the affine parameters."""
        if self.count.item() <= 0:
            raise RuntimeError("Cannot stop updates before receiving statistics.")

        if self.shift is not None:
            self.shift.data.copy_(-self.avg)

        if self.scale is not None:
            safe_var = torch.where(self.var == 0, torch.ones_like(self.var), self.var)
            inv_std = torch.rsqrt(safe_var)
            self.scale.data.copy_(inv_std)

        self._clear_stats()
        self.waiting_updates = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_stats(self):
        device = (
            self.shift.device
            if self.shift is not None
            else (self.scale.device if self.scale is not None else torch.device("cpu"))
        )
        dtype = (
            self.shift.dtype
            if self.shift is not None
            else (self.scale.dtype if self.scale is not None else torch.float32)
        )
        self.avg = torch.zeros(self.n_units, device=device, dtype=dtype)
        self.var = torch.zeros(self.n_units, device=device, dtype=dtype)
        self.m2 = torch.zeros(self.n_units, device=device, dtype=dtype)
        self.count = torch.tensor(0.0, device=device, dtype=dtype)

    def _clear_stats(self):
        self.avg = None
        self.var = None
        self.m2 = None
        self.count = None

    def update_stats(self, input: torch.Tensor):
        if self.n_units != 1 and input.shape[-1] != self.n_units:
            raise ValueError(
                f"Expected last dimension {self.n_units}, got {input.shape[-1]}."
            )

        data = input.reshape(-1, self.n_units)
        sample_avg = data.mean(dim=0)
        sample_var = ((data - sample_avg) ** 2).mean(dim=0)
        sample_count = torch.tensor(
            float(data.shape[0]), device=data.device, dtype=data.dtype
        )

        if self.count.item() == 0:
            self.avg = sample_avg
            self.var = sample_var
            self.m2 = sample_var * sample_count
            self.count = sample_count
            return

        delta = sample_avg - self.avg
        total_count = self.count + sample_count
        self.m2 = (
            self.var * self.count
            + sample_var * sample_count
            + (delta**2) * self.count * sample_count / total_count
        )
        self.avg = self.avg + delta * (sample_count / total_count)
        self.var = self.m2 / total_count
        self.count = total_count


class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(
        self,
        emb_size: int,
        *,
        activation: Optional[nn.Module] = None,
        right_to_left: bool = False,
        edge_feat_dim: int = 1,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.right_to_left = right_to_left
        self.activation = activation or nn.ReLU()

        self.feature_module_left = nn.Linear(emb_size, emb_size, bias=True)
        self.feature_module_edge = nn.Linear(edge_feat_dim, emb_size, bias=False)
        self.feature_module_right = nn.Linear(emb_size, emb_size, bias=False)
        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False),
            self.activation,
            nn.Linear(emb_size, emb_size, bias=True),
        )

        self.post_conv_module = PreNormLayer(1, shift=False)

        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size, bias=True),
        )

        self._apply_orthogonal_init()

    def forward(
        self,
        left_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        right_features: torch.Tensor,
        scatter_out_size: int,
    ) -> torch.Tensor:
        original_left = left_features
        original_right = right_features

        left_proj = self.feature_module_left(left_features)
        edge_proj = self.feature_module_edge(edge_features)
        right_proj = self.feature_module_right(right_features)

        left_messages = left_proj[edge_indices[0]]
        right_messages = right_proj[edge_indices[1]]
        joint_features = self.feature_module_final(
            left_messages + edge_proj + right_messages
        )

        if self.right_to_left:
            scatter_idx = edge_indices[0]
            prev_features = original_left
        else:
            scatter_idx = edge_indices[1]
            prev_features = original_right

        scatter_out_size = int(scatter_out_size)
        conv_output = joint_features.new_zeros((scatter_out_size, self.emb_size))
        conv_output.index_add_(0, scatter_idx, joint_features)
        conv_output = self.post_conv_module(conv_output)

        output = torch.cat((conv_output, prev_features), dim=1)
        return self.output_module(output)

    def _apply_orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class BaseModel(nn.Module):
    """
    Base class providing pre-training utilities for PreNorm layers.
    """

    def pre_train_init(self):
        for layer in self.modules():
            if isinstance(layer, PreNormLayer):
                layer.start_updates()

    def pre_train_next(self):
        for layer in self.modules():
            if isinstance(layer, PreNormLayer):
                if layer.waiting_updates and layer.received_updates:
                    layer.stop_updates()
                    return layer
        return None

    def pre_train(self, *args, **kwargs):
        try:
            _ = self(*args, **kwargs)
            return False
        except PreNormException:
            return True

    def save_state(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def restore_state(self, path: str):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)


class GCNPolicy(BaseModel):
    """
    Bipartite Graph Convolutional Network (GCN) policy implemented with PyTorch.
    """

    def __init__(self):
        super().__init__()
        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19

        self.cons_embedding = nn.Sequential(
            PreNormLayer(self.cons_nfeats),
            nn.Linear(self.cons_nfeats, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
        )

        self.edge_embedding = PreNormLayer(self.edge_nfeats)

        self.var_embedding = nn.Sequential(
            PreNormLayer(self.var_nfeats),
            nn.Linear(self.var_nfeats, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(
            self.emb_size,
            activation=nn.ReLU(),
            right_to_left=True,
            edge_feat_dim=self.edge_nfeats,
        )
        self.conv_c_to_v = BipartiteGraphConvolution(
            self.emb_size,
            activation=nn.ReLU(),
            right_to_left=False,
            edge_feat_dim=self.edge_nfeats,
        )

        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1, bias=False),
        )

        self._apply_orthogonal_init()

    def forward(self, data: HeteroData) -> torch.Tensor:
        constraint_x = data["constraint"].x
        variable_x = data["variable"].x
        edge_index = data["constraint", "to", "variable"].edge_index
        edge_storage = data["constraint", "to", "variable"]
        edge_attr = getattr(edge_storage, "edge_attr", None)
        edge_attr = (
            edge_attr
            if edge_attr is not None
            else constraint_x.new_zeros((edge_index.size(1), self.edge_nfeats))
        )
        edge_attr = edge_attr.view(-1, self.edge_nfeats)

        constraint_features = self.cons_embedding(constraint_x)
        edge_features = self.edge_embedding(edge_attr)
        variable_features = self.var_embedding(variable_x)

        n_constraints = constraint_features.size(0)
        constraint_features = self.conv_v_to_c(
            constraint_features,
            edge_index,
            edge_features,
            variable_features,
            n_constraints,
        )
        constraint_features = constraint_features.relu()

        n_variables = variable_features.size(0)
        variable_features = self.conv_c_to_v(
            constraint_features,
            edge_index,
            edge_features,
            variable_features,
            n_variables,
        )
        variable_features = variable_features.relu()

        logits = self.output_module(variable_features).squeeze(-1)

        n_vars_per_sample = self._num_nodes_per_graph(data, node_type="variable")
        if len(n_vars_per_sample) > 1:
            logits = self._pad_output(logits, n_vars_per_sample)
        else:
            logits = logits.unsqueeze(0)
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _pad_output(
        self, logits: torch.Tensor, n_vars_per_sample: Sequence[int], pad_value: float = -1e8
    ) -> torch.Tensor:
        splits = torch.split(logits, tuple(int(x) for x in n_vars_per_sample))
        max_len = max(int(x) for x in n_vars_per_sample)
        padded: List[torch.Tensor] = []
        for segment in splits:
            padded.append(
                F.pad(
                    segment,
                    (0, max_len - segment.numel()),
                    value=pad_value,
                )
            )
        return torch.stack(padded, dim=0)

    @staticmethod
    def _num_nodes_per_graph(data: HeteroData, node_type: str) -> List[int]:
        batch = getattr(data[node_type], "batch", None)
        if batch is None:
            return [data[node_type].num_nodes or data[node_type].x.size(0)]
        counts = torch.bincount(batch, minlength=int(batch.max()) + 1)
        return counts.tolist()

    def _apply_orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
