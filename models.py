import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.nn import Embedding, Linear
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, MLP, GCNConv, SAGEConv

from typing import Optional

from model_torch import PreNormLayer


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        *,
        alpha=0.1,
        theta=0.5,
        shared_weights=True,
        dropout=0.0,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCNConv(
                    in_channels,
                    hidden_channels,
                    normalize=False,
                )
            )
            in_channels = hidden_channels

        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, adj_t)
            x = x.relu()

        return F.dropout(x, self.dropout, training=self.training)


class GCNDataEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.gnn_encoder = GCNEncoder(
            in_channels, hidden_channels, num_layers, dropout=dropout
        )

    def forward(self, data, tuples_coo) -> tuple[Data, torch.Tensor]:
        x, adj_t = data.x, data.adj_t
        X = self.gnn_encoder(x, adj_t)
        return X, tuples_coo


class BipartiteDataEncoder(torch.nn.Module):
    """Encode bipartite MILP graphs stored as HeteroData into variable embeddings."""

    def __init__(
        self,
        *,
        emb_size: int = 64,
        cons_nfeats: int = 5,
        var_nfeats: int = 19,
        edge_nfeats: int = 1,
        num_layers: int = 2,
        conv_type: str = "sage",
    ):
        super().__init__()
        self.emb_size = emb_size
        self.cons_nfeats = cons_nfeats
        self.var_nfeats = var_nfeats
        self.edge_nfeats = edge_nfeats
        self.num_layers = num_layers

        self.cons_embedding = nn.Sequential(
            PreNormLayer(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = PreNormLayer(edge_nfeats)
        self.var_embedding = nn.Sequential(
            PreNormLayer(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.break_indicator_encoder = nn.Linear(1, emb_size, bias=False)

        conv_type = conv_type.lower()
        conv_cls = {
            "sage": lambda: SAGEConv((-1, -1), emb_size),
            "gcn": lambda: GCNConv((-1, -1), emb_size),
        }.get(conv_type)
        if conv_cls is None:
            raise ValueError(f"Unsupported conv_type '{conv_type}'. Use 'sage' or 'gcn'.")

        self.use_edge_weight = conv_type == "gcn"

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("constraint", "to", "variable"): conv_cls(),
                    ("variable", "to_rev", "constraint"): conv_cls(),
                },
                aggr="sum",
            )
            self.convs.append(conv)
        self.out_dim = emb_size

    def forward(
        self,
        data: HeteroData,
        break_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        constraint_x = data["constraint"].x
        variable_x = data["variable"].x
        edge_storage = data["constraint", "to", "variable"]
        edge_index = edge_storage.edge_index
        edge_attr = getattr(edge_storage, "edge_attr", None)
        if edge_attr is None:
            edge_attr = constraint_x.new_zeros(edge_index.size(1), self.edge_nfeats)
        edge_attr = edge_attr.view(-1, self.edge_nfeats)
        edge_index = edge_index.contiguous()

        constraint_features = self.cons_embedding(constraint_x)
        edge_features = self.edge_embedding(edge_attr)
        variable_features = self.var_embedding(variable_x)
        if break_indicator is not None:
            if break_indicator.dim() == 1:
                break_indicator = break_indicator.unsqueeze(-1)
            break_indicator = break_indicator.to(
                variable_features.device, dtype=variable_features.dtype
            )
            variable_features = variable_features + self.break_indicator_encoder(
                break_indicator
            )

        x_dict = {
            "constraint": constraint_features,
            "variable": variable_features,
        }
        edge_index_dict = {
            ("constraint", "to", "variable"): edge_index,
            ("variable", "to_rev", "constraint"): edge_index.flip(0).contiguous(),
        }
        edge_attr_dict = None
        if self.use_edge_weight:
            edge_weight = edge_features.squeeze(-1)
            edge_attr_dict = {
                ("constraint", "to", "variable"): edge_weight,
                ("variable", "to_rev", "constraint"): edge_weight,
            }

        for conv in self.convs:
            if edge_attr_dict is None:
                x_dict = conv(x_dict, edge_index_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict["variable"]


class VariableTupleEncoder(torch.nn.Module):
    """Tuple encoder selecting variable embeddings for candidate indices."""

    def __init__(self):
        super().__init__()
        self.out_dim: Optional[int] = None

    def forward(
        self,
        variable_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = variable_embeddings.index_select(0, candidate_indices)
        if self.out_dim is None:
            self.out_dim = candidate_embeddings.size(-1)
        return candidate_embeddings


class BipartiteHoloTupleEncoder(torch.nn.Module):
    """Holo-inspired tuple encoder for bipartite graphs with node-level predictions."""

    def __init__(
        self,
        *,
        n_breakings: int,
        symmetry_breaking_encoder: Optional[BipartiteDataEncoder] = None,
        encoder_kwargs: Optional[dict] = None,
        reduce: str = "mean",
    ):
        super().__init__()
        if n_breakings <= 0:
            raise ValueError("n_breakings must be a positive integer.")
        if reduce not in {"mean", "max"}:
            raise ValueError("reduce must be either 'mean' or 'max'.")
        self.n_breakings = n_breakings
        self.reduce = reduce
        if symmetry_breaking_encoder is None:
            encoder_kwargs = encoder_kwargs or {}
            self.symmetry_breaking_encoder = BipartiteDataEncoder(**encoder_kwargs)
        else:
            if encoder_kwargs:
                raise ValueError("encoder_kwargs and symmetry_breaking_encoder are mutually exclusive.")
            self.symmetry_breaking_encoder = symmetry_breaking_encoder
        self.out_dim = self.symmetry_breaking_encoder.out_dim

    def _variable_degrees(self, data: HeteroData, device: torch.device) -> torch.Tensor:
        num_variables = data["variable"].num_nodes
        degrees = torch.zeros(num_variables, device=device, dtype=torch.float32)
        if ("constraint", "to", "variable") not in data.edge_types:
            return degrees
        edge_index = data["constraint", "to", "variable"].edge_index
        if edge_index.numel() == 0:
            return degrees
        edge_index = edge_index.to(device)
        ones = torch.ones(edge_index.size(1), device=device, dtype=degrees.dtype)
        degrees.scatter_add_(0, edge_index[1], ones)
        return degrees

    def _select_break_nodes(self, data: HeteroData, device: torch.device) -> torch.Tensor:
        degrees = self._variable_degrees(data, device)
        k = min(self.n_breakings, degrees.size(0))
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=degrees.device)
        _, indices = torch.topk(degrees, k, sorted=False)
        return indices

    def forward(
        self,
        variable_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        *,
        data: HeteroData,
    ) -> torch.Tensor:
        if data is None:
            raise ValueError("BipartiteHoloTupleEncoder requires the associated HeteroData object.")

        device = variable_embeddings.device
        candidate_indices = candidate_indices.to(device)
        self.symmetry_breaking_encoder.to(device)

        break_nodes = self._select_break_nodes(data, device)
        if break_nodes.numel() == 0:
            return variable_embeddings.index_select(0, candidate_indices)

        num_variables = data["variable"].num_nodes
        holo_embeddings = []
        for node_idx in break_nodes:
            indicator = torch.zeros(num_variables, 1, device=device, dtype=variable_embeddings.dtype)
            indicator[node_idx] = 1.0
            var_repr = self.symmetry_breaking_encoder(data, break_indicator=indicator)
            var_repr = var_repr.to(device=device, dtype=variable_embeddings.dtype)
            holo_embeddings.append(var_repr.index_select(0, candidate_indices))

        stacked = torch.stack(holo_embeddings, dim=0)
        if self.reduce == "mean":
            aggregated = stacked.mean(dim=0)
        else:  # self.reduce == "max"
            aggregated, _ = stacked.max(dim=0)
        return aggregated


class GNNPolicy(torch.nn.Module):
    """
    Policy network composed of a bipartite data encoder, tuple encoder, and head.
    """

    def __init__(
        self,
        *,
        data_encoder: Optional[torch.nn.Module] = None,
        tuple_encoder: Optional[torch.nn.Module] = None,
        emb_size: int = 64,
        linear_classifier: bool = False,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.data_encoder = (
            data_encoder
            if data_encoder is not None
            else BipartiteDataEncoder(emb_size=emb_size)
        )
        self.tuple_encoder = (
            tuple_encoder if tuple_encoder is not None else VariableTupleEncoder()
        )

        head_in_dim = getattr(self.tuple_encoder, "out_dim", None)
        if head_in_dim is None:
            head_in_dim = getattr(self.data_encoder, "out_dim", emb_size)
        self.head_in_dim = head_in_dim
        self.projection_dim = projection_dim
        self.linear_classifier = linear_classifier

        if projection_dim is not None and projection_dim <= 0:
            raise ValueError("projection_dim must be positive when provided.")

        if linear_classifier:
            self.pre_head = None
            self.head = nn.Linear(head_in_dim, 1)
        else:
            if projection_dim is not None:
                # Mimic Holo's usage of out_dim: two hidden layers followed by projection_dim output.
                self.pre_head = MLP(
                    [head_in_dim, head_in_dim, projection_dim], norm=None, dropout=0.0
                )
                self.head = nn.Linear(projection_dim, 1)
            else:
                self.pre_head = None
                self.head = MLP([head_in_dim, head_in_dim, 1], norm=None, dropout=0.0)

    def forward(
        self,
        data: HeteroData,
        candidate_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_indices is None:
            candidate_indices = getattr(data, "candidate_indices", None)
            if candidate_indices is None:
                raise ValueError("candidate_indices must be provided with the batch.")

        variable_embeddings = self.data_encoder(data)

        tuple_repr = self.tuple_encoder(
            variable_embeddings,
            candidate_indices,
            data=data,
        )

        if not self.linear_classifier and self.pre_head is not None:
            tuple_repr = self.pre_head(tuple_repr)

        logits = self.head(tuple_repr).squeeze(-1)
        return logits
