import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.nn import Embedding, Linear
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, MLP, GCNConv, SAGEConv

from typing import Optional

from model_torch import PreNormLayer


def tied_topk_indices(values, k, expansion=2):
    """
    >>> tied_topk_indices(torch.tensor([4,4,4,5,5]), 2, 2).sort()[0]
    tensor([3, 4])
    >>> tied_topk_indices(torch.tensor([4,1,4,5,5,1]), 3, 2).sort()[0]
    tensor([0, 2, 3, 4])
    """
    assert len(values) >= k * expansion

    values, indices = torch.topk(values, k * expansion)
    assert values[k - 1] != values[-1], (
        "Cannot break ties within expansion.\nTry a larger expansion value"
    )

    return indices[: k + ((values[k - 1] == values[k:]).sum())]


class PowerMethod(torch.nn.Module):
    """Symmetry breaking model using the power method."""

    def __init__(self, k, out_dim):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

    def forward(self, v0, adj_t):
        v = v0
        for _ in range(self.k):
            v = adj_t.matmul(v)
        return v


class SymmetryBreakingGNN(torch.nn.Module):
    """Symmetry breaking model using a 2-layer GCN."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels, hidden_channels, normalize=False
        )  # Note: This assumes the adj matrix is normalized
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.out_dim = hidden_channels

    def forward(self, v0, adj_t):
        x = self.conv1(v0, adj_t).relu()
        return self.conv2(x, adj_t)


class ProductTupleEncoder(torch.nn.Module):
    """A baseline tuple encoder that takes the element-wise product of node embeddings."""

    def __init__(self):
        super().__init__()

    def forward(self, X, adj_t, tuples_coo, **kwargs):
        return X[tuples_coo].prod(dim=0)


class Classifier(torch.nn.Module):
    """A wrapper module combining a data encoder, tuple encoder, and MLP head."""

    def __init__(
        self,
        data_encoder,
        tuple_encoder,
        in_dim,
        out_dim,
        linear_classifier,
        train_head_only=False,
    ):
        super().__init__()
        self.data_encoder = data_encoder
        self.tuple_encoder = tuple_encoder
        self.mlp = (
            MLP([in_dim, in_dim, out_dim], norm=None, dropout=0.0)
            if not linear_classifier
            else Linear(in_dim, out_dim)
        )
        self.train_head_only = train_head_only

    def forward(self, data, tuples_coo, adj_t):
        with torch.set_grad_enabled(not self.train_head_only and self.training):
            X, tuples_coo = self.data_encoder(data, tuples_coo)
            X = self.tuple_encoder(X, adj_t, tuples_coo)
        return self.mlp(X)


class Holo(torch.nn.Module):
    """The Holo-GNN tuple encoder using symmetry breaking."""

    def __init__(self, *, n_breakings: int, symmetry_breaking_model):
        super().__init__()
        self.n_breakings = n_breakings
        self.symmetry_breaking_model = symmetry_breaking_model

        self.ln = torch.nn.LayerNorm(symmetry_breaking_model.out_dim)

    def get_nodes_to_break(self, adj_t, n_breakings=8):
        node_degrees = adj_t.sum(-1)
        return tied_topk_indices(node_degrees, k=n_breakings)

    def forward(self, X, adj_t, tuples_coo, group_idx=None):
        # X: (n_nodes, hidden_channels)
        break_node_indices = self.get_nodes_to_break(
            adj_t, n_breakings=self.n_breakings
        )

        # break_node_indices: (n_breakings,)
        one_hot_breakings = F.one_hot(break_node_indices, X.size(0)).unsqueeze(-1)
        # one_hot_breakings: (n_breakings, n_nodes, 1)
        holo_repr = self.symmetry_breaking_model(
            torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    one_hot_breakings,
                ),
                dim=-1,
            ),
            adj_t,
        )  # (t, n, f), where n includes both movies and users
        # holo_repr: (n_breakings, n_nodes, symmetry_breaking_model.out_dim)
        holo_repr = self.ln(holo_repr)

        set_of_link_repr = holo_repr[:, tuples_coo].prod(dim=1)  # (t, k, f)
        # set_of_link_repr: (n_breakings, n_tuples, out_dim)

        if group_idx is not None:
            link_repr = torch_scatter.scatter(
                set_of_link_repr, group_idx, 0, reduce="mean"
            )
        else:
            link_repr = set_of_link_repr.mean(0, keepdim=True)  # (l=1, k, f)

        return link_repr.transpose(0, 1).flatten(1, 2)  # (k, f*l)


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

    def forward(self, data: HeteroData) -> torch.Tensor:
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
    ) -> torch.Tensor:
        candidate_embeddings = variable_embeddings.index_select(0, candidate_indices)
        if self.out_dim is None:
            self.out_dim = candidate_embeddings.size(-1)
        return candidate_embeddings


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
        self.head = (
            nn.Linear(head_in_dim, 1)
            if linear_classifier
            else MLP([head_in_dim, head_in_dim, 1], norm=None, dropout=0.0)
        )

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
        )

        logits = self.head(tuple_repr).squeeze(-1)
        return logits
