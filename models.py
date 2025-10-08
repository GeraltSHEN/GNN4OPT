import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GCNConv, SAGEConv


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


# =================================================================================
# Planetoid-Specific Modules
# =================================================================================
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
