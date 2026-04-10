from typing import Dict, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, Linear, MLP
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import NodeType, EdgeType


class MultiheadAttentionBlock(nn.Module):
    """Multihead attention block mirroring Set Transformer MAB."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_V, num_heads=num_heads, batch_first=True
        )

    def forward(self, Q, K, key_padding_mask):
        Q = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V = self.fc_v(K)
        attn_out, _ = self.attn(Q, K_proj, V, need_weights=False, 
                                key_padding_mask=key_padding_mask)
        O = Q + attn_out
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SetAttentionBlock(nn.Module):
    """Set Transformer SAB block implemented with torch MultiheadAttention."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, key_padding_mask)


class InducedSetAttentionBlock(nn.Module):
    """Set Transformer ISAB block implemented with torch MultiheadAttention."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        if num_inds <= 0:
            raise ValueError("num_inds must be >= 1.")
        self.inducing_points = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab0 = MultiheadAttentionBlock(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask):
        # X: (batch, n, dim_in)
        batch_size = X.size(0)
        inducing = self.inducing_points.repeat(batch_size, 1, 1)
        H = self.mab0(inducing, X, key_padding_mask)
        return self.mab1(X, H, key_padding_mask=None)


class PoolingMultiheadAttention(nn.Module):
    """Set Transformer PMA block implemented with torch MultiheadAttention."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads, ln=ln)
    
    def forward(self, X, key_padding_mask):
        # X: (batch, n, dim)
        batch_size = X.size(0)
        return self.mab(self.S.repeat(batch_size, 1, 1), X, key_padding_mask)


class StackedBipartiteGNN(torch.nn.Module):
    """Stack of BipartiteGraphConvolution layers for constraint-variable message passing."""

    def __init__(self, hidden_channels, n_layers=2):
        super().__init__()
        self.out_dim = hidden_channels
        self.n_layers = n_layers

        if n_layers == 1:
            self.conv_v_to_c = BipartiteGraphConvolution(
                emb_size=hidden_channels
            )
            self.conv_c_to_v = BipartiteGraphConvolution(
                emb_size=hidden_channels
            )
        else:
            for i in range(n_layers):
                setattr(
                    self,
                    f"conv_{i}_v_to_c",
                    BipartiteGraphConvolution(
                        emb_size=hidden_channels
                    ),
                )
                setattr(
                    self,
                    f"conv_{i}_c_to_v",
                    BipartiteGraphConvolution(
                        emb_size=hidden_channels
                    ),
                )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        if self.n_layers == 1:
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )
        else:
            for i in range(self.n_layers):
                conv_v_to_c = getattr(self, f"conv_{i}_v_to_c")
                conv_c_to_v = getattr(self, f"conv_{i}_c_to_v")
                constraint_features = constraint_features + conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                )
                variable_features = variable_features + conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
        return constraint_features, variable_features


class BipartiteGraphConvolution(MessagePassing):
    """
    src code: milp-evolve
    simple implementation of bipartite graph convolution
    """
    def __init__(self, emb_size=64):
        super().__init__("add")
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class SAGEConv(MessagePassing):
    def __init__(self, emb_size=64, mlp_layers=2):
        super().__init__("add")
        self.act = nn.ReLU()
        self.lin_src = nn.Linear(emb_size, emb_size)
        self.lin_dst = nn.Linear(emb_size, emb_size)
        self.mlp = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        left_features = self.lin_src(left_features)
        out = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        out = out + self.lin_dst(right_features)
        return self.mlp(out)

    def message(self, node_features_j, edge_features):
        return self.act(node_features_j) * edge_features


class StackedSAGEBipartiteGNN(nn.Module):
    """Stack of SAGE bipartite layers"""

    def __init__(self, hidden_channels, n_layers=2, mlp_layers=2):
        super().__init__()
        self.out_dim = hidden_channels
        self.n_layers = n_layers

        if n_layers == 1:
            self.conv_v_to_c = SAGEConv(
                emb_size=hidden_channels, mlp_layers=mlp_layers
            )
            self.conv_c_to_v = SAGEConv(
                emb_size=hidden_channels, mlp_layers=mlp_layers
            )
        else:
            for i in range(n_layers):
                setattr(self, f"conv_{i}_v_to_c",
                    SAGEConv(
                        emb_size=hidden_channels, mlp_layers=mlp_layers
                    ),
                )
                setattr(
                    self,
                    f"conv_{i}_c_to_v",
                    SAGEConv(
                        emb_size=hidden_channels, mlp_layers=mlp_layers
                    ),
                )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        if self.n_layers == 1:
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )
        else:
            for i in range(self.n_layers):
                conv_v_to_c = getattr(self, f"conv_{i}_v_to_c")
                conv_c_to_v = getattr(self, f"conv_{i}_c_to_v")
                constraint_features = constraint_features + conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                )
                variable_features = variable_features + conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
        return constraint_features, variable_features
    