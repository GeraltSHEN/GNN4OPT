import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from .modules import StackedBipartiteGNN, StackedSAGEBipartiteGNN


class GNNPolicy(nn.Module):
    def __init__(
        self,
        emb_size,
        cons_nfeats,
        var_nfeats,
        n_layers,
        gnn_backbone
    ):
        super().__init__()
        self.n_layers = n_layers
        self.gnn_backbone = gnn_backbone

        # EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
                # torch.nn.LayerNorm(cons_nfeats),
                MLP([cons_nfeats, emb_size, emb_size], act="relu", norm=None, plain_last=False))
        self.var_embedding = torch.nn.Sequential(
                # torch.nn.LayerNorm(var_nfeats),
                MLP([var_nfeats, emb_size, emb_size], act="relu", norm=None, plain_last=False))

        # DATA ENCODER
        if self.gnn_backbone == "bipartite":
            self.data_encoder = StackedBipartiteGNN(
                hidden_channels=emb_size, n_layers=n_layers
            )
        elif self.gnn_backbone == "sage":
            self.data_encoder = StackedSAGEBipartiteGNN(
                hidden_channels=emb_size, n_layers=n_layers
            )
        else:
            raise ValueError(f"{self.gnn_backbone} not available.")

        # FINAL MLP
        self.output_module = MLP([emb_size, emb_size, 1],
                                 act="relu", norm=None, plain_last=True, bias=[True, False])

    def forward(self, data):
        constraint_features = data["constraint_features"]
        edge_indices = data["edge_index"]
        edge_features = data["edge_attr"]
        variable_features = data["variable_features"]

        # 1. raw features to embeddings in common dimension
        Y = self.cons_embedding(constraint_features)
        X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        Y, X = self.data_encoder(Y, edge_indices, edge_features, X)

        # 3. transform variable features to branching logits
        output = self.output_module(X).squeeze(-1)
        return output
