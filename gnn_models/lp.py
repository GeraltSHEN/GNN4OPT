import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_scatter import scatter

from .modules import StackedBipartiteGNN, StackedSAGEBipartiteGNN


class LPGNN(nn.Module):
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
        self.cons_out = torch.nn.Sequential(
                # torch.nn.LayerNorm(emb_size),
                MLP([emb_size, emb_size, 1], act="relu", norm=None, plain_last=True))
        self.vars_out = torch.nn.Sequential(
                # torch.nn.LayerNorm(emb_size),
                MLP([emb_size, emb_size, 1], act="relu", norm=None, plain_last=True))

    def forward(self, data):
        constraint_features = data["constraint_features"]
        edge_indices = data["edge_index"]
        edge_features = data["edge_attr"]
        variable_features = data["variable_features"]
        n_constraints_per_graph = data["n_constraints_per_graph"]
        n_variables_per_graph = data["n_variables_per_graph"]

        # 1. raw features to embeddings in common dimension
        Y = self.cons_embedding(constraint_features)
        X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        Y, X = self.data_encoder(Y, edge_indices, edge_features, X)

        # 3. reduce channel to 1 per node, then aggregate node dimension per LP graph
        cons_contribution = self.cons_out(Y).squeeze(-1)
        vars_contribution = self.vars_out(X).squeeze(-1)

        n_graphs = int(n_constraints_per_graph.numel())
        cons_graph_ids = torch.repeat_interleave(
            torch.arange(n_graphs, device=cons_contribution.device),
            n_constraints_per_graph,
        )
        vars_graph_ids = torch.repeat_interleave(
            torch.arange(n_graphs, device=vars_contribution.device),
            n_variables_per_graph,
        )

        cons_graph_mean = scatter(cons_contribution, cons_graph_ids, dim=0, dim_size=n_graphs, reduce="mean")
        vars_graph_mean = scatter(vars_contribution, vars_graph_ids, dim=0, dim_size=n_graphs, reduce="mean")
        output = cons_graph_mean + vars_graph_mean
        return output
