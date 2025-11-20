from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

# TODO: add dropout option to models

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

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V = self.fc_v(K)
        attn_out, _ = self.attn(Q, K_proj, V, need_weights=False)
        O = Q + attn_out
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


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

    def forward(self, X):
        # X: (batch, n, dim_in)
        batch_size = X.size(0)
        inducing = self.inducing_points.repeat(batch_size, 1, 1)
        H = self.mab0(inducing, X)
        return self.mab1(X, H)


class PoolingMultiheadAttention(nn.Module):
    """Set Transformer PMA block implemented with torch MultiheadAttention."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads, ln=ln)
    
    def forward(self, X):
        # X: (batch, n, dim)
        batch_size = X.size(0)
        return self.mab(self.S.repeat(batch_size, 1, 1), X)


class StackedBipartiteGNN(torch.nn.Module):
    """Stack of BipartiteGraphConvolution layers for constraint-variable message passing."""

    def __init__(self, hidden_channels, edge_nfeats=1, n_layers=2):
        super().__init__()
        self.out_dim = hidden_channels
        self.edge_nfeats = edge_nfeats
        self.n_layers = n_layers

        if n_layers == 1:
            self.conv_v_to_c = BipartiteGraphConvolution(
                emb_size=hidden_channels, edge_nfeats=edge_nfeats
            )
            self.conv_c_to_v = BipartiteGraphConvolution(
                emb_size=hidden_channels, edge_nfeats=edge_nfeats
            )
        else:
            for i in range(n_layers):
                setattr(
                    self,
                    f"conv_{i}_v_to_c",
                    BipartiteGraphConvolution(
                        emb_size=hidden_channels, edge_nfeats=edge_nfeats
                    ),
                )
                setattr(
                    self,
                    f"conv_{i}_c_to_v",
                    BipartiteGraphConvolution(
                        emb_size=hidden_channels, edge_nfeats=edge_nfeats
                    ),
                )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
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
    def __init__(self, emb_size=64, edge_nfeats=1):
        super().__init__("add")
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, emb_size, bias=False)
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


class GNNPolicy(nn.Module):
    """A wrapper module combining
    - an initial MLP to convert raw features to embeddings in common dimension,
    - a data encoder (BipartiteGraphConvolution) for MILP bipartite graphs, 
    - a set-cover-specific module (SetCoverHolo)
    - a final MLP on the variable features for candidate choice/scoring
    """
    def __init__(self, emb_size, cons_nfeats, edge_nfeats, var_nfeats, output_size,
                 n_layers, holo):
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        
        # DATA ENCODER
        self.n_layers = n_layers
        self.data_encoder = StackedBipartiteGNN(
            hidden_channels=emb_size, edge_nfeats=edge_nfeats, n_layers=n_layers
        )
        
        # TUPLE ENCODER
        self.holo = holo

        # FINAL MLP
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.holo.emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_size, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        # 1. raw features to embeddings in common dimension
        Y = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        Y, X = self.data_encoder(Y, edge_indices, edge_features, X)
        
        # 3. break symmetry
        Y, X = self.holo(Y, X, constraint_features, edge_indices, edge_features, variable_features)

        # 4. transform variable features to strong branching decision
        output = self.output_module(X).squeeze(-1)
        return output


class SetCoverHolo(torch.nn.Module):
    """
    A Set-Cover-specific Holo-GNN tuple encoder using symmetry breaking.
    1. oracle or heuristic (customized get_nodes_to_break method) to select n_breakings (n_branching) candidates, say t in total
    
    2. add one-hot encodings in the form of 2-element-set to nodes, i.e. 
    Y:= {[Y, 0, 0], [Y, 0, 0]} \in R^{2t * n_constraints * (d+2)}
    X:= {[X, 1_v, 0], [X, 0, 1_v]} \in R^{2t * n_variables * (d+2)}
    
    3. r-gated constraint embeddings. The forward pass will take the original graph input, constraint_feature:=[r] \in R^{n_constraints * 1},
    Y:= {[r * Y, 0, 0], [Y, 0, 0]} \in R^{2t * n_constraints * (d+2)}
    The forward pass will take the original graph input, edge_index:=[constraint_indices, variable_indices] \in R^{2 * n_edges},
    for each v in n_breaking, find the constraint_indices_connected_to_v, and 
    get revised r' by setting r[constraint_indices_connected_to_v] = 0
    Y:= {[r * Y, 0, 0], [r' * Y, 0, 0]} \in R^{2t * n_constraints * (d+2)}
    
    4. break symmetry. 
    Y, X:= symmetry_breaking_model(Y, X, adj_t) 

    5. Y go into set transformer and let constraint nodes talk to each other
    Y:= InducedSetAttentionBlock(Y) \in R^{2t * n_constraints * (d+2)} where n_constraints get mixed information from each other

    6. X and Y get updated through message passing, i.e. let constraint nodes talk to variable nodes
    Y or X:= BipartiteGraphConvolution(X, edge_index, Y) \in R^{2t * n_constraints or n_variables * (d+2)} for a few rounds
    
    7. X get grouped across breakings for future ranking task, i.e.
    for each variable node v, get X_v \in R^{2t * (d+2)}, 
    potentially followed by some reduction to aggregare across views t, across sets 2, across embedding dimension (d+2),
    end up with X \in R^{n_variables * final_embedding_dimension} (You shouldn't do anything for ranking, I will handle it myself in the future)
    
    """

    def __init__(
        self,
        n_breakings: int,
        breaking_selector_model,
        symmetry_breaking_model,
        num_heads: int = 0,
        isab_num_inds: int = 0,
        mp_layers: int = 1,
        edge_nfeats: int = 1
    ):
        super().__init__()

        # select and break
        self.n_breakings = n_breakings
        self.breaking_selector_model = breaking_selector_model
        if self.breaking_selector_model is not None:
            for param in self.breaking_selector_model.parameters():
                param.requires_grad = False
            self.breaking_selector_model.eval()
        self.symmetry_breaking_model = symmetry_breaking_model
        self.emb_size = self.symmetry_breaking_model.out_dim
        self.edge_nfeats = edge_nfeats
        self.num_heads = num_heads
        # self.ln = torch.nn.LayerNorm(self.emb_size)

        # set transformer: constraint talks to constraint
        if num_heads > 0:
            if self.emb_size % num_heads != 0:
                raise ValueError(
                    f"symmetry_breaking_model.out_dim ({self.emb_size}) "
                    f"must be divisible by num_heads ({num_heads})."
                )
            num_inds = isab_num_inds
            if num_inds <= 0:
                raise ValueError("isab_num_inds must be >= 1 when attention is enabled.")
            self.constraint_set_block = InducedSetAttentionBlock(
                dim_in=self.emb_size,
                dim_out=self.emb_size,
                num_heads=num_heads,
                num_inds=num_inds,
                ln=True,
            )
        else:
            self.constraint_set_block = None

        # GNN: constraint talks to variable
        self.mp_layers = mp_layers
        if self.mp_layers > 0:
            self.constraint_variable_gnn = StackedBipartiteGNN(
                hidden_channels=self.emb_size,
                edge_nfeats=self.edge_nfeats,
                n_layers=self.mp_layers,
            )
        else:
            self.constraint_variable_gnn = None
        
        self.constraint_pma = PoolingMultiheadAttention(dim=self.emb_size, num_heads=1, num_seeds=1, ln=True)
        self.variable_pma = PoolingMultiheadAttention(dim=self.emb_size, num_heads=1, num_seeds=1, ln=True)

    def get_nodes_to_break(
        self,
        constraint_features,
        variable_features,
        edge_indices,
        edge_features
    ):
        """Select variable nodes to break using an external selector model."""
        k = self.n_breakings
        with torch.no_grad():
            scores = self.breaking_selector_model(
                constraint_features, edge_indices, edge_features, variable_features
            )
        return torch.topk(scores, k=k).indices
    
    def revise_r(
            self, r, edge_indices, branching_variable_indices):
        """
        r: base r (n_constraints, 1)
        edge_indices: (2, E)
        branching_variable_indices: (t,)

        Returns:
        r_after_branching: (t, n_constraints, 1)
            where r_after_branching[k] is r updated with branching_variable_indices[k] fixed to 1
        """
        constraint_indices, variable_indices = edge_indices
        n_constraints = r.size(0)
        device = r.device
        dtype= r.dtype

        t = branching_variable_indices.size(0)
        r_after_branching = r.unsqueeze(0).repeat(t, 1, 1) # (t, n_constraints, 1)

        # (t, E) mask indicating the branching variables
        mask = (variable_indices.unsqueeze(0) == branching_variable_indices.unsqueeze(1)).to(dtype)
        # Scatter the mask
        row_idx = constraint_indices.unsqueeze(0).expand_as(mask)
        connected = torch.zeros((t, n_constraints), dtype=dtype, device=device)
        connected.scatter_add_(1, row_idx, mask)
        r_after_branching[connected > 0] = 0.0
        return r_after_branching

    def format_for_stacked_bipartite(self, Y, X, edge_indices, edge_features):
        """
        (t, n, d) nodes -> (t * n, d) nodes
        (n_constraints, n_variables) edges -> (t * n_constraints, t * n_variables) edges
        """
        num_views, n_constraints, _ = Y.shape
        _, n_variables, _ = X.shape

        formatted_Y = Y.reshape(num_views * n_constraints, -1)
        formatted_X = X.reshape(num_views * n_variables, -1)

        constraint_offsets = torch.arange(num_views, device=Y.device).unsqueeze(1) * n_constraints
        variable_offsets = torch.arange(num_views, device=Y.device).unsqueeze(1) * n_variables
        constraint_edges = edge_indices[0].unsqueeze(0) + constraint_offsets
        variable_edges = edge_indices[1].unsqueeze(0) + variable_offsets
        formatted_edge_indices = torch.stack(
            (constraint_edges.reshape(-1), variable_edges.reshape(-1)), dim=0
        )

        formatted_edge_features = (
                edge_features.unsqueeze(0)
                .repeat(num_views, 1, 1)
                .reshape(num_views * edge_features.size(0), -1)
            )
        shape_info = {"num_views": num_views, "n_constraints": n_constraints, "n_variables": n_variables}

        return formatted_Y, formatted_X, formatted_edge_indices, formatted_edge_features, shape_info

    def format_from_stacked_bipartite(self, Y, X, shape_info):
        """
        (t * n, d) nodes -> (t, n, d) nodes
        edges are not changing over time so no need to format them back
        """
        num_views = shape_info["num_views"]
        n_constraints = shape_info["n_constraints"]
        n_variables = shape_info["n_variables"]
        Y = Y.reshape(num_views, n_constraints, -1)
        X = X.reshape(num_views, n_variables, -1)
        return Y, X

    def forward(
        self,
        Y,
        X,
        constraint_features, # [r] \in R^{n_constraints * 1}
        edge_indices,
        edge_features,
        variable_features, # [c, is_fixed_to_1, is_fixed_to_0, is_not_fixed]
    ):
        device = Y.device
        dtype = Y.dtype
        n_constraints = Y.size(0)
        n_variables = X.size(0)

        # select nodes to break
        break_node_indices = self.get_nodes_to_break(
            constraint_features=constraint_features,
            variable_features=variable_features,
            edge_indices=edge_indices,
            edge_features=edge_features,
        ) # (t,)
        # add one-hot encodings to X
        one_hot_breakings = F.one_hot(break_node_indices, n_variables).unsqueeze(-1) # (t, n_variables, 1)
        all_zeros = torch.zeros((one_hot_breakings.size(0), n_variables, 1), device=device, dtype=dtype)
        one_hot_breakings_a = torch.cat([one_hot_breakings, all_zeros], dim=-1) # (t, n_variables, 2)
        one_hot_breakings_b = torch.cat([all_zeros, one_hot_breakings], dim=-1) # (t, n_variables, 2)
        X_a = torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    one_hot_breakings_a,
                ),
                dim=-1) # (t, n_variables, d+2)
        X_b = torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    one_hot_breakings_b,
                ),
                dim=-1) # (t, n_variables, d+2)
        # add r-gating to Y
        r = constraint_features
        r_after_branching = self.revise_r(r=r, 
                                          edge_indices=edge_indices, 
                                          branching_variable_indices=break_node_indices) # (t, n_constraints, 1)
        Y_a = torch.cat(
                (
                    r * Y.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    torch.zeros((one_hot_breakings.size(0), n_constraints, 2), device=device, dtype=dtype)
                ),
                dim=-1) # (t, n_constraints, d+2)
        Y_b = torch.cat(
                (
                    r_after_branching * Y.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    torch.zeros((one_hot_breakings.size(0), n_constraints, 2), device=device, dtype=dtype)
                ), 
                dim=-1) # (t, n_constraints, d+2)
        # break symmetry
        formatted_Y_a, formatted_X_a, formatted_edge_indices, formatted_edge_features, shape_info = \
            self.format_for_stacked_bipartite(Y_a, X_a, edge_indices, edge_features)
        holo_repr_Y_a, holo_repr_X_a = self.symmetry_breaking_model(
            formatted_Y_a, formatted_edge_indices, formatted_edge_features, formatted_X_a
        )
        holo_repr_Y_a, holo_repr_X_a = self.format_from_stacked_bipartite(
            holo_repr_Y_a, holo_repr_X_a, shape_info
        )

        formatted_Y_b, formatted_X_b, formatted_edge_indices, formatted_edge_features, shape_info = \
            self.format_for_stacked_bipartite(Y_b, X_b, edge_indices, edge_features)
        holo_repr_Y_b, holo_repr_X_b = self.symmetry_breaking_model(
            formatted_Y_b, formatted_edge_indices, formatted_edge_features, formatted_X_b
        )
        holo_repr_Y_b, holo_repr_X_b = self.format_from_stacked_bipartite(
            holo_repr_Y_b, holo_repr_X_b, shape_info
        )
        # holo_repr: (t, n, d+2)
        # set transformer: constraint talks to constraint
        if self.constraint_set_block is not None:
            holo_repr_Y_a = self.constraint_set_block(holo_repr_Y_a)
            holo_repr_Y_b = self.constraint_set_block(holo_repr_Y_b) # (t, n_constraints, d+2)
        
        # Question: apply pooling over Y and use mixed Y to update X_a and X_b? 
        holo_repr_Y = self.constraint_pma(
                        torch.stack((holo_repr_Y_a, holo_repr_Y_b), 
                                    dim=2).reshape(-1, 2, holo_repr_Y_a.size(-1))).reshape(
                                        one_hot_breakings.size(0), n_constraints, -1
                                    )
        # holo_repr_Y: (t, n_constraints, d+2)

        # gnn: constraint talks to variable
        if self.constraint_variable_gnn is not None:
            formatted_Y, formatted_X_a, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(holo_repr_Y, holo_repr_X_a, edge_indices, edge_features)
            holo_repr_Y_a_updated, holo_repr_X_a = self.constraint_variable_gnn(
                formatted_Y, formatted_edge_indices, formatted_edge_features, formatted_X_a
            )
            _, holo_repr_X_a = self.format_from_stacked_bipartite(holo_repr_Y_a_updated, holo_repr_X_a, shape_info)

            formatted_Y, formatted_X_b, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(holo_repr_Y, holo_repr_X_b, edge_indices, edge_features)
            holo_repr_Y_b_updated, holo_repr_X_b = self.constraint_variable_gnn(
                formatted_Y, formatted_edge_indices, formatted_edge_features, formatted_X_b
            )
            _, holo_repr_X_b = self.format_from_stacked_bipartite(holo_repr_Y_b_updated, holo_repr_X_b, shape_info)
        holo_repr_X = self.variable_pma(
                        torch.stack((holo_repr_X_a, holo_repr_X_b), 
                                    dim=2).reshape(-1, 2, holo_repr_X_a.size(-1))).reshape(
                                        one_hot_breakings.size(0), n_variables, -1
                                    )
        # holo_repr_X: (t, n_variables, d+2)
        return holo_repr_Y, holo_repr_X
