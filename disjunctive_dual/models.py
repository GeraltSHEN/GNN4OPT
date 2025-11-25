from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor


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

    def forward(self, X, key_padding_mask):
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
        return self.mab1(X, H)


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
            torch.nn.Linear(self.holo.emb_size if self.holo is not None else emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_size, bias=False),
        )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates=None,
        n_constraints_per_graph=None,
        n_variables_per_graph=None,
    ):
        # 1. raw features to embeddings in common dimension
        Y = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        Y, X = self.data_encoder(Y, edge_indices, edge_features, X)
        
        # 3. break symmetry
        if self.holo is not None:
            Y, X = self.holo(
                Y,
                X,
                constraint_features,
                edge_indices,
                edge_features,
                variable_features,
                candidates=candidates,
                n_constraints_per_graph=n_constraints_per_graph,
                n_variables_per_graph=n_variables_per_graph,
            )

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

    5. Y go into set transformer and let constraint nodes talk to each other, problem channels also talk to each other
    Y:= setTransformer(Y) \in R^{2t * n_constraints * (d+2)} where n_constraints get mixed information from each other, sub-problems get mixed information from each other
    
    6. X and Y get updated through message passing, i.e. let constraint nodes talk to variable nodes
    Y or X:= BipartiteGraphConvolution(X, edge_index, Y) \in R^{2t * n_constraints or n_variables * (d+2)} for a few rounds
    X:= setTransformer(X) \in R^{t * n_variables * (d+2)} let sub-problems get mixed information by learned pooling
    
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
        
        self.constraint_sab = SetAttentionBlock(dim_in=self.emb_size, dim_out=self.emb_size, num_heads=1, ln=True)
        self.variable_pma = PoolingMultiheadAttention(dim=self.emb_size, num_heads=1, num_seeds=1, ln=True)

    def get_nodes_to_break(
        self,
        constraint_features,
        variable_features,
        edge_indices,
        edge_features,
        candidates=None,
        n_variables_per_graph=None,
    ):
        """Select variable nodes to break using an external selector model
        return (bsz, t) and (bsz, t) where indices at the second dim are 
        local indices and global indices
        """
        k = self.n_breakings
        with torch.no_grad():
            scores = self.breaking_selector_model(
                constraint_features, edge_indices, edge_features, variable_features
            )
        if n_variables_per_graph is None:
            n_variables_per_graph = torch.tensor(
                [variable_features.size(0)], device=scores.device, dtype=torch.long
            )

        variable_offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, device=scores.device, dtype=torch.long),
                    n_variables_per_graph[:-1],
                )
            ),
            dim=0,
        )

        selected_local = []
        selected_global = []
        for offset, n_vars in zip(variable_offsets.tolist(), n_variables_per_graph.tolist()):
            start = int(offset)
            n_vars = int(n_vars)
            end = start + n_vars

            if candidates is not None:
                mask = (candidates >= start) & (candidates < end)
                graph_candidates = candidates[mask]
                num_avail = graph_candidates.numel()
                if num_avail < k:
                    raise ValueError(f"No candidates available = {num_avail}, "
                                     f"but requested to break k = {k} variables in the current graph.")
                scores_local = scores[graph_candidates]
                chosen_local = graph_candidates[torch.topk(scores_local, k=k).indices] - start
                chosen_global = graph_candidates[torch.topk(scores_local, k=k).indices]
            else:
                scores_local = scores[start:end]
                num_avail = scores_local.numel()
                if num_avail < k:
                    raise ValueError(f"Number of variables in the graph = {num_avail}, "
                                     f"but requested to break k = {k} variables.")
                chosen_local = torch.topk(scores_local, k=k).indices
                chosen_global = torch.topk(scores_local, k=k).indices + start
            selected_local.append(chosen_local)
            selected_global.append(chosen_global)
        return torch.stack(selected_local, dim=0), torch.stack(selected_global, dim=0)
    
    def revise_r(
            self, r, edge_indices, branching_variable_indices):
        """
        r: base r (n_constraints_total, 1)
        edge_indices: (2, E)
        branching_variable_indices: (t, bsz) where indices at the second dim 
        must be global indices

        Returns:
        r_after_branching: (t, n_constraints_total, 1)
            where r_after_branching[k] is r updated with branching_variable_indices[k] fixed to 1
        """
        constraint_indices, variable_indices = edge_indices
        n_constraints_total = r.size(0)
        device = r.device
        dtype= r.dtype

        t, bsz = branching_variable_indices.size(0), branching_variable_indices.size(1)
        r_after_branching = r.unsqueeze(0).repeat(t, 1, 1) # (t, n_constraints_total, 1)

        #  branching_variable_indices -> (t, bsz, 1)
        #  variable_indices           -> (1, 1, E)
        #  mask                       -> (t, bsz, E) -> (t, E)
        mask = (branching_variable_indices.view(t, bsz, 1) == 
                variable_indices.view(1, 1, -1)).any(dim=1).to(dtype)
        # Scatter the mask
        row_idx = constraint_indices.unsqueeze(0).expand_as(mask)
        connected = torch.zeros((t, n_constraints_total), dtype=dtype, device=device)
        connected.scatter_add_(1, row_idx, mask)
        r_after_branching[connected > 0] = 0.0
        return r_after_branching

    def format_for_stacked_bipartite(self, Y, X, edge_indices, edge_features):
        """
        (t, n, d) nodes -> (t * n, d) nodes
        (2, E) edge indices -> (2, t * E) edge indices
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
                .expand(num_views, edge_features.size(0), -1)
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
    
    def format_for_batched_and_padded_nodes(self, nodes, n_per_graph):
        """
        nodes: (t, n_total, d)
        n_per_graph: (num_graphs, ) sizes per graph

        return
        nodes: (t * num_graphs, n_max, d)
        key_padding_mask: None if all graphs share the same size, otherwise
            (t * num_graphs, n_max) where True marks padding to ignore in attention
        """
        device, dtype = nodes.device, nodes.dtype
        num_graphs = n_per_graph.numel()
        t, _, d = nodes.shape

        n_max = int(n_per_graph.max().item())
        is_consistent = bool((n_per_graph == n_max).all().item())
        if is_consistent:
            nodes = nodes.view(t, num_graphs, n_max, d)
            return nodes.reshape(t * num_graphs, n_max, d), None

        offsets = torch.cumsum(
            torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_per_graph[:-1])),
            dim=0,
        )

        padded_nodes = []
        key_padding_masks = []

        for g in range(num_graphs):
            n_current = int(n_per_graph[g].item())
            start = int(offsets[g].item())
            end = start + n_current

            nodes_slice = nodes[:, start:end, :]

            pad_len = n_max - n_current
            if pad_len > 0:
                pad = torch.zeros((t, pad_len, d), device=device, dtype=dtype)
                nodes_slice = torch.cat([nodes_slice, pad], dim=1)
                mask = torch.cat(
                    [
                        torch.zeros((t, n_current), device=device, dtype=torch.bool),
                        torch.ones((t, pad_len), device=device, dtype=torch.bool),
                    ],
                    dim=1,
                )
            else:
                mask = torch.zeros((t, n_max), device=device, dtype=torch.bool)

            padded_nodes.append(nodes_slice)
            key_padding_masks.append(mask)

        padded_nodes = torch.cat(padded_nodes, dim=0)
        key_padding_mask = torch.cat(key_padding_masks, dim=0)

        return padded_nodes, key_padding_mask

    def format_from_batched_and_padded_nodes(self, nodes, n_per_graph):
        """
        nodes: (t * num_graphs, n_max, d)
        n_per_graph: (num_graphs, ) sizes per graph

        return
        nodes: (t, n_total, d)
        """
        num_graphs = n_per_graph.numel()
        t = nodes.size(0) // num_graphs
        n_max = nodes.size(1)
        n_total = int(n_per_graph.sum().item())

        is_consistent = bool((n_per_graph == n_max).all().item())
        if is_consistent:
            nodes = nodes.view(num_graphs, t, n_max, -1).transpose(0, 1)
            nodes = nodes.reshape(t, num_graphs * n_max, -1)
            return nodes

        restored_nodes = []

        for g in range(num_graphs):
            n_current = int(n_per_graph[g].item())

            start = g * t
            end = (g + 1) * t

            nodes_slice = nodes[start:end, :n_current, :]
            restored_nodes.append(nodes_slice)

        restored_nodes = torch.cat(restored_nodes, dim=1)
        return restored_nodes

    def forward(
        self,
        Y,
        X,
        constraint_features, # [r] \in R^{n_constraints * 1}
        edge_indices,
        edge_features,
        variable_features, # [c, is_fixed_to_1, is_fixed_to_0, is_not_fixed]
        candidates=None,
        n_constraints_per_graph=None,
        n_variables_per_graph=None,
    ):
        device = Y.device
        dtype = Y.dtype
        if n_constraints_per_graph is None:
            n_constraints_per_graph = torch.tensor(
                [Y.size(0)], device=device, dtype=torch.long
            )
        if n_variables_per_graph is None:
            n_variables_per_graph = torch.tensor(
                [X.size(0)], device=device, dtype=torch.long
            )
        num_graphs = n_constraints_per_graph.numel()
        n_constraints_total = n_constraints_per_graph.sum().int()
        n_variables_every = n_variables_per_graph.sum().int() // num_graphs

        break_node_indices_local, break_node_indices_global = self.get_nodes_to_break(
            constraint_features=constraint_features,
            variable_features=variable_features,
            edge_indices=edge_indices,
            edge_features=edge_features,
            candidates=candidates,
            n_variables_per_graph=n_variables_per_graph,
        )  # (bsz, t)
        bsz, t = break_node_indices_local.shape
        # add one-hot encodings to X
        one_hot_breakings = F.one_hot(break_node_indices_local.T, 
                                      n_variables_every).reshape(t, -1).unsqueeze(-1)
        # (t, bsz * n_variables_every, 1)
        all_zeros = torch.zeros_like(one_hot_breakings)
        one_hot_breakings_a = torch.cat([one_hot_breakings, all_zeros], dim=-1)
        one_hot_breakings_b = torch.cat([all_zeros, one_hot_breakings], dim=-1) # (t, bsz * n_variables_every, 2)
        X_a = torch.cat([X.unsqueeze(0).expand(t, X.size(0), X.size(1)), 
                         one_hot_breakings_a], dim=-1)  # (t, bsz * n_variables_every, d+2)
        X_b = torch.cat([X.unsqueeze(0).expand(t, X.size(0), X.size(1)), 
                         one_hot_breakings_b], dim=-1) # (t, bsz * n_variables_every, d+2)
        # add r-gating to Y
        r = constraint_features
        r_after_branching = self.revise_r(r=r, 
                                          edge_indices=edge_indices, 
                                          branching_variable_indices=break_node_indices_global.T)
        # (t, n_constraints_total, 1)
        Y_a = torch.cat([r * Y.unsqueeze(0).expand(t, n_constraints_total, Y.size(1)),
                         torch.zeros((t, n_constraints_total, 2), device=device, dtype=dtype)], 
                         dim=-1) # (t, n_constraints_total, d+2)
        Y_b = torch.cat([r_after_branching * Y.unsqueeze(0).expand(t, n_constraints_total, Y.size(1)),
                         torch.zeros((t, n_constraints_total, 2), device=device, dtype=dtype)], 
                         dim=-1) # (t, n_constraints_total, d+2)
        # break symmetry
        Y_a, X_a, formatted_edge_indices, formatted_edge_features, shape_info = \
            self.format_for_stacked_bipartite(Y_a, X_a, edge_indices, edge_features)
        Y_a, X_a = self.symmetry_breaking_model(
            Y_a, formatted_edge_indices, formatted_edge_features, X_a
        )
        Y_a, X_a = self.format_from_stacked_bipartite(Y_a, X_a, shape_info)

        Y_b, X_b, formatted_edge_indices, formatted_edge_features, shape_info = \
            self.format_for_stacked_bipartite(Y_b, X_b, edge_indices, edge_features)
        Y_b, X_b = self.symmetry_breaking_model(
            Y_b, formatted_edge_indices, formatted_edge_features, X_b
        )
        Y_b, X_b = self.format_from_stacked_bipartite(Y_b, X_b, shape_info)
        # holo_repr: (t, n, d+2)

        # set transformer: constraint talks to constraint
        Y_a, key_padding_mask_a = \
            self.format_for_batched_and_padded_nodes(Y_a, n_constraints_per_graph)
        Y_b, key_padding_mask_b = \
            self.format_for_batched_and_padded_nodes(Y_b, n_constraints_per_graph)
        # (bsz*t, n_constraints_max, d+2)
        if self.constraint_set_block is not None:
            Y_a = self.constraint_set_block(Y_a, key_padding_mask_a)
            Y_b = self.constraint_set_block(Y_b, key_padding_mask_b) 
        # (bsz*t, n_constraints_max, d+2)
        Y_a = self.format_from_batched_and_padded_nodes(Y_a, n_constraints_per_graph)
        Y_b = self.format_from_batched_and_padded_nodes(Y_b, n_constraints_per_graph)
        # (t, n_constraints_total, d+2)
        
        # set transformer: constraint's 2 problems talk to each other
        Y = self.constraint_sab(
                        torch.stack((Y_a, Y_b), dim=2).reshape(-1, 2, Y_a.size(-1)),
                        key_padding_mask=None).reshape(
                                        one_hot_breakings.size(0), n_constraints_total, 2, -1
                                    )
        Y_a, Y_b = Y[:,:,0], Y[:,:,1]
        # (t, n_constraints, d+2)

        # gnn: constraint talks to variable
        if self.constraint_variable_gnn is not None:
            Y_a, X_a, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(Y_a, X_a, edge_indices, edge_features)
            Y_a, X_a = self.constraint_variable_gnn(
                Y_a, formatted_edge_indices, formatted_edge_features, X_a
            )
            Y_a, X_a = self.format_from_stacked_bipartite(Y_a, X_a, shape_info)

            Y_b, X_b, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(Y_b, X_b, edge_indices, edge_features)
            Y_b, X_b = self.constraint_variable_gnn(
                Y_b, formatted_edge_indices, formatted_edge_features, X_b
            )
            Y_b, X_b = self.format_from_stacked_bipartite(Y_b, X_b, shape_info)
        X = self.variable_pma(
                        torch.stack((X_a, X_b), dim=2).reshape(-1, 2, X_a.size(-1)),
                        key_padding_mask=None).reshape(
                                        one_hot_breakings.size(0), n_variables_every * num_graphs, -1
                                    )
        # (t, n_variables, d+2)
        return Y, X
