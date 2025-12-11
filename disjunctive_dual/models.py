from collections.abc import Sequence
from contextlib import contextmanager
from typing import Optional, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

PERFORMANCE_DEBUG = False  # Toggle to print timing and memory information


@contextmanager
def _perf_timer(label: str):
    if not PERFORMANCE_DEBUG:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start)
        print(f"[Perf] {label}: {elapsed_ms:.3f} s")


def _reset_peak_memory(device: torch.device):
    if not PERFORMANCE_DEBUG:
        return None
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return device
    if PERFORMANCE_DEBUG:
        print("[Perf] SetCoverHolo peak memory: skipped (CUDA not available or tensor on CPU).")
    return None


def _log_peak_memory(device: Optional[torch.device]):
    if device is None or not PERFORMANCE_DEBUG:
        return
    peak_bytes = torch.cuda.max_memory_allocated(device)
    print(
        f"[Perf] SetCoverHolo peak CUDA memory on {device}: {peak_bytes / (1024 ** 2):.2f} MB"
    )


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
        with _perf_timer("GNNPolicy step 1: embed raw features"):
            Y = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features)
            X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        with _perf_timer("GNNPolicy step 2: constraint-variable message passing"):
            Y, X = self.data_encoder(Y, edge_indices, edge_features, X)
        
        # 3. break symmetry
        if self.holo is not None:
            with _perf_timer("GNNPolicy step 3: symmetry breaking / SetCoverHolo"):
                X = self.holo(
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
        with _perf_timer("GNNPolicy step 4: output head"):
            output = self.output_module(X).squeeze(-1)
        return output


class SetCoverHolo(torch.nn.Module):
    """
    A Set-Cover-specific Holo-GNN tuple encoder using symmetry breaking.
    1. oracle or heuristic (customized get_nodes_to_break method) to select n_breakings (n_branching) candidates, say t in total
    
    2. add one-hot encodings in the form of 2-element-set to nodes, i.e. 
    Y:= {[Y, 0, 0], [Y, 0, 0]} in R^{2t * n_constraints * (d+2)}
    X:= {[X, 1_v, 0], [X, 0, 1_v]} in R^{2t * n_variables * (d+2)}
    
    3. r-gated constraint embeddings. The forward pass takes the original graph input, whose first constraint feature is the r-gate (extra constraint features may follow),
    Y:= {[r * Y, 0, 0], [Y, 0, 0]} in R^{2t * n_constraints * (d+2)}
    The forward pass will take the original graph input, edge_index:=[constraint_indices, variable_indices] in R^{2 * n_edges},
    for each v in n_breaking, find the constraint_indices_connected_to_v, and 
    get revised r' by setting r[constraint_indices_connected_to_v] = 0
    Y:= {[r * Y, 0, 0], [r' * Y, 0, 0]} in R^{2t * n_constraints * (d+2)}
    
    4. break symmetry. 
    Y, X:= symmetry_breaking_model(Y, X, adj_t) 

    5. Y go into set transformer and let constraint nodes talk to each other, problem channels also talk to each other
    Y:= setTransformer(Y) in R^{2t * n_constraints * (d+2)} where n_constraints get mixed information from each other, sub-problems get mixed information from each other
    
    6. X and Y get updated through message passing, i.e. let constraint nodes talk to variable nodes
    Y or X:= BipartiteGraphConvolution(X, edge_index, Y) in R^{2t * n_constraints or n_variables * (d+2)} for a few rounds
    X:= setTransformer(X) in R^{t * n_variables * (d+2)} let sub-problems get mixed information by learned pooling
    
    7. X go into set transformer again, allowing breaking views to talk to each other, 
    X:= setTransformer(X) in R^{n_variables * (d+2)}
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
        
        # set transformer: sub-problem talks to sub-problem (two dual LP)
        self.constraint_sab = SetAttentionBlock(dim_in=self.emb_size, dim_out=self.emb_size, num_heads=1, ln=True)
        # set transformer: sub-problems are mixed (primal aspect)
        self.variable_problem_pma = PoolingMultiheadAttention(dim=self.emb_size, num_heads=1, num_seeds=1, ln=True)
        # set transformer: views are mixed (primal aspect)
        self.variable_view_pma = PoolingMultiheadAttention(dim=self.emb_size, num_heads=1, num_seeds=1, ln=True)

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
        r: base r gating column (n_constraints_total, 1)
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
        constraint_features, # first column is r-gate; remaining columns are extra features
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

        peak_mem_device = _reset_peak_memory(Y.device)

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
        with _perf_timer("SetCoverHolo step 1: add one-hot encodings to X"):
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
        with _perf_timer("SetCoverHolo step 2: apply r-gating to Y"):
            # Use only the original r-gating column; extra features remain available in constraint_features
            r = constraint_features[:, :1]
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
        with _perf_timer("SetCoverHolo step 3: break symmetry"):
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
        with _perf_timer("SetCoverHolo step 4: constraint set transformer"):
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
        with _perf_timer("SetCoverHolo step 5: cross-problem constraint attention"):
            Y = self.constraint_sab(
                            torch.stack((Y_a, Y_b), dim=2).reshape(-1, 2, Y_a.size(-1)),
                            key_padding_mask=None).reshape(
                                            t, n_constraints_total, 2, -1
                                        )
            Y_a, Y_b = Y[:,:,0], Y[:,:,1]
        # (t, n_constraints, d+2)

        # gnn: constraint talks to variable
        with _perf_timer("SetCoverHolo step 6: constraint-variable message passing"):
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
        X = self.variable_problem_pma(
                        torch.stack((X_a, X_b), dim=2).reshape(-1, 2, X_a.size(-1)),
                        key_padding_mask=None).reshape(
                                        t, n_variables_every * num_graphs, -1
                                    )
        # (t, n_variables, d+2)
        X = self.variable_view_pma(X.transpose(0, 1), key_padding_mask=None).squeeze(1)
        # (n_variables, d+2)
        _log_peak_memory(peak_mem_device)
        return X


class SetCoverGumbel(torch.nn.Module):
    """
    A Set-Cover-specific GumbelModel for symmetry breaking.
    """

    def __init__(
        self,
        selection_model: Optional[torch.nn.Module],
        symmetry_breaking_model: StackedBipartiteGNN,
        num_subgraphs: int,
        num_marked: int = 1,
        tau: float = 1.0,
        hard: bool = True,
        use_noise: bool = True,
        detach_marking: bool = False,
    ):
        """
        Args:
            selection_model: Network producing a scalar score per variable node.
                The model is expected to accept (constraint_features, edge_indices,
                edge_features, variable_features) and return shape (n_vars,) or
                (n_vars, 1). When None, variables are selected uniformly at random.
            symmetry_breaking_model: Prediction model applied after marking the
                selected variables. Typically a `StackedBipartiteGNN` whose
                `hidden_channels` matches `base_dim + num_marked`.
            num_subgraphs: Number of subgraphs (views) to sample. The original
                graph counts as one view; sampling happens `num_subgraphs - 1` times.
            num_marked: How many nodes to mark per sampled subgraph.
            tau: Gumbel-Softmax temperature.
            hard: When True, uses straight-through discrete samples; otherwise
                returns soft probabilities.
            use_noise: Enable Gumbel noise during training; when False behaves
                deterministically given the logits.
            detach_marking: If True, detaches the marking channels before they
                are concatenated to the variable embeddings.
        """
        super().__init__()
        if num_subgraphs < 1:
            raise ValueError("num_subgraphs must be at least 1.")
        if num_marked < 1:
            raise ValueError("num_marked must be at least 1.")

        self.selection_model = selection_model
        self.symmetry_breaking_model = symmetry_breaking_model
        self.num_subgraphs = num_subgraphs
        self.num_marked = num_marked
        self.tau = tau
        self.hard = hard
        self.use_noise = use_noise
        self.detach_marking = detach_marking

        # The prediction model operates on augmented embeddings
        self.emb_size = self.symmetry_breaking_model.out_dim
        self._mask_value = float("-inf")
        self.last_selection = {}

    @staticmethod
    def _top_k_gumbel_softmax(
        logits: torch.Tensor,
        k: int,
        tau: float = 1.0,
        hard: bool = False,
        use_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised top-k Gumbel-Softmax sampling.

        Args:
            logits: Tensor of shape (..., num_nodes).
            k: Number of nodes to sample.
            tau: Temperature for the softmax.
            hard: If True, returns straight-through one-hot samples.
            use_noise: Whether to inject Gumbel noise.

        Returns:
            samples: Tensor with shape (..., num_nodes, k) containing
                soft/hard samples for each of the k selections.
            indices: Tensor with shape (..., k) containing the argmax indices.
        """
        if use_noise:
            gumbels = torch.distributions.Gumbel(
                torch.zeros_like(logits), torch.ones_like(logits)
            ).sample()
            gumbels = (logits + gumbels) / tau
        else:
            gumbels = logits
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            topk = y_soft.topk(k, dim=-1)
            indices = topk.indices
            rets: List[torch.Tensor] = []
            for i in range(k):
                y_hard = torch.zeros_like(logits).scatter_(
                    -1, indices[..., i : i + 1], 1.0
                )
                ret = y_hard - y_soft.detach() + y_soft
                rets.append(ret.unsqueeze(-1))
            samples = torch.cat(rets, dim=-1)
        else:
            indices = torch.empty_like(logits, dtype=torch.long)
            samples = y_soft.unsqueeze(-1).expand(*y_soft.shape, k)
        return samples, indices

    def preprocess_observation(self, n_variables_per_graph: torch.Tensor, device) -> dict:
        """
        Create a minimal observation dictionary mirroring the RL utilities in
        `policy-learn` but tailored to the static tensors available here.

        The observation only tracks which variable indices have been selected so
        far; everything else is passed directly to the model in the forward call.
        """
        num_rounds = max(self.num_subgraphs - 1, 0)
        num_slots = num_rounds * self.num_marked
        which_subgraphs = torch.full(
            (n_variables_per_graph.numel(), num_slots),
            -1,
            device=device,
            dtype=torch.long,
        )
        return {
            "which_subgraphs": which_subgraphs,
            "n_variables_per_graph": n_variables_per_graph,
        }

    @staticmethod
    def update_observation_inplace_no_replace(
        obs: dict, selection_indices: torch.Tensor
    ) -> None:
        """
        Append new selections to the observation without overwriting previous
        ones. The observation stores the *global* variable ids. The incoming
        `selection_indices` can be flat (num_graphs * k) or shaped as
        (num_graphs, k).
        """
        which_subgraphs = obs["which_subgraphs"]
        if which_subgraphs.numel() == 0:
            return
        selection_indices = selection_indices.view(which_subgraphs.size(0), -1)
        for g in range(which_subgraphs.size(0)):
            empty_slots = (which_subgraphs[g] == -1).nonzero(as_tuple=False).flatten()
            if empty_slots.numel() == 0:
                continue
            num_to_fill = min(empty_slots.numel(), selection_indices.size(1))
            which_subgraphs[g, empty_slots[:num_to_fill]] = selection_indices[
                g, :num_to_fill
            ]

    def _compute_selection_logits(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the provided selector to obtain logits per variable. If no selector
        is provided, fall back to uniform logits.
        """
        if self.selection_model is None:
            return torch.zeros(
                variable_features.size(0),
                device=variable_features.device,
                dtype=variable_features.dtype,
            )
        logits = self.selection_model(
            constraint_features, edge_indices, edge_features, variable_features
        )
        logits = logits.squeeze(-1)
        if logits.dim() != 1:
            raise ValueError(
                f"Selection model must return shape (n_vars,) or (n_vars, 1); got {tuple(logits.shape)}"
            )
        return logits

    def _build_dense_logits(
        self,
        logits: torch.Tensor,
        n_variables_per_graph: torch.Tensor,
        candidates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a flat logits vector into a padded matrix of shape
        (num_graphs, max_num_vars) so that the Gumbel sampling can happen
        independently per graph.
        """
        device, dtype = logits.device, logits.dtype
        mask_value = torch.finfo(dtype).min
        num_graphs = int(n_variables_per_graph.numel())
        max_vars = int(n_variables_per_graph.max().item())
        dense_logits = torch.full(
            (num_graphs, max_vars), mask_value, device=device, dtype=dtype
        )
        offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, device=device, dtype=torch.long),
                    n_variables_per_graph[:-1],
                )
            ),
            dim=0,
        )

        for g, (start, n_vars) in enumerate(zip(offsets.tolist(), n_variables_per_graph.tolist())):
            start = int(start)
            n_vars = int(n_vars)
            end = start + n_vars
            if candidates is None:
                dense_logits[g, :n_vars] = logits[start:end]
            else:
                mask = (candidates >= start) & (candidates < end)
                graph_candidates = candidates[mask]
                if graph_candidates.numel() == 0:
                    continue
                dense_logits[g, (graph_candidates - start).long()] = logits[graph_candidates]
        return dense_logits, offsets

    @staticmethod
    def _mask_selected_logits(
        dense_logits: torch.Tensor, indices: torch.Tensor, mask_value: float
    ) -> torch.Tensor:
        """Mask out already selected nodes so future rounds sample without replacement."""
        batch_idx = torch.arange(
            dense_logits.size(0), device=dense_logits.device
        ).unsqueeze(-1)
        dense_logits[batch_idx, indices] = mask_value
        return dense_logits

    def mark_subgraphs(
        self,
        samples: List[torch.Tensor],
        n_variables_per_graph: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Convert a list of sampled subgraphs into per-view marking features.

        Args:
            samples: Each element has shape (batch, max_vars, num_marked).
            n_variables_per_graph: 1D tensor with variable counts per graph.
            dtype/device: Used to materialise zero-marking for the base graph.
            detach: Whether to detach the marking channels from autograd.

        Returns:
            Tensor with shape (num_views, sum(n_variables_per_graph), num_marked)
            where view 0 corresponds to the unmarked original graph.
        """
        total_vars = int(n_variables_per_graph.sum().item())
        if len(samples) == 0:
            base = torch.zeros(
                (1, total_vars, self.num_marked), device=device, dtype=dtype
            )
            return base.detach() if detach else base

        offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, device=device, dtype=torch.long),
                    n_variables_per_graph[:-1],
                )
            ),
            dim=0,
        )
        marks: List[torch.Tensor] = [
            torch.zeros(
                (total_vars, self.num_marked), device=device, dtype=samples[0].dtype
            )
        ]
        for sample in samples:
            view_marks = torch.zeros_like(marks[0])
            for g, (start, n_vars) in enumerate(zip(offsets.tolist(), n_variables_per_graph.tolist())):
                start = int(start)
                n_vars = int(n_vars)
                end = start + n_vars
                view_marks[start:end] = sample[g, :n_vars, :]
            marks.append(view_marks)
        marks = torch.stack(marks, dim=0)
        return marks.detach() if detach else marks

    @staticmethod
    def format_for_stacked_bipartite(Y, X, edge_indices, edge_features):
        """
        Expand the bipartite graph for multiple views by stacking the node
        features and adjusting the edge indices accordingly.
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
        shape_info = {
            "num_views": num_views,
            "n_constraints": n_constraints,
            "n_variables": n_variables,
        }
        return formatted_Y, formatted_X, formatted_edge_indices, formatted_edge_features, shape_info

    @staticmethod
    def format_from_stacked_bipartite(Y, X, shape_info):
        """
        Undo `format_for_stacked_bipartite` by restoring the view dimension.
        """
        num_views = shape_info["num_views"]
        n_constraints = shape_info["n_constraints"]
        n_variables = shape_info["n_variables"]
        Y = Y.reshape(num_views, n_constraints, -1)
        X = X.reshape(num_views, n_variables, -1)
        return Y, X

    def forward(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
        candidates: Optional[torch.Tensor] = None,
        n_constraints_per_graph: Optional[torch.Tensor] = None,
        n_variables_per_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Selects variable subgraphs with Gumbel-Softmax, appends the resulting
        marks to the node features, runs the symmetry-breaking GNN, and finally
        aggregates the per-view variable embeddings.

        Args match those of `SetCoverHolo` so this module can be swapped into
        `GNNPolicy` with minimal changes.
        """
        device, dtype = Y.device, Y.dtype
        if n_constraints_per_graph is None:
            n_constraints_per_graph = torch.tensor(
                [Y.size(0)], device=device, dtype=torch.long
            )
        if n_variables_per_graph is None:
            n_variables_per_graph = torch.tensor(
                [X.size(0)], device=device, dtype=torch.long
            )

        dense_logits, offsets = self._build_dense_logits(
            logits=self._compute_selection_logits(
                constraint_features, edge_indices, edge_features, variable_features
            ),
            n_variables_per_graph=n_variables_per_graph,
            candidates=candidates,
        )
        obs = self.preprocess_observation(n_variables_per_graph, device)

        samples: List[torch.Tensor] = []
        indices_per_round: List[torch.Tensor] = []
        num_rounds = max(self.num_subgraphs - 1, 0)
        working_logits = dense_logits.clone()
        mask_value = torch.finfo(working_logits.dtype).min

        for _ in range(num_rounds):
            available = (~torch.isinf(working_logits)).sum(dim=1)
            if (available < self.num_marked).any():
                raise ValueError(
                    "Not enough available variables to sample without replacement; "
                    f"requested {self.num_marked}, available {available.min().item()}."
                )
            sample, idx = self._top_k_gumbel_softmax(
                working_logits,
                k=self.num_marked,
                tau=self.tau,
                hard=self.hard,
                use_noise=self.training and self.use_noise,
            )
            samples.append(sample)
            indices_per_round.append(idx)
            working_logits = self._mask_selected_logits(
                working_logits, idx, mask_value=mask_value
            )

        # Record selections as global indices for potential debugging/analysis
        if indices_per_round:
            global_indices: List[torch.Tensor] = []
            for idx in indices_per_round:
                global_idx = idx + offsets.view(-1, 1)
                global_indices.append(global_idx)
                self.update_observation_inplace_no_replace(obs, global_idx.flatten())
            self.last_selection = {
                "dense_logits": dense_logits.detach(),
                "indices": [g.detach() for g in global_indices],
            }
        else:
            self.last_selection = {"dense_logits": dense_logits.detach(), "indices": []}

        markings = self.mark_subgraphs(
            samples,
            n_variables_per_graph=n_variables_per_graph,
            dtype=dtype,
            device=device,
            detach=self.detach_marking,
        )
        num_views = markings.size(0)

        zeros_for_constraints = torch.zeros(
            (num_views, Y.size(0), self.num_marked), device=device, dtype=dtype
        )
        Y_aug = torch.cat(
            (Y.unsqueeze(0).expand(num_views, -1, -1), zeros_for_constraints), dim=-1
        )
        X_aug = torch.cat(
            (X.unsqueeze(0).expand(num_views, -1, -1), markings), dim=-1
        )

        Y_formatted, X_formatted, edge_indices_f, edge_features_f, shape_info = (
            self.format_for_stacked_bipartite(Y_aug, X_aug, edge_indices, edge_features)
        )
        Y_formatted, X_formatted = self.symmetry_breaking_model(
            Y_formatted, edge_indices_f, edge_features_f, X_formatted
        )
        _, X_views = self.format_from_stacked_bipartite(
            Y_formatted, X_formatted, shape_info
        )
        
        X_out = X_views.mean(dim=0)
        return X_out
