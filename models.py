from collections.abc import Sequence
from contextlib import contextmanager
from typing import Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, MessagePassing

# from extensions import repeat_interleave, vrange

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

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        n_variables_per_graph=None,
    ):
        del n_variables_per_graph
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


class SAGEConv(MessagePassing):
    def __init__(self, emb_size=64, edge_nfeats=1, mlp_layers=2):
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

    def __init__(self, hidden_channels, edge_nfeats=1, n_layers=2, mlp_layers=2):
        super().__init__()
        self.out_dim = hidden_channels
        self.edge_nfeats = edge_nfeats
        self.n_layers = n_layers

        if n_layers == 1:
            self.conv_v_to_c = SAGEConv(
                emb_size=hidden_channels, edge_nfeats=edge_nfeats, mlp_layers=mlp_layers
            )
            self.conv_c_to_v = SAGEConv(
                emb_size=hidden_channels, edge_nfeats=edge_nfeats, mlp_layers=mlp_layers
            )
        else:
            for i in range(n_layers):
                setattr(self, f"conv_{i}_v_to_c",
                    SAGEConv(
                        emb_size=hidden_channels, edge_nfeats=edge_nfeats, mlp_layers=mlp_layers
                    ),
                )
                setattr(
                    self,
                    f"conv_{i}_c_to_v",
                    SAGEConv(
                        emb_size=hidden_channels, edge_nfeats=edge_nfeats, mlp_layers=mlp_layers
                    ),
                )

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        n_variables_per_graph=None,
    ):
        del n_variables_per_graph
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


class SecondOrderPPGNBlock(nn.Module):
    """Second-order Folklore block operating on dense pair tensors (B, M_max, N, F) and (B, N, N, F)."""

    def __init__(self, emb_size=64, mlp_layers=2, layernorm=True):
        super().__init__()
        self.mlp1_cv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.mlp2_cv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.skip_cv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.ln_cv = nn.LayerNorm(emb_size) if layernorm else nn.Identity()

        self.mlp1_vv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.mlp2_vv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.skip_vv = MLP([emb_size] * (mlp_layers + 1), act="relu", norm=None, plain_last=False)
        self.ln_vv = nn.LayerNorm(emb_size) if layernorm else nn.Identity()
    
    def cv_forward(self, con_var_features, var_var_features, con_var_mask, var_var_mask):
        x1 = self.mlp1_cv(con_var_features)
        x2 = self.mlp2_cv(var_var_features)
        if con_var_mask is not None:
            x1 = x1.masked_fill(~con_var_mask.unsqueeze(-1), 0.0)
        if var_var_mask is not None:
            x2 = x2.masked_fill(~var_var_mask.unsqueeze(-1), 0.0)
        mult = torch.einsum("bmnf,bnlf->bmlf", x1, x2)
        mult = self.ln_cv(mult)
        return self.skip_cv(con_var_features + mult)
    
    def vv_forward(self, var_var_features, var_var_mask):
        x1 = self.mlp1_vv(var_var_features)
        x2 = self.mlp2_vv(var_var_features)
        if var_var_mask is not None:
            mask = var_var_mask.unsqueeze(-1)
            x1 = x1.masked_fill(~mask, 0.0)
            x2 = x2.masked_fill(~mask, 0.0)
        mult = torch.einsum("bmnf,bnlf->bmlf", x1, x2)
        mult = self.ln_vv(mult)
        return self.skip_vv(var_var_features + mult)

    def forward(self, con_var_features, var_var_features, con_var_mask, var_var_mask):
        con_var_features = self.cv_forward(
            con_var_features, var_var_features, con_var_mask, var_var_mask
        )
        var_var_features = self.vv_forward(var_var_features, var_var_mask)

        if con_var_mask is not None:
            con_var_features = con_var_features.masked_fill(~con_var_mask.unsqueeze(-1), 0.0)
        if var_var_mask is not None:
            var_var_features = var_var_features.masked_fill(~var_var_mask.unsqueeze(-1), 0.0)

        return con_var_features, var_var_features
    

class StackedPPGNBipartiteGNN(nn.Module):
    """Stack of SecondOrderPPGNBlock"""
    def __init__(self, hidden_channels, edge_nfeats=1, n_layers=2, 
                 ppgn_mlp_layers=2, ppgn_layernorm=True):
        super().__init__()
        self.out_dim = hidden_channels
        self.edge_nfeats = edge_nfeats
        self.n_layers = n_layers

        self.ppgn_layers = nn.ModuleList(
            [
                SecondOrderPPGNBlock(
                    emb_size=hidden_channels,
                    mlp_layers=ppgn_mlp_layers,
                    layernorm=ppgn_layernorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.readout_mlp = MLP([2 * hidden_channels, hidden_channels, hidden_channels], act="relu", 
                               norm=None, plain_last=False)

    def prepare_mask(self, con_var_features, var_var_features, n_constraints_per_graph, n_variables_per_graph):
        if con_var_features.dim() != 4:
            raise ValueError("con_var_features must be a 4D tensor with shape (B, M_max, N_max, F).")
        if var_var_features.dim() != 4:
            raise ValueError("var_var_features must be a 4D tensor with shape (B, N_max, N_max, F).")

        bsz, m_max, n_max, _ = con_var_features.shape
        bsz_vv, n_max_a, n_max_b, _ = var_var_features.shape
        if bsz != bsz_vv:
            raise ValueError("Batch sizes of con_var_features and var_var_features must match.")
        if n_max_a != n_max or n_max_b != n_max:
            raise ValueError("N_max dimensions of con_var_features and var_var_features must match.")

        if n_constraints_per_graph is None or n_variables_per_graph is None:
            return None, None

        device = con_var_features.device
        n_constraints_per_graph = n_constraints_per_graph.reshape(-1)
        n_variables_per_graph = n_variables_per_graph.reshape(-1)

        constraint_node_mask = (torch.arange(m_max, device=device).unsqueeze(0) < n_constraints_per_graph.unsqueeze(1))
        variable_node_mask = (torch.arange(n_max, device=device).unsqueeze(0) < n_variables_per_graph.unsqueeze(1))
        con_var_mask = torch.einsum("bm,bn->bmn", constraint_node_mask, variable_node_mask).bool()
        var_var_mask = torch.einsum("bn,bm->bnm", variable_node_mask, variable_node_mask).bool()
        return con_var_mask, var_var_mask
    
    def readout(self, con_var_features, var_var_features, con_var_mask, var_var_mask):
        """
        con_var_features: (B, M_max, N_max, F)
        var_var_features: (B, N_max, N_max, F)
        get [con_var_features.sum(dim=1), var_var_features.sum(dim=1)] in shape of (B, N_max, 2F)
        apply MLP([con_var_features.sum(dim=1), var_var_features.sum(dim=1)]) to get 
        variable_features in shape of (B, N_max, F), then unpad to (N_1 + ... + N_bsz, F)
        """
        if con_var_mask is not None:
            con_var_features = con_var_features.masked_fill(~con_var_mask.unsqueeze(-1), 0.0)
        if var_var_mask is not None:
            var_var_features = var_var_features.masked_fill(~var_var_mask.unsqueeze(-1), 0.0)

        con_var_summary = con_var_features.sum(dim=1)
        var_var_summary = var_var_features.sum(dim=1)
        variable_features = self.readout_mlp(torch.cat([con_var_summary, var_var_summary], dim=-1)) # (B, N_max, 2F)

        if var_var_mask is None:
            return variable_features.reshape(-1, variable_features.size(-1))

        variable_node_mask = var_var_mask.any(dim=-1)  # (B, N_max)
        return variable_features[variable_node_mask]  # (N_1 + ... + N_bsz, F)

    def forward(self, con_var_features, var_var_features, n_constraints_per_graph, n_variables_per_graph):
        con_var_mask, var_var_mask = self.prepare_mask(
            con_var_features,
            var_var_features,
            n_constraints_per_graph,
            n_variables_per_graph,
        )

        if con_var_mask is not None:
            con_var_features = con_var_features.masked_fill(~con_var_mask.unsqueeze(-1), 0.0)
        if var_var_mask is not None:
            var_var_features = var_var_features.masked_fill(~var_var_mask.unsqueeze(-1), 0.0)

        for layer in self.ppgn_layers:
            con_var_features, var_var_features = layer(
                con_var_features, var_var_features, con_var_mask, var_var_mask
            )

        variable_features = self.readout(con_var_features, var_var_features, con_var_mask, var_var_mask)
        return variable_features


class GNNPolicy(nn.Module):
    """A wrapper module combining
    - an initial MLP to convert raw features to embeddings in common dimension,
    - a configurable data encoder (bipartite / sage / ppgn) for MILP bipartite graphs,
    - a set-cover-specific module (SetCoverHolo) ASSUME NONE ALWAYS FOR NOW.
    - a final MLP on the variable features for candidate choice/scoring
    """
    def __init__(
        self,
        emb_size,
        cons_nfeats,
        edge_nfeats,
        var_nfeats,
        output_size,
        n_layers,
        holo,
        gnn_backbone: str = "bipartite", # sage, ppgn
        sage_mlp_layers: int = 2,
        ppgn_mlp_layers: int = 2,
        ppgn_layernorm: bool = True,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be >= 1.")
        self.n_layers = n_layers
        self.gnn_backbone = gnn_backbone

        # CONSTRAINT EMBEDDING
        if self.gnn_backbone == "ppgn":
            self.con_var_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(cons_nfeats + var_nfeats + 1),
                MLP([cons_nfeats + var_nfeats + 1, emb_size, emb_size], act="relu", norm=None, plain_last=False))
            self.cons_embedding = None
            self.var_var_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(var_nfeats + var_nfeats + 1),
                MLP([var_nfeats + var_nfeats + 1, emb_size, emb_size], act="relu", norm=None, plain_last=False))
            self.var_embedding = None
        else:
            self.con_var_embedding = None
            self.cons_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(cons_nfeats),
                MLP([cons_nfeats, emb_size, emb_size], act="relu", norm=None, plain_last=False))
            self.var_var_embedding = None
            self.var_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(var_nfeats),
                MLP([var_nfeats, emb_size, emb_size], act="relu", norm=None, plain_last=False))

        # DATA ENCODER
        if self.gnn_backbone == "bipartite":
            self.data_encoder = StackedBipartiteGNN(
                hidden_channels=emb_size, edge_nfeats=edge_nfeats, n_layers=n_layers
            )
        elif self.gnn_backbone == "sage":
            self.data_encoder = StackedSAGEBipartiteGNN(
                hidden_channels=emb_size,
                edge_nfeats=edge_nfeats,
                n_layers=n_layers,
                mlp_layers=sage_mlp_layers,
            )
        elif self.gnn_backbone == "ppgn":
            self.data_encoder = StackedPPGNBipartiteGNN(
                hidden_channels=emb_size,
                edge_nfeats=edge_nfeats,
                n_layers=n_layers,
                ppgn_mlp_layers=ppgn_mlp_layers,
                ppgn_layernorm=ppgn_layernorm,
            )
        else:
            raise ValueError(f"{self.gnn_backbone} not available.")

        # TUPLE ENCODER
        self.holo = holo

        # FINAL MLP
        self.output_module = MLP([self.holo.emb_size if self.holo is not None else emb_size, emb_size, output_size],
                                 act="relu", norm=None, plain_last=True, bias=[True, False])

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        con_var_features,
        var_var_features,
        candidates=None,
        n_constraints_per_graph=None,
        n_variables_per_graph=None,
    ):
        # 1. raw features to embeddings in common dimension
        with _perf_timer("GNNPolicy step 1: embed raw features"):
            if self.gnn_backbone == "ppgn":
                Y = self.con_var_embedding(con_var_features)
                X = self.var_var_embedding(var_var_features)
            else:
                Y = self.cons_embedding(constraint_features)
                X = self.var_embedding(variable_features)

        # 2. constraint-variable message passing
        with _perf_timer("GNNPolicy step 2: constraint-variable message passing"):
            if self.gnn_backbone == "ppgn":
                Y, X = self.data_encoder(Y, 
                                         X, 
                                         n_constraints_per_graph, 
                                         n_variables_per_graph)
            else:
                Y, X = self.data_encoder(Y,
                                         edge_indices,
                                         edge_features,
                                         X,
                                         n_variables_per_graph=n_variables_per_graph)

        # 3. break symmetry NOTE NOT MODIFIED FOR PPGN YET
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
        edge_nfeats: int = 1,
        use_set_transformer: bool = True,
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
        self.use_set_transformer = use_set_transformer
        # self.ln = torch.nn.LayerNorm(self.emb_size)

        # set transformer: constraint talks to constraint
        if self.use_set_transformer and num_heads > 0:
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
        if self.use_set_transformer and self.mp_layers > 0:
            self.constraint_variable_gnn = StackedBipartiteGNN(
                hidden_channels=self.emb_size,
                edge_nfeats=self.edge_nfeats,
                n_layers=self.mp_layers,
            )
        else:
            self.constraint_variable_gnn = None
        
        # set transformer: sub-problem talks to sub-problem (two dual LP)
        self.constraint_sab = (
            SetAttentionBlock(dim_in=self.emb_size, dim_out=self.emb_size, num_heads=1, ln=True)
            if self.use_set_transformer
            else None
        )
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
    

    def remove_edges_via_r(self, r, edge_indices):
        """
        r: base r gating column (t, n_constraints_total, 1)
            or r_after_branching: (t, n_constraints_total, 1)
            Note: remember to repeat base r t times to make it in shape of (t, n_constraints_total, 1) in forward
        edge_indices: (2, t * E) original edges repeated t times

        Returns:
        reduced_edge_indices: (2, E'_1 + E'_'2 + ... + E'_t)
            for each view, the original E_l is reduced to E'_l in this manner:
            when r_i = 0, all edges connected to node i are removed. 
            Note that edge_indices[0] are constraint nodes and edge_indices[1] are variable nodes (correct?)
        """
        constraint_gate = r.squeeze(-1)  # (t, n_constraints_total)
        flat_gate = constraint_gate.reshape(-1)  # (t * n_constraints_total,)

        constraint_nodes = edge_indices[0]
        edge_mask = flat_gate[constraint_nodes] > 0
        reduced_edge_indices = edge_indices[:, edge_mask]
        return reduced_edge_indices, edge_mask


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

        # peak_mem_device = _reset_peak_memory(Y.device)

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
            r_base_views = r.unsqueeze(0).expand(t, n_constraints_total, 1)
            Y_a = torch.cat([r_base_views * Y.unsqueeze(0),
                             torch.zeros((t, n_constraints_total, 2), device=device, dtype=dtype)], 
                             dim=-1) # (t, n_constraints_total, d+2)
            Y_b = torch.cat([r_after_branching * Y.unsqueeze(0),
                             torch.zeros((t, n_constraints_total, 2), device=device, dtype=dtype)], 
                             dim=-1) # (t, n_constraints_total, d+2)

        # break symmetry
        with _perf_timer("SetCoverHolo step 3: break symmetry"):
            Y_a, X_a, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(Y_a, X_a, edge_indices, edge_features)
            formatted_edge_indices_a, edge_mask_a = self.remove_edges_via_r(
                r_base_views, formatted_edge_indices
            )
            formatted_edge_features_a = formatted_edge_features[edge_mask_a]
            Y_a, X_a = self.symmetry_breaking_model(
                Y_a, formatted_edge_indices_a, formatted_edge_features_a, X_a
            )
            Y_a, X_a = self.format_from_stacked_bipartite(Y_a, X_a, shape_info)

            Y_b, X_b, formatted_edge_indices, formatted_edge_features, shape_info = \
                self.format_for_stacked_bipartite(Y_b, X_b, edge_indices, edge_features)
            formatted_edge_indices_b, edge_mask_b = self.remove_edges_via_r(
                r_after_branching, formatted_edge_indices
            )
            formatted_edge_features_b = formatted_edge_features[edge_mask_b]
            Y_b, X_b = self.symmetry_breaking_model(
                Y_b, formatted_edge_indices_b, formatted_edge_features_b, X_b
            )
            Y_b, X_b = self.format_from_stacked_bipartite(Y_b, X_b, shape_info)
        # holo_repr: (t, n, d+2)

        # set transformer: constraint talks to constraint
        if self.use_set_transformer:
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
        if self.constraint_sab is not None:
            with _perf_timer("SetCoverHolo step 5: cross-problem constraint attention"):
                Y = self.constraint_sab(
                                torch.stack((Y_a, Y_b), dim=2).reshape(-1, 2, Y_a.size(-1)),
                                key_padding_mask=None).reshape(
                                                t, n_constraints_total, 2, -1
                                            )
                Y_a, Y_b = Y[:,:,0], Y[:,:,1]
        # (t, n_constraints, d+2)

        # gnn: constraint talks to variable
        if self.constraint_variable_gnn is not None:
            with _perf_timer("SetCoverHolo step 6: constraint-variable message passing"):
                Y_a, X_a, _, __, shape_info = \
                    self.format_for_stacked_bipartite(Y_a, X_a, edge_indices, edge_features)
                Y_a, X_a = self.constraint_variable_gnn(
                    Y_a, formatted_edge_indices_a, formatted_edge_features_a, X_a
                )
                Y_a, X_a = self.format_from_stacked_bipartite(Y_a, X_a, shape_info)

                Y_b, X_b, _, __, shape_info = \
                    self.format_for_stacked_bipartite(Y_b, X_b, edge_indices, edge_features)
                Y_b, X_b = self.constraint_variable_gnn(
                    Y_b, formatted_edge_indices_b, formatted_edge_features_b, X_b
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
        # _log_peak_memory(peak_mem_device)
        return X


class new_GNNPolicy(nn.Module):
    """A wrapper module combining
    - an initial MLP to convert raw features to embeddings in common dimension,
    - a set transformer (InducedSetAttentionBlock, ISAB) to update constraint nodes (constraint attends to each other), a set transformer (InducedSetAttentionBlock, ISAB) to update variable nodes (variable attends to each other)
    - a MP-GNN (StackedBipartiteGNN) for MILP bipartite graphs to exchange information
The set transformer + GNN process can be repeated for a few times controlled by an arg
    - a final MLP on the variable features for candidate choice/scoring
    Remember to add this new_GNNPolicy to utils.py to load -> args.model == 'STGNN', use this model
    """

    def __init__(
        self,
        emb_size,
        cons_nfeats,
        edge_nfeats,
        var_nfeats,
        output_size,
        n_layers,
        num_heads=1,
        isab_num_inds=4,
        use_set_transformer=True,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be >= 1.")

        self.emb_size = emb_size
        self.n_layers = n_layers
        self.use_set_transformer = use_set_transformer and num_heads > 0 and isab_num_inds > 0

        # Embeddings
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.edge_embedding = torch.nn.Sequential(torch.nn.LayerNorm(edge_nfeats))
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Set transformer layers
        self.constraint_set_layers = nn.ModuleList()
        self.variable_set_layers = nn.ModuleList()
        if self.use_set_transformer:
            if emb_size % num_heads != 0:
                raise ValueError(
                    f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})."
                )
            for _ in range(n_layers):
                self.constraint_set_layers.append(
                    InducedSetAttentionBlock(
                        dim_in=emb_size,
                        dim_out=emb_size,
                        num_heads=num_heads,
                        num_inds=isab_num_inds,
                        ln=True,
                    )
                )
                self.variable_set_layers.append(
                    InducedSetAttentionBlock(
                        dim_in=emb_size,
                        dim_out=emb_size,
                        num_heads=num_heads,
                        num_inds=isab_num_inds,
                        ln=True,
                    )
                )

        # Message passing layers
        self.gnn_layers = nn.ModuleList(
            [
                StackedBipartiteGNN(
                    hidden_channels=emb_size, edge_nfeats=edge_nfeats, n_layers=1
                )
                for _ in range(n_layers)
            ]
        )

        # Output head
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_size, bias=False),
        )

    @staticmethod
    def _normalize_sizes(size_tensor, ref, total_fallback):
        if size_tensor is None:
            return torch.tensor([total_fallback], device=ref.device, dtype=torch.long)
        if isinstance(size_tensor, int):
            return torch.tensor([size_tensor], device=ref.device, dtype=torch.long)
        return size_tensor.to(device=ref.device, dtype=torch.long)

    @staticmethod
    def _pad_nodes(nodes, n_per_graph):
        device, dtype = nodes.device, nodes.dtype
        n_per_graph = n_per_graph.to(device=device)
        num_graphs = n_per_graph.numel()
        n_max = int(n_per_graph.max().item())

        if (n_per_graph == n_max).all():
            return nodes.view(num_graphs, n_max, -1), None

        offsets = torch.cumsum(
            torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_per_graph[:-1])),
            dim=0,
        )
        padded = []
        masks = []
        for g in range(num_graphs):
            start = int(offsets[g].item())
            n_current = int(n_per_graph[g].item())
            end = start + n_current
            chunk = nodes[start:end]

            pad_len = n_max - n_current
            if pad_len > 0:
                pad = torch.zeros((pad_len, nodes.size(-1)), device=device, dtype=dtype)
                chunk = torch.cat((chunk, pad), dim=0)
                mask = torch.cat(
                    (
                        torch.zeros(n_current, device=device, dtype=torch.bool),
                        torch.ones(pad_len, device=device, dtype=torch.bool),
                    ),
                    dim=0,
                )
            else:
                mask = torch.zeros(n_max, device=device, dtype=torch.bool)

            padded.append(chunk)
            masks.append(mask)

        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    @staticmethod
    def _unpad_nodes(nodes, n_per_graph):
        n_per_graph = n_per_graph.to(device=nodes.device)
        n_max = nodes.size(1)
        valid_mask = (
            torch.arange(n_max, device=nodes.device).unsqueeze(0) < n_per_graph.unsqueeze(1)
        )
        return nodes[valid_mask].reshape(-1, nodes.size(-1))

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
        del candidates  # unused but kept for API compatibility

        # 1) Embedding
        Y = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        X = self.var_embedding(variable_features)

        n_constraints_per_graph = self._normalize_sizes(
            n_constraints_per_graph, Y, total_fallback=Y.size(0)
        )
        n_variables_per_graph = self._normalize_sizes(
            n_variables_per_graph, X, total_fallback=X.size(0)
        )

        # 2) Set transformer + message passing blocks
        for layer_idx in range(self.n_layers):
            if self.use_set_transformer:
                Y_batched, cons_mask = self._pad_nodes(Y, n_constraints_per_graph)
                Y = self.constraint_set_layers[layer_idx](
                    Y_batched, key_padding_mask=cons_mask
                )
                Y = self._unpad_nodes(Y, n_constraints_per_graph)

                X_batched, var_mask = self._pad_nodes(X, n_variables_per_graph)
                X = self.variable_set_layers[layer_idx](X_batched, key_padding_mask=var_mask)
                X = self._unpad_nodes(X, n_variables_per_graph)

            Y, X = self.gnn_layers[layer_idx](Y, edge_indices, edge_features, X)

        # 3) Output head
        output = self.output_module(X).squeeze(-1)
        return output
