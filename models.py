from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor


class PowerMethod(nn.Module):
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


class Holo(torch.nn.Module):
    """The Holo-GNN tuple encoder using symmetry breaking."""

    def __init__(self, *, n_breakings: int, symmetry_breaking_model):
        super().__init__()
        self.n_breakings = n_breakings
        self.symmetry_breaking_model = symmetry_breaking_model

        self.ln = torch.nn.LayerNorm(symmetry_breaking_model.out_dim)
    
    def tied_topk_indices(self, values, k, expansion=2):
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

    def get_nodes_to_break(self, adj_t, n_breakings=8):
        node_degrees = adj_t.sum(-1)
        return self.tied_topk_indices(values=node_degrees, k=n_breakings)

    def forward(self, X, adj_t, tuples_coo, group_idx=None):
        # X: (n, d)
        break_node_indices = self.get_nodes_to_break(
            adj_t, n_breakings=self.n_breakings
        )

        # break_node_indices: (t,)
        one_hot_breakings = F.one_hot(break_node_indices, X.size(0)).unsqueeze(-1)
        # one_hot_breakings: (t, n, 1)
        # expand to holographic node representations
        holo_repr = self.symmetry_breaking_model(
            torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    one_hot_breakings,
                ),
                dim=-1,
            ),  # (t, n, d+1)
            adj_t,
        )  # (t, n, d+1), where n includes both movies and users
        holo_repr = self.ln(holo_repr)
        # holo_repr: (t, n, d+1=symmetry_breaking_model.out_dim)

        # aggregate (elementwise multiplication) orders, r
        set_of_link_repr = holo_repr[:, tuples_coo].prod(dim=1)  # (t, r, k, d+1) -> (t, k, d+1)
        # set_of_link_repr: (t, k=n_tuples, d+1)

        # aggregate (average over) all views, t
        if group_idx is not None:
            link_repr = torch_scatter.scatter(
                set_of_link_repr, group_idx, 0, reduce="mean"
            )
        else:
            link_repr = set_of_link_repr.mean(0, keepdim=True)  # (l=1, k, d+1)
        # flatten to a list if multiple symmetry breakings groups are used
        return link_repr.transpose(0, 1).flatten(1, 2)  # (k, (d+1)*l)


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
    - a tuple encoder (Holo, or ProductTupleEncoder),
    - a final MLP on the variable features for candidate choice/scoring
    """
    def __init__(self, emb_size, cons_nfeats, edge_nfeats, var_nfeats, output_size,
                 n_layers, tuple_encoder, r=1):
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
        if n_layers == 1:
            self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats)
            self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats)
        else:
            for i in range(n_layers):
                setattr(self, f"conv_{i}_v_to_c", BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats))
                setattr(self, f"conv_{i}_c_to_v", BipartiteGraphConvolution(emb_size=emb_size, edge_nfeats=edge_nfeats))
        
        # TUPLE ENCODER
        self.r = r
        self.tuple_encoder = tuple_encoder

        # FINAL MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_size, bias=False),
        )

    def get_holo_input(
        self,
        variable_features,
        constraint_features,
        edge_indices,
        reversed_edge_indices,
    ):
        """
        X: stacked node features (variable + constraint)
        adj_t: homogeneous and unweighted adjacency matrix of the bipartite graph
            make it symmetric to allow power method, i.e., adj_t^k X
        tuples_coo: batched indices (r=1: node, r=2: edge, r=3: triplet, etc.) for the task
            as the task's dataset is multi-graphs rather than a single graph, 
            no CooSampler was used to sample sub-graphs to produce batches. 
            Instead, PyG default data and loader produces batches.
        """
        n_variables = variable_features.size(0)
        n_constraints = constraint_features.size(0)
        X = torch.vstack((variable_features, constraint_features))

        var_to_cons = reversed_edge_indices.clone()
        var_to_cons[1] = var_to_cons[1] + n_variables
        cons_to_var = edge_indices.clone()
        cons_to_var[0] = cons_to_var[0] + n_variables
        homogeneous_edge_index = torch.hstack((var_to_cons, cons_to_var))
        adj_t = SparseTensor(
                row=homogeneous_edge_index[0],
                col=homogeneous_edge_index[1],
                sparse_sizes=(n_variables + n_constraints, n_variables + n_constraints),
            )
        adj_t = gcn_norm(adj_t, add_self_loops=True)

        if self.r == 1:
            # tuples_coo is the variable indices, i.e., the first n_variables rows of X
            tuples_coo = torch.arange(n_variables).unsqueeze(0)
            # tuples_coo: (r=1, k=n_variables)
        elif self.r == 2:
            # tuples_coo is the reversed_edge_indices with offset
            new_entity_index = []
            reversed_relation_schema = ("variable", "constraint")
            for entity, entity_index in zip(reversed_relation_schema, reversed_edge_indices):
                offset = 0 if entity == "variable" else n_variables
                new_entity_index.append(entity_index + offset)
            tuples_coo = torch.vstack(new_entity_index)
            # tuples_coo: (r=2, k=n_edges)
            raise ValueError(f"r should be 1 for branching task")
        else:
            raise NotImplementedError(f"get_tuples_coo for `r={self.r}` not implemented yet.")

        return X, adj_t, tuples_coo

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        n_constraints = constraint_features.size(0)
        n_variables = variable_features.size(0)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        adj_t = self.get_adj_t(edge_indices, reversed_edge_indices, edge_features,
                               n_constraints, n_variables)

        # 1. raw features to embeddings in common dimension
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # 2. two half convolutions
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
        
        # 3. break symmetry
        X, adj_t, tuples_coo = self.get_holo_input(
            variable_features,
            constraint_features,
            edge_indices,
            reversed_edge_indices,
        )
        variable_features = self.tuple_encoder(X, adj_t, tuples_coo, group_idx=None)

        # 4. transform variable features to strong branching decision
        output = self.output_module(variable_features).squeeze(-1)
        return output
