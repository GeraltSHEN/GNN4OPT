# Comparison: `models.py` vs. `holognn/models.py`

This document outlines the architectural relationship between the local MILP-focused implementation (`models.py`) and the original HoloGNN reference (`holognn/models.py`). It is written as a side-by-side guide so you can quickly reason about how the symmetry breaking idea was adapted for bipartite node classification.

## Cheat Sheet

| Theme | `models.py` (this repo) | `holognn/models.py` | Notes |
| --- | --- | --- | --- |
| Data encoder | `BipartiteDataEncoder` (heterogeneous, MILP specific) | Task-specific encoders (`PlanetoidEncoder`, `MovieLensEncoder`, etc.) | Local encoder handles constraint/variable nodes with feature preprocessing and optional anchor injection. |
| Symmetry breaking | `BipartiteHoloTupleEncoder` using the same encoder with one-hot anchors | `Holo` wrapper around an auxiliary breaker (`PowerMethod` or `SymmetryBreakingGNN`) | Local version reuses the hetero encoder rather than a dedicated breaker network. |
| Tuple baselines | `VariableTupleEncoder` (`index_select`) | `ProductTupleEncoder` (element-wise product) | Each baseline mirrors the task: node-level vs. tuple/link scoring. |
| Policy/head | `GNNPolicy` with optional `projection_dim` | `Classifier` with `MLP([in_dim, in_dim, out_dim])` | When `projection_dim` is set, local policy mirrors Holo’s two-hidden-layer projection before the final logit. |
| Support layers | `PreNormLayer`, `BipartiteDataEncoder`, `BipartiteHoloTupleEncoder` | `PowerMethod`, `SymmetryBreakingGNN`, MovieLens SAGE stacks | Auxiliary utilities diverge because of the domain (MILP vs. recommender/citation tasks). |

## Component-by-Component Breakdown

### 1. Data Encoders

- **Local (`models.py:42-140`)**  
  - `BipartiteDataEncoder` consumes `torch_geometric.data.HeteroData`, embedding constraint, edge, and variable features via `PreNormLayer` + MLPs.  
  - Message passing is handled with `HeteroConv` (SAGE or GCN) over the constraint→variable bipartite structure.  
  - A `break_indicator_encoder` injects symmetry-breaking one-hot signals directly into variable features when requested.

- **HoloGNN (`holognn/models.py`, varied)**  
  - Encoders are task-specific: e.g., `PlanetoidEncoder` for homogeneous citation graphs, `MovieLensEncoder` for user–movie bipartite interactions.  
  - Symmetry-breaking information is *not* fused into the encoder; instead, a downstream tuples module handles it.

**Interpretation.** The local implementation consolidates feature embedding and symmetry breaking into one hetero module tailored for MILP branch-and-bound graphs, whereas HoloGNN keeps the core encoder symmetry-agnostic and relies on a separate breaker network.

### 2. Symmetry-Breaking Tuple Encoders

- **Local (`models.py:202-281`)**  
  - `BipartiteHoloTupleEncoder` selects top-degree variables, creates an indicator vector, and re-encodes the graph for each anchor.  
  - It averages or max-pools the candidate embeddings across breakings to produce logits for node-level classification.

- **HoloGNN (`holognn/models.py:81-133`)**  
  - `Holo` duplicates the node embeddings, concatenates one-hot anchor channels, and pushes them through a symmetry-breaking model (`PowerMethod` or `SymmetryBreakingGNN`).  
  - Tuple representations are built via element-wise products across tuple entries, reflecting the original link/set tasks.

**Interpretation.** In the local fork, symmetry breaking is implemented by reusing the main encoder and aggregating per-node embeddings; in HoloGNN it is a distinct module that outputs link-level representations.

### 3. Tuple Baselines

- **Local (`models.py:166-199`)**  
  - `VariableTupleEncoder` simply `index_select`s candidate variable embeddings and caches dimensionality for downstream heads.

- **HoloGNN (`holognn/models.py:44-58`)**  
  - `ProductTupleEncoder` multiplies embeddings for all nodes in a tuple, which suits link prediction and combinatorial set scoring.

**Interpretation.** Baselines mirror the task: the local project cares about node classification among candidate variables, while HoloGNN manipulates sets/edges of nodes.

### 4. Head / Policy Structure

- **Local (`models.py:283-353`)**  
  - `GNNPolicy` composes the encoder, tuple encoder, and head.  
  - The optional `projection_dim` (wired to `args.out_dim`) builds an MLP `[head_dim, head_dim, projection_dim]` before a final linear logit, closely matching Holo’s classifier shape.  
  - Defaults to `MLP([d, d, 1])` when no projection is specified and supports a purely linear head when `linear_classifier=True`.

- **HoloGNN (`holognn/models.py:60-78`)**  
  - `Classifier` wraps arbitrary encoders and uses either an MLP `[in_dim, in_dim, out_dim]` or a single linear layer.  
  - Includes `train_head_only` to freeze encoders when loading pretrained symmetry-breaking weights.

**Interpretation.** The local policy inlines what Holo does with its classifier, while making the projection optional so node-level tasks can fallback to a simple scorer if desired.

### 5. Supporting Utilities

- **Local**  
  - `PreNormLayer` (`model_torch.py`) pre-computes statistics for feature normalization.  
  - Degree-based breaker selection (`_select_break_nodes`) chooses anchors suited to MILP candidate sets.

- **HoloGNN**  
  - `PowerMethod` iteratively multiplies by the adjacency to produce symmetry-breaking signals.  
  - `SymmetryBreakingGNN` is a two-layer GCN dedicated to the breaking process.  
  - Additional dataset-specific SAGE stacks (e.g., `BipartiteSAGEEncoder`) live alongside the general Holo tuple encoder.

**Interpretation.** Each codebase ships auxiliary modules tuned to its domain: MILP scheduling versus link prediction / recommendation.

## How to Use This Summary

1. Keep this document open alongside `models.py` and `holognn/models.py` when adapting ideas between projects.  
2. Extend the comparison table whenever new encoders or tuple modules are introduced to maintain a clear mapping.  
3. If you plan to port further Holo components, consider whether to keep the “shared encoder” design or reintroduce dedicated symmetry-breaking networks for tighter parity.
