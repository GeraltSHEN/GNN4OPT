#!/usr/bin/env python3
"""
Demonstrate candidate padding + top-k selection line by line.

Run:
  python scripts/demo_topk_tensor_ops.py
"""

import torch


def show(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(tensor)
    print("-" * 80)


def main():
    # Example setup:
    # - 3 graphs in the batch
    # - scores are over GLOBAL variable indices [0..8]
    # - candidates are flattened graph-by-graph and use global indices
    scores = torch.tensor([0.20, 0.70, 0.10, 0.90, 0.30, 0.80, 0.60, 0.50, 0.40], dtype=torch.float32)
    candidates = torch.tensor(
        [
            0, 3, 5,        # graph 0 candidates (3)
            6, 8,           # graph 1 candidates (2)
            1, 2, 4, 7,     # graph 2 candidates (4)
        ],
        dtype=torch.long,
    )
    nb_candidates = torch.tensor([3, 2, 4], dtype=torch.long)
    bsz = nb_candidates.numel()
    k = 2

    print("\nInput tensors")
    show("scores (global variable logits)", scores)
    show("candidates (flattened by graph, global ids)", candidates)
    show("nb_candidates (per graph)", nb_candidates)
    print(f"bsz={bsz}, k={k}")
    print("=" * 80)

    candidate_scores = scores[candidates]
    show("candidate_scores = scores[candidates]", candidate_scores)

    # ----- Exact lines from your code -----
    max_cands = int(nb_candidates.max().item())
    show("max_cands (scalar wrapped for display)", torch.tensor(max_cands))

    row_ids = torch.repeat_interleave(torch.arange(bsz, device=scores.device), nb_candidates)
    show("row_ids", row_ids)

    cand_offsets = torch.cumsum(
        torch.cat((torch.zeros(1, device=scores.device, dtype=torch.long), nb_candidates[:-1])),
        dim=0,
    )
    show("cand_offsets", cand_offsets)

    local_idx = torch.arange(candidates.numel(), device=scores.device) - cand_offsets[row_ids]
    show("local_idx", local_idx)

    print("Each candidate goes to padded[row_ids[i], local_idx[i]]:")
    print(torch.stack([row_ids, local_idx], dim=1))
    print("-" * 80)

    padded_scores = scores.new_full((bsz, max_cands), -1e8)
    padded_candidates = candidates.new_full((bsz, max_cands), -1)
    show("padded_scores (initialized)", padded_scores)
    show("padded_candidates (initialized)", padded_candidates)

    padded_scores[row_ids, local_idx] = candidate_scores
    padded_candidates[row_ids, local_idx] = candidates
    show("padded_scores (after scatter)", padded_scores)
    show("padded_candidates (after scatter)", padded_candidates)

    top_local = padded_scores.topk(k=k, dim=-1).indices
    show("top_local = padded_scores.topk(k, dim=-1).indices", top_local)

    branching_candidates_global = padded_candidates.gather(1, top_local)
    show("branching_candidates_global = padded_candidates.gather(1, top_local)", branching_candidates_global)

    top_scores = padded_scores.gather(1, top_local)
    show("top_scores = padded_scores.gather(1, top_local)", top_scores)

    print("Sanity check by graph:")
    cand_splits = torch.split(candidates, nb_candidates.tolist())
    score_splits = torch.split(candidate_scores, nb_candidates.tolist())
    for g, (cand_g, score_g) in enumerate(zip(cand_splits, score_splits)):
        manual_top = torch.topk(score_g, k=k).indices
        print(
            f"graph {g}: candidates={cand_g.tolist()}, scores={score_g.tolist()}, "
            f"manual top candidates={cand_g[manual_top].tolist()}"
        )


if __name__ == "__main__":
    main()
