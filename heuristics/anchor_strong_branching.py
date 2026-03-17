"""Anchor strong branching with SCIP explicit-dual substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from .utils import (
        SCIPBranchingContext,
        compute_strong_branch_score,
        evaluate_explicit_dual_on_child,
        load_sample,
        unpack_sample_data,
    )
except Exception:  # pragma: no cover - script execution fallback
    from utils import (
        SCIPBranchingContext,
        compute_strong_branch_score,
        evaluate_explicit_dual_on_child,
        load_sample,
        unpack_sample_data,
    )


@dataclass
class AnchorSBResult:
    selected_candidate_global: int
    selected_candidate_pos: int
    candidate_global_indices: np.ndarray
    anchor_candidate_global_indices: np.ndarray
    pseudo_scores: np.ndarray
    child_zero_obj: np.ndarray
    child_one_obj: np.ndarray
    parent_obj: float
    dual_pool_size: int
    topk_positions: np.ndarray
    topk_globals: np.ndarray


def _sanitize_anchor_action_set(
    action_set: np.ndarray,
    top_k_action_set: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_set = set(action_set.tolist())
    invalid = [int(v) for v in top_k_action_set.tolist() if int(v) not in candidate_set]
    if invalid:
        raise ValueError(f"top_k_action_set contains vars not in action_set: {invalid}")

    seen = set()
    unique = []
    for v in top_k_action_set.tolist():
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            unique.append(iv)
    anchor_lp_positions = np.asarray(unique, dtype=np.int64)

    cand_pos_lookup = {int(lp_pos): idx for idx, lp_pos in enumerate(action_set.tolist())}
    anchor_positions = np.asarray([cand_pos_lookup[int(v)] for v in anchor_lp_positions.tolist()], dtype=np.int64)
    return anchor_lp_positions, anchor_positions


def run_anchor_strong_branching(
    context: SCIPBranchingContext,
    action_set: Sequence[int],
    top_k_action_set: Sequence[int],
    cutoffbound: float,
    top_k: Optional[int] = None,
) -> AnchorSBResult:
    """Run anchor SB on one sample using SCIP explicit dual solutions."""

    candidate_lp_positions = np.asarray(action_set, dtype=np.int64)
    anchor_input = np.asarray(top_k_action_set, dtype=np.int64).reshape(-1)

    if candidate_lp_positions.size == 0:
        raise ValueError("No candidates available in action_set.")
    if anchor_input.size == 0:
        raise ValueError("top_k_action_set must be non-empty.")

    anchor_lp_positions, _ = _sanitize_anchor_action_set(candidate_lp_positions, anchor_input)

    parent = context.solve_parent_primal()
    if not parent.success or parent.objective_value is None:
        raise RuntimeError(f"Parent LP solve failed: {parent.status} ({parent.message})")
    parent_obj = float(parent.objective_value)

    candidate_branches = [context.create_branch_bounds(int(v)) for v in candidate_lp_positions.tolist()]
    n_candidates = int(candidate_lp_positions.shape[0])

    dual_pool = []
    for anchor_var in anchor_lp_positions.tolist():
        down_dual = context.solve_child_dual(int(anchor_var), direction="down", cutoffbound=cutoffbound)
        up_dual = context.solve_child_dual(int(anchor_var), direction="up", cutoffbound=cutoffbound)
        if down_dual.success and down_dual.y is not None:
            dual_pool.append({"source": "anchor_down", "anchor_lp_pos": int(anchor_var), "dual": down_dual})
        if up_dual.success and up_dual.y is not None:
            dual_pool.append({"source": "anchor_up", "anchor_lp_pos": int(anchor_var), "dual": up_dual})

    if not dual_pool:
        raise RuntimeError("No valid dual solutions collected from top_k_action_set children.")

    child_zero_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)
    child_one_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)

    for dual_row in dual_pool:
        dual = dual_row["dual"]
        down_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)
        up_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)
        for idx, branch in enumerate(candidate_branches):
            down_eval[idx] = evaluate_explicit_dual_on_child(
                context.lp,
                dual.y,
                dual.alpha,
                dual.beta,
                bound_overrides=branch.down_overrides,
            )
            up_eval[idx] = evaluate_explicit_dual_on_child(
                context.lp,
                dual.y,
                dual.alpha,
                dual.beta,
                bound_overrides=branch.up_overrides,
            )

        np.maximum(child_zero_obj_est, down_eval, out=child_zero_obj_est)
        np.maximum(child_one_obj_est, up_eval, out=child_one_obj_est)

    pseudo_scores = np.array(
        [
            compute_strong_branch_score(parent_obj, child_one_obj_est[i], child_zero_obj_est[i], cutoffbound)
            for i in range(n_candidates)
        ],
        dtype=np.float64,
    )
    best_position = int(np.nanargmax(pseudo_scores))

    k_rank = int(min(max(int(top_k if top_k is not None else anchor_lp_positions.size), 1), n_candidates))
    topk_positions = np.argsort(pseudo_scores)[-k_rank:][::-1].astype(np.int64)
    topk_lp_positions = candidate_lp_positions[topk_positions].astype(np.int64)

    return AnchorSBResult(
        selected_candidate_global=int(candidate_lp_positions[best_position]),
        selected_candidate_pos=best_position,
        candidate_global_indices=candidate_lp_positions,
        anchor_candidate_global_indices=anchor_lp_positions,
        pseudo_scores=pseudo_scores,
        child_zero_obj=child_zero_obj_est,
        child_one_obj=child_one_obj_est,
        parent_obj=parent_obj,
        dual_pool_size=len(dual_pool),
        topk_positions=topk_positions,
        topk_globals=topk_lp_positions,
    )


def run_anchor_strong_branching_from_sample_file(
    sample_path: str,
    k: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> AnchorSBResult:
    sample = load_sample(sample_path)
    record = unpack_sample_data(sample["data"])
    n_candidates = int(record.action_set.shape[0])
    if n_candidates == 0:
        raise ValueError("No candidates available in sample.")
    if rng is None:
        rng = np.random.default_rng()
    k_eff = int(min(max(int(k), 1), n_candidates))
    anchor_positions = np.sort(rng.choice(n_candidates, size=k_eff, replace=False))
    top_k_action_set = record.action_set[anchor_positions]
    context = SCIPBranchingContext.from_sample_state(record.sample_state)
    return run_anchor_strong_branching(
        context=context,
        action_set=record.action_set,
        top_k_action_set=top_k_action_set,
        cutoffbound=float(record.cutoffbound),
        top_k=k_eff,
    )
