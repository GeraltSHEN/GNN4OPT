"""Anchor strong branching implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix

try:
    from .utils import Problem, compute_sbs, evaluate_obj, solve_dual
except Exception:  # pragma: no cover - script execution fallback
    from utils import Problem, compute_sbs, evaluate_obj, solve_dual


@dataclass(frozen=True)
class PreparedSample:
    problem: Problem
    candidate_global_indices: np.ndarray
    candidate_local_indices: np.ndarray
    unfixed_global_indices: np.ndarray


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


def _feature_indices(use_default_features: bool) -> tuple[int, int, int]:
    # Mirrors feature construction in root utils.py / GraphDataset.get.
    if use_default_features:
        return 1, 4, 16  # constraint bias, variable coef_normalized, variable sol_val
    return 0, 9, 10


def prepare_problem_from_graph(graph, use_default_features: bool = False) -> PreparedSample:
    """Recover (b, A_F, x_F, A_U, c_U) style data from one GraphDataset sample."""

    bias_idx, coef_idx, sol_idx = _feature_indices(use_default_features)

    constraint_features = graph.constraint_features.detach().cpu().numpy()
    variable_features = graph.variable_features.detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_attr = graph.edge_attr.detach().cpu().numpy().reshape(-1)

    n_constraints = constraint_features.shape[0]
    n_variables = variable_features.shape[0]

    # `extract_state` stores constraints in <= orientation. Convert to >=.
    A_ub = coo_matrix(
        (edge_attr, (edge_index[0], edge_index[1])),
        shape=(n_constraints, n_variables),
    ).toarray()
    A = -A_ub
    b = -constraint_features[:, bias_idx]

    c = variable_features[:, coef_idx]
    x_val = variable_features[:, sol_idx]

    # Last two columns are appended in GraphDataset.get:
    #   -2: is_not_fixed (from original action_set)
    #   -1: ranking candidate indicator (possibly cleaned)
    is_not_fixed_mask = variable_features[:, -2] > 0.5
    unfixed_global = np.flatnonzero(is_not_fixed_mask).astype(np.int64)
    fixed_global = np.flatnonzero(~is_not_fixed_mask).astype(np.int64)

    A_U = A[:, unfixed_global]
    A_F = A[:, fixed_global]
    x_F = x_val[fixed_global]
    c_U = c[unfixed_global]

    lb_U = np.zeros_like(c_U, dtype=np.float64)
    ub_U = np.ones_like(c_U, dtype=np.float64)

    problem = Problem(
        b=b.astype(np.float64, copy=False),
        A_F=A_F.astype(np.float64, copy=False),
        x_F=x_F.astype(np.float64, copy=False),
        A_U=A_U.astype(np.float64, copy=False),
        c_U=c_U.astype(np.float64, copy=False),
        lb_U=lb_U,
        ub_U=ub_U,
    )

    candidate_global = graph.candidates.detach().cpu().numpy().astype(np.int64)
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(unfixed_global.tolist())}
    candidate_local = np.array([global_to_local[int(g_idx)] for g_idx in candidate_global], dtype=np.int64)

    return PreparedSample(
        problem=problem,
        candidate_global_indices=candidate_global,
        candidate_local_indices=candidate_local,
        unfixed_global_indices=unfixed_global,
    )


def run_anchor_strong_branching(
    graph,
    k: int = 8,
    rng: Optional[np.random.Generator] = None,
    use_default_features: bool = False,
) -> AnchorSBResult:
    """Run anchor strong branching on one sample graph."""

    prepared = prepare_problem_from_graph(graph, use_default_features=use_default_features)
    problem = prepared.problem

    candidate_globals = prepared.candidate_global_indices
    candidate_locals = prepared.candidate_local_indices
    n_candidates = int(candidate_locals.size)
    if n_candidates == 0:
        raise ValueError("No branching candidates available.")

    if rng is None:
        rng = np.random.default_rng()

    k_eff = min(int(k), n_candidates)
    anchor_pos = np.sort(rng.choice(n_candidates, size=k_eff, replace=False))
    anchor_globals = candidate_globals[anchor_pos]

    parent_dual = solve_dual(problem)
    if parent_dual.y is None or parent_dual.alpha is None or parent_dual.beta is None:
        raise RuntimeError(f"Failed to solve parent dual LP: {parent_dual.message}")
    parent_obj = float(parent_dual.objective)

    child_zero_problems = [problem.branch_zero(int(local_idx)) for local_idx in candidate_locals]
    child_one_problems = [problem.branch_one(int(local_idx)) for local_idx in candidate_locals]

    child_zero_obj = np.full(n_candidates, float("-inf"), dtype=np.float64)
    child_one_obj = np.full(n_candidates, float("-inf"), dtype=np.float64)

    dual_pool = [(parent_dual.y, parent_dual.alpha, parent_dual.beta)]

    # Baseline dual bounds from the parent dual solution.
    for i in range(n_candidates):
        child_zero_obj[i] = evaluate_obj(child_zero_problems[i], *dual_pool[0])
        child_one_obj[i] = evaluate_obj(child_one_problems[i], *dual_pool[0])

    # Exact anchor solves + dual pool enrichment.
    for i in anchor_pos:
        zero_res = solve_dual(child_zero_problems[i])
        one_res = solve_dual(child_one_problems[i])

        if np.isfinite(zero_res.objective):
            child_zero_obj[i] = max(child_zero_obj[i], float(zero_res.objective))
        else:
            child_zero_obj[i] = float(zero_res.objective)
        if np.isfinite(one_res.objective):
            child_one_obj[i] = max(child_one_obj[i], float(one_res.objective))
        else:
            child_one_obj[i] = float(one_res.objective)

        if zero_res.y is not None and zero_res.alpha is not None and zero_res.beta is not None:
            dual_pool.append((zero_res.y, zero_res.alpha, zero_res.beta))
        if one_res.y is not None and one_res.alpha is not None and one_res.beta is not None:
            dual_pool.append((one_res.y, one_res.alpha, one_res.beta))

    # Evaluate all pool duals on all child objectives to get pseudo bounds.
    for y, alpha, beta in dual_pool[1:]:
        for i in range(n_candidates):
            if not np.isinf(child_zero_obj[i]):
                child_zero_obj[i] = max(child_zero_obj[i], evaluate_obj(child_zero_problems[i], y, alpha, beta))
            if not np.isinf(child_one_obj[i]):
                child_one_obj[i] = max(child_one_obj[i], evaluate_obj(child_one_problems[i], y, alpha, beta))

    pseudo_scores = np.array(
        [compute_sbs(parent_obj, child_one_obj[i], child_zero_obj[i]) for i in range(n_candidates)],
        dtype=np.float64,
    )

    selected_pos = int(np.nanargmax(pseudo_scores))
    selected_global = int(candidate_globals[selected_pos])

    return AnchorSBResult(
        selected_candidate_global=selected_global,
        selected_candidate_pos=selected_pos,
        candidate_global_indices=candidate_globals,
        anchor_candidate_global_indices=anchor_globals,
        pseudo_scores=pseudo_scores,
        child_zero_obj=child_zero_obj,
        child_one_obj=child_one_obj,
        parent_obj=parent_obj,
    )
