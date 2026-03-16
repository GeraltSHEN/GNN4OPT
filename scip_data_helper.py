import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pyscipopt as scip

import numpy as np
import scipy.sparse as sp


SampleState = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
Bound = Tuple[Optional[float], Optional[float]]
BoundOverrides = Dict[int, Bound]


@dataclass(frozen=True)
class SampleRecord:
    sample_state: SampleState
    action: int
    action_set: np.ndarray
    scores: np.ndarray
    cutoffbound: float


@dataclass
class LPComponents:
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    objective_coefficients: np.ndarray
    lp_solution: np.ndarray
    bounds: List[Bound]
    objective_sense: str
    objective_offset: float
    row_duals: Optional[np.ndarray]
    reduced_costs: Optional[np.ndarray]


@dataclass
class SolveResult:
    success: bool
    status: str
    message: str
    objective_value: Optional[float]
    x: Optional[np.ndarray]
    row_duals: Optional[np.ndarray]
    reduced_costs: Optional[np.ndarray]
    raw_result: Any = None


@dataclass
class DualSolveResult:
    success: bool
    status: str
    message: str
    objective_value: float
    y: Optional[np.ndarray]
    alpha: Optional[np.ndarray]
    beta: Optional[np.ndarray]
    raw_result: Any = None


@dataclass(frozen=True)
class BranchBounds:
    var_idx: int
    lpsol: float
    lb_local: float
    ub_local: float
    down_ub: float
    up_lb: float
    down_overrides: BoundOverrides
    up_overrides: BoundOverrides


def load_sample(path: Union[str, Path]) -> Dict[str, Any]:
    with gzip.open(Path(path), "rb") as fh:
        return pickle.load(fh)


def unpack_sample_data(data: Sequence[Any]) -> SampleRecord:
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected sample['data'] to be list/tuple, got {type(data)}")
    if len(data) < 5:
        raise ValueError(f"Expected at least 5 entries in sample['data'], got {len(data)}")

    sample_state = data[0]
    sample_action = int(data[2])
    sample_action_set = np.asarray(data[3], dtype=np.int64)
    sample_scores = np.asarray(data[4], dtype=np.float64)
    cutoffbound = float(data[5]) if len(data) >= 6 and data[5] is not None else float("inf")

    return SampleRecord(
        sample_state=sample_state,
        action=sample_action,
        action_set=sample_action_set,
        scores=sample_scores,
        cutoffbound=cutoffbound,
    )


def _column(values: np.ndarray, indices: Dict[str, int], name: str) -> np.ndarray:
    if name not in indices:
        raise KeyError(f"Missing required feature '{name}'. Available: {list(indices)}")
    idx = indices[name]
    if values.ndim == 1:
        if idx != 0:
            raise ValueError(f"Cannot access feature '{name}' at index {idx} from 1D array")
        return values
    return values[:, idx]


def reconstruct_unnormalized_state(sample_state: SampleState) -> SampleState:
    if not isinstance(sample_state, (list, tuple)) or len(sample_state) != 3:
        raise ValueError("sample_state must be a (constraint_dict, edge_dict, variable_dict) tuple/list")

    constraint_dict, edge_dict, variable_dict = sample_state
    normalization = constraint_dict.get("normalization")
    if normalization is None:
        raise ValueError(
            "Missing normalization metadata. Regenerate samples with updated "
            "legacy_code_generator/utilities.py to reconstruct unnormalized features."
        )

    obj_norm = float(normalization["obj_norm"])
    age_norm_denom = float(normalization["age_norm_denom"])
    constraint_row_norms = np.asarray(normalization["constraint_row_norms"], dtype=np.float64).reshape(-1)

    constraint_values = np.asarray(constraint_dict["values"], dtype=np.float64).copy()
    edge_values = np.asarray(edge_dict["values"], dtype=np.float64).copy()
    variable_values = np.asarray(variable_dict["values"], dtype=np.float64).copy()
    edge_indices = np.asarray(edge_dict["indices"], dtype=np.int64)

    if constraint_values.shape[0] != constraint_row_norms.shape[0]:
        raise ValueError(
            "constraint_row_norms length does not match number of constraints "
            f"({constraint_row_norms.shape[0]} vs {constraint_values.shape[0]})"
        )

    constraint_feature_indices = {name: idx for idx, name in enumerate(constraint_dict["names"])}
    edge_feature_indices = {name: idx for idx, name in enumerate(edge_dict["names"])}
    variable_feature_indices = {name: idx for idx, name in enumerate(variable_dict["names"])}

    if "bias" in constraint_feature_indices:
        constraint_values[:, constraint_feature_indices["bias"]] *= constraint_row_norms
    if "dualsol_val_normalized" in constraint_feature_indices:
        constraint_values[:, constraint_feature_indices["dualsol_val_normalized"]] *= (
            constraint_row_norms * obj_norm
        )
    if "age" in constraint_feature_indices:
        constraint_values[:, constraint_feature_indices["age"]] *= age_norm_denom

    if "coef_normalized" in edge_feature_indices:
        edge_rows = edge_indices[0]
        edge_values[:, edge_feature_indices["coef_normalized"]] *= constraint_row_norms[edge_rows]

    if "coef_normalized" in variable_feature_indices:
        variable_values[:, variable_feature_indices["coef_normalized"]] *= obj_norm
    if "reduced_cost" in variable_feature_indices:
        variable_values[:, variable_feature_indices["reduced_cost"]] *= obj_norm
    if "age" in variable_feature_indices:
        variable_values[:, variable_feature_indices["age"]] *= age_norm_denom

    return (
        {
            "names": constraint_dict["names"],
            "values": constraint_values,
            "normalization": normalization,
        },
        {
            "names": edge_dict["names"],
            "indices": edge_indices,
            "values": edge_values,
        },
        {
            "names": variable_dict["names"],
            "values": variable_values,
            "reconstruction": variable_dict.get("reconstruction", {}),
        },
    )


def _build_bounds(
    variable_dict: Dict[str, Any],
    variable_values: np.ndarray,
    variable_feature_indices: Dict[str, int],
) -> List[Bound]:
    n_vars = variable_values.shape[0]
    reconstruction = variable_dict.get("reconstruction", {})

    lbs = np.full(n_vars, np.nan, dtype=np.float64)
    ubs = np.full(n_vars, np.nan, dtype=np.float64)

    if "lbs" in reconstruction:
        rec_lbs = np.asarray(reconstruction["lbs"], dtype=np.float64).reshape(-1)
        if rec_lbs.shape[0] == n_vars:
            lbs = rec_lbs
    if "ubs" in reconstruction:
        rec_ubs = np.asarray(reconstruction["ubs"], dtype=np.float64).reshape(-1)
        if rec_ubs.shape[0] == n_vars:
            ubs = rec_ubs

    # v2 convention: always assume missing binary bounds are [0, 1].
    if "type_0" in variable_feature_indices:
        is_binary = variable_values[:, variable_feature_indices["type_0"]] > 0.5
        lbs[np.isnan(lbs) & is_binary] = 0.0
        ubs[np.isnan(ubs) & is_binary] = 1.0

    bounds: List[Bound] = []
    for lb, ub in zip(lbs, ubs):
        lbv = None if np.isnan(lb) or np.isinf(lb) else float(lb)
        ubv = None if np.isnan(ub) or np.isinf(ub) else float(ub)
        bounds.append((lbv, ubv))
    return bounds


def extract_lp_components(unnormalized_state: SampleState) -> LPComponents:
    constraint_dict, edge_dict, variable_dict = unnormalized_state

    constraint_values = np.asarray(constraint_dict["values"], dtype=np.float64)
    edge_values = np.asarray(edge_dict["values"], dtype=np.float64)
    variable_values = np.asarray(variable_dict["values"], dtype=np.float64)
    edge_indices = np.asarray(edge_dict["indices"], dtype=np.int64)

    if edge_indices.ndim != 2 or edge_indices.shape[0] != 2:
        raise ValueError(f"edge indices must have shape [2, nnz], got {edge_indices.shape}")

    constraint_feature_indices = {name: idx for idx, name in enumerate(constraint_dict["names"])}
    edge_feature_indices = {name: idx for idx, name in enumerate(edge_dict["names"])}
    variable_feature_indices = {name: idx for idx, name in enumerate(variable_dict["names"])}

    b_ub = _column(constraint_values, constraint_feature_indices, "bias")
    c = _column(variable_values, variable_feature_indices, "coef_normalized")
    x_lp = _column(variable_values, variable_feature_indices, "sol_val")
    edge_coef = _column(edge_values, edge_feature_indices, "coef_normalized")

    n_constraints = constraint_values.shape[0]
    n_variables = variable_values.shape[0]
    A_ub = sp.coo_matrix(
        (edge_coef, (edge_indices[0], edge_indices[1])),
        shape=(n_constraints, n_variables),
    ).tocsr()

    reduced_costs = None
    if "reduced_cost" in variable_feature_indices:
        reduced_costs = _column(variable_values, variable_feature_indices, "reduced_cost")

    row_duals = None
    if "dualsol_val_normalized" in constraint_feature_indices:
        row_duals = _column(constraint_values, constraint_feature_indices, "dualsol_val_normalized")

    reconstruction = variable_dict.get("reconstruction", {})
    objective_sense = str(reconstruction.get("objective_sense", "minimize")).lower()
    objective_offset = float(reconstruction.get("objective_offset", 0.0))

    bounds = _build_bounds(variable_dict, variable_values, variable_feature_indices)

    return LPComponents(
        A_ub=A_ub,
        b_ub=np.asarray(b_ub, dtype=np.float64),
        objective_coefficients=np.asarray(c, dtype=np.float64),
        lp_solution=np.asarray(x_lp, dtype=np.float64),
        bounds=bounds,
        objective_sense=objective_sense,
        objective_offset=objective_offset,
        row_duals=None if row_duals is None else np.asarray(row_duals, dtype=np.float64),
        reduced_costs=None if reduced_costs is None else np.asarray(reduced_costs, dtype=np.float64),
    )


def compute_primal_objective(lp: LPComponents, x: Optional[np.ndarray] = None) -> float:
    coeffs = np.asarray(lp.objective_coefficients, dtype=np.float64)
    vals = lp.lp_solution if x is None else np.asarray(x, dtype=np.float64)
    return float(np.dot(coeffs, vals) + lp.objective_offset)


def compute_dual_plus_reduced_cost_objective(
    lp: LPComponents,
    row_duals: Optional[np.ndarray] = None,
    reduced_costs: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    include_offset: bool = False,
) -> float:
    y = lp.row_duals if row_duals is None else np.asarray(row_duals, dtype=np.float64)
    rc = lp.reduced_costs if reduced_costs is None else np.asarray(reduced_costs, dtype=np.float64)
    if y is None or rc is None:
        raise ValueError("row_duals and reduced_costs are required")

    vals = lp.lp_solution if x is None else np.asarray(x, dtype=np.float64)
    value = float(np.dot(y, lp.b_ub) + np.dot(rc, vals))
    if include_offset:
        value += lp.objective_offset
    return value


def compute_strong_branch_score(
    parent_obj: float,
    child_one_obj: float,
    child_zero_obj: float,
    cutoffbound: float,
) -> float:
    child_one_obj_capped = min(float(child_one_obj), float(cutoffbound))
    child_zero_obj_capped = min(float(child_zero_obj), float(cutoffbound))
    gain_one = max(child_one_obj_capped - float(parent_obj), 1e-9)
    gain_zero = max(child_zero_obj_capped - float(parent_obj), 1e-9)
    return float(gain_one * gain_zero)


def _bounds_to_arrays(bounds: Sequence[Bound], n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    lbs = np.full(n_vars, -np.inf, dtype=np.float64)
    ubs = np.full(n_vars, +np.inf, dtype=np.float64)
    for j in range(n_vars):
        lb, ub = bounds[j]
        if lb is not None:
            lbs[j] = float(lb)
        if ub is not None:
            ubs[j] = float(ub)
    return lbs, ubs


def _apply_bound_overrides(
    lbs: np.ndarray,
    ubs: np.ndarray,
    bound_overrides: Optional[BoundOverrides],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    lbs = np.asarray(lbs, dtype=np.float64).copy()
    ubs = np.asarray(ubs, dtype=np.float64).copy()
    if bound_overrides is None:
        return lbs, ubs, True

    for j, (lb_override, ub_override) in bound_overrides.items():
        if lb_override is not None:
            lbs[j] = max(lbs[j], float(lb_override))
        if ub_override is not None:
            ubs[j] = min(ubs[j], float(ub_override))
        if lbs[j] > ubs[j]:
            return lbs, ubs, False
    return lbs, ubs, True


def solve_lp_with_scip(
    lp: LPComponents,
    bound_overrides: Optional[BoundOverrides] = None,
    display_verblevel: int = 0,
) -> SolveResult:
    c = np.asarray(lp.objective_coefficients, dtype=np.float64)
    b_ub = np.asarray(lp.b_ub, dtype=np.float64)
    A_ub = lp.A_ub.tocsr()
    maximize = lp.objective_sense.startswith("max")

    parent_lbs, parent_ubs = _bounds_to_arrays(lp.bounds, int(c.shape[0]))
    lbs, ubs, feasible = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not feasible:
        return SolveResult(
            success=False,
            status="infeasible",
            message="infeasible after applying bound overrides",
            objective_value=None,
            x=None,
            row_duals=None,
            reduced_costs=None,
            raw_result=None,
        )

    model = scip.Model()
    model.setIntParam("display/verblevel", int(display_verblevel))

    inf = model.infinity()
    vars_ = []
    for j in range(c.shape[0]):
        lbv = -inf if not np.isfinite(lbs[j]) else float(lbs[j])
        ubv = +inf if not np.isfinite(ubs[j]) else float(ubs[j])
        vars_.append(model.addVar(name=f"x_{j}", vtype="C", lb=lbv, ub=ubv))

    obj_expr = scip.quicksum(float(c[j]) * vars_[j] for j in range(c.shape[0]))
    model.setObjective(obj_expr, sense="maximize" if maximize else "minimize")

    cons_ = []
    for i in range(A_ub.shape[0]):
        row_start, row_end = A_ub.indptr[i], A_ub.indptr[i + 1]
        idxs = A_ub.indices[row_start:row_end]
        vals = A_ub.data[row_start:row_end]
        expr = scip.quicksum(float(v) * vars_[int(j)] for j, v in zip(idxs, vals))
        cons_.append(model.addCons(expr <= float(b_ub[i]), name=f"c_{i}"))

    model.optimize()
    status = str(model.getStatus()).lower()
    if status != "optimal":
        return SolveResult(
            success=False,
            status=status,
            message=status,
            objective_value=None,
            x=None,
            row_duals=None,
            reduced_costs=None,
            raw_result=model,
        )

    sol = model.getBestSol()
    x = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
    row_duals = np.asarray([float(model.getDualsolLinear(cons)) for cons in cons_], dtype=np.float64)
    reduced_costs = np.asarray([float(model.getVarRedcost(var)) for var in vars_], dtype=np.float64)
    obj = float(model.getObjVal()) + lp.objective_offset
    return SolveResult(
        success=True,
        status=status,
        message=status,
        objective_value=obj,
        x=x,
        row_duals=row_duals,
        reduced_costs=reduced_costs,
        raw_result=model,
    )


def solve_explicit_dual_with_scip(
    lp: LPComponents,
    bound_overrides: Optional[BoundOverrides] = None,
    display_verblevel: int = 0,
) -> DualSolveResult:
    if lp.objective_sense.startswith("max"):
        raise ValueError("Explicit dual solver currently supports minimization problems only.")

    c = np.asarray(lp.objective_coefficients, dtype=np.float64)
    A_ub = lp.A_ub.tocsr()
    b_ub = np.asarray(lp.b_ub, dtype=np.float64)

    A = (-A_ub).tocsr()
    rhs = -b_ub

    parent_lbs, parent_ubs = _bounds_to_arrays(lp.bounds, int(c.shape[0]))
    lbs, ubs, feasible = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not feasible:
        return DualSolveResult(
            success=False,
            status="infeasible",
            message="infeasible after applying bound overrides",
            objective_value=float("inf"),
            y=None,
            alpha=None,
            beta=None,
            raw_result=None,
        )

    if np.any(~np.isfinite(lbs)) or np.any(~np.isfinite(ubs)):
        return DualSolveResult(
            success=False,
            status="unsupported_infinite_bounds",
            message="Explicit dual solve expects finite variable bounds.",
            objective_value=float("nan"),
            y=None,
            alpha=None,
            beta=None,
            raw_result=None,
        )

    m_rows, n_vars = A.shape
    model = scip.Model()
    model.setIntParam("display/verblevel", int(display_verblevel))

    y = [model.addVar(name=f"y_{i}", vtype="C", lb=0.0) for i in range(m_rows)]
    alpha = [model.addVar(name=f"alpha_{j}", vtype="C", lb=0.0) for j in range(n_vars)]
    beta = [model.addVar(name=f"beta_{j}", vtype="C", lb=0.0) for j in range(n_vars)]

    A_csc = A.tocsc()
    for j in range(n_vars):
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        row_ids = A_csc.indices[start:end]
        vals = A_csc.data[start:end]
        lhs = scip.quicksum(float(v) * y[int(i)] for i, v in zip(row_ids, vals)) - alpha[j] + beta[j]
        model.addCons(lhs == float(c[j]), name=f"dual_eq_{j}")

    obj_expr = (
        scip.quicksum(float(rhs[i]) * y[i] for i in range(m_rows))
        - scip.quicksum(float(ubs[j]) * alpha[j] for j in range(n_vars))
        + scip.quicksum(float(lbs[j]) * beta[j] for j in range(n_vars))
    )
    model.setObjective(obj_expr, sense="maximize")
    model.optimize()

    status = str(model.getStatus()).lower()
    if status != "optimal":
        obj = float("inf") if status in {"unbounded", "inforunbd"} else float("nan")
        return DualSolveResult(
            success=False,
            status=status,
            message=status,
            objective_value=obj,
            y=None,
            alpha=None,
            beta=None,
            raw_result=model,
        )

    sol = model.getBestSol()
    y_val = np.asarray([model.getSolVal(sol, var) for var in y], dtype=np.float64)
    alpha_val = np.asarray([model.getSolVal(sol, var) for var in alpha], dtype=np.float64)
    beta_val = np.asarray([model.getSolVal(sol, var) for var in beta], dtype=np.float64)
    obj = float(model.getObjVal()) + lp.objective_offset
    return DualSolveResult(
        success=True,
        status=status,
        message=status,
        objective_value=obj,
        y=y_val,
        alpha=alpha_val,
        beta=beta_val,
        raw_result=model,
    )


def evaluate_explicit_dual_on_child(
    lp: LPComponents,
    y: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    bound_overrides: Optional[BoundOverrides] = None,
) -> float:
    c = np.asarray(lp.objective_coefficients, dtype=np.float64)
    b_ub = np.asarray(lp.b_ub, dtype=np.float64)
    rhs = -b_ub
    lbs, ubs, feasible = _apply_bound_overrides(*_bounds_to_arrays(lp.bounds, int(c.shape[0])), bound_overrides)
    if not feasible:
        return float("inf")
    if np.any(~np.isfinite(lbs)) or np.any(~np.isfinite(ubs)):
        return float("nan")
    value = float(np.dot(rhs, y) - np.dot(ubs, alpha) + np.dot(lbs, beta) + lp.objective_offset)
    return value


class SCIPBranchingContext:
    def __init__(self, lp_components: LPComponents):
        self.lp = lp_components
        self._parent_primal: Optional[SolveResult] = None
        self._parent_dual: Optional[DualSolveResult] = None

        n_vars = int(np.asarray(lp_components.objective_coefficients, dtype=np.float64).shape[0])
        self.parent_lbs, self.parent_ubs = _bounds_to_arrays(lp_components.bounds, n_vars)

    @classmethod
    def from_sample_state(cls, sample_state: SampleState) -> "SCIPBranchingContext":
        unnormalized = reconstruct_unnormalized_state(sample_state)
        lp = extract_lp_components(unnormalized)
        return cls(lp)

    def solve_parent_primal(self) -> SolveResult:
        if self._parent_primal is None:
            self._parent_primal = solve_lp_with_scip(self.lp)
        return self._parent_primal

    def solve_parent_dual(self) -> DualSolveResult:
        if self._parent_dual is None:
            self._parent_dual = solve_explicit_dual_with_scip(self.lp)
        return self._parent_dual

    def create_branch_bounds(self, var_idx: int) -> BranchBounds:
        parent = self.solve_parent_primal()
        if not parent.success or parent.x is None:
            raise RuntimeError(f"Parent LP solve failed: {parent.status} ({parent.message})")

        lpsol = float(parent.x[var_idx])
        lb_local = float(self.parent_lbs[var_idx])
        ub_local = float(self.parent_ubs[var_idx])

        down_ub = min(float(np.floor(lpsol)), ub_local)
        up_lb = max(float(np.ceil(lpsol)), lb_local)

        return BranchBounds(
            var_idx=int(var_idx),
            lpsol=lpsol,
            lb_local=lb_local,
            ub_local=ub_local,
            down_ub=down_ub,
            up_lb=up_lb,
            down_overrides={int(var_idx): (None, down_ub)},
            up_overrides={int(var_idx): (up_lb, None)},
        )

    def solve_child_primal(self, var_idx: int, direction: str) -> SolveResult:
        bounds = self.create_branch_bounds(int(var_idx))
        if direction == "down":
            return solve_lp_with_scip(self.lp, bound_overrides=bounds.down_overrides)
        if direction == "up":
            return solve_lp_with_scip(self.lp, bound_overrides=bounds.up_overrides)
        raise ValueError(f"Unknown direction '{direction}', expected 'down' or 'up'")

    def solve_child_dual(self, var_idx: int, direction: str) -> DualSolveResult:
        bounds = self.create_branch_bounds(int(var_idx))
        if direction == "down":
            return solve_explicit_dual_with_scip(self.lp, bound_overrides=bounds.down_overrides)
        if direction == "up":
            return solve_explicit_dual_with_scip(self.lp, bound_overrides=bounds.up_overrides)
        raise ValueError(f"Unknown direction '{direction}', expected 'down' or 'up'")


def reconstruct_topk_strong_branching_scores(
    context: SCIPBranchingContext,
    action_set: Sequence[int],
    candidate_scores: Sequence[float],
    cutoffbound: float,
    top_k: int = 8,
) -> Dict[str, Any]:
    candidate_lp_positions = np.asarray(action_set, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)

    if candidate_lp_positions.shape[0] != candidate_scores.shape[0]:
        raise ValueError(
            f"action_set and candidate_scores size mismatch "
            f"({candidate_lp_positions.shape[0]} vs {candidate_scores.shape[0]})"
        )
    if candidate_lp_positions.size == 0:
        return {
            "k": 0,
            "parent_obj": None,
            "topk_by_score": [],
            "topk_rank_original": [],
            "topk_rank_computed": [],
        }

    parent = context.solve_parent_primal()
    if not parent.success or parent.objective_value is None:
        raise RuntimeError(f"Parent LP solve failed: {parent.status} ({parent.message})")
    parent_obj = float(parent.objective_value)

    k_eff = int(min(max(int(top_k), 0), candidate_scores.size))
    topk_positions = np.argsort(candidate_scores)[-k_eff:][::-1]

    records: List[Dict[str, Any]] = []
    for pos in topk_positions.tolist():
        var_idx = int(candidate_lp_positions[pos])
        branch = context.create_branch_bounds(var_idx)

        down_res = solve_lp_with_scip(context.lp, bound_overrides=branch.down_overrides)
        up_res = solve_lp_with_scip(context.lp, bound_overrides=branch.up_overrides)

        child_zero_obj = float("inf") if (not down_res.success or down_res.objective_value is None) else float(down_res.objective_value)
        child_one_obj = float("inf") if (not up_res.success or up_res.objective_value is None) else float(up_res.objective_value)

        computed_score = compute_strong_branch_score(
            parent_obj=parent_obj,
            child_one_obj=child_one_obj,
            child_zero_obj=child_zero_obj,
            cutoffbound=float(cutoffbound),
        )

        records.append(
            {
                "cand_position": int(pos),
                "cand_lp_pos": var_idx,
                "cand_score": float(candidate_scores[pos]),
                "lpsol": branch.lpsol,
                "lb_local": branch.lb_local,
                "ub_local": branch.ub_local,
                "down_ub": branch.down_ub,
                "up_lb": branch.up_lb,
                "parent_lp_obj": parent_obj,
                "child_zero_lp_obj": child_zero_obj,
                "child_one_lp_obj": child_one_obj,
                "computed_score": computed_score,
                "down_status": down_res.status,
                "up_status": up_res.status,
            }
        )

    original_rank = sorted(records, key=lambda row: (-float(row["cand_score"]), int(row["cand_position"])))
    computed_rank = sorted(records, key=lambda row: (-float(row["computed_score"]), int(row["cand_position"])))

    return {
        "k": k_eff,
        "parent_obj": parent_obj,
        "topk_by_score": records,
        "topk_rank_original": [int(row["cand_position"]) for row in original_rank],
        "topk_rank_computed": [int(row["cand_position"]) for row in computed_rank],
    }


def run_anchor_dual_substitution_with_scip(
    context: SCIPBranchingContext,
    action_set: Sequence[int],
    candidate_scores: Sequence[float],
    cutoffbound: float,
    anchor_k: int = 8,
) -> Dict[str, Any]:
    candidate_lp_positions = np.asarray(action_set, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)

    if candidate_lp_positions.shape[0] != candidate_scores.shape[0]:
        raise ValueError(
            f"action_set and candidate_scores size mismatch "
            f"({candidate_lp_positions.shape[0]} vs {candidate_scores.shape[0]})"
        )
    n_candidates = int(candidate_lp_positions.shape[0])
    if n_candidates == 0:
        raise ValueError("No candidates available.")

    parent_primal = context.solve_parent_primal()
    if not parent_primal.success or parent_primal.objective_value is None:
        raise RuntimeError(f"Parent LP solve failed: {parent_primal.status} ({parent_primal.message})")
    parent_obj = float(parent_primal.objective_value)

    parent_dual = context.solve_parent_dual()
    if not parent_dual.success or parent_dual.y is None:
        raise RuntimeError(f"Parent dual solve failed: {parent_dual.status} ({parent_dual.message})")

    k_eff = int(min(max(int(anchor_k), 0), n_candidates))
    anchor_positions = np.argsort(candidate_scores)[-k_eff:][::-1]

    child_zero_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)
    child_one_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)
    dual_pool_records: List[Dict[str, Any]] = []
    anchor_child_records: List[Dict[str, Any]] = []

    def _add_dual_source(source_name: str, src_pos: Optional[int], dual: DualSolveResult) -> None:
        if dual.y is None or dual.alpha is None or dual.beta is None:
            return

        base_obj = evaluate_explicit_dual_on_child(
            context.lp,
            dual.y,
            dual.alpha,
            dual.beta,
            bound_overrides=None,
        )

        down_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)
        up_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)

        for idx, var_idx in enumerate(candidate_lp_positions.tolist()):
            branch = context.create_branch_bounds(int(var_idx))
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

        dual_pool_records.append(
            {
                "source": source_name,
                "source_candidate_position": None if src_pos is None else int(src_pos),
                "base_obj_eval": base_obj,
                "down_eval": down_eval,
                "up_eval": up_eval,
            }
        )

    _add_dual_source("parent", None, parent_dual)

    for pos in anchor_positions.tolist():
        var_idx = int(candidate_lp_positions[pos])
        branch = context.create_branch_bounds(var_idx)

        down_primal = solve_lp_with_scip(context.lp, bound_overrides=branch.down_overrides)
        up_primal = solve_lp_with_scip(context.lp, bound_overrides=branch.up_overrides)

        down_dual = None
        down_dual_eval_self = None
        down_dual_eval_gap = None
        if down_primal.success and down_primal.objective_value is not None:
            child_zero_obj_est[pos] = max(child_zero_obj_est[pos], float(down_primal.objective_value))
            down_dual = solve_explicit_dual_with_scip(context.lp, bound_overrides=branch.down_overrides)
            if down_dual.success and down_dual.y is not None:
                down_dual_eval_self = evaluate_explicit_dual_on_child(
                    context.lp,
                    down_dual.y,
                    down_dual.alpha,
                    down_dual.beta,
                    bound_overrides=branch.down_overrides,
                )
                down_dual_eval_gap = abs(float(down_primal.objective_value) - float(down_dual_eval_self))
                _add_dual_source("anchor_down", int(pos), down_dual)

        up_dual = None
        up_dual_eval_self = None
        up_dual_eval_gap = None
        if up_primal.success and up_primal.objective_value is not None:
            child_one_obj_est[pos] = max(child_one_obj_est[pos], float(up_primal.objective_value))
            up_dual = solve_explicit_dual_with_scip(context.lp, bound_overrides=branch.up_overrides)
            if up_dual.success and up_dual.y is not None:
                up_dual_eval_self = evaluate_explicit_dual_on_child(
                    context.lp,
                    up_dual.y,
                    up_dual.alpha,
                    up_dual.beta,
                    bound_overrides=branch.up_overrides,
                )
                up_dual_eval_gap = abs(float(up_primal.objective_value) - float(up_dual_eval_self))
                _add_dual_source("anchor_up", int(pos), up_dual)

        anchor_child_records.append(
            {
                "cand_position": int(pos),
                "cand_lp_pos": var_idx,
                "lpsol": branch.lpsol,
                "down_status": down_primal.status,
                "up_status": up_primal.status,
                "down_obj": float("inf") if down_primal.objective_value is None else float(down_primal.objective_value),
                "up_obj": float("inf") if up_primal.objective_value is None else float(up_primal.objective_value),
                "down_dual_status": None if down_dual is None else down_dual.status,
                "up_dual_status": None if up_dual is None else up_dual.status,
                "down_dual_eval_self": down_dual_eval_self,
                "down_dual_eval_gap": down_dual_eval_gap,
                "up_dual_eval_self": up_dual_eval_self,
                "up_dual_eval_gap": up_dual_eval_gap,
            }
        )

    pseudo_scores = np.array(
        [
            compute_strong_branch_score(parent_obj, child_one_obj_est[i], child_zero_obj_est[i], cutoffbound)
            for i in range(n_candidates)
        ],
        dtype=np.float64,
    )
    selected_pos = int(np.nanargmax(pseudo_scores))

    return {
        "parent_obj": parent_obj,
        "anchor_positions": anchor_positions.astype(np.int64),
        "candidate_lp_positions": candidate_lp_positions,
        "candidate_scores": candidate_scores,
        "dual_pool_records": dual_pool_records,
        "anchor_child_records": anchor_child_records,
        "child_zero_obj_est": child_zero_obj_est,
        "child_one_obj_est": child_one_obj_est,
        "pseudo_scores": pseudo_scores,
        "selected_position": selected_pos,
        "selected_lp_pos": int(candidate_lp_positions[selected_pos]),
    }
