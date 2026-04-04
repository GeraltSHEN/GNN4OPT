"""SCIP-based utilities for anchor strong branching."""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyscipopt as scip
import scs
import scipy.sparse as sp


SampleState = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
Bound = Tuple[Optional[float], Optional[float]]
BoundOverrides = Dict[int, Bound]
DUAL_OPTION_DEFAULT = 1
UNIVERSAL_CUTOFFBOUND_DEFAULT = 1e6


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


def resolve_effective_cutoffbound(
    cutoffbound: Optional[float],
    dual_option: int = DUAL_OPTION_DEFAULT,
    universal_cutoffbound: float = UNIVERSAL_CUTOFFBOUND_DEFAULT,
) -> Optional[float]:
    option = int(dual_option)
    if option in (1, 3):
        return cutoffbound
    if option in (2, 4):
        return float(universal_cutoffbound)
    raise ValueError(f"Unsupported dual_option '{dual_option}'. Use one of: 1, 2, 3, 4.")


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



def solve_explicit_dual_with_scs(
    lp: LPComponents,
    bound_overrides: Optional[BoundOverrides] = None,
    obj_val: Optional[float] = None,
    dual_option: int = DUAL_OPTION_DEFAULT,
    universal_cutoffbound: float = UNIVERSAL_CUTOFFBOUND_DEFAULT,
    time_limit_sec: Optional[float] = None,
    display_verblevel: int = 0,
) -> DualSolveResult:
    """Solve dual option 3/4 via SCS QP with fixed objective target and min L2-norm solution."""
    option = int(dual_option)
    if option not in (3, 4):
        raise ValueError(f"Unsupported dual_option '{dual_option}'. Use one of: 1, 2, 3, 4.")
    if lp.objective_sense.startswith("max"):
        raise ValueError("Explicit dual solver currently supports minimization problems only.")

    c = np.asarray(lp.objective_coefficients, dtype=np.float64)
    A_ub = lp.A_ub.tocsr()
    b_ub = np.asarray(lp.b_ub, dtype=np.float64)

    A = (-A_ub).tocsr()
    rhs = -b_ub

    parent_lbs, parent_ubs = _bounds_to_arrays(lp.bounds, int(c.shape[0]))
    lbs, ubs, bounds_consistent = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not bounds_consistent:
        return DualSolveResult(
            success=False,
            status="unbounded",
            message="primal infeasible after applying bound overrides (inconsistent bounds)",
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

    if obj_val is None or not np.isfinite(float(obj_val)):
        return DualSolveResult(
            success=False,
            status="missing_objective_target",
            message="SCS option 3/4 requires a finite obj_val from option 1/2 solve.",
            objective_value=float("nan"),
            y=None,
            alpha=None,
            beta=None,
            raw_result=None,
        )

    m_rows, n_vars = A.shape
    yab_dim = m_rows + 2 * n_vars
    objective_target = float(obj_val) - float(lp.objective_offset)

    eq_block = sp.hstack(
        [
            A.transpose().tocsr(),
            -sp.eye(n_vars, dtype=np.float64, format="csr"),
            sp.eye(n_vars, dtype=np.float64, format="csr"),
        ],
        format="csr",
    )

    target_coeffs = np.zeros(yab_dim, dtype=np.float64)
    target_coeffs[:m_rows] = rhs
    target_coeffs[m_rows : m_rows + n_vars] = -ubs
    target_coeffs[m_rows + n_vars : m_rows + (2 * n_vars)] = lbs

    eq_block = sp.vstack([eq_block, sp.csr_matrix(target_coeffs.reshape(1, -1))], format="csr")
    b_eq = np.concatenate([np.asarray(c, dtype=np.float64), np.asarray([objective_target], dtype=np.float64)])

    nonneg_block = -sp.eye(yab_dim, dtype=np.float64, format="csr")
    A_scs = sp.vstack([eq_block, nonneg_block], format="csc")
    b_scs = np.concatenate([b_eq, np.zeros(yab_dim, dtype=np.float64)]).astype(np.float64, copy=False)
    c_scs = np.zeros(yab_dim, dtype=np.float64)
    P_scs = sp.eye(yab_dim, dtype=np.float64, format="csc")
    data = {"P": P_scs, "A": A_scs, "b": b_scs, "c": c_scs}
    cone = {"z": int(n_vars + 1), "l": int(yab_dim)}

    settings: Dict[str, Any] = {"verbose": bool(int(display_verblevel) > 0)}
    if time_limit_sec is not None and np.isfinite(float(time_limit_sec)) and float(time_limit_sec) > 0.0:
        settings["time_limit_secs"] = float(time_limit_sec)

    try:
        raw_result = scs.solve(data, cone, **settings)
    except Exception as exc:
        return DualSolveResult(
            success=False,
            status="solver_error",
            message=f"scs exception: {exc}",
            objective_value=float("nan"),
            y=None,
            alpha=None,
            beta=None,
            raw_result=None,
        )
    info = raw_result.get("info", {}) if isinstance(raw_result, dict) else {}
    status_raw = str(info.get("status", "")).lower()
    status_val = info.get("status_val")
    if status_val in (1, 2) and isinstance(raw_result, dict) and raw_result.get("x") is not None:
        x_scs = np.asarray(raw_result["x"], dtype=np.float64).reshape(-1)
        if x_scs.shape[0] < yab_dim:
            return DualSolveResult(
                success=False,
                status="invalid_solution_dimension",
                message=f"scs returned x with dim={x_scs.shape[0]}, expected at least {yab_dim}",
                objective_value=float("nan"),
                y=None,
                alpha=None,
                beta=None,
                raw_result=raw_result,
            )
        y_val = x_scs[:m_rows].copy()
        alpha_val = x_scs[m_rows : m_rows + n_vars].copy()
        beta_val = x_scs[m_rows + n_vars : m_rows + (2 * n_vars)].copy()
        obj_value = float(np.dot(rhs, y_val) - np.dot(ubs, alpha_val) + np.dot(lbs, beta_val) + lp.objective_offset)

        target_obj = float(obj_val)
        obj_gap = abs(float(obj_value) - target_obj)
        res_pri = abs(float(info.get("res_pri", 0.0) or 0.0))
        res_dual = abs(float(info.get("res_dual", 0.0) or 0.0))
        gap = abs(float(info.get("gap", 0.0) or 0.0))
        obj_tol = max(
            1e-5,
            10.0 * gap,
            10.0 * res_pri * max(1.0, abs(target_obj)),
            10.0 * res_dual * max(1.0, abs(target_obj)),
        )
        min_dual = float(min(np.min(y_val), np.min(alpha_val), np.min(beta_val)))
        nonneg_tol = max(1e-6, 10.0 * res_pri)
        if min_dual < -nonneg_tol or obj_gap > obj_tol:
            return DualSolveResult(
                success=False,
                status="numerical_mismatch",
                message=(
                    f"scs solution inconsistent with objective target: "
                    f"obj_gap={obj_gap:.3e}, obj_tol={obj_tol:.3e}, "
                    f"min_dual={min_dual:.3e}, nonneg_tol={nonneg_tol:.3e}, status={status_raw}"
                ),
                objective_value=float("nan"),
                y=None,
                alpha=None,
                beta=None,
                raw_result=raw_result,
            )
        return DualSolveResult(
            success=True,
            status="optimal" if status_val == 1 else "optimal_inaccurate",
            message=status_raw,
            objective_value=obj_value,
            y=y_val,
            alpha=alpha_val,
            beta=beta_val,
            raw_result=raw_result,
        )

    if "infeasible" in status_raw or "unbounded" in status_raw:
        mapped_status = "unbounded"
        obj_value = float("inf")
    else:
        mapped_status = status_raw or "unknown"
        obj_value = float("nan")
    return DualSolveResult(
        success=False,
        status=mapped_status,
        message=status_raw or "unknown",
        objective_value=obj_value,
        y=None,
        alpha=None,
        beta=None,
        raw_result=raw_result,
    )

def solve_explicit_dual_with_scip(
    lp: LPComponents,
    bound_overrides: Optional[BoundOverrides] = None,
    cutoffbound: Optional[float] = None,
    dual_option: int = DUAL_OPTION_DEFAULT,
    universal_cutoffbound: float = UNIVERSAL_CUTOFFBOUND_DEFAULT,
    time_limit_sec: Optional[float] = None,
    display_verblevel: int = 0,
) -> DualSolveResult:
    option = int(dual_option)
    if option in (3, 4):
        base_option = 1 if option == 3 else 2
        base_result = solve_explicit_dual_with_scip(
            lp=lp,
            bound_overrides=bound_overrides,
            cutoffbound=cutoffbound,
            dual_option=base_option,
            universal_cutoffbound=universal_cutoffbound,
            time_limit_sec=time_limit_sec,
            display_verblevel=display_verblevel,
        )
        if not base_result.success:
            return base_result
        return solve_explicit_dual_with_scs(
            lp=lp,
            bound_overrides=bound_overrides,
            obj_val=float(base_result.objective_value),
            dual_option=option,
            universal_cutoffbound=universal_cutoffbound,
            time_limit_sec=time_limit_sec,
            display_verblevel=display_verblevel,
        )
    if option not in (1, 2):
        raise ValueError(f"Unsupported dual_option '{dual_option}'. Use one of: 1, 2, 3, 4.")

    if lp.objective_sense.startswith("max"):
        raise ValueError("Explicit dual solver currently supports minimization problems only.")

    c = np.asarray(lp.objective_coefficients, dtype=np.float64)
    A_ub = lp.A_ub.tocsr()
    b_ub = np.asarray(lp.b_ub, dtype=np.float64)

    A = (-A_ub).tocsr()
    rhs = -b_ub

    parent_lbs, parent_ubs = _bounds_to_arrays(lp.bounds, int(c.shape[0]))
    lbs, ubs, bounds_consistent = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not bounds_consistent:
        return DualSolveResult(
            success=False,
            status="unbounded",
            message="primal infeasible after applying bound overrides (inconsistent bounds)",
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
    model.setIntParam("presolving/maxrounds", 0)
    model.setIntParam("presolving/maxrestarts", 0)
    if time_limit_sec is not None and np.isfinite(float(time_limit_sec)) and float(time_limit_sec) > 0.0:
        model.setRealParam("limits/time", float(time_limit_sec))

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
    effective_cutoffbound = resolve_effective_cutoffbound(
        cutoffbound=cutoffbound,
        dual_option=int(dual_option),
        universal_cutoffbound=float(universal_cutoffbound),
    )
    if effective_cutoffbound is not None and np.isfinite(float(effective_cutoffbound)):
        cutoff_rhs = float(effective_cutoffbound) - float(lp.objective_offset)
        model.addCons(obj_expr <= float(cutoff_rhs), name="dual_obj_cutoff_cap")

    model.setObjective(obj_expr, sense="maximize")
    model.optimize()

    status = str(model.getStatus()).lower()
    message = status
    if status != "optimal":
        if status in {"unbounded", "inforunbd", "infeasible"}:
            mapped_status = "unbounded"
            obj_value = float("inf")
        else:
            mapped_status = status
            obj_value = float("nan")
        return DualSolveResult(
            success=False,
            status=mapped_status,
            message=message,
            objective_value=obj_value,
            y=None,
            alpha=None,
            beta=None,
            raw_result=model,
        )

    sol = model.getBestSol()
    y_val = np.asarray([model.getSolVal(sol, var) for var in y], dtype=np.float64)
    alpha_val = np.asarray([model.getSolVal(sol, var) for var in alpha], dtype=np.float64)
    beta_val = np.asarray([model.getSolVal(sol, var) for var in beta], dtype=np.float64)
    obj_value = float(np.dot(rhs, y_val) - np.dot(ubs, alpha_val) + np.dot(lbs, beta_val) + lp.objective_offset)
    return DualSolveResult(
        success=True,
        status="optimal",
        message=message,
        objective_value=obj_value,
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
    return float(np.dot(rhs, y) - np.dot(ubs, alpha) + np.dot(lbs, beta) + lp.objective_offset)


class SCIPBranchingContext:
    """LP/dual solve context for one sample state."""

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

    def solve_parent_dual(
        self,
        cutoffbound: Optional[float] = None,
        dual_option: int = DUAL_OPTION_DEFAULT,
        universal_cutoffbound: float = UNIVERSAL_CUTOFFBOUND_DEFAULT,
        time_limit_sec: Optional[float] = None,
    ) -> DualSolveResult:
        if (
            cutoffbound is not None
            or int(dual_option) != int(DUAL_OPTION_DEFAULT)
            or float(universal_cutoffbound) != float(UNIVERSAL_CUTOFFBOUND_DEFAULT)
        ):
            return solve_explicit_dual_with_scip(
                self.lp,
                cutoffbound=cutoffbound,
                dual_option=dual_option,
                universal_cutoffbound=universal_cutoffbound,
                time_limit_sec=time_limit_sec,
            )
        if self._parent_dual is None:
            self._parent_dual = solve_explicit_dual_with_scip(
                self.lp,
                dual_option=dual_option,
                universal_cutoffbound=universal_cutoffbound,
                time_limit_sec=time_limit_sec,
            )
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

    def solve_child_dual(
        self,
        var_idx: int,
        direction: str,
        cutoffbound: Optional[float] = None,
        dual_option: int = DUAL_OPTION_DEFAULT,
        universal_cutoffbound: float = UNIVERSAL_CUTOFFBOUND_DEFAULT,
        time_limit_sec: Optional[float] = None,
    ) -> DualSolveResult:
        bounds = self.create_branch_bounds(int(var_idx))
        if direction == "down":
            return solve_explicit_dual_with_scip(
                self.lp,
                bound_overrides=bounds.down_overrides,
                cutoffbound=cutoffbound,
                dual_option=dual_option,
                universal_cutoffbound=universal_cutoffbound,
                time_limit_sec=time_limit_sec,
            )
        if direction == "up":
            return solve_explicit_dual_with_scip(
                self.lp,
                bound_overrides=bounds.up_overrides,
                cutoffbound=cutoffbound,
                dual_option=dual_option,
                universal_cutoffbound=universal_cutoffbound,
                time_limit_sec=time_limit_sec,
            )
        raise ValueError(f"Unknown direction '{direction}', expected 'down' or 'up'")
