import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp


SampleState = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]


def load_sample(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def unpack_sample_data(data: Sequence[Any]) -> Tuple[SampleState, int, Sequence[int], Sequence[float], float]:
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected sample['data'] to be list/tuple, got {type(data)}")
    if len(data) < 5:
        raise ValueError(f"Expected at least 5 entries in sample['data'], got {len(data)}")
    sample_state = data[0]
    sample_action = data[2]
    sample_action_set = data[3]
    sample_scores = data[4]
    cutoffbound = float(data[5]) if len(data) >= 6 and data[5] is not None else float("inf")
    return sample_state, sample_action, sample_action_set, sample_scores, cutoffbound


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
    """
    Reconstruct unnormalized SCIP features from the normalized state in sample['data'][0].

    Requires metadata emitted by legacy_code_generator.utilities.extract_state.
    """
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
        constraint_values[:, constraint_feature_indices["dualsol_val_normalized"]] *= (constraint_row_norms * obj_norm)
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


def _build_bounds(variable_dict: Dict[str, Any], variable_values: np.ndarray, variable_feature_indices: Dict[str, int],
                  assume_binary_bounds: bool) -> List[Tuple[Union[float, None], Union[float, None]]]:
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

    if assume_binary_bounds and "type_0" in variable_feature_indices:
        type_binary = variable_values[:, variable_feature_indices["type_0"]] > 0.5
        missing_lb = type_binary & np.isnan(lbs)
        missing_ub = type_binary & np.isnan(ubs)
        lbs[missing_lb] = 0.0
        ubs[missing_ub] = 1.0

    bounds: List[Tuple[Union[float, None], Union[float, None]]] = []
    for lb, ub in zip(lbs, ubs):
        lbv = None if np.isnan(lb) or np.isinf(lb) else float(lb)
        ubv = None if np.isnan(ub) or np.isinf(ub) else float(ub)
        bounds.append((lbv, ubv))
    return bounds


def extract_lp_components(unnormalized_state: SampleState, assume_binary_bounds: bool = True) -> Dict[str, Any]:
    """
    Convert a recovered state into LP data in canonical form:
    minimize/maximize c^T x + offset, subject to A_ub x <= b_ub and variable bounds.

    The row splitting/sign convention matches 02_generate_samples.py.
    """
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

    bounds = _build_bounds(variable_dict, variable_values, variable_feature_indices, assume_binary_bounds)

    return {
        "A_ub": A_ub,
        "b_ub": b_ub,
        "objective_coefficients": c,
        "lp_solution": x_lp,
        "reduced_costs": reduced_costs,
        "row_duals": row_duals,
        "objective_sense": objective_sense,
        "objective_offset": objective_offset,
        "bounds": bounds,
        "variable_values": variable_values,
        "variable_feature_indices": variable_feature_indices,
    }


def compute_primal_objective(lp_components: Dict[str, Any], x: Union[np.ndarray, None] = None) -> float:
    coeffs = np.asarray(lp_components["objective_coefficients"], dtype=np.float64)
    if x is None:
        x = np.asarray(lp_components["lp_solution"], dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
    return float(np.dot(coeffs, x) + float(lp_components.get("objective_offset", 0.0)))


def compute_dual_reduced_cost_objective(lp_components: Dict[str, Any], include_offset: bool = False) -> float:
    row_duals = lp_components.get("row_duals")
    reduced_costs = lp_components.get("reduced_costs")
    if row_duals is None or reduced_costs is None:
        raise ValueError("row_duals and reduced_costs are required to compute dual+rc objective")

    row_duals = np.asarray(row_duals, dtype=np.float64)
    reduced_costs = np.asarray(reduced_costs, dtype=np.float64)
    row_bias = np.asarray(lp_components["b_ub"], dtype=np.float64)
    x_lp = np.asarray(lp_components["lp_solution"], dtype=np.float64)

    value = float(np.dot(row_duals, row_bias) + np.dot(reduced_costs, x_lp))
    if include_offset:
        value += float(lp_components.get("objective_offset", 0.0))
    return value


def solve_reconstructed_lp_with_scip(
    lp_components: Dict[str, Any],
    as_mip: bool = False,
    display_verblevel: int = 0,
    bound_overrides: Union[None, Dict[int, Tuple[Union[float, None], Union[float, None]]]] = None,
) -> Dict[str, Any]:
    """
    Solve reconstructed LP/MIP directly with SCIP defaults via PySCIPOpt.

    Parameters
    ----------
    lp_components : dict
        Output of `extract_lp_components`.
    as_mip : bool
        If True, recover variable integrality from one-hot variable type features
        (`type_0`, `type_1`, `type_2`, `type_3`). If False, solve LP relaxation.
    display_verblevel : int
        SCIP display verbosity (`display/verblevel`).
    """
    try:
        import pyscipopt as scip
        from pyscipopt import quicksum
    except Exception as exc:
        raise RuntimeError(
            "PySCIPOpt is required for SCIP-based reconstruction solve."
        ) from exc

    sense = str(lp_components.get("objective_sense", "minimize")).lower()
    maximize = sense.startswith("max")
    c = np.asarray(lp_components["objective_coefficients"], dtype=np.float64)
    b_ub = np.asarray(lp_components["b_ub"], dtype=np.float64)
    A_ub = lp_components["A_ub"].tocsr()
    bounds = lp_components.get("bounds")
    objective_offset = float(lp_components.get("objective_offset", 0.0))

    variable_values = np.asarray(lp_components.get("variable_values"), dtype=np.float64)
    variable_feature_indices = lp_components.get("variable_feature_indices", {})

    model = scip.Model()
    model.setIntParam("display/verblevel", int(display_verblevel))

    vars_ = []
    inf = model.infinity()
    for j in range(c.shape[0]):
        lb, ub = bounds[j] if bounds is not None else (None, None)
        if bound_overrides is not None and j in bound_overrides:
            lb_override, ub_override = bound_overrides[j]
            if lb_override is not None:
                lb = float(lb_override) if lb is None else max(float(lb), float(lb_override))
            if ub_override is not None:
                ub = float(ub_override) if ub is None else min(float(ub), float(ub_override))

        if lb is not None and ub is not None and float(lb) > float(ub):
            return {
                "success": False,
                "status": "infeasible",
                "message": "infeasible after applying bound overrides",
                "x": None,
                "objective_value": None,
                "raw_result": None,
            }

        lbv = -inf if lb is None else float(lb)
        ubv = +inf if ub is None else float(ub)

        vtype = "C"
        if as_mip and variable_values.size > 0:
            is_bin = "type_0" in variable_feature_indices and variable_values[j, variable_feature_indices["type_0"]] > 0.5
            is_int = "type_1" in variable_feature_indices and variable_values[j, variable_feature_indices["type_1"]] > 0.5
            is_impl = "type_2" in variable_feature_indices and variable_values[j, variable_feature_indices["type_2"]] > 0.5
            if is_bin:
                vtype = "B"
            elif is_int or is_impl:
                vtype = "I"

        var = model.addVar(
            name=f"x_{j}",
            vtype=vtype,
            lb=lbv,
            ub=ubv,
        )
        vars_.append(var)

    obj_expr = quicksum(float(c[j]) * vars_[j] for j in range(c.shape[0]))
    model.setObjective(obj_expr, sense="maximize" if maximize else "minimize")

    cons_ = []
    for i in range(A_ub.shape[0]):
        row_start, row_end = A_ub.indptr[i], A_ub.indptr[i + 1]
        idxs = A_ub.indices[row_start:row_end]
        vals = A_ub.data[row_start:row_end]
        cons_expr = quicksum(float(v) * vars_[int(j)] for j, v in zip(idxs, vals))
        cons_.append(model.addCons(cons_expr <= float(b_ub[i]), name=f"c_{i}"))

    model.optimize()
    status = str(model.getStatus()).lower()

    out = {
        "success": status == "optimal",
        "status": status,
        "message": status,
        "x": None,
        "objective_value": None,
        "row_duals": None,
        "row_bias": None,
        "reduced_costs": None,
        "dual_obj_row_rc": None,
        "raw_result": model,
    }

    if status == "optimal":
        sol = model.getBestSol()
        x = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
        obj = float(model.getObjVal()) + objective_offset
        out["x"] = x
        out["objective_value"] = obj

        row_bias = np.asarray(b_ub, dtype=np.float64).copy()
        row_duals = np.asarray([float(model.getDualsolLinear(cons)) for cons in cons_], dtype=np.float64)
        reduced_costs = np.asarray([float(model.getVarRedcost(var)) for var in vars_], dtype=np.float64)
        dual_obj_row_rc = None

        out["row_duals"] = row_duals
        out["row_bias"] = row_bias
        out["reduced_costs"] = reduced_costs
        out["dual_obj_row_rc"] = dual_obj_row_rc

    return out


def compute_strong_branch_score(parent_obj: float, child_one_obj: float, child_zero_obj: float, cutoffbound: float) -> float:
    child_one_obj_capped = min(float(child_one_obj), float(cutoffbound))
    child_zero_obj_capped = min(float(child_zero_obj), float(cutoffbound))
    gain_one = max(child_one_obj_capped - float(parent_obj), 1e-9)
    gain_zero = max(child_zero_obj_capped - float(parent_obj), 1e-9)
    return float(gain_one * gain_zero)


def _bounds_to_arrays(bounds: Sequence[Tuple[Union[float, None], Union[float, None]]], n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
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
    bound_overrides: Union[None, Dict[int, Tuple[Union[float, None], Union[float, None]]]] = None,
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


def solve_lp_with_dual_signature_scip(
    lp_components: Dict[str, Any],
    bound_overrides: Union[None, Dict[int, Tuple[Union[float, None], Union[float, None]]]] = None,
    display_verblevel: int = 0,
) -> Dict[str, Any]:
    """
    Solve LP with SCIP and return an explicit dual signature:
    - main row duals (A_ub x <= b_ub)
    - lower-bound row duals (-x_j <= -lb_j)
    - upper-bound row duals ( x_j <=  ub_j)
    """
    try:
        import pyscipopt as scip
        from pyscipopt import quicksum
    except Exception as exc:
        raise RuntimeError("PySCIPOpt is required for SCIP-based LP solve.") from exc

    c = np.asarray(lp_components["objective_coefficients"], dtype=np.float64)
    b_ub = np.asarray(lp_components["b_ub"], dtype=np.float64)
    A_ub = lp_components["A_ub"].tocsr()
    objective_offset = float(lp_components.get("objective_offset", 0.0))
    bounds = lp_components.get("bounds")
    if bounds is None:
        raise ValueError("lp_components['bounds'] is required for dual-signature solve.")

    parent_lbs, parent_ubs = _bounds_to_arrays(bounds, int(c.shape[0]))
    lbs, ubs, feasible = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not feasible:
        return {
            "success": False,
            "status": "infeasible",
            "message": "infeasible after applying bound overrides",
            "x": None,
            "objective_value": None,
            "dual_signature": None,
            "raw_result": None,
        }

    model = scip.Model()
    model.setIntParam("display/verblevel", int(display_verblevel))

    vars_ = [model.addVar(name=f"x_{j}", vtype="C") for j in range(c.shape[0])]
    obj_expr = quicksum(float(c[j]) * vars_[j] for j in range(c.shape[0]))
    model.setObjective(obj_expr, sense="minimize")

    main_cons = []
    for i in range(A_ub.shape[0]):
        row_start, row_end = A_ub.indptr[i], A_ub.indptr[i + 1]
        idxs = A_ub.indices[row_start:row_end]
        vals = A_ub.data[row_start:row_end]
        cons_expr = quicksum(float(v) * vars_[int(j)] for j, v in zip(idxs, vals))
        main_cons.append(model.addCons(cons_expr <= float(b_ub[i]), name=f"main_{i}"))

    lb_cons = [None] * c.shape[0]
    ub_cons = [None] * c.shape[0]
    for j in range(c.shape[0]):
        if np.isfinite(lbs[j]):
            lb_cons[j] = model.addCons(-vars_[j] <= float(-lbs[j]), name=f"lb_{j}")
        if np.isfinite(ubs[j]):
            ub_cons[j] = model.addCons(vars_[j] <= float(ubs[j]), name=f"ub_{j}")

    model.optimize()
    status = str(model.getStatus()).lower()
    out = {
        "success": status == "optimal",
        "status": status,
        "message": status,
        "x": None,
        "objective_value": None,
        "dual_signature": None,
        "raw_result": model,
    }
    if status != "optimal":
        return out

    sol = model.getBestSol()
    x = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
    obj = float(model.getObjVal()) + objective_offset

    y_main = np.asarray([float(model.getDualsolLinear(cons)) for cons in main_cons], dtype=np.float64)
    y_lb = np.zeros(c.shape[0], dtype=np.float64)
    y_ub = np.zeros(c.shape[0], dtype=np.float64)
    for j in range(c.shape[0]):
        if lb_cons[j] is not None:
            y_lb[j] = float(model.getDualsolLinear(lb_cons[j]))
        if ub_cons[j] is not None:
            y_ub[j] = float(model.getDualsolLinear(ub_cons[j]))

    dual_signature = {
        "y_main": y_main,
        "y_lb": y_lb,
        "y_ub": y_ub,
        "b_main": b_ub.copy(),
        "lb_rhs": -lbs.copy(),  # from -x <= -lb
        "ub_rhs": ubs.copy(),   # from  x <=  ub
    }

    out["x"] = x
    out["objective_value"] = obj
    out["dual_signature"] = dual_signature
    return out


def evaluate_objective_from_dual_signature(
    dual_signature: Dict[str, np.ndarray],
    objective_offset: float = 0.0,
    lb_rhs_override: Union[None, np.ndarray] = None,
    ub_rhs_override: Union[None, np.ndarray] = None,
) -> float:
    y_main = np.asarray(dual_signature["y_main"], dtype=np.float64)
    y_lb = np.asarray(dual_signature["y_lb"], dtype=np.float64)
    y_ub = np.asarray(dual_signature["y_ub"], dtype=np.float64)
    b_main = np.asarray(dual_signature["b_main"], dtype=np.float64)
    lb_rhs = np.asarray(dual_signature["lb_rhs"] if lb_rhs_override is None else lb_rhs_override, dtype=np.float64)
    ub_rhs = np.asarray(dual_signature["ub_rhs"] if ub_rhs_override is None else ub_rhs_override, dtype=np.float64)

    raw = float(np.dot(y_main, b_main) + np.dot(y_lb, lb_rhs) + np.dot(y_ub, ub_rhs))
    return float(-raw + float(objective_offset))


def _evaluate_dual_signature_on_all_candidates(
    dual_signature: Dict[str, np.ndarray],
    objective_offset: float,
    parent_lbs: np.ndarray,
    parent_ubs: np.ndarray,
    parent_sol: np.ndarray,
    candidate_lp_positions: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    lb_rhs_parent = -parent_lbs.copy()
    ub_rhs_parent = parent_ubs.copy()
    base_obj = evaluate_objective_from_dual_signature(
        dual_signature=dual_signature,
        objective_offset=objective_offset,
        lb_rhs_override=lb_rhs_parent,
        ub_rhs_override=ub_rhs_parent,
    )

    down_est = np.full(candidate_lp_positions.shape[0], float("-inf"), dtype=np.float64)
    up_est = np.full(candidate_lp_positions.shape[0], float("-inf"), dtype=np.float64)

    for idx, var_idx in enumerate(candidate_lp_positions.tolist()):
        lpsol = float(parent_sol[var_idx])
        down_ub = min(float(np.floor(lpsol)), float(parent_ubs[var_idx]))
        up_lb = max(float(np.ceil(lpsol)), float(parent_lbs[var_idx]))

        ub_rhs_down = ub_rhs_parent.copy()
        ub_rhs_down[var_idx] = down_ub
        down_est[idx] = evaluate_objective_from_dual_signature(
            dual_signature=dual_signature,
            objective_offset=objective_offset,
            lb_rhs_override=lb_rhs_parent,
            ub_rhs_override=ub_rhs_down,
        )

        lb_rhs_up = lb_rhs_parent.copy()
        lb_rhs_up[var_idx] = -up_lb
        up_est[idx] = evaluate_objective_from_dual_signature(
            dual_signature=dual_signature,
            objective_offset=objective_offset,
            lb_rhs_override=lb_rhs_up,
            ub_rhs_override=ub_rhs_parent,
        )

    return base_obj, down_est, up_est


def solve_explicit_dual_with_scip(
    lp_components: Dict[str, Any],
    bound_overrides: Union[None, Dict[int, Tuple[Union[float, None], Union[float, None]]]] = None,
    display_verblevel: int = 0,
) -> Dict[str, Any]:
    """
    Solve the explicit dual (heuristics/utils.py form) with SCIP:
        max rhs^T y - ub^T alpha + lb^T beta
        s.t. A^T y - alpha + beta = c
             y, alpha, beta >= 0
    where A = -A_ub and rhs = -b_ub.
    """
    try:
        import pyscipopt as scip
        from pyscipopt import quicksum
    except Exception as exc:
        raise RuntimeError("PySCIPOpt is required for SCIP-based dual solve.") from exc

    c = np.asarray(lp_components["objective_coefficients"], dtype=np.float64)
    A_ub = lp_components["A_ub"].tocsr()
    b_ub = np.asarray(lp_components["b_ub"], dtype=np.float64)
    bounds = lp_components.get("bounds")
    if bounds is None:
        raise ValueError("lp_components['bounds'] is required for dual solve.")

    A = (-A_ub).tocsr()
    rhs = -b_ub
    parent_lbs, parent_ubs = _bounds_to_arrays(bounds, int(c.shape[0]))
    lbs, ubs, feasible = _apply_bound_overrides(parent_lbs, parent_ubs, bound_overrides)
    if not feasible:
        return {
            "success": False,
            "status": "infeasible",
            "message": "infeasible after applying bound overrides",
            "objective_value": float("inf"),
            "y": None,
            "alpha": None,
            "beta": None,
            "raw_result": None,
        }

    if np.any(~np.isfinite(lbs)) or np.any(~np.isfinite(ubs)):
        return {
            "success": False,
            "status": "unsupported_infinite_bounds",
            "message": "Explicit dual solve currently expects finite variable bounds.",
            "objective_value": float("nan"),
            "y": None,
            "alpha": None,
            "beta": None,
            "raw_result": None,
        }

    m_rows, n_vars = A.shape
    model = scip.Model()
    model.setIntParam("display/verblevel", int(display_verblevel))

    y = [model.addVar(name=f"y_{i}", vtype="C", lb=0.0) for i in range(m_rows)]
    alpha = [model.addVar(name=f"alpha_{j}", vtype="C", lb=0.0) for j in range(n_vars)]
    beta = [model.addVar(name=f"beta_{j}", vtype="C", lb=0.0) for j in range(n_vars)]

    A_csc = A.tocsc()
    for j in range(n_vars):
        col_start, col_end = A_csc.indptr[j], A_csc.indptr[j + 1]
        row_ids = A_csc.indices[col_start:col_end]
        vals = A_csc.data[col_start:col_end]
        lhs = quicksum(float(v) * y[int(i)] for i, v in zip(row_ids, vals)) - alpha[j] + beta[j]
        model.addCons(lhs == float(c[j]), name=f"dual_eq_{j}")

    obj_expr = (
        quicksum(float(rhs[i]) * y[i] for i in range(m_rows))
        - quicksum(float(ubs[j]) * alpha[j] for j in range(n_vars))
        + quicksum(float(lbs[j]) * beta[j] for j in range(n_vars))
    )
    model.setObjective(obj_expr, sense="maximize")
    model.optimize()

    status = str(model.getStatus()).lower()
    out = {
        "success": status == "optimal",
        "status": status,
        "message": status,
        "objective_value": float("nan"),
        "y": None,
        "alpha": None,
        "beta": None,
        "raw_result": model,
    }
    if status == "optimal":
        sol = model.getBestSol()
        y_val = np.asarray([model.getSolVal(sol, v) for v in y], dtype=np.float64)
        alpha_val = np.asarray([model.getSolVal(sol, v) for v in alpha], dtype=np.float64)
        beta_val = np.asarray([model.getSolVal(sol, v) for v in beta], dtype=np.float64)
        out["y"] = y_val
        out["alpha"] = alpha_val
        out["beta"] = beta_val
        out["objective_value"] = float(model.getObjVal()) + float(lp_components.get("objective_offset", 0.0))
    elif status in {"unbounded", "inforunbd"}:
        out["objective_value"] = float("inf")
    return out


def evaluate_explicit_dual_on_child(
    lp_components: Dict[str, Any],
    y: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    bound_overrides: Union[None, Dict[int, Tuple[Union[float, None], Union[float, None]]]] = None,
) -> float:
    bounds = lp_components.get("bounds")
    if bounds is None:
        raise ValueError("lp_components['bounds'] is required.")
    c = np.asarray(lp_components["objective_coefficients"], dtype=np.float64)
    b_ub = np.asarray(lp_components["b_ub"], dtype=np.float64)
    rhs = -b_ub
    lbs, ubs, feasible = _apply_bound_overrides(*_bounds_to_arrays(bounds, int(c.shape[0])), bound_overrides)
    if not feasible:
        return float("inf")
    if np.any(~np.isfinite(lbs)) or np.any(~np.isfinite(ubs)):
        return float("nan")
    objective_offset = float(lp_components.get("objective_offset", 0.0))
    value = float(np.dot(rhs, y) - np.dot(ubs, alpha) + np.dot(lbs, beta) + objective_offset)
    return value


def run_anchor_dual_substitution_with_scip(
    lp_components: Dict[str, Any],
    action_set: Sequence[int],
    candidate_scores: Sequence[float],
    cutoffbound: float,
    anchor_k: int = 8,
    as_mip: bool = False,
) -> Dict[str, Any]:
    """
    SCIP-only anchor dual substitution using explicit dual variables (y, alpha, beta).
    """
    candidate_lp_positions = np.asarray(action_set, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if as_mip:
        raise ValueError("Anchor dual substitution expects LP solves; set as_mip=False.")
    if candidate_lp_positions.shape[0] != candidate_scores.shape[0]:
        raise ValueError(
            f"action_set and candidate_scores size mismatch ({candidate_lp_positions.shape[0]} vs {candidate_scores.shape[0]})"
        )
    n_candidates = int(candidate_lp_positions.shape[0])
    if n_candidates == 0:
        raise ValueError("No candidates available.")

    parent_res = solve_reconstructed_lp_with_scip(lp_components, as_mip=False)
    if not parent_res["success"]:
        raise RuntimeError(
            f"Failed to solve parent LP with SCIP: {parent_res['status']} ({parent_res['message']})"
        )
    parent_obj = float(parent_res["objective_value"])
    parent_sol = np.asarray(parent_res["x"], dtype=np.float64)

    parent_dual = solve_explicit_dual_with_scip(lp_components)
    if not parent_dual["success"] or parent_dual["y"] is None:
        raise RuntimeError(
            f"Failed to solve parent explicit dual with SCIP: {parent_dual['status']} ({parent_dual['message']})"
        )

    bounds = lp_components.get("bounds")
    if bounds is None:
        raise ValueError("lp_components['bounds'] is required for anchor dual substitution.")
    n_vars = int(np.asarray(lp_components["objective_coefficients"], dtype=np.float64).shape[0])
    parent_lbs, parent_ubs = _bounds_to_arrays(bounds, n_vars)

    k_eff = int(min(max(int(anchor_k), 0), n_candidates))
    anchor_positions = np.argsort(candidate_scores)[-k_eff:][::-1]

    child_zero_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)
    child_one_obj_est = np.full(n_candidates, float("-inf"), dtype=np.float64)
    dual_pool_records: List[Dict[str, Any]] = []
    anchor_child_records: List[Dict[str, Any]] = []

    def _add_dual_source(source_name: str, src_pos: Union[int, None], y: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
        base_obj = evaluate_explicit_dual_on_child(lp_components, y, alpha, beta, bound_overrides=None)
        down_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)
        up_eval = np.full(n_candidates, float("-inf"), dtype=np.float64)
        for idx, var_idx in enumerate(candidate_lp_positions.tolist()):
            lpsol = float(parent_sol[var_idx])
            down_ub = min(float(np.floor(lpsol)), float(parent_ubs[var_idx]))
            up_lb = max(float(np.ceil(lpsol)), float(parent_lbs[var_idx]))
            down_eval[idx] = evaluate_explicit_dual_on_child(
                lp_components, y, alpha, beta, bound_overrides={var_idx: (None, down_ub)}
            )
            up_eval[idx] = evaluate_explicit_dual_on_child(
                lp_components, y, alpha, beta, bound_overrides={var_idx: (up_lb, None)}
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

    _add_dual_source("parent", None, parent_dual["y"], parent_dual["alpha"], parent_dual["beta"])

    for pos in anchor_positions.tolist():
        var_idx = int(candidate_lp_positions[pos])
        lpsol = float(parent_sol[var_idx])
        lb_local = float(parent_lbs[var_idx])
        ub_local = float(parent_ubs[var_idx])
        down_ub = min(float(np.floor(lpsol)), ub_local)
        up_lb = max(float(np.ceil(lpsol)), lb_local)

        down_res = solve_reconstructed_lp_with_scip(
            lp_components,
            as_mip=False,
            bound_overrides={var_idx: (None, down_ub)},
        )
        up_res = solve_reconstructed_lp_with_scip(
            lp_components,
            as_mip=False,
            bound_overrides={var_idx: (up_lb, None)},
        )

        down_dual_eval_self = None
        down_dual_eval_gap = None
        down_dual = None
        if down_res["success"]:
            down_dual = solve_explicit_dual_with_scip(lp_components, bound_overrides={var_idx: (None, down_ub)})
            if down_dual["success"] and down_dual["y"] is not None:
                down_dual_eval_self = evaluate_explicit_dual_on_child(
                    lp_components,
                    down_dual["y"],
                    down_dual["alpha"],
                    down_dual["beta"],
                    bound_overrides={var_idx: (None, down_ub)},
                )
                down_dual_eval_gap = abs(float(down_res["objective_value"]) - float(down_dual_eval_self))

        up_dual_eval_self = None
        up_dual_eval_gap = None
        up_dual = None
        if up_res["success"]:
            up_dual = solve_explicit_dual_with_scip(lp_components, bound_overrides={var_idx: (up_lb, None)})
            if up_dual["success"] and up_dual["y"] is not None:
                up_dual_eval_self = evaluate_explicit_dual_on_child(
                    lp_components,
                    up_dual["y"],
                    up_dual["alpha"],
                    up_dual["beta"],
                    bound_overrides={var_idx: (up_lb, None)},
                )
                up_dual_eval_gap = abs(float(up_res["objective_value"]) - float(up_dual_eval_self))

        if down_res["success"] and down_res["objective_value"] is not None:
            child_zero_obj_est[pos] = max(child_zero_obj_est[pos], float(down_res["objective_value"]))
            if down_dual is not None and down_dual["success"] and down_dual["y"] is not None:
                _add_dual_source("anchor_down", pos, down_dual["y"], down_dual["alpha"], down_dual["beta"])
        if up_res["success"] and up_res["objective_value"] is not None:
            child_one_obj_est[pos] = max(child_one_obj_est[pos], float(up_res["objective_value"]))
            if up_dual is not None and up_dual["success"] and up_dual["y"] is not None:
                _add_dual_source("anchor_up", pos, up_dual["y"], up_dual["alpha"], up_dual["beta"])

        anchor_child_records.append(
            {
                "cand_position": int(pos),
                "cand_lp_pos": var_idx,
                "down_status": str(down_res["status"]),
                "up_status": str(up_res["status"]),
                "down_obj": float("inf") if not down_res["success"] else float(down_res["objective_value"]),
                "up_obj": float("inf") if not up_res["success"] else float(up_res["objective_value"]),
                "down_dual_eval_self": down_dual_eval_self,
                "down_dual_eval_gap": down_dual_eval_gap,
                "up_dual_eval_self": up_dual_eval_self,
                "up_dual_eval_gap": up_dual_eval_gap,
                "down_dual_status": None if down_dual is None else str(down_dual["status"]),
                "up_dual_status": None if up_dual is None else str(up_dual["status"]),
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


def reconstruct_topk_strong_branching_scores(
    lp_components: Dict[str, Any],
    action_set: Sequence[int],
    candidate_scores: Sequence[float],
    cutoffbound: float,
    top_k: int = 8,
    as_mip: bool = False,
) -> Dict[str, Any]:
    """
    Recompute top-k strong-branching scores on reconstructed LP/MIP using SCIP.

    Candidate ordering and score formula follow `legacy_code_generator/02_generate_samples.py`.
    """
    action_set = np.asarray(action_set, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if action_set.shape[0] != candidate_scores.shape[0]:
        raise ValueError(
            f"action_set and candidate_scores size mismatch ({action_set.shape[0]} vs {candidate_scores.shape[0]})"
        )
    if action_set.size == 0:
        return {
            "k": 0,
            "parent_obj": None,
            "topk_by_score": [],
            "topk_rank_original": [],
            "topk_rank_computed": [],
        }

    parent_res = solve_reconstructed_lp_with_scip(lp_components, as_mip=as_mip)
    if not parent_res["success"]:
        raise RuntimeError(
            f"Failed to solve reconstructed parent LP/MIP with SCIP: {parent_res['status']} ({parent_res['message']})"
        )
    parent_obj = float(parent_res["objective_value"])
    x_parent = np.asarray(parent_res["x"], dtype=np.float64)
    bounds = lp_components.get("bounds")

    k_eff = int(min(max(int(top_k), 0), candidate_scores.size))
    topk_positions = np.argsort(candidate_scores)[-k_eff:][::-1]

    records: List[Dict[str, Any]] = []
    for pos in topk_positions.tolist():
        var_idx = int(action_set[pos])
        lpsol = float(x_parent[var_idx])

        lb_local, ub_local = bounds[var_idx] if bounds is not None else (None, None)
        lb_local_v = -np.inf if lb_local is None else float(lb_local)
        ub_local_v = +np.inf if ub_local is None else float(ub_local)
        down_ub = min(float(np.floor(lpsol)), ub_local_v)
        up_lb = max(float(np.ceil(lpsol)), lb_local_v)

        down_res = solve_reconstructed_lp_with_scip(
            lp_components,
            as_mip=as_mip,
            bound_overrides={var_idx: (None, down_ub)},
        )
        up_res = solve_reconstructed_lp_with_scip(
            lp_components,
            as_mip=as_mip,
            bound_overrides={var_idx: (up_lb, None)},
        )

        child_zero_obj = float("inf") if not down_res["success"] else float(down_res["objective_value"])
        child_one_obj = float("inf") if not up_res["success"] else float(up_res["objective_value"])
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
                "lpsol": lpsol,
                "lb_local": lb_local_v,
                "ub_local": ub_local_v,
                "down_ub": down_ub,
                "up_lb": up_lb,
                "parent_lp_obj": parent_obj,
                "child_zero_lp_obj": child_zero_obj,
                "child_one_lp_obj": child_one_obj,
                "computed_score": computed_score,
                "down_status": str(down_res["status"]),
                "up_status": str(up_res["status"]),
            }
        )

    original_rank = sorted(records, key=lambda row: (-float(row["cand_score"]), int(row["cand_position"])))
    computed_rank = sorted(records, key=lambda row: (-float(row["computed_score"]), int(row["cand_position"])))
    original_order = [int(row["cand_position"]) for row in original_rank]
    computed_order = [int(row["cand_position"]) for row in computed_rank]

    return {
        "k": k_eff,
        "parent_obj": parent_obj,
        "topk_by_score": records,
        "topk_rank_original": original_order,
        "topk_rank_computed": computed_order,
    }
