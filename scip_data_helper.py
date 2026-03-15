import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


SampleState = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]


def load_sample(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def unpack_sample_data(data: Sequence[Any]) -> Tuple[SampleState, int, Sequence[int], Sequence[float]]:
    sample_state = data[0]
    sample_action = data[2]
    sample_action_set = data[3]
    sample_scores = data[4]
    cutoffbound = data[5]
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

    for i in range(A_ub.shape[0]):
        row_start, row_end = A_ub.indptr[i], A_ub.indptr[i + 1]
        idxs = A_ub.indices[row_start:row_end]
        vals = A_ub.data[row_start:row_end]
        cons_expr = quicksum(float(v) * vars_[int(j)] for j, v in zip(idxs, vals))
        model.addCons(cons_expr <= float(b_ub[i]), name=f"c_{i}")

    model.optimize()
    status = str(model.getStatus()).lower()

    out = {
        "success": status == "optimal",
        "status": status,
        "message": status,
        "x": None,
        "objective_value": None,
        "raw_result": model,
    }

    if status == "optimal":
        sol = model.getBestSol()
        x = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
        obj = float(model.getObjVal()) + objective_offset
        out["x"] = x
        out["objective_value"] = obj

    return out


def compute_strong_branch_score(parent_obj: float, child_one_obj: float, child_zero_obj: float, cutoffbound: float) -> float:
    child_one_obj_capped = min(float(child_one_obj), float(cutoffbound))
    child_zero_obj_capped = min(float(child_zero_obj), float(cutoffbound))
    gain_one = max(child_one_obj_capped - float(parent_obj), 1e-9)
    gain_zero = max(child_zero_obj_capped - float(parent_obj), 1e-9)
    return float(gain_one * gain_zero)


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
