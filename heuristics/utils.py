"""Core utilities for anchor strong branching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


@dataclass(frozen=True)
class Problem:
    """Dual-ready LP description at one node.

    Primal form on unfixed variables `x_U`:
        min c_U^T x_U
        s.t. A_U x_U >= (b - A_F x_F)
             lb_U <= x_U <= ub_U

    The fixed variables contribution is represented by `(A_F, x_F)`.
    """

    b: np.ndarray
    A_F: np.ndarray
    x_F: np.ndarray
    A_U: np.ndarray
    c_U: np.ndarray
    lb_U: np.ndarray
    ub_U: np.ndarray

    def rhs(self) -> np.ndarray:
        if self.A_F.size == 0:
            return self.b.copy()
        return self.b - self.A_F @ self.x_F

    def branch_zero(self, local_var_idx: int) -> "Problem":
        lb = self.lb_U.copy()
        ub = self.ub_U.copy()
        lb[local_var_idx] = 0.0
        ub[local_var_idx] = 0.0
        return Problem(self.b, self.A_F, self.x_F, self.A_U, self.c_U, lb, ub)

    def branch_one(self, local_var_idx: int) -> "Problem":
        lb = self.lb_U.copy()
        ub = self.ub_U.copy()
        lb[local_var_idx] = 1.0
        ub[local_var_idx] = 1.0
        return Problem(self.b, self.A_F, self.x_F, self.A_U, self.c_U, lb, ub)


@dataclass
class DualSolveResult:
    y: Optional[np.ndarray]
    alpha: Optional[np.ndarray]
    beta: Optional[np.ndarray]
    objective: float
    success: bool
    status: int
    message: str


def evaluate_obj(problem: Problem, y: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
    """Evaluate a dual feasible point on the given problem objective."""

    rhs = problem.rhs()
    return float(rhs @ y - problem.ub_U @ alpha + problem.lb_U @ beta)


def solve_dual(problem: Problem) -> DualSolveResult:
    """Solve the LP dual with Pyomo + GLPK.

    Dual form:
        max rhs^T y - ub_U^T alpha + lb_U^T beta
        s.t. A_U^T y - alpha + beta = c_U
             y, alpha, beta >= 0
    """

    rhs = problem.rhs().astype(np.float64, copy=False)
    A_u = problem.A_U.astype(np.float64, copy=False)
    c_u = problem.c_U.astype(np.float64, copy=False)
    lb_u = problem.lb_U.astype(np.float64, copy=False)
    ub_u = problem.ub_U.astype(np.float64, copy=False)

    m, n = A_u.shape
    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=list(range(m)))
    model.J = pyo.Set(initialize=list(range(n)))

    model.y = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.alpha = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.beta = pyo.Var(model.J, domain=pyo.NonNegativeReals)

    def dual_eq_rule(mod, j):
        return (
            pyo.quicksum(A_u[i, j] * mod.y[i] for i in mod.I)
            - mod.alpha[j]
            + mod.beta[j]
            == c_u[j]
        )

    model.dual_eq = pyo.Constraint(model.J, rule=dual_eq_rule)
    model.obj = pyo.Objective(
        expr=(
            pyo.quicksum(rhs[i] * model.y[i] for i in model.I)
            - pyo.quicksum(ub_u[j] * model.alpha[j] for j in model.J)
            + pyo.quicksum(lb_u[j] * model.beta[j] for j in model.J)
        ),
        sense=pyo.maximize,
    )

    solver = pyo.SolverFactory("glpk")
    if not solver.available(False):
        return DualSolveResult(
            y=None,
            alpha=None,
            beta=None,
            objective=float("nan"),
            success=False,
            status=-1,
            message="GLPK solver is not available.",
        )

    results = solver.solve(model, tee=False, load_solutions=False)
    term = results.solver.termination_condition
    message = (
        f"status={results.solver.status}, "
        f"termination={term}, "
        f"message={results.solver.message}"
    )

    if term == TerminationCondition.optimal:
        model.solutions.load_from(results)
        def _safe_value(v):
            value = pyo.value(v, exception=False)
            return 0.0 if value is None else float(value)

        y = np.array([_safe_value(model.y[i]) for i in range(m)], dtype=np.float64)
        alpha = np.array([_safe_value(model.alpha[j]) for j in range(n)], dtype=np.float64)
        beta = np.array([_safe_value(model.beta[j]) for j in range(n)], dtype=np.float64)
        objective = evaluate_obj(problem, y, alpha, beta)
        return DualSolveResult(
            y=y,
            alpha=alpha,
            beta=beta,
            objective=objective,
            success=True,
            status=0,
            message=message,
        )

    # Dual unbounded commonly corresponds to primal child infeasibility.
    if term in {TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded}:
        return DualSolveResult(
            y=None,
            alpha=None,
            beta=None,
            objective=float("inf"),
            success=False,
            status=3,
            message=message,
        )

    return DualSolveResult(
        y=None,
        alpha=None,
        beta=None,
        objective=float("nan"),
        success=False,
        status=2,
        message=message,
    )


def compute_sbs(parent_obj: float, child_one_obj: float, child_zero_obj: float) -> float:
    """Strong-branching style score from parent/children objectives."""

    if any(np.isnan(v) for v in (parent_obj, child_one_obj, child_zero_obj)):
        return float("-inf")

    gain_one = max(child_one_obj - parent_obj, 1e-9)
    gain_zero = max(child_zero_obj - parent_obj, 1e-9)
    return float(gain_one * gain_zero)
