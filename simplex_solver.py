from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class SimplexResult:
    """Container for simplex outcomes."""

    status: str
    optimal_value: float | None
    solution: List[float] | None
    iterations: int
    basis: List[int]
    tableau: List[List[float]]
    num_variables: int
    num_constraints: int


class SimplexError(Exception):
    pass


def _validate_dimensions(
    c: Sequence[float], A: Sequence[Sequence[float]], b: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not c:
        raise SimplexError("objective vector c must be non-empty.")
    if not A:
        raise SimplexError("constraint matrix A must contain at least one row.")

    try:
        c_arr = np.asarray(c, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimplexError("objective vector c must contain numeric values.") from exc
    try:
        A_arr = np.asarray(A, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimplexError("constraint matrix A must be rectangular with numeric values.") from exc
    try:
        b_arr = np.asarray(b, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimplexError("constraint vector b must contain numeric values.") from exc

    if c_arr.ndim != 1:
        raise SimplexError("objective vector c must be one-dimensional.")
    if A_arr.ndim != 2:
        raise SimplexError("constraint matrix A must be two-dimensional.")
    if b_arr.ndim != 1:
        raise SimplexError("constraint vector b must be one-dimensional.")

    num_constraints, num_variables = A_arr.shape

    if c_arr.size != num_variables:
        raise SimplexError(
            "length of c does not match number of columns in A "
            f"({c_arr.size} vs {num_variables})."
        )
    if b_arr.size != num_constraints:
        raise SimplexError(
            "length of b does not match number of rows in A "
            f"({b_arr.size} vs {num_constraints})."
        )

    if np.any(b_arr < 0):
        raise SimplexError(
            "all entries of b must be non-negative for <= constraints."
        )

    return c_arr, A_arr, b_arr


def simplex(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> SimplexResult:
    c_arr, A_arr, b_arr = _validate_dimensions(c, A, b)
    m, n = A_arr.shape

    width = n + m + 1
    tableau = np.zeros((m + 1, width), dtype=float)

    tableau[:m, :n] = A_arr
    tableau[:m, n:n + m] = np.eye(m, dtype=float)
    tableau[:m, -1] = b_arr
    tableau[-1, :n] = -c_arr

    basis: List[int] = [n + i for i in range(m)]
    iterations = 0

    while iterations < max_iter:
        reduced_costs = tableau[-1, :-1]
        entering_candidates = np.where(reduced_costs < -tol)[0]

        if entering_candidates.size == 0:
            full_solution = np.zeros(n + m, dtype=float)
            for row_index, var_index in enumerate(basis):
                full_solution[var_index] = tableau[row_index, -1]
            primal_solution = full_solution[:n]
            optimal_value = float(np.dot(c_arr, primal_solution))

            return SimplexResult(
                status="optimal",
                optimal_value=optimal_value,
                solution=primal_solution.tolist(),
                iterations=iterations,
                basis=basis.copy(),
                tableau=tableau.tolist(),
                num_variables=n,
                num_constraints=m,
            )

        entering = int(entering_candidates[np.argmin(reduced_costs[entering_candidates])])
        column = tableau[:m, entering]

        if np.all(column <= tol):
            return SimplexResult(
                status="unbounded",
                optimal_value=None,
                solution=None,
                iterations=iterations,
                basis=basis.copy(),
                tableau=tableau.tolist(),
                num_variables=n,
                num_constraints=m,
            )

        feasible_rows = np.where(column > tol)[0]
        if feasible_rows.size == 0:
            return SimplexResult(
                status="unbounded",
                optimal_value=None,
                solution=None,
                iterations=iterations,
                basis=basis.copy(),
                tableau=tableau.tolist(),
                num_variables=n,
                num_constraints=m,
            )

        ratios = [
            (tableau[i, -1] / column[i], basis[i], i) for i in feasible_rows
        ]
        _, _, leaving = min(ratios, key=lambda item: (item[0], item[1]))

        pivot = tableau[leaving, entering]
        if abs(pivot) <= tol:
            raise SimplexError("Encountered zero pivot; consider perturbing the problem.")

        tableau[leaving] = tableau[leaving] / pivot

        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i, entering]
            if abs(factor) <= tol:
                continue
            tableau[i] = tableau[i] - factor * tableau[leaving]

        basis[leaving] = entering
        iterations += 1

    return SimplexResult(
        status="iteration_limit",
        optimal_value=None,
        solution=None,
        iterations=iterations,
        basis=basis.copy(),
        tableau=tableau.tolist(),
        num_variables=n,
        num_constraints=m,
    )
