"""
Pure Python simplex solver for maximization problems:
    max c^T x
    s.t. A x <= b, x >= 0.

No external numerical libraries are used; all tableau operations rely on lists of
floats. The solver assumes the linear program is already in standard form with
non-negative right-hand sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


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
    """Raised when inputs violate solver assumptions."""


def _ensure_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
        raise SimplexError(f"Cannot convert value {value!r} to float.") from exc


def _validate_dimensions(
    c: Sequence[float], A: Sequence[Sequence[float]], b: Sequence[float]
) -> Tuple[List[float], List[List[float]], List[float]]:
    if not c:
        raise SimplexError("Objective vector c must be non-empty.")
    if not A:
        raise SimplexError("Constraint matrix A must contain at least one row.")

    c_vec = [_ensure_float(v) for v in c]
    A_mat = [[_ensure_float(a_ij) for a_ij in row] for row in A]
    b_vec = [_ensure_float(v) for v in b]

    num_constraints = len(A_mat)
    num_variables = len(A_mat[0])

    if len(c_vec) != num_variables:
        raise SimplexError(
            "Length of c does not match number of columns in A "
            f"({len(c_vec)} vs {num_variables})."
        )
    if len(b_vec) != num_constraints:
        raise SimplexError(
            "Length of b does not match number of rows in A "
            f"({len(b_vec)} vs {num_constraints})."
        )

    for row in A_mat:
        if len(row) != num_variables:
            raise SimplexError("All rows in A must have the same length.")
    for b_i in b_vec:
        if b_i < 0:
            raise SimplexError(
                "All entries of b must be non-negative for <= constraints."
            )

    return c_vec, A_mat, b_vec


def simplex(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> SimplexResult:
    """
    Solve a standard-form maximization linear program using the simplex method.

    Parameters
    ----------
    c, A, b :
        Problem data for max c^T x subject to A x <= b, x >= 0.
    max_iter : int
        Maximum number of pivot iterations.
    tol : float
        Tolerance used when selecting entering and leaving variables.
    """
    c_vec, A_mat, b_vec = _validate_dimensions(c, A, b)
    m = len(A_mat)
    n = len(c_vec)

    width = n + m + 1
    tableau: List[List[float]] = [[0.0] * width for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            tableau[i][j] = A_mat[i][j]
        tableau[i][n + i] = 1.0  # Slack variable coefficient.
        tableau[i][-1] = b_vec[i]

    for j in range(n):
        tableau[-1][j] = -c_vec[j]

    basis: List[int] = [n + i for i in range(m)]
    iterations = 0

    while iterations < max_iter:
        reduced_costs = tableau[-1][:-1]
        entering_candidates = [idx for idx, value in enumerate(reduced_costs) if value < -tol]

        if not entering_candidates:
            full_solution = [0.0] * (n + m)
            for row_index, var_index in enumerate(basis):
                full_solution[var_index] = tableau[row_index][-1]
            primal_solution = full_solution[:n]
            optimal_value = sum(c_vec[j] * primal_solution[j] for j in range(n))

            return SimplexResult(
                status="optimal",
                optimal_value=optimal_value,
                solution=primal_solution,
                iterations=iterations,
                basis=basis.copy(),
                tableau=[row[:] for row in tableau],
                num_variables=n,
                num_constraints=m,
            )

        entering = min(entering_candidates, key=lambda idx: reduced_costs[idx])
        column = [tableau[i][entering] for i in range(m)]

        if all(value <= tol for value in column):
            return SimplexResult(
                status="unbounded",
                optimal_value=None,
                solution=None,
                iterations=iterations,
                basis=basis.copy(),
                tableau=[row[:] for row in tableau],
                num_variables=n,
                num_constraints=m,
            )

        ratios = []
        for i in range(m):
            if column[i] > tol:
                ratios.append((tableau[i][-1] / column[i], i))
        if not ratios:
            return SimplexResult(
                status="unbounded",
                optimal_value=None,
                solution=None,
                iterations=iterations,
                basis=basis.copy(),
                tableau=[row[:] for row in tableau],
                num_variables=n,
                num_constraints=m,
            )

        _, leaving = min(ratios, key=lambda item: (item[0], basis[item[1]]))
        pivot = tableau[leaving][entering]
        if abs(pivot) <= tol:
            raise SimplexError("Encountered zero pivot; consider perturbing the problem.")

        pivot_row = [value / pivot for value in tableau[leaving]]
        tableau[leaving] = pivot_row

        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            if abs(factor) <= tol:
                continue
            tableau[i] = [
                tableau[i][j] - factor * pivot_row[j] for j in range(width)
            ]

        basis[leaving] = entering
        iterations += 1

    return SimplexResult(
        status="iteration_limit",
        optimal_value=None,
        solution=None,
        iterations=iterations,
        basis=basis.copy(),
        tableau=[row[:] for row in tableau],
        num_variables=n,
        num_constraints=m,
    )
