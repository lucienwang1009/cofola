from __future__ import annotations

from cofola.backend.wfomc.api import (
    Algo,
    EncodedProblem,
    LinearOrderEncoding,
    UnaryEvidenceStrategy,
    WFOMCResult,
    solve_problem,
)


def solve_wfomc(
    problem: EncodedProblem,
    algo: Algo,
    unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.CCS,
    linear_order_encoding: LinearOrderEncoding | str | None = None,
) -> WFOMCResult:
    """Solve the given WFOMC problem using the given algorithm.

    Args:
        problem: The WFOMC problem to solve.
        algo: The WFOMC algorithm to use.
        unary_evidence_strategy: How unary evidence is handled. ``AUTO`` lets
            each algorithm use its best supported (lifted) implementation;
            ``CCS`` forces the auxiliary-predicate and cardinality-constraint
            encoding. For ``algo == Algo.PROPOSITIONAL`` the solver resolves the
            effective strategy itself based on the order axioms / encoding.
        linear_order_encoding: Only consulted when ``algo == Algo.PROPOSITIONAL``;
            picks ``PIN`` (cheap, default) or ``AXIOMS`` (FO³ ground truth).

    Returns:
        The WFOMC result object.
    """
    return solve_problem(
        problem,
        algo,
        unary_evidence_strategy,
        linear_order_encoding,
    )
