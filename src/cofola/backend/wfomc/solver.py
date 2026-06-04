from __future__ import annotations

from typing import Union

from wfomc import wfomc, Algo, WFOMCProblem, UnaryEvidenceStrategy, WFOMCResult
from wfomc.algo import LinearOrderEncoding


def solve_wfomc(
    problem: WFOMCProblem,
    algo: Algo,
    unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.CCS,
    linear_order_encoding: Union[LinearOrderEncoding, str, None] = None,
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
    return wfomc(
        problem,
        algo,
        unary_evidence_strategy=unary_evidence_strategy,
        linear_order_encoding=linear_order_encoding,
    )
