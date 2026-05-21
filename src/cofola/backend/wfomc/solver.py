from __future__ import annotations

from typing import Union

from wfomc import wfomc, Algo, WFOMCProblem, UnaryEvidenceEncoding
from wfomc.algo import LinearOrderEncoding
from wfomc.utils import RingElement


def solve_wfomc(
    problem: WFOMCProblem,
    algo: Algo,
    use_partition_constraint: bool = False,
    linear_order_encoding: Union[LinearOrderEncoding, str, None] = None,
) -> RingElement:
    """Solve the given WFOMC problem using the given algorithm.

    Args:
        problem: The WFOMC problem to solve.
        algo: The WFOMC algorithm to use.
        use_partition_constraint: Whether to use partition constraint encoding.
            Ignored when ``algo == Algo.PROPOSITIONAL`` because that algorithm
            selects its own unary-evidence encoding (``CCS`` or ``NONE``)
            based on whether order axioms are pinned or axiomatized.
        linear_order_encoding: Only consulted when ``algo == Algo.PROPOSITIONAL``;
            picks ``PIN`` (cheap, default) or ``AXIOMS`` (FO³ ground truth).

    Returns:
        The raw ring element result.
    """
    unary_evidence_encoding = (
        UnaryEvidenceEncoding.PC if use_partition_constraint
        else UnaryEvidenceEncoding.CCS
    )
    return wfomc(
        problem,
        algo,
        unary_evidence_encoding=unary_evidence_encoding,
        linear_order_encoding=linear_order_encoding,
    )
