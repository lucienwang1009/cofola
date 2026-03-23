from wfomc import wfomc, Algo, WFOMCProblem, UnaryEvidenceEncoding
from wfomc.utils import RingElement


def solve_wfomc(problem: WFOMCProblem, algo: Algo,
                use_partition_constraint: bool = False) -> RingElement:
    """Solve the given WFOMC problem using the given algorithm.

    Args:
        problem: The WFOMC problem to solve.
        algo: The WFOMC algorithm to use.
        use_partition_constraint: Whether to use partition constraint encoding.

    Returns:
        The raw ring element result.
    """
    if use_partition_constraint:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.PC)
    else:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.CCS)
    return ret
