from wfomc import wfomc, Algo, WFOMCProblem, UnaryEvidenceEncoding
from wfomc.utils import RingElement


def solve(problem: WFOMCProblem, algo: Algo,
          use_partition_constraint: bool = False) -> RingElement:
    """Solve the given problem using the given algorithm.

    Args:
        problem (WFOMCProblem): The problem to solve.

    Returns:
        int: The number of models of the given problem.
    """
    if use_partition_constraint:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.PC)
    else:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.CCS)
    return ret
