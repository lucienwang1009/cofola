from wfomc import wfomc, Algo, WFOMCProblem, UnaryEvidenceEncoding


def solve(problem: WFOMCProblem, algo: Algo,
          use_partition_constraint: bool = False) -> int:
    """Solve the given problem using the given algorithm.

    Args:
        problem (WFOMCProblem): The problem to solve.

    Returns:
        int: The number of models of the given problem.
    """
    if use_partition_constraint:
        return wfomc(problem, algo, UnaryEvidenceEncoding.PC)
    else:
        return wfomc(problem, algo, UnaryEvidenceEncoding.CCS)
