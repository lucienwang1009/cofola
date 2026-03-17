"""WFOMC backend — implements Backend ABC using the wfomc library."""
from __future__ import annotations

from wfomc import Algo
from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.wfomc.solver import solve_wfomc
from cofola.backend.wfomc.encoder_ir import encode_ir
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult

__all__ = ["WFOMCBackend"]


class WFOMCBackend(Backend):
    """Solves a problem by translating it to a WFOMC problem via encode_ir()."""

    name = "wfomc"

    def __init__(
        self,
        algo: Algo = Algo.FASTv2,
        use_partition_constraint: bool = True,
        lifted: bool = False,
    ) -> None:
        self.algo = algo
        self.use_partition_constraint = use_partition_constraint
        self.lifted = lifted

    def solve(
        self,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> int:
        """Encode and solve a single atomic IR problem via WFOMC.

        Args:
            problem: A fully-lowered, simplified IR Problem (single connected
                     component, no compound constraints).
            analysis: BagClassification result carrying SetInfo/BagInfo for
                      every ref in problem.

        Returns:
            The integer count, or 0 if unsatisfiable.
        """
        logger.info("WFOMCBackend.solve: encoding IR problem ({} objects, {} constraints)",
                    len(list(problem.iter_objects())), len(problem.constraints))

        wfomc_problem, decoder = encode_ir(problem, analysis, self.lifted)

        algo = self.algo
        use_partition_constraint = self.use_partition_constraint
        if wfomc_problem.contain_linear_order_axiom() and \
                algo != Algo.INCREMENTAL and algo != Algo.RECURSIVE:
            logger.warning(
                'Linear order axiom with the predicate LEQ is found, '
                'while the algorithm is not INCREMENTAL or RECURSIVE. '
                'Switching to INCREMENTAL algorithm...'
            )
            algo = Algo.INCREMENTAL
            use_partition_constraint = True

        logger.debug("WFOMCBackend: algo={}", algo)

        try:
            raw = solve_wfomc(wfomc_problem, algo, use_partition_constraint)
        except (IndexError, Exception) as exc:
            # WFOMC library crashes on degenerate problems (e.g. empty domains).
            # Treat as unsatisfiable → 0.
            logger.warning("WFOMCBackend: WFOMC solver raised {}: {} — returning 0", type(exc).__name__, exc)
            return 0
        logger.debug("WFOMCBackend: raw wfomc result = {}", raw)

        result = decoder.decode_result(raw)
        logger.debug("WFOMCBackend: decoded result = {}", result)

        if result is None:
            logger.info("WFOMCBackend: result is None (unsatisfiable) -> 0")
            return 0
        logger.info("WFOMCBackend: final result = {}", result)
        return result
