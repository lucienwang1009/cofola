"""WFOMC backend — implements Backend ABC using the wfomc library."""
from __future__ import annotations

from typing import Union

from wfomc import Algo, UnaryEvidenceStrategy
from wfomc.algo import LinearOrderEncoding
from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.wfomc.solver import solve_wfomc
from cofola.backend.wfomc.encoder import encode
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import (
    ConstantFolder,
    FullChoiceOptimizer,
    SizeConstraintFolder,
)
from cofola.planing.passes.simplify import SimplifyPass
from cofola.planing.pipeline import PlanningProfile

__all__ = ["WFOMC_GLOBAL_PASSES", "WFOMC_LOCAL_PASSES", "WFOMCBackend"]


WFOMC_GLOBAL_PASSES = (
    FixedPointPass(ConstantFolder),
    FixedPointPass(FullChoiceOptimizer),
    MergeIdenticalObjects,
)

WFOMC_LOCAL_PASSES = (
    SizeConstraintFolder,
    FixedPointPass(LoweringPass),
    MergeIdenticalObjects,
    SimplifyPass,
)


class WFOMCBackend(Backend):
    """Solves a problem by translating it to a WFOMC problem via encode()."""

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "wfomc"

    def __init__(
        self,
        algo: Algo = Algo.FASTv2,
        unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.AUTO,
        lifted: bool = False,
        linear_order_encoding: Union[LinearOrderEncoding, str, None] = None,
    ) -> None:
        self.algo = algo
        self.unary_evidence_strategy = unary_evidence_strategy
        self.lifted = lifted
        # Only consulted when algo == Algo.PROPOSITIONAL; ignored otherwise.
        # None lets the wfomc library use its default (PIN).
        self.linear_order_encoding = linear_order_encoding

    def planning_profile(self) -> PlanningProfile:
        """Return the WFOMC-compatible planning profile."""

        return PlanningProfile(
            global_passes=WFOMC_GLOBAL_PASSES,
            local_passes=WFOMC_LOCAL_PASSES,
        )

    def solve(
        self,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> int:
        """Encode and solve a single atomic planning problem via WFOMC.

        Args:
            problem: A fully-lowered, simplified planning Problem (single connected
                     component, no compound constraints).
            analysis: BagClassification result carrying SetInfo/BagInfo for
                      every ref in problem.

        Returns:
            The integer count, or 0 if unsatisfiable.
        """
        logger.info("WFOMCBackend.solve: encoding planning problem ({} objects, {} constraints)",
                    len(list(problem.iter_objects())), len(problem.constraints))

        wfomc_problem, decoder = encode(problem, analysis, self.lifted)

        algo = self.algo
        unary_evidence_strategy = self.unary_evidence_strategy
        _order_capable = (Algo.INCREMENTAL, Algo.RECURSIVE, Algo.PROPOSITIONAL)
        if wfomc_problem.contain_linear_order_axiom() and algo not in _order_capable:
            logger.warning(
                'Linear order axiom with the predicate LEQ is found, '
                'while the algorithm does not support it '
                '(supported: INCREMENTAL, RECURSIVE, PROPOSITIONAL). '
                'Switching to INCREMENTAL algorithm...'
            )
            algo = Algo.INCREMENTAL
            unary_evidence_strategy = UnaryEvidenceStrategy.AUTO

        logger.debug("WFOMCBackend: algo={}", algo)

        try:
            raw = solve_wfomc(
                wfomc_problem,
                algo,
                unary_evidence_strategy,
                linear_order_encoding=self.linear_order_encoding,
            )
        except IndexError as exc:
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
