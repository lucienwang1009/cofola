"""Direct ASP/clingo backend for Cofola planning problems."""
from __future__ import annotations

from loguru import logger

from cofola.backend.asp.encoder import ASPEncoder
from cofola.backend.asp.solver import run_clingo
from cofola.backend.base import Backend
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import ForAllPartsExpansionStep, LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import ConstantFolder, SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass
from cofola.planing.pipeline import PlanningProfile

__all__ = ["ASPBackend", "ASP_GLOBAL_PASSES", "ASP_LOCAL_PASSES"]


class _ASPForAllPartsLoweringPass(LoweringPass):
    """Expand forall-part templates without lowering ordered objects."""

    STEP_CLASSES = (ForAllPartsExpansionStep,)


ASP_GLOBAL_PASSES = (
    FixedPointPass(ConstantFolder),
    MergeIdenticalObjects,
)

ASP_LOCAL_PASSES = (
    FixedPointPass(_ASPForAllPartsLoweringPass),
    SizeConstraintFolder,
    MergeIdenticalObjects,
    SimplifyPass,
)


class ASPBackend(Backend):
    """Solve a backend-ready planning component by counting clingo models."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "asp"

    def planning_profile(self) -> PlanningProfile:
        """Return the direct-ASP planning profile."""

        return PlanningProfile(
            global_passes=ASP_GLOBAL_PASSES,
            local_passes=ASP_LOCAL_PASSES,
        )

    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        """Encode the component to ASP and count stable models."""

        if analysis.unsatisfiable:
            return 0
        program = ASPEncoder(problem, analysis).encode()
        logger.debug("ASPBackend generated program:\n{}", program)
        return run_clingo(program)
