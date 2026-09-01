"""Direct Essence/Conjure backend for Cofola planning problems."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.essence.encoder import EssenceEncoder
from cofola.backend.essence.solver import EssenceSolverConfig, run_conjure
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import ForAllPartsExpansionStep, LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import ConstantFolder, SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass
from cofola.planing.pipeline import PlanningProfile

__all__ = ["EssenceBackend", "ESSENCE_GLOBAL_PASSES", "ESSENCE_LOCAL_PASSES"]


class _EssenceForAllPartsLoweringPass(LoweringPass):
    """Expand forall-part templates without lowering ordered objects."""

    STEP_CLASSES = (ForAllPartsExpansionStep,)


ESSENCE_GLOBAL_PASSES = (
    FixedPointPass(ConstantFolder),
    MergeIdenticalObjects,
)

ESSENCE_LOCAL_PASSES = (
    FixedPointPass(_EssenceForAllPartsLoweringPass),
    SizeConstraintFolder,
    MergeIdenticalObjects,
    SimplifyPass,
)


class EssenceBackend(Backend):
    """Solve a backend-ready planning component by counting Conjure solutions."""

    def __init__(
        self,
        *,
        conjure_dir: str | Path | None = None,
        java_bin: str | Path | None = None,
        timeout: float | None = None,
        debug: bool = False,
    ) -> None:
        self.config = EssenceSolverConfig(
            conjure_dir=Path(conjure_dir) if conjure_dir is not None else None,
            java_bin=Path(java_bin) if java_bin is not None else None,
            timeout=timeout,
        )
        self.debug = debug

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "essence"

    def planning_profile(self) -> PlanningProfile:
        """Return the direct-Essence planning profile."""

        return PlanningProfile(
            global_passes=ESSENCE_GLOBAL_PASSES,
            local_passes=ESSENCE_LOCAL_PASSES,
        )

    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        """Encode the component to Essence and count Conjure solutions."""

        if analysis.unsatisfiable:
            return 0
        program = EssenceEncoder(problem, analysis).encode()
        logger.debug("EssenceBackend generated model:\n{}", program)
        return run_conjure(program, self.config)
