"""CoSo backend - implements Backend ABC by emitting CoLa."""
from __future__ import annotations

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.coso.encoder import encode
from cofola.backend.coso.solver import run_coso_program
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import ConstantFolder, SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass
from cofola.planing.pipeline import PlanningProfile

__all__ = ["COSO_GLOBAL_PASSES", "COSO_LOCAL_PASSES", "CoSoBackend"]


COSO_GLOBAL_PASSES = (
    FixedPointPass(ConstantFolder),
    MergeIdenticalObjects,
)


# CoSo can directly represent tuple/permutation configurations with absolute
# positional and counting constraints. Preserve frontend TupleDef nodes instead
# of lowering them to FuncDef, and let the encoder reject sequence/circle
# constructs with relative positional constraints.
COSO_LOCAL_PASSES = (
    SizeConstraintFolder,
    MergeIdenticalObjects,
    SimplifyPass,
)


class CoSoBackend(Backend):
    """Solves a single-configuration problem by translating it to CoLa."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "coso"

    def planning_profile(self) -> PlanningProfile:
        """Return the CoSo-compatible planning profile."""

        return PlanningProfile(
            global_passes=COSO_GLOBAL_PASSES,
            local_passes=COSO_LOCAL_PASSES,
        )

    def solve(
        self,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> int:
        """Encode and solve a single atomic planning problem via CoSo."""

        logger.info(
            "CoSoBackend.solve: encoding planning problem ({} objects, {} constraints)",
            len(list(problem.iter_objects())),
            len(problem.constraints),
        )
        program = encode(problem, analysis)
        if program.is_trivial:
            logger.debug("CoSoBackend: trivial component -> {}", program.trivial_count)
            return program.trivial_count

        logger.debug("CoSoBackend: generated CoLa program:\n{}", program.cola)
        result = run_coso_program(program.cola, debug=self.debug)
        if program.count_divisor != 1:
            if result % program.count_divisor != 0:
                raise ValueError(
                    "CoSo result is not divisible by indexed composition "
                    f"normalization factor {program.count_divisor}: {result}"
                )
            result //= program.count_divisor
        logger.info("CoSoBackend: final result = {}", result)
        return result
