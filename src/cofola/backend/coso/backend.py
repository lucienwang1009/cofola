"""CoSo backend - implements Backend ABC by emitting CoLa."""
from __future__ import annotations

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.coso.encoder import encode
from cofola.backend.coso.solver import run_coso_program
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult

__all__ = ["CoSoBackend"]


class CoSoBackend(Backend):
    """Solves a single-configuration problem by translating it to CoLa."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "coso"

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
        logger.info("CoSoBackend: final result = {}", result)
        return result
