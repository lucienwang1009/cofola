"""UnaryFOMC backend for the basic monadic set fragment of Cofola."""

from __future__ import annotations

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.unaryfomc.encoder import encode_ir
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult

__all__ = ["UnaryFOMCBackend"]


class UnaryFOMCBackend(Backend):
    """Solve supported normalized Cofola problems through UnaryFOMC."""

    name = "unaryfomc"

    def __init__(
        self,
        *,
        profile_pruning: bool = True,
        twin_compression: bool = True,
        max_cardinality_cases: int = 100_000,
    ) -> None:
        self.profile_pruning = profile_pruning
        self.twin_compression = twin_compression
        self.max_cardinality_cases = max_cardinality_cases

    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        try:
            from unaryfomc import c1f_fomc
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "The UnaryFOMC backend requires Cofola's 'unaryfomc' extra"
            ) from exc

        encoding = encode_ir(
            problem,
            analysis,
            max_cardinality_cases=self.max_cardinality_cases,
        )
        logger.info(
            "UnaryFOMCBackend.solve: domain={}, normalization={}",
            encoding.domain_size,
            encoding.normalization_factor,
        )
        raw = c1f_fomc(
            encoding.sentence,
            encoding.domain_size,
            profile_pruning=self.profile_pruning,
            twin_compression=self.twin_compression,
        )
        return encoding.decode_result(raw)
