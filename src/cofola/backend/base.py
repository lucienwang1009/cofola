"""Backend ABC — decouples the IR from any specific solver."""
from __future__ import annotations

from abc import ABC, abstractmethod

from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult

__all__ = ["Backend"]


class Backend(ABC):
    """Abstract solver backend.

    Implementations translate a fully-analysed, lowered IR Problem into
    an integer count.

    The Problem passed to :meth:`solve` must have been through:
    - EntityAnalysis
    - ConstantFolder
    - MaxSizeInference
    - LoweringPass
    - SimplifyPass
    - BagClassification

    Inspired by Z3's solver interface pattern.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abstractmethod
    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        """Translate and solve a single atomic sub-problem.

        Args:
            problem: A fully-simplified :class:`~cofola.ir.problem.Problem`.
                     Must be a single connected component with no compound
                     constraints.
            analysis: The :class:`~cofola.ir.analysis.entities.AnalysisResult`
                      carrying SetInfo/BagInfo for every ref in problem.

        Returns:
            The integer count.
        """
        ...
