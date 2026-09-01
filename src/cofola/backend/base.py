"""Backend ABC — decouples planning Problems from concrete solvers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pipeline import PlanningProfile

__all__ = ["Backend"]


class Backend(ABC):
    """Abstract solver backend.

    Implementations translate a fully-analysed planning Problem into an integer
    count.

    The Problem passed to :meth:`solve` has been prepared by the planning
    pipeline according to this backend's :meth:`planning_profile`.

    Inspired by Z3's solver interface pattern.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    def planning_profile(self) -> PlanningProfile:
        """Return the planning profile requested by this backend."""

        return PlanningProfile()

    @abstractmethod
    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        """Translate and solve a single atomic sub-problem.

        Args:
            problem: A fully-simplified :class:`~cofola.planing.problem.Problem`.
                     Must be a single connected component with no compound
                     constraints.
            analysis: The :class:`~cofola.planing.analysis.entities.AnalysisResult`
                      carrying SetInfo/BagInfo for every ref in problem.

        Returns:
            The integer count.
        """
        ...
