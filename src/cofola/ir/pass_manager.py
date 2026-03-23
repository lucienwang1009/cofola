"""Pass infrastructure and Analysis Manager for the Cofola IR pipeline.

Provides abstract base classes for analysis and transformation passes,
plus an LLVM-style AnalysisManager for on-demand caching with automatic
invalidation when transformation passes modify the Problem.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from loguru import logger

A = TypeVar("A")


class AnalysisPass(ABC):
    """Abstract base class for analysis passes.

    An analysis pass reads a Problem and produces some derived data,
    but does not transform the Problem itself.

    Class attributes:
        required_analyses: List of AnalysisPass subclasses that must be
            computed before this pass runs. The AnalysisManager resolves
            these automatically.
    """

    required_analyses: list[type] = []

    @abstractmethod
    def run(self, problem: Any, am: Any = None) -> Any:
        """Run the analysis pass.

        Args:
            problem: The Problem to analyse.
            am: AnalysisManager instance for accessing required analyses.

        Returns:
            The analysis result (type depends on the concrete pass).
        """
        ...


class TransformPass(ABC):
    """Abstract base class for transformation passes.

    A transformation pass takes a Problem and returns a (possibly modified)
    Problem.

    Class attributes:
        required_analyses: List of AnalysisPass subclasses whose results
            this pass may read via the AnalysisManager.
    """

    required_analyses: list[type] = []

    @abstractmethod
    def run(self, problem: Any, am: Any = None) -> Any:
        """Run the transformation pass.

        Args:
            problem: The Problem to transform.
            am: AnalysisManager instance for accessing required analyses.

        Returns:
            The transformed Problem.
        """
        ...


class AnalysisManager:
    """Caches analysis results and invalidates them when the Problem changes.

    Each analysis pass declares a ``required_analyses`` class attribute listing
    other analysis passes that must run first. ``AnalysisManager.get()`` resolves
    these dependencies automatically and caches results until the Problem changes.

    Usage::

        am = AnalysisManager(problem)
        result = am.get(EntityAnalysis)   # computed and cached

        new_problem = LoweringPass().run(am.problem, am)
        am.update(new_problem)            # cache invalidated

        result = am.get(EntityAnalysis)   # re-computed
    """

    def __init__(self, problem: Any) -> None:
        self._problem = problem
        self._cache: dict[type, Any] = {}

    @property
    def problem(self) -> Any:
        """The current Problem being analysed."""
        return self._problem

    def update(self, new_problem: Any) -> None:
        """Update the current Problem and invalidate all cached analyses.

        Call this after any transformation pass that may have modified the Problem.
        If the new Problem is the same object as the current one, the cache is
        not cleared.

        Args:
            new_problem: The Problem returned by a transformation pass.
        """
        if new_problem is not self._problem:
            logger.debug(
                "AnalysisManager.update: problem changed, invalidating {} cached analyses",
                len(self._cache),
            )
            self._problem = new_problem
            self._cache.clear()

    def get(self, analysis_cls: type[A]) -> A:
        """Return the cached result for an analysis, computing it if needed.

        Recursively resolves ``required_analyses`` before running the analysis
        itself, so dependencies are always satisfied.

        Args:
            analysis_cls: The analysis class to retrieve.

        Returns:
            The cached (or freshly computed) analysis result.
        """
        if analysis_cls in self._cache:
            return self._cache[analysis_cls]  # type: ignore[return-value]

        # Resolve dependencies first (recursive, but each is cached after first run)
        for dep_cls in getattr(analysis_cls, "required_analyses", []):
            self.get(dep_cls)

        logger.debug("AnalysisManager: running {}", analysis_cls.__name__)
        instance = analysis_cls()
        result = instance.run(self._problem, self)
        self._cache[analysis_cls] = result
        return result  # type: ignore[return-value]
