"""Pass infrastructure for the Cofola frontend.

Provides abstract base classes for analysis and transformation passes,
along with a PassManager for chaining passes together.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AnalysisPass(ABC):
    """Abstract base class for analysis passes.

    An analysis pass reads a Problem and produces some derived data,
    but does not transform the Problem itself.
    """

    @abstractmethod
    def run(self, problem: Any) -> Any:
        """Run the analysis pass.

        Args:
            problem: The Problem to analyse.

        Returns:
            The analysis result (type depends on the concrete pass).
        """
        ...


class TransformPass(ABC):
    """Abstract base class for transformation passes.

    A transformation pass takes a Problem and returns a (possibly modified)
    Problem.
    """

    @abstractmethod
    def run(self, problem: Any) -> Any:
        """Run the transformation pass.

        Args:
            problem: The Problem to transform.

        Returns:
            The transformed Problem.
        """
        ...


class PassManager:
    """Manager that chains transformation passes together.

    Passes are run in the order they were added. Each pass receives the
    output of the previous pass.

    Args:
        passes: Optional initial list of passes.
    """

    def __init__(self, passes: list[TransformPass] | None = None) -> None:
        self._passes: list[TransformPass] = list(passes) if passes is not None else []

    def add(self, pass_: TransformPass) -> "PassManager":
        """Add a pass to the end of the chain.

        Args:
            pass_: The pass to add.

        Returns:
            self (for chaining).
        """
        self._passes.append(pass_)
        return self

    def run(self, problem: Any) -> Any:
        """Run all passes in order.

        Args:
            problem: The initial Problem.

        Returns:
            The Problem after all passes have been applied.
        """
        result = problem
        for pass_ in self._passes:
            result = pass_.run(result)
        return result
