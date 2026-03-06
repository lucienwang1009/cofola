"""Tests for the Pass ABC, PassManager, and AnalysisManager."""
from __future__ import annotations

import pytest


def test_analysis_pass_cannot_be_instantiated_directly():
    from cofola.ir.pass_manager import AnalysisPass

    with pytest.raises(TypeError):
        AnalysisPass()  # type: ignore[abstract]


def test_transform_pass_cannot_be_instantiated_directly():
    from cofola.ir.pass_manager import TransformPass

    with pytest.raises(TypeError):
        TransformPass()  # type: ignore[abstract]


def test_analysis_manager_caches_result():
    """get() should return the same object on repeated calls."""
    from cofola.ir.pass_manager import AnalysisManager
    from cofola.ir.pass_manager import AnalysisPass

    class CountingAnalysis(AnalysisPass):
        required_analyses = []
        count = 0

        def run(self, problem, am=None):
            CountingAnalysis.count += 1
            return "result"

    CountingAnalysis.count = 0
    sentinel = object()
    am = AnalysisManager(sentinel)
    r1 = am.get(CountingAnalysis)
    r2 = am.get(CountingAnalysis)
    assert r1 == r2 == "result"
    assert CountingAnalysis.count == 1  # ran only once


def test_analysis_manager_invalidates_on_update():
    """update() should clear cache so next get() re-runs the analysis."""
    from cofola.ir.pass_manager import AnalysisManager
    from cofola.ir.pass_manager import AnalysisPass

    class CountingAnalysis(AnalysisPass):
        required_analyses = []
        count = 0

        def run(self, problem, am=None):
            CountingAnalysis.count += 1
            return "result"

    CountingAnalysis.count = 0
    sentinel1, sentinel2 = object(), object()
    am = AnalysisManager(sentinel1)
    am.get(CountingAnalysis)
    assert CountingAnalysis.count == 1

    am.update(sentinel2)
    assert am.problem is sentinel2

    am.get(CountingAnalysis)
    assert CountingAnalysis.count == 2  # re-ran after invalidation


def test_analysis_manager_resolves_required_analyses():
    """get() should auto-resolve required_analyses recursively."""
    from cofola.ir.pass_manager import AnalysisManager
    from cofola.ir.pass_manager import AnalysisPass

    class BaseAnalysis(AnalysisPass):
        required_analyses = []

        def run(self, problem, am=None):
            return "base"

    class DerivedAnalysis(AnalysisPass):
        required_analyses = [BaseAnalysis]

        def run(self, problem, am=None):
            base = am.get(BaseAnalysis)
            return f"derived({base})"

    am = AnalysisManager(object())
    result = am.get(DerivedAnalysis)
    assert result == "derived(base)"


def test_merged_analysis_returns_analysis_result():
    """MergedAnalysis should return an AnalysisResult with sizes populated."""
    from cofola.ir.analysis.merged import MergedAnalysis
    from cofola.ir.analysis.entities import AnalysisResult
    from cofola.ir.pass_manager import AnalysisManager
    from cofola.frontend.problem import ProblemBuilder
    from cofola.frontend.objects import SetInit
    from cofola.frontend.types import Entity

    builder = ProblemBuilder()
    builder.add(SetInit(entities=frozenset([Entity("a"), Entity("b")])), name="A")
    problem = builder.build()

    am = AnalysisManager(problem)
    result = am.get(MergedAnalysis)
    assert isinstance(result, AnalysisResult)
    assert not result.unsatisfiable
