"""Tests for PassManager and AnalysisManager behavior."""
from __future__ import annotations

import pytest


def test_analysis_manager_caches_result():
    """get() should return the same object on repeated calls."""
    from cofola.planing.pass_manager import AnalysisManager
    from cofola.planing.pass_manager import AnalysisPass

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
    from cofola.planing.pass_manager import AnalysisManager
    from cofola.planing.pass_manager import AnalysisPass

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
    from cofola.planing.pass_manager import AnalysisManager
    from cofola.planing.pass_manager import AnalysisPass

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


def test_fixed_point_pass_invalidates_analyses_between_iterations():
    """FixedPointPass should recompute analyses after every changed step."""
    from cofola.frontend.objects import Entity
    from cofola.frontend.objects import ObjRef
    from cofola.frontend.objects import SetInit
    from cofola.frontend.problem import Problem
    from cofola.planing.pass_manager import AnalysisPass
    from cofola.planing.pass_manager import FixedPointPass
    from cofola.planing.pass_manager import PassResult
    from cofola.planing.pass_manager import TransformPass
    from cofola.planing.pipeline import PlaningPipeline

    class CountingAnalysis(AnalysisPass):
        required_analyses = []
        count = 0

        def run(self, problem, am=None):
            CountingAnalysis.count += 1
            return len(problem.defs)

    class AddUntilTwo(TransformPass):
        required_analyses = [CountingAnalysis]

        def run(self, problem, am=None):
            am.get(CountingAnalysis)
            if len(problem.defs) >= 2:
                return PassResult(problem=problem, changed=False)
            ref = ObjRef(len(problem.defs))
            new_def = SetInit(entities=frozenset({Entity(f"e{ref.id}")}))
            return PassResult(
                problem=Problem(
                    defs=problem.defs + ((ref, new_def),),
                    constraints=problem.constraints,
                    names=problem.names,
                    locs=problem.locs,
                ),
                changed=True,
            )

    CountingAnalysis.count = 0
    am = PlaningPipeline.run_passes(
        Problem(defs=(), constraints=(), names=()),
        [FixedPointPass(AddUntilTwo)],
    )

    assert len(am.problem.defs) == 2
    assert CountingAnalysis.count == 3


def test_merged_analysis_returns_analysis_result():
    """MergedAnalysis should return an AnalysisResult with sizes populated."""
    from cofola.planing.analysis.merged import MergedAnalysis
    from cofola.planing.analysis.entities import AnalysisResult
    from cofola.planing.pass_manager import AnalysisManager
    from cofola.frontend.problem import ProblemBuilder
    from cofola.frontend.objects import SetInit
    from cofola.frontend.objects import Entity

    builder = ProblemBuilder()
    builder.add(SetInit(entities=frozenset([Entity("a"), Entity("b")])), name="A")
    problem = builder.build()

    am = AnalysisManager(problem)
    result = am.get(MergedAnalysis)
    assert isinstance(result, AnalysisResult)
    assert not result.unsatisfiable
