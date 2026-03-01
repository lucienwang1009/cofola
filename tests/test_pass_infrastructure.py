"""Tests for the Pass ABC and PassManager."""
from __future__ import annotations

import pytest


def test_pass_manager_chains_passes():
    from cofola.frontend.passes import PassManager, TransformPass

    class CountingPass(TransformPass):
        name = "counter"
        runs = 0

        def run(self, problem):
            CountingPass.runs += 1
            return problem

    pm = PassManager([CountingPass(), CountingPass()])
    # Use a simple sentinel object (not a real CofolaProblem)
    sentinel = object()
    CountingPass.runs = 0
    result = pm.run(sentinel)
    assert CountingPass.runs == 2


def test_analysis_pass_cannot_be_instantiated_directly():
    from cofola.frontend.passes import AnalysisPass

    with pytest.raises(TypeError):
        AnalysisPass()  # type: ignore[abstract]


def test_transform_pass_cannot_be_instantiated_directly():
    from cofola.frontend.passes import TransformPass

    with pytest.raises(TypeError):
        TransformPass()  # type: ignore[abstract]


def test_pass_manager_add_chains():
    from cofola.frontend.passes import PassManager, TransformPass

    class NoopPass(TransformPass):
        name = "noop"

        def run(self, problem):
            return problem

    pm = PassManager()
    result = pm.add(NoopPass()).add(NoopPass())
    assert result is pm
    assert len(pm._passes) == 2
