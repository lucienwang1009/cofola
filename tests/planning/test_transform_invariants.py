"""Cross-pass transform invariants on the planning pipeline."""
from __future__ import annotations

import pytest

from cofola.frontend import Problem
from cofola.frontend.utils import (
    constraint_refs,
    object_refs,
)
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.pipeline import PlaningPipeline
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import (
    ConstantFolder,
    SizeConstraintFolder,
)
from cofola.planing.passes.simplify import SimplifyPass
from cofola.parser.parser import parse


def _assert_problem_refs_are_defined(problem: Problem) -> None:
    defined = set(problem.refs())
    for _ref, defn in problem.iter_objects():
        assert set(object_refs(defn)) <= defined
    for constraint in problem.constraints:
        assert set(constraint_refs(constraint)) <= defined

class TestPlanningTransformInvariants(object):
    """Cross-pass structural invariants."""

    @pytest.mark.parametrize(
        "pass_spec,program",
        [
            (
                ConstantFolder,
                """
A = set(a)
B = set(b)
C = A + B
|C| == 2
""",
            ),
            (
                SizeConstraintFolder,
                """
S = set(a, b, c)
T = choose(S)
|T| == 2
""",
            ),
            (
                FixedPointPass(LoweringPass),
                """
S = set(a, b, c)
T = tuple(S)
T[0] == a
""",
            ),
            (
                MergeIdenticalObjects,
                """
A = set(a)
B = set(a)
C = A + B
a in C
""",
            ),
            (
                SimplifyPass,
                """
A = set(a)
B = set(b)
a in A
""",
            ),
        ],
        ids=[
            "constant-folder",
            "size-constraint-folder",
            "lowering",
            "merge-identical",
            "simplify",
        ],
    )
    def test_transform_pass_preserves_defined_ref_invariant(self, pass_spec, program: str) -> None:
        """Every transform pass should leave object and constraint refs defined."""
        problem = parse(program)

        am = PlaningPipeline.run_passes(problem, [pass_spec])

        _assert_problem_refs_are_defined(am.problem)
