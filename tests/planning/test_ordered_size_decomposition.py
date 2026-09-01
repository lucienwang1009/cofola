"""Planner materialization for full ordered object/source size coupling."""
from __future__ import annotations

from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.objects import ObjRef
from cofola.planing.pipeline import PlaningPipeline
from cofola.parser.parser import parse
from tests.helpers import _part_ref, _ref_named


def _source_size_equalities(constraints: tuple[object, ...], ref: ObjRef) -> set[int]:
    return {
        constraint.rhs
        for constraint in constraints
        if isinstance(constraint, SizeConstraint)
        and constraint.terms == ((ref, 1),)
        and constraint.comparator == "=="
    }


class TestOrderedSizeDecomposition(object):
    """Full ordered objects should persist |source| == |ordered| before backend."""

    def test_decomposed_sequence_size_branches_also_constrain_source(self) -> None:
        problem = parse(
            """
S = set(a, b)
P = compose(S, 2)
T = sequence(P[0])
"""
        )
        part = _part_ref(problem, _ref_named(problem, "P"), 0)

        schedule = PlaningPipeline(WFOMCBackend().planning_profile()).process(problem)

        assert len(schedule.branches) == 3
        branch_sizes = []
        for branch in schedule.branches:
            constraints = tuple(
                constraint
                for component, _analysis in branch.components
                for constraint in component.constraints
            )
            branch_sizes.append(_source_size_equalities(constraints, part))
        assert sorted(branch_sizes, key=lambda s: tuple(sorted(s))) == [{0}, {1}, {2}]


    def test_fixed_tuple_size_materializes_source_size_before_lowering(self) -> None:
        problem = parse(
            """
S = set(a, b)
P = compose(S, 2)
T = tuple(P[0])
|T| == 1
"""
        )
        part = _part_ref(problem, _ref_named(problem, "P"), 0)

        schedule = PlaningPipeline(WFOMCBackend().planning_profile()).process(problem)

        assert len(schedule.branches) == 1
        constraints = tuple(
            constraint
            for component, _analysis in schedule.branches[0].components
            for constraint in component.constraints
        )
        assert _source_size_equalities(constraints, part) == {1}
