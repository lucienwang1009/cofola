"""Planning optimization passes: constant folding, full-choice, size-constraint folding, merge."""
from __future__ import annotations

import pytest

from cofola.frontend import (
    BagChoose,
    Entity,
    ObjRef,
    Problem,
    SetChoose,
    SetInit,
    SetUnion,
    SizeConstraint,
    TupleDef,
)
from cofola.planing.pass_manager import (
    AnalysisManager,
    FixedPointPass,
    UnsatisfiableConstraint,
)
from cofola.planing.pipeline import PlaningPipeline
from cofola.planing.passes.optimize import (
    ConstantFolder,
    FullChoiceOptimizer,
    SizeConstraintFolder,
)
from cofola.parser.parser import parse
from tests.helpers import _ref_named


class TestPlanningOptimizationPasses(object):
    """Optimization and folding passes before lowering."""

    def test_full_choice_optimizer_defaults_unsized_ordered_choose_to_source_size(self) -> None:
        """An unconstrained choose_tuple(B) defaults to choosing the whole bag."""
        problem = parse("""
S = bag(A: 5, B: 4, C: 2)
T = choose_tuple(S)
""")
        tuple_ref = _ref_named(problem, "T")

        result = FullChoiceOptimizer().run(problem, AnalysisManager(problem))
        tuple_defn = result.get_object(tuple_ref)

        assert isinstance(tuple_defn, TupleDef)
        assert tuple_defn.choose is False
        assert tuple_defn.size == 11


    def test_full_choice_optimizer_aliases_full_set_and_bag_choose(self) -> None:
        """A full-size choose over a set or bag is just its source object."""
        problem = parse("""
S = set(a, b, c)
T = choose(S, 3)
B = bag(a: 2, b: 1)
C = choose(B, 3)
""")
        source_set = _ref_named(problem, "S")
        source_bag = _ref_named(problem, "B")
        chosen_set = _ref_named(problem, "T")
        chosen_bag = _ref_named(problem, "C")

        result = FullChoiceOptimizer().run(problem, AnalysisManager(problem))

        assert result.get_object(chosen_set) is None
        assert result.get_object(chosen_bag) is None
        assert _ref_named(result, "T") == source_set
        assert _ref_named(result, "C") == source_bag
        assert not any(isinstance(defn, (SetChoose, BagChoose)) for _, defn in result.defs)


    def test_full_choice_optimizer_keeps_variable_unsized_choose(self) -> None:
        """Plain choose(S) remains a variable subset unless size is known."""
        problem = parse("""
S = set(a, b, c)
T = choose(S)
""")
        chosen = _ref_named(problem, "T")

        result = FullChoiceOptimizer().run(problem, AnalysisManager(problem))

        assert result.get_object(chosen) == SetChoose(
            source=_ref_named(problem, "S"),
            size=None,
        )


    def test_constant_folder_returns_same_problem_when_unchanged(self) -> None:
        """No-op constant folding should not invalidate analyses by identity churn."""
        ref = ObjRef(0)
        problem = Problem(
            defs=((ref, SetInit(entities=frozenset({Entity("a")}))),),
            constraints=(),
            names=((ref, "A"),),
            locs=((ref, (1, 1)),),
        )

        result = ConstantFolder().run(problem)

        assert result is problem


    def test_constant_folder_fixed_point_is_owned_by_pass_runner(self) -> None:
        """ConstantFolder performs one step; FixedPointPass owns convergence."""
        a_ref = ObjRef(0)
        b_ref = ObjRef(1)
        inner_ref = ObjRef(2)
        outer_ref = ObjRef(3)
        problem = Problem(
            defs=(
                (a_ref, SetInit(entities=frozenset({Entity("a")}))),
                (b_ref, SetInit(entities=frozenset({Entity("b")}))),
                (inner_ref, SetUnion(left=a_ref, right=b_ref)),
                (outer_ref, SetUnion(left=inner_ref, right=b_ref)),
            ),
            constraints=(),
            names=(),
        )

        one_step = ConstantFolder().run(problem)
        assert isinstance(one_step.get_object(outer_ref), SetUnion)

        am = PlaningPipeline.run_passes(problem, [FixedPointPass(ConstantFolder)])
        assert am.problem.get_object(outer_ref) == SetInit(
            entities=frozenset({Entity("a"), Entity("b")})
        )


    def test_size_constraint_folder_substitutes_known_terms(self) -> None:
        """Exact terms should be removed from mixed SizeConstraints."""
        problem = parse("""
S = set(a, b)
T = choose(S)
|S| + |T| == 3
""")

        result = SizeConstraintFolder().run(problem)

        assert result.constraints == (
            SizeConstraint(
                terms=((_ref_named(problem, "T"), 1),),
                comparator="==",
                rhs=1,
            ),
        )


    def test_size_constraint_folder_embeds_dropped_choose_size(self) -> None:
        """Dropped true constraints should preserve choose size on the def."""
        problem = parse("""
S = set(a, b, c)
T = choose(S)
|T| == 2
""")
        source = _ref_named(problem, "S")
        chosen = _ref_named(problem, "T")

        result = SizeConstraintFolder().run(problem)

        assert result.constraints == ()
        assert result.get_object(chosen) == SetChoose(source=source, size=2)


    def test_size_constraint_folder_raises_when_analysis_is_unsat(self) -> None:
        """Direct pass use should surface contradictory analysis facts."""
        problem = parse("""
S = set(a, b)
|S| <= 1
""")

        with pytest.raises(UnsatisfiableConstraint):
            SizeConstraintFolder().run(problem)
