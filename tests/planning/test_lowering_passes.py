"""Planning lowering passes: tuple/partition/function/linear-def lowering and for-all expansion."""
from __future__ import annotations

import pytest

from cofola.frontend import (
    BagChoose,
    BagCountAtom,
    Entity,
    ForAllParts,
    FuncDef,
    FuncImage,
    FuncInverseImage,
    MembershipConstraint,
    ObjRef,
    PartPlaceholderDef,
    PartitionDef,
    Problem,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetInit,
    SetIntersection,
    SetPartDef,
    SizeConstraint,
    TupleCountAtom,
)
from cofola.planing.pass_manager import (
    AnalysisManager,
    FixedPointPass,
)
from cofola.planing.pipeline import (
    PlaningPipeline,
    PlanningProfile,
)
from cofola.planing.passes.lowering import (
    ForAllPartsExpansionStep,
    InjectiveFunctionLoweringStep,
    LinearDefLoweringStep,
    LoweringPass,
    TupleCountAtomLoweringStep,
    TupleDefLoweringStep,
)
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import FullChoiceOptimizer
from cofola.parser.parser import parse
from tests.helpers import (
    _first_def_ref,
    _part_ref,
    _ref_named,
    _size_constraint_for_ref,
)


class TestPlanningLoweringPasses(object):
    """Lowering passes for forall parts, tuples, sequences, and functions."""

    def test_lowering_expands_for_all_parts(self) -> None:
        """ForAllParts should become one concrete constraint per real part."""
        problem = parse("""
S = set(a, b)
P = partition(S, 2)
|p| == 1 for p in P
""")

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        partition = _ref_named(problem, "P")

        assert not any(isinstance(c, ForAllParts) for c in result.constraints)
        assert result.constraints == (
            SizeConstraint(terms=((_part_ref(problem, partition, 0), 1),), comparator="==", rhs=1),
            SizeConstraint(terms=((_part_ref(problem, partition, 1), 1),), comparator="==", rhs=1),
        )


    def test_lowering_steps_are_grouped_under_one_fixed_point_driver(self) -> None:
        """Split lowering steps should preserve the original combined fixpoint."""

        assert LoweringPass.STEP_CLASSES == (
            ForAllPartsExpansionStep,
            TupleDefLoweringStep,
            LinearDefLoweringStep,
            InjectiveFunctionLoweringStep,
            TupleCountAtomLoweringStep,
        )


    def test_empty_planning_profile_returns_intact_problem(self) -> None:
        """No profile means no planner transformations or decomposition."""

        problem = parse("""
S = set(a, b)
T = choose(S)
|T| == 1
""")

        implicit_schedule = PlaningPipeline().process(problem)
        explicit_schedule = PlaningPipeline(PlanningProfile()).process(problem)

        for schedule in (implicit_schedule, explicit_schedule):
            assert len(schedule.branches) == 1
            branch = schedule.branches[0]
            assert len(branch.components) == 1
            planned_problem, _analysis = branch.components[0]
            assert planned_problem is problem


    def test_lowering_split_steps_share_tuple_state_across_iterations(self) -> None:
        """Tuple-count lowering depends on metadata from an earlier tuple step."""

        problem = parse("""
S = set(a, b)
T = tuple(S)
T.count(a) == 1
""")
        lowering = LoweringPass()
        am = AnalysisManager(problem)

        first = lowering.run(am.problem, am)
        assert first.changed
        am.update(first.problem)

        second = lowering.run(am.problem, am)
        assert second.changed
        assert not any(
            isinstance(atom, TupleCountAtom)
            for constraint in second.problem.constraints
            if isinstance(constraint, SizeConstraint)
            for atom, _coef in constraint.terms
        )


    def test_lowering_tuple_membership_uses_tuple_image(self) -> None:
        """Tuple membership should lower through the tuple function image."""
        problem = parse("""
S = set(a, b)
T = tuple(S)
a in T
b not in T
""")

        am = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)])
        image_ref = _first_def_ref(am.problem, FuncImage)

        assert not any(
            isinstance(atom, TupleCountAtom)
            for constraint in am.problem.constraints
            if isinstance(constraint, SizeConstraint)
            for atom, _coef in constraint.terms
        )
        assert am.problem.constraints[-2:] == (
            MembershipConstraint(entity=Entity("a"), container=image_ref, positive=True),
            MembershipConstraint(entity=Entity("b"), container=image_ref, positive=False),
        )


    def test_lowering_bag_tuple_membership_uses_inverse_image_size(self) -> None:
        """Bag tuple membership should reuse inverse images instead of tuple images."""
        problem = parse("""
B = bag(a: 2, b: 1)
T = choose_tuple(B, 2)
a in T
b not in T
""")

        am = PlaningPipeline.run_passes(
            problem,
            [FixedPointPass(LoweringPass), MergeIdenticalObjects],
        )
        inverse_images = {
            defn.argument: ref
            for ref, defn in am.problem.defs
            if isinstance(defn, FuncInverseImage)
        }

        assert Entity("a") in inverse_images
        assert Entity("b") in inverse_images
        assert not any(isinstance(defn, FuncImage) for _, defn in am.problem.defs)
        assert SizeConstraint(
            terms=((inverse_images[Entity("a")], 1),),
            comparator=">",
            rhs=0,
        ) in am.problem.constraints
        assert SizeConstraint(
            terms=((inverse_images[Entity("b")], 1),),
            comparator="==",
            rhs=0,
        ) in am.problem.constraints


    def test_lowering_rejects_for_all_parts_placeholder_partition_mismatch(self) -> None:
        """Malformed builder input should not silently rewrite with the wrong part."""
        a = Entity("a")
        source = ObjRef(0)
        partition = ObjRef(1)
        part = ObjRef(2)
        other_partition = ObjRef(3)
        placeholder = ObjRef(4)
        problem = Problem(
            defs=(
                (source, SetInit(entities=frozenset({a}))),
                (partition, PartitionDef(source=source, num_parts=1)),
                (part, SetPartDef(partition=partition, index=0)),
                (other_partition, PartitionDef(source=source, num_parts=1)),
                (placeholder, PartPlaceholderDef(partition=other_partition)),
            ),
            constraints=(
                ForAllParts(
                    constraint_template=MembershipConstraint(entity=a, container=placeholder),
                    partition=partition,
                    part_ref=placeholder,
                ),
            ),
            names=(),
        )

        with pytest.raises(ValueError, match="belongs to partition"):
            LoweringPass().run(problem, AnalysisManager(problem))


    def test_lowering_rejects_for_all_parts_placeholder_escape(self) -> None:
        """Placeholder refs outside the forall template would become dangling refs."""
        a = Entity("a")
        source = ObjRef(0)
        partition = ObjRef(1)
        part = ObjRef(2)
        placeholder = ObjRef(3)
        problem = Problem(
            defs=(
                (source, SetInit(entities=frozenset({a}))),
                (partition, PartitionDef(source=source, num_parts=1)),
                (part, SetPartDef(partition=partition, index=0)),
                (placeholder, PartPlaceholderDef(partition=partition)),
            ),
            constraints=(
                MembershipConstraint(entity=a, container=placeholder),
                ForAllParts(
                    constraint_template=MembershipConstraint(entity=a, container=placeholder),
                    partition=partition,
                    part_ref=placeholder,
                ),
            ),
            names=(),
        )

        with pytest.raises(ValueError, match="escaped the forall template"):
            LoweringPass().run(problem, AnalysisManager(problem))


    def test_lowering_bag_like_part_source_reports_invalid_partition(self) -> None:
        """Malformed PartDef sources should fail explicitly in bag-like detection."""
        source = ObjRef(0)
        part = SetPartDef(partition=source, index=0)
        problem = Problem(
            defs=((source, SetInit(entities=frozenset({Entity("a")}))),),
            constraints=(),
            names=(),
        )

        with pytest.raises(ValueError, match="PartDef source references partition"):
            LoweringPass()._is_bag_like(part, problem)


    def test_lowering_choose_sequence_uses_inferred_size(self) -> None:
        """The inserted choose object should use analysis-resolved sequence size."""
        problem = parse("""
S = set(a, b, c)
Q = choose_sequence(S)
|Q| == 2
""")
        seq_ref = _ref_named(problem, "Q")

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        seq_defn = result.get_object(seq_ref)

        assert isinstance(seq_defn, SequenceDef)
        assert seq_defn.choose is False
        assert seq_defn.size == 2
        assert result.get_object(seq_defn.source) == SetChoose(
            source=_ref_named(problem, "S"),
            size=2,
        )


    def test_lowering_choose_replace_sequence_from_derived_set_uses_bag_choice(self) -> None:
        """Derived set sources must constrain repeated sequence entries to the source."""
        problem = parse("""
S = set(a, b, c, d, e, f)
C = choose(S, 5)
Q = choose_replace_sequence(C, 7)
""")
        seq_ref = _ref_named(problem, "Q")

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        seq_defn = result.get_object(seq_ref)

        assert isinstance(seq_defn, SequenceDef)
        assert seq_defn.choose is False
        assert seq_defn.replace is False
        assert seq_defn.size == 7
        assert result.get_object(seq_defn.source) == SetChooseReplace(
            source=_ref_named(problem, "C"),
            size=7,
        )


    def test_lowering_choose_replace_sequence_from_fixed_set_keeps_flatten(self) -> None:
        """Fixed set sources can use the simpler direct flatten encoding."""
        problem = parse("""
S = set(a, b, c)
Q = choose_replace_sequence(S, 4)
""")
        seq_ref = _ref_named(problem, "Q")

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        seq_defn = result.get_object(seq_ref)

        assert isinstance(seq_defn, SequenceDef)
        assert seq_defn.choose is True
        assert seq_defn.replace is True
        assert seq_defn.flatten is not None
        assert isinstance(result.get_object(seq_defn.flatten), SetInit)


    @pytest.mark.parametrize(
        "tuple_program",
        [
            "T = choose_tuple(S, 11)",
            "T = choose_tuple(S)\n|T| == 11",
        ],
    )
    def test_lowering_full_size_bag_choose_tuple_skips_redundant_bag_choose(self,
        tuple_program: str,
    ) -> None:
        """A proven full-source bag tuple choice is the source permutation."""
        problem = parse(f"""
S = bag(A: 5, B: 4, C: 2)
{tuple_program}
""")

        am = PlaningPipeline.run_passes(
            problem,
            [FixedPointPass(FullChoiceOptimizer), FixedPointPass(LoweringPass)],
        )

        assert not any(isinstance(defn, BagChoose) for _, defn in am.problem.defs)
        assert not any(
            isinstance(term, BagCountAtom)
            for constraint in am.problem.constraints
            if isinstance(constraint, SizeConstraint)
            for term, _coef in constraint.terms
        )


    def test_lowering_injective_function_uses_symbolic_domain_size_when_variable(self) -> None:
        """Injectivity should not turn a variable domain bound into an exact size."""
        source = ObjRef(0)
        domain = ObjRef(1)
        codomain = ObjRef(2)
        func = ObjRef(3)
        problem = Problem(
            defs=(
                (source, SetInit(entities=frozenset({Entity("a"), Entity("b"), Entity("c")}))),
                (domain, SetChoose(source=source)),
                (codomain, SetInit(entities=frozenset({Entity("x"), Entity("y"), Entity("z")}))),
                (func, FuncDef(domain=domain, codomain=codomain, injective=True)),
            ),
            constraints=(),
            names=(),
        )

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        image_ref = next(
            ref for ref, defn in result.defs
            if isinstance(defn, FuncImage) and defn.func == func
        )

        assert result.get_object(func) == FuncDef(
            domain=domain,
            codomain=codomain,
            injective=False,
        )
        assert result.constraints[-1] == SizeConstraint(
            terms=((image_ref, 1), (domain, -1)),
            comparator="==",
            rhs=0,
        )


    def test_lowering_injective_function_uses_exact_domain_size_when_known(self) -> None:
        """Known exact domain size can still become a constant injectivity constraint."""
        domain = ObjRef(0)
        codomain = ObjRef(1)
        func = ObjRef(2)
        problem = Problem(
            defs=(
                (domain, SetInit(entities=frozenset({Entity("a"), Entity("b")}))),
                (codomain, SetInit(entities=frozenset({Entity("x"), Entity("y")}))),
                (func, FuncDef(domain=domain, codomain=codomain, injective=True)),
            ),
            constraints=(),
            names=(),
        )

        result = LoweringPass().run(problem, AnalysisManager(problem)).problem
        image_ref = next(
            ref for ref, defn in result.defs
            if isinstance(defn, FuncImage) and defn.func == func
        )

        assert result.constraints[-1] == SizeConstraint(
            terms=((image_ref, 1),),
            comparator="==",
            rhs=2,
        )


    def test_lowering_tuple_count_atom_to_inverse_image(self) -> None:
        """Plain tuple counts should become FuncInverseImage cardinalities."""
        problem = parse("""
S = set(a, b)
T = tuple(S)
T.count(a) == 1
""")
        am = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)])
        inv_ref = _first_def_ref(am.problem, FuncInverseImage)

        assert am.problem.constraints[-1] == SizeConstraint(
            terms=((inv_ref, 1),),
            comparator="==",
            rhs=1,
        )


    def test_lowering_tuple_dedup_count_full_tuple_to_codomain_intersection(self) -> None:
        """Dedup count on full tuples should intersect the mapping codomain."""
        problem = parse("""
S = set(a, b, c)
A = set(a, b)
T = tuple(S)
T.dedup_count(A) == 2
""")
        am = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)])
        intersection_ref = _first_def_ref(am.problem, SetIntersection)
        intersection = am.problem.get_object(intersection_ref)

        assert isinstance(intersection, SetIntersection)
        assert intersection.right == _ref_named(problem, "A")
        assert _size_constraint_for_ref(am.problem, intersection_ref) == SizeConstraint(
            terms=((intersection_ref, 1),),
            comparator="==",
            rhs=2,
        )


    def test_lowering_tuple_dedup_count_choose_tuple_to_image_intersection(self) -> None:
        """Dedup count on choose tuples should first restrict to the tuple image."""
        problem = parse("""
S = set(a, b, c)
A = set(a, b)
T = choose_tuple(S, 2)
T.dedup_count(A) == 1
""")
        am = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)])

        assert any(isinstance(defn, FuncImage) for _, defn in am.problem.defs)
        intersection_ref = _first_def_ref(am.problem, SetIntersection)
        assert _size_constraint_for_ref(am.problem, intersection_ref) == SizeConstraint(
            terms=((intersection_ref, 1),),
            comparator="==",
            rhs=1,
        )
