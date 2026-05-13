"""Tests for planning-layer utilities and analysis policies."""
from __future__ import annotations

import pytest

from cofola.frontend import (
    BagChoose,
    BagCountAtom,
    BagInit,
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
    ProblemBuilder,
    SequenceDef,
    SetPartDef,
    SetChoose,
    SetInit,
    SetIntersection,
    SetUnion,
    SizeConstraint,
    TupleCountAtom,
)
from cofola.frontend.utils import constraint_refs
from cofola.frontend.utils import object_refs
from cofola.planing.analysis.bag_classify import BagClassification
from cofola.planing.analysis.entities import EntityAnalysis
from cofola.planing.analysis.merged import MergedAnalysis
from cofola.planing.analysis.max_size import MaxSizeInference
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.pass_manager import AnalysisManager
from cofola.planing.pass_manager import RefAllocator
from cofola.planing.pass_manager import UnsatisfiableConstraint
from cofola.planing.pipeline import PlaningPipeline
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import ConstantFolder, SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass
from cofola.parser.parser import parse


def _ref_named(problem: Problem, name: str) -> ObjRef:
    for ref, candidate in problem.names:
        if candidate == name:
            return ref
    raise AssertionError(f"missing ref named {name!r}")


def _part_ref(problem: Problem, partition: ObjRef, index: int) -> ObjRef:
    for ref, defn in problem.defs:
        if isinstance(defn, SetPartDef) and defn.partition == partition and defn.index == index:
            return ref
    raise AssertionError(f"missing part {partition.id}[{index}]")


def _first_def_ref(problem: Problem, cls: type) -> ObjRef:
    for ref, defn in problem.defs:
        if isinstance(defn, cls):
            return ref
    raise AssertionError(f"missing def of type {cls.__name__}")


def _size_constraint_for_ref(problem: Problem, target: ObjRef) -> SizeConstraint:
    for constraint in problem.constraints:
        if not isinstance(constraint, SizeConstraint):
            continue
        if any(term == target for term, _coef in constraint.terms):
            return constraint
    raise AssertionError(f"missing SizeConstraint for ref {target.id}")


def test_ref_allocator_starts_after_existing_refs() -> None:
    """Generated refs should not depend on a magic-number id range."""
    problem = Problem(
        defs=((ObjRef(10000), SetInit(entities=frozenset({Entity("a")}))),),
        constraints=(),
        names=(),
    )

    allocator = RefAllocator(problem)

    assert allocator.new_ref() == ObjRef(10001)
    assert allocator.new_ref() == ObjRef(10002)


def test_constraint_refs_recurses_into_size_atoms() -> None:
    """Shared ref walking should see refs nested inside size atoms."""
    bag = ObjRef(1)
    constraint = SizeConstraint(
        terms=((BagCountAtom(bag=bag, entity=Entity("a")), 1),),
        comparator="==",
        rhs=1,
    )

    assert constraint_refs(constraint) == [bag]


def test_max_size_inference_skips_size_atom_constraints() -> None:
    """LP inference only reasons about raw object-cardinality terms."""
    builder = ProblemBuilder()
    a = Entity("a")
    source = builder.add(BagInit(entity_multiplicity=((a, 2),)), name="B")
    chosen = builder.add(BagChoose(source=source), name="C")
    builder.add_constraint(
        SizeConstraint(
            terms=((BagCountAtom(bag=chosen, entity=a), 1),),
            comparator="==",
            rhs=1,
        )
    )
    problem = builder.build()

    result = AnalysisManager(problem).get(MaxSizeInference)

    assert result.exact_sizes == {}
    assert result.max_sizes == {}
    assert not result.unsatisfiable


def test_max_size_inference_keeps_raw_ref_constraints() -> None:
    """Raw ObjRef size constraints remain LP-compatible."""
    builder = ProblemBuilder()
    source = builder.add(
        SetInit(entities=frozenset({Entity("a"), Entity("b"), Entity("c")})),
        name="S",
    )
    chosen = builder.add(SetChoose(source=source), name="T")
    builder.add_constraint(
        SizeConstraint(terms=((chosen, 1),), comparator="==", rhs=2)
    )
    problem = builder.build()

    result = AnalysisManager(problem).get(MaxSizeInference)

    assert result.exact_sizes[chosen] == 2


def test_max_size_inference_uses_entity_capacity_bounds() -> None:
    """LP inference should reject constraints exceeding known object capacity."""
    problem = parse("""
S = set(a, b)
T = choose(S)
|T| >= 3
""")

    result = AnalysisManager(problem).get(MaxSizeInference)

    assert result.unsatisfiable


def test_max_size_inference_rejects_exact_size_above_capacity() -> None:
    """Known exact objects should be fixed by LP bounds."""
    problem = parse("""
S = set(a, b)
|S| == 3
""")

    result = AnalysisManager(problem).get(MaxSizeInference)

    assert result.unsatisfiable


def test_max_size_inference_tightens_upper_bound_from_inequality() -> None:
    """Upper-bound constraints should refine max_sizes through bounded LP."""
    problem = parse("""
S = set(a, b, c)
T = choose(S)
|T| <= 1
""")
    chosen = _ref_named(problem, "T")

    result = AnalysisManager(problem).get(MaxSizeInference)

    assert not result.unsatisfiable
    assert result.max_sizes[chosen] == 1


def test_merged_analysis_rejects_exact_size_above_tightened_max() -> None:
    """Merging LP max bounds must not leave exact_size > max_size."""
    problem = parse("""
S = set(a, b)
|S| <= 1
""")

    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert analysis.unsatisfiable


def test_merged_analysis_does_not_mutate_entity_analysis_cache() -> None:
    """MergedAnalysis should refine a deep copy of EntityAnalysis facts."""
    problem = parse("""
S = set(a, b, c)
T = choose(S)
|T| <= 1
""")
    chosen = _ref_named(problem, "T")
    am = AnalysisManager(problem)

    base = am.get(EntityAnalysis)
    merged = am.get(MergedAnalysis)

    assert base.set_info[chosen].max_size == 3
    assert merged.set_info[chosen].max_size == 1


def test_constant_folder_returns_same_problem_when_unchanged() -> None:
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


def test_constant_folder_fixed_point_is_owned_by_pass_runner() -> None:
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


def test_size_constraint_folder_substitutes_known_terms() -> None:
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


def test_size_constraint_folder_embeds_dropped_choose_size() -> None:
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


def test_size_constraint_folder_raises_when_analysis_is_unsat() -> None:
    """Direct pass use should surface contradictory analysis facts."""
    problem = parse("""
S = set(a, b)
|S| <= 1
""")

    with pytest.raises(UnsatisfiableConstraint):
        SizeConstraintFolder().run(problem)


def test_lowering_expands_for_all_parts() -> None:
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


def test_lowering_rejects_for_all_parts_placeholder_partition_mismatch() -> None:
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


def test_lowering_rejects_for_all_parts_placeholder_escape() -> None:
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


def test_lowering_bag_like_part_source_reports_invalid_partition() -> None:
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


def test_lowering_choose_sequence_uses_inferred_size() -> None:
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


def test_lowering_injective_function_uses_symbolic_domain_size_when_variable() -> None:
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


def test_lowering_injective_function_uses_exact_domain_size_when_known() -> None:
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


def test_lowering_tuple_count_atom_to_inverse_image() -> None:
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


def test_lowering_tuple_dedup_count_full_tuple_to_codomain_intersection() -> None:
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


def test_lowering_tuple_dedup_count_choose_tuple_to_image_intersection() -> None:
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


def test_set_choose_size_larger_than_source_is_unsatisfiable() -> None:
    """A no-replacement set choice cannot request more elements than its source."""
    problem = parse("""
S = set(a, b)
T = choose(S, 3)
""")

    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert analysis.unsatisfiable


def test_bag_choose_size_larger_than_source_is_unsatisfiable() -> None:
    """A no-replacement bag choice cannot request more items than source capacity."""
    problem = parse("""
B = bag(a: 2)
C = choose(B, 3)
""")

    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert analysis.unsatisfiable


def test_entity_analysis_set_difference_keeps_conservative_capacity() -> None:
    """A disjoint RHS does not shrink the possible size of a set difference."""
    problem = parse("""
A = set(a)
B = set(b)
C = A - B
""")

    analysis = AnalysisManager(problem).get(EntityAnalysis)

    assert analysis.set_info[_ref_named(problem, "C")].max_size == 1


def test_entity_analysis_bag_support_caps_by_distinct_entities() -> None:
    """Support size is bounded by distinct entities, not total multiplicity."""
    problem = parse("""
B = bag(a: 2, b: 2)
S = supp(B)
T = choose(S, 3)
""")

    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert analysis.set_info[_ref_named(problem, "S")].max_size == 2
    assert analysis.unsatisfiable


def test_ordered_collection_size_larger_than_source_is_unsatisfiable() -> None:
    """A no-replacement ordered choice cannot exceed source capacity."""
    problem = parse("""
S = set(a, b)
T = choose_tuple(S, 3)
""")

    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert analysis.unsatisfiable


def test_entity_analysis_infos_satisfy_size_invariants() -> None:
    """EntityAnalysis should centrally normalize SetInfo and BagInfo facts."""
    problem = parse("""
A = set(a, b)
B = set(b, c)
C = A + B
M = bag(a: 2, b: 2)
S = supp(M)
T = choose_tuple(A, 3)
""")

    analysis = AnalysisManager(problem).get(EntityAnalysis)

    assert analysis.unsatisfiable
    for info in analysis.set_info.values():
        assert 0 <= info.max_size <= len(info.p_entities)
        if info.exact_size is not None:
            assert 0 <= info.exact_size <= info.max_size
    for info in analysis.bag_info.values():
        assert all(mult >= 0 for mult in info.p_entities_multiplicity.values())
        assert 0 <= info.max_size <= sum(info.p_entities_multiplicity.values())
        if info.exact_size is not None:
            assert 0 <= info.exact_size <= info.max_size


def test_entity_analysis_rejects_negative_bag_multiplicity() -> None:
    """Malformed public-builder input should not create negative BagInfo facts."""
    a = Entity("a")
    builder = ProblemBuilder()
    bag = builder.add(BagInit(entity_multiplicity=((a, -1),)), name="B")
    problem = builder.build()

    analysis = AnalysisManager(problem).get(EntityAnalysis)

    assert analysis.unsatisfiable
    assert analysis.bag_info[bag].p_entities_multiplicity[a] == 0
    assert analysis.bag_info[bag].max_size == 0
    assert analysis.bag_info[bag].exact_size == 0


def test_entity_analysis_reports_invalid_part_partition_ref() -> None:
    """Malformed hand-built PartDef refs should fail with a controlled error."""
    builder = ProblemBuilder()
    builder.add(SetPartDef(partition=ObjRef(999), index=0), name="bad")
    problem = builder.build()

    with pytest.raises(ValueError, match="partition ref=999"):
        AnalysisManager(problem).get(EntityAnalysis)


def test_bag_classification_reports_invalid_part_partition_ref() -> None:
    """The public bag-classification boundary should not leak AttributeError."""
    builder = ProblemBuilder()
    builder.add(SetPartDef(partition=ObjRef(999), index=0), name="bad")
    problem = builder.build()

    with pytest.raises(ValueError, match="partition ref=999"):
        AnalysisManager(problem).get(BagClassification)


def _assert_problem_refs_are_defined(problem: Problem) -> None:
    defined = set(problem.refs())
    for _ref, defn in problem.iter_objects():
        assert set(object_refs(defn)) <= defined
    for constraint in problem.constraints:
        assert set(constraint_refs(constraint)) <= defined


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
def test_transform_pass_preserves_defined_ref_invariant(pass_spec, program: str) -> None:
    """Every transform pass should leave object and constraint refs defined."""
    problem = parse(program)

    am = PlaningPipeline.run_passes(problem, [pass_spec])

    _assert_problem_refs_are_defined(am.problem)
