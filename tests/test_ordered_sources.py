"""Regression tests for full ordered objects and dynamic choice sources."""
from __future__ import annotations

from itertools import product
from math import factorial, prod

import pytest

from cofola.frontend import (
    ObjRef,
    Problem,
    SequenceDef,
    SetChooseReplace,
    SetInit,
    SizeConstraint,
    TupleDef,
)
from cofola.parser.parser import parse
from cofola.planing.analysis.entities import EntityAnalysis
from cofola.planing.analysis.merged import MergedAnalysis
from cofola.planing.pass_manager import AnalysisManager
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


def _ref_named(problem: Problem, name: str) -> ObjRef:
    return next(ref for ref, candidate in problem.names if candidate == name)


def _source_size_equalities(constraints: tuple[object, ...], ref: ObjRef) -> set[int]:
    return {
        constraint.rhs
        for constraint in constraints
        if isinstance(constraint, SizeConstraint)
        and constraint.terms == ((ref, 1),)
        and constraint.comparator == "=="
    }


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("U = set(a, b)\nS = choose(U)\nrow = sequence(S)\n", 5),
        ("U = set(a, b, c, d)\nS = choose(U)\n|S| == 2\nrow = sequence(S)\n", 12),
        ("S = set(a, b)\nP = compose(S, 2)\nT = tuple(P[0])\n", 5),
        ("S = set(a, b, c)\nC = choose(S, 2)\nQ = choose_replace_sequence(C, 3)\n", 24),
        ("S = set(a, b, c)\nC = choose(S, 2)\nQ = choose_replace_sequence(C, 4)\n", 48),
        ("B = bag(a: 2, b: 1)\nC = choose(B)\n|C| == 2\nrow = sequence(C)\n", 3),
        ("B = bag(a: 2, b: 1)\nC = choose(B)\nrow = sequence(C)\n", 9),
        ("S = set(a, b)\nC = choose(S)\n|C| == 0\nrow = sequence(C)\n", 1),
    ],
)
def test_ordered_objects_respect_dynamic_sources(program: str, expected: int) -> None:
    assert parse_and_solve(program) == expected


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
@pytest.mark.parametrize("multiplicities", [(1, 1), (2, 1), (2, 1, 1), (2, 2)])
@pytest.mark.parametrize("size", [None, 0, 2])
def test_ordered_bag_sources_match_subbag_enumeration(
    constructor: str, multiplicities: tuple[int, ...], size: int | None,
) -> None:
    """Sum distinct permutations of each possible sub-bag independently."""
    expected = sum(
        factorial(sum(counts)) // prod(factorial(count) for count in counts)
        for counts in product(*(range(multiplicity + 1) for multiplicity in multiplicities))
        if size is None or sum(counts) == size
    )
    contents = ", ".join(
        f"e{index}: {multiplicity}" for index, multiplicity in enumerate(multiplicities)
    )
    constraint = "" if size is None else f"|C| == {size}\n"

    assert parse_and_solve(
        f"B = bag({contents})\nC = choose(B)\nrow = {constructor}(C)\n{constraint}"
    ) == expected


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
@pytest.mark.parametrize(("singleton_count", "expected"), [(0, 3), (1, 6)])
def test_ordered_bag_source_preserves_singleton_count_constraints(
    constructor: str, singleton_count: int, expected: int,
) -> None:
    """The source count and ordered encoding must share a bound singleton count."""
    assert parse_and_solve(
        f"B = bag(a: 2, b: 1)\nC = choose(B)\nrow = {constructor}(C)\n"
        + f"C.count(b) == {singleton_count}\n" * 2
    ) == expected


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
def test_ordered_bag_part_keeps_existing_singleton_weights(constructor: str) -> None:
    # Each sub-bag in P[0] determines P[1]; its distinct permutations contribute
    # 1 + 2 + 3 + 3 solutions over sizes zero through three.
    assert parse_and_solve(
        f"B = bag(a: 2, b: 1)\nP = compose(B, 2)\nrow = {constructor}(P[0])\n"
    ) == 9


@pytest.mark.parametrize("size", [0, 1, 3])
def test_replacement_sequence_respects_variable_set_membership(size: int) -> None:
    # Each of the four subsets contributes |C|**size sequences (including
    # one empty sequence for an empty C when size == 0).
    expected = sum(cardinality**size for cardinality in (0, 1, 1, 2))
    assert parse_and_solve(
        "U = set(a, b)\nC = choose(U)\n"
        f"row = choose_replace_sequence(C, {size})\n"
    ) == expected


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
@pytest.mark.parametrize("constrained", ["C", "row"])
def test_full_ordered_exact_size_propagates_both_ways(
    constructor: str, constrained: str,
) -> None:
    problem = parse(
        f"U = set(a, b, c, d)\nC = choose(U)\nrow = {constructor}(C)\n"
        f"|{constrained}| == 2\n"
    )
    analysis = AnalysisManager(problem).get(MergedAnalysis)

    assert not analysis.unsatisfiable
    for name in ("C", "row"):
        assert analysis.set_info[_ref_named(problem, name)].exact_size == 2


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
def test_full_ordered_size_bounds_keep_bag_multiplicities_tight(constructor: str) -> None:
    problem = parse(
        "B = bag(a: 100, b: 100)\nC = choose(B)\n"
        f"row = {constructor}(C)\n|row| <= 2\n"
    )
    manager = AnalysisManager(problem)
    base = manager.get(EntityAnalysis)
    analysis = manager.get(MergedAnalysis)

    assert not analysis.unsatisfiable
    for name in ("C", "row"):
        ref = _ref_named(problem, name)
        assert base.bag_info[ref].max_size == 200
        assert analysis.bag_info[ref].max_size == 2
        assert set(analysis.bag_info[ref].p_entities_multiplicity.values()) == {2}
    assert analysis.bag_info[_ref_named(problem, "B")].max_size == 200


@pytest.mark.parametrize("constructor", ["tuple", "sequence"])
@pytest.mark.parametrize("ordered_bound", ["== 1", "<= 1"])
def test_full_ordered_rejects_conflicting_source_size(
    constructor: str, ordered_bound: str,
) -> None:
    program = (
        f"U = set(a, b, c)\nC = choose(U)\nrow = {constructor}(C)\n"
        f"|C| == 2\n|row| {ordered_bound}\n"
    )
    assert AnalysisManager(parse(program)).get(MergedAnalysis).unsatisfiable
    assert parse_and_solve(program) == 0


@pytest.mark.parametrize(
    "constructor", ["choose_tuple", "choose_sequence", "choose_replace_sequence"]
)
def test_chosen_ordered_objects_do_not_fix_their_source_size(constructor: str) -> None:
    program = (
        f"U = set(a, b, c)\nC = choose(U)\nrow = {constructor}(C, 1)\n"
    )
    problem = parse(program)
    analysis = AnalysisManager(problem).get(MergedAnalysis)
    source = analysis.set_info[_ref_named(problem, "C")]

    assert not analysis.unsatisfiable
    assert source.max_size == 3
    assert source.exact_size is None
    # Three possible ordered elements, each contained in four source subsets.
    assert parse_and_solve(program) == 12


def test_decomposed_sequence_size_branches_also_constrain_source() -> None:
    problem = parse("S = set(a, b)\nP = compose(S, 2)\nT = sequence(P[0])\n")
    ordered = problem.get_object(_ref_named(problem, "T"))
    assert isinstance(ordered, SequenceDef)

    schedule = PlaningPipeline().process(problem)

    assert len(schedule.branches) == 3
    branch_sizes = []
    for branch in schedule.branches:
        constraints = tuple(
            constraint
            for component, _analysis in branch.components
            for constraint in component.constraints
        )
        branch_sizes.append(_source_size_equalities(constraints, ordered.source))
    assert sorted(branch_sizes, key=lambda sizes: tuple(sorted(sizes))) == [{0}, {1}, {2}]


def test_fixed_tuple_size_materializes_source_size_before_lowering() -> None:
    problem = parse(
        "S = set(a, b)\nP = compose(S, 2)\nT = tuple(P[0])\n|T| == 1\n"
    )
    ordered = problem.get_object(_ref_named(problem, "T"))
    assert isinstance(ordered, TupleDef)

    schedule = PlaningPipeline().process(problem)

    assert len(schedule.branches) == 1
    constraints = tuple(
        constraint
        for component, _analysis in schedule.branches[0].components
        for constraint in component.constraints
    )
    assert _source_size_equalities(constraints, ordered.source) == {1}


def test_replacement_sequence_from_derived_set_uses_bag_choice() -> None:
    problem = parse(
        "S = set(a, b, c)\nC = choose(S, 2)\nQ = choose_replace_sequence(C, 4)\n"
    )

    result = LoweringPass().run(problem, AnalysisManager(problem)).problem
    sequence = result.get_object(_ref_named(problem, "Q"))

    assert isinstance(sequence, SequenceDef)
    assert not sequence.choose
    assert not sequence.replace
    assert sequence.size == 4
    assert result.get_object(sequence.source) == SetChooseReplace(
        source=_ref_named(problem, "C"), size=4,
    )


def test_replacement_sequence_from_fixed_set_keeps_direct_flatten() -> None:
    problem = parse("S = set(a, b, c)\nQ = choose_replace_sequence(S, 4)\n")

    result = LoweringPass().run(problem, AnalysisManager(problem)).problem
    sequence = result.get_object(_ref_named(problem, "Q"))

    assert isinstance(sequence, SequenceDef)
    assert sequence.choose
    assert sequence.replace
    assert sequence.flatten is not None
    assert isinstance(result.get_object(sequence.flatten), SetInit)
