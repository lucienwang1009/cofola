"""Executable examples for the public ``ProblemBuilder`` API."""
from __future__ import annotations

import pytest
import cofola

from cofola.frontend import (
    BagChoose,
    BagCountAtom,
    BagInit,
    CofolaTypeError,
    Entity,
    FuncDef,
    MembershipConstraint,
    ObjRef,
    PartPlaceholderDef,
    PartitionDef,
    ProblemBuilder,
    SetPartDef,
    SetChoose,
    SetInit,
    SizeConstraint,
    TupleDef,
    TupleIndexEq,
    validate_problem,
)
from cofola.solver import solve


def test_problem_builder_set_example_solves() -> None:
    """Build and solve the set example from the usage guide."""
    builder = ProblemBuilder()

    a = Entity("a")
    b = Entity("b")
    c = Entity("c")

    source = builder.add(SetInit(entities=frozenset({a, b, c})), name="S")
    chosen = builder.add(SetChoose(source=source, size=2), name="T")

    builder.add_constraint(
        SizeConstraint(
            terms=((chosen, 1),),
            comparator="==",
            rhs=2,
        )
    )
    builder.add_constraint(
        MembershipConstraint(
            entity=a,
            container=chosen,
            positive=True,
        )
    )

    problem = builder.build()
    validate_problem(problem)

    assert solve(problem, validate=False) == 2


def test_problem_builder_effective_object_resolves_partition_parts() -> None:
    """Effective lookup should treat parts as their partition source kind."""
    builder = ProblemBuilder()
    source = builder.add(SetInit(entities=frozenset({Entity("a")})), name="S")
    partition = builder.add(PartitionDef(source=source, num_parts=1), name="P")
    part = builder.add(SetPartDef(partition=partition, index=0), name="P0")
    placeholder = builder.add(PartPlaceholderDef(partition=partition))

    assert builder.get_effective_object(part) == builder.get_object(source)
    assert builder.get_effective_object(placeholder) == builder.get_object(source)

    problem = builder.build()
    assert problem.get_effective_object(part) == problem.get_object(source)
    assert problem.get_effective_object(placeholder) == problem.get_object(source)


def test_problem_builder_bag_count_example_solves() -> None:
    """Build and solve the bag multiplicity example from the usage guide."""
    builder = ProblemBuilder()

    a = Entity("a")
    b = Entity("b")

    source = builder.add(
        BagInit(entity_multiplicity=((a, 2), (b, 1))),
        name="B",
    )
    chosen = builder.add(BagChoose(source=source), name="C")

    builder.add_constraint(
        SizeConstraint(
            terms=((BagCountAtom(bag=chosen, entity=a), 1),),
            comparator="==",
            rhs=1,
        )
    )

    problem = builder.build()
    validate_problem(problem)

    assert solve(problem, validate=False) == 2


def test_validation_helpers_are_public_api() -> None:
    """Validation helpers should be available from the top-level package."""
    assert cofola.validate_problem is validate_problem
    assert cofola.CofolaTypeError is CofolaTypeError


def test_problem_builder_tuple_index_example_type_checks() -> None:
    """Build and validate the tuple indexing example from the usage guide."""
    builder = ProblemBuilder()

    a = Entity("a")
    b = Entity("b")

    source = builder.add(SetInit(entities=frozenset({a, b})), name="S")
    tuple_ref = builder.add(TupleDef(source=source, choose=True, size=2), name="T")

    builder.add_constraint(
        TupleIndexEq(
            tuple_ref=tuple_ref,
            index=0,
            entity=a,
            positive=True,
        )
    )

    problem = builder.build()

    validate_problem(problem)


def test_validate_problem_rejects_unknown_size_constraint_ref() -> None:
    """Public validation should catch dangling refs in hand-built problems."""
    builder = ProblemBuilder()
    builder.add_constraint(
        SizeConstraint(terms=((ObjRef(999), 1),), comparator="==", rhs=1)
    )

    with pytest.raises(CofolaTypeError, match="unknown object"):
        validate_problem(builder.build())


def test_validate_problem_rejects_function_size_terms() -> None:
    """Function refs should not be accepted as raw cardinality terms."""
    builder = ProblemBuilder()
    domain = builder.add(SetInit(entities=frozenset({Entity("a")})), name="A")
    codomain = builder.add(SetInit(entities=frozenset({Entity("b")})), name="B")
    func = builder.add(FuncDef(domain=domain, codomain=codomain), name="f")
    builder.add_constraint(SizeConstraint(terms=((func, 1),), comparator="==", rhs=1))

    with pytest.raises(CofolaTypeError, match="cardinality object"):
        validate_problem(builder.build())


def test_solve_validates_hand_built_problem_by_default() -> None:
    """solve() is also a validation boundary for public Problem inputs."""
    builder = ProblemBuilder()
    builder.add_constraint(
        SizeConstraint(terms=((ObjRef(999), 1),), comparator="==", rhs=1)
    )

    with pytest.raises(CofolaTypeError, match="unknown object"):
        solve(builder.build())

