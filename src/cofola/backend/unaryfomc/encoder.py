"""Translate the monadic set fragment of Cofola IR to UnaryFOMC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import factorial
from operator import eq, ge, gt, le, lt, ne
from typing import Callable

from cofola.frontend.constraints import (
    DisjointConstraint,
    EqualityConstraint,
    MembershipConstraint,
    SizeConstraint,
    SubsetConstraint,
)
from cofola.frontend.objects import (
    SetChoose,
    SetDifference,
    SetInit,
    SetIntersection,
    SetUnion,
)
from cofola.frontend.problem import Problem
from cofola.frontend.types import Entity, ObjRef
from cofola.ir.analysis.entities import AnalysisResult

__all__ = [
    "UnaryFOMCEncoding",
    "UnaryFOMCUnsupportedError",
    "encode_ir",
]


class UnaryFOMCUnsupportedError(ValueError):
    """Raised when a normalized Cofola problem leaves the supported overlap."""


@dataclass(frozen=True, slots=True)
class UnaryFOMCEncoding:
    """A UnaryFOMC sentence plus its finite Cofola decoding information."""

    sentence: str
    domain_size: int
    normalization_factor: int

    def decode_result(self, raw_count: object) -> int:
        """Remove the equivalent encodings of Cofola's fixed input sets."""

        integer_count = int(raw_count)
        if raw_count != integer_count:
            raise ArithmeticError(
                "UnaryFOMC returned a non-integral count for an unweighted encoding"
            )
        result, remainder = divmod(integer_count, self.normalization_factor)
        if remainder:
            raise ArithmeticError(
                "UnaryFOMC count is not divisible by the static-set "
                f"normalization factor {self.normalization_factor}"
            )
        return result


_COMPARATORS: dict[str, Callable[[int, int], bool]] = {
    "==": eq,
    "!=": ne,
    "<": lt,
    "<=": le,
    ">": gt,
    ">=": ge,
}


def _object_predicate(ref: ObjRef) -> str:
    return f"CofolaObject_{ref.id}"


def _disjunction(parts: list[str]) -> str:
    if not parts:
        return "FALSE"
    if len(parts) == 1:
        return parts[0]
    return "(" + " | ".join(parts) + ")"


def _conjunction(parts: list[str]) -> str:
    if not parts:
        return "TRUE"
    if len(parts) == 1:
        return parts[0]
    return "(" + " & ".join(parts) + ")"


def _exact_cardinality(body: str, size: int) -> str:
    return f"(exists={size} x: {body})"


def _linear_cardinality_formula(
    constraint: SizeConstraint,
    domain_size: int,
    cardinality_bounds: dict[ObjRef, int],
    *,
    max_cardinality_cases: int,
) -> str:
    coefficients: dict[ObjRef, int] = {}
    for term, coefficient in constraint.terms:
        if not isinstance(term, ObjRef):
            raise UnaryFOMCUnsupportedError(
                "UnaryFOMC cardinality translation currently supports object "
                f"sizes, not {type(term).__name__}"
            )
        coefficients[term] = coefficients.get(term, 0) + coefficient
    coefficients = {
        ref: coefficient
        for ref, coefficient in coefficients.items()
        if coefficient != 0
    }

    comparator = _COMPARATORS.get(constraint.comparator)
    if comparator is None:
        raise UnaryFOMCUnsupportedError(
            f"Unsupported size comparator {constraint.comparator!r}"
        )
    if not coefficients:
        return "TRUE" if comparator(0, constraint.rhs) else "FALSE"

    ordered = sorted(coefficients.items(), key=lambda item: item[0].id)
    bounds = [
        min(domain_size, cardinality_bounds.get(ref, domain_size))
        for ref, _ in ordered
    ]
    case_count = 1
    for bound in bounds:
        case_count *= bound + 1
    if case_count > max_cardinality_cases:
        raise UnaryFOMCUnsupportedError(
            "The finite linear-cardinality expansion would require "
            f"{case_count} cases; increase max_cardinality_cases or use the "
            "WFOMC backend"
        )

    accepted: list[str] = []
    for sizes in product(*(range(bound + 1) for bound in bounds)):
        value = sum(
            coefficient * size
            for (_, coefficient), size in zip(ordered, sizes)
        )
        if comparator(value, constraint.rhs):
            accepted.append(
                _conjunction(
                    [
                        _exact_cardinality(
                            f"{_object_predicate(ref)}(x)",
                            size,
                        )
                        for (ref, _), size in zip(ordered, sizes)
                    ]
                )
            )
    return _disjunction(accepted)


def _encode_object(
    ref: ObjRef,
    definition: object,
) -> list[str]:
    predicate = _object_predicate(ref)
    match definition:
        case SetInit():
            # All SetInit predicates are fixed together by their Venn-cell
            # cardinalities in _encode_static_sets.
            return []
        case SetChoose(source=source, size=size):
            clauses = [
                f"forall x: ({predicate}(x) -> {_object_predicate(source)}(x))"
            ]
            if size is not None:
                clauses.append(_exact_cardinality(f"{predicate}(x)", size))
            return clauses
        case SetUnion(left=left, right=right):
            return [
                f"forall x: ({predicate}(x) <-> "
                f"({_object_predicate(left)}(x) | {_object_predicate(right)}(x)))"
            ]
        case SetIntersection(left=left, right=right):
            return [
                f"forall x: ({predicate}(x) <-> "
                f"({_object_predicate(left)}(x) & {_object_predicate(right)}(x)))"
            ]
        case SetDifference(left=left, right=right):
            return [
                f"forall x: ({predicate}(x) <-> "
                f"({_object_predicate(left)}(x) & ~{_object_predicate(right)}(x)))"
            ]
        case _:
            raise UnaryFOMCUnsupportedError(
                "UnaryFOMC's basic Cofola translation does not yet support "
                f"{type(definition).__name__} (object {ref.id})"
            )


def _encode_constraint(
    constraint: object,
    entity_predicates: dict[Entity, str],
    domain_size: int,
    cardinality_bounds: dict[ObjRef, int],
    *,
    max_cardinality_cases: int,
) -> str:
    match constraint:
        case SizeConstraint():
            return _linear_cardinality_formula(
                constraint,
                domain_size,
                cardinality_bounds,
                max_cardinality_cases=max_cardinality_cases,
            )
        case MembershipConstraint(entity=entity, container=container, positive=positive):
            member = f"{_object_predicate(container)}(x)"
            if not positive:
                member = f"~{member}"
            return f"forall x: ({entity_predicates[entity]}(x) -> {member})"
        case SubsetConstraint(sub=sub, sup=sup, positive=positive):
            witness = f"({_object_predicate(sub)}(x) & ~{_object_predicate(sup)}(x))"
            return (
                f"forall x: ~{witness}"
                if positive
                else f"exists x: {witness}"
            )
        case DisjointConstraint(left=left, right=right, positive=positive):
            witness = f"({_object_predicate(left)}(x) & {_object_predicate(right)}(x))"
            return (
                f"forall x: ~{witness}"
                if positive
                else f"exists x: {witness}"
            )
        case EqualityConstraint(left=left, right=right, positive=positive):
            left_atom = f"{_object_predicate(left)}(x)"
            right_atom = f"{_object_predicate(right)}(x)"
            if positive:
                return f"forall x: ({left_atom} <-> {right_atom})"
            return (
                "exists x: "
                f"(({left_atom} & ~{right_atom}) | (~{left_atom} & {right_atom}))"
            )
        case _:
            raise UnaryFOMCUnsupportedError(
                "UnaryFOMC's basic Cofola translation does not yet support "
                f"{type(constraint).__name__}"
            )


def _static_cell_formula(
    static_refs: list[ObjRef],
    signature: tuple[bool, ...],
) -> str:
    return _conjunction(
        [
            (
                f"{_object_predicate(ref)}(x)"
                if present
                else f"~{_object_predicate(ref)}(x)"
            )
            for ref, present in zip(static_refs, signature)
        ]
    )


def _encode_static_sets(
    problem: Problem,
    entities: list[Entity],
    distinguished_entities: set[Entity],
) -> tuple[list[str], dict[Entity, str], int]:
    """Fix all input sets by Venn-cell sizes and return their symmetry factor."""

    static_definitions = [
        (ref, definition)
        for ref, definition in problem.iter_objects()
        if isinstance(definition, SetInit)
    ]
    static_refs = [ref for ref, _ in static_definitions]
    signatures = {
        entity: tuple(
            entity in definition.entities
            for _, definition in static_definitions
        )
        for entity in entities
    }
    signature_sizes: dict[tuple[bool, ...], int] = {}
    distinguished_sizes: dict[tuple[bool, ...], int] = {}
    for entity, signature in signatures.items():
        signature_sizes[signature] = signature_sizes.get(signature, 0) + 1
        if entity in distinguished_entities:
            distinguished_sizes[signature] = (
                distinguished_sizes.get(signature, 0) + 1
            )

    clauses = [
        _exact_cardinality(
            _static_cell_formula(static_refs, signature),
            size,
        )
        for signature, size in sorted(signature_sizes.items())
    ]

    entity_predicates = {
        entity: f"CofolaEntity_{index}"
        for index, entity in enumerate(entities)
        if entity in distinguished_entities
    }
    clauses.extend(
        _exact_cardinality(f"{predicate}(x)", 1)
        for predicate in entity_predicates.values()
    )
    clauses.extend(
        f"forall x: ~({left}(x) & {right}(x))"
        for left, right in combinations(entity_predicates.values(), 2)
    )
    clauses.extend(
        f"forall x: ({entity_predicates[entity]}(x) -> "
        f"{_static_cell_formula(static_refs, signatures[entity])})"
        for entity in entities
        if entity in distinguished_entities
    )

    denominator = 1
    for signature, size in signature_sizes.items():
        anonymous_size = size - distinguished_sizes.get(signature, 0)
        denominator *= factorial(anonymous_size)
    normalization_factor = factorial(len(entities)) // denominator
    return clauses, entity_predicates, normalization_factor


def encode_ir(
    problem: Problem,
    analysis: AnalysisResult,
    *,
    max_cardinality_cases: int = 100_000,
) -> UnaryFOMCEncoding:
    """Encode one normalized atomic Cofola problem into ``C1_=[f]``.

    The explicit input sets are fixed by the cardinalities of their nonempty
    Venn cells. Entities named by membership constraints additionally receive
    pairwise-disjoint singleton predicates. The resulting labeled encodings
    contribute a known symmetry factor, which
    :meth:`UnaryFOMCEncoding.decode_result` removes. The otherwise unused unary
    function is fixed to the identity, so it adds no multiplicity.
    """

    if analysis.unsatisfiable:
        return UnaryFOMCEncoding("FALSE", 0, 1)
    if max_cardinality_cases < 1:
        raise ValueError("max_cardinality_cases must be positive")

    entities = sorted(analysis.all_entities, key=lambda item: item.name)
    distinguished_entities = {
        constraint.entity
        for constraint in problem.constraints
        if isinstance(constraint, MembershipConstraint)
    }
    clauses = ["forall x: f(x) = x"]
    static_clauses, entity_predicates, normalization_factor = _encode_static_sets(
        problem,
        entities,
        distinguished_entities,
    )
    clauses.extend(static_clauses)
    cardinality_bounds = {
        ref: info.max_size
        for ref, info in analysis.set_info.items()
    }
    for ref, definition in problem.iter_objects():
        clauses.extend(_encode_object(ref, definition))
    clauses.extend(
        _encode_constraint(
            constraint,
            entity_predicates,
            len(entities),
            cardinality_bounds,
            max_cardinality_cases=max_cardinality_cases,
        )
        for constraint in problem.constraints
    )
    sentence = " & ".join(f"({clause})" for clause in clauses)
    return UnaryFOMCEncoding(
        sentence=sentence,
        domain_size=len(entities),
        normalization_factor=normalization_factor,
    )
