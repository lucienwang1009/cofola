"""Operator/constraint type signatures for TypeChecker.

Signatures are keyed by frontend-node class — TypeChecker walks the Problem, not the
parser tree. Polymorphism (e.g. choose over a Set vs a Bag) was already
resolved by the transformer when it created the frontend node, so each node
class has a single, monomorphic signature.

A Signature describes:
- params: the typed positional parameters (each is a Param with an `accepts`
  type expectation and a human-readable name).
- extra: spec-prescribed predicates that run after positional checks pass.
  Each predicate has signature `(node, problem) -> str | None`; it returns
  None on success, or a human-readable error message on failure.

`accepts` is either a single class or a tuple of classes — exactly the shape
Python's built-in `isinstance` accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from cofola.frontend.constraints import (
    BagCountAtom,
    BagEqConstraint,
    BagSubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    FuncPairConstraint,
    LessThanPattern,
    MembershipConstraint,
    NextToPattern,
    PredecessorPattern,
    SeqPatternCountAtom,
    SequencePatternConstraint,
    SubsetConstraint,
    TogetherPattern,
    TupleCountAtom,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagIntersection,
    BagObjDef,
    BagPartDef,
    BagSupport,
    BagUnion,
    CircleDef,
    CompositionDef,
    FuncImage,
    FuncInverseImage,
    FuncObjDef,
    Grouped,
    Linear,
    PartitionDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetIntersection,
    SetLike,
    SetObjDef,
    SetPartDef,
    SetUnion,
    TupleDef,
)
from cofola.frontend.objects import Entity


# A type expectation: a single class, or a tuple of classes (matching the
# argument shape of built-in `isinstance`).
TypeExpect = type | tuple[type, ...]

# Each `extra` predicate runs against the frontend node after positional checks.
# Returns None if OK, or a human-readable message describing the problem.
# `problem` is the surrounding Problem (passed through so predicates can
# resolve refs to definitions).
ExtraCheck = Callable[[object, object], Optional[str]]


@dataclass(frozen=True)
class Param:
    """One positional parameter of an operator/constraint.

    Attributes:
        accepts: A class (exact match via inheritance) or a tuple of
            classes (any-of match). Same semantics as ``isinstance``.
        name: Human-readable parameter name for error messages.
        field: The dataclass field to read on the frontend node.
    """

    accepts: TypeExpect
    name: str
    field: str = ""


@dataclass(frozen=True)
class Signature:
    """A type signature for a frontend-node class.

    Attributes:
        params: The typed parameters this op accepts (in declaration order
            on the frontend dataclass).
        extra: Optional spec-prescribed predicates run after positional checks.
    """

    params: tuple[Param, ...]
    extra: tuple[ExtraCheck, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Spec-prescribed extra predicates
# ---------------------------------------------------------------------------


def _circle_pattern_compat(node: object, problem: object) -> Optional[str]:
    """Circles only support `together` and `next_to` / adjacency patterns
    (and the implicit identity patterns). They reject `<` and predecessor
    patterns (which require a fixed starting position).
    """
    seq_ref = getattr(node, "seq")
    seq_defn = problem.get_object(seq_ref)  # type: ignore[attr-defined]
    if not isinstance(seq_defn, CircleDef):
        return None
    pattern = getattr(node, "pattern")
    if isinstance(pattern, LessThanPattern):
        return "Circle does not support `<` pattern"
    if isinstance(pattern, PredecessorPattern):
        return "Circle does not support predecessor pattern (e1, e2)"
    return None


def _disjoint_same_kind(node: object, problem: object) -> Optional[str]:
    """`disjoint` requires both operands to be the same kind (Set/Set or
    Bag/Bag). The base SetLike check on each operand is done by Param;
    here we just enforce that the kinds match.
    """
    left_ref = getattr(node, "left", None)
    right_ref = getattr(node, "right", None)
    if left_ref is None or right_ref is None:
        return None
    left_def = problem.get_object(left_ref)  # type: ignore[attr-defined]
    right_def = problem.get_object(right_ref)  # type: ignore[attr-defined]
    if left_def is None or right_def is None:
        return None
    left_set = isinstance(left_def, SetObjDef)
    right_set = isinstance(right_def, SetObjDef)
    left_bag = isinstance(left_def, BagObjDef)
    right_bag = isinstance(right_def, BagObjDef)
    if (left_set and right_set) or (left_bag and right_bag):
        return None
    if (left_set or left_bag) and (right_set or right_bag):
        return "disjoint requires both operands to be of the same kind (Set/Set or Bag/Bag)"
    return None


def _no_together_count(node: object, problem: object) -> Optional[str]:
    """Per spec (ordered_objects.tex): `together` does NOT have a count
    variant — `seq.count(together(...))` is forbidden.
    """
    pattern = getattr(node, "pattern")
    if isinstance(pattern, TogetherPattern):
        return "seq.count(together(...)) is not supported (no count variant for `together` per spec)"
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SIGNATURES: dict[type, Signature] = {
    # ------------------------------------------------------------------
    # Object-producing operators
    # ------------------------------------------------------------------
    SetChoose: Signature(
        params=(Param(SetObjDef, "source", field="source"),),
    ),
    BagChoose: Signature(
        params=(Param(BagObjDef, "source", field="source"),),
    ),
    SetChooseReplace: Signature(
        params=(Param(SetObjDef, "source", field="source"),),
    ),
    SetUnion: Signature(
        params=(
            Param(SetObjDef, "left", field="left"),
            Param(SetObjDef, "right", field="right"),
        ),
    ),
    SetIntersection: Signature(
        params=(
            Param(SetObjDef, "left", field="left"),
            Param(SetObjDef, "right", field="right"),
        ),
    ),
    SetDifference: Signature(
        params=(
            Param(SetObjDef, "left", field="left"),
            Param(SetObjDef, "right", field="right"),
        ),
    ),
    BagUnion: Signature(
        params=(
            Param(BagObjDef, "left", field="left"),
            Param(BagObjDef, "right", field="right"),
        ),
    ),
    BagAdditiveUnion: Signature(
        params=(
            Param(BagObjDef, "left", field="left"),
            Param(BagObjDef, "right", field="right"),
        ),
    ),
    BagIntersection: Signature(
        params=(
            Param(BagObjDef, "left", field="left"),
            Param(BagObjDef, "right", field="right"),
        ),
    ),
    BagDifference: Signature(
        params=(
            Param(BagObjDef, "left", field="left"),
            Param(BagObjDef, "right", field="right"),
        ),
    ),
    BagSupport: Signature(
        params=(Param(BagObjDef, "source", field="source"),),
    ),
    FuncImage: Signature(
        params=(Param(FuncObjDef, "func", field="func"),),
    ),
    FuncInverseImage: Signature(
        params=(Param(FuncObjDef, "func", field="func"),),
    ),
    # Tuples: source must be SetLike per spec (ordered_objects.tex).
    TupleDef: Signature(
        params=(Param(SetLike, "source", field="source"),),
    ),
    # Sequences/circles: source must be SetLike (same rationale).
    SequenceDef: Signature(
        params=(Param(SetLike, "source", field="source"),),
    ),
    CircleDef: Signature(
        params=(Param(SetLike, "source", field="source"),),
    ),
    # Partitions/compositions: source must be SetLike.
    PartitionDef: Signature(
        params=(Param(SetLike, "source", field="source"),),
    ),
    CompositionDef: Signature(
        params=(Param(SetLike, "source", field="source"),),
    ),
    # Partition refs: partition must be PartitionDef or CompositionDef (Grouped).
    SetPartDef: Signature(
        params=(Param(Grouped, "partition", field="partition"),),
    ),
    BagPartDef: Signature(
        params=(Param(Grouped, "partition", field="partition"),),
    ),
    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    MembershipConstraint: Signature(
        # `e in X` is only allowed for set-like containers per spec.
        # (entity is typed by construction; not validated here.)
        params=(Param(SetLike, "container", field="container"),),
    ),
    SubsetConstraint: Signature(
        # Pure SubsetConstraint is for SETs only (BagSubsetConstraint covers
        # the bag case and is dispatched separately by the transformer).
        params=(
            Param(SetObjDef, "sub", field="sub"),
            Param(SetObjDef, "sup", field="sup"),
        ),
    ),
    BagSubsetConstraint: Signature(
        params=(
            Param(BagObjDef, "sub", field="sub"),
            Param(BagObjDef, "sup", field="sup"),
        ),
    ),
    EqualityConstraint: Signature(
        params=(
            Param(SetObjDef, "left", field="left"),
            Param(SetObjDef, "right", field="right"),
        ),
    ),
    BagEqConstraint: Signature(
        params=(
            Param(BagObjDef, "left", field="left"),
            Param(BagObjDef, "right", field="right"),
        ),
    ),
    DisjointConstraint: Signature(
        # Both refs must be SetLike; same-kind constraint enforced below.
        params=(
            Param(SetLike, "left", field="left"),
            Param(SetLike, "right", field="right"),
        ),
        extra=(_disjoint_same_kind,),
    ),
    TupleIndexEq: Signature(
        params=(Param(TupleDef, "tuple_ref", field="tuple_ref"),),
    ),
    TupleIndexMembership: Signature(
        params=(
            Param(TupleDef, "tuple_ref", field="tuple_ref"),
            Param(SetLike, "container", field="container"),
        ),
    ),
    SequencePatternConstraint: Signature(
        params=(Param(Linear, "seq", field="seq"),),
        extra=(_circle_pattern_compat,),
    ),
    FuncPairConstraint: Signature(
        params=(Param(FuncObjDef, "func", field="func"),),
    ),
    # ------------------------------------------------------------------
    # Size atoms
    # ------------------------------------------------------------------
    BagCountAtom: Signature(
        params=(Param(BagObjDef, "bag", field="bag"),),
    ),
    TupleCountAtom: Signature(
        # count_obj may be an Entity (count of equal positions) or Set/Bag
        # (count of positions whose value is in the collection).
        params=(
            Param(TupleDef, "tuple_ref", field="tuple_ref"),
            Param((SetLike, Entity), "count_obj", field="count_obj"),
        ),
    ),
    SeqPatternCountAtom: Signature(
        params=(Param(Linear, "seq", field="seq"),),
        extra=(_circle_pattern_compat, _no_together_count),
    ),
    # ------------------------------------------------------------------
    # Sequence patterns themselves: the type-check on their fields runs
    # in TypeChecker (not as standalone signatures) because the fields are
    # `ObjRef | Entity` unions. See type_check.py.
    # ------------------------------------------------------------------
}


# Pattern field expectations.
# `entity_ok` says whether an Entity literal is also valid in the field.
# `kinds` is the acceptable type expectation when the value is an ObjRef.
#
# Per spec (ordered_objects.tex):
# - TogetherPattern.group: must be a Set/Bag (entity makes no sense alone).
# - LessThanPattern.left/right, PredecessorPattern.first/second,
#   NextToPattern.first/second: may be a single entity or a Set/Bag.
PATTERN_FIELD_EXPECT: dict[
    type,
    tuple[tuple[str, TypeExpect, bool], ...],
] = {
    TogetherPattern: (("group", SetLike, False),),
    LessThanPattern: (
        ("left", SetLike, True),
        ("right", SetLike, True),
    ),
    PredecessorPattern: (
        ("first", SetLike, True),
        ("second", SetLike, True),
    ),
    NextToPattern: (
        ("first", SetLike, True),
        ("second", SetLike, True),
    ),
}
