"""Operator/constraint type signatures for TypeCheckPass.

Signatures are keyed by IR-node class — TypeCheckPass walks the IR, not the
parser tree. Polymorphism (e.g. choose over a Set vs a Bag) was already
resolved by the transformer when it created the IR node, so each IR-node
class has a single, monomorphic signature.

A Signature describes:
- params: the typed positional parameters (each is a Param with an `accepts`
  type expectation and a human-readable name).
- result: the spec type produced by this op, or None for constraints (which
  don't produce a value).
- extra: spec-prescribed predicates that run after positional checks pass.
  Each predicate has signature `(node, problem) -> str | None`; it returns
  None on success, or a human-readable error message on failure.
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
    BagPartRef,
    BagSupport,
    BagUnion,
    FuncImage,
    FuncInverseImage,
    PartitionDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetIntersection,
    SetPartRef,
    SetUnion,
    TupleDef,
)
from cofola.ir.analysis.type_lattice import (
    CONTAINER,
    CofolaType,
    LINEAR,
    SET_LIKE,
)


# Each `extra` predicate runs against the IR node after positional checks.
# Returns None if OK, or a human-readable message describing the problem.
# `problem` is the surrounding Problem (passed through so predicates can
# resolve refs to definitions).
ExtraCheck = Callable[[object, object], Optional[str]]


@dataclass(frozen=True)
class Param:
    """One positional parameter of an operator/constraint.

    Attributes:
        accepts: Single CofolaType for an exact match, or a frozenset for
            "any of these".
        name: Human-readable parameter name for error messages.
        field: The dataclass field to read on the IR node. If None, the
            param is matched positionally to the dataclass's fields() order
            and is not used here (callers always pass an explicit field name).
    """

    accepts: CofolaType | frozenset[CofolaType]
    name: str
    field: str = ""


@dataclass(frozen=True)
class Signature:
    """A type signature for an IR-node class.

    Attributes:
        params: The typed parameters this op accepts (in declaration order
            on the IR dataclass).
        result: The type produced by the op, or None for constraints.
        extra: Optional spec-prescribed predicates run after positional checks.
    """

    params: tuple[Param, ...]
    result: Optional[CofolaType]
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
    if not isinstance(seq_defn, SequenceDef) or not seq_defn.circular:
        return None
    pattern = getattr(node, "pattern")
    if isinstance(pattern, LessThanPattern):
        return "Circle does not support `<` pattern"
    if isinstance(pattern, PredecessorPattern):
        return "Circle does not support predecessor pattern (e1, e2)"
    return None


def _disjoint_same_kind(node: object, problem: object) -> Optional[str]:
    """`disjoint` requires both operands to be the same kind (Set/Set or
    Bag/Bag). The base SET_LIKE check on each operand is done by Param;
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
    lt = type_of_safe(left_def)
    rt = type_of_safe(right_def)
    if lt is None or rt is None:
        return None
    if lt is rt:
        return None
    if {lt, rt} <= {CofolaType.SET, CofolaType.BAG}:
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
    # SetChoose / BagChoose: source must match the result kind. The
    # transformer created the right class already, but we still validate
    # the source field.
    SetChoose: Signature(
        params=(
            Param(CofolaType.SET, "source", field="source"),
        ),
        result=CofolaType.SET,
    ),
    BagChoose: Signature(
        params=(
            Param(CofolaType.BAG, "source", field="source"),
        ),
        result=CofolaType.BAG,
    ),
    SetChooseReplace: Signature(
        params=(
            Param(CofolaType.SET, "source", field="source"),
        ),
        result=CofolaType.BAG,
    ),
    SetUnion: Signature(
        params=(
            Param(CofolaType.SET, "left", field="left"),
            Param(CofolaType.SET, "right", field="right"),
        ),
        result=CofolaType.SET,
    ),
    SetIntersection: Signature(
        params=(
            Param(CofolaType.SET, "left", field="left"),
            Param(CofolaType.SET, "right", field="right"),
        ),
        result=CofolaType.SET,
    ),
    SetDifference: Signature(
        params=(
            Param(CofolaType.SET, "left", field="left"),
            Param(CofolaType.SET, "right", field="right"),
        ),
        result=CofolaType.SET,
    ),
    BagUnion: Signature(
        params=(
            Param(CofolaType.BAG, "left", field="left"),
            Param(CofolaType.BAG, "right", field="right"),
        ),
        result=CofolaType.BAG,
    ),
    BagAdditiveUnion: Signature(
        params=(
            Param(CofolaType.BAG, "left", field="left"),
            Param(CofolaType.BAG, "right", field="right"),
        ),
        result=CofolaType.BAG,
    ),
    BagIntersection: Signature(
        params=(
            Param(CofolaType.BAG, "left", field="left"),
            Param(CofolaType.BAG, "right", field="right"),
        ),
        result=CofolaType.BAG,
    ),
    BagDifference: Signature(
        params=(
            Param(CofolaType.BAG, "left", field="left"),
            Param(CofolaType.BAG, "right", field="right"),
        ),
        result=CofolaType.BAG,
    ),
    BagSupport: Signature(
        params=(
            Param(CofolaType.BAG, "source", field="source"),
        ),
        result=CofolaType.SET,
    ),
    # Func image / inverse image: argument may be a set or an entity. We
    # treat it as ENTITY-or-SET_LIKE; checked specially in the walker.
    FuncImage: Signature(
        params=(
            Param(CofolaType.FUNCTION, "func", field="func"),
        ),
        result=CofolaType.SET,
    ),
    FuncInverseImage: Signature(
        params=(
            Param(CofolaType.FUNCTION, "func", field="func"),
        ),
        result=CofolaType.SET,
    ),
    # Tuples: source must be SET_LIKE per spec (ordered_objects.tex).
    TupleDef: Signature(
        params=(
            Param(SET_LIKE, "source", field="source"),
        ),
        result=CofolaType.TUPLE,
    ),
    # Sequences/circles: source must be SET_LIKE (same rationale).
    SequenceDef: Signature(
        params=(
            Param(SET_LIKE, "source", field="source"),
        ),
        # Result is SEQUENCE or CIRCLE depending on `circular`; type_of
        # handles the dispatch. We declare SEQUENCE here as a placeholder;
        # the result field is informational only and not checked downstream.
        result=CofolaType.SEQUENCE,
    ),
    # Partitions/compositions: source must be SET_LIKE.
    PartitionDef: Signature(
        params=(
            Param(SET_LIKE, "source", field="source"),
        ),
        result=CofolaType.PARTITION,
    ),
    # Partition refs: partition must be PARTITION or COMPOSITION.
    SetPartRef: Signature(
        params=(
            Param(frozenset({CofolaType.PARTITION, CofolaType.COMPOSITION}),
                  "partition", field="partition"),
        ),
        result=CofolaType.SET,
    ),
    BagPartRef: Signature(
        params=(
            Param(frozenset({CofolaType.PARTITION, CofolaType.COMPOSITION}),
                  "partition", field="partition"),
        ),
        result=CofolaType.BAG,
    ),
    # ------------------------------------------------------------------
    # Constraints (no result)
    # ------------------------------------------------------------------
    MembershipConstraint: Signature(
        params=(
            # entity: typed by construction (Entity), not validated here
            Param(CONTAINER, "container", field="container"),
        ),
        result=None,
    ),
    SubsetConstraint: Signature(
        # Pure SubsetConstraint is for SETs only (BagSubsetConstraint covers
        # the bag case and is dispatched separately by the transformer).
        params=(
            Param(CofolaType.SET, "sub", field="sub"),
            Param(CofolaType.SET, "sup", field="sup"),
        ),
        result=None,
    ),
    BagSubsetConstraint: Signature(
        params=(
            Param(CofolaType.BAG, "sub", field="sub"),
            Param(CofolaType.BAG, "sup", field="sup"),
        ),
        result=None,
    ),
    EqualityConstraint: Signature(
        # Plain EqualityConstraint is for SETs only (BagEqConstraint covers
        # the bag case). Tuple==Tuple etc. is not a thing per spec.
        params=(
            Param(CofolaType.SET, "left", field="left"),
            Param(CofolaType.SET, "right", field="right"),
        ),
        result=None,
    ),
    BagEqConstraint: Signature(
        params=(
            Param(CofolaType.BAG, "left", field="left"),
            Param(CofolaType.BAG, "right", field="right"),
        ),
        result=None,
    ),
    DisjointConstraint: Signature(
        # Both refs must be SET_LIKE; the same-kind constraint is enforced
        # by an extra predicate below.
        params=(
            Param(SET_LIKE, "left", field="left"),
            Param(SET_LIKE, "right", field="right"),
        ),
        result=None,
        extra=(_disjoint_same_kind,),
    ),
    TupleIndexEq: Signature(
        params=(
            Param(CofolaType.TUPLE, "tuple_ref", field="tuple_ref"),
            # entity is typed by construction
        ),
        result=None,
    ),
    TupleIndexMembership: Signature(
        params=(
            Param(CofolaType.TUPLE, "tuple_ref", field="tuple_ref"),
            Param(SET_LIKE, "container", field="container"),
        ),
        result=None,
    ),
    SequencePatternConstraint: Signature(
        params=(
            Param(LINEAR, "seq", field="seq"),
        ),
        result=None,
        extra=(_circle_pattern_compat,),
    ),
    FuncPairConstraint: Signature(
        params=(
            Param(CofolaType.FUNCTION, "func", field="func"),
        ),
        result=None,
    ),
    # ------------------------------------------------------------------
    # Size atoms
    # ------------------------------------------------------------------
    BagCountAtom: Signature(
        params=(
            Param(CofolaType.BAG, "bag", field="bag"),
        ),
        result=CofolaType.INT,
    ),
    TupleCountAtom: Signature(
        params=(
            Param(CofolaType.TUPLE, "tuple_ref", field="tuple_ref"),
            # Per existing semantics, count_obj may be an Entity (count of
            # positions equal to the entity) or a Set/Bag (count of
            # positions whose value is in the collection). The Entity
            # case is handled separately in _check_signature; here we
            # only constrain the ObjRef case.
            Param(SET_LIKE | frozenset({CofolaType.ENTITY}), "count_obj",
                  field="count_obj"),
        ),
        result=CofolaType.INT,
    ),
    SeqPatternCountAtom: Signature(
        params=(
            Param(LINEAR, "seq", field="seq"),
        ),
        result=CofolaType.INT,
        extra=(_circle_pattern_compat, _no_together_count),
    ),
    # ------------------------------------------------------------------
    # Sequence patterns themselves: the type-check on their fields runs
    # in the walker (not as standalone signatures) because the fields are
    # `ObjRef | Entity` unions. See type_check.py.
    # ------------------------------------------------------------------
}


def type_of_safe(defn: object) -> CofolaType | None:
    """type_of() that returns None on unknown defs, for use in lambdas."""
    from cofola.ir.analysis.type_lattice import type_of as _type_of

    try:
        return _type_of(defn)  # type: ignore[arg-type]
    except TypeError:
        return None


# Pattern field expectations.
# `entity_ok` says whether an Entity literal is also valid in the field.
# `kinds` are the acceptable CofolaTypes when the value is an ObjRef.
#
# Per spec (ordered_objects.tex):
# - TogetherPattern.group: must be a Set/Bag (entity makes no sense alone).
# - LessThanPattern.left/right, PredecessorPattern.first/second,
#   NextToPattern.first/second: may be a single entity or a Set/Bag.
PATTERN_FIELD_EXPECT: dict[
    type,
    tuple[tuple[str, frozenset[CofolaType], bool], ...],
] = {
    TogetherPattern: (("group", SET_LIKE, False),),
    LessThanPattern: (
        ("left", SET_LIKE, True),
        ("right", SET_LIKE, True),
    ),
    PredecessorPattern: (
        ("first", SET_LIKE, True),
        ("second", SET_LIKE, True),
    ),
    NextToPattern: (
        ("first", SET_LIKE, True),
        ("second", SET_LIKE, True),
    ),
}
