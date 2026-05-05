"""Type lattice for the Cofola IR.

Defines the spec types used by TypeCheckPass and a `type_of` mapping from IR
object definitions to their spec type.

Subtype groups:
    SET_LIKE  = {SET, BAG}
    LINEAR    = {SEQUENCE, CIRCLE}
    GROUPED   = {PARTITION, COMPOSITION}
    ORDERED   = {TUPLE, SEQUENCE, CIRCLE}
    CONTAINER = SET_LIKE  # `e in X` only allows set-like containers per spec
"""

from __future__ import annotations

from enum import Enum

from cofola.frontend.objects import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagInit,
    BagIntersection,
    BagPartRef,
    BagSupport,
    BagUnion,
    FuncDef,
    FuncImage,
    FuncInverse,
    FuncInverseImage,
    ObjDef,
    PartitionDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetPartRef,
    SetUnion,
    TupleDef,
)


class CofolaType(Enum):
    """The 11 spec types in the Cofola type lattice."""

    SET = "Set"
    BAG = "Bag"
    TUPLE = "Tuple"
    SEQUENCE = "Sequence"
    CIRCLE = "Circle"
    PARTITION = "Partition"
    COMPOSITION = "Composition"
    FUNCTION = "Function"
    ENTITY = "Entity"
    INT = "Int"
    PATTERN = "Pattern"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# Subtype groups (spec-prescribed).
SET_LIKE: frozenset[CofolaType] = frozenset({CofolaType.SET, CofolaType.BAG})
LINEAR: frozenset[CofolaType] = frozenset({CofolaType.SEQUENCE, CofolaType.CIRCLE})
GROUPED: frozenset[CofolaType] = frozenset({CofolaType.PARTITION, CofolaType.COMPOSITION})
ORDERED: frozenset[CofolaType] = frozenset(
    {CofolaType.TUPLE, CofolaType.SEQUENCE, CofolaType.CIRCLE}
)
# `e in X` is only allowed for set-like containers per the spec
# (basic_objects.tex). Ordered objects do NOT support `e in X`.
CONTAINER: frozenset[CofolaType] = SET_LIKE


def type_of(defn: ObjDef) -> CofolaType:
    """Map an IR object definition to its spec type.

    Polymorphism (e.g. choose over a Set vs a Bag) is already resolved
    structurally by the transformer — SetChoose is only created for set
    sources, BagChoose only for bag sources, and so on.

    Args:
        defn: The IR object definition.

    Returns:
        The CofolaType for this definition.

    Raises:
        TypeError: If the definition is not a recognised IR node.
    """
    # Set-producing definitions
    if isinstance(defn, (
        SetInit,
        SetChoose,
        SetUnion,
        SetIntersection,
        SetDifference,
        BagSupport,
        FuncImage,
        FuncInverseImage,
        SetPartRef,
    )):
        return CofolaType.SET
    # Bag-producing definitions
    if isinstance(defn, (
        BagInit,
        BagChoose,
        SetChooseReplace,
        BagUnion,
        BagAdditiveUnion,
        BagIntersection,
        BagDifference,
        BagPartRef,
    )):
        return CofolaType.BAG
    if isinstance(defn, TupleDef):
        return CofolaType.TUPLE
    if isinstance(defn, SequenceDef):
        return CofolaType.CIRCLE if defn.circular else CofolaType.SEQUENCE
    if isinstance(defn, PartitionDef):
        return CofolaType.COMPOSITION if defn.ordered else CofolaType.PARTITION
    if isinstance(defn, (FuncDef, FuncInverse)):
        return CofolaType.FUNCTION
    raise TypeError(
        f"type_of: unrecognised IR object definition {type(defn).__name__}"
    )


def is_subtype(
    actual: CofolaType,
    expected: CofolaType | frozenset[CofolaType],
) -> bool:
    """Return True if `actual` satisfies the type expectation `expected`.

    `expected` may be a single CofolaType (exact match) or a frozenset of
    CofolaTypes (membership). This intentionally does not implement a
    full lattice with transitive joins — the spec only uses flat unions.
    """
    if isinstance(expected, CofolaType):
        return actual is expected
    return actual in expected
