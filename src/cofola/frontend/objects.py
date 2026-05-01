"""Object definitions for the immutable IR.

All object definitions are frozen dataclasses. They store ONLY structural
information — no derived properties like p_entities, max_size, dis_entities.
Those are computed by analysis passes.
"""

from __future__ import annotations

from dataclasses import dataclass

from cofola.frontend.types import Entity, ObjRef


# =============================================================================
# Base Sets
# =============================================================================


@dataclass(frozen=True, slots=True)
class SetInit:
    """A set explicitly defined by its entities.

    Example: set A = {a, b, c}
    """

    entities: frozenset[Entity]


@dataclass(frozen=True, slots=True)
class SetChoose:
    """A subset chosen from a source set.

    Example: set B = choose 2 from A
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class SetChooseReplace:
    """A multiset chosen from a source set with replacement.

    Example: set C = choose 3 from A with replacement
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class SetUnion:
    """Union of two sets.

    Example: set D = A union B
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class SetIntersection:
    """Intersection of two sets.

    Example: set E = A intersect B
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class SetDifference:
    """Difference of two sets.

    Example: set F = A minus B
    """

    left: ObjRef
    right: ObjRef


# Union type for all set objects
SetObjDef = SetInit | SetChoose | SetUnion | SetIntersection | SetDifference


# =============================================================================
# Bags
# =============================================================================


@dataclass(frozen=True, slots=True)
class BagInit:
    """A bag explicitly defined by entity multiplicities.

    Example: bag B = {a: 2, b: 3}
    Use tuple of pairs for hashability; ordered by entity name for determinism.
    """

    entity_multiplicity: tuple[tuple[Entity, int], ...]


@dataclass(frozen=True, slots=True)
class BagChoose:
    """A sub-bag chosen from a source bag.

    Example: bag C = choose 2 from B
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class BagUnion:
    """Union of two bags (max multiplicity).

    Example: bag D = B union C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagAdditiveUnion:
    """Additive union of two bags (sum multiplicities).

    Example: bag E = B add C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagIntersection:
    """Intersection of two bags (min multiplicity).

    Example: bag F = B intersect C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagDifference:
    """Difference of two bags.

    Example: bag G = B minus C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagSupport:
    """The support set of a bag (unique elements).

    Example: set S = support of B
    """

    source: ObjRef


# Union type for pure bag objects (BagSupport is set-like, see DerivedSetObjDef below)
BagObjDef = (
    BagInit
    | BagChoose
    | BagUnion
    | BagAdditiveUnion
    | BagIntersection
    | BagDifference
    | SetChooseReplace
)


# =============================================================================
# Functions
# =============================================================================


@dataclass(frozen=True, slots=True)
class FuncDef:
    """A function definition.

    Example: func f: A -> B [injective] [surjective]
    """

    domain: ObjRef
    codomain: ObjRef
    injective: bool = False
    surjective: bool = False


@dataclass(frozen=True, slots=True)
class FuncImage:
    """The image of a function applied to an argument.

    Example: set I = f(A) or f(a) for entity a
    """

    func: ObjRef
    argument: ObjRef  # Set or Entity ref


@dataclass(frozen=True, slots=True)
class FuncInverseImage:
    """The inverse image of a function.

    Example: set P = f^{-1}(B)
    """

    func: ObjRef
    argument: ObjRef


@dataclass(frozen=True, slots=True)
class FuncInverse:
    """The inverse function of f (f⁻¹ as a function object, domain/codomain swapped).

    This represents f⁻¹ as a function object, not as an image.
    The inverse function maps codomain(f) → domain(f).

    Example: func g = inverse of f
    """

    func: ObjRef


# Union type for function mappings only (FuncImage/FuncInverseImage are set-like, see DerivedSetObjDef)
FuncObjDef = FuncDef | FuncInverse


# =============================================================================
# Tuples
# =============================================================================


@dataclass(frozen=True, slots=True)
class TupleDef:
    """A tuple definition.

    Example: tuple T = tuple of A
             tuple T = choose 3 tuple from A
             tuple T = choose 3 tuple from A with replacement
    """

    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None


# =============================================================================
# Sequences
# =============================================================================


@dataclass(frozen=True, slots=True)
class SequenceDef:
    """A sequence definition.

    Example: seq S = sequence of A
             seq S = choose 3 sequence from A
             seq S = circle of A [with reflection]
    """

    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None
    circular: bool = False
    reflection: bool = False
    flatten: ObjRef | None = None  # SetInit of position-index entities for Bag sources


# =============================================================================
# Partitions
# =============================================================================


@dataclass(frozen=True, slots=True)
class PartitionDef:
    """A partition/composition definition.

    Example: partition P = partition A into 3
             composition C = compose A into 3
    """

    source: ObjRef
    num_parts: int
    ordered: bool  # True = composition, False = partition


@dataclass(frozen=True, slots=True)
class SetPartRef:
    """Reference to the i-th part of a set partition/composition.

    Created when the partition's source is a set-like object.
    Example: P[0], C[1]
    """

    partition: ObjRef
    index: int


@dataclass(frozen=True, slots=True)
class BagPartRef:
    """Reference to the i-th part of a bag partition/composition.

    Created when the partition's source is a bag-like object.
    Example: P[0], C[1]
    """

    partition: ObjRef
    index: int


# Backwards-compatible union: any code doing `isinstance(x, PartRef)` keeps working.
PartRef = SetPartRef | BagPartRef


# =============================================================================
# Union Types
# =============================================================================

# Objects that always produce a set, derived from bags or functions:
#   BagSupport(bag)          → support set of a bag
#   FuncImage(f, A)          → image f(A), always a set
#   FuncInverseImage(f, B)   → preimage f⁻¹(B), always a set
DerivedSetObjDef = BagSupport | FuncImage | FuncInverseImage

# All objects that unconditionally produce a set (primary + derived)
AnySetObjDef = SetObjDef | DerivedSetObjDef

# Note: SetPartRef / BagPartRef are intentionally NOT folded into AnySetObjDef /
# BagObjDef. The aliases describe primary set/bag definitions; PartRefs are
# references to a piece of one and are handled separately by analysis passes.

# All object definition types
ObjDef = (
    AnySetObjDef | BagObjDef | FuncObjDef | TupleDef | SequenceDef
    | PartitionDef | SetPartRef | BagPartRef
)

# All container objects (can contain entities)
ContainerObjDef = (
    AnySetObjDef | BagObjDef | SetPartRef | BagPartRef | TupleDef | SequenceDef
)
