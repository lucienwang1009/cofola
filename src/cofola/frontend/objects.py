"""Object definitions for the Cofola frontend core problem model.

All object definitions are frozen dataclasses. They store ONLY structural
information — no derived properties like p_entities, max_size, dis_entities.
Those are computed by analysis passes.

The class hierarchy doubles as the spec type system: each concrete dataclass
inherits from one or more ABCs that name its spec type and group memberships
(e.g. ``SetLike``, ``Linear``, ``Ordered``). Type checks throughout the
codebase use plain ``isinstance``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjRef:
    """Unique reference to an object in the frontend problem graph."""

    id: int

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjRef) and self.id == other.id

    def __lt__(self, other: "ObjRef") -> bool:
        return self.id < other.id

    def __repr__(self) -> str:
        return f"ObjRef({self.id})"


@dataclass(frozen=True, slots=True)
class Entity:
    """An atomic entity in a combinatorics problem."""

    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.name == other.name

    def __repr__(self) -> str:
        return f"Entity({self.name!r})"

    def __str__(self) -> str:
        return self.name


RefOrEntity = ObjRef | Entity


# =============================================================================
# Spec-type ABCs (empty marker classes)
# =============================================================================
#
# Group mixins (correspond to the spec's flat unions):
#   SetLike  = {SET, BAG}                — `e in X` containers
#   Linear   = {SEQUENCE, CIRCLE}        — sequence-like
#   Grouped  = {PARTITION, COMPOSITION}  — partition-like
#   Ordered  = {TUPLE, SEQUENCE, CIRCLE} — index-addressable
#
# Leaf-type bases:
#   ObjDef         — root of every object definition
#   SetObjDef      — concrete: SetInit, SetChoose, SetUnion, SetIntersection,
#                    SetDifference, BagSupport, FuncImage, FuncInverseImage,
#                    SetPartDef
#   BagObjDef      — concrete: BagInit, BagChoose, SetChooseReplace, BagUnion,
#                    BagAdditiveUnion, BagIntersection, BagDifference,
#                    BagPartDef
#   FuncObjDef     — concrete: FuncDef, FuncInverse
#   TupleDef, SequenceDef, CircleDef (Phase C),
#   PartitionDef, CompositionDef (Phase D) are themselves the spec-type
#   markers (single concrete class per type).
#
# `PartDef` is an orthogonal marker mixin for the "i-th part of a partition"
# pattern; SetPartDef/BagPartDef inherit it alongside their SetObjDef/
# BagObjDef base.


class SetLike:
    """Marker for set-like spec types (SET ∪ BAG)."""

    __slots__ = ()


class Linear:
    """Marker for linear spec types (SEQUENCE ∪ CIRCLE)."""

    __slots__ = ()


class Grouped:
    """Marker for grouped spec types (PARTITION ∪ COMPOSITION)."""

    __slots__ = ()


class Ordered:
    """Marker for ordered spec types (TUPLE ∪ SEQUENCE ∪ CIRCLE)."""

    __slots__ = ()


class ObjDef:
    """Root of every object definition."""

    __slots__ = ()


class SetObjDef(ObjDef, SetLike):
    """Base for every concrete SET-typed object definition."""

    __slots__ = ()


class BagObjDef(ObjDef, SetLike):
    """Base for every concrete BAG-typed object definition."""

    __slots__ = ()


class FuncObjDef(ObjDef):
    """Base for every concrete FUNCTION-typed object definition."""

    __slots__ = ()


class PartDef:
    """Marker mixin for partition-part references (carry a `partition` ref
    and an `index`). SetPartDef/BagPartDef inherit this alongside their
    spec-type base.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class PartPlaceholderDef(ObjDef):
    """Scoped placeholder used inside ``ForAllParts`` templates.

    It is not a concrete partition part. Lowering replaces references to this
    placeholder with each real part of the target partition.
    """

    partition: ObjRef


# =============================================================================
# Base Sets
# =============================================================================


@dataclass(frozen=True, slots=True)
class SetInit(SetObjDef):
    """A set explicitly defined by its entities.

    Example: set A = {a, b, c}
    """

    entities: frozenset[Entity]


@dataclass(frozen=True, slots=True)
class SetChoose(SetObjDef):
    """A subset chosen from a source set.

    Example: set B = choose 2 from A
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class SetChooseReplace(BagObjDef):
    """A multiset chosen from a source set with replacement.

    Example: set C = choose 3 from A with replacement
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class SetUnion(SetObjDef):
    """Union of two sets.

    Example: set D = A union B
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class SetIntersection(SetObjDef):
    """Intersection of two sets.

    Example: set E = A intersect B
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class SetDifference(SetObjDef):
    """Difference of two sets.

    Example: set F = A minus B
    """

    left: ObjRef
    right: ObjRef


# =============================================================================
# Bags
# =============================================================================


@dataclass(frozen=True, slots=True)
class BagInit(BagObjDef):
    """A bag explicitly defined by entity multiplicities.

    Example: bag B = {a: 2, b: 3}
    Use tuple of pairs for hashability; ordered by entity name for determinism.
    """

    entity_multiplicity: tuple[tuple[Entity, int], ...]


@dataclass(frozen=True, slots=True)
class BagChoose(BagObjDef):
    """A sub-bag chosen from a source bag.

    Example: bag C = choose 2 from B
    """

    source: ObjRef
    size: int | None = None


@dataclass(frozen=True, slots=True)
class BagUnion(BagObjDef):
    """Union of two bags (max multiplicity).

    Example: bag D = B union C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagAdditiveUnion(BagObjDef):
    """Additive union of two bags (sum multiplicities).

    Example: bag E = B add C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagIntersection(BagObjDef):
    """Intersection of two bags (min multiplicity).

    Example: bag F = B intersect C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagDifference(BagObjDef):
    """Difference of two bags.

    Example: bag G = B minus C
    """

    left: ObjRef
    right: ObjRef


@dataclass(frozen=True, slots=True)
class BagSupport(SetObjDef):
    """The support set of a bag (unique elements).

    Example: set S = support of B
    """

    source: ObjRef


# =============================================================================
# Functions
# =============================================================================


@dataclass(frozen=True, slots=True)
class FuncDef(FuncObjDef):
    """A function definition.

    Example: func f: A -> B [injective] [surjective]
    """

    domain: ObjRef
    codomain: ObjRef
    injective: bool = False
    surjective: bool = False


@dataclass(frozen=True, slots=True)
class FuncImage(SetObjDef):
    """The image of a function applied to an argument.

    Example: set I = f(A) or f(a) for entity a
    """

    func: ObjRef
    argument: ObjRef  # Set or Entity ref


@dataclass(frozen=True, slots=True)
class FuncInverseImage(SetObjDef):
    """The inverse image of a function.

    Example: set P = f^{-1}(B)
    """

    func: ObjRef
    argument: ObjRef


@dataclass(frozen=True, slots=True)
class FuncInverse(FuncObjDef):
    """The inverse function of f (f⁻¹ as a function object, domain/codomain swapped).

    This represents f⁻¹ as a function object, not as an image.
    The inverse function maps codomain(f) → domain(f).

    Example: func g = inverse of f
    """

    func: ObjRef


# =============================================================================
# Tuples
# =============================================================================


@dataclass(frozen=True, slots=True)
class TupleDef(ObjDef, Ordered):
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
class SequenceDef(ObjDef, Linear, Ordered):
    """A linear (non-circular) sequence definition.

    Example: seq S = sequence of A
             seq S = choose 3 sequence from A
    """

    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None
    flatten: ObjRef | None = None  # SetInit of position-index entities for Bag sources


@dataclass(frozen=True, slots=True)
class CircleDef(ObjDef, Linear, Ordered):
    """A circular sequence (rotation-equivalence; optional reflection symmetry).

    Example: circle C = circle of A
             circle C = circle of A reflectional
    """

    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None
    reflection: bool = False
    flatten: ObjRef | None = None


# =============================================================================
# Partitions
# =============================================================================


@dataclass(frozen=True, slots=True)
class PartitionDef(ObjDef, Grouped):
    """An unordered partition definition (set of parts; part order is irrelevant).

    Example: partition P = partition A into 3
    """

    source: ObjRef
    num_parts: int


@dataclass(frozen=True, slots=True)
class CompositionDef(ObjDef, Grouped):
    """An ordered partition definition (parts are indexed; order matters).

    Example: composition C = compose A into 3
    """

    source: ObjRef
    num_parts: int


@dataclass(frozen=True, slots=True)
class SetPartDef(SetObjDef, PartDef):
    """Reference to the i-th part of a set partition/composition.

    Created when the partition's source is a set-like object.
    Example: P[0], C[1]
    """

    partition: ObjRef
    index: int


@dataclass(frozen=True, slots=True)
class BagPartDef(BagObjDef, PartDef):
    """Reference to the i-th part of a bag partition/composition.

    Created when the partition's source is a bag-like object.
    Example: P[0], C[1]
    """

    partition: ObjRef
    index: int


# =============================================================================
# Container alias
# =============================================================================
#
# `ContainerObjDef` is kept as a union alias for type hints; runtime
# isinstance checks use the base classes directly. After Phase F, this
# alias may be retired in favour of `(SetLike, Ordered)` checks.
ContainerObjDef = SetObjDef | TupleDef | SequenceDef | CircleDef
