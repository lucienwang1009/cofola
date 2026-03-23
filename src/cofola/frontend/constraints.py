"""Constraint definitions for the immutable IR.

All constraints are frozen dataclasses. They represent the constraints
on objects in a combinatorics problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from cofola.frontend.types import Entity, ObjRef


# =============================================================================
# Size Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class SizeConstraint:
    """A linear size constraint.

    Example: |A| + 2*|B| == 5
    """

    terms: tuple[tuple[ObjRef, int], ...]  # (obj_ref, coefficient)
    comparator: str  # "==", "<", "<=", ">", ">="
    rhs: int


# =============================================================================
# Membership Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class MembershipConstraint:
    """Entity membership in a container.

    Example: a in A, a not in B
    """

    entity: Entity
    container: ObjRef
    positive: bool = True


@dataclass(frozen=True, slots=True)
class SubsetConstraint:
    """Subset relationship between containers.

    Example: A subset B, A not subset C
    """

    sub: ObjRef
    sup: ObjRef
    positive: bool = True


@dataclass(frozen=True, slots=True)
class DisjointConstraint:
    """Disjointness constraint between containers.

    Example: A disjoint B, A not disjoint C
    """

    left: ObjRef
    right: ObjRef
    positive: bool = True


@dataclass(frozen=True, slots=True)
class EqualityConstraint:
    """Equality constraint between objects.

    Example: A == B, A != C
    """

    left: ObjRef
    right: ObjRef
    positive: bool = True


# =============================================================================
# Tuple Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class TupleIndexEq:
    """Tuple index equality constraint.

    Example: T[0] == a, T[1] != b
    """

    tuple_ref: ObjRef
    index: int
    entity: Entity
    positive: bool = True


@dataclass(frozen=True, slots=True)
class TupleIndexMembership:
    """Tuple index membership constraint.

    Example: T[0] in A, T[1] not in B
    """

    tuple_ref: ObjRef
    index: int
    container: ObjRef
    positive: bool = True


# =============================================================================
# Sequence Patterns
# =============================================================================


@dataclass(frozen=True, slots=True)
class TogetherPattern:
    """Entities that must appear together in a sequence.

    Example: A together in S
    """

    group: ObjRef


@dataclass(frozen=True, slots=True)
class LessThanPattern:
    """Ordering constraint: left < right in sequence.

    Example: a < b in S
    """

    left: ObjRef | Entity
    right: ObjRef | Entity


@dataclass(frozen=True, slots=True)
class PredecessorPattern:
    """Predecessor constraint: first immediately precedes second.

    Example: a predecessor b in S
    """

    first: ObjRef | Entity
    second: ObjRef | Entity


@dataclass(frozen=True, slots=True)
class NextToPattern:
    """Next-to constraint: first and second are adjacent.

    Example: a nextto b in S
    """

    first: ObjRef | Entity
    second: ObjRef | Entity


# Union type for all sequence patterns
SeqPattern = TogetherPattern | LessThanPattern | PredecessorPattern | NextToPattern


@dataclass(frozen=True, slots=True)
class SequencePatternConstraint:
    """Sequence pattern constraint.

    Example: a < b in S, A together in S
    """

    seq: ObjRef
    pattern: SeqPattern
    positive: bool = True


# =============================================================================
# Function Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class FuncPairConstraint:
    """Function pair constraint.

    Example: f(a) == b, f(a) in A
    """

    func: ObjRef
    arg_entity: Entity
    result: ObjRef | Entity
    positive: bool = True


# =============================================================================
# Size Atoms (for ordered objects)
# =============================================================================


@dataclass(frozen=True, slots=True)
class TupleCountAtom:
    """Size atom: T.count(S) — positions in T occupied by elements of S.

    Example: T.count(A) == 2
    """

    tuple_ref: ObjRef
    count_obj: ObjRef
    deduplicate: bool = False


@dataclass(frozen=True, slots=True)
class SeqPatternCountAtom:
    """Size atom: seq.count(pattern) — occurrences of pattern in seq.

    Example: S.count(a < b) == 3
    """

    seq: ObjRef
    pattern: SeqPattern


@dataclass(frozen=True, slots=True)
class BagCountAtom:
    """Size atom: B.count(e) — multiplicity of entity e in bag B.

    Example: B.count(a) == 2
    """

    bag: ObjRef
    entity: Entity


# Union type for size atoms
SizeAtom = TupleCountAtom | SeqPatternCountAtom | BagCountAtom


# =============================================================================
# Compound Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class NotConstraint:
    """Negation of a constraint.

    Example: not (A subset B)
    """

    sub: Constraint


@dataclass(frozen=True, slots=True)
class AndConstraint:
    """Conjunction of constraints.

    Example: A subset B and B subset C
    """

    left: Constraint
    right: Constraint


@dataclass(frozen=True, slots=True)
class OrConstraint:
    """Disjunction of constraints.

    Example: A subset B or A subset C
    """

    left: Constraint
    right: Constraint


# =============================================================================
# Partition Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class ForAllParts:
    """Constraint applied to every part of a partition.

    Example: A subset part for part in P
    The constraint_template contains a sentinel PartRef that will be
    instantiated for each part of the partition.
    """

    partition: ObjRef
    constraint_template: Constraint


# =============================================================================
# Union Types
# =============================================================================

# Forward declaration for recursive type
# =============================================================================
# Bag Subset / Equality Constraints
# =============================================================================


@dataclass(frozen=True, slots=True)
class BagSubsetConstraint:
    """Bag subset constraint (by multiplicity): sub ⊆ sup.

    Each entity's multiplicity in sub must be <= that in sup.
    This is the bag analogue of SubsetConstraint for sets.

    Example: B subset C  (where B and C are bags)
    """

    sub: ObjRef
    sup: ObjRef
    positive: bool = True


@dataclass(frozen=True, slots=True)
class BagEqConstraint:
    """Bag equality constraint (by multiplicity): left == right.

    Each entity's multiplicity must be identical in both bags.
    This is the bag analogue of EqualityConstraint for sets.

    Example: B == C  (where B and C are bags)
    """

    left: ObjRef
    right: ObjRef
    positive: bool = True


Constraint = Union[
    SizeConstraint,
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    SequencePatternConstraint,
    FuncPairConstraint,
    BagSubsetConstraint,
    BagEqConstraint,
    NotConstraint,
    AndConstraint,
    OrConstraint,
    ForAllParts,
]