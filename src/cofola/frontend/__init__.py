"""Cofola frontend — combinatorial problem definition types.

This module defines the public types for representing combinatorics counting
problems as immutable data structures. It is the stable API for constructing
and inspecting problems.

Example usage::

    from cofola.frontend import ProblemBuilder, SetInit, Entity, ObjRef

    builder = ProblemBuilder()
    ref = builder.add(SetInit(entities=frozenset([Entity('a'), Entity('b')])), name='A')
    problem = builder.build()
"""

from .objects import (
    # Core types
    Entity,
    ObjRef,
    RefOrEntity,
    # Spec-type ABCs (group + leaf bases)
    ObjDef,
    SetObjDef,
    BagObjDef,
    FuncObjDef,
    SetLike,
    Linear,
    Grouped,
    Ordered,
    PartDef,
    PartPlaceholderDef,
    ContainerObjDef,
    # Primary sets
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    # Derived sets (always produce a set, but derived from bags/functions)
    BagSupport,
    FuncImage,
    FuncInverseImage,
    # Bags
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    # Function mappings
    FuncDef,
    # Tuples
    TupleDef,
    # Sequences / circles
    SequenceDef,
    CircleDef,
    # Partitions / compositions
    PartitionDef,
    CompositionDef,
    # Partition parts (set vs bag for precise dispatch)
    SetPartDef,
    BagPartDef,
)
from .constraints import (
    # Size constraints
    SizeConstraint,
    # Membership constraints
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    # Tuple constraints
    TupleIndexEq,
    TupleIndexMembership,
    # Sequence patterns
    TogetherPattern,
    LessThanPattern,
    PredecessorPattern,
    NextToPattern,
    SeqPattern,
    SequencePatternConstraint,
    # Function constraints
    FuncPairConstraint,
    # Bag constraints
    BagSubsetConstraint,
    BagEqConstraint,
    # Size atoms
    TupleCountAtom,
    SeqPatternCountAtom,
    BagCountAtom,
    SizeAtom,
    # Compound constraints
    NotConstraint,
    AndConstraint,
    OrConstraint,
    # Partition constraints
    ForAllParts,
    # Union type
    CONSTRAINT_CLASSES,
    Constraint,
)
from .problem import (
    Problem,
    ProblemBuilder,
)
from .utils import (
    constraint_refs,
    iter_refs,
    map_refs,
    object_refs,
)
from .type_check import CofolaTypeError, TypeCheckError, TypeChecker, validate_problem
from .pretty import fmt_problem, fmt_analysis

__all__ = [
    # Core types
    "Entity",
    "ObjRef",
    "RefOrEntity",
    "iter_refs",
    "map_refs",
    "object_refs",
    "constraint_refs",
    # Problem
    "Problem",
    "ProblemBuilder",
    "TypeChecker",
    "TypeCheckError",
    "CofolaTypeError",
    "validate_problem",
    # Spec-type ABCs (group + leaf bases)
    "ObjDef",
    "SetObjDef",
    "BagObjDef",
    "FuncObjDef",
    "SetLike",
    "Linear",
    "Grouped",
    "Ordered",
    "PartDef",
    "PartPlaceholderDef",
    "ContainerObjDef",
    # Primary set objects
    "SetInit",
    "SetChoose",
    "SetChooseReplace",
    "SetUnion",
    "SetIntersection",
    "SetDifference",
    # Derived set objects (always produce a set)
    "BagSupport",
    "FuncImage",
    "FuncInverseImage",
    # Bag objects
    "BagInit",
    "BagChoose",
    "BagUnion",
    "BagAdditiveUnion",
    "BagIntersection",
    "BagDifference",
    # Function mapping objects
    "FuncDef",
    # Tuple objects
    "TupleDef",
    # Sequence / circle objects
    "SequenceDef",
    "CircleDef",
    # Partition / composition objects
    "PartitionDef",
    "CompositionDef",
    # Partition parts (set vs bag for precise dispatch)
    "SetPartDef",
    "BagPartDef",
    # Constraints
    "SizeConstraint",
    "MembershipConstraint",
    "SubsetConstraint",
    "DisjointConstraint",
    "EqualityConstraint",
    "TupleIndexEq",
    "TupleIndexMembership",
    # Sequence patterns
    "TogetherPattern",
    "LessThanPattern",
    "PredecessorPattern",
    "NextToPattern",
    "SeqPattern",
    "SequencePatternConstraint",
    # Function constraints
    "FuncPairConstraint",
    # Bag constraints
    "BagSubsetConstraint",
    "BagEqConstraint",
    # Size atoms
    "TupleCountAtom",
    "SeqPatternCountAtom",
    "BagCountAtom",
    "SizeAtom",
    # Compound constraints
    "NotConstraint",
    "AndConstraint",
    "OrConstraint",
    # Partition constraints
    "ForAllParts",
    # Constraint union type
    "CONSTRAINT_CLASSES",
    "Constraint",
    # Pretty-printing
    "fmt_problem",
    "fmt_analysis",
]
