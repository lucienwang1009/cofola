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

from .types import Entity, ObjRef, RefOrEntity
from .objects import (
    # Sets
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    SetObjDef,
    SetLikeObjDef,
    # Bags
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    BagObjDef,
    # Functions
    FuncDef,
    FuncImage,
    FuncInverseImage,
    FuncObjDef,
    # Tuples
    TupleDef,
    # Sequences
    SequenceDef,
    # Partitions
    PartitionDef,
    PartRef,
    # Union types
    ObjDef,
    ContainerObjDef,
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
    Constraint,
)
from .problem import Problem, ProblemBuilder
from .pretty import fmt_problem, fmt_analysis

__all__ = [
    # Core types
    "Entity",
    "ObjRef",
    "RefOrEntity",
    # Problem
    "Problem",
    "ProblemBuilder",
    # Set objects
    "SetInit",
    "SetChoose",
    "SetChooseReplace",
    "SetUnion",
    "SetIntersection",
    "SetDifference",
    "SetObjDef",
    "SetLikeObjDef",
    # Bag objects
    "BagInit",
    "BagChoose",
    "BagUnion",
    "BagAdditiveUnion",
    "BagIntersection",
    "BagDifference",
    "BagSupport",
    "BagObjDef",
    # Function objects
    "FuncDef",
    "FuncImage",
    "FuncInverseImage",
    "FuncObjDef",
    # Tuple objects
    "TupleDef",
    # Sequence objects
    "SequenceDef",
    # Partition objects
    "PartitionDef",
    "PartRef",
    # Union types
    "ObjDef",
    "ContainerObjDef",
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
    "Constraint",
    # Pretty-printing
    "fmt_problem",
    "fmt_analysis",
]
