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
    # Object kind aliases
    SetObjDef,
    BagObjDef,
    # Function mappings
    FuncDef,
    FuncObjDef,
    # Tuples
    TupleDef,
    # Sequences
    SequenceDef,
    # Partitions (set vs bag PartRefs are split for precise dispatch)
    PartitionDef,
    SetPartRef,
    BagPartRef,
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
    # Object kind aliases
    "SetObjDef",
    "BagObjDef",
    # Function mapping objects
    "FuncDef",
    "FuncObjDef",
    # Tuple objects
    "TupleDef",
    # Sequence objects
    "SequenceDef",
    # Partition objects (set vs bag PartRefs are split)
    "PartitionDef",
    "SetPartRef",
    "BagPartRef",
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
