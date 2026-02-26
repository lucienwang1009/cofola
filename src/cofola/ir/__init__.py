"""Cofola IR — analysis and processing of combinatorics problems.

This module provides:
- Problem definition types (re-exported from cofola.frontend for backward compat)
- Analysis passes: EntityAnalysis, BagClassification, MaxSizeInference
- Transformation passes: ConstantFolder, SimplifyPass, LoweringPass
- Pipeline: IRPipeline

Example usage::

    from cofola.ir import ProblemBuilder, SetInit, Entity, ObjRef

    builder = ProblemBuilder()
    ref = builder.add(SetInit(entities=frozenset([Entity('a'), Entity('b')])), name='A')
    problem = builder.build()
"""

# Re-export all problem definition types from cofola.frontend for backward compatibility
from cofola.frontend import (
    Entity,
    ObjRef,
    RefOrEntity,
    Problem,
    ProblemBuilder,
    # Set objects
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    SetObjDef,
    SetLikeObjDef,
    # Bag objects
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    BagObjDef,
    # Function objects
    FuncDef,
    FuncImage,
    FuncInverseImage,
    FuncObjDef,
    # Tuple objects
    TupleDef,
    # Sequence objects
    SequenceDef,
    # Partition objects
    PartitionDef,
    PartRef,
    # Union types
    ObjDef,
    ContainerObjDef,
    # Constraints
    SizeConstraint,
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    TogetherPattern,
    LessThanPattern,
    PredecessorPattern,
    NextToPattern,
    SeqPattern,
    SequencePatternConstraint,
    FuncPairConstraint,
    TupleCountAtom,
    SeqPatternCountAtom,
    BagCountAtom,
    SizeAtom,
    NotConstraint,
    AndConstraint,
    OrConstraint,
    ForAllParts,
    Constraint,
    # Rewriter
    Rewriter,
    RewriterWithSubstitution,
)

# Analysis and passes (local to ir/)
from .analysis import EntityAnalysis, MaxSizeInference, BagClassification, AnalysisResult, SetInfo, BagInfo
from .passes import ConstantFolder, SimplifyPass, LoweringPass
from .pipeline import IRPipeline

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
    # Rewriter
    "Rewriter",
    "RewriterWithSubstitution",
    # Pipeline
    "IRPipeline",
    # Analysis
    "EntityAnalysis",
    "MaxSizeInference",
    "BagClassification",
    "AnalysisResult",
    "SetInfo",
    "BagInfo",
    # Passes
    "ConstantFolder",
    "SimplifyPass",
    "LoweringPass",
]
