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
    # Primary set objects
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    SetObjDef,
    # Derived set objects (always produce a set)
    BagSupport,
    FuncImage,
    FuncInverseImage,
    DerivedSetObjDef,
    AnySetObjDef,
    # Bag objects
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagObjDef,
    # Function mapping objects
    FuncDef,
    FuncObjDef,
    # Tuple objects
    TupleDef,
    # Sequence objects
    SequenceDef,
    # Partition objects (PartRef is polymorphic)
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
    BagSubsetConstraint,
    BagEqConstraint,
    TupleCountAtom,
    SeqPatternCountAtom,
    BagCountAtom,
    SizeAtom,
    NotConstraint,
    AndConstraint,
    OrConstraint,
    ForAllParts,
    Constraint,
)

# Analysis and passes (local to ir/)
from .analysis import EntityAnalysis, MaxSizeInference, MergedAnalysis, BagClassification, AnalysisResult, SetInfo, BagInfo
from .passes import ConstantFolder, SimplifyPass, LoweringPass, MergeIdenticalObjects
from .pass_manager import AnalysisManager, AnalysisPass, TransformPass
from .pipeline import IRPipeline, SolveBranch, SolveSchedule

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
    "SetObjDef",
    # Derived set objects (always produce a set)
    "BagSupport",
    "FuncImage",
    "FuncInverseImage",
    "DerivedSetObjDef",
    "AnySetObjDef",
    # Bag objects
    "BagInit",
    "BagChoose",
    "BagUnion",
    "BagAdditiveUnion",
    "BagIntersection",
    "BagDifference",
    "BagObjDef",
    # Function mapping objects
    "FuncDef",
    "FuncObjDef",
    # Tuple objects
    "TupleDef",
    # Sequence objects
    "SequenceDef",
    # Partition objects (PartRef is polymorphic)
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
    # Pipeline
    "IRPipeline",
    "SolveBranch",
    "SolveSchedule",
    # Pass infrastructure
    "AnalysisPass",
    "TransformPass",
    "AnalysisManager",
    # Analysis
    "EntityAnalysis",
    "MaxSizeInference",
    "MergedAnalysis",
    "BagClassification",
    "AnalysisResult",
    "SetInfo",
    "BagInfo",
    # Passes
    "ConstantFolder",
    "MergeIdenticalObjects",
    "SimplifyPass",
    "LoweringPass",
]
