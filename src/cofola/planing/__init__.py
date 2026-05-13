"""Cofola planing layer — analysis and processing of combinatorics problems.

This module provides:
- Problem definition types (re-exported from cofola.frontend for backward compat)
- Analysis passes: EntityAnalysis, BagClassification, MaxSizeInference
- Transformation passes: ConstantFolder, SimplifyPass, LoweringPass
- Pipeline: PlaningPipeline

Example usage::

    from cofola.planing import ProblemBuilder, SetInit, Entity, ObjRef

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
    TypeChecker,
    TypeCheckError,
    CofolaTypeError,
    validate_problem,
    # Primary set objects
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    # Derived set objects (always produce a set)
    BagSupport,
    FuncImage,
    FuncInverseImage,
    # Bag objects
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    # Spec-type ABCs
    ObjDef,
    SetObjDef,
    BagObjDef,
    FuncObjDef,
    SetLike,
    Linear,
    Grouped,
    Ordered,
    PartDef,
    ContainerObjDef,
    # Function mapping objects
    FuncDef,
    # Tuple objects
    TupleDef,
    # Sequence / circle objects
    SequenceDef,
    CircleDef,
    # Partition / composition objects
    PartitionDef,
    CompositionDef,
    # Partition parts
    SetPartDef,
    BagPartDef,
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

# Analysis and passes (local to planing/)
from .analysis import EntityAnalysis, MaxSizeInference, MergedAnalysis, BagClassification, AnalysisResult, SetInfo, BagInfo
from .passes import ConstantFolder, SimplifyPass, LoweringPass, MergeIdenticalObjects
from .pass_manager import (
    AnalysisManager,
    AnalysisPass,
    FixedPointPass,
    PassResult,
    RefAllocator,
    TransformPass,
    UnsatisfiableConstraint,
)
from .pipeline import IRPipeline, PlaningPipeline, SolveBranch, SolveSchedule

__all__ = [
    # Core types
    "Entity",
    "ObjRef",
    "RefOrEntity",
    # Problem
    "Problem",
    "ProblemBuilder",
    "TypeChecker",
    "TypeCheckError",
    "CofolaTypeError",
    "validate_problem",
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
    # Spec-type ABCs
    "ObjDef",
    "SetObjDef",
    "BagObjDef",
    "FuncObjDef",
    "SetLike",
    "Linear",
    "Grouped",
    "Ordered",
    "PartDef",
    "ContainerObjDef",
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
    # Partition parts
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
    "Constraint",
    # Pipeline
    "PlaningPipeline",
    "IRPipeline",
    "SolveBranch",
    "SolveSchedule",
    # Pass infrastructure
    "AnalysisPass",
    "TransformPass",
    "PassResult",
    "FixedPointPass",
    "RefAllocator",
    "AnalysisManager",
    "UnsatisfiableConstraint",
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
