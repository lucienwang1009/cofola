"""Planning-layer APIs for analysis, rewrites, and solve scheduling."""

from .analysis import (
    AnalysisResult,
    BagClassification,
    BagInfo,
    EntityAnalysis,
    MaxSizeInference,
    MergedAnalysis,
    SetInfo,
)
from .pass_manager import (
    AnalysisManager,
    AnalysisPass,
    FixedPointPass,
    PassResult,
    RefAllocator,
    TransformPass,
    UnsatisfiableConstraint,
)
from .passes import (
    ConstantFolder,
    FullChoiceOptimizer,
    LoweringPass,
    MergeIdenticalObjects,
    SimplifyPass,
)
from .pipeline import PlaningPipeline, PlanningProfile, SolveBranch, SolveSchedule

__all__ = [
    "AnalysisResult",
    "BagClassification",
    "BagInfo",
    "EntityAnalysis",
    "MaxSizeInference",
    "MergedAnalysis",
    "SetInfo",
    "AnalysisManager",
    "AnalysisPass",
    "FixedPointPass",
    "PassResult",
    "RefAllocator",
    "TransformPass",
    "UnsatisfiableConstraint",
    "ConstantFolder",
    "FullChoiceOptimizer",
    "LoweringPass",
    "MergeIdenticalObjects",
    "SimplifyPass",
    "PlaningPipeline",
    "PlanningProfile",
    "SolveBranch",
    "SolveSchedule",
]
