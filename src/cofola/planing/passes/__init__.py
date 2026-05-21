"""Rewriter passes for planning Problems.

This module provides the rewriter framework for IR-to-IR transformations:
- Rewriter: Base class for transformation passes
- ConstantFolder: Folds constant expressions
- SimplifyPass: Removes unused objects
- LoweringPass: Runs fine-grained lowering steps under one fixed-point driver
"""

from .optimize import ConstantFolder
from .simplify import SimplifyPass
from .lowering import (
    ForAllPartsExpansionStep,
    InjectiveFunctionLoweringStep,
    LinearDefLoweringStep,
    LoweringPass,
    LoweringStep,
    TupleCountAtomLoweringStep,
    TupleDefLoweringStep,
)
from .merge_identical import MergeIdenticalObjects

__all__ = [
    "ConstantFolder",
    "ForAllPartsExpansionStep",
    "InjectiveFunctionLoweringStep",
    "LinearDefLoweringStep",
    "SimplifyPass",
    "LoweringPass",
    "LoweringStep",
    "MergeIdenticalObjects",
    "TupleCountAtomLoweringStep",
    "TupleDefLoweringStep",
]
