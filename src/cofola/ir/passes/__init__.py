"""Rewriter passes for the immutable IR.

This module provides the rewriter framework for IR-to-IR transformations:
- Rewriter: Base class for transformation passes
- ConstantFolder: Folds constant expressions
- SimplifyPass: Removes unused objects
- LoweringPass: Lowers high-level constructs (tuple→func, etc.)
"""

from .optimize import ConstantFolder
from .simplify import SimplifyPass
from .lowering import LoweringPass

__all__ = [
    "ConstantFolder",
    "SimplifyPass",
    "LoweringPass",
]