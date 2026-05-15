"""Planning configuration for the CoSo backend."""
from __future__ import annotations

from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass

__all__ = ["COSO_LOCAL_PASSES"]


# CoSo can directly represent tuple/permutation configurations with absolute
# positional and counting constraints. Preserve frontend TupleDef nodes instead
# of lowering them to FuncDef, and let the encoder reject sequence/circle
# constructs with relative positional constraints.
COSO_LOCAL_PASSES = [
    SizeConstraintFolder,
    MergeIdenticalObjects,
    SimplifyPass,
]
