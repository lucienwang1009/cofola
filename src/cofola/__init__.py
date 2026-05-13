"""Cofola - Combinatorial counting with First-Order Logic Language.

A DSL and solver for combinatorial counting problems via Weighted First-Order
Model Counting (WFOMC).
"""

from cofola.solver import solve, parse_and_solve
from cofola.parser.parser import parse
from cofola.frontend import CofolaTypeError, validate_problem

__all__ = [
    "solve",
    "parse_and_solve",
    "parse",
    "validate_problem",
    "CofolaTypeError",
]
