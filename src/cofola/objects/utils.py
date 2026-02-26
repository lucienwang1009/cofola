from __future__ import annotations

from typing import Callable
from sympy import Eq, Ne
from enum import Enum


class _AuxCounter:
    """Encapsulated counter for generating unique auxiliary object names."""
    _count: int = 0

    @classmethod
    def next_name(cls) -> str:
        """Get the next auxiliary object name and increment the counter."""
        name = f"AUX_{cls._count}"
        cls._count += 1
        return name

    @classmethod
    def reset(cls, start_from: int = 0) -> None:
        """Reset the counter to a specific starting value."""
        cls._count = start_from


IDX_PREFIX = 'IDX_'


def reset_aux_obj_counter(start_from: int = 0):
    """Reset the auxiliary object counter."""
    _AuxCounter.reset(start_from)


def aux_obj_name() -> str:
    """Generate a unique auxiliary object name."""
    return _AuxCounter.next_name()


def parse_comparator(comparator: str) -> Callable:
    if comparator == '==':
        return lambda a, b: Eq(a, b)
    if comparator == '!=':
        return lambda a, b: Ne(a, b)
    if comparator == '<':
        return lambda a, b: a < b
    if comparator == '<=':
        return lambda a, b: a <= b
    if comparator == '>':
        return lambda a, b: a > b
    if comparator == '>=':
        return lambda a, b: a >= b
    raise ValueError(f"Unknown comparator: {comparator}")


def invert_comparator(comparator: str) -> str:
    if comparator == '==':
        return '!='
    if comparator == '!=':
        return '=='
    if comparator == '<':
        return '>='
    if comparator == '<=':
        return '>'
    if comparator == '>':
        return '<='
    if comparator == '>=':
        return '<'
    raise ValueError(f"Unknown comparator: {comparator}")


class Quantifier(Enum):
    FORALL = 1
    EXISTS = 2

    def __str__(self) -> str:
        return self.name.lower()
