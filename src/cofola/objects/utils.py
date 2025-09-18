from __future__ import annotations
from typing import Callable
from symengine import Eq, Ne
from enum import Enum

AUX_COUNTER = 0
IDX_PREFIX = 'idx_'

def reset_aux_obj_counter(start_from: int = 0):
    global AUX_COUNTER
    AUX_COUNTER = start_from

def aux_obj_name() -> str:
    global AUX_COUNTER
    name = 'aux_' + str(AUX_COUNTER)
    AUX_COUNTER += 1
    return name


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
