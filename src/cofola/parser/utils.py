"""Parser utility definitions shared by the parser transformer modules.

This module intentionally sits at the bottom of the parser package import
graph. It contains parser-only constants, exceptions, and temporary helper
objects that are safe for the transformer and mixins to import.
"""
from __future__ import annotations

from cofola.frontend.objects import ObjRef


RESERVED_KEYWORDS = [
    "set",
    "bag",
    "choose",
    "choose_replace",
    "count",
    "in",
    "subset",
    "disjoint",
    "supp",
    "compose",
    "partition",
    "tuple",
    "choose_tuple",
    "choose_replace_tuple",
    "sequence",
    "choose_sequence",
    "choose_replace_sequence",
    "circle",
    "choose_circle",
    "choose_replace_circle",
    "together",
    "next_to",
    "not",
    "and",
    "or",
    "dedup_count",
    "for",
    "true",
    "false",
    "True",
    "False",
]

RESERVED_PREFIXES = [
    "AUX_",
    "IDX_",
]


class CofolaParsingError(Exception):
    """Base exception for Cofola parsing errors."""


class TupleIndexSentinel:
    """Sentinel for tuple index expressions (T[i]) used in constraints.

    Not a real frontend object — only lives during parsing to carry tuple_ref and index
    until a constraint is built.
    """

    __slots__ = ("tuple_ref", "index")

    def __init__(self, tuple_ref: ObjRef, index: int) -> None:
        self.tuple_ref = tuple_ref
        self.index = index

    def __repr__(self) -> str:
        return f"TupleIndexSentinel({self.tuple_ref}, {self.index})"
