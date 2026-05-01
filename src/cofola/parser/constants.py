"""Parser-level constants and intermediate sentinels.

Lives at the bottom of the parser package's import graph so that both the
top-level transformer and its mixins can import from here without creating
cycles.
"""
from __future__ import annotations

from cofola.frontend.types import ObjRef


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
    "together",
    "not",
    "and",
    "or",
]

RESERVED_PREFIXES = [
    "AUX_",
    "IDX_",
]


class TupleIndexSentinel:
    """Sentinel for tuple index expressions (T[i]) used in constraints.

    Not a real IR object — only lives during parsing to carry tuple_ref and index
    until a constraint is built.
    """

    __slots__ = ("tuple_ref", "index")

    def __init__(self, tuple_ref: ObjRef, index: int) -> None:
        self.tuple_ref = tuple_ref
        self.index = index

    def __repr__(self) -> str:
        return f"TupleIndexSentinel({self.tuple_ref}, {self.index})"
