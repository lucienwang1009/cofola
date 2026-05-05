"""Tests for TypeCheckPass — Phase 1 of the parser type-checking refactor.

Each parameterized case asserts that `parse_and_solve(program)` raises
CofolaTypeError and that the formatted message contains an expected
substring identifying the violated rule.
"""

from __future__ import annotations

import pytest

from cofola.ir.analysis.type_check import CofolaTypeError
from cofola.solver import parse_and_solve


# Each case: (label, program, expected substring in error message).
TYPE_ERROR_CASES: list[tuple[str, str, str]] = [
    # 1) tuple() over a partition — partition is not Set/Bag.
    (
        "tuple_of_partition",
        """
S = set(a, b, c, d)
P = partition(S, 2)
T = tuple(P)
""",
        "TupleDef",
    ),
    # 2) sequence() over a tuple — sequence source must be SET_LIKE.
    (
        "sequence_of_tuple",
        """
S = set(a, b, c)
T = tuple(S)
seq = sequence(T)
""",
        "requires Bag or Set",
    ),
    # 3) together() over an entity (entities are not Set/Bag).
    (
        "together_of_entity",
        """
S = set(a, b, c)
arr = sequence(S)
together(a) in arr
""",
        "together",
    ),
    # 4) Cross-kind subset (set subset bag).
    (
        "set_subset_bag",
        """
S = set(a, b)
B = bag(a, b)
S subset B
""",
        "subset",
    ),
    # 5) disjoint between Tuple and Set (Tuple is not set-like).
    (
        "disjoint_tuple_set",
        """
S = set(a, b)
T = set(c, d)
T1 = tuple(S)
T1 disjoint T
""",
        "Bag or Set",
    ),
    # 6) set == bag — pure EqualityConstraint should reject mixed kinds.
    (
        "set_eq_bag",
        """
S = set(a, b)
B = bag(a, b)
S == B
""",
        "Set",
    ),
    # 7) Tuple == Set — equality is for sets only.
    (
        "tuple_eq_set",
        """
S = set(a, b)
T = tuple(S)
T == S
""",
        "Set",
    ),
    # 8) Circle with `<` pattern — circles reject ordering patterns.
    (
        "circle_less_than",
        """
S = set(a, b, c, d)
C = circle(S)
a < b in C
""",
        "Circle",
    ),
    # 9) seq.count(together(...)) — together has no count variant.
    (
        "seq_count_together",
        """
S = set(a, b, c, d)
seq = sequence(S)
seq.count(together(S)) > 0
""",
        "together",
    ),
    # 10) Indexing into an unordered Partition — must use Composition.
    (
        "partition_indexing",
        """
S = set(a, b, c, d)
P = partition(S, 2)
|P[0]| == 2
""",
        "unordered Partition",
    ),
    # 11) entity in Tuple — only Set/Bag containers per CONTAINER = SET_LIKE.
    (
        "entity_in_tuple",
        """
S = set(a, b, c)
T = tuple(S)
a in T
""",
        "Set",
    ),
    # 12) T[i] in tuple — TupleIndexMembership requires SET_LIKE container.
    (
        "tuple_index_in_tuple",
        """
S = set(a, b, c)
T1 = tuple(S)
T2 = tuple(S)
T1[0] in T2
""",
        "TupleIndexMembership",
    ),
]


@pytest.mark.parametrize(
    "label,program,expected_substring",
    TYPE_ERROR_CASES,
    ids=[label for label, _, _ in TYPE_ERROR_CASES],
)
def test_type_error(label: str, program: str, expected_substring: str) -> None:
    """Each program must trigger a CofolaTypeError mentioning the rule."""
    with pytest.raises(CofolaTypeError) as exc_info:
        parse_and_solve(program)
    msg = str(exc_info.value)
    assert expected_substring.lower() in msg.lower(), (
        f"[{label}] expected substring {expected_substring!r} in error message, "
        f"got:\n{msg}"
    )
