"""WFOMC regressions for Boolean combinations of size comparisons."""
from __future__ import annotations

import pytest

from cofola.solver import parse_and_solve


@pytest.mark.parametrize(
    ("comparator", "expected"),
    [("<", 7), ("<=", 4), (">", 4), (">=", 1), ("==", 5), ("!=", 3)],
)
def test_negated_size_comparison_count(comparator: str, expected: int) -> None:
    assert parse_and_solve(
        f"S = set(a, b, c)\nA = choose(S)\nnot (|A| {comparator} 1)"
    ) == expected


@pytest.mark.parametrize(("size", "expected"), [(0, 1), (3, 1), (4, 0)])
def test_negated_size_inequality_at_boundaries(size: int, expected: int) -> None:
    assert parse_and_solve(
        f"S = set(a, b, c)\nA = choose(S)\nnot (|A| != {size})"
    ) == expected


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        ("(|A| != 1) or (a in A)", 6),
        ("not ((|A| != 1) or (a in A))", 2),
        ("not ((|A| != 1) and (a in A))", 5),
        ("not (not (|A| != 1))", 5),
    ],
)
def test_size_inequality_boolean_count(constraint: str, expected: int) -> None:
    """Shannon branches must complement != even without an explicit not."""
    assert parse_and_solve(
        f"S = set(a, b, c)\nA = choose(S)\n{constraint}"
    ) == expected


def test_negated_size_inequality_with_multiple_terms() -> None:
    """2|A| - |B| = 1 forces both subsets to have size one, giving four ways."""
    assert parse_and_solve(
        "S = set(a, b)\nA = choose(S)\nB = choose(S)\n"
        "not (2|A| - |B| != 1)"
    ) == 4
