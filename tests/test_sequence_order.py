"""Strict global sequence ordering and its Boolean negation."""
from __future__ import annotations

import pytest

from cofola.solver import parse_and_solve


@pytest.mark.parametrize(
    ("definitions", "relation", "positive", "negative"),
    [
        ("S = set(a)\nrow = sequence(S)\n", "a < a", 0, 1),
        ("A = set(a0, a1)\nB = set(b)\nrow = sequence(A + B)\n", "A < B", 2, 4),
        ("A = set(a)\nB = set(b)\nS = set(c)\nrow = sequence(S)\n", "A < B", 1, 0),
        ("A = set(a)\nS = set(b)\nrow = sequence(S)\n", "A < b", 1, 0),
        ("S = bag(a: 2, b: 1)\nrow = sequence(S)\n", "a < b", 1, 2),
        ("S = bag(a: 2, b: 1)\nrow = sequence(S)\n", "a < a", 0, 3),
        ("S = set(a, b)\nC = choose(S, 0)\nrow = sequence(C)\n", "a < b", 1, 0),
    ],
)
def test_global_order_and_negation_partition_solutions(
    definitions: str, relation: str, positive: int, negative: int,
) -> None:
    assert parse_and_solve(definitions + f"{relation} in row\n") == positive
    assert parse_and_solve(definitions + f"not ({relation} in row)\n") == negative
    assert parse_and_solve(definitions + f"{relation} not in row\n") == negative
    assert parse_and_solve(definitions) == positive + negative


def test_counted_order_is_also_strict() -> None:
    assert parse_and_solve(
        "S = set(a)\nrow = sequence(S)\nrow.count(a < a) == 0\n"
    ) == 1


def test_local_group_adjacency_keeps_existing_semantics() -> None:
    assert parse_and_solve(
        "A = set(a0, a1)\nS = A + set(b)\nrow = sequence(S)\n"
        "next_to(A, b) in row\n"
    ) == 2
