"""A one-part grouping preserves its source bag, including multiplicities."""
from __future__ import annotations

import pytest

from cofola.solver import parse_and_solve


@pytest.mark.parametrize("grouping", ["compose", "partition"])
@pytest.mark.parametrize("lifted_bags", [False, True])
@pytest.mark.parametrize(
    ("bag", "constraint"),
    [
        ("bag(a: 2)", "|part| == 2"),
        ("bag(a: 2)", "part.count(a) == 2"),
        ("bag(a: 2, b: 1)", "|part| == 3"),
    ],
)
def test_single_part_bag_preserves_fixed_multiplicities(
    grouping: str, lifted_bags: bool, bag: str, constraint: str,
) -> None:
    assert parse_and_solve(
        f"B = {bag}\nG = {grouping}(B, 1)\n{constraint} for part in G\n",
        lifted_bags=lifted_bags,
    ) == 1


@pytest.mark.parametrize("grouping", ["compose", "partition"])
@pytest.mark.parametrize("lifted_bags", [False, True])
@pytest.mark.parametrize(
    ("bag", "size", "expected"),
    [
        ("bag(a: 2)", 0, 1),
        ("bag(a: 2)", 1, 1),
        ("bag(a: 2)", 2, 1),
        ("bag(a: 2, b: 1)", 1, 2),
        ("bag(a: 1, b: 1)", 1, 2),
    ],
)
def test_single_part_bag_tracks_chosen_source(
    grouping: str, lifted_bags: bool, bag: str, size: int, expected: int,
) -> None:
    """The part equals the chosen bag, not its structural upper bound."""
    assert parse_and_solve(
        f"B = {bag}\nX = choose(B)\nG = {grouping}(X, 1)\n"
        f"|part| == {size} for part in G\n",
        lifted_bags=lifted_bags,
    ) == expected


@pytest.mark.parametrize("lifted_bags", [False, True])
@pytest.mark.parametrize(("source", "expected"), [("B", 3), ("choose(B)", 6)])
def test_choice_from_single_bag_part(
    lifted_bags: bool, source: str, expected: int,
) -> None:
    """A part is also a valid bag source for a subsequent choice."""
    assert parse_and_solve(
        f"B = bag(a: 2)\nX = {source}\nG = compose(X, 1)\nC = choose(G[0])\n",
        lifted_bags=lifted_bags,
    ) == expected
