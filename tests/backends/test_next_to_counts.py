"""Undirected adjacency counts for the direct ASP and Essence backends."""
from __future__ import annotations

import os
import shutil
from itertools import product
from pathlib import Path

import pytest

from cofola.backend.asp.backend import ASPBackend
from cofola.backend.base import Backend
from cofola.backend.essence.backend import EssenceBackend
from cofola.backend.essence.encoder import EssenceEncoder
from cofola.parser.parser import parse
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


@pytest.fixture(scope="module", params=["asp", "essence"])
def backend(request: pytest.FixtureRequest) -> Backend:
    if request.param == "asp":
        return ASPBackend()
    configured = os.environ.get("COFOLA_CONJURE_DIR")
    executable = shutil.which("conjure")
    if configured is not None:
        directory = Path(configured)
    elif executable is not None:
        directory = Path(executable).parent
    else:
        bundled = sorted(
            (Path(__file__).resolve().parents[2] / ".tools" / "conjure").glob(
                "conjure-v*-*-combined/conjure"
            )
        )
        if not bundled:
            pytest.skip("Conjure is not installed")
        directory = bundled[0].parent
    return EssenceBackend(conjure_dir=directory, timeout=30.0)


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("a", "a", id="repeated-entity"),
        pytest.param("ab", "ab", id="identical-groups"),
        pytest.param("ab", "bc", id="overlapping-groups"),
        pytest.param("a", "ab", id="entity-in-group"),
        pytest.param("a", "b", id="disjoint-forward"),
        pytest.param("b", "a", id="disjoint-reverse"),
    ],
)
def test_next_to_counts_match_brute_force(
    backend: Backend, first: str, second: str, target: int,
) -> None:
    """Count edges, not the number of orientations satisfying the endpoint test."""
    def argument(entities: str) -> str:
        return entities if len(entities) == 1 else f"set({', '.join(entities)})"

    expected = sum(
        sum(
            (left in first and right in second)
            or (left in second and right in first)
            for left, right in zip(row, row[1:])
        ) == target
        for row in product("abc", repeat=3)
    )
    source = (
        "S = set(a, b, c)\nrow = choose_replace_sequence(S, 3)\n"
        f"row.count(next_to({argument(first)}, {argument(second)})) == {target}\n"
    )
    assert parse_and_solve(source, backend=backend) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "S = set(a, b)\nrow = sequence(S)\n"
            "row.count(next_to(S, S)) == 1\n", 2, id="two-position-group",
        ),
        pytest.param(
            "B = bag(a: 2, b)\nrow = sequence(B)\n"
            "row.count(next_to(a, a)) == 1\n", 2, id="literal-bag",
        ),
        pytest.param(
            "B = bag(a: 2, b)\nrow = sequence(B)\n"
            "row.count(next_to(a, a)) == 1\n"
            "row.count(next_to(a, b)) == 1\n", 2, id="multiple-counts",
        ),
        pytest.param(
            "S = set(a, b)\nrow = choose_replace_sequence(S, 0)\n"
            "row.count(next_to(S, S)) == 0\n", 1, id="empty-sequence",
        ),
        pytest.param(
            "S = set(a, b)\nrow = choose_replace_sequence(S, 1)\n"
            "row.count(next_to(S, S)) == 0\n", 2, id="one-position",
        ),
        pytest.param(
            "S = set(a, b)\nrow = sequence(S)\n"
            "row.count(next_to(S, S)) != 1\n", 0, id="unequal-count",
        ),
        pytest.param(
            "S = set(a, b)\nrow = sequence(S)\n"
            "not (row.count(next_to(S, S)) != 1)\n", 2, id="negated-count",
        ),
    ],
)
def test_next_to_count_boundaries(
    backend: Backend, source: str, expected: int,
) -> None:
    assert parse_and_solve(source, backend=backend) == expected


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        ("next_to(a, b) in row", 2),
        ("next_to(b, a) in row", 2),
        ("not (next_to(a, b) in row)", 0),
        ("next_to(a, b) in row for each a", 2),
        ("next_to(a, b) in row for each b", 2),
        ("next_to(S, S) in row for each S", 2),
        ("not (next_to(S, S) in row for each S)", 0),
        ("row.count((a, b)) == 1", 1),
        ("row.count((b, a)) == 1", 1),
    ],
)
def test_next_to_count_preserves_directional_constraints(
    backend: Backend, constraint: str, expected: int,
) -> None:
    source = (
        "S = set(a, b)\nrow = sequence(S)\n"
        "row.count(next_to(S, S)) == 1\n" + constraint + "\n"
    )
    assert parse_and_solve(source, backend=backend) == expected


@pytest.mark.parametrize("length", [2, 3])
def test_essence_next_to_count_ors_orientations_before_to_int(length: int) -> None:
    """Always check the emitted count, even when external Conjure is unavailable."""
    source = (
        f"S = set(a, b)\nrow = choose_replace_sequence(S, {length})\n"
        f"row.count(next_to(S, S)) == {length - 1}\n"
    )
    schedule = PlaningPipeline(EssenceBackend().planning_profile()).process(parse(source))
    assert len(schedule.branches) == 1
    problem, analysis = schedule.branches[0].components[0]
    model = EssenceEncoder(problem, analysis).encode()
    expected_terms = [
        "toInt(or(["
        f"(o_1({pos}) in o_0 /\\ o_1({pos + 1}) in o_0), "
        f"(o_1({pos + 1}) in o_0 /\\ o_1({pos}) in o_0)]))"
        for pos in range(1, length)
    ]
    assert f"sum([{', '.join(expected_terms)}]) = {length - 1}" in model
