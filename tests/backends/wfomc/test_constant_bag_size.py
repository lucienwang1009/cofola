"""Fixed bags carry literal multiplicities, not weighted entity variables."""
from __future__ import annotations

import pytest

from cofola.backend.wfomc.encoder import encode
from cofola.parser.parser import parse
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


@pytest.mark.parametrize(("bag", "size"), [
    ("bag(a: 2)", 2),
    ("bag(a: 2, b: 1)", 3),
    ("bag(a: 2, b: 2)", 4),
    ("bag(a: 1, b: 1)", 2),
    ("bag(a: 2) - bag(a: 2)", 0),
])
@pytest.mark.parametrize(("comparator", "delta", "expected"), [
    ("==", 0, 1), ("==", 1, 0), ("<=", 0, 1),
    (">=", 0, 1), ("<", 0, 0), (">", 0, 0),
])
def test_constant_bag_size(bag: str, size: int, comparator: str, delta: int, expected: int) -> None:
    assert parse_and_solve(f"B = {bag}\n|B| {comparator} {size + delta}") == expected


@pytest.mark.parametrize("constraint", [
    "not (|B| == 2)",
    "(|B| == 2) or (|B| == 3)",
    "(|B| >= 3) and (|B| <= 3)",
    "2 |B| == 6",
])
def test_constant_bag_compound_and_linear_constraints(constraint: str) -> None:
    assert parse_and_solve(f"B = bag(a: 2, b: 1)\n{constraint}") == 1


def test_folded_constant_bag_size() -> None:
    assert parse_and_solve("B = bag(a: 2, b: 1)\nD = B & B\n|D| == 3") == 1


def test_constant_size_does_not_change_dynamic_choice() -> None:
    assert parse_and_solve("B = bag(a: 2, b: 1)\nC = choose(B)\n|B| == 3\n|C| == 2") == 2


@pytest.mark.parametrize("lifted_bags", [False, True])
def test_constant_size_does_not_create_weight_variables(lifted_bags: bool) -> None:
    problem = parse("B = bag(a: 2, b: 1)\n|B| == 3")
    schedule = PlaningPipeline().process(problem)
    component, analysis = schedule.branches[0].components[0]
    _, decoder = encode(component, analysis, lifted_bags=lifted_bags)
    assert decoder.gens == []
    assert all(bool(validator) for validator in decoder.validator)
