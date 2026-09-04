from __future__ import annotations

import pytest

from cofola.backend.unaryfomc import (
    UnaryFOMCUnsupportedError,
    encode_ir,
)
from cofola.ir.pipeline import IRPipeline
from cofola.parser.parser import parse
from cofola.solver import parse_and_solve
from unaryfomc import c1f_fomc


def _solve(program: str) -> int:
    return parse_and_solve(program, backend="unaryfomc")


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (
            "people = set(person0...4)\n"
            "chosen = choose(people, 2)",
            6,
        ),
        (
            "people = set(person0...4)\n"
            "chosen = choose(people)",
            16,
        ),
        (
            "people = set(a, b, c, d)\n"
            "chosen = choose(people)\n"
            "fixed = set(a, b)\n"
            "overlap = chosen & fixed\n"
            "|overlap| == 1",
            8,
        ),
        (
            "people = set(a, b, c)\n"
            "chosen = choose(people)\n"
            "(a in chosen) or (b in chosen)",
            6,
        ),
        (
            "people = set(a, b, c, d)\n"
            "left = choose(people)\n"
            "right = choose(people - left)\n"
            "left disjoint right\n"
            "a in left\n"
            "b not in right",
            18,
        ),
    ],
)
def test_basic_set_fragment(program: str, expected: int) -> None:
    assert _solve(program) == expected


def test_linear_constraint_over_multiple_cardinalities() -> None:
    program = (
        "boys = set(boy0...3)\n"
        "girls = set(girl0...4)\n"
        "club = boys + girls\n"
        "team = choose(club)\n"
        "team_boys = team & boys\n"
        "team_girls = team & girls\n"
        "|team_girls| - |team_boys| > 0"
    )

    # Sum C(3, b) C(4, g) over g > b.
    assert _solve(program) == 64


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        ("left subset right", 27),
        ("not (left subset right)", 37),
        ("left disjoint right", 27),
        ("not (left disjoint right)", 37),
        ("left == right", 8),
        ("left != right", 56),
    ],
)
def test_set_relations(relation: str, expected: int) -> None:
    program = (
        "people = set(a, b, c)\n"
        "left = choose(people)\n"
        "right = choose(people)\n"
        f"{relation}"
    )

    assert _solve(program) == expected


def test_encoding_exposes_and_removes_static_set_factor() -> None:
    problem = parse(
        "people = set(a, b, c, d)\n"
        "special = set(a, b)\n"
        "chosen = choose(people, 2)\n"
        "overlap = chosen & special\n"
        "|overlap| == 1"
    )
    schedule = IRPipeline().process(problem)
    component, analysis = schedule.branches[0].components[0]

    encoding = encode_ir(component, analysis)
    raw_count = c1f_fomc(encoding.sentence, encoding.domain_size)

    assert encoding.domain_size == 4
    assert encoding.normalization_factor == 6
    assert int(raw_count) == 4 * 6
    assert encoding.decode_result(raw_count) == 4


def test_named_membership_uses_only_one_singleton_marker() -> None:
    problem = parse(
        "people = set(a, b, c, d)\n"
        "chosen = choose(people)\n"
        "a in chosen"
    )
    schedule = IRPipeline().process(problem)
    component, analysis = schedule.branches[0].components[0]

    encoding = encode_ir(component, analysis)

    assert encoding.normalization_factor == 4
    assert "CofolaEntity_0" in encoding.sentence
    assert "CofolaEntity_1" not in encoding.sentence
    assert encoding.decode_result(
        c1f_fomc(encoding.sentence, encoding.domain_size)
    ) == 8


def test_unsupported_construct_has_actionable_error() -> None:
    program = (
        "items = bag(a: 2, b)\n"
        "chosen = choose(items)"
    )

    with pytest.raises(UnaryFOMCUnsupportedError, match="Bag"):
        _solve(program)


def test_cardinality_expansion_limit_is_enforced() -> None:
    problem = parse(
        "people = set(a, b, c)\n"
        "left = choose(people)\n"
        "right = choose(people)\n"
        "|left| - |right| > 0"
    )
    schedule = IRPipeline().process(problem)
    component, analysis = schedule.branches[0].components[0]

    with pytest.raises(UnaryFOMCUnsupportedError, match="16 cases"):
        encode_ir(component, analysis, max_cardinality_cases=15)
