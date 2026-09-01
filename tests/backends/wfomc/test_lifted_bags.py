"""Differential tests for the experimental lifted bag 1-type encoding."""
from __future__ import annotations

import sys

from sympy import Poly

from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.backend.wfomc.encoder import encode
from cofola.backend.wfomc import lifted_bags
from cofola.parser.parser import parse
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve, parse_args


def _solve(source: str, *, lifted_bags: bool) -> int:
    return parse_and_solve(source, lifted_bags=lifted_bags)


def _encode(source: str, *, lifted_bags: bool):
    schedule = PlaningPipeline().process(parse(source))
    assert len(schedule.branches) == 1
    assert len(schedule.branches[0].components) == 1
    problem, analysis = schedule.branches[0].components[0]
    return encode(problem, analysis, lifted_bags=lifted_bags)


def test_lifted_bags_is_disabled_by_default() -> None:
    backend = WFOMCBackend()

    assert isinstance(backend, WFOMCBackend)
    assert backend.lifted_bags is False


def test_lifted_bags_can_be_enabled_through_backend_option() -> None:
    backend = WFOMCBackend(lifted_bags=True)

    assert isinstance(backend, WFOMCBackend)
    assert backend.lifted_bags is True


def test_cli_exposes_lifted_bags_as_an_opt_in_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cofola", "-i", "problem.cfl", "--lifted-bags"],
    )

    assert parse_args().lifted_bags is True


def test_cli_leaves_lifted_bags_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cofola", "-i", "problem.cfl"])

    assert parse_args().lifted_bags is False


def test_lifted_choose_merges_equal_multiplicity_symbols() -> None:
    source = "B = bag(a: 2, b: 2)\nC = choose(B, 2)\n"

    assert _solve(source, lifted_bags=False) == 3
    assert _solve(source, lifted_bags=True) == 3

    _, unlifted_decoder = _encode(source, lifted_bags=False)
    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert len(unlifted_decoder.gens) == 2
    assert [str(var) for var in lifted_decoder.gens] == ["v_C#2"]


def test_lifted_one_types_and_weights_are_compiled_by_ganak(monkeypatch) -> None:
    calls: list[tuple[int, tuple[frozenset[int], ...], dict[int, tuple]]] = []
    results: list[object] = []
    original = lifted_bags.symbolic_ganak_count

    def recording_count(n_vars, clauses, weights):
        materialized_clauses = tuple(clauses)
        calls.append((n_vars, materialized_clauses, dict(weights)))
        result = original(n_vars, materialized_clauses, weights)
        results.append(result)
        return result

    monkeypatch.setattr(lifted_bags, "symbolic_ganak_count", recording_count)

    assert _solve("B = bag(a: 2, b: 2)\nC = choose(B, 2)\n", lifted_bags=True) == 3
    assert len(calls) == 1
    n_vars, clauses, weights = calls[0]
    symbolic_weights = {
        str(weight[0])
        for weight in weights.values()
        if getattr(weight[0], "free_symbols", frozenset())
    }
    assert symbolic_weights == {
        "v_C#2",
        "v_C#2**2",
    }
    assert n_vars == len(weights)
    assert clauses  # range and choose constraints are encoded as a CNF circuit
    assert Poly(results[0]).all_coeffs() == [1, 1, 1]


def test_lifted_choose_chain_preserves_per_entity_multiplicity_order() -> None:
    source = """
B = bag(a: 3, b: 3, c: 3)
X = choose(B, 4)
Y = choose(X, 2)
"""

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False)


def test_unobserved_intermediate_bag_does_not_get_a_symbol() -> None:
    source = """
B = bag(a: 2, b: 2, c: 2)
X = choose(B)
Y = choose(X, 2)
"""

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False)
    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert [str(var) for var in lifted_decoder.gens] == ["v_Y#2"]


def test_lifted_intersection_factor_matches_unlifted_encoding() -> None:
    source = """
B = bag(a: 2, b: 2, c: 2, d: 2)
X = choose(B)
Y = choose(B)
Z = X & Y
|X| == 3
|Y| == 3
|Z| == 2
"""

    assert _solve(source, lifted_bags=False) == 96
    assert _solve(source, lifted_bags=True) == 96

    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert {str(var) for var in lifted_decoder.gens} == {
        "v_X#2",
        "v_Y#2",
        "v_Z#2",
    }


def test_membership_observer_falls_back_to_per_entity_encoding() -> None:
    source = """
B = bag(a: 2, b: 2)
X = choose(B, 2)
a in X
"""

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False) == 2
    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert {str(var) for var in lifted_decoder.gens} == {"v_X_a", "v_X_b"}


def test_named_count_splits_only_the_observed_entity() -> None:
    source = """
B = bag(a: 2, b: 2, c: 2)
X = choose(B)
X.count(a) == 1
|X| == 3
"""

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False)
    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert {str(var) for var in lifted_decoder.gens} == {"v_X_a", "v_X#2"}


def test_unsupported_bag_operation_falls_back_to_per_entity_encoding() -> None:
    source = """
B = bag(a: 2, b: 2)
X = choose(B)
Y = choose(B)
Z = X + Y
|Z| == 2
"""

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False)


def test_factor_budget_exhaustion_falls_back_before_encoding(monkeypatch) -> None:
    source = "B = bag(a: 2, b: 2)\nX = choose(B, 2)\n"
    monkeypatch.setattr(lifted_bags, "MAX_FACTOR_STATES", 1)

    assert _solve(source, lifted_bags=True) == _solve(source, lifted_bags=False) == 3
    _, lifted_decoder = _encode(source, lifted_bags=True)
    assert {str(var) for var in lifted_decoder.gens} == {"v_X_a", "v_X_b"}
