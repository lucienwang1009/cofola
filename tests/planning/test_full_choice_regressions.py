"""Full-source choices simplify only when their size is proven exact."""
from __future__ import annotations

from dataclasses import replace

import pytest

from cofola.backend.wfomc import WFOMC_GLOBAL_PASSES
from cofola.frontend import BagInit, ObjRef, SequenceDef, SetInit, TupleDef
from cofola.parser.parser import parse
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


def _globals(text: str):
    return PlaningPipeline.run_passes(parse(text), WFOMC_GLOBAL_PASSES).problem


@pytest.mark.parametrize(("source", "size", "kind"), [
    ("set(a,b)", 2, SetInit), ("bag(a:2,b:1)", 3, BagInit),
])
@pytest.mark.parametrize("explicit", [True, False])
def test_full_choices_are_aliases(source: str, size: int, kind: type, explicit: bool) -> None:
    choice = f"choose(S, {size})" if explicit else "choose(S)"
    constraints = "" if explicit else f"\n|C| == {size}"
    text = f"S = {source}\nC = {choice}{constraints}"
    result = _globals(text)
    assert len(result.defs) == 1
    assert isinstance(result.defs[0][1], kind)
    assert {name for _, name in result.names} == {"S", "C"}
    assert len({ref for ref, _ in result.names}) == 1
    assert parse_and_solve(text) == 1


@pytest.mark.parametrize("constructor", ["choose_tuple", "choose_sequence"])
@pytest.mark.parametrize("explicit", [True, False])
def test_full_ordered_choice_becomes_permutation(constructor: str, explicit: bool) -> None:
    args = "S, 2" if explicit else "S"
    text = f"S = set(a,b)\nT = {constructor}({args})"
    # A redundant size constraint must be safe even if the definition already
    # carries the size (including sizes embedded by FullChoiceOptimizer).
    text += "\n|T| == 2"
    result = _globals(text)
    ordered = [d for _, d in result.defs if isinstance(d, (TupleDef, SequenceDef))]
    assert len(ordered) == 1
    assert not ordered[0].choose
    assert parse_and_solve(text) == 2


@pytest.mark.parametrize(("constructor", "source", "total", "small"), [
    ("choose_tuple", "set(a,b)", 5, 3),
    ("choose_sequence", "set(a,b)", 5, 3),
    ("choose_tuple", "bag(a:2,b:1)", 9, 3),
    ("choose_sequence", "bag(a:2,b:1)", 9, 3),
])
def test_unsized_ordered_choices_keep_all_lengths(constructor: str, source: str, total: int, small: int) -> None:
    text = f"S = {source}\nT = {constructor}(S)"
    assert parse_and_solve(text) == total
    assert parse_and_solve(text + "\n|T| <= 1") == small
    assert any(isinstance(d, (TupleDef, SequenceDef)) and d.choose for _, d in _globals(text).defs)


@pytest.mark.parametrize(("expression", "count"), [
    ("choose(S)", 4), ("choose(S,1)", 2),
    ("choose_replace(S,2)", 3), ("choose_replace_tuple(S,2)", 4),
    ("choose_replace_sequence(S,2)", 4),
])
def test_partial_and_replacement_choices_are_unchanged(expression: str, count: int) -> None:
    assert parse_and_solve(f"S = set(a,b)\nC = {expression}") == count


def test_compound_size_constraints_are_not_assumed_unconditional() -> None:
    assert parse_and_solve("S = set(a,b)\nT = choose_tuple(S)\n(|T| == 1) or (|T| == 2)") == 4


def test_dynamic_source_full_choice_keeps_source_choices() -> None:
    text = "S = set(a,b,c)\nA = choose(S,2)\nC = choose(A,2)\na in C"
    result = _globals(text)
    assert len(result.defs) == 2
    assert parse_and_solve(text) == 2


@pytest.mark.parametrize("constructor", ["choose_tuple", "choose_sequence"])
def test_full_bag_ordered_choice_preserves_multiplicity(constructor: str) -> None:
    assert parse_and_solve(f"B = bag(a:2,b:1)\nT = {constructor}(B,3)\n|T| == 3") == 3


def test_unknown_source_size_does_not_imply_identity() -> None:
    text = "S = set(a,b)\nA = choose(S)\nC = choose(A,1)"
    assert len(_globals(text).defs) == 3
    assert parse_and_solve(text) == 4


def test_alias_chain_is_refolded_and_preserves_names() -> None:
    text = "B = bag(a:2,b:1)\nC = choose(B,3)\nD = choose(C,3)\nI = D & B\n|I| == 3"
    result = _globals(text)
    assert len(result.defs) == 1
    assert {name for _, name in result.names} == {"B", "C", "D", "I"}
    assert parse_and_solve(text) == 1


def test_substitute_preserves_all_aliases_and_target_canonical_name() -> None:
    problem = parse("S = set(a,b)\nC = choose(S,2)")
    source, choice = (ref for ref, _ in problem.defs)
    problem = replace(problem, names=((choice, "C"), (source, "S"), (choice, "alias")))
    result = problem.substitute(choice, source)
    assert result.names == ((source, "S"), (source, "C"), (source, "alias"))
    assert result.get_name(source) == "S"
    assert problem.substitute(source, source) is problem
    renamed = problem.substitute(choice, ObjRef(9))
    assert (ObjRef(9), "C") in renamed.names
    assert (ObjRef(9), "alias") in renamed.names
    assert renamed.get_loc(ObjRef(9)) == problem.get_loc(choice)


@pytest.mark.parametrize("alias_count", [0, 1, 1000])
def test_substitute_preserves_order_and_deduplicates_existing_aliases(alias_count: int) -> None:
    problem = parse("S = set(a,b)\nC = choose(S,2)\nU = set(c)")
    source, choice, unrelated = (ref for ref, _ in problem.defs)
    retained_names = ((unrelated, "C"), (source, "S")) + tuple(
        (source, f"alias_{i}") for i in range(alias_count)
    )
    choice_names = ((choice, "C"), (choice, "S")) + tuple(
        (choice, f"alias_{i}") for i in range(2 * alias_count)
    )
    original_names = choice_names + retained_names
    problem = replace(problem, names=original_names)

    result = problem.substitute(choice, source)

    # Existing target aliases are not re-added, while a matching name on an
    # unrelated ref must not suppress an alias. Retained names stay first.
    assert result.names == retained_names + ((source, "C"),) + tuple(
        (source, f"alias_{i}") for i in range(alias_count, 2 * alias_count)
    )
    assert result.get_name(source) == "S"
    assert result.get_name(unrelated) == "C"
    assert problem.names == original_names
