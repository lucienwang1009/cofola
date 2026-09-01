"""Tuple membership has the same frontend and lowering semantics in both APIs."""
from __future__ import annotations

from dataclasses import replace

import pytest

from cofola.frontend import Entity, FuncImage, FuncInverseImage, MembershipConstraint, SizeConstraint
from cofola.parser.parser import parse
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve, solve


@pytest.mark.parametrize(("source", "tuple_expr", "entity", "positive_count", "negative_count"), [
    ("set(a, b)", "choose_tuple(S, 1)", "a", 1, 1),
    ("set(a, b)", "tuple(S)", "a", 2, 0),
    ("set(a, b)", "choose_replace_tuple(S, 2)", "a", 3, 1),
    ("set(a, b)", "choose_tuple(S, 0)", "a", 0, 1),
    ("bag(a: 2, b: 1)", "choose_tuple(S, 2)", "a", 3, 0),
    ("bag(a: 2, b: 1)", "choose_tuple(S, 2)", "b", 2, 1),
    ("bag(a: 2, b: 1)", "tuple(S)", "b", 3, 0),
])
def test_text_and_python_membership_agree(
    source: str, tuple_expr: str, entity: str, positive_count: int, negative_count: int,
) -> None:
    text = f"S = {source}\nT = {tuple_expr}\n"
    problem = parse(text)
    tuple_ref = next(ref for ref, name in problem.names if name == "T")
    for positive, expected in ((True, positive_count), (False, negative_count)):
        constraint = MembershipConstraint(Entity(entity), tuple_ref, positive)
        assert solve(replace(problem, constraints=(constraint,))) == expected
        assert parse_and_solve(text + f"{entity} {'in' if positive else 'not in'} T") == expected


def test_set_membership_reuses_one_image() -> None:
    problem = parse("S = set(a, b)\nT = tuple(S)\na in T\nb not in T\na in T")
    result = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)]).problem
    images = [ref for ref, defn in result.defs if isinstance(defn, FuncImage)]
    assert len(images) == 1
    assert all(isinstance(c, MembershipConstraint) and c.container == images[0]
               for c in result.constraints)


def test_bag_membership_reuses_multiplicity_inverse_images() -> None:
    problem = parse("B = bag(a: 2, b: 1)\nT = choose_tuple(B, 2)\na in T\nb not in T\na in T")
    result = PlaningPipeline.run_passes(problem, [FixedPointPass(LoweringPass)]).problem
    inverse_images = [(ref, defn) for ref, defn in result.defs if isinstance(defn, FuncInverseImage)]
    assert len(inverse_images) == 2
    assert not any(isinstance(defn, FuncImage) for _, defn in result.defs)
    by_entity = {defn.argument: ref for ref, defn in inverse_images}
    assert SizeConstraint(((by_entity[Entity("a")], 1),), ">", 0) in result.constraints
    assert SizeConstraint(((by_entity[Entity("b")], 1),), "==", 0) in result.constraints
    assert parse_and_solve("B = bag(a: 2, b: 1)\nT = choose_tuple(B, 2)\na in T\nb not in T") == 1
