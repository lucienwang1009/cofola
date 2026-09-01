"""Shared helpers for navigating a planning ``Problem`` in tests."""
from __future__ import annotations

from cofola.frontend import ObjRef, Problem, SetPartDef, SizeConstraint


def _ref_named(problem: Problem, name: str) -> ObjRef:
    for ref, candidate in problem.names:
        if candidate == name:
            return ref
    raise AssertionError(f"missing ref named {name!r}")


def _part_ref(problem: Problem, partition: ObjRef, index: int) -> ObjRef:
    for ref, defn in problem.defs:
        if isinstance(defn, SetPartDef) and defn.partition == partition and defn.index == index:
            return ref
    raise AssertionError(f"missing part {partition.id}[{index}]")


def _first_def_ref(problem: Problem, cls: type) -> ObjRef:
    for ref, defn in problem.defs:
        if isinstance(defn, cls):
            return ref
    raise AssertionError(f"missing def of type {cls.__name__}")


def _size_constraint_for_ref(problem: Problem, target: ObjRef) -> SizeConstraint:
    for constraint in problem.constraints:
        if not isinstance(constraint, SizeConstraint):
            continue
        if any(term == target for term, _coef in constraint.terms):
            return constraint
    raise AssertionError(f"missing SizeConstraint for ref {target.id}")
