"""Base rewriter for IR-to-IR transformations.

This module provides the base Rewriter class for implementing transformation
passes that convert one Problem into another.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

from cofola.frontend.types import ObjRef
from cofola.frontend.objects import ObjDef
from cofola.frontend.constraints import Constraint
from cofola.frontend.problem import Problem


class Rewriter:
    """Base class for IR-to-IR transformation passes.

    A Rewriter transforms a Problem into a new Problem by visiting each
    object definition and optionally replacing it. The base class handles
    the mechanics of rebuilding the Problem.

    Subclasses override visit() to implement specific transformations.
    """

    def rewrite(self, problem: Problem) -> Problem:
        """Rewrite a Problem by visiting each object.

        Args:
            problem: The Problem to rewrite.

        Returns:
            A new Problem with transformations applied.
        """
        new_defs: list[tuple[ObjRef, ObjDef]] = []
        new_constraints: list[Constraint] = list(problem.constraints)
        ref_map: dict[ObjRef, ObjRef] = {}  # old ref -> new ref

        # Visit each object in topological order
        for ref in problem.topological_order():
            defn = problem.get_object(ref)
            if defn is None:
                continue

            result = self.visit(ref, defn, problem)

            if result is not None:
                # Replace with new definition
                new_defs.append((ref, result))
            else:
                # Keep original
                new_defs.append((ref, defn))

        return Problem(
            defs=tuple(new_defs),
            constraints=tuple(new_constraints),
            names=problem.names,
        )

    def visit(
        self, ref: ObjRef, defn: ObjDef, problem: Problem
    ) -> ObjDef | None:
        """Visit an object definition.

        Override in subclass to implement transformations.

        Args:
            ref: The reference to this object.
            defn: The object definition.
            problem: The full Problem context.

        Returns:
            A new ObjDef to replace the current one, or None to keep it.
        """
        return None


class RewriterWithSubstitution(Rewriter):
    """A Rewriter that supports reference substitution.

    This is useful for passes that create new objects or replace existing ones,
    requiring all references to be updated.
    """

    def __init__(self) -> None:
        self._next_id: int = 10000  # Start high to avoid collision
        self._ref_map: dict[ObjRef, ObjRef] = {}  # old -> new

    def _new_ref(self) -> ObjRef:
        """Create a new unique reference."""
        ref = ObjRef(self._next_id)
        self._next_id += 1
        return ref

    def _substitute_refs(
        self, defn: ObjDef, ref_map: dict[ObjRef, ObjRef]
    ) -> ObjDef:
        """Substitute references in an object definition.

        Args:
            defn: The definition to update.
            ref_map: Mapping from old refs to new refs.

        Returns:
            A new ObjDef with substituted references.
        """
        new_fields = {}
        for f in fields(defn):
            val = getattr(defn, f.name)
            new_val = self._substitute_in_value(val, ref_map)
            new_fields[f.name] = new_val

        return type(defn)(**new_fields)

    def _substitute_in_value(
        self, val: object, ref_map: dict[ObjRef, ObjRef]
    ) -> object:
        """Substitute refs in a field value."""
        if isinstance(val, ObjRef):
            return ref_map.get(val, val)
        elif isinstance(val, tuple):
            return tuple(self._substitute_in_value(item, ref_map) for item in val)
        elif isinstance(val, frozenset):
            return frozenset(self._substitute_in_value(item, ref_map) for item in val)
        else:
            return val