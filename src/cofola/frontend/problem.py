"""Problem and ProblemBuilder for the immutable IR.

Problem is the immutable IR representation of a combinatorics counting problem.
ProblemBuilder is the mutable builder used to construct a Problem.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Iterator

from cofola.frontend.types import ObjRef, Entity
from cofola.frontend.constraints import (
    AndConstraint,
    Constraint,
    ForAllParts,
    NotConstraint,
    OrConstraint,
)
from cofola.frontend.objects import ObjDef


@dataclass(frozen=True)
class Problem:
    """Immutable IR for a combinatorics counting problem.

    A Problem contains:
    - defs: Mapping from ObjRef to object definitions
    - constraints: Tuple of constraints on the objects
    - names: Mapping from ObjRef to user-given identifiers (for debugging/encoding)

    All methods are pure (no mutation) and return new values.
    """

    defs: tuple[tuple[ObjRef, ObjDef], ...]  # tuple of pairs for determinism
    constraints: tuple[Constraint, ...]
    names: tuple[tuple[ObjRef, str], ...]

    def get_object(self, ref: ObjRef) -> ObjDef | None:
        """Get the object definition for a reference.

        Args:
            ref: The object reference to look up.

        Returns:
            The object definition, or None if not found.
        """
        for r, defn in self.defs:
            if r == ref:
                return defn
        return None

    def iter_objects(self) -> Iterator[tuple[ObjRef, ObjDef]]:
        """Iterate over all (ref, defn) pairs in definition order."""
        yield from self.defs

    def refs(self) -> list[ObjRef]:
        """Get all object references in definition order."""
        return [ref for ref, _ in self.defs]

    def get_name(self, ref: ObjRef) -> str | None:
        """Get the user-given name for an object reference.

        Args:
            ref: The object reference to look up.

        Returns:
            The name string, or None if unnamed.
        """
        for r, name in self.names:
            if r == ref:
                return name
        return None

    def get_refs(self, defn: ObjDef) -> list[ObjRef]:
        """Extract all ObjRef fields from an IR node.

        Args:
            defn: The object definition to extract references from.

        Returns:
            List of ObjRef fields found in the definition.
        """
        result = []
        for f in fields(defn):
            val = getattr(defn, f.name)
            if isinstance(val, ObjRef):
                result.append(val)
            elif isinstance(val, frozenset):
                # Handle sets of entities (no ObjRef in there)
                pass
            elif isinstance(val, tuple):
                # Handle tuples that might contain (ObjRef, int) pairs
                for item in val:
                    if isinstance(item, ObjRef):
                        result.append(item)
                    elif isinstance(item, tuple) and len(item) == 2:
                        if isinstance(item[0], ObjRef):
                            result.append(item[0])
        return result

    def dep_graph(self) -> dict[ObjRef, list[ObjRef]]:
        """Build dependency graph: ref -> list of refs it depends on.

        Returns:
            Adjacency list mapping each ref to its dependencies.
        """
        return {ref: self.get_refs(defn) for ref, defn in self.defs}

    def topological_order(self) -> list[ObjRef]:
        """Compute topological order using Kahn's algorithm.

        Returns:
            List of refs in topological order (dependencies before dependents).
        """
        graph = self.dep_graph()
        in_degree: dict[ObjRef, int] = {ref: 0 for ref, _ in self.defs}

        # Compute in-degrees
        for ref, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[ref] += 1

        # Find all nodes with no incoming edges
        queue = [ref for ref, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for deterministic ordering
            queue.sort(key=lambda r: r.id)
            ref = queue.pop(0)
            result.append(ref)

            # For each node that depends on ref, decrease its in-degree
            for other_ref, deps in graph.items():
                if ref in deps:
                    in_degree[other_ref] -= 1
                    if in_degree[other_ref] == 0:
                        queue.append(other_ref)

        return result

    @staticmethod
    def _sub_field(val: object, old_ref: ObjRef, new_ref: ObjRef) -> object:
        """Substitute old_ref with new_ref in a single field value.

        Handles ObjRef directly and tuples that may contain ObjRefs or
        (ObjRef, int) pairs (e.g. SizeConstraint.terms).
        """
        if isinstance(val, ObjRef):
            return new_ref if val == old_ref else val
        elif isinstance(val, tuple):
            new_items = []
            for item in val:
                if isinstance(item, ObjRef):
                    new_items.append(new_ref if item == old_ref else item)
                elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ObjRef):
                    new_items.append((new_ref if item[0] == old_ref else item[0], item[1]))
                elif isinstance(item, tuple) and len(item) == 2 and is_dataclass(item[0]) and not isinstance(item[0], type):
                    # Recurse into size atoms (TupleCountAtom, BagCountAtom, SeqPatternCountAtom)
                    # that contain ObjRef fields
                    atom, coef = item
                    new_atom_fields = {
                        f.name: Problem._sub_field(getattr(atom, f.name), old_ref, new_ref)
                        for f in fields(atom)
                    }
                    new_items.append((type(atom)(**new_atom_fields), coef))
                else:
                    new_items.append(item)
            return tuple(new_items)
        return val

    def substitute(self, old_ref: ObjRef, new_ref: ObjRef) -> Problem:
        """Replace all uses of old_ref with new_ref.

        Args:
            old_ref: The reference to replace.
            new_ref: The reference to replace with.

        Returns:
            A new Problem with the substitution applied.
        """
        sub = self._sub_field

        # Substitute in definitions
        new_defs = []
        for ref, defn in self.defs:
            if ref == old_ref:
                continue
            new_fields = {f.name: sub(getattr(defn, f.name), old_ref, new_ref) for f in fields(defn)}
            new_defs.append((ref, type(defn)(**new_fields)))

        # If new_ref is a new definition, add it
        new_def_exists = any(ref == new_ref for ref, _ in new_defs)
        old_def = self.get_object(old_ref)
        if not new_def_exists and old_def is not None:
            new_defs.append((new_ref, old_def))

        # Substitute in constraints
        new_constraints = tuple(
            self._sub_constraint(c, old_ref, new_ref) for c in self.constraints
        )

        # Keep names
        new_names = tuple(
            (ref, name) for ref, name in self.names if ref != old_ref
        )

        return Problem(defs=tuple(new_defs), constraints=new_constraints, names=new_names)

    def _sub_constraint(self, c: Constraint, old_ref: ObjRef, new_ref: ObjRef) -> Constraint:
        """Substitute old_ref with new_ref in a constraint."""
        if isinstance(c, NotConstraint):
            return NotConstraint(self._sub_constraint(c.sub, old_ref, new_ref))
        elif isinstance(c, AndConstraint):
            return AndConstraint(
                self._sub_constraint(c.left, old_ref, new_ref),
                self._sub_constraint(c.right, old_ref, new_ref),
            )
        elif isinstance(c, OrConstraint):
            return OrConstraint(
                self._sub_constraint(c.left, old_ref, new_ref),
                self._sub_constraint(c.right, old_ref, new_ref),
            )
        elif isinstance(c, ForAllParts):
            return ForAllParts(
                new_ref if c.partition == old_ref else c.partition,
                self._sub_constraint(c.constraint_template, old_ref, new_ref),
            )
        else:
            sub = self._sub_field
            c_fields = {f.name: sub(getattr(c, f.name), old_ref, new_ref) for f in fields(c)}
            return type(c)(**c_fields)

    def __repr__(self) -> str:
        """Return an unambiguous string representation."""
        return f"Problem(defs={self.defs!r}, constraints={self.constraints!r}, names={self.names!r})"

    def __str__(self) -> str:
        """Return a readable string representation of the problem."""
        lines = ["Problem("]

        # Definitions
        lines.append("  defs=[")
        for ref, defn in self.defs:
            name = self.get_name(ref)
            if name:
                lines.append(f"    {name}: {defn!r},")
            else:
                lines.append(f"    {ref}: {defn!r},")
        lines.append("  ],")

        # Constraints
        lines.append("  constraints=[")
        for c in self.constraints:
            lines.append(f"    {c},")
        lines.append("  ],")

        lines.append(")")
        return "\n".join(lines)


class ProblemBuilder:
    """Mutable builder for constructing a Problem.

    Used by the parser to incrementally build a Problem from parsed input.
    After construction, call build() to get the immutable Problem.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self._defs: list[tuple[ObjRef, ObjDef]] = []
        self._constraints: list[Constraint] = []
        self._names: dict[ObjRef, str] = {}

    def add(self, defn: ObjDef, name: str | None = None) -> ObjRef:
        """Add an object definition and return its reference.

        Args:
            defn: The object definition to add.
            name: Optional user-given name for the object.

        Returns:
            The ObjRef for the added definition.
        """
        ref = ObjRef(self._next_id)
        self._next_id += 1
        self._defs.append((ref, defn))
        if name is not None:
            self._names[ref] = name
        return ref

    def add_constraint(self, c: Constraint) -> None:
        """Add a constraint to the problem.

        Args:
            c: The constraint to add.
        """
        self._constraints.append(c)

    def set_name(self, ref: ObjRef, name: str) -> None:
        """Set or update the name for an existing object reference."""
        self._names[ref] = name

    def find_equivalent(self, defn: ObjDef) -> ObjRef | None:
        """Find existing ref with an equivalent definition (deduplication).

        Args:
            defn: The definition to search for.

        Returns:
            The ObjRef of an equivalent definition, or None if not found.
        """
        for ref, existing in self._defs:
            if existing == defn:  # frozen dataclass __eq__ does structural comparison
                return ref
        return None

    def get_object(self, ref: ObjRef) -> ObjDef | None:
        """Get the object definition for a reference.

        Args:
            ref: The object reference to look up.

        Returns:
            The object definition, or None if not found.
        """
        for r, defn in self._defs:
            if r == ref:
                return defn
        return None

    def build(self) -> Problem:
        """Build the immutable Problem from the accumulated definitions.

        Returns:
            The immutable Problem.
        """
        names_tuple = tuple((ref, name) for ref, name in self._names.items())
        return Problem(
            defs=tuple(self._defs),
            constraints=tuple(self._constraints),
            names=names_tuple,
        )
