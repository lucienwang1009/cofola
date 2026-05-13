"""Problem and ProblemBuilder for the Cofola frontend core problem model.

Problem is the immutable frontend representation of a combinatorics counting
problem. ProblemBuilder is the mutable builder used to construct a Problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, cast

from cofola.frontend.constraints import Constraint
from cofola.frontend.objects import ObjDef, ObjRef, PartDef, PartPlaceholderDef
from cofola.frontend.utils import map_refs, object_refs


# A source location is a (line, column) pair from the parser, or None when
# the object was not produced by the parser (e.g. constructed by a lowering pass).
SourceLoc = tuple[int, int]


@dataclass(frozen=True)
class Problem:
    """Immutable frontend core model for a combinatorics counting problem.

    A Problem contains:
    - defs: Mapping from ObjRef to object definitions
    - constraints: Tuple of constraints on the objects
    - names: Mapping from ObjRef to user-given identifiers (for debugging/encoding)

    All methods are pure (no mutation) and return new values.
    """

    defs: tuple[tuple[ObjRef, ObjDef], ...]  # tuple of pairs for determinism
    constraints: tuple[Constraint, ...]
    names: tuple[tuple[ObjRef, str], ...]
    # Source locations for ObjRefs introduced by the parser. Side-table so we
    # don't need to plumb (line, col) through every frontend dataclass. Defaults to
    # empty for objects/passes that don't track positions.
    locs: tuple[tuple[ObjRef, SourceLoc], ...] = field(default_factory=tuple)

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

    def get_effective_object(self, ref: ObjRef) -> ObjDef | None:
        """Return the semantic object definition for type dispatch.

        Concrete part refs and forall-part placeholders behave like their
        partition source for parser/type-dispatch purposes. Raw structural
        callers should use ``get_object`` instead.
        """

        defn = self.get_object(ref)
        if isinstance(defn, (PartDef, PartPlaceholderDef)):
            partition_defn = self.get_object(defn.partition)
            source = getattr(partition_defn, "source", None)
            if isinstance(source, ObjRef):
                return self.get_effective_object(source)
            return None
        return defn

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

    def get_loc(self, ref: ObjRef) -> SourceLoc | None:
        """Return the (line, col) source location for an object reference.

        Returns None if no source location was recorded (e.g. the object was
        constructed by a lowering pass, or the parser did not propagate positions).
        """
        for r, loc in self.locs:
            if r == ref:
                return loc
        return None

    def get_refs(self, defn: ObjDef) -> list[ObjRef]:
        """Extract all ObjRef fields from a frontend node.

        Args:
            defn: The object definition to extract references from.

        Returns:
            List of ObjRef fields found in the definition.
        """
        return object_refs(defn)

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

    def substitute(self, old_ref: ObjRef, new_ref: ObjRef) -> Problem:
        """Replace all uses of old_ref with new_ref.

        Args:
            old_ref: The reference to replace.
            new_ref: The reference to replace with.

        Returns:
            A new Problem with the substitution applied.
        """
        def sub(ref: ObjRef) -> ObjRef:
            return new_ref if ref == old_ref else ref

        # Substitute in definitions
        new_defs = []
        for ref, defn in self.defs:
            if ref == old_ref:
                continue
            new_defs.append((ref, cast(ObjDef, map_refs(defn, sub))))

        # If new_ref is a new definition, add it
        new_def_exists = any(ref == new_ref for ref, _ in new_defs)
        old_def = self.get_object(old_ref)
        if not new_def_exists and old_def is not None:
            new_defs.append((new_ref, old_def))

        # Substitute in constraints
        new_constraints = tuple(
            cast(Constraint, map_refs(c, sub)) for c in self.constraints
        )

        # Keep names
        new_names = tuple(
            (ref, name) for ref, name in self.names if ref != old_ref
        )

        new_locs = list((ref, loc) for ref, loc in self.locs if ref != old_ref)
        if not new_def_exists:
            old_loc = self.get_loc(old_ref)
            if old_loc is not None and not any(ref == new_ref for ref, _ in new_locs):
                new_locs.append((new_ref, old_loc))

        return Problem(
            defs=tuple(new_defs),
            constraints=new_constraints,
            names=new_names,
            locs=tuple(new_locs),
        )

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
        self._locs: dict[ObjRef, SourceLoc] = {}

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

    def set_loc(self, ref: ObjRef, line: int, col: int) -> None:
        """Record the parser source location (line, col) for an object reference.

        Idempotent: if the ref already has a location, the first call wins.
        Subsequent calls (e.g. from an outer rule re-visiting the same ref)
        are ignored so we keep the innermost / earliest position.
        """
        if ref not in self._locs:
            self._locs[ref] = (int(line), int(col))

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

    def get_effective_object(self, ref: ObjRef) -> ObjDef | None:
        """Return the semantic object definition for type dispatch."""

        defn = self.get_object(ref)
        if isinstance(defn, (PartDef, PartPlaceholderDef)):
            partition_defn = self.get_object(defn.partition)
            source = getattr(partition_defn, "source", None)
            if isinstance(source, ObjRef):
                return self.get_effective_object(source)
            return None
        return defn

    def iter_defs(self) -> list[tuple[ObjRef, ObjDef]]:
        """Return the (ref, defn) pairs added so far, in insertion order."""
        return list(self._defs)

    def build(self) -> Problem:
        """Build the immutable Problem from the accumulated definitions.

        Returns:
            The immutable Problem.
        """
        names_tuple = tuple((ref, name) for ref, name in self._names.items())
        locs_tuple = tuple((ref, loc) for ref, loc in self._locs.items())
        return Problem(
            defs=tuple(self._defs),
            constraints=tuple(self._constraints),
            names=names_tuple,
            locs=locs_tuple,
        )
