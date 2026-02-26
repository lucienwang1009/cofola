"""Simplification pass for the immutable IR.

This module implements SimplifyPass, which removes unused objects
from the IR.
"""

from __future__ import annotations

from cofola.frontend.types import ObjRef
from cofola.frontend.constraints import (
    SizeConstraint,
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    SequencePatternConstraint,
    FuncPairConstraint,
    ForAllParts,
    NotConstraint,
    AndConstraint,
    OrConstraint,
    TogetherPattern,
    LessThanPattern,
    PredecessorPattern,
    NextToPattern,
)
from cofola.frontend.problem import Problem
from loguru import logger


class SimplifyPass:
    """Removes unused objects from the IR.

    This pass identifies objects that are not referenced by any constraint
    or other used object, and removes them.
    """

    def run(self, problem: Problem) -> Problem:
        """Simplify a Problem by removing unused objects.

        Args:
            problem: The Problem to simplify.

        Returns:
            A new Problem with unused objects removed.
        """
        logger.info("SimplifyPass: {} objects before", len(list(problem.iter_objects())))

        # Find all objects referenced by constraints
        used_refs = self._find_used_refs(problem)

        # Propagate usage through dependency graph
        used_refs = self._propagate_usage(problem, used_refs)

        all_refs = {ref for ref, _ in problem.iter_objects()}
        removed = all_refs - used_refs
        if removed:
            logger.info("SimplifyPass: removed {} unused objects: {}",
                        len(removed), {r.id for r in removed})
        else:
            logger.debug("SimplifyPass: no objects removed")

        logger.debug("SimplifyPass: {} used refs out of {} total",
                     len(used_refs), len(all_refs))

        # Filter definitions to only used ones
        new_defs = [
            (ref, defn) for ref, defn in problem.iter_objects()
            if ref in used_refs
        ]

        logger.info("SimplifyPass: {} objects after", len(new_defs))

        return Problem(
            defs=tuple(new_defs),
            constraints=problem.constraints,
            names=problem.names,
        )

    def _find_used_refs(self, problem: Problem) -> set:
        """Find all ObjRefs that are initially "used".

        An ObjRef is used if it is referenced by a constraint, OR if it is
        a leaf node (no other object depends on it).  Leaf nodes are always
        kept because the answer to the counting problem is always the last /
        leaf object in the dependency graph.

        Args:
            problem: The Problem to analyze.

        Returns:
            Set of ObjRefs used by constraints or leaf nodes.
        """
        used = set()

        for c in problem.constraints:
            used.update(self._get_constraint_refs(c))

        # Always keep leaf objects: objects that no other object depends on.
        dep_graph = problem.dep_graph()
        all_deps: set = set()
        for deps in dep_graph.values():
            all_deps.update(deps)
        all_refs = {ref for ref, _ in problem.iter_objects()}
        leaf_refs = all_refs - all_deps
        used.update(leaf_refs)

        return used

    def _get_constraint_refs(self, c) -> set:
        """Extract ObjRefs from a constraint."""
        refs = set()

        if isinstance(c, SizeConstraint):
            for ref, _ in c.terms:
                refs.add(ref)

        elif isinstance(c, MembershipConstraint):
            refs.add(c.container)

        elif isinstance(c, SubsetConstraint):
            refs.add(c.sub)
            refs.add(c.sup)

        elif isinstance(c, DisjointConstraint):
            refs.add(c.left)
            refs.add(c.right)

        elif isinstance(c, EqualityConstraint):
            refs.add(c.left)
            refs.add(c.right)

        elif isinstance(c, TupleIndexEq):
            refs.add(c.tuple_ref)

        elif isinstance(c, TupleIndexMembership):
            refs.add(c.tuple_ref)
            refs.add(c.container)

        elif isinstance(c, SequencePatternConstraint):
            refs.add(c.seq)
            refs.update(self._get_pattern_refs(c.pattern))

        elif isinstance(c, FuncPairConstraint):
            refs.add(c.func)
            if isinstance(c.result, ObjRef):
                refs.add(c.result)

        elif isinstance(c, ForAllParts):
            refs.add(c.partition)
            refs.update(self._get_constraint_refs(c.constraint_template))

        elif isinstance(c, NotConstraint):
            refs.update(self._get_constraint_refs(c.sub))

        elif isinstance(c, (AndConstraint, OrConstraint)):
            refs.update(self._get_constraint_refs(c.left))
            refs.update(self._get_constraint_refs(c.right))

        return refs

    def _get_pattern_refs(self, pattern) -> set:
        """Extract ObjRefs from a sequence pattern."""
        refs = set()

        if isinstance(pattern, TogetherPattern):
            refs.add(pattern.group)

        elif isinstance(pattern, LessThanPattern):
            if isinstance(pattern.left, ObjRef):
                refs.add(pattern.left)
            if isinstance(pattern.right, ObjRef):
                refs.add(pattern.right)

        elif isinstance(pattern, (PredecessorPattern, NextToPattern)):
            if isinstance(pattern.first, ObjRef):
                refs.add(pattern.first)
            if isinstance(pattern.second, ObjRef):
                refs.add(pattern.second)

        return refs

    def _propagate_usage(self, problem: Problem, used: set) -> set:
        """Propagate usage through the dependency graph.

        An object is used if it's in the used set, or if an object
        that depends on it is used.

        Args:
            problem: The Problem to analyze.
            used: Initially used refs (from constraints).

        Returns:
            Complete set of used refs.
        """
        # Build reverse dependency graph (ref -> refs that depend on it)
        dep_graph = problem.dep_graph()
        reverse_deps: dict = {}

        for ref, deps in dep_graph.items():
            for dep in deps:
                if dep not in reverse_deps:
                    reverse_deps[dep] = set()
                reverse_deps[dep].add(ref)

        # BFS from used refs
        result = set(used)
        queue = list(used)

        while queue:
            current = queue.pop(0)

            # Find refs that current depends on
            deps = dep_graph.get(current, [])

            for dep in deps:
                if dep not in result:
                    result.add(dep)
                    queue.append(dep)

        return result