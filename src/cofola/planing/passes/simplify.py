"""Simplification pass for the immutable IR.

This module implements SimplifyPass, which removes unused objects
from the IR.
"""

from __future__ import annotations

from cofola.planing.pass_manager import TransformPass
from cofola.frontend.problem import Problem
from cofola.frontend.utils import constraint_refs
from loguru import logger


class SimplifyPass(TransformPass):
    """Removes unused objects from the IR.

    This pass identifies objects that are not referenced by any constraint
    or other used object, and removes them.
    """

    required_analyses: list[type] = []

    def run(self, problem: Problem, am=None) -> Problem:
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
            used.update(constraint_refs(c))

        # Always keep leaf objects: objects that no other object depends on.
        dep_graph = problem.dep_graph()
        all_deps: set = set()
        for deps in dep_graph.values():
            all_deps.update(deps)
        all_refs = {ref for ref, _ in problem.iter_objects()}
        leaf_refs = all_refs - all_deps
        used.update(leaf_refs)

        return used

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
        dep_graph = problem.dep_graph()

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
