"""Pass for merging identical SetInit and BagInit objects.

This module provides MergeIdenticalObjects, which finds and merges
objects that are semantically identical (same type and same content).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

from cofola.frontend.objects import (
    BagInit,
    ObjDef,
    ObjRef,
    SetInit,
)
from cofola.frontend.problem import Problem


@dataclass
class _SetInitKey:
    """Hash key for SetInit objects."""
    entities: frozenset  # frozenset[Entity]

    def __hash__(self) -> int:
        return hash(("SetInit", self.entities))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SetInitKey) and self.entities == other.entities


@dataclass
class _BagInitKey:
    """Hash key for BagInit objects."""
    entity_multiplicity: tuple  # tuple[tuple[Entity, int], ...]

    def __hash__(self) -> int:
        return hash(("BagInit", self.entity_multiplicity))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _BagInitKey) and self.entity_multiplicity == other.entity_multiplicity


class MergeIdenticalObjects:
    """Merge semantically identical SetInit and BagInit objects.

    This pass finds objects that are semantically identical:
    - Two SetInit objects with the same entities
    - Two BagInit objects with the same entity_multiplicity

    For each group of identical objects, it keeps one (the one with
    the smallest ObjRef.id for determinism) and substitutes all
    references to the others.

    This is a form of common subexpression elimination that can
    reduce the problem size and improve solver efficiency.
    """

    def run(self, problem: Problem) -> Problem:
        """Merge identical objects in a Problem.

        Args:
            problem: The Problem to optimize.

        Returns:
            A new Problem with identical objects merged.
        """
        # Step 1: Find all SetInit and BagInit objects and group by content
        set_groups: dict[_SetInitKey, list[ObjRef]] = defaultdict(list)
        bag_groups: dict[_BagInitKey, list[ObjRef]] = defaultdict(list)

        for ref, defn in problem.iter_objects():
            if isinstance(defn, SetInit):
                key = _SetInitKey(entities=defn.entities)
                set_groups[key].append(ref)
            elif isinstance(defn, BagInit):
                key = _BagInitKey(entity_multiplicity=defn.entity_multiplicity)
                bag_groups[key].append(ref)

        # Step 2: Find duplicates and create substitution map
        # Map: old_ref -> new_ref (canonical ref to keep)
        substitutions: dict[ObjRef, ObjRef] = {}

        for key, refs in set_groups.items():
            if len(refs) > 1:
                # Sort by ObjRef.id for determinism, keep the smallest
                refs_sorted = sorted(refs, key=lambda r: r.id)
                canonical = refs_sorted[0]
                for ref in refs_sorted[1:]:
                    substitutions[ref] = canonical
                    logger.info(
                        "MergeIdenticalObjects: Merging SetInit %s -> %s (entities: %s)",
                        ref, canonical, key.entities
                    )

        for key, refs in bag_groups.items():
            if len(refs) > 1:
                refs_sorted = sorted(refs, key=lambda r: r.id)
                canonical = refs_sorted[0]
                for ref in refs_sorted[1:]:
                    substitutions[ref] = canonical
                    logger.info(
                        "MergeIdenticalObjects: Merging BagInit %s -> %s",
                        ref, canonical
                    )

        if not substitutions:
            logger.debug("MergeIdenticalObjects: No identical objects found")
            return problem

        logger.info(
            "MergeIdenticalObjects: Merging %d duplicate objects",
            len(substitutions)
        )

        # Step 3: Apply substitutions iteratively
        # We need to do this one at a time since Problem.substitute returns a new Problem
        current = problem
        for old_ref, new_ref in substitutions.items():
            current = current.substitute(old_ref, new_ref)

        return current