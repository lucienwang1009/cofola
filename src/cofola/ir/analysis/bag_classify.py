"""Bag classification for the immutable IR.

This module implements BagClassification, which classifies bag entities
as distinguishable or indistinguishable for lifted WFOMC encoding.

Ports the legacy preprocess_bags function to work with the new IR.
"""

from __future__ import annotations

from dataclasses import replace

from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import (
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    ObjDef,
    PartitionDef,
    PartRef,
    SetChooseReplace,
    TupleDef,
)
from cofola.ir.pass_manager import AnalysisPass
from cofola.frontend.problem import Problem
from cofola.frontend.constraints import SizeConstraint, BagCountAtom
from loguru import logger

from cofola.ir.analysis.entities import AnalysisResult, BagInfo
from cofola.ir.analysis.merged import MergedAnalysis


class BagClassification(AnalysisPass):
    """Classifies bag entities as distinguishable or indistinguishable.

    This analysis determines which entities can be "lifted" (treated as
    indistinguishable) in the WFOMC encoding, which can significantly
    improve performance.

    Ports the legacy preprocess_bags function to work with the new IR.
    """

    required_analyses = [MergedAnalysis]

    def run(self, problem: Problem, am=None) -> AnalysisResult:
        """Run bag classification on a Problem.

        Args:
            problem: The Problem to analyze.
            am: AnalysisManager for accessing MergedAnalysis result.

        Returns:
            New AnalysisResult (deep-copied) with dis_entities and indis_entities filled.
        """
        from copy import deepcopy

        analysis = deepcopy(am.get(MergedAnalysis))

        logger.info("BagClassification.run: {} refs", len(list(problem.refs())))

        # Step 1: Find non-liftable bags
        # Bags derived from BagAdditiveUnion, BagDifference, TupleDef, and PartitionDef are non-liftable
        non_lifted_refs: set[ObjRef] = set()
        logger.debug("[Step 1] identifying non-liftable refs")

        for ref in problem.refs():
            defn = problem.get_object(ref)
            if defn is None:
                continue

            # These object types are inherently non-liftable
            if isinstance(defn, (BagAdditiveUnion, BagDifference)):
                non_lifted_refs.add(ref)
                self._mark_dependencies_non_lifted(ref, problem, non_lifted_refs)

            # TupleDef is non-liftable: the source bag's entities must be distinguishable
            elif isinstance(defn, TupleDef):
                # Mark the source bag as non-liftable
                source_ref = defn.source
                if source_ref not in non_lifted_refs:
                    non_lifted_refs.add(source_ref)
                    self._mark_dependencies_non_lifted(source_ref, problem, non_lifted_refs)

            # PartitionDef (compose) is also non-liftable, and its source bag must be dis
            elif isinstance(defn, PartitionDef):
                non_lifted_refs.add(ref)
                self._mark_dependencies_non_lifted(ref, problem, non_lifted_refs)

            # PartRef inherits the non-liftability of its partition
            elif isinstance(defn, PartRef):
                non_lifted_refs.add(ref)
                self._mark_dependencies_non_lifted(ref, problem, non_lifted_refs)

        logger.debug("  non_lifted_refs={}", {r.id for r in non_lifted_refs})

        # Step 2: Mark entities in non-liftable bags as distinguishable
        logger.debug("[Step 2] marking entities in non-liftable bags as dis")
        for ref in non_lifted_refs:
            if ref in analysis.bag_info:
                info = analysis.bag_info[ref]
                info.dis_entities = set(info.p_entities_multiplicity.keys())
                info.indis_entities = {}
                logger.debug("  bag ref={} -> all entities marked dis", ref.id)

        # Step 3: Mark entities referenced in BagCountAtom constraints as distinguishable
        # (entities used in multiplicity constraints cannot be lifted)
        logger.debug("[Step 3] marking BagCountAtom entities as dis")
        for constraint in problem.constraints:
            if isinstance(constraint, SizeConstraint):
                for term, _ in constraint.terms:
                    if isinstance(term, BagCountAtom):
                        bag_ref = term.bag
                        entity = term.entity
                        logger.debug("  BagCountAtom bag={} entity={}", bag_ref.id, entity.name)
                        # Mark entity as distinguishable in this bag and all its dependencies
                        self._mark_entity_dis_in_deps(bag_ref, entity, problem, analysis)

        # Step 4: For BagInit/SetChooseReplace, classify remaining entities by multiplicity
        logger.debug("[Step 4] classifying BagInit/SetChooseReplace entities by multiplicity")
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if defn is None:
                continue

            if isinstance(defn, BagInit) and ref in analysis.bag_info:
                info = analysis.bag_info[ref]
                self._classify_by_multiplicity(info)

            elif isinstance(defn, SetChooseReplace) and ref in analysis.bag_info:
                # SetChooseReplace: all entities must be dis_entities because
                # the encoder uses individual polynomial weights per entity
                info = analysis.bag_info[ref]
                if ref not in non_lifted_refs:
                    info.dis_entities = set(info.p_entities_multiplicity.keys())
                    info.indis_entities = {}

        # Step 5: Propagate dis_entities in topological order
        # For each derived bag, inherit dis_entities from its sources
        logger.debug("[Step 5] propagating dis_entities in topological order")
        for ref in problem.topological_order():
            defn = problem.get_object(ref)
            if defn is None or ref not in analysis.bag_info:
                continue

            # Skip BagInit — already classified in step 4
            if isinstance(defn, BagInit):
                continue

            self._propagate_from_sources(ref, defn, problem, analysis)

        # Final summary
        for ref, info in analysis.bag_info.items():
            logger.debug("  BagInfo ref={}: dis={}, indis={}",
                         ref.id,
                         {e.name for e in info.dis_entities},
                         {e.name: m for m, es in info.indis_entities.items() for e in es})

        return analysis

    def _mark_entity_dis_in_deps(
        self,
        ref: ObjRef,
        entity: Entity,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> None:
        """Mark a specific entity as distinguishable in ref and all its bag dependencies."""
        dep_graph = problem.dep_graph()
        visited: set[ObjRef] = set()
        queue = [ref]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in analysis.bag_info:
                info = analysis.bag_info[current]
                if entity in info.p_entities_multiplicity:
                    # Only mark if entity exists in this bag; e.g. the right-hand
                    # side of a BagDifference may not contain every entity from
                    # the result, so we must not add it to dis_entities there.
                    info.dis_entities.add(entity)
                    # Remove from indis_entities if present
                    for mult, entities in list(info.indis_entities.items()):
                        entities.discard(entity)
                        if not entities:
                            del info.indis_entities[mult]

            for dep in dep_graph.get(current, []):
                if dep not in visited:
                    queue.append(dep)

    def _mark_dependencies_non_lifted(
        self,
        ref: ObjRef,
        problem: Problem,
        non_lifted_refs: set[ObjRef],
    ) -> None:
        """Mark all dependencies of a non-liftable bag as non-liftable."""
        # Get dependencies from the problem's dependency graph
        dep_graph = problem.dep_graph()

        # BFS to find all dependencies
        visited = {ref}
        queue = [ref]

        while queue:
            current = queue.pop(0)
            deps = dep_graph.get(current, [])

            for dep in deps:
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
                    non_lifted_refs.add(dep)

    def _classify_by_multiplicity(self, info: BagInfo) -> None:
        """Classify entities in a bag by their multiplicity.

        Entities with the same multiplicity > 1 can be indistinguishable.
        Entities already marked as distinguishable stay that way.
        """
        # Group entities by multiplicity
        mult_to_entities: dict[int, set[Entity]] = {}

        for entity, mult in info.p_entities_multiplicity.items():
            if entity in info.dis_entities:
                continue

            if mult not in mult_to_entities:
                mult_to_entities[mult] = set()
            mult_to_entities[mult].add(entity)

        # Entities with multiplicity > 1 and more than one entity can be indistinguishable
        for mult, entities in mult_to_entities.items():
            if mult > 1 and len(entities) > 1:
                info.indis_entities[mult] = entities
            else:
                info.dis_entities.update(entities)

    def _find_root_bags(
        self,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> list[ObjRef]:
        """Find root bags (top-level BagChoose with indis_entities)."""
        roots = []

        for ref in problem.refs():
            defn = problem.get_object(ref)
            if defn is None:
                continue

            # A root bag is a BagChoose whose source is not a BagChoose
            if isinstance(defn, BagChoose):
                src_defn = problem.get_object(defn.source)
                if not isinstance(src_defn, BagChoose):
                    if ref in analysis.bag_info:
                        info = analysis.bag_info[ref]
                        if info.indis_entities:
                            roots.append(ref)

        return roots

    def _propagate_from_sources(
        self,
        ref: ObjRef,
        defn,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> None:
        """Propagate dis_entities from source bags to a derived bag."""
        info = analysis.bag_info[ref]

        if isinstance(defn, BagChoose):
            src_info = analysis.bag_info.get(defn.source)
            if src_info is not None:
                info.dis_entities = src_info.dis_entities.copy()
                info.indis_entities = {k: v.copy() for k, v in src_info.indis_entities.items()}

        elif isinstance(defn, BagUnion):
            left_info = analysis.bag_info.get(defn.left)
            right_info = analysis.bag_info.get(defn.right)
            if left_info is not None and right_info is not None:
                info.dis_entities = left_info.dis_entities | right_info.dis_entities
                info.indis_entities = {}

        elif isinstance(defn, BagAdditiveUnion):
            # These are non-liftable, already handled by step 2
            left_info = analysis.bag_info.get(defn.left)
            right_info = analysis.bag_info.get(defn.right)
            if left_info is not None and right_info is not None:
                info.dis_entities = left_info.dis_entities | right_info.dis_entities
                info.indis_entities = {}

        elif isinstance(defn, (BagIntersection, BagDifference)):
            # These are non-liftable, already handled by step 2.
            # dis_entities must be restricted to entities that actually appear
            # in the result: intersection result drops entities not in both bags,
            # difference result drops entities from the right bag.
            left_info = analysis.bag_info.get(defn.left)
            right_info = analysis.bag_info.get(defn.right)
            if left_info is not None and right_info is not None:
                combined = left_info.dis_entities | right_info.dis_entities
                info.dis_entities = combined & info.p_entities_multiplicity.keys()
                info.indis_entities = {}

        elif isinstance(defn, PartRef):
            partition_defn = problem.get_object(defn.partition)
            src_info = analysis.bag_info.get(partition_defn.source)
            if src_info is not None:
                info.dis_entities = src_info.dis_entities.copy()
                info.indis_entities = {k: v.copy() for k, v in src_info.indis_entities.items()}

    def _propagate_dis_entities(
        self,
        ref: ObjRef,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> None:
        """Propagate distinguishable entities from a bag to its descendants (legacy stub)."""
        pass