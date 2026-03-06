"""Entity analysis for the immutable IR.

This module implements EntityAnalysis, which computes derived properties
like p_entities, max_size, and dis_entities for all objects in a Problem.

Replaces the legacy inherit() + propagate() methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import (
    SetInit,
    SetChoose,
    SetChooseReplace,
    SetUnion,
    SetIntersection,
    SetDifference,
    BagInit,
    BagChoose,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    ObjDef,
    PartitionDef,
)
from cofola.ir.pass_manager import AnalysisPass
from cofola.frontend.problem import Problem
from cofola.frontend.constraints import SizeConstraint, BagCountAtom
from loguru import logger


@dataclass
class SetInfo:
    """Analysis result for set objects.

    Attributes:
        p_entities: Potential entities that could be in this set.
        max_size: Maximum possible size of this set.
    """

    p_entities: set[Entity]
    max_size: int


@dataclass
class BagInfo:
    """Analysis result for bag objects.

    Attributes:
        p_entities_multiplicity: Potential entities with their multiplicities.
        max_size: Maximum possible size (sum of multiplicities).
        dis_entities: Distinguishable entities (non-liftable).
        indis_entities: Indistinguishable entities grouped by multiplicity.
    """

    p_entities_multiplicity: dict[Entity, int]
    max_size: int
    dis_entities: set[Entity] = field(default_factory=set)
    indis_entities: dict[int, set[Entity]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result for a Problem.

    Attributes:
        set_info: Analysis results for set objects.
        bag_info: Analysis results for bag objects.
        all_entities: All entities used in the problem.
        singletons: Entities that appear in exactly one base object.
        unsatisfiable: True if a size conflict was detected during analysis.
    """

    set_info: dict[ObjRef, SetInfo]
    bag_info: dict[ObjRef, BagInfo]
    all_entities: set[Entity]
    singletons: set[Entity]
    unsatisfiable: bool = False


class EntityAnalysis(AnalysisPass):
    """Computes entity-related properties for all objects in a Problem.

    This is a bottom-up analysis that processes objects in topological order,
    computing p_entities, max_size for each object based on its dependencies.

    Replaces the legacy inherit() methods on each object class.
    """

    required_analyses: list[type] = []

    def run(self, problem: Problem, am=None) -> AnalysisResult:
        """Run the entity analysis on a Problem.

        Args:
            problem: The Problem to analyze.
            am: AnalysisManager (unused; EntityAnalysis has no required_analyses).

        Returns:
            AnalysisResult containing all computed information.
        """
        set_info: dict[ObjRef, SetInfo] = {}
        bag_info: dict[ObjRef, BagInfo] = {}

        topo_order = list(problem.topological_order())
        logger.debug("EntityAnalysis.run: processing {} objects in topological order",
                     len(topo_order))

        # Process objects in topological order
        for ref in topo_order:
            defn = problem.get_object(ref)
            if defn is None:
                continue

            # Dispatch based on object type
            if isinstance(defn, SetInit):
                self._analyze_set_init(ref, defn, set_info)
            elif isinstance(defn, SetChoose):
                self._analyze_set_choose(ref, defn, set_info, problem)
            elif isinstance(defn, SetChooseReplace):
                self._analyze_set_choose_replace(ref, defn, set_info, bag_info, problem)
            elif isinstance(defn, SetUnion):
                self._analyze_set_union(ref, defn, set_info, problem)
            elif isinstance(defn, SetIntersection):
                self._analyze_set_intersection(ref, defn, set_info, problem)
            elif isinstance(defn, SetDifference):
                self._analyze_set_difference(ref, defn, set_info, problem)
            elif isinstance(defn, BagInit):
                self._analyze_bag_init(ref, defn, bag_info)
            elif isinstance(defn, BagChoose):
                self._analyze_bag_choose(ref, defn, bag_info, problem)
            elif isinstance(defn, BagUnion):
                self._analyze_bag_union(ref, defn, bag_info, problem)
            elif isinstance(defn, BagAdditiveUnion):
                self._analyze_bag_additive_union(ref, defn, bag_info, problem)
            elif isinstance(defn, BagIntersection):
                self._analyze_bag_intersection(ref, defn, bag_info, problem)
            elif isinstance(defn, BagDifference):
                self._analyze_bag_difference(ref, defn, bag_info, problem)
            elif isinstance(defn, BagSupport):
                self._analyze_bag_support(ref, defn, set_info, bag_info, problem)
            # Note: TupleDef, SequenceDef, FuncDef, PartitionDef handled separately
            logger.debug("  analyzed {} ref={}", type(defn).__name__, ref.id)

        # Compute all_entities and singletons
        all_entities: set[Entity] = set()
        for info in set_info.values():
            all_entities.update(info.p_entities)
        for info in bag_info.values():
            all_entities.update(info.p_entities_multiplicity.keys())

        singletons = self._compute_singletons(set_info, bag_info, problem)

        logger.debug("EntityAnalysis result: all_entities={}, singletons={}",
                     {e.name for e in all_entities},
                     {e.name for e in singletons})

        for ref, info in set_info.items():
            logger.debug("  SetInfo ref={}: p_entities={}, max_size={}",
                         ref.id, {e.name for e in info.p_entities}, info.max_size)
        for ref, info in bag_info.items():
            logger.debug("  BagInfo ref={}: p_entities={}, max_size={}",
                         ref.id,
                         {e.name: m for e, m in info.p_entities_multiplicity.items()},
                         info.max_size)

        return AnalysisResult(
            set_info=set_info,
            bag_info=bag_info,
            all_entities=all_entities,
            singletons=singletons,
        )

    def _analyze_set_init(
        self, ref: ObjRef, defn: SetInit, set_info: dict[ObjRef, SetInfo]
    ) -> None:
        """SetInit: entities are directly given, max_size = len(entities)."""
        entities = set(defn.entities)
        set_info[ref] = SetInfo(
            p_entities=entities,
            max_size=len(entities),
        )

    def _analyze_set_choose(
        self,
        ref: ObjRef,
        defn: SetChoose,
        set_info: dict[ObjRef, SetInfo],
        problem: Problem,
    ) -> None:
        """SetChoose: inherits from source, max_size bounded by choose size."""
        src_info = set_info.get(defn.source)
        if src_info is None:
            # Source not yet processed (shouldn't happen in topological order)
            return

        max_s = src_info.max_size
        if defn.size is not None:
            max_s = min(defn.size, max_s)

        set_info[ref] = SetInfo(
            p_entities=src_info.p_entities.copy(),
            max_size=max_s,
        )

    def _analyze_set_choose_replace(
        self,
        ref: ObjRef,
        defn: SetChooseReplace,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """SetChooseReplace: treated as a bag (multiset) with multiplicities.

        For entities from the source set, each can appear multiple times.
        The multiplicity is constrained by the size if given.
        """
        src_info = set_info.get(defn.source)
        if src_info is None:
            return

        # SetChooseReplace behaves like a bag where each entity can have
        # multiplicity up to size (if given) or unlimited.
        # Use sys.maxsize as sentinel when size is None so MergedAnalysis.min()
        # correctly adopts the LP-inferred bound (same pattern as
        # _analyze_ordered_collection for replace=True).
        max_mult = defn.size if defn.size is not None else sys.maxsize
        max_size = defn.size if defn.size is not None else sys.maxsize

        p_entities_multiplicity: dict[Entity, int] = {
            entity: max_mult for entity in src_info.p_entities
        }

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=p_entities_multiplicity,
            max_size=max_size,
            dis_entities=set(),
            indis_entities={},
        )

    def _analyze_set_union(
        self,
        ref: ObjRef,
        defn: SetUnion,
        set_info: dict[ObjRef, SetInfo],
        problem: Problem,
    ) -> None:
        """SetUnion: union of p_entities, sum of max_sizes."""
        left_info = set_info.get(defn.left)
        right_info = set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        set_info[ref] = SetInfo(
            p_entities=left_info.p_entities | right_info.p_entities,
            max_size=left_info.max_size + right_info.max_size,
        )

    def _analyze_set_intersection(
        self,
        ref: ObjRef,
        defn: SetIntersection,
        set_info: dict[ObjRef, SetInfo],
        problem: Problem,
    ) -> None:
        """SetIntersection: intersection of p_entities, min of max_sizes."""
        left_info = set_info.get(defn.left)
        right_info = set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        set_info[ref] = SetInfo(
            p_entities=left_info.p_entities & right_info.p_entities,
            max_size=min(left_info.max_size, right_info.max_size),
        )

    def _analyze_set_difference(
        self,
        ref: ObjRef,
        defn: SetDifference,
        set_info: dict[ObjRef, SetInfo],
        problem: Problem,
    ) -> None:
        """SetDifference: p_entities from left, max_size from left."""
        left_info = set_info.get(defn.left)
        right_info = set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        # p_entities are those from left (conservative approximation)
        # max_size is bounded by left's max_size
        set_info[ref] = SetInfo(
            p_entities=left_info.p_entities.copy(),
            max_size=left_info.max_size,
        )

    def _analyze_bag_init(
        self, ref: ObjRef, defn: BagInit, bag_info: dict[ObjRef, BagInfo]
    ) -> None:
        """BagInit: multiplicities directly given."""
        em_dict = dict(defn.entity_multiplicity)
        max_size = sum(em_dict.values())

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=em_dict,
            max_size=max_size,
            dis_entities=set(),
            indis_entities={},
        )

    def _analyze_bag_choose(
        self,
        ref: ObjRef,
        defn: BagChoose,
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagChoose: inherits from source, max_size bounded by choose size."""
        src_info = bag_info.get(defn.source)
        if src_info is None:
            return

        max_s = src_info.max_size
        if defn.size is not None:
            max_s = min(defn.size, max_s)

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=src_info.p_entities_multiplicity.copy(),
            max_size=max_s,
            dis_entities=src_info.dis_entities.copy(),
            indis_entities={k: v.copy() for k, v in src_info.indis_entities.items()},
        )

    def _analyze_bag_union(
        self,
        ref: ObjRef,
        defn: BagUnion,
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagUnion: max multiplicity per entity, max_size is sum of max multiplicities."""
        left_info = bag_info.get(defn.left)
        right_info = bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        p_entities_multiplicity: dict[Entity, int] = {}
        for entity, mult in left_info.p_entities_multiplicity.items():
            if entity in right_info.p_entities_multiplicity:
                p_entities_multiplicity[entity] = max(
                    mult, right_info.p_entities_multiplicity[entity]
                )
            else:
                p_entities_multiplicity[entity] = mult

        for entity, mult in right_info.p_entities_multiplicity.items():
            if entity not in p_entities_multiplicity:
                p_entities_multiplicity[entity] = mult

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=p_entities_multiplicity,
            max_size=sum(p_entities_multiplicity.values()),
            dis_entities=left_info.dis_entities | right_info.dis_entities,
            indis_entities={},
        )

    def _analyze_bag_additive_union(
        self,
        ref: ObjRef,
        defn: BagAdditiveUnion,
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagAdditiveUnion: sum of multiplicities, all entities distinguishable."""
        left_info = bag_info.get(defn.left)
        right_info = bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        all_entities = (
            set(left_info.p_entities_multiplicity.keys())
            | set(right_info.p_entities_multiplicity.keys())
        )
        p_entities_multiplicity: dict[Entity, int] = {}
        for entity in all_entities:
            p_entities_multiplicity[entity] = (
                left_info.p_entities_multiplicity.get(entity, 0)
                + right_info.p_entities_multiplicity.get(entity, 0)
            )

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=p_entities_multiplicity,
            max_size=left_info.max_size + right_info.max_size,
            dis_entities=left_info.dis_entities | right_info.dis_entities,
            indis_entities={},  # All entities are distinguishable in additive union
        )

    def _analyze_bag_intersection(
        self,
        ref: ObjRef,
        defn: BagIntersection,
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagIntersection: min multiplicity per entity."""
        left_info = bag_info.get(defn.left)
        right_info = bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        p_entities_multiplicity: dict[Entity, int] = {}
        for entity, mult in left_info.p_entities_multiplicity.items():
            if entity in right_info.p_entities_multiplicity:
                p_entities_multiplicity[entity] = min(
                    mult, right_info.p_entities_multiplicity[entity]
                )

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=p_entities_multiplicity,
            max_size=min(left_info.max_size, right_info.max_size),
            dis_entities=left_info.dis_entities & right_info.dis_entities,
            indis_entities={},
        )

    def _analyze_bag_difference(
        self,
        ref: ObjRef,
        defn: BagDifference,
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagDifference: inherits from first, dis_entities intersection."""
        left_info = bag_info.get(defn.left)
        right_info = bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        bag_info[ref] = BagInfo(
            p_entities_multiplicity=left_info.p_entities_multiplicity.copy(),
            max_size=left_info.max_size,
            dis_entities=left_info.dis_entities & right_info.dis_entities,
            indis_entities={},
        )

    def _analyze_bag_support(
        self,
        ref: ObjRef,
        defn: BagSupport,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
        problem: Problem,
    ) -> None:
        """BagSupport: unique entities from bag, max_size from bag."""
        src_info = bag_info.get(defn.source)
        if src_info is None:
            return

        entities = set(src_info.p_entities_multiplicity.keys())
        set_info[ref] = SetInfo(
            p_entities=entities,
            max_size=src_info.max_size,
        )

    def _compute_singletons(
        self,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
        problem,
    ) -> set[Entity]:
        """Compute singleton entities following legacy preprocess_bags logic.

        An entity is a singleton if:
        - It appears exactly once across all SetInit/BagInit objects, AND
        - Its multiplicity in BagInit is 1 (not a repeated bag element), AND
        - The problem has no PartitionDef (partitions require full encoding), AND
        - It does not appear in any BagCountAtom constraint (multiplicities must be tracked).

        Ports the legacy CofolaProblem.update_singletons() logic.
        """
        # If any PartitionDef exists, no singletons (partitions need entity tracking)
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if isinstance(defn, PartitionDef):
                return set()

        singletons: set[Entity] = set()

        # From BagInit: only entities with multiplicity == 1
        for ref, info in bag_info.items():
            defn = problem.get_object(ref)
            if isinstance(defn, BagInit):
                for entity, mult in info.p_entities_multiplicity.items():
                    if mult == 1:
                        singletons.add(entity)

        # From SetInit: all entities (count = 1 by definition)
        entity_count: dict[Entity, int] = {}
        for ref, info in set_info.items():
            for entity in info.p_entities:
                entity_count[entity] = entity_count.get(entity, 0) + 1
        # Only keep entities that appear in exactly one set
        set_singletons = {e for e, c in entity_count.items() if c == 1}
        singletons.update(set_singletons)

        # Remove entities that appear in any bag (including SetChooseReplace) with mult > 1
        # This mirrors the legacy update_singletons which discarded entities with mult != 1
        for ref, info in bag_info.items():
            for entity, mult in info.p_entities_multiplicity.items():
                if mult != 1:
                    singletons.discard(entity)

        # Remove entities that appear in BagCountAtom constraints
        for constraint in problem.constraints:
            if isinstance(constraint, SizeConstraint):
                for term, _ in constraint.terms:
                    if isinstance(term, BagCountAtom):
                        singletons.discard(term.entity)

        return singletons