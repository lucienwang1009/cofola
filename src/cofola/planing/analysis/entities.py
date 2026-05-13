"""Entity analysis for the immutable IR.

This module implements EntityAnalysis, which computes derived properties
like p_entities, max_size, and dis_entities for all objects in a Problem.

Replaces the legacy inherit() + propagate() methods.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from cofola.frontend.objects import Entity, ObjRef
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
    CircleDef,
    FuncDef,
    FuncImage,
    FuncInverseImage,
    Ordered,
    TupleDef,
    SequenceDef,
    PartDef,
    Grouped,
)
from cofola.planing.pass_manager import AnalysisPass
from cofola.frontend.problem import Problem
from cofola.frontend.constraints import SizeConstraint, BagCountAtom
from loguru import logger


@dataclass
class SetInfo:
    """Analysis result for set objects.

    Attributes:
        p_entities: Potential entities that could be in this set.
        max_size: Maximum possible size of this set.
        exact_size: Known exact size if fixed at analysis time, else None.
    """

    p_entities: set[Entity]
    max_size: int
    exact_size: int | None = None


@dataclass
class BagInfo:
    """Analysis result for bag objects.

    Attributes:
        p_entities_multiplicity: Potential entities with their multiplicities.
        max_size: Maximum possible size (sum of multiplicities).
        dis_entities: Distinguishable entities (non-liftable).
        indis_entities: Indistinguishable entities grouped by multiplicity.
        exact_size: Known exact size if fixed at analysis time, else None.
    """

    p_entities_multiplicity: dict[Entity, int]
    max_size: int
    dis_entities: set[Entity] = field(default_factory=set)
    indis_entities: dict[int, set[Entity]] = field(default_factory=dict)
    exact_size: int | None = None


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


@dataclass
class _AnalysisState:
    """Mutable construction state for EntityAnalysis results.

    All SetInfo/BagInfo writes go through this builder so local analysis
    branches cannot accidentally bypass core size invariants.
    """

    set_info: dict[ObjRef, SetInfo] = field(default_factory=dict)
    bag_info: dict[ObjRef, BagInfo] = field(default_factory=dict)
    unsatisfiable: bool = False

    def mark_unsatisfiable(self, message: str, *args: object) -> None:
        logger.info(message, *args)
        self.unsatisfiable = True

    @staticmethod
    def _nonnegative(value: int, label: str) -> tuple[int, bool]:
        if value >= 0:
            return value, False
        logger.info("EntityAnalysis: {} is negative ({})", label, value)
        return 0, True

    def add_set_info(
        self,
        ref: ObjRef,
        p_entities: set[Entity],
        max_size: int,
        exact_size: int | None = None,
    ) -> None:
        entities = set(p_entities)
        max_size, bad_max = self._nonnegative(max_size, f"SetInfo ref={ref.id} max_size")
        self.unsatisfiable |= bad_max

        max_size = min(max_size, len(entities))
        if exact_size is not None:
            exact_size, bad_exact = self._nonnegative(
                exact_size,
                f"SetInfo ref={ref.id} exact_size",
            )
            self.unsatisfiable |= bad_exact
            if exact_size > max_size:
                self.mark_unsatisfiable(
                    "EntityAnalysis: SetInfo ref={} exact_size {} exceeds max_size {}",
                    ref.id,
                    exact_size,
                    max_size,
                )
                exact_size = max_size

        self.set_info[ref] = SetInfo(
            p_entities=entities,
            max_size=max_size,
            exact_size=exact_size,
        )

    def add_bag_info(
        self,
        ref: ObjRef,
        p_entities_multiplicity: dict[Entity, int],
        max_size: int,
        exact_size: int | None = None,
        dis_entities: set[Entity] | None = None,
        indis_entities: dict[int, set[Entity]] | None = None,
    ) -> None:
        multiplicities: dict[Entity, int] = {}
        for entity, multiplicity in p_entities_multiplicity.items():
            clean_multiplicity, bad_mult = self._nonnegative(
                multiplicity,
                f"BagInfo ref={ref.id} multiplicity[{entity.name}]",
            )
            self.unsatisfiable |= bad_mult
            multiplicities[entity] = clean_multiplicity

        max_size, bad_max = self._nonnegative(max_size, f"BagInfo ref={ref.id} max_size")
        self.unsatisfiable |= bad_max
        max_size = min(max_size, sum(multiplicities.values()))

        if exact_size is not None:
            exact_size, bad_exact = self._nonnegative(
                exact_size,
                f"BagInfo ref={ref.id} exact_size",
            )
            self.unsatisfiable |= bad_exact
            if exact_size > max_size:
                self.mark_unsatisfiable(
                    "EntityAnalysis: BagInfo ref={} exact_size {} exceeds max_size {}",
                    ref.id,
                    exact_size,
                    max_size,
                )
                exact_size = max_size

        self.bag_info[ref] = BagInfo(
            p_entities_multiplicity=multiplicities,
            max_size=max_size,
            dis_entities=set(dis_entities or set()),
            indis_entities={k: set(v) for k, v in (indis_entities or {}).items()},
            exact_size=exact_size,
        )


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
        state = _AnalysisState()

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
                self._analyze_set_init(ref, defn, state)
            elif isinstance(defn, SetChoose):
                self._analyze_set_choose(ref, defn, state)
            elif isinstance(defn, SetChooseReplace):
                self._analyze_set_choose_replace(ref, defn, state)
            elif isinstance(defn, SetUnion):
                self._analyze_set_union(ref, defn, state)
            elif isinstance(defn, SetIntersection):
                self._analyze_set_intersection(ref, defn, state)
            elif isinstance(defn, SetDifference):
                self._analyze_set_difference(ref, defn, state, problem)
            elif isinstance(defn, BagInit):
                self._analyze_bag_init(ref, defn, state)
            elif isinstance(defn, BagChoose):
                self._analyze_bag_choose(ref, defn, state)
            elif isinstance(defn, BagUnion):
                self._analyze_bag_union(ref, defn, state)
            elif isinstance(defn, BagAdditiveUnion):
                self._analyze_bag_additive_union(ref, defn, state)
            elif isinstance(defn, BagIntersection):
                self._analyze_bag_intersection(ref, defn, state)
            elif isinstance(defn, BagDifference):
                self._analyze_bag_difference(ref, defn, state)
            elif isinstance(defn, BagSupport):
                self._analyze_bag_support(ref, defn, state)
            elif isinstance(defn, FuncImage):
                self._analyze_func_image(ref, defn, state, problem)
            elif isinstance(defn, FuncInverseImage):
                self._analyze_func_inverse_image(ref, defn, state, problem)
            elif isinstance(defn, Ordered):
                self._analyze_ordered_collection(ref, defn, state)
            elif isinstance(defn, PartDef):
                self._analyze_part_ref(ref, defn, state, problem)
            # Note: FuncDef, FuncInverse, PartitionDef not analyzed
            logger.debug("  analyzed {} ref={}", type(defn).__name__, ref.id)

        # Compute all_entities and singletons
        all_entities: set[Entity] = set()
        for info in state.set_info.values():
            all_entities.update(info.p_entities)
        for info in state.bag_info.values():
            all_entities.update(info.p_entities_multiplicity.keys())

        singletons = self._compute_singletons(state.set_info, state.bag_info, problem)

        logger.debug("EntityAnalysis result: all_entities={}, singletons={}",
                     {e.name for e in all_entities},
                     {e.name for e in singletons})

        for ref, info in state.set_info.items():
            logger.debug("  SetInfo ref={}: p_entities={}, max_size={}, exact_size={}",
                         ref.id, {e.name for e in info.p_entities}, info.max_size, info.exact_size)
        for ref, info in state.bag_info.items():
            logger.debug("  BagInfo ref={}: p_entities={}, max_size={}, exact_size={}",
                         ref.id,
                         {e.name: m for e, m in info.p_entities_multiplicity.items()},
                         info.max_size, info.exact_size)

        return AnalysisResult(
            set_info=state.set_info,
            bag_info=state.bag_info,
            all_entities=all_entities,
            singletons=singletons,
            unsatisfiable=state.unsatisfiable,
        )

    def _analyze_set_init(
        self, ref: ObjRef, defn: SetInit, state: _AnalysisState
    ) -> None:
        """SetInit: entities are directly given, max_size = len(entities)."""
        entities = set(defn.entities)
        n = len(entities)
        state.add_set_info(ref, entities, max_size=n, exact_size=n)

    def _analyze_set_choose(
        self,
        ref: ObjRef,
        defn: SetChoose,
        state: _AnalysisState,
    ) -> None:
        """SetChoose: inherits from source, max_size bounded by choose size."""
        src_info = state.set_info.get(defn.source)
        if src_info is None:
            # Source not yet processed (shouldn't happen in topological order)
            return

        max_s = src_info.max_size
        if defn.size is not None:
            if defn.size > src_info.max_size:
                state.mark_unsatisfiable(
                    "EntityAnalysis: SetChoose ref={} requests size {} "
                    "from source max_size {}",
                    ref.id,
                    defn.size,
                    src_info.max_size,
                )
            max_s = min(defn.size, max_s)

        state.add_set_info(
            ref,
            src_info.p_entities,
            max_size=max_s,
            exact_size=defn.size,
        )

    def _analyze_set_choose_replace(
        self,
        ref: ObjRef,
        defn: SetChooseReplace,
        state: _AnalysisState,
    ) -> None:
        """SetChooseReplace: treated as a bag (multiset) with multiplicities.

        For entities from the source set, each can appear multiple times.
        The multiplicity is constrained by the size if given.
        """
        src_info = state.set_info.get(defn.source)
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

        state.add_bag_info(
            ref,
            p_entities_multiplicity,
            max_size=max_size,
            exact_size=defn.size,
        )

    def _analyze_set_union(
        self,
        ref: ObjRef,
        defn: SetUnion,
        state: _AnalysisState,
    ) -> None:
        """SetUnion: union of p_entities, sum of max_sizes."""
        left_info = state.set_info.get(defn.left)
        right_info = state.set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        p_entities = left_info.p_entities | right_info.p_entities
        state.add_set_info(
            ref,
            p_entities,
            max_size=left_info.max_size + right_info.max_size,
        )

    def _analyze_set_intersection(
        self,
        ref: ObjRef,
        defn: SetIntersection,
        state: _AnalysisState,
    ) -> None:
        """SetIntersection: intersection of p_entities, min of max_sizes."""
        left_info = state.set_info.get(defn.left)
        right_info = state.set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        state.add_set_info(
            ref,
            left_info.p_entities & right_info.p_entities,
            max_size=min(left_info.max_size, right_info.max_size),
        )

    def _analyze_set_difference(
        self,
        ref: ObjRef,
        defn: SetDifference,
        state: _AnalysisState,
        problem: Problem,
    ) -> None:
        """SetDifference: p_entities from left, max_size from left."""
        left_info = state.set_info.get(defn.left)
        right_info = state.set_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        # p_entities are those from left (conservative approximation).
        # The RHS may be disjoint from the LHS, so its exact_size alone cannot
        # reduce the upper bound.
        max_s = left_info.max_size

        # exact_size: known iff right is structurally a subset of left
        # (e.g. SetChoose/SetChooseReplace with source == left), so |left - right| = |left| - |right|
        right_defn = problem.get_object(defn.right)
        right_is_subset_of_left = (
            isinstance(right_defn, (SetChoose, SetChooseReplace))
            and right_defn.source == defn.left
        )
        if (right_is_subset_of_left
                and left_info.exact_size is not None
                and right_info.exact_size is not None):
            exact_s: int | None = left_info.exact_size - right_info.exact_size
        else:
            exact_s = None

        state.add_set_info(
            ref,
            left_info.p_entities,
            max_size=max_s,
            exact_size=exact_s,
        )

    def _analyze_bag_init(
        self, ref: ObjRef, defn: BagInit, state: _AnalysisState
    ) -> None:
        """BagInit: multiplicities directly given."""
        em_dict = dict(defn.entity_multiplicity)
        max_size = sum(em_dict.values())

        state.add_bag_info(ref, em_dict, max_size=max_size, exact_size=max_size)

    def _analyze_bag_choose(
        self,
        ref: ObjRef,
        defn: BagChoose,
        state: _AnalysisState,
    ) -> None:
        """BagChoose: inherits from source, max_size bounded by choose size."""
        src_info = state.bag_info.get(defn.source)
        if src_info is None:
            return

        max_s = src_info.max_size
        if defn.size is not None:
            if defn.size > src_info.max_size:
                state.mark_unsatisfiable(
                    "EntityAnalysis: BagChoose ref={} requests size {} "
                    "from source max_size {}",
                    ref.id,
                    defn.size,
                    src_info.max_size,
                )
            max_s = min(defn.size, max_s)

        state.add_bag_info(
            ref,
            src_info.p_entities_multiplicity,
            max_size=max_s,
            exact_size=defn.size,
            dis_entities=src_info.dis_entities.copy(),
            indis_entities={k: v.copy() for k, v in src_info.indis_entities.items()},
        )

    def _analyze_bag_union(
        self,
        ref: ObjRef,
        defn: BagUnion,
        state: _AnalysisState,
    ) -> None:
        """BagUnion: max multiplicity per entity, max_size is sum of max multiplicities."""
        left_info = state.bag_info.get(defn.left)
        right_info = state.bag_info.get(defn.right)

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

        state.add_bag_info(
            ref,
            p_entities_multiplicity,
            max_size=sum(p_entities_multiplicity.values()),
            dis_entities=left_info.dis_entities | right_info.dis_entities,
            indis_entities={},
        )

    def _analyze_bag_additive_union(
        self,
        ref: ObjRef,
        defn: BagAdditiveUnion,
        state: _AnalysisState,
    ) -> None:
        """BagAdditiveUnion: sum of multiplicities, all entities distinguishable."""
        left_info = state.bag_info.get(defn.left)
        right_info = state.bag_info.get(defn.right)

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

        # exact_size: only valid if entity sets are disjoint and both operands have exact_size
        left_keys = set(left_info.p_entities_multiplicity.keys())
        right_keys = set(right_info.p_entities_multiplicity.keys())
        if (
            left_keys.isdisjoint(right_keys)
            and left_info.exact_size is not None
            and right_info.exact_size is not None
        ):
            exact_s: int | None = left_info.exact_size + right_info.exact_size
        else:
            exact_s = None

        state.add_bag_info(
            ref,
            p_entities_multiplicity,
            max_size=left_info.max_size + right_info.max_size,
            exact_size=exact_s,
            dis_entities=left_info.dis_entities | right_info.dis_entities,
            indis_entities={},  # All entities are distinguishable in additive union
        )

    def _analyze_bag_intersection(
        self,
        ref: ObjRef,
        defn: BagIntersection,
        state: _AnalysisState,
    ) -> None:
        """BagIntersection: min multiplicity per entity."""
        left_info = state.bag_info.get(defn.left)
        right_info = state.bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        p_entities_multiplicity: dict[Entity, int] = {}
        for entity, mult in left_info.p_entities_multiplicity.items():
            if entity in right_info.p_entities_multiplicity:
                p_entities_multiplicity[entity] = min(
                    mult, right_info.p_entities_multiplicity[entity]
                )

        state.add_bag_info(
            ref,
            p_entities_multiplicity,
            max_size=min(left_info.max_size, right_info.max_size),
            dis_entities=left_info.dis_entities & right_info.dis_entities,
            indis_entities={},
        )

    def _analyze_bag_difference(
        self,
        ref: ObjRef,
        defn: BagDifference,
        state: _AnalysisState,
    ) -> None:
        """BagDifference: possible multiplicity is bounded by the left operand."""
        left_info = state.bag_info.get(defn.left)
        right_info = state.bag_info.get(defn.right)

        if left_info is None or right_info is None:
            return

        state.add_bag_info(
            ref,
            dict(left_info.p_entities_multiplicity),
            max_size=left_info.max_size,
            dis_entities=set(left_info.p_entities_multiplicity),
            indis_entities={},
        )

    def _analyze_bag_support(
        self,
        ref: ObjRef,
        defn: BagSupport,
        state: _AnalysisState,
    ) -> None:
        """BagSupport: unique entities from bag, max_size from bag."""
        src_info = state.bag_info.get(defn.source)
        if src_info is None:
            return

        entities = set(src_info.p_entities_multiplicity.keys())
        state.add_set_info(ref, entities, max_size=src_info.max_size)

    def _analyze_func_image(
        self,
        ref: ObjRef,
        defn: FuncImage,
        state: _AnalysisState,
        problem: Problem,
    ) -> None:
        """FuncImage: image f(A), always a set, subset of codomain.

        p_entities = codomain's p_entities (conservative).
        max_size   = min(|argument|, |codomain|).
        """
        func_defn = problem.get_object(defn.func)
        if not isinstance(func_defn, FuncDef):
            return
        codomain_info = state.set_info.get(func_defn.codomain)
        if codomain_info is None:
            return

        argument = defn.argument
        if isinstance(argument, Entity):
            # f(a) for a single entity: at most 1 result element
            max_size = 1
        else:
            arg_info = state.set_info.get(argument)
            if arg_info is not None:
                max_size = min(arg_info.max_size, codomain_info.max_size)
            else:
                max_size = codomain_info.max_size

        state.add_set_info(ref, codomain_info.p_entities, max_size=max_size)

    def _analyze_func_inverse_image(
        self,
        ref: ObjRef,
        defn: FuncInverseImage,
        state: _AnalysisState,
        problem: Problem,
    ) -> None:
        """FuncInverseImage: preimage f⁻¹(B), always a set, subset of domain.

        p_entities = domain's p_entities (conservative).
        max_size   = |domain|.
        """
        func_defn = problem.get_object(defn.func)
        if not isinstance(func_defn, FuncDef):
            return
        domain_info = state.set_info.get(func_defn.domain)
        if domain_info is None:
            return

        state.add_set_info(ref, domain_info.p_entities, max_size=domain_info.max_size)

    def _analyze_ordered_collection(
        self,
        ref: ObjRef,
        defn: TupleDef | SequenceDef | CircleDef,
        state: _AnalysisState,
    ) -> None:
        """Shared analysis for TupleDef and SequenceDef.

        Routes to set_info or bag_info based on replace flag and source type:
        - replace=False + set source  → SetInfo  (each entity at most once)
        - replace=False + bag source  → BagInfo  (inherit source multiplicities)
        - replace=True  + set source  → BagInfo  (each entity up to size times)
        """
        src_set = state.set_info.get(defn.source)
        src_bag = state.bag_info.get(defn.source)

        if src_set is None and src_bag is None:
            raise ValueError(
                f"{type(defn).__name__} ref={ref.id}: source ref={defn.source.id} "
                f"has no set_info or bag_info — source must be analyzed before this object"
            )

        src_max = src_set.max_size if src_set is not None else src_bag.max_size
        src_exact = src_set.exact_size if src_set is not None else src_bag.exact_size

        # exact_size:
        #   - defn.size is given → use it
        #   - choose=False, no size → this is a full permutation of the source;
        #     exact_size = source's exact_size (may still be None if source is dynamic)
        #   - choose=True, no size → size is free (determined by constraints)
        exact_s = defn.size if defn.size is not None else (
            src_exact if not defn.choose else None
        )
        if (
            defn.size is not None
            and not defn.replace
            and defn.size > src_max
        ):
            state.mark_unsatisfiable(
                "EntityAnalysis: {} ref={} requests size {} from source max_size {}",
                type(defn).__name__,
                ref.id,
                defn.size,
                src_max,
            )

        if defn.replace:
            # replace=True: size is unconstrained by source; use sys.maxsize so
            # MergedAnalysis min() correctly adopts the LP-inferred bound.
            max_s = defn.size if defn.size is not None else sys.maxsize
            mult = defn.size if defn.size is not None else sys.maxsize
            entities = (
                src_set.p_entities if src_set is not None
                else set(src_bag.p_entities_multiplicity.keys())
            )
            state.add_bag_info(
                ref,
                {e: mult for e in entities},
                max_size=max_s,
                exact_size=exact_s,
            )
        elif src_bag is not None:
            # replace=False + bag source: max bounded by source, inherit multiplicities
            max_s = min(src_max, defn.size) if defn.size is not None else src_max
            state.add_bag_info(
                ref,
                src_bag.p_entities_multiplicity,
                max_size=max_s,
                exact_size=exact_s,
            )
        else:
            # replace=False + set source: max bounded by source, no repetitions
            max_s = min(src_max, defn.size) if defn.size is not None else src_max
            state.add_set_info(
                ref,
                src_set.p_entities,
                max_size=max_s,
                exact_size=exact_s,
            )

    def _analyze_part_ref(
        self,
        ref: ObjRef,
        defn: PartDef,
        state: _AnalysisState,
        problem: Problem,
    ) -> None:
        """PartDef: inherit entity information from the partition's source.

        Each part has the same *potential* entities as the source (any entity
        could end up in any part).  max_size is conservatively the source's
        max_size — MaxSizeInference can tighten this later via SizeConstraints.

        Set partition  → SetInfo
        Bag partition  → BagInfo
        """
        partition_defn = problem.get_object(defn.partition)
        if not isinstance(partition_defn, Grouped):
            raise ValueError(
                f"PartDef ref={ref.id}: partition ref={defn.partition.id} "
                "does not reference a Partition or Composition"
            )
        source = partition_defn.source

        if source in state.set_info:
            src = state.set_info[source]
            state.add_set_info(ref, src.p_entities, max_size=src.max_size)
        elif source in state.bag_info:
            src = state.bag_info[source]
            state.add_bag_info(ref, src.p_entities_multiplicity, max_size=src.max_size)

    def _compute_singletons(
        self,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
        problem,
    ) -> set[Entity]:
        """Compute singleton entities.

        An entity is a singleton if its multiplicity is at most 1 in every
        bag object in the problem — i.e., it cannot be a repeated element
        (multiplicity > 1) in any bag.

        SetInit entities always satisfy this (sets have no repeated elements).
        """
        # All entities in the problem are candidates
        all_entities: set[Entity] = set()
        for info in set_info.values():
            all_entities.update(info.p_entities)
        for info in bag_info.values():
            all_entities.update(info.p_entities_multiplicity.keys())

        # Disqualify entities whose max multiplicity exceeds 1 in any bag object
        non_singletons: set[Entity] = set()
        for info in bag_info.values():
            for entity, mult in info.p_entities_multiplicity.items():
                if mult > 1:
                    non_singletons.add(entity)

        # Disqualify entities referenced by BagCountAtom constraints
        # (their multiplicities must be tracked per-instance, not fixed)
        for constraint in problem.constraints:
            if isinstance(constraint, SizeConstraint):
                for term, _ in constraint.terms:
                    if isinstance(term, BagCountAtom):
                        non_singletons.add(term.entity)

        return all_entities - non_singletons
