"""MergedAnalysis combines EntityAnalysis and MaxSizeInference."""
from __future__ import annotations

from copy import deepcopy

from loguru import logger

from cofola.frontend.objects import CircleDef, ObjRef, SequenceDef, TupleDef
from cofola.planing.pass_manager import AnalysisPass
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import EntityAnalysis, AnalysisResult, BagInfo, SetInfo
from cofola.planing.analysis.max_size import MaxSizeInference


class MergedAnalysis(AnalysisPass[AnalysisResult]):
    """Merge EntityAnalysis and MaxSizeInference into a single AnalysisResult.

    Returns a new AnalysisResult (not mutated) with max_size and exact_size
    updated from the LP results. Sets unsatisfiable=True if a size conflict
    is detected.
    """

    required_analyses = [EntityAnalysis, MaxSizeInference]

    def run(self, problem: Problem, am=None) -> AnalysisResult:
        """Run merged analysis.

        Args:
            problem: The Problem to analyze.
            am: AnalysisManager for accessing EntityAnalysis and MaxSizeInference.

        Returns:
            AnalysisResult with sizes merged from LP inference.
        """
        from cofola.planing.pass_manager import AnalysisManager
        if not isinstance(am, AnalysisManager):
            raise ValueError("MergedAnalysis requires an AnalysisManager")

        base = am.get(EntityAnalysis)
        sizes = am.get(MaxSizeInference)

        if base.unsatisfiable:
            logger.info("MergedAnalysis: EntityAnalysis found unsatisfiable constraints")
            return self._unsat_result(base)

        if sizes.unsatisfiable:
            logger.info("MergedAnalysis: MaxSizeInference found unsatisfiable constraints")
            return self._unsat_result(base)

        # Deep-copy set_info and bag_info so we don't mutate the cached base result
        set_info = deepcopy(base.set_info)
        bag_info = deepcopy(base.bag_info)

        for ref, size in sizes.max_sizes.items():
            if ref in set_info:
                set_info[ref].max_size = min(set_info[ref].max_size, size)
                logger.debug("  MergedAnalysis set ref={}: max_size → {}", ref.id, set_info[ref].max_size)
            elif ref in bag_info:
                info = bag_info[ref]
                info.max_size = min(info.max_size, size)
                self._cap_bag_multiplicities(info)
                logger.debug("  MergedAnalysis bag ref={}: max_size → {}", ref.id, info.max_size)

        for ref, exact in sizes.exact_sizes.items():
            if ref in set_info:
                info = set_info[ref]
                if info.exact_size is not None and info.exact_size != exact:
                    logger.info(
                        "MergedAnalysis: exact_size conflict on ref={}: EA={} LP={}",
                        ref.id, info.exact_size, exact,
                    )
                    return self._unsat_result(base, set_info=set_info, bag_info=bag_info)
                info.exact_size = exact
                info.max_size = min(info.max_size, exact)
            elif ref in bag_info:
                info = bag_info[ref]
                if info.exact_size is not None and info.exact_size != exact:
                    logger.info(
                        "MergedAnalysis: exact_size conflict on ref={}: EA={} LP={}",
                        ref.id, info.exact_size, exact,
                    )
                    return self._unsat_result(base, set_info=set_info, bag_info=bag_info)
                info.exact_size = exact
                info.max_size = min(info.max_size, exact)
                self._cap_bag_multiplicities(info)

        if not self._propagate_full_ordered_sizes(problem, set_info, bag_info):
            return self._unsat_result(base, set_info=set_info, bag_info=bag_info)

        if self._has_size_conflict(set_info, bag_info):
            return self._unsat_result(base, set_info=set_info, bag_info=bag_info)

        return AnalysisResult(
            set_info=set_info,
            bag_info=bag_info,
            all_entities=base.all_entities,
            singletons=base.singletons,
        )

    def _unsat_result(
        self,
        base: AnalysisResult,
        *,
        set_info: dict | None = None,
        bag_info: dict | None = None,
    ) -> AnalysisResult:
        """Return an AnalysisResult that preserves available facts but is unsat."""

        return AnalysisResult(
            set_info=base.set_info if set_info is None else set_info,
            bag_info=base.bag_info if bag_info is None else bag_info,
            all_entities=base.all_entities,
            singletons=base.singletons,
            unsatisfiable=True,
        )

    @staticmethod
    def _get_info(
        ref: ObjRef,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
    ) -> SetInfo | BagInfo | None:
        return set_info.get(ref) or bag_info.get(ref)

    @classmethod
    def _set_max_size(cls, info: SetInfo | BagInfo, max_size: int) -> bool:
        if info.max_size == max_size:
            return False
        info.max_size = max_size
        if isinstance(info, BagInfo):
            cls._cap_bag_multiplicities(info)
        return True

    @classmethod
    def _set_exact_size(cls, info: SetInfo | BagInfo, exact_size: int) -> bool:
        changed = info.exact_size != exact_size
        info.exact_size = exact_size
        if exact_size < info.max_size:
            info.max_size = exact_size
            changed = True
        if isinstance(info, BagInfo):
            cls._cap_bag_multiplicities(info)
        return changed

    @classmethod
    def _propagate_full_ordered_sizes(
        cls,
        problem: Problem,
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
    ) -> bool:
        """Propagate |ordered| == |source| for non-choose ordered objects."""

        changed = True
        while changed:
            changed = False
            for ref, defn in problem.defs:
                if not isinstance(defn, (TupleDef, SequenceDef, CircleDef)):
                    continue
                if defn.choose or defn.replace:
                    continue

                own_info = cls._get_info(ref, set_info, bag_info)
                source_info = cls._get_info(defn.source, set_info, bag_info)
                if own_info is None or source_info is None:
                    continue

                shared_max = min(own_info.max_size, source_info.max_size)
                changed |= cls._set_max_size(own_info, shared_max)
                changed |= cls._set_max_size(source_info, shared_max)

                own_exact = own_info.exact_size
                source_exact = source_info.exact_size
                if own_exact is not None and source_exact is not None:
                    if own_exact != source_exact:
                        logger.info(
                            "MergedAnalysis: full ordered size conflict on ref={} "
                            "and source ref={}: {} != {}",
                            ref.id,
                            defn.source.id,
                            own_exact,
                            source_exact,
                        )
                        return False
                    continue

                exact = own_exact if own_exact is not None else source_exact
                if exact is not None:
                    changed |= cls._set_exact_size(own_info, exact)
                    changed |= cls._set_exact_size(source_info, exact)

        return True

    @staticmethod
    def _cap_bag_multiplicities(info: BagInfo) -> None:
        """Keep each per-entity multiplicity compatible with bag max_size."""

        for entity in info.p_entities_multiplicity:
            info.p_entities_multiplicity[entity] = min(
                info.p_entities_multiplicity[entity],
                info.max_size,
            )

    @staticmethod
    def _has_size_conflict(
        set_info: dict[ObjRef, SetInfo],
        bag_info: dict[ObjRef, BagInfo],
    ) -> bool:
        """Validate post-merge size invariants."""

        for ref, info in set_info.items():
            if info.exact_size is not None and info.exact_size > info.max_size:
                logger.info(
                    "MergedAnalysis: set ref={} exact_size {} exceeds max_size {}",
                    getattr(ref, "id", ref),
                    info.exact_size,
                    info.max_size,
                )
                return True

        for ref, info in bag_info.items():
            if info.exact_size is not None and info.exact_size > info.max_size:
                logger.info(
                    "MergedAnalysis: bag ref={} exact_size {} exceeds max_size {}",
                    getattr(ref, "id", ref),
                    info.exact_size,
                    info.max_size,
                )
                return True
        return False
