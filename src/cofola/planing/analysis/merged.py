"""MergedAnalysis: combines EntityAnalysis + MaxSizeInference into one AnalysisResult.

This replaces the _merge_size_inference method that was previously in pipeline.py.
LoweringPass and BagClassification both declare MergedAnalysis as a required analysis,
so they only need to call am.get(MergedAnalysis) to get fully-sized analysis results.
"""
from __future__ import annotations

from copy import deepcopy

from loguru import logger

from cofola.planing.pass_manager import AnalysisPass
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import EntityAnalysis, AnalysisResult, BagInfo, SetInfo
from cofola.planing.analysis.max_size import MaxSizeInference


class MergedAnalysis(AnalysisPass):
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
    def _cap_bag_multiplicities(info: BagInfo) -> None:
        """Keep each per-entity multiplicity compatible with bag max_size."""

        for entity in info.p_entities_multiplicity:
            info.p_entities_multiplicity[entity] = min(
                info.p_entities_multiplicity[entity],
                info.max_size,
            )

    @staticmethod
    def _has_size_conflict(
        set_info: dict[object, SetInfo],
        bag_info: dict[object, BagInfo],
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
