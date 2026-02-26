"""New IR-based pipeline for Cofola.

This module provides the IRPipeline class, which implements the
immutable IR-based pipeline for solving combinatorics problems.

The pipeline stages are:
1. Entity Analysis
2. Constant Folding
3. Max Size Inference
4. Lowering
5. Simplification
6. Bag Classification
7. Solve via IR-native backend (WFOMCBackend)
"""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

from cofola.frontend.problem import Problem
from cofola.frontend.constraints import OrConstraint
from cofola.frontend.pretty import fmt_problem, fmt_analysis
from cofola.ir.analysis.entities import EntityAnalysis, AnalysisResult
from cofola.ir.analysis.max_size import MaxSizeInference
from cofola.ir.analysis.bag_classify import BagClassification
from cofola.ir.passes.optimize import ConstantFolder
from cofola.ir.passes.lowering import LoweringPass
from cofola.ir.passes.simplify import SimplifyPass
from cofola.backend.wfomc.backend import WFOMCBackend


class IRPipeline:
    """Pipeline using immutable IR.

    This pipeline processes a combinatorics problem through the
    immutable IR, running analysis and transformation passes, then
    calling the WFOMC backend.
    """

    def solve(self, problem: "Problem") -> int:
        """Solve a combinatorics problem.

        Args:
            problem: The cofola.frontend.Problem to solve.

        Returns:
            The count of solutions.
        """
        # Initial state
        logger.debug("\n{}", fmt_problem(problem, stage="[Input] Parsed Problem"))

        # Stage 1: Entity Analysis
        logger.info("[Stage 1] EntityAnalysis — {} objects", len(problem.defs))
        analysis = EntityAnalysis().run(problem)
        logger.debug("[Stage 1] entities={}, singletons={}",
                     {e.name for e in analysis.all_entities},
                     {e.name for e in analysis.singletons})
        logger.debug("\n{}", fmt_analysis(analysis, problem, stage="[After Stage 1] EntityAnalysis"))

        # Stage 2: Optimize (constant folding)
        logger.info("[Stage 2] ConstantFolder")
        problem = ConstantFolder().rewrite(problem)
        logger.debug("[Stage 2] after fold: {} objects", len(problem.defs))
        logger.debug("\n{}", fmt_problem(problem, stage="[After Stage 2] ConstantFolder"))

        # Re-analyze after optimization
        logger.debug("[Stage 2b] Re-running EntityAnalysis after fold")
        analysis = EntityAnalysis().run(problem)
        logger.debug("\n{}", fmt_analysis(analysis, problem, stage="[After Stage 2b] EntityAnalysis (post-fold)"))

        # Stage 3: Infer max sizes (LP)
        logger.info("[Stage 3] MaxSizeInference")
        max_sizes = MaxSizeInference().run(problem, analysis)
        logger.debug("[Stage 3] inferred max_sizes={}",
                     {getattr(ref, 'id', repr(ref)): sz for ref, sz in max_sizes.items()})

        # Merge max sizes into analysis
        self._merge_max_sizes(analysis, max_sizes)
        logger.debug("[Stage 3] merged max_sizes into analysis")
        logger.debug("\n{}", fmt_analysis(analysis, problem, stage="[After Stage 3] MaxSizeInference (sizes merged)"))

        # Stage 4: Lowering
        logger.info("[Stage 4] LoweringPass — {} objects before", len(problem.defs))
        problem = LoweringPass().run(problem, analysis)
        logger.info("[Stage 4] LoweringPass done — {} objects after", len(problem.defs))
        logger.debug("\n{}", fmt_problem(problem, stage="[After Stage 4] LoweringPass"))

        # Re-analyze after lowering
        logger.debug("[Stage 4b] Re-running EntityAnalysis after lowering")
        analysis = EntityAnalysis().run(problem)
        logger.debug("\n{}", fmt_analysis(analysis, problem, stage="[After Stage 4b] EntityAnalysis (post-lowering)"))

        # Stage 5: Simplify
        logger.info("[Stage 5] SimplifyPass — {} objects before", len(problem.defs))
        problem = SimplifyPass().run(problem)
        logger.info("[Stage 5] SimplifyPass done — {} objects after", len(problem.defs))
        logger.debug("\n{}", fmt_problem(problem, stage="[After Stage 5] SimplifyPass → backend input"))

        # Stage 6: Bag classification
        logger.info("[Stage 6] BagClassification")
        analysis = BagClassification().run(problem, analysis)
        logger.debug("[Stage 6] dis_entities={}",
                     {ref.id: {e.name for e in info.dis_entities}
                      for ref, info in analysis.bag_info.items()})
        logger.debug("\n{}", fmt_analysis(analysis, problem, stage="[After Stage 6] BagClassification (dis/indis)"))

        # Stage 7: Solve via IR-native backend
        logger.info("[Stage 7] Solving via WFOMCBackend")
        result = self._solve_ir(problem, analysis)
        logger.info("[Stage 7] result={}", result)
        return result

    def _merge_max_sizes(
        self, analysis: AnalysisResult, max_sizes: dict
    ) -> None:
        """Merge inferred max sizes into analysis result."""
        logger.debug("_merge_max_sizes: updating {} refs", len(max_sizes))
        for ref, size in max_sizes.items():
            if ref in analysis.set_info:
                current = analysis.set_info[ref].max_size
                analysis.set_info[ref].max_size = min(current, size)
                logger.debug("  ref={} (set): old={} -> new={}", ref.id, current, min(current, size))
            elif ref in analysis.bag_info:
                current = analysis.bag_info[ref].max_size
                analysis.bag_info[ref].max_size = min(current, size)
                logger.debug("  ref={} (bag): old={} -> new={}", ref.id, current, min(current, size))

    def _solve_ir(self, problem: "Problem", analysis: "AnalysisResult") -> int:
        """Solve using the IR-native WFOMC backend.

        Handles OrConstraint via inclusion-exclusion before dispatching to
        the backend.

        Args:
            problem: The fully-lowered, simplified IR Problem.
            analysis: The AnalysisResult from BagClassification (final stage).

        Returns:
            The integer count of solutions.
        """
        logger.debug("_solve_ir: {} constraints", len(problem.constraints))

        # Find the first OrConstraint and expand it via inclusion-exclusion:
        # |A ∪ B| = |A| + |B| - |A ∩ B|
        for i, c in enumerate(problem.constraints):
            if isinstance(c, OrConstraint):
                logger.info("Expanding OrConstraint[{}] via inclusion-exclusion", i)
                logger.debug("  left={}  right={}", c.left, c.right)
                other = [x for j, x in enumerate(problem.constraints) if j != i]
                prob_a = replace(problem, constraints=tuple(other + [c.left]))
                prob_b = replace(problem, constraints=tuple(other + [c.right]))
                prob_ab = replace(problem, constraints=tuple(other + [c.left, c.right]))
                count_a = self._solve_ir(prob_a, analysis)
                count_b = self._solve_ir(prob_b, analysis)
                count_ab = self._solve_ir(prob_ab, analysis)
                logger.debug("  |A|={}  |B|={}  |A∩B|={}  result={}",
                             count_a, count_b, count_ab, count_a + count_b - count_ab)
                return count_a + count_b - count_ab

        backend = WFOMCBackend(lifted=False)
        return backend.solve(problem, analysis)
