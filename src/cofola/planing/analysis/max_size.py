"""Max size inference for the planning layer.

This module implements MaxSizeInference, which uses linear programming
to infer tighter max_size bounds from raw object-cardinality constraints.

Size constraints containing derived size atoms such as BagCountAtom,
TupleCountAtom, or SeqPatternCountAtom are deliberately skipped here. Those
terms are lowered or handled elsewhere; treating them as LP variables would
produce bounds that MergedAnalysis cannot safely merge back into object info.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from cofola.planing.pass_manager import AnalysisPass
from cofola.frontend.objects import ObjRef
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import EntityAnalysis, AnalysisResult
from loguru import logger


@dataclass
class SizeInferenceResult:
    """Result of MaxSizeInference.

    Attributes:
        max_sizes: Tighter upper bounds inferred by LP (existing behaviour).
        exact_sizes: Exact sizes where LP min == LP max (newly inferred).
        unsatisfiable: True if size constraints are contradictory or conflict
                       with EntityAnalysis exact_size values.
    """

    max_sizes: dict[ObjRef, int] = field(default_factory=dict)
    exact_sizes: dict[ObjRef, int] = field(default_factory=dict)
    unsatisfiable: bool = False


@dataclass(frozen=True)
class _LPProblem:
    """LP matrices and variable bounds for raw object-cardinality constraints."""

    refs: list[ObjRef]
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    A_eq: np.ndarray | None
    b_eq: np.ndarray | None
    bounds: list[tuple[int, int | None]]


class MaxSizeInference(AnalysisPass):
    """Infers maximum sizes for objects from size constraints via LP.

    This pass examines SizeConstraints in the problem and uses linear
    programming to find tighter bounds on object sizes.

    Ports the legacy infer_max_size function to work with the new IR.
    """

    required_analyses = [EntityAnalysis]

    def run(self, problem: Problem, am=None) -> SizeInferenceResult:
        """Run max size inference on a Problem.

        Args:
            problem: The Problem to analyze.
            am: AnalysisManager; used to retrieve EntityAnalysis result.

        Returns:
            SizeInferenceResult with inferred max_sizes, exact_sizes, and
            unsatisfiable flag.
        """
        analysis: AnalysisResult = am.get(EntityAnalysis)

        result = SizeInferenceResult()

        size_constraints, constrained_refs, skipped_constraints = (
            self._collect_lp_constraints(problem)
        )

        logger.debug(
            "MaxSizeInference: {} LP constraints, {} constrained_refs, {} skipped",
            len(size_constraints), len(constrained_refs), skipped_constraints,
        )

        if not size_constraints or not constrained_refs:
            logger.debug("MaxSizeInference: no LP-compatible size constraints, skipping LP")
            return result

        lp_problem = self._build_lp_problem(size_constraints, constrained_refs, analysis)
        if self._is_infeasible(lp_problem):
            result.unsatisfiable = True
            return result

        # Solve LP for each object to find max and min size.
        for index, ref in enumerate(lp_problem.refs):
            min_val, max_val, unsat = self._solve_ref_bounds(lp_problem, index)
            if unsat:
                result.unsatisfiable = True
                continue
            if min_val is None or max_val is None:
                continue

            logger.debug("  LP ref={}: min={} max={}", ref.id, min_val, max_val)

            initial_max = self._get_initial_max(ref, analysis)
            if initial_max is None or max_val < initial_max:
                result.max_sizes[ref] = max_val

            if min_val == max_val:
                result.exact_sizes[ref] = max_val

        # Conflict detection: LP exact_size vs EntityAnalysis exact_size
        for ref, lp_exact in result.exact_sizes.items():
            info = analysis.set_info.get(ref) or analysis.bag_info.get(ref)
            if info is not None and info.exact_size is not None:
                if info.exact_size != lp_exact:
                    logger.info(
                        "MaxSizeInference: conflict on ref={}: "
                        "EntityAnalysis.exact_size={} vs LP exact_size={}",
                        getattr(ref, 'id', repr(ref)), info.exact_size, lp_exact,
                    )
                    result.unsatisfiable = True

        logger.info(
            "MaxSizeInference: max_sizes={}, exact_sizes={}, unsatisfiable={}",
            len(result.max_sizes), len(result.exact_sizes), result.unsatisfiable,
        )
        return result

    def _collect_lp_constraints(
        self,
        problem: Problem,
    ) -> tuple[list[SizeConstraint], list[ObjRef], int]:
        """Collect SizeConstraints whose terms are raw ObjRefs."""

        constrained_refs: list[ObjRef] = []
        size_constraints: list[SizeConstraint] = []
        skipped_constraints = 0

        for c in problem.constraints:
            if not isinstance(c, SizeConstraint):
                continue
            if c.comparator == "!=":
                skipped_constraints += 1
                logger.debug("MaxSizeInference: skipping != size constraint {}", c)
                continue
            if not all(isinstance(term, ObjRef) for term, _ in c.terms):
                skipped_constraints += 1
                logger.debug("MaxSizeInference: skipping size-atom constraint {}", c)
                continue
            size_constraints.append(c)
            for ref, _ in c.terms:
                if ref not in constrained_refs:
                    constrained_refs.append(ref)

        return size_constraints, constrained_refs, skipped_constraints

    def _build_lp_problem(
        self,
        size_constraints: list[SizeConstraint],
        refs: list[ObjRef],
        analysis: AnalysisResult,
    ) -> _LPProblem:
        """Build LP rows plus EntityAnalysis-derived bounds."""

        ref_index = {ref: i for i, ref in enumerate(refs)}
        n_vars = len(refs)
        upper_rows: list[list[float]] = []
        upper_rhs: list[float] = []
        equal_rows: list[list[float]] = []
        equal_rhs: list[float] = []

        for c in size_constraints:
            comp = c.comparator
            row = [0.0] * n_vars
            for ref, coef in c.terms:
                row[ref_index[ref]] = float(coef)

            if comp == "==":
                equal_rows.append(row)
                equal_rhs.append(float(c.rhs))
            elif comp == "<=":
                upper_rows.append(row)
                upper_rhs.append(float(c.rhs))
            elif comp == ">=":
                upper_rows.append([-coef for coef in row])
                upper_rhs.append(float(-c.rhs))
            elif comp == "<":
                upper_rows.append(row)
                upper_rhs.append(float(c.rhs - 1))
            elif comp == ">":
                upper_rows.append([-coef for coef in row])
                upper_rhs.append(float(-c.rhs + 1))

        bounds = [
            (0, self._get_initial_max(ref, analysis))
            for ref in refs
        ]
        return _LPProblem(
            refs=refs,
            A_ub=np.array(upper_rows) if upper_rows else None,
            b_ub=np.array(upper_rhs) if upper_rhs else None,
            A_eq=np.array(equal_rows) if equal_rows else None,
            b_eq=np.array(equal_rhs) if equal_rhs else None,
            bounds=bounds,
        )

    def _linprog(self, lp_problem: _LPProblem, objective: np.ndarray) -> Any:
        return linprog(
            objective,
            A_ub=lp_problem.A_ub,
            b_ub=lp_problem.b_ub,
            A_eq=lp_problem.A_eq,
            b_eq=lp_problem.b_eq,
            bounds=lp_problem.bounds,
        )

    def _is_infeasible(self, lp_problem: _LPProblem) -> bool:
        """Check global feasibility once before per-variable optimization."""

        ret = self._linprog(lp_problem, np.zeros(len(lp_problem.refs)))
        if ret.status == 2:
            logger.info("MaxSizeInference: LP constraints are infeasible")
            return True
        return False

    def _solve_ref_bounds(
        self,
        lp_problem: _LPProblem,
        index: int,
    ) -> tuple[int | None, int | None, bool]:
        """Return integer min/max bounds for one LP variable."""

        n_vars = len(lp_problem.refs)
        ref = lp_problem.refs[index]
        c_min = np.zeros(n_vars)
        c_min[index] = 1
        ret_min = self._linprog(lp_problem, c_min)
        if not ret_min.success:
            logger.debug("  LP ref={}: infeasible or unbounded (min)", ref.id)
            return None, None, ret_min.status == 2

        c_max = np.zeros(n_vars)
        c_max[index] = -1
        ret_max = self._linprog(lp_problem, c_max)
        if not ret_max.success:
            logger.debug("  LP ref={}: infeasible or unbounded (max)", ref.id)
            return None, None, ret_max.status == 2

        min_val = int(np.ceil(ret_min.x[index] - 1e-9))
        max_val = int(np.floor(ret_max.x[index] + 1e-9))
        return min_val, max_val, False

    def _get_initial_max(self, ref: ObjRef, analysis: AnalysisResult) -> int | None:
        """Get the initial max_size for a reference from analysis results."""
        if ref in analysis.set_info:
            return analysis.set_info[ref].max_size
        if ref in analysis.bag_info:
            return analysis.bag_info[ref].max_size
        return None
