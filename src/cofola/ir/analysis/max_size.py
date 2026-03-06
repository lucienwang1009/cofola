"""Max size inference for the immutable IR.

This module implements MaxSizeInference, which uses linear programming
to infer tighter max_size bounds from size constraints.

Ports the legacy InferMaxSizePass to work with the new IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from cofola.ir.pass_manager import AnalysisPass
from cofola.frontend.types import ObjRef
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import EntityAnalysis, AnalysisResult
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

        # Collect constrained objects and constraints
        constrained_refs: list[ObjRef] = []
        size_constraints: list[SizeConstraint] = []

        for c in problem.constraints:
            if isinstance(c, SizeConstraint):
                size_constraints.append(c)
                for ref, _ in c.terms:
                    if ref not in constrained_refs:
                        constrained_refs.append(ref)

        logger.debug("MaxSizeInference: {} size_constraints, {} constrained_refs",
                     len(size_constraints), len(constrained_refs))

        if not size_constraints:
            logger.debug("MaxSizeInference: no size constraints, skipping LP")
            return result

        # Build LP matrices
        n_constraints = len(size_constraints)
        n_vars = len(constrained_refs)

        A_u = np.zeros((n_constraints, n_vars))  # Upper bound constraints
        b_u = np.zeros(n_constraints)  # Upper bound RHS
        A_e = np.zeros((n_constraints, n_vars))  # Equality constraints
        b_e = np.zeros(n_constraints)  # Equality RHS

        for i, c in enumerate(size_constraints):
            comp = c.comparator

            # Set up RHS based on comparator
            if comp == "==":
                b_e[i] = c.rhs
            elif comp == "<=":
                b_u[i] = c.rhs
            elif comp == ">=":
                b_u[i] = -c.rhs
            elif comp == "<":
                b_u[i] = c.rhs - 1
            elif comp == ">":
                b_u[i] = -c.rhs + 1
            else:
                continue

            # Set up coefficients
            for ref, coef in c.terms:
                index = constrained_refs.index(ref)
                if comp == "==":
                    A_e[i, index] = coef
                elif comp in ("<=", "<"):
                    A_u[i, index] = coef
                else:  # >=, >
                    A_u[i, index] = -coef

        # Solve LP for each object to find max and min size
        for ref in constrained_refs:
            index = constrained_refs.index(ref)
            ref_id = getattr(ref, 'id', repr(ref))

            # Maximise
            c_max = np.zeros(n_vars)
            c_max[index] = -1
            ret_max = linprog(c_max, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq=b_e)

            if not ret_max.success:
                logger.debug("  LP ref={}: infeasible or unbounded (max)", ref_id)
                # Infeasible LP means the constraints are contradictory
                if ret_max.status == 2:  # infeasible
                    result.unsatisfiable = True
                continue

            max_val = int(ret_max.x[index])

            # Minimise
            c_min = np.zeros(n_vars)
            c_min[index] = 1
            ret_min = linprog(c_min, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq=b_e)

            if not ret_min.success:
                logger.debug("  LP ref={}: infeasible or unbounded (min)", ref_id)
                if ret_min.status == 2:
                    result.unsatisfiable = True
                continue

            min_val = int(ret_min.x[index])

            logger.debug("  LP ref={}: min={} max={}", ref_id, min_val, max_val)

            # Update max_sizes if tighter than initial bound
            initial_max = self._get_initial_max(ref, analysis)
            if initial_max is None or max_val < initial_max:
                result.max_sizes[ref] = max_val

            # If min == max, we have an exact size
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

    def _get_initial_max(self, ref: ObjRef, analysis: AnalysisResult) -> int | None:
        """Get the initial max_size for a reference from analysis results."""
        if ref in analysis.set_info:
            return analysis.set_info[ref].max_size
        if ref in analysis.bag_info:
            return analysis.bag_info[ref].max_size
        return None
