"""Max size inference for the immutable IR.

This module implements MaxSizeInference, which uses linear programming
to infer tighter max_size bounds from size constraints.

Ports the legacy InferMaxSizePass to work with the new IR.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from cofola.frontend.types import ObjRef
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult
from loguru import logger


class MaxSizeInference:
    """Infers maximum sizes for objects from size constraints via LP.

    This pass examines SizeConstraints in the problem and uses linear
    programming to find tighter bounds on object sizes.

    Ports the legacy infer_max_size function to work with the new IR.
    """

    def run(self, problem: Problem, analysis: AnalysisResult) -> dict[ObjRef, int]:
        """Run max size inference on a Problem.

        Args:
            problem: The Problem to analyze.
            analysis: The entity analysis result (provides initial bounds).

        Returns:
            Dictionary mapping ObjRef to inferred max_size.
        """
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
            return {}

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

        # Solve LP for each object to find max size
        inferred_sizes: dict[ObjRef, int] = {}

        for ref in constrained_refs:
            index = constrained_refs.index(ref)
            c = np.zeros(n_vars)
            c[index] = -1  # Maximize this variable

            ret = linprog(c, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq=b_e)

            ref_id = getattr(ref, 'id', repr(ref))
            if ret.success:
                size = int(ret.x[index])
                # Only update if this is tighter than the initial bound
                initial_max = self._get_initial_max(ref, analysis)
                if initial_max is None or size < initial_max:
                    inferred_sizes[ref] = size
                    logger.debug("  LP ref={}: max_size={} (initial_max={})",
                                 ref_id, size, initial_max)
            else:
                logger.debug("  LP ref={}: infeasible or unbounded", ref_id)

        logger.info("MaxSizeInference: inferred sizes for {} refs", len(inferred_sizes))
        return inferred_sizes

    def _get_initial_max(self, ref: ObjRef, analysis: AnalysisResult) -> int | None:
        """Get the initial max_size for a reference from analysis results."""
        if ref in analysis.set_info:
            return analysis.set_info[ref].max_size
        if ref in analysis.bag_info:
            return analysis.bag_info[ref].max_size
        return None