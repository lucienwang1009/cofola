"""Constant folding for the immutable IR.

This module implements ConstantFolder, which folds constant expressions
like SetUnion(SetInit, SetInit) -> SetInit.

Ports the legacy fold_constants function to work with the new IR.
"""

from __future__ import annotations

from dataclasses import fields, replace as dc_replace

from loguru import logger

from cofola.frontend.types import ObjRef, Entity
from cofola.frontend.objects import (
    ObjDef,
    SetInit,
    SetUnion,
    SetIntersection,
    SetDifference,
    BagInit,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    SetChoose,
    BagChoose,
    SetChooseReplace,
)
from cofola.ir.pass_manager import TransformPass
from cofola.frontend.problem import Problem
from cofola.ir.analysis.merged import MergedAnalysis


class UnsatisfiableConstraint(Exception):
    """Raised when a SizeConstraint folds to a numerically false statement."""


def _eval_comparator(comp: str, lhs: int, rhs: int) -> bool:
    """Evaluate a numeric comparator expression."""
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<":  lhs <  rhs,
        "<=": lhs <= rhs,
        ">":  lhs >  rhs,
        ">=": lhs >= rhs,
    }[comp]



class ConstantFolder(TransformPass):
    """Folds constant sub-expressions in the IR.

    This pass transforms constant expressions into simpler forms:
    - SetUnion(SetInit, SetInit) -> SetInit (union of entities)
    - SetIntersection(SetInit, SetInit) -> SetInit (intersection)
    - SetDifference(SetInit, SetInit) -> SetInit (difference)
    - BagSupport(BagInit) -> SetInit (support)
    - BagAdditiveUnion(BagInit, BagInit) -> BagInit
    - BagIntersection(BagInit, BagInit) -> BagInit
    - BagDifference(BagInit, BagInit) -> BagInit
    """

    required_analyses: list[type] = []

    def run(self, problem: Problem, am=None) -> Problem:
        """Fold constants in a Problem.

        Runs folding to a fixed point.

        Args:
            problem: The Problem to optimize.

        Returns:
            A new Problem with constants folded.
        """
        current = problem
        changed = True
        iteration = 0

        while changed:
            iteration += 1
            current, changed = self._fold_once(current)
            logger.debug("ConstantFolder iteration {}: changed={}", iteration, changed)

        logger.info("ConstantFolder: converged after {} iterations", iteration)
        return current

    def _fold_once(self, problem: Problem) -> tuple[Problem, bool]:
        """Run one round of constant folding.

        Args:
            problem: The Problem to fold.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        new_defs: list[tuple[ObjRef, ObjDef]] = []
        changed = False

        for ref, defn in problem.iter_objects():
            folded = self._try_fold(ref, defn, problem)

            if folded is not None:
                new_defs.append((ref, folded))
                changed = True
                logger.info(f"Folded {defn} to {folded}")
            else:
                new_defs.append((ref, defn))

        return Problem(
            defs=tuple(new_defs),
            constraints=problem.constraints,
            names=problem.names,
        ), changed

    def _try_fold(
        self, ref: ObjRef, defn: ObjDef, problem: Problem
    ) -> ObjDef | None:
        """Try to fold a single object definition.

        Args:
            ref: Reference to the object.
            defn: The object definition.
            problem: The full Problem context.

        Returns:
            A folded ObjDef, or None if not foldable.
        """
        # SetUnion of two SetInits
        if isinstance(defn, SetUnion):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, SetInit) and isinstance(right_defn, SetInit):
                return SetInit(
                    entities=left_defn.entities | right_defn.entities
                )

        # SetIntersection of two SetInits
        elif isinstance(defn, SetIntersection):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, SetInit) and isinstance(right_defn, SetInit):
                return SetInit(
                    entities=left_defn.entities & right_defn.entities
                )

        # SetDifference of two SetInits
        elif isinstance(defn, SetDifference):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, SetInit) and isinstance(right_defn, SetInit):
                return SetInit(
                    entities=left_defn.entities - right_defn.entities
                )

        # BagSupport of BagInit
        elif isinstance(defn, BagSupport):
            src_defn = problem.get_object(defn.source)

            if isinstance(src_defn, BagInit):
                entities = frozenset(e for e, _ in src_defn.entity_multiplicity)
                return SetInit(entities=entities)

        # BagAdditiveUnion of two BagInits
        elif isinstance(defn, BagAdditiveUnion):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, BagInit) and isinstance(right_defn, BagInit):
                merged = self._merge_bag_mults(
                    left_defn.entity_multiplicity,
                    right_defn.entity_multiplicity,
                    operation="add",
                )
                return BagInit(entity_multiplicity=merged)

        # BagUnion of two BagInits (max multiplicity)
        elif isinstance(defn, BagUnion):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, BagInit) and isinstance(right_defn, BagInit):
                merged = self._merge_bag_mults(
                    left_defn.entity_multiplicity,
                    right_defn.entity_multiplicity,
                    operation="max",
                )
                return BagInit(entity_multiplicity=merged)

        # BagIntersection of two BagInits (min multiplicity)
        elif isinstance(defn, BagIntersection):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, BagInit) and isinstance(right_defn, BagInit):
                merged = self._merge_bag_mults(
                    left_defn.entity_multiplicity,
                    right_defn.entity_multiplicity,
                    operation="min",
                )
                return BagInit(entity_multiplicity=merged)

        # BagDifference of two BagInits
        elif isinstance(defn, BagDifference):
            left_defn = problem.get_object(defn.left)
            right_defn = problem.get_object(defn.right)

            if isinstance(left_defn, BagInit) and isinstance(right_defn, BagInit):
                merged = self._merge_bag_mults(
                    left_defn.entity_multiplicity,
                    right_defn.entity_multiplicity,
                    operation="sub",
                )
                return BagInit(entity_multiplicity=merged)

        return None

    def _merge_bag_mults(
        self,
        left: tuple[tuple[Entity, int], ...],
        right: tuple[tuple[Entity, int], ...],
        operation: str,
    ) -> tuple[tuple[Entity, int], ...]:
        """Merge two bag multiplicities.

        Args:
            left: Left bag's entity multiplicities.
            right: Right bag's entity multiplicities.
            operation: One of "add", "max", "min", "sub".

        Returns:
            Merged entity multiplicities as a tuple.
        """
        left_dict = dict(left)
        right_dict = dict(right)

        all_entities = set(left_dict.keys()) | set(right_dict.keys())
        result: dict[Entity, int] = {}

        for entity in all_entities:
            l_val = left_dict.get(entity, 0)
            r_val = right_dict.get(entity, 0)

            if operation == "add":
                result[entity] = l_val + r_val
            elif operation == "max":
                if entity in left_dict and entity in right_dict:
                    result[entity] = max(l_val, r_val)
                elif entity in left_dict:
                    result[entity] = l_val
                else:
                    result[entity] = r_val
            elif operation == "min":
                if entity in left_dict and entity in right_dict:
                    val = min(l_val, r_val)
                    if val > 0:
                        result[entity] = val
            elif operation == "sub":
                val = l_val - r_val
                if val > 0:
                    result[entity] = val

        # Sort by entity name for determinism
        return tuple(sorted(result.items(), key=lambda x: x[0].name))


class SizeConstraintFolder(TransformPass):
    """Substitutes known exact sizes into SizeConstraints.

    For each SizeConstraint term whose ref has a known exact_size (from
    MergedAnalysis / LP inference), substitute the integer value directly
    and remove the term from the constraint. This prevents the WFOMC encoder
    from creating unnecessary symbolic variables and polynomial weights for
    objects whose sizes are already fully determined.

    Example::

        |A| + |B| == 5,  exact_size(A) == 3  →  |B| == 2

    After this pass, A no longer appears in any SizeConstraint, so the
    encoder never calls get_obj_var(A) and never registers a WFOMC weight
    for it.
    """

    required_analyses = [MergedAnalysis]

    def run(self, problem: Problem, am=None) -> Problem:
        """Substitute known exact sizes into SizeConstraints.

        When all terms in a SizeConstraint are substituted:
        - If the resulting numerical comparison is True (e.g. 0 == 0): drop the
          constraint and embed the exact size directly into any SetChoose /
          BagChoose / SetChooseReplace defs whose size was None. This preserves
          the size information the encoder needs without keeping a redundant
          SizeConstraint.
        - If the resulting comparison is False (e.g. 0 == 3): raise
          UnsatisfiableConstraint, which the pipeline catches to return 0.

        Args:
            problem: The Problem to fold.
            am: AnalysisManager for accessing MergedAnalysis.

        Returns:
            A new Problem with size-known terms substituted, or the original
            Problem unchanged if nothing could be folded.

        Raises:
            UnsatisfiableConstraint: if a fully-substituted constraint evaluates
                to False.
        """
        from cofola.frontend.constraints import SizeConstraint

        analysis = am.get(MergedAnalysis)
        new_constraints = []
        # Track def updates: ref → new ObjDef (only for choose objects)
        updated_defs: dict[ObjRef, object] = {}
        changed = False

        for c in problem.constraints:
            if not isinstance(c, SizeConstraint):
                new_constraints.append(c)
                continue

            remaining_terms: list[tuple] = []
            rhs = c.rhs
            folded_any = False
            # Track which ObjRefs were folded (for def-embedding)
            folded_obj_refs: list[tuple[ObjRef, int]] = []  # (ref, exact_size)

            for term, coef in c.terms:
                if not isinstance(term, ObjRef):
                    remaining_terms.append((term, coef))
                    continue
                info = analysis.set_info.get(term) or analysis.bag_info.get(term)
                if info is not None and info.exact_size is not None:
                    # Substitute: coef * exact_size is a known constant, move to RHS
                    rhs = rhs - coef * info.exact_size
                    folded_any = True
                    folded_obj_refs.append((term, info.exact_size))
                    logger.debug(
                        "SizeConstraintFolder: folded ref={} exact_size={} coef={} → rhs adjusted to {}",
                        term.id, info.exact_size, coef, rhs,
                    )
                else:
                    remaining_terms.append((term, coef))

            if not folded_any:
                new_constraints.append(c)
                continue

            if not remaining_terms:
                # All terms substituted: evaluate the purely numerical result.
                if _eval_comparator(c.comparator, 0, rhs):
                    # Trivially true: try to drop the constraint.
                    # We can only safely drop if every folded ObjRef term is a
                    # SetChoose / BagChoose / SetChooseReplace whose size field is
                    # None — in that case we embed the exact size into the def so
                    # the encoder can still constrain the WFOMC weight via defn.size
                    # without needing a SizeConstraint.
                    embeddable = all(
                        isinstance(problem.get_object(ref), (SetChoose, BagChoose, SetChooseReplace))
                        and problem.get_object(ref).size is None
                        for ref, _ in folded_obj_refs
                    )
                    if embeddable:
                        for ref, exact_size in folded_obj_refs:
                            defn = problem.get_object(ref)
                            updated_defs[ref] = dc_replace(defn, size=exact_size)
                            logger.debug(
                                "SizeConstraintFolder: embedded size={} into {} ref={}",
                                exact_size, type(defn).__name__, ref.id,
                            )
                        logger.debug(
                            "SizeConstraintFolder: dropped trivially-true constraint {}",
                            c,
                        )
                        changed = True
                        # Don't append c — it is dropped
                    else:
                        # Some terms are not embeddable (e.g. PartRef); keep
                        # original constraint to preserve encoder correctness.
                        new_constraints.append(c)
                        # Do not set changed=True: this constraint is unchanged.
                else:
                    # Trivially false: problem is unsatisfiable
                    raise UnsatisfiableConstraint(
                        f"SizeConstraint folded to 0 {c.comparator} {rhs}: unsatisfiable"
                    )
            else:
                new_constraints.append(SizeConstraint(
                    terms=tuple(remaining_terms),
                    comparator=c.comparator,
                    rhs=rhs,
                ))
                changed = True

        if not changed:
            return problem

        # Rebuild defs, applying any size embeddings
        new_defs = tuple(
            (ref, updated_defs.get(ref, defn))
            for ref, defn in problem.defs
        )
        return Problem(
            defs=new_defs,
            constraints=tuple(new_constraints),
            names=problem.names,
        )
