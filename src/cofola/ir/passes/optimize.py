"""Constant folding for the immutable IR.

This module implements ConstantFolder, which folds constant expressions
like SetUnion(SetInit, SetInit) -> SetInit.

Ports the legacy fold_constants function to work with the new IR.
"""

from __future__ import annotations

from dataclasses import fields

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
)
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import EntityAnalysis


class ConstantFolder:
    """Folds constant sub-expressions in the IR.

    This pass transforms constant expressions into simpler forms:
    - SetUnion(SetInit, SetInit) -> SetInit (union of entities)
    - SetIntersection(SetInit, SetInit) -> SetInit (intersection)
    - SetDifference(SetInit, SetInit) -> SetInit (difference)
    - BagSupport(BagInit) -> SetInit (support)
    - BagAdditiveUnion(BagInit, BagInit) -> BagInit
    - BagIntersection(BagInit, BagInit) -> BagInit
    - BagDifference(BagInit, BagInit) -> BagInit

    Ports the legacy fold_constants function to work with the new IR.
    """

    def rewrite(self, problem: Problem) -> Problem:
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

        # Fold size constraints
        new_constraints, constraint_changed = self._fold_constraints(problem)

        return Problem(
            defs=tuple(new_defs),
            constraints=tuple(new_constraints),
            names=problem.names,
        ), changed or constraint_changed

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

    def _fold_constraints(
        self, problem: Problem
    ) -> tuple[list, bool]:
        """Fold constants in constraints.

        For SizeConstraints, if objects have known sizes, substitute them.

        Args:
            problem: The Problem to fold.

        Returns:
            Tuple of (new constraints, whether any changes were made).
        """
        # Run entity analysis to get sizes
        analysis = EntityAnalysis().run(problem)

        new_constraints = []
        changed = False

        for c in problem.constraints:
            if isinstance(c, SizeConstraint):
                folded_terms = []
                rhs = c.rhs

                for ref, coef in c.terms:
                    # Get the object's size if known
                    if ref in analysis.set_info:
                        # We can only fold if the size is exactly determined
                        pass
                    folded_terms.append((ref, coef))

                if changed:
                    new_constraints.append(
                        SizeConstraint(
                            terms=tuple(folded_terms),
                            comparator=c.comparator,
                            rhs=rhs,
                        )
                    )
                else:
                    new_constraints.append(c)
            else:
                new_constraints.append(c)

        return new_constraints, changed