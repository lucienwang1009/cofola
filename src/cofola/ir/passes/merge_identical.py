"""Pass for merging identical object definitions (CSE).

This module provides MergeIdenticalObjects, which finds and merges
structurally identical derived/constant objects. Only types whose identity
is fully determined by their field values are eligible; "random variable"
types such as SetChoose, BagChoose, FuncDef, TupleDef, SequenceDef, and
PartitionDef are excluded because two refs with the same definition represent
independent random draws and must not be conflated.
"""

from __future__ import annotations

from loguru import logger

from cofola.frontend.objects import (
    ObjDef,
    ObjRef,
    SetInit,
    BagInit,
    SetUnion,
    SetIntersection,
    SetDifference,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    FuncImage,
    FuncInverseImage,
    FuncInverse,
    PartRef,
)
from cofola.ir.pass_manager import TransformPass
from cofola.frontend.problem import Problem


# ObjDef types for which structural equality implies semantic equality.
# Excluded: SetChoose, SetChooseReplace, BagChoose, FuncDef, TupleDef,
#           SequenceDef, PartitionDef — each occurrence is an independent
#           random variable even when the definition fields are identical.
MERGEABLE_TYPES = (
    SetInit,
    BagInit,
    SetUnion,
    SetIntersection,
    SetDifference,
    BagUnion,
    BagAdditiveUnion,
    BagIntersection,
    BagDifference,
    BagSupport,
    FuncImage,
    FuncInverseImage,
    FuncInverse,
    PartRef,
)


class MergeIdenticalObjects(TransformPass):
    """Merge structurally identical object definitions (CSE).

    Scans problem.defs for objects whose type is in MERGEABLE_TYPES and
    whose field values are equal. For each group of duplicates, keeps the
    ref with the smallest id (for determinism) and substitutes all others
    throughout the problem.

    This handles both leaf constants (SetInit, BagInit) and derived objects
    produced by LoweringPass (FuncImage, FuncInverseImage, BagSupport, etc.).
    """

    required_analyses: list[type] = []

    def run(self, problem: Problem, am=None) -> Problem:
        # Map ObjDef value → canonical ObjRef (smallest id seen so far)
        canonical: dict[ObjDef, ObjRef] = {}
        substitutions: dict[ObjRef, ObjRef] = {}

        for ref, defn in problem.iter_objects():
            if not isinstance(defn, MERGEABLE_TYPES):
                continue
            if defn in canonical:
                existing = canonical[defn]
                if ref.id < existing.id:
                    substitutions[existing] = ref
                    canonical[defn] = ref
                else:
                    substitutions[ref] = existing
            else:
                canonical[defn] = ref

        if not substitutions:
            logger.debug("MergeIdenticalObjects: nothing to merge")
            return problem

        logger.info(
            "MergeIdenticalObjects: merging {} duplicate defs",
            len(substitutions),
        )
        for old, new in substitutions.items():
            logger.debug("  #{} -> #{}", old.id, new.id)

        current = problem
        for old_ref, new_ref in substitutions.items():
            current = current.substitute(old_ref, new_ref)
        return current
