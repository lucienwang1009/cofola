"""IR-native WFOMC encoder.

This module replaces encoder.py. Instead of accepting a legacy CofolaProblem
with mutable object instances, it accepts an immutable ir.Problem together with
an ir.AnalysisResult and encodes them directly to a WFOMCProblem.

OVERALL STRATEGY
================
The encoder iterates objects in topological order and dispatches on the IR
dataclass type using match-case. For each object it calls the corresponding
private helper (e.g. _encode_set_init, _encode_bag_init, etc.) which writes
predicates/sentences/weightings into the ContextIR.

After all objects, constraints are encoded.
Finally context.build() is called to produce (WFOMCProblem, Decoder).

ANALYSIS DATA ACCESS
====================
All derived data (dis_entities, p_entities_multiplicity, indis_entities, max_size,
p_entities) is read from the AnalysisResult, NOT from object attributes:

    bag_info  = analysis.bag_info[ref]     # → BagInfo
    set_info  = analysis.set_info[ref]     # → SetInfo

BagInfo fields:
    .dis_entities: set[IREntity]
    .p_entities_multiplicity: dict[IREntity, int]
    .indis_entities: dict[int, set[IREntity]]   # multiplicity → set of entities
    .max_size: int

SetInfo fields:
    .p_entities: set[IREntity]
    .max_size: int

NAME RESOLUTION
===============
    name = problem.get_name(ref) or f"obj_{ref.id}"

MAPPING FROM LEGACY ENCODER
============================
Each section below documents which function in encoder.py it replaces.
Cross-reference by searching the legacy file for the corresponding object class.

IR TYPE IMPORTS (for match-case)
=================================
    from cofola.frontend import objects as ir_obj, constraints as ir_cst

WFOMC IMPORTS
=============
    from wfomc import (WFOMCProblem, X, Const, QuantifiedFormula, Universal,
                       top, MultinomialCoefficients, Pred,
                       fol_parse as parse, exclusive, exactly_one_qf, Algo)
"""
from __future__ import annotations

import math
from functools import reduce
from sympy import Eq, Ge, Gt, Le, Lt, Min
from wfomc import (
    Algo,
    Const,
    Pred,
    QuantifiedFormula,
    Universal,
    WFOMCProblem,
    X,
    exactly_one_qf,
    exclusive,
    fol_parse as parse,
    top,
    MultinomialCoefficients,
)

import cofola.frontend.objects as ir_obj
import cofola.frontend.constraints as ir_cst
from cofola.backend.wfomc.context_ir import ContextIR
from cofola.backend.wfomc.decoder import Decoder
from cofola.utils import create_aux_pred, create_cofola_pred, ListLessThan
from cofola.frontend.problem import Problem
from cofola.frontend.types import ObjRef, Entity
from cofola.ir.analysis.entities import AnalysisResult, SetInfo, BagInfo
from loguru import logger


# =============================================================================
# Public entry point
# =============================================================================


def encode_ir(
    problem: Problem,
    analysis: AnalysisResult,
    lifted: bool = False,
) -> tuple[WFOMCProblem, Decoder]:
    """Encode an IR Problem + AnalysisResult to a (WFOMCProblem, Decoder).

    This is the top-level entry point called by WFOMCBackend.solve().

    Args:
        problem: The fully lowered, simplified immutable IR Problem.
                 At this point, TupleDef has been lowered to FuncDef, etc.
        analysis: The entity analysis result (includes bag classification).
        lifted: Whether to use lifted inference (passed to Decoder logic).

    Returns:
        Tuple of (WFOMCProblem, Decoder).

    IMPLEMENTATION:
        1. context = ContextIR(problem, analysis)
        2. _encode_entities(analysis, context)
        3. for ref in problem.topological_order():
               defn = problem.get_object(ref)
               if defn is None: continue
               _encode_object(ref, defn, problem, analysis, context)
        4. for c in problem.constraints:
               _encode_constraint(c, problem, analysis, context)
        5. return context.build()
    """
    logger.debug("encode_ir: {} objects to encode", len(list(problem.iter_objects())))
    context = ContextIR(problem, analysis)
    MultinomialCoefficients.setup(len(context.domain))

    logger.debug(f"IR encode: singletons={context.singletons}")

    # Encode singletons predicate (if any)
    if context.singletons:
        _encode_singleton(context)

    # Encode objects in topological order
    for ref in problem.topological_order():
        defn = problem.get_object(ref)
        if defn is None:
            continue
        logger.debug(f"Encoding {type(defn).__name__} ref={ref.id}")
        _encode_object(ref, defn, problem, analysis, context)

    logger.debug(
        "encode_ir: objects encoded — predicates={}, entity_preds={}, "
        "evidence={}, weighting={}, sentence_len={}",
        len(context.ref2pred),
        len(context.ref_entity2pred),
        len(context.unary_evidence),
        len(context.weighting),
        len(str(context.sentence)),
    )

    # Encode constraints
    logger.debug("encode_ir: encoding {} constraints", len(problem.constraints))
    for c in problem.constraints:
        _encode_constraint(c, problem, analysis, context)

    logger.debug(
        "encode_ir: constraints encoded — validator={}, gen_vars={}, overcount={}",
        len(context.validator),
        len(context.gen_vars),
        context.overcount,
    )

    result = context.build()
    logger.info("encode_ir complete: encoding done")
    return result


# =============================================================================
# Entity encoding
# =============================================================================


def _encode_entities(analysis: AnalysisResult, context: ContextIR) -> None:
    """Create one WFOMC constant for each entity in the problem.

    Entities that are singletons get special treatment: their constant
    is added as unary evidence for all set predicates.

    LEGACY EQUIVALENT: _encode_entity() loop in encoder.py (lines ~90–130)

    Args:
        analysis: The analysis result (provides all_entities, singletons).
        context: The encoding context (mutated in place).

    IMPLEMENTATION:
        For each entity in analysis.all_entities:
            entity_const = Const(entity.name)  # already in context.domain
            # Singleton handling is done per-object during object encoding,
            # because only then do we know which predicates the entity belongs to.
        # No action needed here — entities become domain constants automatically.
        # This function may encode singleton unary evidence if the strategy requires it.
        pass  # or handle singletons here
    """
    pass  # entities become domain constants via context.domain; singleton evidence added per-object


def _encode_singleton(context: ContextIR) -> None:
    """Create a predicate for the set of singleton entities."""
    singletons = context.singletons
    pred = create_aux_pred(1, "singletons")
    context.singletons_pred = pred
    for e in context.analysis.all_entities:
        if e in singletons:
            context.unary_evidence.add(pred(Const(e.name)))
        else:
            context.unary_evidence.add(~pred(Const(e.name)))


def _encode_entity_in_ctx(entity: IREntity, context: ContextIR) -> object:
    """Encode a single entity as a unary predicate with evidence.

    Returns the Pred. Idempotent — encoding the same entity twice is a no-op.
    """
    key = (None, entity)
    if key in context.ref_entity2pred:
        return context.ref_entity2pred[key]
    pred = create_cofola_pred(f"entity_{entity.name}", 1)
    context.ref_entity2pred[key] = pred
    # Add positive evidence for this entity
    context.unary_evidence.add(pred(Const(entity.name)))
    # Add negative evidence for all other entities
    for e in context.analysis.all_entities:
        if e != entity:
            context.unary_evidence.add(~pred(Const(e.name)))
    return pred


# =============================================================================
# Object encoding dispatcher
# =============================================================================


def _encode_object(
    ref: ObjRef,
    defn: object,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Dispatch to the appropriate object encoder based on IR node type.

    Args:
        ref: The ObjRef for this node.
        defn: The IR dataclass node (SetInit, BagInit, etc.).
        problem: The Problem (for lookups).
        analysis: The analysis result.
        context: The encoding context.

    IMPLEMENTATION: Use match-case on defn type, call the corresponding helper.
    Do NOT handle PartRef here — it is processed inside _encode_partition.
    """
    match defn:
        # ── Sets ──────────────────────────────────────────────────────────────
        case ir_obj.SetInit():
            _encode_set_init(ref, defn, analysis, context)

        case ir_obj.SetChoose():
            _encode_set_choose(ref, defn, analysis, context)

        case ir_obj.SetChooseReplace():
            _encode_set_choose_replace(ref, defn, analysis, context)

        case ir_obj.SetUnion():
            _encode_set_union(ref, defn, context)

        case ir_obj.SetIntersection():
            _encode_set_intersection(ref, defn, context)

        case ir_obj.SetDifference():
            _encode_set_difference(ref, defn, context)

        # ── Bags ──────────────────────────────────────────────────────────────
        case ir_obj.BagInit():
            _encode_bag_init(ref, defn, analysis, context)

        case ir_obj.BagChoose():
            _encode_bag_choose(ref, defn, analysis, context)

        case ir_obj.BagUnion():
            _encode_bag_union(ref, defn, analysis, context)

        case ir_obj.BagAdditiveUnion():
            _encode_bag_additive_union(ref, defn, analysis, context)

        case ir_obj.BagIntersection():
            _encode_bag_intersection(ref, defn, analysis, context)

        case ir_obj.BagDifference():
            _encode_bag_difference(ref, defn, analysis, context)

        case ir_obj.BagSupport():
            _encode_bag_support(ref, defn, context)

        # ── Functions ─────────────────────────────────────────────────────────
        case ir_obj.FuncDef():
            _encode_func_def(ref, defn, context)

        case ir_obj.FuncInverse():
            _encode_func_inverse(ref, defn, context)

        case ir_obj.FuncImage():
            _encode_func_image(ref, defn, problem, analysis, context)

        case ir_obj.FuncInverseImage():
            _encode_func_inverse_image(ref, defn, context)

        # ── Sequences ─────────────────────────────────────────────────────────
        case ir_obj.SequenceDef():
            _encode_sequence(ref, defn, problem, analysis, context)

        # ── Partitions ────────────────────────────────────────────────────────
        case ir_obj.PartitionDef():
            _encode_partition(ref, defn, problem, analysis, context)

        case ir_obj.PartRef():
            pass  # Handled inside _encode_partition

        case _:
            logger.warning(f"encode_ir: unhandled IR node type {type(defn)} for ref {ref}")


# =============================================================================
# Set encoders
# =============================================================================


def _encode_set_init(
    ref: ObjRef,
    defn: ir_obj.SetInit,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a SetInit node.

    A SetInit is a base set defined by explicit entities: e.g. A = {a, b, c}.

    LEGACY EQUIVALENT: encode() branch for SetInit (encoder.py ~line 547).

    Strategy:
    - Create one unary predicate p_<name> for the set.
    - For each entity in defn.entities: add unary evidence p_<name>(entity) = True.
    - For entities NOT in this set: add p_<name>(entity) = False (exclusion).
    - Singletons: entities in analysis.singletons get singleton treatment
      (their constant appears in exactly one set, so LEQ predicate applies).

    Args:
        ref: ObjRef.
        defn: SetInit dataclass (fields: entities: frozenset[IREntity]).
        analysis: AnalysisResult.
        context: ContextIR.
    """
    logger.debug("_encode_set_init: ref={}, entities={}", ref.id, len(defn.entities))
    obj_pred = context.get_pred(ref, create=True, use=False)
    for entity in defn.entities:
        context.unary_evidence.add(obj_pred(Const(entity.name)))
    for entity in context.analysis.all_entities:
        if entity not in defn.entities:
            context.unary_evidence.add(~obj_pred(Const(entity.name)))


def _encode_set_choose(
    ref: ObjRef,
    defn: ir_obj.SetChoose,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a SetChoose node (subset of known size or any size).

    LEGACY EQUIVALENT: encode() branch for SetChoose (encoder.py ~line 560).

    A SetChoose B picks a subset of B (without replacement).
    - If defn.size is set: the chosen subset has exactly that size.
    - The set_info for ref contains p_entities and max_size.

    Args:
        ref: ObjRef.
        defn: SetChoose dataclass (source: ObjRef, size: int | None).
        analysis: AnalysisResult.
        context: ContextIR.
    """
    obj_pred = context.get_pred(ref, create=True, use=False)
    from_pred = context.get_pred(defn.source)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
    )
    if defn.size is not None:
        var = context.get_obj_var(ref)
        context.validator.append(Eq(var, defn.size))


def _encode_set_choose_replace(
    ref: ObjRef,
    defn: ir_obj.SetChooseReplace,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a SetChooseReplace node (multiset chosen with replacement).

    LEGACY EQUIVALENT: encode() branch for SetChooseReplace (encoder.py ~line 573).

    SetChooseReplace picks elements from a source set WITH replacement,
    producing a multiset. Encoded like a bag: uses dis_entities and
    indis_entities from bag_info (SetChooseReplace has bag_info, not set_info).

    Note: analysis.bag_info[ref] is populated for SetChooseReplace by EntityAnalysis.

    Args:
        ref: ObjRef.
        defn: SetChooseReplace dataclass (source: ObjRef, size: int | None).
        analysis: AnalysisResult.
        context: ContextIR.
    """
    obj_pred = context.get_pred(ref, create=True, use=False)
    from_pred = context.get_pred(defn.source)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
    )
    bag_info = analysis.bag_info[ref]

    # Use a single shared variable for all entity polynomial weights.
    # This enables the multiset cardinality constraint to work correctly:
    # the WFOMC polynomial becomes (1 + x + x^2 + ... + x^mult)^n, and
    # Eq(x, size) extracts the coefficient of x^size = C(n+size-1, size).
    shared_var = context.get_obj_var(ref, set_weight=False)

    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        multiplicity = bag_info.p_entities_multiplicity[entity]
        if multiplicity == float("inf"):
            raise ValueError("SetChooseReplace with infinite size is not supported")
        entity_pred = _encode_entity_in_ctx(entity, context)
        bag_entity_pred = context.get_entity_pred(ref, entity)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
        )
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + shared_var**y, range(1, multiplicity + 1), 0),
            1,
        )

    if defn.size is not None:
        context.validator.append(Eq(shared_var, defn.size))


def _encode_set_union(
    ref: ObjRef,
    defn: ir_obj.SetUnion,
    context: ContextIR,
) -> None:
    """Encode a SetUnion node: ref = left ∪ right."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) | {right_pred}(X)))"
    )


def _encode_set_intersection(
    ref: ObjRef,
    defn: ir_obj.SetIntersection,
    context: ContextIR,
) -> None:
    """Encode a SetIntersection node: ref = left ∩ right."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) & {right_pred}(X)))"
    )


def _encode_set_difference(
    ref: ObjRef,
    defn: ir_obj.SetDifference,
    context: ContextIR,
) -> None:
    """Encode a SetDifference node: ref = left \\ right."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) & ~{right_pred}(X)))"
    )


# =============================================================================
# Bag encoders
# =============================================================================

# COMMON BAG ENCODING PATTERN
# ============================
# Bags are encoded via two disjoint groups of entities:
#
# 1. DISTINGUISHABLE entities (bag_info.dis_entities):
#    Each gets its own unary predicate p_<bag>_<entity> representing
#    whether that entity is in the bag (with its multiplicity as weight).
#
# 2. INDISTINGUISHABLE entities (bag_info.indis_entities: dict[int, set[Entity]]):
#    Entities with the same multiplicity are treated as interchangeable.
#    They share a single symbolic variable, and the count is a multinomial.
#
# See encoder.py ~lines 380–590 for the legacy implementation.


def _encode_bag_init(
    ref: ObjRef,
    defn: ir_obj.BagInit,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagInit node (base bag with explicit multiplicities).

    Since all multiplicities are statically known, no entity variables or
    cardinality constraints are needed. Membership is captured entirely by
    unary evidence on the main bag predicate. Derived bags (BagChoose, etc.)
    read the exact multiplicities directly from analysis.bag_info.
    """
    logger.debug(
        "_encode_bag_init: ref={}, entity_multiplicity={}",
        ref.id,
        list(defn.entity_multiplicity),
    )
    obj_pred = context.get_pred(ref, create=True, use=False)
    # Build set of entities in this bag
    entities_in_bag = {entity for entity, _ in defn.entity_multiplicity}
    # Add unary evidence for entities in the bag
    for entity in entities_in_bag:
        context.unary_evidence.add(obj_pred(Const(entity.name)))
    # Add negative evidence for entities NOT in the bag
    for entity in context.analysis.all_entities:
        if entity not in entities_in_bag:
            context.unary_evidence.add(~obj_pred(Const(entity.name)))
    # No entity variables needed: all multiplicities are fixed by defn.
    # BagChoose (and other derived bags) use integer constants from
    # analysis.bag_info[ref].p_entities_multiplicity as upper bounds.


def _encode_bag_choose(
    ref: ObjRef,
    defn: ir_obj.BagChoose,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagChoose node (sub-bag chosen from source)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    from_pred = context.get_pred(defn.source)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
    )
    bag_info = analysis.bag_info[ref]
    source_bag_info = analysis.bag_info[defn.source]
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        multiplicity = bag_info.p_entities_multiplicity[entity]
        entity_pred = _encode_entity_in_ctx(entity, context)
        bag_entity_pred = context.get_entity_pred(ref, entity)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
        )
        entity_var = context.get_entity_var(ref, entity)
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + entity_var**y, range(1, multiplicity + 1), 0),
            1,
        )
    # Add constraint that chosen multiplicity <= source multiplicity
    source_defn = context.problem.get_object(defn.source)
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        entity_var = context.get_entity_var(ref, entity)
        # Get source multiplicity
        if isinstance(source_defn, ir_obj.BagInit):
            source_mul = source_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            source_entity_var = context.get_entity_var(defn.source, entity)
            source_mul = source_entity_var
        context.validator.append(entity_var <= source_mul)


def _encode_bag_union(
    ref: ObjRef,
    defn: ir_obj.BagUnion,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagUnion node: ref = max(left, right) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) | {right_pred}(X)))"
    )


def _encode_bag_additive_union(
    ref: ObjRef,
    defn: ir_obj.BagAdditiveUnion,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagAdditiveUnion node: ref = left + right (sum multiplicities)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) | {right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
    left_defn = context.problem.get_object(defn.left)
    right_defn = context.problem.get_object(defn.right)
    left_bag_info = analysis.bag_info.get(defn.left)
    right_bag_info = analysis.bag_info.get(defn.right)
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        multiplicity = bag_info.p_entities_multiplicity[entity]
        entity_pred = _encode_entity_in_ctx(entity, context)
        bag_entity_pred = context.get_entity_pred(ref, entity)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
        )
        entity_var = context.get_entity_var(ref, entity)
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + entity_var**y, range(1, multiplicity + 1), 0),
            1,
        )
        # Get left multiplicity
        if isinstance(left_defn, ir_obj.BagInit):
            left_mul = left_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            left_mul = context.get_entity_var(defn.left, entity)
        # Get right multiplicity
        if isinstance(right_defn, ir_obj.BagInit):
            right_mul = right_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            right_mul = context.get_entity_var(defn.right, entity)
        context.validator.append(Eq(entity_var, left_mul + right_mul))


def _encode_bag_intersection(
    ref: ObjRef,
    defn: ir_obj.BagIntersection,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagIntersection node: ref = min(left, right) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) & {right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
    left_defn = context.problem.get_object(defn.left)
    right_defn = context.problem.get_object(defn.right)
    left_bag_info = analysis.bag_info.get(defn.left)
    right_bag_info = analysis.bag_info.get(defn.right)
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        multiplicity = bag_info.p_entities_multiplicity[entity]
        entity_pred = _encode_entity_in_ctx(entity, context)
        bag_entity_pred = context.get_entity_pred(ref, entity)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
        )
        entity_var = context.get_entity_var(ref, entity)
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + entity_var**y, range(1, multiplicity + 1), 0),
            1,
        )
        # Get left multiplicity
        if isinstance(left_defn, ir_obj.BagInit):
            left_mul = left_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            left_mul = context.get_entity_var(defn.left, entity)
        # Get right multiplicity
        if isinstance(right_defn, ir_obj.BagInit):
            right_mul = right_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            right_mul = context.get_entity_var(defn.right, entity)
        context.validator.append(Eq(entity_var, Min(left_mul, right_mul)))


def _encode_bag_difference(
    ref: ObjRef,
    defn: ir_obj.BagDifference,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagDifference node: ref = max(left - right, 0) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) & ~{right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
    left_defn = context.problem.get_object(defn.left)
    right_defn = context.problem.get_object(defn.right)
    left_bag_info = analysis.bag_info.get(defn.left)
    right_bag_info = analysis.bag_info.get(defn.right)
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        multiplicity = bag_info.p_entities_multiplicity[entity]
        entity_pred = _encode_entity_in_ctx(entity, context)
        bag_entity_pred = context.get_entity_pred(ref, entity)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
        )
        entity_var = context.get_entity_var(ref, entity)
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + entity_var**y, range(1, multiplicity + 1), 0),
            1,
        )
        # Get left multiplicity
        if isinstance(left_defn, ir_obj.BagInit):
            left_mul = left_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            left_mul = context.get_entity_var(defn.left, entity)
        # Get right multiplicity
        if isinstance(right_defn, ir_obj.BagInit):
            right_mul = right_bag_info.p_entities_multiplicity.get(entity, 0)
        else:
            right_mul = context.get_entity_var(defn.right, entity)
        context.validator.append(Eq(entity_var, left_mul - right_mul))


def _encode_bag_support(
    ref: ObjRef,
    defn: ir_obj.BagSupport,
    context: ContextIR,
) -> None:
    """Encode a BagSupport node: ref = support set of source bag."""
    # BagSupport is an alias for the source's predicate
    source_pred = context.get_pred(defn.source)
    context.set_pred(ref, source_pred)
    context.used_refs.add(defn.source)


# =============================================================================
# Function encoders
# =============================================================================


def _encode_func_def(
    ref: ObjRef,
    defn: ir_obj.FuncDef,
    context: ContextIR,
) -> None:
    """Encode a FuncDef node (function from domain to codomain)."""
    logger.debug(
        "_encode_func_def: ref={}, domain={}, codomain={}, surjective={}",
        ref.id,
        defn.domain.id,
        defn.codomain.id,
        defn.surjective,
    )
    # Create a BINARY predicate for the function (arity=2)
    name = context.problem.get_name(ref) or f"obj_{ref.id}"
    obj_pred = create_cofola_pred(name, 2)
    context.set_pred(ref, obj_pred)
    domain_pred = context.get_pred(defn.domain)
    codomain_pred = context.get_pred(defn.codomain)
    # Function sentence: every domain element maps to exactly one codomain element
    context.sentence = context.sentence & parse(
        f"\\forall X: (\\exists Y: ({domain_pred}(X) -> {obj_pred}(X, Y)))"
        f" & \\forall X: (\\forall Y: ({obj_pred}(X, Y) -> ({domain_pred}(X) & {codomain_pred}(Y))))"
    )
    # Set up size variable for the function
    obj_var = context.get_obj_var(ref)
    domain_size = context.get_obj_var(defn.domain)
    context.validator.append(Eq(obj_var, domain_size))
    # If surjective, add surjectivity sentence
    if defn.surjective:
        context.sentence = context.sentence & parse(
            f"\\forall Y: (\\exists X: ({codomain_pred}(Y) -> {obj_pred}(X, Y)))"
        )


def _encode_func_inverse(
    ref: ObjRef,
    defn: ir_obj.FuncInverse,
    context: ContextIR,
) -> None:
    """Encode a FuncInverse node (f⁻¹ as a function object)."""
    # Create a BINARY predicate for the inverse function
    name = context.problem.get_name(ref) or f"obj_{ref.id}"
    obj_pred = create_cofola_pred(name, 2)
    context.set_pred(ref, obj_pred)
    func_pred = context.get_pred(defn.func)
    # Inverse: p_ref(X, Y) <-> p_func(Y, X)
    context.sentence = context.sentence & parse(
        f"\\forall X: (\\forall Y: ({obj_pred}(X, Y) <-> {func_pred}(Y, X)))"
    )


def _encode_func_image(
    ref: ObjRef,
    defn: ir_obj.FuncImage,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a FuncImage node: ref = f(argument)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    func_pred = context.get_pred(defn.func)
    argument = defn.argument
    # Check if argument is an Entity or ObjRef
    if isinstance(argument, Entity):
        # Image of a single entity
        entity_pred = _encode_entity_in_ctx(argument, context)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: ({entity_pred}(X) & {func_pred}(X, Y) -> {obj_pred}(Y))) "
            f"& \\forall X: (\\forall Y: ({entity_pred}(X) & ~{func_pred}(X, Y) -> ~{obj_pred}(Y)))"
        )
    else:
        # Image of a set
        arg_pred = context.get_pred(argument)
        context.sentence = context.sentence & parse(
            f"\\forall Y: ({obj_pred}(Y) <-> (\\exists X: ({arg_pred}(X) & {func_pred}(X, Y))))"
        )


def _encode_func_inverse_image(
    ref: ObjRef,
    defn: ir_obj.FuncInverseImage,
    context: ContextIR,
) -> None:
    """Encode a FuncInverseImage node: ref = f⁻¹(argument)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    func_pred = context.get_pred(defn.func)
    argument = defn.argument
    # Check if argument is an Entity or ObjRef
    if isinstance(argument, Entity):
        # Inverse image of a single entity
        entity_pred = _encode_entity_in_ctx(argument, context)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: ({entity_pred}(Y) & {func_pred}(X, Y) -> {obj_pred}(X))) "
            f"& \\forall X: (\\forall Y: ({entity_pred}(Y) & ~{func_pred}(X, Y) -> ~{obj_pred}(X)))"
        )
    else:
        # Inverse image of a set
        arg_pred = context.get_pred(argument)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> (\\exists Y: ({arg_pred}(Y) & {func_pred}(X, Y))))"
        )


# =============================================================================
# Sequence encoder
# =============================================================================


def _encode_sequence(
    ref: ObjRef,
    defn: ir_obj.SequenceDef,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a SequenceDef node.

    After LoweringPass:
    - choose=True, replace=False → already lowered to SetChoose/BagChoose + SequenceDef(choose=False)
    - choose=True, replace=True → currently not fully supported
    - choose=False → sequence of a fixed source

    For a sequence from a Set:
    - Use source predicate directly
    - Add linear order via LEQ predicate
    - Apply overcount correction (factorial of domain-size elements)

    For a sequence from a Bag:
    - Encode entity predicates with multiplicity variables
    - Apply overcount correction
    """
    logger.debug(
        "_encode_sequence: ref={}, source={}, size={}, circular={}, reflection={}",
        ref.id,
        defn.source.id,
        defn.size,
        defn.circular,
        defn.reflection,
    )
    domain_size = len(context.domain)
    source_defn = problem.get_object(defn.source)

    # Check if source is a set or bag
    set_info = analysis.set_info.get(defn.source)
    bag_info = analysis.bag_info.get(defn.source)

    if set_info is not None:
        logger.debug("_encode_sequence: source is a Set")
        # Sequence from a Set
        source_pred = context.get_pred(defn.source)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({source_pred}(X) & ~{source_pred}(Y)) -> {context.leq_pred}(X,Y)))"
        )
        # Overcount correction: (domain_size - size)!
        if defn.size is not None:
            context.overcount = context.overcount * math.factorial(domain_size - defn.size)
    elif bag_info is not None:
        logger.debug(
            "_encode_sequence: source is a Bag, dis_entities={}",
            len(bag_info.dis_entities),
        )
        # Sequence from a Bag - need entity predicates
        source_pred = context.get_pred(defn.source)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({source_pred}(X) & ~{source_pred}(Y)) -> {context.leq_pred}(X,Y)))"
        )
        # Create entity predicates for each distinguishable entity
        entity_preds = []
        for entity in bag_info.dis_entities:
            if entity in context.singletons:
                continue
            entity_pred = context.get_entity_pred(ref, entity)
            entity_preds.append(entity_pred)
            entity_var = context.get_entity_var(ref, entity)
            context.weighting[entity_pred] = (entity_var, 1)
            # Get multiplicity from source
            if isinstance(source_defn, ir_obj.BagInit):
                multi = bag_info.p_entities_multiplicity.get(entity, 0)
                context.validator.append(Eq(entity_var, multi))
            else:
                # Source is derived bag - use its entity var
                source_entity_var = context.get_entity_var(defn.source, entity)
                context.validator.append(Eq(entity_var, source_entity_var))
        # Exclusive covering
        if entity_preds:
            or_formula = " | ".join(f"{pred}(X)" for pred in entity_preds)
            context.sentence = context.sentence & parse(
                f"\\forall X: (({or_formula}) <-> {source_pred}(X))"
            )
            context.sentence = context.sentence & exclusive(entity_preds)
        # Overcount correction for bag sequences
        if defn.size is not None:
            context.overcount = (
                context.overcount
                * math.factorial(domain_size - defn.size)
                * math.factorial(defn.size)
            )
    else:
        raise ValueError(f"Sequence source {defn.source} is neither a set nor a bag")

    # Handle circular and reflection
    if defn.circular:
        context.circle_len = defn.size or domain_size
        context.overcount = context.overcount * (defn.size or domain_size)
        if defn.reflection:
            context.overcount = context.overcount * 2


# =============================================================================
# Partition encoder
# =============================================================================


def _encode_partition(
    ref: ObjRef,
    defn: ir_obj.PartitionDef,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a PartitionDef node (set partition or bag partition, ordered or not).

    Parts: obtained via context.get_parts_of(ref) → list[ObjRef] of PartRef nodes.
    """
    logger.debug(
        "_encode_partition: ref={}, source={}, ordered={}",
        ref.id,
        defn.source.id,
        defn.ordered,
    )
    parts = context.get_parts_of(ref)
    source_pred = context.get_pred(defn.source)
    part_preds = [context.get_pred(p, create=True) for p in parts]

    # Coverage: source(X) <-> (part1(X) | part2(X) | ... | partN(X))
    or_formula = " | ".join(f"{pred}(X)" for pred in part_preds)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({source_pred}(X) <-> ({or_formula}))"
    )

    # Check if source is a set or bag
    set_info = analysis.set_info.get(defn.source)
    bag_info = analysis.bag_info.get(defn.source)

    logger.debug("_encode_partition: {} parts, source is {}", len(parts), "Set" if set_info is not None else "Bag")
    if set_info is not None:
        # Set partition - parts must be disjoint
        for i, pred1 in enumerate(part_preds):
            for j, pred2 in enumerate(part_preds):
                if i >= j:
                    continue
                context.sentence = context.sentence & parse(
                    f"\\forall X: (~({pred1}(X) & {pred2}(X)))"
                )

        # If not ordered, add symmetry breaking via size variables
        if not defn.ordered:
            indis_vars = []
            pre_var = None
            for part in parts:
                var = context.get_obj_var(part)
                if pre_var is not None:
                    context.validator.append(pre_var <= var)
                pre_var = var
                context.weighting[context.get_pred(part)] = (var, 1)
                indis_vars.append([var])
            context.indis_vars.append(indis_vars)

    elif bag_info is not None:
        # Bag partition - parts are sub-bags
        source_defn = problem.get_object(defn.source)

        # For singletons, add exactly-one constraint
        if context.singletons and context.singletons_pred is not None:
            exactly_one_formula = exactly_one_qf(part_preds)
            context.sentence = context.sentence & parse(
                f"\\forall X: (({context.singletons_pred}(X) & {source_pred}(X)) -> ({exactly_one_formula}))"
            )

        # Entity multiplicity partitioning
        ordered_vars = [[] for _ in range(len(parts))]
        for entity in bag_info.dis_entities:
            if entity in context.singletons:
                continue
            multi = bag_info.p_entities_multiplicity[entity]
            entity_pred = _encode_entity_in_ctx(entity, context)

            # Get source multiplicity
            if isinstance(source_defn, ir_obj.BagInit):
                multi_var = multi
            else:
                multi_var = context.get_entity_var(defn.source, entity)

            partitioned_vars = []
            for idx, part in enumerate(parts):
                pred = context.get_entity_pred(part, entity)
                part_pred = context.get_pred(part)
                context.sentence = context.sentence & parse(
                    f"\\forall X: ({pred}(X) <-> ({part_pred}(X) & {entity_pred}(X)))"
                )
                var = context.get_entity_var(part, entity)
                context.weighting[pred] = (sum(var**i for i in range(1, multi + 1)), 1)
                partitioned_vars.append(var)
                if not defn.ordered:
                    ordered_vars[idx].append(var)

            context.validator.append(Eq(sum(partitioned_vars), multi_var))

        # Symmetry breaking for unordered partitions
        if not defn.ordered:
            for i in range(len(ordered_vars) - 1):
                context.validator.append(ListLessThan(ordered_vars[i], ordered_vars[i + 1]))


# =============================================================================
# Constraint encoders
# =============================================================================


def _encode_constraint(
    c: object,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Dispatch to the appropriate constraint encoder.

    Args:
        c: The IR constraint dataclass.
        problem: Problem.
        analysis: AnalysisResult.
        context: ContextIR.
    """
    logger.debug("Encoding constraint {}", type(c).__name__)
    match c:
        case ir_cst.SizeConstraint():
            _encode_size_constraint(c, problem, analysis, context)

        case ir_cst.MembershipConstraint():
            _encode_membership_constraint(c, context)

        case ir_cst.SubsetConstraint():
            _encode_subset_constraint(c, context)

        case ir_cst.DisjointConstraint():
            _encode_disjoint_constraint(c, context)

        case ir_cst.EqualityConstraint():
            _encode_equality_constraint(c, context)

        case ir_cst.FuncPairConstraint():
            _encode_func_pair_constraint(c, problem, context)

        case ir_cst.SequencePatternConstraint():
            _encode_sequence_pattern_constraint(c, context)

        case ir_cst.BagSubsetConstraint():
            _encode_bag_subset_constraint(c, analysis, context)

        case ir_cst.BagEqConstraint():
            _encode_bag_eq_constraint(c, analysis, context)

        case ir_cst.TupleIndexEq():
            _encode_tuple_index_eq(c, context)

        case ir_cst.TupleIndexMembership():
            _encode_tuple_index_membership(c, context)

        case ir_cst.NotConstraint():
            raise NotImplementedError("NotConstraint should be expanded by solver")

        case ir_cst.AndConstraint():
            _encode_constraint(c.left, problem, analysis, context)
            _encode_constraint(c.right, problem, analysis, context)

        case ir_cst.OrConstraint():
            raise NotImplementedError("OrConstraint requires DNF expansion")

        case _:
            logger.warning(f"encode_ir: unhandled constraint type {type(c)}")


def _encode_size_constraint(
    c: ir_cst.SizeConstraint,
    problem: Problem,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a SizeConstraint."""
    # Build the left-hand expression from terms
    left = 0
    for term, coef in c.terms:
        if isinstance(term, ObjRef):
            var = context.get_obj_var(term)
        elif isinstance(term, ir_cst.BagCountAtom):
            var = _get_bag_count_var(term, context)
        elif isinstance(term, ir_cst.SeqPatternCountAtom):
            var = _get_seq_pattern_count_var(term, context)
        else:
            raise TypeError(f"Unknown term type in SizeConstraint: {type(term)}")
        left = left + coef * var

    # Map comparator string to sympy function
    comparator_map = {
        "==": Eq,
        "<": Lt,
        "<=": Le,
        ">": Gt,
        ">=": Ge,
    }
    comparator_fn = comparator_map.get(c.comparator)
    if comparator_fn is None:
        raise ValueError(f"Unknown comparator: {c.comparator}")
    context.validator.append(comparator_fn(left, c.rhs))


def _encode_membership_constraint(
    c: ir_cst.MembershipConstraint,
    context: ContextIR,
) -> None:
    """Encode a MembershipConstraint: entity [not] in container."""
    obj_pred = context.get_pred(c.container)
    if c.positive:
        context.unary_evidence.add(obj_pred(Const(c.entity.name)))
    else:
        context.unary_evidence.add(~obj_pred(Const(c.entity.name)))


def _encode_subset_constraint(
    c: ir_cst.SubsetConstraint,
    context: ContextIR,
) -> None:
    """Encode a SubsetConstraint: sub [not] subset sup."""
    sub_pred = context.get_pred(c.sub)
    sup_pred = context.get_pred(c.sup)
    if c.positive:
        context.sentence = context.sentence & parse(
            f"\\forall X: ({sub_pred}(X) -> {sup_pred}(X))"
        )
    else:
        context.sentence = context.sentence & parse(
            f"\\exists X: ({sub_pred}(X) & ~{sup_pred}(X))"
        )


def _encode_disjoint_constraint(
    c: ir_cst.DisjointConstraint,
    context: ContextIR,
) -> None:
    """Encode a DisjointConstraint: left [not] disjoint right."""
    left_pred = context.get_pred(c.left)
    right_pred = context.get_pred(c.right)
    if c.positive:
        context.sentence = context.sentence & parse(
            f"\\forall X: (~({left_pred}(X) & {right_pred}(X)))"
        )
    else:
        context.sentence = context.sentence & parse(
            f"\\exists X: ({left_pred}(X) & {right_pred}(X))"
        )


def _encode_equality_constraint(
    c: ir_cst.EqualityConstraint,
    context: ContextIR,
) -> None:
    """Encode an EqualityConstraint: left == right (as sets)."""
    left_pred = context.get_pred(c.left)
    right_pred = context.get_pred(c.right)
    if c.positive:
        context.sentence = context.sentence & parse(
            f"\\forall X: ({left_pred}(X) <-> {right_pred}(X))"
        )
    else:
        context.sentence = context.sentence & parse(
            f"\\exists X: (({left_pred}(X) & ~{right_pred}(X)) | (~{left_pred}(X) & {right_pred}(X)))"
        )


def _encode_func_pair_constraint(
    c: ir_cst.FuncPairConstraint,
    problem: Problem,
    context: ContextIR,
) -> None:
    """Encode a FuncPairConstraint: f(arg_entity) [not] in result."""
    func_pred = context.get_pred(c.func)
    arg_pred = _encode_entity_in_ctx(c.arg_entity, context)

    # Check if result is an Entity or ObjRef
    if isinstance(c.result, Entity):
        result_pred = _encode_entity_in_ctx(c.result, context)
    else:
        result_pred = context.get_pred(c.result)

    if c.positive:
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: ({func_pred}(X, Y) & {arg_pred}(X) -> {result_pred}(Y)))"
        )
    else:
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: ({func_pred}(X, Y) & {arg_pred}(X) -> ~{result_pred}(Y)))"
        )


def _encode_sequence_pattern_constraint(
    c: ir_cst.SequencePatternConstraint,
    context: ContextIR,
) -> None:
    """Encode a SequencePatternConstraint."""
    match c.pattern:
        case ir_cst.TogetherPattern():
            _encode_together_pattern(c.seq, c.pattern, c.positive, context)
        case ir_cst.LessThanPattern():
            _encode_less_than_pattern(c.seq, c.pattern, c.positive, context)
        case ir_cst.PredecessorPattern():
            _encode_predecessor_pattern(c.seq, c.pattern, c.positive, context)
        case ir_cst.NextToPattern():
            _encode_next_to_pattern(c.seq, c.pattern, c.positive, context)


def _encode_bag_subset_constraint(
    c: ir_cst.BagSubsetConstraint,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagSubsetConstraint: sub ⊆ sup (by multiplicity)."""
    sub_info = analysis.bag_info.get(c.sub)
    sup_info = analysis.bag_info.get(c.sup)
    if sub_info is None or sup_info is None:
        return

    for entity in sub_info.dis_entities:
        if entity in context.singletons:
            continue
        sub_var = context.get_entity_var(c.sub, entity)
        sup_var = context.get_entity_var(c.sup, entity)
        if c.positive:
            context.validator.append(sub_var <= sup_var)
        else:
            # Not subset means at least one entity has strictly greater multiplicity
            context.validator.append(sub_var > sup_var)


def _encode_bag_eq_constraint(
    c: ir_cst.BagEqConstraint,
    analysis: AnalysisResult,
    context: ContextIR,
) -> None:
    """Encode a BagEqConstraint: left == right (by multiplicity)."""
    left_info = analysis.bag_info.get(c.left)
    right_info = analysis.bag_info.get(c.right)
    if left_info is None or right_info is None:
        return

    # Get all entities from both bags
    all_entities = left_info.dis_entities | right_info.dis_entities
    for entity in all_entities:
        if entity in context.singletons:
            continue
        left_var = context.get_entity_var(c.left, entity)
        right_var = context.get_entity_var(c.right, entity)
        if c.positive:
            context.validator.append(Eq(left_var, right_var))
        else:
            # Not equal means at least one entity differs
            context.validator.append(~Eq(left_var, right_var))


def _encode_tuple_index_eq(
    c: ir_cst.TupleIndexEq,
    context: ContextIR,
) -> None:
    """Encode a TupleIndexEq: T[index] [!=] entity.

    LEGACY EQUIVALENT: not directly in encoder.py (tuples are lowered to functions
    by LoweringPass before encoding). After lowering, tuple index constraints
    become FuncPairConstraints. If this is still present, it's an error.

    After LoweringPass, TupleIndexEq should not appear. Raise a warning/error.

    Args:
        c: TupleIndexEq dataclass.
        context: ContextIR.
    """
    logger.warning(
        "TupleIndexEq reached encoder — TupleDef should have been lowered. "
        "This constraint will be ignored."
    )


def _encode_tuple_index_membership(
    c: ir_cst.TupleIndexMembership,
    context: ContextIR,
) -> None:
    """Encode a TupleIndexMembership: T[index] [not] in container.

    LEGACY EQUIVALENT: Same as TupleIndexEq — should be lowered away.

    Args:
        c: TupleIndexMembership dataclass.
        context: ContextIR.
    """
    logger.warning(
        "TupleIndexMembership reached encoder — TupleDef should have been lowered. "
        "This constraint will be ignored."
    )


# =============================================================================
# Sequence pattern sub-encoders
# =============================================================================


def _encode_together_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.TogetherPattern,
    positive: bool,
    context: ContextIR,
) -> None:
    """Encode a TogetherPattern: elements of group appear consecutively in seq."""
    group_ref = pattern.group

    # Get predicate for the group
    group_defn = context.problem.get_object(group_ref)
    if isinstance(group_defn, Entity):
        # Check if sequence source is a bag
        seq_defn = context.problem.get_object(seq_ref)
        if seq_defn is not None and hasattr(seq_defn, 'source'):
            source_defn = context.problem.get_object(seq_defn.source)
            if source_defn is not None and isinstance(source_defn, ir_obj.BagInit):
                obj_pred = context.get_entity_pred(seq_ref, group_defn)
            else:
                obj_pred = _encode_entity_in_ctx(group_defn, context)
        else:
            obj_pred = _encode_entity_in_ctx(group_defn, context)
    else:
        obj_pred = context.get_pred(group_ref)

    # Get predecessor predicate for the sequence
    pred_pred = context.get_predecessor_pred(seq_ref)
    source_pred = context.get_pred(context.problem.get_object(seq_ref).source)

    # Create "first" predicate
    name = context.problem.get_name(group_ref) or f"obj_{group_ref.id}"
    first_pred = create_aux_pred(1, f"{name}_first")

    # Define first(X) as the first element of the group in the sequence
    context.sentence = context.sentence & parse(
        f"\\forall X: ({first_pred}(X) <-> ({obj_pred}(X) & {source_pred}(X) & "
        f"\\forall Y: (({obj_pred}(Y) & {source_pred}(Y)) -> ~{pred_pred}(Y,X))))"
    )

    # Create variable for counting first elements
    first_var = context.create_var(f"{name}_first")
    context.weighting[first_pred] = (first_var, 1)

    if positive:
        context.validator.append(first_var <= 1)
    else:
        context.validator.append(first_var > 1)


def _encode_less_than_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.LessThanPattern,
    positive: bool,
    context: ContextIR,
) -> None:
    """Encode a LessThanPattern: left appears before right in seq."""

    # Get predicates for left and right
    if isinstance(pattern.left, Entity):
        left_pred = _encode_entity_in_ctx(pattern.left, context)
    else:
        left_pred = context.get_pred(pattern.left)

    if isinstance(pattern.right, Entity):
        right_pred = _encode_entity_in_ctx(pattern.right, context)
    else:
        right_pred = context.get_pred(pattern.right)

    # Get LEQ predicate for the sequence
    leq_pred = context.get_leq_pred(seq_ref)

    # Create auxiliary predicate for the relation
    left_name = context.problem.get_name(pattern.left) if isinstance(pattern.left, ObjRef) else str(pattern.left)
    right_name = context.problem.get_name(pattern.right) if isinstance(pattern.right, ObjRef) else str(pattern.right)
    obj_leq_pred = create_aux_pred(2, f"{left_name}_leq_{right_name}")

    context.sentence = context.sentence & parse(
        f"\\forall X: (\\forall Y: (({left_pred}(X) & {right_pred}(Y) & {leq_pred}(X,Y)) <-> {obj_leq_pred}(X,Y)))"
    )

    leq_var = context.create_var(obj_leq_pred.name)
    context.weighting[obj_leq_pred] = (leq_var, 1)

    if positive:
        context.validator.append(leq_var > 0)
    else:
        context.validator.append(Eq(leq_var, 0))


def _encode_predecessor_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.PredecessorPattern,
    positive: bool,
    context: ContextIR,
) -> None:
    """Encode a PredecessorPattern: first immediately precedes second in seq."""

    # Get predicates for first and second
    if isinstance(pattern.first, Entity):
        first_pred = _encode_entity_in_ctx(pattern.first, context)
    else:
        first_pred = context.get_pred(pattern.first)

    if isinstance(pattern.second, Entity):
        second_pred = _encode_entity_in_ctx(pattern.second, context)
    else:
        second_pred = context.get_pred(pattern.second)

    # Get predecessor predicate for the sequence
    pred_pred = context.get_predecessor_pred(seq_ref)

    # Create auxiliary predicate
    first_name = context.problem.get_name(pattern.first) if isinstance(pattern.first, ObjRef) else str(pattern.first)
    second_name = context.problem.get_name(pattern.second) if isinstance(pattern.second, ObjRef) else str(pattern.second)
    obj_pred_pred = create_aux_pred(2, f"{first_name}_pred_{second_name}")

    context.sentence = context.sentence & parse(
        f"\\forall X: (\\forall Y: (({first_pred}(X) & {second_pred}(Y) & {pred_pred}(X,Y)) <-> {obj_pred_pred}(X,Y)))"
    )

    pred_var = context.create_var(obj_pred_pred.name)
    context.weighting[obj_pred_pred] = (pred_var, 1)

    if positive:
        context.validator.append(pred_var > 0)
    else:
        context.validator.append(Eq(pred_var, 0))


def _encode_next_to_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.NextToPattern,
    positive: bool,
    context: ContextIR,
) -> None:
    """Encode a NextToPattern: first and second are adjacent in seq."""

    # Get predicates for first and second
    if isinstance(pattern.first, Entity):
        first_pred = _encode_entity_in_ctx(pattern.first, context)
    else:
        first_pred = context.get_pred(pattern.first)

    if isinstance(pattern.second, Entity):
        second_pred = _encode_entity_in_ctx(pattern.second, context)
    else:
        second_pred = context.get_pred(pattern.second)

    # Get next-to predicate for the sequence
    next_to_pred = context.get_next_to_pred(seq_ref)

    # Create auxiliary predicate
    first_name = context.problem.get_name(pattern.first) if isinstance(pattern.first, ObjRef) else str(pattern.first)
    second_name = context.problem.get_name(pattern.second) if isinstance(pattern.second, ObjRef) else str(pattern.second)
    obj_next_to_pred = create_aux_pred(2, f"{first_name}_next_to_{second_name}")

    context.sentence = context.sentence & parse(
        f"\\forall X: (\\forall Y: (({first_pred}(X) & {second_pred}(Y) & {next_to_pred}(X,Y)) <-> {obj_next_to_pred}(X,Y)))"
    )

    next_to_var = context.create_var(obj_next_to_pred.name)
    context.weighting[obj_next_to_pred] = (next_to_var, 1)

    if positive:
        context.validator.append(next_to_var > 0)
    else:
        context.validator.append(Eq(next_to_var, 0))


# =============================================================================
# Size-atom variable helpers
# =============================================================================


def _get_bag_count_var(
    atom: ir_cst.BagCountAtom,
    context: ContextIR,
) -> object:
    """Get the symbolic variable for a BagCountAtom (B.count(e)).

    LEGACY EQUIVALENT: BagMultiplicity weight variable in encoder.py ~line 101.

    Returns context.get_entity_var(atom.bag, atom.entity).

    Args:
        atom: BagCountAtom dataclass (bag: ObjRef, entity: IREntity).
        context: ContextIR.

    Returns:
        Symbolic Expr for the entity multiplicity variable.
    """
    return context.get_entity_var(atom.bag, atom.entity)


def _get_seq_pattern_count_var(
    atom: ir_cst.SeqPatternCountAtom,
    context: ContextIR,
) -> object:
    """Get the symbolic variable for a SeqPatternCountAtom (S.count(pattern)).

    This creates a variable representing the count of the given pattern
    in the given sequence by encoding the pattern and extracting the count var.
    """
    seq_ref = atom.seq
    pattern = atom.pattern

    # Encode the pattern and extract the count variable

    match pattern:
        case ir_cst.LessThanPattern():
            # Get predicates for left and right
            if isinstance(pattern.left, Entity):
                left_pred = _encode_entity_in_ctx(pattern.left, context)
            else:
                left_pred = context.get_pred(pattern.left)

            if isinstance(pattern.right, Entity):
                right_pred = _encode_entity_in_ctx(pattern.right, context)
            else:
                right_pred = context.get_pred(pattern.right)

            # Get LEQ predicate for the sequence
            leq_pred = context.get_leq_pred(seq_ref)

            # Create auxiliary predicate
            left_name = context.problem.get_name(pattern.left) if isinstance(pattern.left, ObjRef) else str(pattern.left)
            right_name = context.problem.get_name(pattern.right) if isinstance(pattern.right, ObjRef) else str(pattern.right)
            obj_leq_pred = create_aux_pred(2, f"{left_name}_leq_{right_name}")

            context.sentence = context.sentence & parse(
                f"\\forall X: (\\forall Y: (({left_pred}(X) & {right_pred}(Y) & {leq_pred}(X,Y)) <-> {obj_leq_pred}(X,Y)))"
            )

            leq_var = context.create_var(obj_leq_pred.name)
            context.weighting[obj_leq_pred] = (leq_var, 1)
            return leq_var

        case ir_cst.PredecessorPattern():
            # Get predicates for first and second
            if isinstance(pattern.first, Entity):
                first_pred = _encode_entity_in_ctx(pattern.first, context)
            else:
                first_pred = context.get_pred(pattern.first)

            if isinstance(pattern.second, Entity):
                second_pred = _encode_entity_in_ctx(pattern.second, context)
            else:
                second_pred = context.get_pred(pattern.second)

            # Get predecessor predicate for the sequence
            pred_pred = context.get_predecessor_pred(seq_ref)

            # Create auxiliary predicate
            first_name = context.problem.get_name(pattern.first) if isinstance(pattern.first, ObjRef) else str(pattern.first)
            second_name = context.problem.get_name(pattern.second) if isinstance(pattern.second, ObjRef) else str(pattern.second)
            obj_pred_pred = create_aux_pred(2, f"{first_name}_pred_{second_name}")

            context.sentence = context.sentence & parse(
                f"\\forall X: (\\forall Y: (({first_pred}(X) & {second_pred}(Y) & {pred_pred}(X,Y)) <-> {obj_pred_pred}(X,Y)))"
            )

            pred_var = context.create_var(obj_pred_pred.name)
            context.weighting[obj_pred_pred] = (pred_var, 1)
            return pred_var

        case ir_cst.NextToPattern():
            # Get predicates for first and second
            if isinstance(pattern.first, Entity):
                first_pred = _encode_entity_in_ctx(pattern.first, context)
            else:
                first_pred = context.get_pred(pattern.first)

            if isinstance(pattern.second, Entity):
                second_pred = _encode_entity_in_ctx(pattern.second, context)
            else:
                second_pred = context.get_pred(pattern.second)

            # Get next-to predicate for the sequence
            next_to_pred = context.get_next_to_pred(seq_ref)

            # Create auxiliary predicate
            first_name = context.problem.get_name(pattern.first) if isinstance(pattern.first, ObjRef) else str(pattern.first)
            second_name = context.problem.get_name(pattern.second) if isinstance(pattern.second, ObjRef) else str(pattern.second)
            obj_next_to_pred = create_aux_pred(2, f"{first_name}_next_to_{second_name}")

            context.sentence = context.sentence & parse(
                f"\\forall X: (\\forall Y: (({first_pred}(X) & {second_pred}(Y) & {next_to_pred}(X,Y)) <-> {obj_next_to_pred}(X,Y)))"
            )

            next_to_var = context.create_var(obj_next_to_pred.name)
            context.weighting[obj_next_to_pred] = (next_to_var, 1)
            return next_to_var

        case ir_cst.TogetherPattern():
            # Together pattern - return the "first" variable
            group_ref = pattern.group
            name = context.problem.get_name(group_ref) or f"obj_{group_ref.id}"
            first_var = context.create_var(f"{name}_first")
            return first_var

        case _:
            raise TypeError(f"Unknown pattern type: {type(pattern)}")
