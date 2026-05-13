"""WFOMC object encoders."""
from __future__ import annotations

import math
from functools import reduce

from sympy import Eq, Max, Min
from wfomc import Const, exactly_one_qf, exclusive, fol_parse as parse

import cofola.frontend.objects as ir_obj
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.encoding_helpers import (
    _bag_entity_expr,
    _encode_entity_in_ctx,
    _get_bag_size_expr,
)
from cofola.backend.wfomc.utils import ListLessThan, create_cofola_pred
from cofola.frontend.objects import Entity, ObjRef
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from loguru import logger


# Object encoding dispatcher
# =============================================================================


def _encode_object(
    ref: ObjRef,
    defn: object,
    problem: Problem,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Dispatch to the appropriate object encoder based on IR node type.

    Args:
        ref: The ObjRef for this node.
        defn: The IR dataclass node (SetInit, BagInit, etc.).
        problem: The Problem (for lookups).
        analysis: The analysis result.
        context: The encoding context.

    IMPLEMENTATION: Use match-case on defn type, call the corresponding helper.
    Do NOT handle PartDef here — it is processed inside _encode_partition.
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
            _encode_func_image(ref, defn, context)

        case ir_obj.FuncInverseImage():
            _encode_func_inverse_image(ref, defn, context)

        # ── Sequences ─────────────────────────────────────────────────────────
        case ir_obj.SequenceDef() | ir_obj.CircleDef():
            _encode_sequence(ref, defn, analysis, context)

        # ── Partitions ────────────────────────────────────────────────────────
        case ir_obj.PartitionDef() | ir_obj.CompositionDef():
            _encode_partition(ref, defn, analysis, context)

        case ir_obj.SetPartDef() | ir_obj.BagPartDef():
            _encode_part_ref(ref, defn, context)

        case _:
            raise NotImplementedError(f"Unhandled object type {type(defn).__name__} for ref {ref}")


# =============================================================================
# Set encoders
# =============================================================================


def _encode_set_init(
    ref: ObjRef,
    defn: ir_obj.SetInit,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a SetInit node.

    A SetInit is a base set defined by explicit entities: e.g. A = {a, b, c}.

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
        context: Context.
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
    context: Context,
) -> None:
    """Encode a SetChoose node (subset of known size or any size).

    A SetChoose B picks a subset of B (without replacement).
    - If defn.size is set: the chosen subset has exactly that size.
    - The set_info for ref contains p_entities and max_size.

    Args:
        ref: ObjRef.
        defn: SetChoose dataclass (source: ObjRef, size: int | None).
        analysis: AnalysisResult.
        context: Context.
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
    context: Context,
) -> None:
    """Encode a SetChooseReplace node (multiset chosen with replacement).

    SetChooseReplace picks elements from a source set WITH replacement,
    producing a multiset. Encoded like a bag: uses dis_entities and
    indis_entities from bag_info (SetChooseReplace has bag_info, not set_info).

    Note: analysis.bag_info[ref] is populated for SetChooseReplace by EntityAnalysis.

    Args:
        ref: ObjRef.
        defn: SetChooseReplace dataclass (source: ObjRef, size: int | None).
        analysis: AnalysisResult.
        context: Context.
    """
    obj_pred = context.get_pred(ref, create=True, use=False)
    from_pred = context.get_pred(defn.source)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
    )
    bag_info = analysis.bag_info[ref]

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
        entity_var = context.get_entity_var(ref, entity)
        context.weighting[bag_entity_pred] = (
            reduce(lambda x, y: x + entity_var**y, range(1, multiplicity + 1), 0),
            1,
        )

    if defn.size is not None:
        size_expr = _get_bag_size_expr(ref, analysis, context)
        context.validator.append(Eq(size_expr, defn.size))


def _encode_set_union(
    ref: ObjRef,
    defn: ir_obj.SetUnion,
    context: Context,
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
    context: Context,
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
    context: Context,
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
def _encode_bag_init(
    ref: ObjRef,
    defn: ir_obj.BagInit,
    analysis: AnalysisResult,
    context: Context,
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
    context: Context,
) -> None:
    """Encode a BagChoose node (sub-bag chosen from source)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    from_pred = context.get_pred(defn.source)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
    )
    bag_info = analysis.bag_info[ref]
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
    for entity in bag_info.dis_entities:
        if entity in context.singletons:
            continue
        entity_var = context.get_entity_var(ref, entity)
        source_mul = _bag_entity_expr(defn.source, entity, analysis, context)
        context.validator.append(entity_var <= source_mul)

    # Enforce the exact bag size when choose(bag, k) was written.
    # Uses the full size expression (entity vars + singletons + indis) so
    # that singletons are counted correctly via bag_singletons_pred.
    if defn.size is not None:
        size_expr = _get_bag_size_expr(ref, analysis, context)
        context.validator.append(Eq(size_expr, defn.size))


def _encode_bag_union(
    ref: ObjRef,
    defn: ir_obj.BagUnion,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagUnion node: ref = max(left, right) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) | {right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
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
        left_mul = _bag_entity_expr(defn.left, entity, analysis, context)
        right_mul = _bag_entity_expr(defn.right, entity, analysis, context)
        context.validator.append(Eq(entity_var, Max(left_mul, right_mul)))


def _encode_bag_additive_union(
    ref: ObjRef,
    defn: ir_obj.BagAdditiveUnion,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagAdditiveUnion node: ref = left + right (sum multiplicities)."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) | {right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
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
        left_mul = _bag_entity_expr(defn.left, entity, analysis, context)
        right_mul = _bag_entity_expr(defn.right, entity, analysis, context)
        context.validator.append(Eq(entity_var, left_mul + right_mul))


def _encode_bag_intersection(
    ref: ObjRef,
    defn: ir_obj.BagIntersection,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagIntersection node: ref = min(left, right) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    right_pred = context.get_pred(defn.right)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) <-> ({left_pred}(X) & {right_pred}(X)))"
    )
    bag_info = analysis.bag_info[ref]
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
        left_mul = _bag_entity_expr(defn.left, entity, analysis, context)
        right_mul = _bag_entity_expr(defn.right, entity, analysis, context)
        context.validator.append(Eq(entity_var, Min(left_mul, right_mul)))


def _encode_bag_difference(
    ref: ObjRef,
    defn: ir_obj.BagDifference,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagDifference node: ref = max(left - right, 0) per entity."""
    obj_pred = context.get_pred(ref, create=True, use=False)
    left_pred = context.get_pred(defn.left)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({obj_pred}(X) -> {left_pred}(X))"
    )
    bag_info = analysis.bag_info[ref]
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
        left_mul = _bag_entity_expr(defn.left, entity, analysis, context)
        right_mul = _bag_entity_expr(defn.right, entity, analysis, context)
        context.validator.append(Eq(entity_var, Max(left_mul - right_mul, 0)))


def _encode_bag_support(
    ref: ObjRef,
    defn: ir_obj.BagSupport,
    context: Context,
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
    context: Context,
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
    # Set up size variable for the function.
    # domain_size: use get_size_expr so that when the domain has a known
    # exact_size (e.g. SetInit, or SetChoose with explicit size) we get a plain
    # integer instead of creating an extra symbolic variable.  The constraint
    # Eq(obj_var, k) is then a simple constant comparison rather than a
    # symbolic equality, which reduces the number of polynomial variables the
    # WFOMC solver needs to track.
    obj_var = context.get_obj_var(ref)
    domain_size = context.get_size_expr(defn.domain)
    context.validator.append(Eq(obj_var, domain_size))
    # If surjective, add surjectivity sentence
    if defn.surjective:
        context.sentence = context.sentence & parse(
            f"\\forall Y: (\\exists X: ({codomain_pred}(Y) -> {obj_pred}(X, Y)))"
        )


def _encode_func_inverse(
    ref: ObjRef,
    defn: ir_obj.FuncInverse,
    context: Context,
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
    context: Context,
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
    context: Context,
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
    analysis: AnalysisResult,
    context: Context,
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
    is_circle = isinstance(defn, ir_obj.CircleDef)
    logger.debug(
        "_encode_sequence: ref={}, source={}, size={}, circular={}, reflection={}",
        ref.id,
        defn.source.id,
        defn.size,
        is_circle,
        is_circle and defn.reflection,
    )
    domain_size = len(context.domain)
    # Check if source is a set or bag
    set_info = analysis.set_info.get(defn.source)
    bag_info = analysis.bag_info.get(defn.source)

    if set_info is not None and defn.flatten is not None:
        logger.debug(
            "_encode_sequence: source is a Set with flatten, p_entities={}, flatten={}",
            len(set_info.p_entities),
            defn.flatten.id,
        )
        # Set source with flatten (choose-with-replacement): use position-index entities
        # as domain. Entity predicates label positions; no multiplicity constraints.
        flatten_pred = context.get_pred(defn.flatten)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({flatten_pred}(X) & ~{flatten_pred}(Y)) -> {context.leq_pred}(X,Y)))"
        )
        entity_preds = []
        for entity in sorted(set_info.p_entities, key=lambda e: e.name):
            entity_pred = context.get_entity_pred(ref, entity)
            entity_preds.append(entity_pred)
        if entity_preds:
            or_formula = " | ".join(f"{pred}(X)" for pred in entity_preds)
            context.sentence = context.sentence & parse(
                f"\\forall X: (({or_formula}) <-> {flatten_pred}(X))"
            )
            context.sentence = context.sentence & exclusive(entity_preds)
        # Overcount correction: idx positions can be permuted freely (factorial(size))
        # plus non-flatten elements have factorial(domain_size - size) orderings
        if defn.size is not None:
            context.overcount = (
                context.overcount
                * math.factorial(domain_size - defn.size)
                * math.factorial(defn.size)
            )
    elif set_info is not None:
        logger.debug("_encode_sequence: source is a Set")
        # Sequence from a Set (no flatten)
        source_pred = context.get_pred(defn.source)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({source_pred}(X) & ~{source_pred}(Y)) -> {context.leq_pred}(X,Y)))"
        )
        # Overcount correction: (domain_size - size)!
        if defn.size is not None:
            context.overcount = context.overcount * math.factorial(domain_size - defn.size)
    elif bag_info is not None:
        logger.debug(
            "_encode_sequence: source is a Bag, dis_entities={}, flatten={}",
            len(bag_info.dis_entities),
            defn.flatten.id if defn.flatten is not None else None,
        )
        if defn.flatten is not None:
            # Bag sequence with flatten: use position-index entities as domain.
            # All dis_entities get entity predicates (no singleton bypass) and
            # cover the flatten domain instead of the source bag predicate.
            flatten_pred = context.get_pred(defn.flatten)
            context.sentence = context.sentence & parse(
                f"\\forall X: (\\forall Y: (({flatten_pred}(X) & ~{flatten_pred}(Y)) -> {context.leq_pred}(X,Y)))"
            )
            entity_preds = []
            for entity in bag_info.dis_entities:
                entity_pred = context.get_entity_pred(ref, entity)
                entity_preds.append(entity_pred)
                entity_var = context.get_entity_var(ref, entity)
                context.weighting[entity_pred] = (entity_var, 1)
                source_mul = _bag_entity_expr(defn.source, entity, analysis, context)
                context.validator.append(Eq(entity_var, source_mul))
            if entity_preds:
                or_formula = " | ".join(f"{pred}(X)" for pred in entity_preds)
                context.sentence = context.sentence & parse(
                    f"\\forall X: (({or_formula}) <-> {flatten_pred}(X))"
                )
                context.sentence = context.sentence & exclusive(entity_preds)
        else:
            # Bag sequence without flatten: use source bag predicate as domain.
            source_pred = context.get_pred(defn.source)
            context.sentence = context.sentence & parse(
                f"\\forall X: (\\forall Y: (({source_pred}(X) & ~{source_pred}(Y)) -> {context.leq_pred}(X,Y)))"
            )
            # Create entity predicates for each distinguishable entity
            entity_preds = []
            singleton_entity_preds = []
            for entity in bag_info.dis_entities:
                if entity in context.singletons:
                    # Singletons are in the source bag but handled via unary evidence.
                    # Include their entity predicate in the covering formula so the
                    # equivalence holds for singleton domain elements too.
                    singleton_pred = _encode_entity_in_ctx(entity, context)
                    singleton_entity_preds.append(singleton_pred)
                    continue
                entity_pred = context.get_entity_pred(ref, entity)
                entity_preds.append(entity_pred)
                entity_var = context.get_entity_var(ref, entity)
                context.weighting[entity_pred] = (entity_var, 1)
                source_mul = _bag_entity_expr(defn.source, entity, analysis, context)
                context.validator.append(Eq(entity_var, source_mul))
            # Exclusive covering: all non-singleton entity preds + singleton entity preds
            # cover source_pred exactly (no overlap allowed between non-singleton preds).
            all_covering_preds = entity_preds + singleton_entity_preds
            if all_covering_preds:
                or_formula = " | ".join(f"{pred}(X)" for pred in all_covering_preds)
                context.sentence = context.sentence & parse(
                    f"\\forall X: (({or_formula}) <-> {source_pred}(X))"
                )
            if entity_preds:
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
    if isinstance(defn, ir_obj.CircleDef):
        source_exact: int | None = None
        if set_info is not None and set_info.exact_size is not None:
            source_exact = set_info.exact_size
        elif bag_info is not None and bag_info.exact_size is not None:
            source_exact = bag_info.exact_size
        circle_size = defn.size or source_exact or domain_size
        context.circle_len = circle_size
        context.overcount = context.overcount * circle_size
        if defn.reflection:
            context.overcount = context.overcount * 2


# =============================================================================
# Partition encoder
# =============================================================================


def _encode_part_ref(
    ref: ObjRef,
    defn: ir_obj.PartDef,
    context: Context,
) -> None:
    """PartDef: predicate and entity vars already registered by _encode_partition.

    _encode_partition runs for defn.partition before this node is reached
    (PartitionDef precedes PartDef in topological order) and has already called:
      context.get_pred(ref, create=True)       # main predicate
      context.get_entity_var(ref, entity)      # per-entity vars (bag partitions)
    Nothing additional is needed here.
    """
    context.get_pred(ref, use=False)  # defensive: verify predicate was created


def _encode_partition(
    ref: ObjRef,
    defn: ir_obj.PartitionDef,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a PartitionDef node (set partition or bag partition, ordered or not).

    Parts: obtained via context.get_parts_of(ref) → list[ObjRef] of PartDef nodes.
    """
    is_composition = isinstance(defn, ir_obj.CompositionDef)
    logger.debug(
        "_encode_partition: ref={}, source={}, ordered={}",
        ref.id,
        defn.source.id,
        is_composition,
    )
    parts = context.get_parts_of(ref)
    source_pred = context.get_pred(defn.source)
    part_preds = [context.get_pred(p, create=True) for p in parts]

    # Coverage: source(X) <-> (part1(X) | part2(X) | ... | partN(X))
    or_formula = " | ".join(f"{pred}(X)" for pred in part_preds)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({source_pred}(X) <-> ({or_formula}))"
    )
    if defn.num_parts <= 1:
        return

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
        if not is_composition:
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
        # For singletons, add exactly-one constraint
        if context.singletons and context.singletons_pred is not None:
            exactly_one_formula = exactly_one_qf(part_preds)
            context.sentence = context.sentence & parse(
                f"\\forall X: (({context.singletons_pred}(X) & {source_pred}(X)) -> ({exactly_one_formula}))"
            )

        # Entity multiplicity partitioning (singletons included: multi=1 → weighting=(var,1))
        ordered_vars = [[] for _ in range(len(parts))]
        for entity in bag_info.dis_entities:
            multi = bag_info.p_entities_multiplicity[entity]
            entity_pred = _encode_entity_in_ctx(entity, context)

            multi_var = _bag_entity_expr(defn.source, entity, analysis, context)

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
                if not is_composition:
                    ordered_vars[idx].append(var)

            context.validator.append(Eq(sum(partitioned_vars), multi_var))

        # Symmetry breaking for unordered partitions
        if not is_composition:
            for i in range(len(ordered_vars) - 1):
                context.validator.append(ListLessThan(ordered_vars[i], ordered_vars[i + 1]))


# =============================================================================
