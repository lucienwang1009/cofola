"""Shared helpers for WFOMC encoding."""
from __future__ import annotations

from sympy import Eq
from wfomc import Const, fol_parse as parse

import cofola.frontend.constraints as ir_cst
import cofola.frontend.objects as ir_obj
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.utils import create_aux_pred, create_cofola_pred
from cofola.frontend.objects import Entity, ObjRef
from cofola.planing.analysis.entities import AnalysisResult


# Entity encoding
# =============================================================================


def _encode_entities(analysis: AnalysisResult, context: Context) -> None:
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


def _encode_singleton(context: Context) -> None:
    """Create a predicate for the set of singleton entities."""
    singletons = context.singletons
    pred = create_aux_pred(1, "singletons")
    context.singletons_pred = pred
    for e in context.analysis.all_entities:
        if e in singletons:
            context.unary_evidence.add(pred(Const(e.name)))
        else:
            context.unary_evidence.add(~pred(Const(e.name)))


def _encode_entity_in_ctx(entity: Entity, context: Context) -> object:
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


# Size-atom variable helpers
# =============================================================================


def _get_bag_size_expr(
    ref: ObjRef,
    analysis: AnalysisResult,
    context: Context,
) -> object:
    """Compute the total size expression for a bag.

    Mirrors Bag.encode_size_var from the reference implementation.

    The total size of a bag is:
      - Sum of entity vars for non-singleton distinguishable entities
      - A singleton contribution counted via bag_singletons_pred (if singletons exist)
      - Sum of indistinguishable entity vars (usually empty for lifted=False)

    This must be called AFTER the bag object has been encoded so that entity
    vars for dis_entities already exist in context.ref_entity2var.

    Args:
        ref: ObjRef of the bag.
        analysis: AnalysisResult.
        context: Context.

    Returns:
        Sympy expression (or int 0) for the total bag multiplicity.
    """
    # Sum entity vars for non-singleton distinguishable entities.
    # These vars are created by _encode_bag_choose / _encode_bag_additive_union etc.
    entity_vars = context.get_entity_var(ref)  # dict[IREntity, Expr]
    term = sum(
        (var for e, var in entity_vars.items() if e not in context.singletons),
        0,
    )

    # Singleton contribution: count how many singleton entities are in this bag.
    if context.singletons and context.singletons_pred is not None:
        if ref not in context.ref2bag_singletons_pred:
            obj_pred = context.get_pred(ref)
            singletons_pred = context.singletons_pred
            bag_singletons_pred = context.create_pred(
                f"{context._get_name(ref)}_singletons", 1
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_singletons_pred}(X) <-> {obj_pred}(X) & {singletons_pred}(X))"
            )
            # set_weight=False: do NOT put a weight on the main obj_pred
            singleton_var = context.get_obj_var(ref, set_weight=False)
            context.weighting[bag_singletons_pred] = (singleton_var, 1)
            context.ref2bag_singletons_pred[ref] = bag_singletons_pred
        else:
            singleton_var = context.get_obj_var(ref, set_weight=False)
        term = term + singleton_var

    # Indistinguishable entity vars (always empty when lifted=False).
    indis_vars = context.get_indis_entity_var(ref)  # dict[int, Expr]
    if indis_vars:
        term = term + sum(indis_vars.values())

    return term


def _bag_entity_expr(
    bag_ref: ObjRef,
    entity: Entity,
    analysis: AnalysisResult,
    context: Context,
) -> object:
    """Return a bag entity multiplicity expression, or 0 if impossible."""
    bag_info = analysis.bag_info.get(bag_ref)
    if bag_info is None or entity not in bag_info.p_entities_multiplicity:
        return 0
    if isinstance(context.problem.get_object(bag_ref), ir_obj.BagInit):
        return bag_info.p_entities_multiplicity.get(entity, 0)
    return context.get_entity_var(bag_ref, entity)


def _get_bag_count_var(
    atom: ir_cst.BagCountAtom,
    context: Context,
) -> object:
    """Get the symbolic variable (or constant) for a BagCountAtom (B.count(e)).

    If the entity is not a possible element of the bag, the count is 0 and
    we return the integer 0 directly. Returning a fresh symbolic variable
    here would leak an unbound symbol into the validator (no weighting ever
    references it), which breaks decode_result.
    """
    return _bag_entity_expr(atom.bag, atom.entity, context.analysis, context)
