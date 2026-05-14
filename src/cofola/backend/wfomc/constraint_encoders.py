"""WFOMC constraint encoders."""
from __future__ import annotations

from sympy import Eq, Ge, Gt, Le, Lt, Ne, Or, false
from wfomc import Const, fol_parse as parse

import cofola.frontend.constraints as ir_cst
import cofola.frontend.objects as ir_obj
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.encoding_helpers import (
    _bag_entity_expr,
    _encode_entity_in_ctx,
    _get_bag_count_var,
    _get_bag_size_expr,
)
from cofola.backend.wfomc.utils import create_aux_pred
from cofola.frontend.objects import Entity, ObjRef
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from loguru import logger


# Constraint encoders
# =============================================================================


def _encode_constraint(
    c: object,
    problem: Problem,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Dispatch to the appropriate constraint encoder.

    Args:
        c: The frontend constraint dataclass.
        problem: Problem.
        analysis: AnalysisResult.
        context: Context.
    """
    logger.debug("Encoding constraint {}", type(c).__name__)
    match c:
        case ir_cst.SizeConstraint():
            _encode_size_constraint(c, analysis, context)

        case ir_cst.MembershipConstraint():
            _encode_membership_constraint(c, context)

        case ir_cst.SubsetConstraint():
            _encode_subset_constraint(c, context)

        case ir_cst.DisjointConstraint():
            _encode_disjoint_constraint(c, context)

        case ir_cst.EqualityConstraint():
            _encode_equality_constraint(c, context)

        case ir_cst.FuncPairConstraint():
            _encode_func_pair_constraint(c, context)

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
            raise NotImplementedError(f"Unhandled constraint type {type(c).__name__}")


def _encode_size_constraint(
    c: ir_cst.SizeConstraint,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a SizeConstraint."""
    # Build the left-hand expression from terms
    left = 0
    for term, coef in c.terms:
        if isinstance(term, ObjRef):
            # Bags need a multi-variable size expression (sum of per-entity vars +
            # singleton contribution); sets/partitions use a single obj_var that
            # counts membership in the predicate.
            if term in analysis.bag_info:
                var = _get_bag_size_expr(term, analysis, context)
            else:
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
        "!=": Ne,
    }
    comparator_fn = comparator_map.get(c.comparator)
    if comparator_fn is None:
        raise ValueError(f"Unknown comparator: {c.comparator}")
    context.validator.append(comparator_fn(left, c.rhs))


def _encode_membership_constraint(
    c: ir_cst.MembershipConstraint,
    context: Context,
) -> None:
    """Encode a MembershipConstraint: entity [not] in container."""
    obj_pred = context.get_pred(c.container)
    if c.positive:
        context.unary_evidence.add(obj_pred(Const(c.entity.name)))
    else:
        context.unary_evidence.add(~obj_pred(Const(c.entity.name)))


def _encode_subset_constraint(
    c: ir_cst.SubsetConstraint,
    context: Context,
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
    context: Context,
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
    context: Context,
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
    context: Context,
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


def _get_seq_entity_pred(
    entity_or_ref: Entity | ObjRef,
    seq_ref: ObjRef,
    context: Context,
) -> object:
    """Return the predicate for an entity/ref in the context of a sequence.

    For sequences with a flatten object (Bag sources or choose-with-replacement),
    domain elements are position-index entities:
    - Entity  → entity_pred(seq_ref, entity)  labels positions by type
    - ObjRef  → aux pred = OR of entity_preds for entities in that set

    For plain Set sequences (no flatten), domain elements are original entities:
    - Entity  → _encode_entity_in_ctx  (unary evidence pinning)
    - ObjRef  → get_pred(ref)  (the set predicate)
    """
    seq_defn = context.problem.get_object(seq_ref)
    has_flatten = seq_defn is not None and seq_defn.flatten is not None

    if isinstance(entity_or_ref, Entity):
        if has_flatten:
            return context.get_entity_pred(seq_ref, entity_or_ref)
        return _encode_entity_in_ctx(entity_or_ref, context)

    # ObjRef (a set/bag reference)
    if has_flatten:
        return _build_group_type_pred_for_seq(entity_or_ref, seq_ref, context)
    return context.get_pred(entity_or_ref)


def _build_group_type_pred_for_seq(
    group_ref: ObjRef,
    seq_ref: ObjRef,
    context: Context,
) -> object:
    """Build an auxiliary predicate true for position X if its entity type is in group.

    Used when encoding patterns in flatten-based sequences (bag source or
    choose-with-replacement). entity_pred(seq_ref, e) labels position entities
    by type, so we OR those together for entities in group_ref.

    Only includes entities whose entity predicate was registered during
    _encode_sequence; entities not in the sequence source are skipped.
    """
    group_defn = context.problem.get_object(group_ref)
    if isinstance(group_defn, ir_obj.SetInit):
        group_entities: frozenset = group_defn.entities
    else:
        group_set_info = context.analysis.set_info.get(group_ref)
        group_entities = group_set_info.p_entities if group_set_info else frozenset()

    # Only reference entity preds already created by _encode_sequence.
    type_preds = [
        context.ref_entity2pred[(seq_ref, e)]
        for e in sorted(group_entities, key=lambda e: e.name)
        if (seq_ref, e) in context.ref_entity2pred
    ]
    if not type_preds:
        false_pred = create_aux_pred(1)
        for e in context.analysis.all_entities:
            context.unary_evidence.add(~false_pred(Const(e.name)))
        return false_pred

    name = context.problem.get_name(group_ref) or f"obj_{group_ref.id}"
    group_type_pred = create_aux_pred(1, f"{name}_type_in_seq")
    or_formula = " | ".join(f"{p}(X)" for p in type_preds)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({group_type_pred}(X) <-> ({or_formula}))"
    )
    return group_type_pred


def _encode_sequence_pattern_constraint(
    c: ir_cst.SequencePatternConstraint,
    context: Context,
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
        case _:
            raise TypeError(f"Unknown sequence pattern type: {type(c.pattern).__name__}")


def _encode_bag_subset_constraint(
    c: ir_cst.BagSubsetConstraint,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagSubsetConstraint: sub ⊆ sup (by multiplicity)."""
    sub_info = analysis.bag_info.get(c.sub)
    sup_info = analysis.bag_info.get(c.sup)
    if sub_info is None or sup_info is None:
        raise TypeError("BagSubsetConstraint requires both sides to be bag-like")

    entities = set(sub_info.p_entities_multiplicity) | set(sup_info.p_entities_multiplicity)
    comparisons = []
    for entity in entities:
        if entity in context.singletons:
            continue
        sub_var = _bag_entity_expr(c.sub, entity, analysis, context)
        sup_var = _bag_entity_expr(c.sup, entity, analysis, context)
        if c.positive:
            context.validator.append(sub_var <= sup_var)
        else:
            comparisons.append(sub_var > sup_var)
    if not c.positive:
        context.validator.append(Or(*comparisons) if comparisons else false)


def _encode_bag_eq_constraint(
    c: ir_cst.BagEqConstraint,
    analysis: AnalysisResult,
    context: Context,
) -> None:
    """Encode a BagEqConstraint: left == right (by multiplicity)."""
    left_info = analysis.bag_info.get(c.left)
    right_info = analysis.bag_info.get(c.right)
    if left_info is None or right_info is None:
        raise TypeError("BagEqConstraint requires both sides to be bag-like")

    # Get all entities from both bags
    all_entities = set(left_info.p_entities_multiplicity) | set(right_info.p_entities_multiplicity)
    comparisons = []
    for entity in all_entities:
        if entity in context.singletons:
            continue
        left_var = _bag_entity_expr(c.left, entity, analysis, context)
        right_var = _bag_entity_expr(c.right, entity, analysis, context)
        if c.positive:
            context.validator.append(Eq(left_var, right_var))
        else:
            comparisons.append(Ne(left_var, right_var))
    if not c.positive:
        context.validator.append(Or(*comparisons) if comparisons else false)


def _encode_tuple_index_eq(
    c: ir_cst.TupleIndexEq,
    context: Context,
) -> None:
    """Encode a TupleIndexEq: T[index] [!=] entity.

    LoweringPass should rewrite tuple index constraints into FuncPairConstraint
    before backend encoding. Reaching this encoder is a pipeline invariant error.

    Args:
        c: TupleIndexEq dataclass.
        context: Context.
    """
    raise NotImplementedError(
        "TupleIndexEq reached encoder; TupleDef should have been lowered"
    )


def _encode_tuple_index_membership(
    c: ir_cst.TupleIndexMembership,
    context: Context,
) -> None:
    """Encode a TupleIndexMembership: T[index] [not] in container.

    LoweringPass should rewrite tuple membership constraints before backend encoding.

    Args:
        c: TupleIndexMembership dataclass.
        context: Context.
    """
    raise NotImplementedError(
        "TupleIndexMembership reached encoder; TupleDef should have been lowered"
    )


# =============================================================================
# Sequence pattern sub-encoders
# =============================================================================


def _encode_together_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.TogetherPattern,
    positive: bool,
    context: Context,
) -> None:
    """Encode a TogetherPattern: elements of group appear consecutively in seq."""
    group_ref = pattern.group

    # Get predicate for the group (handles flatten correctly)
    obj_pred = _get_seq_entity_pred(group_ref, seq_ref, context)

    # Get predecessor predicate and domain predicate for the sequence.
    # domain_pred restricts X to positions that actually belong to the sequence:
    # flatten_pred when the sequence has a flatten (bag source / choose-replace),
    # otherwise the source set/bag predicate.  Without this restriction, entities
    # in the group that are not in the sequence source would be counted as "first"
    # elements (pred_pred(Y,X) is vacuously False for non-source X, so the
    # ∀Y-quantifier is trivially satisfied).
    pred_pred = context.get_predecessor_pred(seq_ref)
    seq_defn = context.problem.get_object(seq_ref)
    domain_pred = context.get_pred(
        seq_defn.flatten if seq_defn.flatten is not None else seq_defn.source
    )

    # Create "first" predicate
    name = context.problem.get_name(group_ref) or f"obj_{group_ref.id}"
    first_pred = create_aux_pred(1, f"{name}_first")

    # Define first(X) as the first element of the group in the sequence.
    # source_pred(Y) is omitted from the ∀Y-guard because pred_pred already
    # restricts both arguments to the sequence domain.
    context.sentence = context.sentence & parse(
        f"\\forall X: ({first_pred}(X) <-> ({obj_pred}(X) & {domain_pred}(X) & "
        f"(\\forall Y: ({obj_pred}(Y) -> ~{pred_pred}(Y,X)))))"
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
    context: Context,
) -> None:
    """Encode a LessThanPattern: left appears before right in seq."""
    _encode_sequence_relation_universal(
        seq_ref,
        pattern.left,
        pattern.right,
        context.get_leq_pred(seq_ref),
        positive,
        context,
    )


def _encode_predecessor_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.PredecessorPattern,
    positive: bool,
    context: Context,
) -> None:
    """Encode a PredecessorPattern: first immediately precedes second in seq."""
    _encode_sequence_relation_universal(
        seq_ref,
        pattern.first,
        pattern.second,
        context.get_predecessor_pred(seq_ref),
        positive,
        context,
    )


def _encode_next_to_pattern(
    seq_ref: ObjRef,
    pattern: ir_cst.NextToPattern,
    positive: bool,
    context: Context,
) -> None:
    """Encode a NextToPattern: first and second are adjacent in seq."""
    _encode_sequence_relation_universal(
        seq_ref,
        pattern.first,
        pattern.second,
        context.get_next_to_pred(seq_ref),
        positive,
        context,
    )


# =============================================================================

def _encode_sequence_relation_universal(
    seq_ref: ObjRef,
    left: Entity | ObjRef,
    right: Entity | ObjRef,
    relation_pred: object,
    positive: bool,
    context: Context,
) -> None:
    """Encode an asserted sequence relation with universal pattern semantics.

    Pattern constraints such as ``A < B in seq`` mean every matching left
    occurrence stands in the relation to every matching right occurrence.
    Negative constraints mean the pattern has no occurrences, so every matching
    pair must fail the relation.
    """
    left_pred = _get_seq_entity_pred(left, seq_ref, context)
    right_pred = _get_seq_entity_pred(right, seq_ref, context)
    if positive:
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({left_pred}(X) & {right_pred}(Y)) -> "
            f"{relation_pred}(X,Y)))"
        )
    else:
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({left_pred}(X) & {right_pred}(Y)) -> "
            f"~{relation_pred}(X,Y)))"
        )


def _encode_sequence_relation_count(
    seq_ref: ObjRef,
    left: Entity | ObjRef,
    right: Entity | ObjRef,
    relation_name: str,
    relation_pred: object,
    context: Context,
) -> object:
    """Encode a binary sequence relation and return its counting variable."""
    left_pred = _get_seq_entity_pred(left, seq_ref, context)
    right_pred = _get_seq_entity_pred(right, seq_ref, context)
    left_name = context.problem.get_name(left) if isinstance(left, ObjRef) else str(left)
    right_name = context.problem.get_name(right) if isinstance(right, ObjRef) else str(right)
    pair_pred = create_aux_pred(2, f"{left_name}_{relation_name}_{right_name}")
    context.sentence = context.sentence & parse(
        f"\\forall X: (\\forall Y: (({left_pred}(X) & {right_pred}(Y) & "
        f"{relation_pred}(X,Y)) <-> {pair_pred}(X,Y)))"
    )
    pair_var = context.create_var(pair_pred.name)
    context.weighting[pair_pred] = (pair_var, 1)
    return pair_var


def _get_seq_pattern_count_var(
    atom: ir_cst.SeqPatternCountAtom,
    context: Context,
) -> object:
    """Get the symbolic variable for a SeqPatternCountAtom (S.count(pattern)).

    This creates a variable representing the count of the given pattern
    in the given sequence by encoding the pattern and extracting the count var.
    """
    seq_ref = atom.seq
    pattern = atom.pattern

    match pattern:
        case ir_cst.LessThanPattern():
            return _encode_sequence_relation_count(
                seq_ref,
                pattern.left,
                pattern.right,
                "leq",
                context.get_leq_pred(seq_ref),
                context,
            )

        case ir_cst.PredecessorPattern():
            return _encode_sequence_relation_count(
                seq_ref,
                pattern.first,
                pattern.second,
                "pred",
                context.get_predecessor_pred(seq_ref),
                context,
            )

        case ir_cst.NextToPattern():
            return _encode_sequence_relation_count(
                seq_ref,
                pattern.first,
                pattern.second,
                "next_to",
                context.get_next_to_pred(seq_ref),
                context,
            )

        case _:
            raise TypeError(f"Unknown pattern type: {type(pattern)}")
