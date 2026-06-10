"""WFOMC constraint encoders."""
from __future__ import annotations

from sympy import Eq, Ge, Gt, Le, Lt, Ne, Or, false
from wfomc import Const, fol_parse as parse

import cofola.frontend.constraints as ir_cst
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.encoding_helpers import (
    _bag_entity_expr,
    _encode_entity_in_ctx,
    _get_bag_count_var,
    _get_bag_size_expr,
)
from cofola.backend.wfomc.utils import create_aux_pred
from cofola.frontend.objects import CircleDef, Entity, ObjRef, SequenceDef
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


def _sequence_domain_pred(seq_ref: ObjRef, context: Context) -> object:
    seq_defn = context.problem.get_object(seq_ref)
    if not isinstance(seq_defn, (SequenceDef, CircleDef)):
        raise ValueError(f"Ref {seq_ref} is not a sequence-like object")
    return context.get_pred(
        seq_defn.flatten if seq_defn.flatten is not None else seq_defn.source
    )


def _sequence_has_flatten(seq_ref: ObjRef, context: Context) -> bool:
    seq_defn = context.problem.get_object(seq_ref)
    return isinstance(seq_defn, (SequenceDef, CircleDef)) and seq_defn.flatten is not None


def _possible_entities(ref: ObjRef, context: Context) -> tuple[Entity, ...]:
    set_info = context.analysis.set_info.get(ref)
    if set_info is not None:
        return tuple(sorted(set_info.p_entities, key=lambda e: e.name))
    bag_info = context.analysis.bag_info.get(ref)
    if bag_info is not None:
        return tuple(sorted(bag_info.p_entities_multiplicity, key=lambda e: e.name))
    return ()


def _arg_possible_entities(arg: Entity | ObjRef, context: Context) -> tuple[Entity, ...]:
    if isinstance(arg, Entity):
        return (arg,)
    return _possible_entities(arg, context)


def _constant_unary_pred(value: bool, context: Context) -> object:
    attr = "_cofola_true_pred" if value else "_cofola_false_pred"
    pred = getattr(context, attr, None)
    if pred is not None:
        return pred

    pred = create_aux_pred(1, "true" if value else "false")
    context.sentence = context.sentence & parse(
        f"\\forall X: ({'' if value else '~'}{pred}(X))"
    )
    setattr(context, attr, pred)
    return pred


def _seq_entity_value_pred(entity: Entity, seq_ref: ObjRef, context: Context) -> object:
    """Predicate for positions in seq whose value is exactly entity."""
    if entity not in _possible_entities(seq_ref, context):
        return _constant_unary_pred(False, context)
    if _sequence_has_flatten(seq_ref, context):
        return context.get_entity_pred(seq_ref, entity)
    return _encode_entity_in_ctx(entity, context)


def _arg_entity_active_pred(
    arg: Entity | ObjRef,
    entity: Entity,
    context: Context,
) -> object:
    """Unary predicate that is globally true iff entity is active in arg.

    For an entity literal this is a constant truth value.  For a set ref it
    reflects membership of that entity in the current value of the object.
    """
    if isinstance(arg, Entity):
        return _constant_unary_pred(arg == entity, context)

    if entity not in _possible_entities(arg, context):
        return _constant_unary_pred(False, context)

    arg_pred = context.get_pred(arg)
    entity_pred = _encode_entity_in_ctx(entity, context)
    name = context.problem.get_name(arg) or f"obj_{arg.id}"
    active_pred = create_aux_pred(1, f"{name}_{entity.name}_active")
    context.sentence = context.sentence & parse(
        f"\\forall X: ({active_pred}(X) <-> "
        f"(\\exists Y: ({entity_pred}(Y) & {arg_pred}(Y))))"
    )
    return active_pred


def _seq_arg_entity_occurrence_pred(
    arg: Entity | ObjRef,
    entity: Entity,
    seq_ref: ObjRef,
    context: Context,
) -> object:
    """Predicate for seq positions occupied by entity as contributed by arg."""
    if isinstance(arg, Entity) and arg != entity:
        return _constant_unary_pred(False, context)

    value_pred = _seq_entity_value_pred(entity, seq_ref, context)
    domain_pred = _sequence_domain_pred(seq_ref, context)
    if isinstance(arg, Entity):
        active_expr = ""
    elif _sequence_has_flatten(seq_ref, context):
        active_pred = _arg_entity_active_pred(arg, entity, context)
        active_expr = f" & {active_pred}(X)"
    else:
        arg_pred = context.get_pred(arg)
        active_expr = f" & {arg_pred}(X)"

    label = (
        arg.name if isinstance(arg, Entity)
        else context.problem.get_name(arg) or f"obj_{arg.id}"
    )
    occ_pred = create_aux_pred(1, f"{label}_{entity.name}_occ")
    context.sentence = context.sentence & parse(
        f"\\forall X: ({occ_pred}(X) <-> "
        f"({value_pred}(X) & {domain_pred}(X){active_expr}))"
    )
    return occ_pred


def _seq_arg_occurrence_pred(
    arg: Entity | ObjRef,
    seq_ref: ObjRef,
    context: Context,
) -> object:
    """Predicate for all seq positions matching an entity or set argument."""
    if isinstance(arg, ObjRef) and not _sequence_has_flatten(seq_ref, context):
        arg_pred = context.get_pred(arg)
        domain_pred = _sequence_domain_pred(seq_ref, context)
        label = context.problem.get_name(arg) or f"obj_{arg.id}"
        occ_pred = create_aux_pred(1, f"{label}_occ")
        context.sentence = context.sentence & parse(
            f"\\forall X: ({occ_pred}(X) <-> ({arg_pred}(X) & {domain_pred}(X)))"
        )
        return occ_pred

    occurrence_preds = [
        _seq_arg_entity_occurrence_pred(arg, entity, seq_ref, context)
        for entity in _arg_possible_entities(arg, context)
    ]
    if not occurrence_preds:
        return _constant_unary_pred(False, context)
    if len(occurrence_preds) == 1:
        return occurrence_preds[0]

    label = (
        arg.name if isinstance(arg, Entity)
        else context.problem.get_name(arg) or f"obj_{arg.id}"
    )
    occ_pred = create_aux_pred(1, f"{label}_occ")
    or_formula = " | ".join(f"{pred}(X)" for pred in occurrence_preds)
    context.sentence = context.sentence & parse(
        f"\\forall X: ({occ_pred}(X) <-> ({or_formula}))"
    )
    return occ_pred


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

    # Positions in this sequence whose value belongs to the group.
    group_occ_pred = _seq_arg_occurrence_pred(group_ref, seq_ref, context)
    pred_pred = context.get_predecessor_pred(seq_ref)

    # Create "first" predicate
    name = context.problem.get_name(group_ref) or f"obj_{group_ref.id}"
    first_pred = create_aux_pred(1, f"{name}_first")

    # Define first(X) as the first element of the group in the sequence.
    context.sentence = context.sentence & parse(
        f"\\forall X: ({first_pred}(X) <-> ({group_occ_pred}(X) & "
        f"(\\forall Y: ({group_occ_pred}(Y) -> ~{pred_pred}(Y,X)))))"
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
    _encode_sequence_relation_all_pairs(
        seq_ref,
        pattern.left,
        pattern.right,
        context.get_lt_pred(seq_ref),
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
    _encode_sequence_relation_left_total(
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
    _encode_sequence_relation_left_total(
        seq_ref,
        pattern.first,
        pattern.second,
        context.get_next_to_pred(seq_ref),
        positive,
        context,
    )


def _encode_sequence_relation_all_pairs(
    seq_ref: ObjRef,
    left: Entity | ObjRef,
    right: Entity | ObjRef,
    relation_pred: object,
    positive: bool,
    context: Context,
) -> None:
    """Encode a relation over every matching left/right occurrence pair."""
    left_pred = _seq_arg_occurrence_pred(left, seq_ref, context)
    right_pred = _seq_arg_occurrence_pred(right, seq_ref, context)
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


def _encode_sequence_relation_left_total(
    seq_ref: ObjRef,
    left: Entity | ObjRef,
    right: Entity | ObjRef,
    relation_pred: object,
    positive: bool,
    context: Context,
) -> None:
    """Encode Pred/NextTo pattern constraints.

    Positive predecessor/adjacency follows the denotational semantics:
    for each active entity pair, every left occurrence must have at least one
    matching right occurrence in the requested relation.  Negative constraints
    keep the language's existing "pattern absent" meaning and forbid all
    matching occurrence pairs.
    """
    if not positive:
        _encode_sequence_relation_all_pairs(
            seq_ref,
            left,
            right,
            relation_pred,
            positive=False,
            context=context,
        )
        return

    for left_entity in _arg_possible_entities(left, context):
        left_pred = _seq_arg_entity_occurrence_pred(
            left,
            left_entity,
            seq_ref,
            context,
        )
        for right_entity in _arg_possible_entities(right, context):
            right_active_pred = _arg_entity_active_pred(right, right_entity, context)
            right_pred = _seq_arg_entity_occurrence_pred(
                right,
                right_entity,
                seq_ref,
                context,
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: (({left_pred}(X) & {right_active_pred}(X)) -> "
                f"(\\exists Y: ({right_pred}(Y) & {relation_pred}(X,Y))))"
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
    left_pred = _seq_arg_occurrence_pred(left, seq_ref, context)
    right_pred = _seq_arg_occurrence_pred(right, seq_ref, context)
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
                "lt",
                context.get_lt_pred(seq_ref),
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
