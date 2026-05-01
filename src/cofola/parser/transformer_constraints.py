"""Constraint transformer mixin for Cofola parser - IR-native version."""
from __future__ import annotations

from typing import TYPE_CHECKING

from cofola.frontend.constraints import (
    AndConstraint,
    BagEqConstraint,
    BagSubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    ForAllParts,
    LessThanPattern,
    MembershipConstraint,
    NextToPattern,
    NotConstraint,
    OrConstraint,
    PredecessorPattern,
    SequencePatternConstraint,
    SizeConstraint,
    SubsetConstraint,
    TogetherPattern,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import BagObjDef, SequenceDef
from cofola.frontend.types import Entity, ObjRef
from cofola.parser.constants import TupleIndexSentinel
from cofola.parser.errors import CofolaParsingError

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ConstraintTransformerMixin:
    """Mixin providing constraint transformation methods for CofolaTransformer (IR-native)."""

    def size_constraint(self: "CofolaTransfomer", args):
        expr, comparator, param = args
        return SizeConstraint(terms=tuple(expr), comparator=comparator, rhs=int(param))

    def size_atom(self: "CofolaTransfomer", args):
        if len(args) == 1:
            coef, obj = 1, args[0]
        else:
            coef, obj = int(args[0]), args[1]
        return coef, obj

    def size_atomic_expr(self: "CofolaTransfomer", args):
        coef, obj = args[0]
        return [(obj, int(coef))]

    def size_add(self: "CofolaTransfomer", args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, coef))
        return expr

    def size_sub(self: "CofolaTransfomer", args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, -coef))
        return expr

    def in_or_not(self: "CofolaTransfomer", args):
        return True if len(args) == 1 else False

    def membership_constraint(self: "CofolaTransfomer", args):
        entity_or_index, in_or_not, obj = args
        if isinstance(entity_or_index, TupleIndexSentinel):
            # T[i] in S → TupleIndexMembership
            return TupleIndexMembership(
                tuple_ref=entity_or_index.tuple_ref,
                index=entity_or_index.index,
                container=obj,
                positive=in_or_not,
            )
        if isinstance(entity_or_index, ObjRef):
            # P[0] in S (tuple from PartRef → resolves to PartRef) → SubsetConstraint
            return SubsetConstraint(sub=entity_or_index, sup=obj, positive=in_or_not)
        if isinstance(entity_or_index, Entity):
            return MembershipConstraint(
                entity=entity_or_index, container=obj, positive=in_or_not
            )
        raise CofolaParsingError(
            f"membership_constraint: unsupported LHS type "
            f"{type(entity_or_index).__name__}"
        )

    def subset_constraint(self: "CofolaTransfomer", args):
        obj1, _, obj2 = args
        d1 = self._defn_of(obj1)
        d2 = self._defn_of(obj2)
        if isinstance(d1, BagObjDef) and isinstance(d2, BagObjDef):
            return BagSubsetConstraint(sub=obj1, sup=obj2)
        return SubsetConstraint(sub=obj1, sup=obj2)

    def disjoint_constraint(self: "CofolaTransfomer", args):
        obj1, _, obj2 = args
        return DisjointConstraint(left=obj1, right=obj2)

    def equivalence_constraint(self: "CofolaTransfomer", args):
        o1, symbol, o2 = args
        symbol = str(symbol)
        positive = symbol == "=="

        if isinstance(o1, TupleIndexSentinel) and isinstance(o2, Entity):
            return TupleIndexEq(
                tuple_ref=o1.tuple_ref,
                index=o1.index,
                entity=o2,
                positive=positive,
            )
        if isinstance(o1, ObjRef) and isinstance(o2, Entity):
            # e.g. P[0] == e1 (tuple from PartRef source): membership constraint
            return MembershipConstraint(entity=o2, container=o1, positive=positive)
        if isinstance(o1, ObjRef) and isinstance(o2, ObjRef):
            d1 = self._defn_of(o1)
            d2 = self._defn_of(o2)
            if isinstance(d1, BagObjDef) and isinstance(d2, BagObjDef):
                return BagEqConstraint(left=o1, right=o2, positive=positive)
            return EqualityConstraint(left=o1, right=o2, positive=positive)
        raise CofolaParsingError(
            f"equivalence_constraint: unsupported types "
            f"({type(o1).__name__} {symbol} {type(o2).__name__})"
        )

    def count_parameter(self: "CofolaTransfomer", args):
        count = int(args[0])
        if count < 0:
            raise CofolaParsingError("Count parameter must be non-negative.")
        return count

    def seq_constraint(self: "CofolaTransfomer", args):
        pattern, is_in, obj = args
        defn = self._defn_of(obj)
        if not isinstance(defn, SequenceDef):
            kind = type(defn).__name__ if defn is not None else "unknown"
            raise CofolaParsingError(f"seq_constraint requires a Sequence, got {kind}")
        return SequencePatternConstraint(seq=obj, pattern=pattern, positive=is_in)

    def seq_pattern(self: "CofolaTransfomer", args):
        # Pattern objects are IR dataclasses — return directly, not added to builder
        return args[0]

    def together(self: "CofolaTransfomer", args):
        obj = args[2]
        return TogetherPattern(group=obj)

    def less_than(self: "CofolaTransfomer", args):
        entity_or_obj1, _, entity_or_obj2 = args
        return LessThanPattern(left=entity_or_obj1, right=entity_or_obj2)

    def next_to(self: "CofolaTransfomer", args):
        _, _, entity_or_obj1, entity_or_obj2, _ = args
        return NextToPattern(first=entity_or_obj1, second=entity_or_obj2)

    def predecessor(self: "CofolaTransfomer", args):
        _, entity_or_obj1, entity_or_obj2, _ = args
        return PredecessorPattern(first=entity_or_obj1, second=entity_or_obj2)

    def negation_constraint(self: "CofolaTransfomer", args):
        constraint = args[1]
        return NotConstraint(sub=constraint)

    def binary_constraint(self: "CofolaTransfomer", args):
        arg1, op, arg2 = args
        op = str(op)
        if op == "and":
            return AndConstraint(left=arg1, right=arg2)
        if op == "or":
            return OrConstraint(left=arg1, right=arg2)
        raise CofolaParsingError(f"Unknown binary constraint operator: {op}")

    def part_constraint(self: "CofolaTransfomer", args):
        constraint_template, _, _part_name_token, _, partition_ref = args
        return ForAllParts(partition=partition_ref, constraint_template=constraint_template)
