"""Constraint transformer mixin for Cofola parser."""
from __future__ import annotations

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
    TupleCountAtom,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import BagObjDef, ObjDef
from cofola.frontend.objects import Entity, ObjRef, TupleDef
from cofola.parser.utils import CofolaParsingError, TupleIndexSentinel


def _resolve_subset_constraint(
    *,
    sub: ObjRef,
    sup: ObjRef,
    sub_defn: ObjDef | None,
    sup_defn: ObjDef | None,
) -> BagSubsetConstraint | SubsetConstraint:
    if isinstance(sub_defn, BagObjDef) and isinstance(sup_defn, BagObjDef):
        return BagSubsetConstraint(sub=sub, sup=sup)
    return SubsetConstraint(sub=sub, sup=sup)


def _resolve_equivalence_constraint(
    *,
    left: object,
    symbol: str,
    right: object,
    left_defn: ObjDef | None,
    right_defn: ObjDef | None,
) -> TupleIndexEq | BagEqConstraint | EqualityConstraint:
    positive = symbol == "=="

    if isinstance(left, TupleIndexSentinel) and isinstance(right, Entity):
        return TupleIndexEq(
            tuple_ref=left.tuple_ref,
            index=left.index,
            entity=right,
            positive=positive,
        )
    if isinstance(left, ObjRef) and isinstance(right, ObjRef):
        if isinstance(left_defn, BagObjDef) and isinstance(right_defn, BagObjDef):
            return BagEqConstraint(left=left, right=right, positive=positive)
        return EqualityConstraint(left=left, right=right, positive=positive)
    raise CofolaParsingError(
        f"equivalence_constraint: unsupported types "
        f"({type(left).__name__} {symbol} {type(right).__name__})"
    )


def _resolve_membership_constraint(
    *,
    member: object,
    positive: bool,
    container: ObjRef,
    container_defn: ObjDef | None,
) -> TupleIndexMembership | SizeConstraint | MembershipConstraint:
    if isinstance(member, TupleIndexSentinel):
        return TupleIndexMembership(
            tuple_ref=member.tuple_ref,
            index=member.index,
            container=container,
            positive=positive,
        )
    if isinstance(container_defn, TupleDef) and isinstance(member, Entity):
        # Workaround: parse `a in T` for tuples as `T.count(a) > 0`.
        return SizeConstraint(
            terms=((TupleCountAtom(container, member, False), 1),),
            comparator=">" if positive else "==",
            rhs=0,
        )
    if isinstance(member, Entity):
        return MembershipConstraint(
            entity=member,
            container=container,
            positive=positive,
        )
    raise CofolaParsingError(
        f"membership_constraint: unsupported LHS type "
        f"{type(member).__name__}"
    )


class ConstraintTransformerMixin:
    """Mixin providing constraint transformation methods for CofolaTransformer."""

    def size_constraint(self, args):
        expr, comparator, param = args
        return SizeConstraint(terms=tuple(expr), comparator=str(comparator), rhs=int(param))

    def size_atom(self, args):
        if len(args) == 1:
            coef, obj = 1, args[0]
        else:
            coef, obj = int(args[0]), args[1]
        return coef, obj

    def size_atomic_expr(self, args):
        coef, obj = args[0]
        return [(obj, int(coef))]

    def size_add(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, coef))
        return expr

    def size_sub(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, -coef))
        return expr

    def in_or_not(self, args):
        return True if len(args) == 1 else False

    def membership_constraint(self, args):
        entity_or_index, in_or_not, obj = args
        return _resolve_membership_constraint(
            member=entity_or_index,
            positive=in_or_not,
            container=obj,
            container_defn=self._defn_of(obj),
        )

    def subset_constraint(self, args):
        obj1, _, obj2 = args
        return _resolve_subset_constraint(
            sub=obj1,
            sup=obj2,
            sub_defn=self._defn_of(obj1),
            sup_defn=self._defn_of(obj2),
        )

    def disjoint_constraint(self, args):
        obj1, _, obj2 = args
        return DisjointConstraint(left=obj1, right=obj2)

    def equivalence_constraint(self, args):
        o1, symbol, o2 = args
        symbol = str(symbol)
        return _resolve_equivalence_constraint(
            left=o1,
            symbol=symbol,
            right=o2,
            left_defn=self._defn_of(o1) if isinstance(o1, ObjRef) else None,
            right_defn=self._defn_of(o2) if isinstance(o2, ObjRef) else None,
        )

    def count_parameter(self, args):
        count = int(args[0])
        if count < 0:
            raise CofolaParsingError("Count parameter must be non-negative.")
        return count

    def seq_constraint(self, args):
        pattern, is_in, obj = args
        # TypeChecker enforces that `seq` is a Sequence/Circle via the
        # SequencePatternConstraint signature.
        return SequencePatternConstraint(seq=obj, pattern=pattern, positive=is_in)

    def seq_pattern(self, args):
        # Pattern objects are frontend dataclasses — return directly, not added to builder
        return args[0]

    def together(self, args):
        obj = args[2]
        return TogetherPattern(group=obj)

    def less_than(self, args):
        entity_or_obj1, _, entity_or_obj2 = args
        return LessThanPattern(left=entity_or_obj1, right=entity_or_obj2)

    def next_to(self, args):
        _, _, entity_or_obj1, entity_or_obj2, _ = args
        return NextToPattern(first=entity_or_obj1, second=entity_or_obj2)

    def predecessor(self, args):
        _, entity_or_obj1, entity_or_obj2, _ = args
        return PredecessorPattern(first=entity_or_obj1, second=entity_or_obj2)

    def negation_constraint(self, args):
        constraint = args[1]
        return NotConstraint(sub=constraint)

    def binary_constraint(self, args):
        arg1, op, arg2 = args
        op = str(op)
        if op == "and":
            return AndConstraint(left=arg1, right=arg2)
        if op == "or":
            return OrConstraint(left=arg1, right=arg2)
        raise CofolaParsingError(f"Unknown binary constraint operator: {op}")

    def part_constraint(self, args):
        constraint_template, _, _part_name_token, _, partition_ref = args
        return ForAllParts(partition=partition_ref, constraint_template=constraint_template)
