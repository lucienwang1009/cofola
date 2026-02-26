"""Constraint transformer mixin for Cofola parser - IR-native version."""
from __future__ import annotations

from typing import TYPE_CHECKING

from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.constraints import (
    SizeConstraint,
    MembershipConstraint, SubsetConstraint, DisjointConstraint, EqualityConstraint,
    TupleIndexEq, TupleIndexMembership,
    SequencePatternConstraint,
    FuncPairConstraint,
    BagSubsetConstraint, BagEqConstraint,
    NotConstraint, AndConstraint, OrConstraint,
    TogetherPattern, LessThanPattern, PredecessorPattern, NextToPattern,
)

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ConstraintTransformerMixin:
    """Mixin providing constraint transformation methods for CofolaTransformer (IR-native)."""

    def size_constraint(self: "CofolaTransfomer", args):
        expr, comparator, param = args
        expr = tuple(expr)
        param = int(param)
        if any(isinstance(obj, list) for obj, _ in expr):
            if len(expr) > 1:
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    "Only support simple size constraints on parts of partitions"
                )
            return list(
                SizeConstraint(terms=((part, expr[0][1]),), comparator=comparator, rhs=param)
                for part in expr[0][0]
            )
        return SizeConstraint(terms=expr, comparator=comparator, rhs=param)

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
        from cofola.parser.transformer import TupleIndexSentinel

        entity_or_index, in_or_not, objs = args

        def single_operation(obj):
            cat = self._ref_category(obj) if isinstance(obj, ObjRef) else 'unknown'
            if isinstance(entity_or_index, TupleIndexSentinel):
                # T[i] in S → TupleIndexMembership
                c = TupleIndexMembership(
                    tuple_ref=entity_or_index.tuple_ref,
                    index=entity_or_index.index,
                    container=obj,
                    positive=in_or_not,
                )
            elif cat == 'tuple' and isinstance(entity_or_index, Entity):
                # e in T → MembershipConstraint(entity, tuple_ref)
                c = MembershipConstraint(
                    entity=entity_or_index, container=obj, positive=in_or_not
                )
            elif isinstance(entity_or_index, Entity):
                c = MembershipConstraint(
                    entity=entity_or_index, container=obj, positive=in_or_not
                )
            else:
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    f"membership_constraint: unsupported types "
                    f"({type(entity_or_index).__name__} in {cat})"
                )
            return c

        return self._op_or_constraint_on_list(single_operation, objs)

    def subset_constraint(self: "CofolaTransfomer", args):
        objs1, _, objs2 = args

        def single_operation(obj1, obj2):
            cat1 = self._ref_category(obj1) if isinstance(obj1, ObjRef) else 'unknown'
            cat2 = self._ref_category(obj2) if isinstance(obj2, ObjRef) else 'unknown'
            if cat1 == 'bag' and cat2 == 'bag':
                return BagSubsetConstraint(sub=obj1, sup=obj2)
            return SubsetConstraint(sub=obj1, sup=obj2)

        return self._op_or_constraint_on_list(single_operation, objs1, objs2)

    def disjoint_constraint(self: "CofolaTransfomer", args):
        obj1, _, obj2 = args

        def single_operation(o1, o2):
            return DisjointConstraint(left=o1, right=o2)

        return self._op_or_constraint_on_list(single_operation, obj1, obj2)

    def equivalence_constraint(self: "CofolaTransfomer", args):
        from cofola.parser.transformer import TupleIndexSentinel

        obj1, symbol, obj2 = args
        symbol = str(symbol)

        def single_operation(o1, o2):
            if isinstance(o1, TupleIndexSentinel) and isinstance(o2, Entity):
                c = TupleIndexEq(
                    tuple_ref=o1.tuple_ref,
                    index=o1.index,
                    entity=o2,
                    positive=(symbol == '=='),
                )
            elif isinstance(o1, ObjRef) and isinstance(o2, ObjRef):
                cat1 = self._ref_category(o1)
                cat2 = self._ref_category(o2)
                if cat1 == 'bag' and cat2 == 'bag':
                    c = BagEqConstraint(
                        left=o1, right=o2, positive=(symbol == '==')
                    )
                else:
                    c = EqualityConstraint(
                        left=o1, right=o2, positive=(symbol == '==')
                    )
            else:
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    f"equivalence_constraint: unsupported types "
                    f"({type(o1).__name__} {symbol} {type(o2).__name__})"
                )
            return c

        return self._op_or_constraint_on_list(single_operation, obj1, obj2)

    def count_parameter(self: "CofolaTransfomer", args):
        count = int(args[0])
        if count < 0:
            from cofola.parser.parser import CofolaParsingError
            raise CofolaParsingError("Count parameter must be non-negative.")
        return count

    def seq_constraint(self: "CofolaTransfomer", args):
        pattern, is_in, obj = args
        cat = self._ref_category(obj) if isinstance(obj, ObjRef) else 'unknown'
        if cat != 'sequence':
            from cofola.parser.parser import CofolaParsingError
            raise CofolaParsingError(
                f"seq_constraint requires a Sequence, got {cat}"
            )
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
        if op == 'and':
            return AndConstraint(left=arg1, right=arg2)
        if op == 'or':
            return OrConstraint(left=arg1, right=arg2)
        from cofola.parser.parser import CofolaParsingError
        raise CofolaParsingError(f"Unknown binary constraint operator: {op}")

    def part_constraint(self: "CofolaTransfomer", args):
        return args[0]
