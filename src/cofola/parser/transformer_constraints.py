"""Constraint transformer mixin for Cofola parser."""
from __future__ import annotations

from typing import TYPE_CHECKING

from cofola.objects.bag import SizeConstraint
from cofola.objects.base import CombinatoricsConstraint, Entity, SizedObject, Tuple
from cofola.objects.sequence import SequenceImpl, SequenceSizedPattern
from cofola.objects.set import MembershipConstraint, SubsetConstraint, DisjointConstraint, SetEqConstraint, Set
from cofola.objects.tuple import TupleIndexEqConstraint, TupleMembershipConstraint, TupleIndex
from cofola.objects.bag import Bag, BagSubsetConstraint, BagEqConstraint

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ConstraintTransformerMixin:
    """Mixin providing constraint transformation methods for CofolaTransformer."""

    def size_constraint(self: "CofolaTransfomer", args):
        expr, comparator, param = args
        expr = tuple(expr)
        param = int(param)
        if any(
            isinstance(obj, list) for obj, _ in expr
        ):
            if len(expr) > 1:
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    "Only support simple size constraints on parts of partitions"
                )
            self._check_obj_type(expr[0][0][0], SizedObject)
            return list(
                SizeConstraint([(part, expr[0][1])], comparator, param)
                for part in expr[0][0]
            )
        return SizeConstraint(expr, comparator, param)

    def size_atom(self: "CofolaTransfomer", args):
        if len(args) == 1:
            coef, obj = 1, args[0]
        else:
            coef, obj = args[0][0], args[1]
        self._check_obj_type(obj, SizedObject, list)
        return coef, obj

    def size_atomic_expr(self: "CofolaTransfomer", args):
        coef, obj = args[0]
        expr = [(obj, int(coef))]
        return expr

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
        entity_or_tuple_index, in_or_not, objs = args
        def single_operation(obj):
            self._check_obj_type(obj, Set, Bag, Tuple)
            if not isinstance(entity_or_tuple_index, TupleIndex) and \
                    not self.problem.contains_entity(entity_or_tuple_index):
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    f"Entity {entity_or_tuple_index} not found."
                )
            if isinstance(obj, Tuple) or \
                    isinstance(entity_or_tuple_index, TupleIndex):
                ret = TupleMembershipConstraint(obj, entity_or_tuple_index)
            else:
                ret = MembershipConstraint(obj, entity_or_tuple_index)
            if not in_or_not:
                ret.negate()
            return ret
        return self._op_or_constraint_on_list(
            single_operation, objs)

    def subset_constraint(self: "CofolaTransfomer", args):
        objs1, _, objs2 = args
        def single_operation(obj1, obj2):
            if isinstance(obj1, Set) or isinstance(obj2, Set):
                self._check_obj_type(obj1, Set, Bag)
                self._check_obj_type(obj2, Set, Bag)
                return SubsetConstraint(obj1, obj2)
            self._check_obj_type(obj1, Bag)
            self._check_obj_type(obj2, Bag)
            return BagSubsetConstraint(obj1, obj2)
        return self._op_or_constraint_on_list(
            single_operation, objs1, objs2)

    def disjoint_constraint(self: "CofolaTransfomer", args):
        obj1, _, obj2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, Bag)
            self._check_obj_type(obj2, Set, Bag)
            return DisjointConstraint(obj1, obj2)
        return self._op_or_constraint_on_list(
            single_operation, obj1, obj2)

    def equivalence_constraint(self: "CofolaTransfomer", args):
        obj1, symbol, obj2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, TupleIndex)
            self._check_obj_type(obj2, Set, Entity)
            if isinstance(obj1, TupleIndex) and isinstance(obj2, Entity):
                ret = TupleIndexEqConstraint(obj1, obj2)
            elif isinstance(obj1, Set) and isinstance(obj2, Set):
                ret = SetEqConstraint(obj1, obj2)
            elif isinstance(obj1, Bag) and isinstance(obj2, Bag):
                ret = BagEqConstraint(obj1, obj2)
            else:
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(f"Equivalence constraint is not supported for the given objects: {obj1}, {obj2}")
            if symbol == '!=':
                ret.negate()
            return ret
        return self._op_or_constraint_on_list(
            single_operation, obj1, obj2)

    def count_parameter(self: "CofolaTransfomer", args):
        count = int(args[0])
        if count < 0:
            from cofola.parser.parser import CofolaParsingError
            raise CofolaParsingError(f"Count parameter must be non-negative.")
        return count

    def seq_constraint(self: "CofolaTransfomer", args):
        from cofola.objects.sequence import SequenceConstraint

        pattern, is_in, obj = args
        self._check_obj_type(obj, SequenceImpl)
        constraint = SequenceConstraint(
            obj, pattern
        )
        if not is_in:
            constraint.negate()
        return constraint

    def seq_pattern(self: "CofolaTransfomer", args):
        pattern = args[0]
        return self.problem.add_object(pattern)

    def together(self: "CofolaTransfomer", args):
        from cofola.objects.sequence import TogetherPattern

        obj = args[2]
        return TogetherPattern(obj)

    def less_than(self: "CofolaTransfomer", args):
        from cofola.objects.sequence import LessThanPattern

        entity_or_obj1, _, entity_or_obj2 = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return LessThanPattern(entity_or_obj1, entity_or_obj2)

    def next_to(self: "CofolaTransfomer", args):
        from cofola.objects.sequence import NextToPattern

        _, _, entity_or_obj1, entity_or_obj2, _ = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return NextToPattern(entity_or_obj1, entity_or_obj2)

    def predecessor(self: "CofolaTransfomer", args):
        from cofola.objects.sequence import PredecessorPattern

        _, entity_or_obj1, entity_or_obj2, _ = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return PredecessorPattern(entity_or_obj1, entity_or_obj2)

    def negation_constraint(self: "CofolaTransfomer", args):
        constraint = args[1]
        self._check_obj_type(
            constraint, CombinatoricsConstraint
        )
        return ~args[1]

    def binary_constraint(self: "CofolaTransfomer", args):
        arg1, op, arg2 = args
        self._check_obj_type(
            arg1, CombinatoricsConstraint
        )
        self._check_obj_type(
            arg2, CombinatoricsConstraint
        )
        if op == 'and':
            return arg1 & arg2
        if op == 'or':
            return arg1 | arg2

    def part_constraint(self: "CofolaTransfomer", args):
        return args[0]