
from collections import defaultdict
from typing import Union

from cofola.objects.base import Entity
from cofola.objects.utils import aux_obj_name

from cofola.parser.common import BaseTransformer, CofolaParsingError
from cofola.objects.set import Set, SetCartesian, SetChoose, SetChooseReplace, SetDifference, SetDisjointConstraint, SetEqConstraint, SetIntersection, SetMembershipConstraint, SetSizeConstraint, SetInit, SetSubsetConstraint, SetUnion


class SetTransformer(BaseTransformer):
    def set_declaration(self, args):
        set_id, set_obj = args[1], args[2]
        self._attach_obj(set_id, set_obj)

    def set_object(self, args):
        obj = args[0]
        if isinstance(obj, Set):
            return obj
        else:
            raise CofolaParsingError(
                "Expect a set object, "
                f"but got {obj} of type {type(obj)}"
            )

    def set_init(self, args):
        set_init_obj = args[0]
        self._update_entities(set_init_obj)
        return self._add_obj(set_init_obj)

    def list_entities(self, args):
        return SetInit(args)

    def slicing_entities(self, args):
        name = ""
        start = 0
        end = 0
        if len(args) == 2:
            start = int(args[0])
            end = int(args[1])
        elif len(args) == 3:
            name = str(args[0])
            start = int(args[1])
            end = int(args[2])
        entities = SetInit(
            [Entity(f'{name}{i}') for i in range(start, end)]
        )
        return entities

    def operation_producing_set(self, args):
        return self._add_obj(args[0])

    def set_choose(self, args):
        set_obj = args[2]
        if len(args) == 5:
            size = int(args[-2])
        else:
            size = None
        return SetChoose(set_obj, size)

    def set_choose_replace(self, args):
        set_obj = args[2]
        if len(args) == 5:
            size = int(args[-2])
        else:
            size = None
        return SetChooseReplace(set_obj, size)

    def set_union(self, args):
        set_obj1, set_obj2 = args[0], args[2]
        return SetUnion(set_obj1, set_obj2)

    def set_intersection(self, args):
        set_obj1, set_obj2 = args[0], args[2]
        return SetIntersection(set_obj1, set_obj2)

    def set_difference(self, args):
        set_obj1, set_obj2 = args[0], args[2]
        return SetDifference(set_obj1, set_obj2)

    def set_cartesian_product(self, args):
        set_obj1, set_obj2 = args[0], args[2]
        return SetCartesian(set_obj1, set_obj2)

    def set_size_constraint(self, args):
        expr, comparator, number = args
        expr = tuple(expr.items())
        return SetSizeConstraint(expr, comparator, number)

    def set_size_atom(self, args):
        if len(args) == 1:
            return 1.0, args[0]
        else:
            return float(args[0][0]), args[1]

    def set_size_atomic_expr(self, args):
        coef, set_obj = args[0]
        expr = defaultdict(lambda : 0.0)
        expr[set_obj] = coef
        return expr

    def set_size_add(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr[set_obj] += coef
        return expr

    def set_size_sub(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr[set_obj] -= coef
        return expr

    def set_membership_constraint(self, args):
        entity, set_obj = args
        if entity not in self.entities:
            raise CofolaParsingError(f"Entity {entity} not found.")
        return SetMembershipConstraint(set_obj, entity)

    def subset_constraint(self, args):
        set_obj1, set_obj2 = args
        return SetSubsetConstraint(set_obj2, set_obj1)

    def disjoint_constraint(self, args):
        set_obj1, set_obj2 = args
        return SetDisjointConstraint(set_obj1, set_obj2)

    def set_equivalence_constraint(self, args):
        set_obj1, set_obj2 = args
        return SetEqConstraint(set_obj1, set_obj2)
