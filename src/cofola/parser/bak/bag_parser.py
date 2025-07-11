
from collections import defaultdict

from cofola.objects.base import Entity

from cofola.parser.common import BaseTransformer, CofolaParsingError
from cofola.objects.bag import Bag, BagChoose, BagDifference, BagIntersection, BagMembershipConstraint, BagMultiplicityConstraint, BagSizeConstraint, BagInit, BagUnion


class BagTransformer(BaseTransformer):
    def bag_declaration(self, args):
        bag_id, bag_obj = args[1], args[2]
        self._attach_obj(bag_id, bag_obj)

    def bag_object(self, args):
        obj = args[0]
        if isinstance(obj, Bag):
            return obj
        else:
            raise CofolaParsingError(
                "Expect a bag object, "
                f"but got {obj} of type {type(obj)}"
            )

    def bag_init(self, args):
        bag_init_obj = args[0]
        self._update_entities(bag_init_obj.keys())
        return self._add_obj(bag_init_obj)

    def bag_body(self, args):
        return BagInit(args)

    def entity_multiplicity(self, args):
        entity, multiplicity = args[0], args[2]
        return entity, multiplicity

    def operation_producing_bag(self, args):
        return self._add_obj(args[0])

    def bag_choose(self, args):
        bag_obj = args[2]
        if len(args) == 5:
            size = int(args[-2])
        else:
            size = None
        return BagChoose(bag_obj, size)

    def bag_union(self, args):
        bag_obj1, bag_obj2 = args[0], args[2]
        return BagUnion(bag_obj1, bag_obj2)

    def bag_intersection(self, args):
        bag_obj1, bag_obj2 = args[0], args[2]
        return BagIntersection(bag_obj1, bag_obj2)

    def bag_difference(self, args):
        bag_obj1, bag_obj2 = args[0], args[2]
        return BagDifference(bag_obj1, bag_obj2)

    def bag_size_constraint(self, args):
        expr, comparator, number = args
        expr = tuple(expr.items())
        return BagSizeConstraint(expr, comparator, number)

    def bag_size_atom(self, args):
        if len(args) == 1:
            return 1.0, args[0]
        else:
            return float(args[0][0]), args[1]

    # TODO: put the praser of size constraints to the base class
    def bag_size_atomic_expr(self, args):
        coef, bag_obj = args[0]
        expr = defaultdict(lambda : 0.0)
        expr[bag_obj] = coef
        return expr

    def bag_size_add(self, args):
        expr, atom = args
        coef, bag_obj = atom
        expr[bag_obj] += coef
        return expr

    def bag_size_sub(self, args):
        expr, atom = args
        coef, bag_obj = atom
        expr[bag_obj] -= coef
        return expr

    def bag_membership_constraint(self, args):
        entity, bag_obj = args
        if entity not in self.entities:
            raise CofolaParsingError(f"Entity {entity} not found.")
        return BagMembershipConstraint(bag_obj, entity)

    def bag_multiplicity_constraint(self, args):
        bag_obj, entity, comparator, number = args[0], args[3], args[5], args[7]
        return BagMultiplicityConstraint(bag_obj, entity, comparator, number)
