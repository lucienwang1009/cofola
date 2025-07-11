from __future__ import annotations

from typing import Union
from cofola.objects.function import FuncComposition, FuncEqConstrainedImage, FuncImage, FuncInverse, FuncInverseImage, FuncMembershipConstraintedImage, FuncSizeConstrainedImage, FuncInit, FuncSubsetConstrainedImage, FuncDisjointConstrainedImage, Function
from cofola.parser.common import BaseTransformer, CofolaParsingError


class FunctionTransformer(BaseTransformer):
    def func_declaration(self, args):
        func_id, func_obj = args[1], args[2]
        self._attach_obj(func_id, func_obj)

    def func_object(self, args):
        obj = args[0]
        if isinstance(obj, Function):
            return obj
        else:
            raise CofolaParsingError(
                "Expect a function object, "
                f"but got {obj} of type {type(obj)}"
            )

    def func_init(self, args):
        return self._add_obj(args[0])

    def general_func_init(self, args):
        domain, codomain = args[0], args[2]
        return FuncInit(domain, codomain)

    def injective_func_init(self, args):
        domain, codomain = args[0], args[2]
        return FuncInit(domain, codomain, injective=True)

    def surjective_func_init(self, args):
        domain, codomain = args[0], args[2]
        return FuncInit(domain, codomain, surjective=True)

    def bijective_func_init(self, args):
        domain, codomain = args[0], args[2]
        return FuncInit(domain, codomain, injective=True, surjective=True)

    def operation_producing_func(self, args):
        return self._add_obj(args[0])

    def func_inverse(self, args):
        func_obj = args[0]
        return FuncInverse(func_obj)

    def func_composition(self, args):
        func1, func2 = args[0], args[2]
        return FuncComposition(func1, func2)

    def func_image_entity(self, args):
        func_obj, entity = args[0], args[2]
        if entity not in self.entities:
            raise CofolaParsingError(f"Entity {entity} not found. Please use f[{entity}] for image of set {entity}.")
        return FuncImage(func_obj, entity)

    def func_image_set(self, args):
        func_obj, set_obj = args[0], args[2]
        return FuncImage(func_obj, set_obj)

    def func_inverse_image_entity(self, args):
        func_obj, entity = args[0], args[3]
        if entity not in self.entities:
            raise CofolaParsingError(f"Entity {entity} not found. Please use f[{entity}] for image of set {entity}.")
        return FuncInverseImage(func_obj, entity)

    def func_inverse_image_set(self, args):
        func_obj, set_obj = args[0], args[3]
        return FuncInverseImage(func_obj, set_obj)

    def func_size_constrained_image(self, args):
        func_obj, _, comparator, size = args
        return FuncSizeConstrainedImage(func_obj, comparator, size)

    def func_subset_constrained_image_1(self, args):
        func_obj, _, set_obj = args[:3]
        return FuncSubsetConstrainedImage(func_obj, set_obj)

    def func_subset_constrained_image_2(self, args):
        set_obj, func_obj = args[:2]
        return FuncSubsetConstrainedImage(func_obj, set_obj, inverse=True)

    def func_membership_constrained_image(self, args):
        entity, func_obj, _ = args
        return FuncMembershipConstraintedImage(func_obj, entity)

    def func_disjoint_constrained_image(self, args):
        func_obj, _, set_obj = args
        return FuncDisjointConstrainedImage(func_obj, set_obj)

    def func_equivalence_constrained_image(self, args):
        func_obj, _, set_obj = args
        return FuncEqConstrainedImage(func_obj, set_obj)
