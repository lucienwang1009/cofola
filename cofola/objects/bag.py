from __future__ import annotations
from decimal import Context
from functools import reduce
from symengine import Eq, Min, Max
from wfomc.fol.sc2 import Pred, Const
from wfomc import RingElement, fol_parse as parse
from typing import TYPE_CHECKING, Generator

from cofola.objects.base import AtomicConstraint, \
    CombinatoricsObject, MockObject, Set, Bag, Entity, SizedObject
from cofola.objects.tuple import TupleCount
from cofola.objects.utils import invert_comparator, parse_comparator

if TYPE_CHECKING:
    from cofola.context import Context


class BagInit(Bag):
    def __init__(self, entity_multiplicity: tuple[tuple[Entity, int]]) -> None:
        """
        :param entity_multiplicity: a dictionary of entity and its multiplicity
        """
        super().__init__(entity_multiplicity)

    def _assign_args(self) -> None:
        self.p_entities_multiplicity = self.args[0]
        self.size = sum(self.p_entities_multiplicity.values())
        self.max_size = self.size
        self.dis_entities = set()
        self.indis_entities = dict()

    def combinatorially_eq(self, o):
        if type(o) is not BagInit:
            return False
        if len(self.p_entities_multiplicity) != len(o.p_entities_multiplicity):
            return False
        for entity, multiplicity in self.p_entities_multiplicity.items():
            if entity not in o or o[entity] != multiplicity:
                return False
        return True

    @property
    def entities(self) -> set[Entity]:
        return set(self.p_entities_multiplicity.keys())

    def items(self) -> Generator[tuple[Entity, int], None, None]:
        return self.p_entities_multiplicity.items()

    def keys(self) -> Generator[Entity, None, None]:
        return self.p_entities_multiplicity.keys()

    def values(self) -> Generator[int, None, None]:
        return self.p_entities_multiplicity.values()

    def __contains__(self, entity: Entity) -> bool:
        return entity in self.p_entities_multiplicity

    def body_str(self) -> str:
        return "{" + ", ".join(
            f"{entity.name}: {multiplicity}" for entity, multiplicity in self.items()
        ) + "}"

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        context.unary_evidence.update(
            obj_pred(Const(entity.name)) for entity in self.keys()
        )
        context.unary_evidence.update(
            ~obj_pred(Const(entity.name)) for entity
            in context.problem.entities
            if entity not in self
        )
        for entity in self.dis_entities:
            context, _ = entity.encode(context)
        return context


class BagChoose(Bag):
    def __init__(self, obj_from: Bag, size: int = None) -> None:
        super().__init__(obj_from, size)

    def _assign_args(self) -> None:
        self.obj_from, self.size = self.args
        if self.size is not None:
            self.max_size = self.size

    def inherit(self) -> None:
        self.update(
            self.obj_from.p_entities_multiplicity,
            max_size=self.obj_from.max_size
        )
        self.dis_entities = self.obj_from.dis_entities
        self.indis_entities = self.obj_from.indis_entities

    def body_str(self) -> str:
        return f"choose({self.obj_from.name})"

    def is_uncertain(self) -> bool:
        return True

    def encode(self, context: Context) -> Context:
        obj_pred = context.get_pred(self, create=True, use=False)
        from_pred = context.get_pred(self.obj_from)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
        )
        for entity in self.dis_entities:
            if entity in context.singletons:
                continue
            multiplicity = self.p_entities_multiplicity[entity]
            _, entity_pred = entity.encode(context)
            bag_entity_pred = context.get_entity_pred(
                self, entity
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
            )
            entity_var = context.get_entity_var(self, entity)
            context.weighting[bag_entity_pred] = (
                reduce(lambda x, y: x + entity_var ** y, range(1, multiplicity + 1), 0), 1
            )
        if not isinstance(self.obj_from, BagInit):
            # add inequality constraints
            for entity in self.dis_entities:
                from_entity_var = context.get_entity_var(self.obj_from, entity)
                entity_var = context.get_entity_var(self, entity)
                context.validator.append(entity_var <= from_entity_var)
        return context


class BagSupport(Set):
    def __init__(self, obj_from: Bag) -> None:
        super().__init__(obj_from)

    def _assign_args(self) -> None:
        self.obj_from = self.args[0]

    def inherit(self) -> None:
        self.update(
            set(self.obj_from.p_entities_multiplicity.keys()),
            self.obj_from.max_size
        )

    def body_str(self) -> str:
        return f"support({self.obj_from.name})"

    def combinatorially_eq(self, o: CombinatoricsObject) -> bool:
        return type(o) is BagSupport and self.obj_from == o.obj_from

    def encode(self, context: Context) -> Context:
        context.obj2pred[self] = context.get_pred(
            self.obj_from
        )
        context.used_objs.add(self.obj_from)
        return context


class BagBinaryOp(Bag):
    def __init__(self, op_name: str, first: Bag, second: Bag) -> None:
        self.op_name: str = op_name
        super().__init__(first, second)

    def _assign_args(self) -> None:
        # only support the bags
        self.first, self.second = self.args

    def body_str(self) -> str:
        return f"({self.first.name} {self.op_name} {self.second.name})"


# TODO
class BagUnion(BagBinaryOp):
    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("∪", first, second)

    def inherit(self) -> None:
        p_entities_multiplicity = dict()
        for entity, multiplicity in self.first.p_entities_multiplicity.items():
            if entity in self.second.p_entities_multiplicity:
                p_entities_multiplicity[entity] = max(
                    multiplicity, self.second.p_entities_multiplicity[entity]
                )
            else:
                p_entities_multiplicity[entity] = multiplicity
        self.update(
            p_entities_multiplicity,
            sum(p_entities_multiplicity.values()),
            self.first.dis_entities | self.second.dis_entities,
            self.first.indis_entities
        )

    def combinatorially_eq(self, o):
        return type(o) is BagUnion and \
            ((self.first == o.first and self.second == o.second) or \
             (self.first == o.second and self.second == o.first))

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) | {second_pred}(X)))"
        )
        return context


class BagAdditiveUnion(BagBinaryOp):
    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("⊎", first, second)
        self.inherit()

    def inherit(self) -> None:
        if self.first.size is not None and self.second.size is not None:
            if set(self.first.p_entities_multiplicity.keys()).isdisjoint(
                    self.second.p_entities_multiplicity.keys()
            ):
                self.size = self.first.size + self.second.size
        if self.size is not None:
            self.max_size = self.size

        all_entities = (
            set(self.first.p_entities_multiplicity.keys()) |
            set(self.second.p_entities_multiplicity.keys())
        )
        p_entities_multiplicity = dict()
        for entity in all_entities:
            p_entities_multiplicity[entity] = (
                self.first.p_entities_multiplicity.get(entity, 0) +
                self.second.p_entities_multiplicity.get(entity, 0)
            )
        self.update(
            p_entities_multiplicity,
            self.first.max_size + self.second.max_size
        )
        # all entities in the BagAdditiveUnion are distinguishable
        self.dis_entities = self.first.dis_entities | self.second.dis_entities
        self.indis_entities = set()

    def combinatorially_eq(self, o):
        return type(o) is BagAdditiveUnion and \
            ((self.first == o.first and self.second == o.second) or \
             (self.first == o.second and self.second == o.first))

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) | {second_pred}(X)))"
        )
        for entity in self.dis_entities:
            if entity in context.singletons:
                continue
            multiplicity = self.p_entities_multiplicity[entity]
            _, entity_pred = entity.encode(context)
            bag_entity_pred = context.get_entity_pred(
                self, entity
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
            )
            entity_var = context.get_entity_var(self, entity)
            context.weighting[bag_entity_pred] = (
                reduce(lambda x, y: x + entity_var ** y, range(1, multiplicity + 1), 0), 1
            )
            if isinstance(self.first, BagInit):
                first_entity_mul = self.first.multiplicity(entity)
            else:
                first_entity_mul = context.get_entity_var(self.first, entity)
            if isinstance(self.second, BagInit):
                second_entity_mul = self.second.multiplicity(entity)
            else:
                second_entity_mul = context.get_entity_var(self.second, entity)
            context.validator.append(
                Eq(entity_var, first_entity_mul + second_entity_mul)
            )
        return context


class BagIntersection(BagBinaryOp):
    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("∩", first, second)

    def inherit(self) -> None:
        p_entities_multiplicity = dict()
        for entity, multiplicity in self.first.p_entities_multiplicity.items():
            if entity in self.second.p_entities_multiplicity:
                p_entities_multiplicity[entity] = min(
                    multiplicity, self.second.p_entities_multiplicity[entity]
                )
        self.update(
            p_entities_multiplicity,
            min(self.first.max_size, self.second.max_size),
        )
        # all entities in the BagAdditiveUnion are distinguishable
        self.dis_entities = self.first.dis_entities & self.second.dis_entities
        self.indis_entities = set()

    def combinatorially_eq(self, o):
        return type(o) is BagIntersection and \
            ((self.first == o.first and self.second == o.second) or \
             (self.first == o.second and self.second == o.first))

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) & {second_pred}(X)))"
        )
        for entity in self.dis_entities:
            if entity in context.singletons:
                continue
            multiplicity = self.p_entities_multiplicity[entity]
            _, entity_pred = entity.encode(context)
            bag_entity_pred = context.get_entity_pred(
                self, entity
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
            )
            entity_var = context.get_entity_var(self, entity)
            context.weighting[bag_entity_pred] = (
                reduce(lambda x, y: x + entity_var ** y, range(1, multiplicity + 1), 0), 1
            )
            if isinstance(self.first, BagInit):
                first_entity_mul = self.first.multiplicity(entity)
            else:
                first_entity_mul = context.get_entity_var(self.first, entity)
            if isinstance(self.second, BagInit):
                second_entity_mul = self.second.multiplicity(entity)
            else:
                second_entity_mul = context.get_entity_var(self.second, entity)
            context.validator.append(
                Eq(entity_var, Min(first_entity_mul, second_entity_mul))
            )
        return context


# TODO
class BagDifference(BagBinaryOp):
    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("\\", first, second)

    def inherit(self) -> None:
        self.update(
            self.first.p_entities_multiplicity,
            self.first.max_size,
            self.first.dis_entities,
            self.first.indis_entities
        )

    def combinatorially_eq(self, o):
        return type(o) is BagDifference and self.first == o.first and self.second == o.second

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) -> {first_pred}(X))"
        )
        for entity in self.dis_entities:
            multiplicity = self.p_entities_multiplicity[entity]
            _, entity_pred = entity.encode(context)
            bag_entity_pred = context.get_entity_pred(
                self, entity
            )
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_entity_pred}(X) <-> {obj_pred}(X) & {entity_pred}(X))"
            )
            entity_var = context.get_entity_var(self, entity)
            context.weighting[bag_entity_pred] = (
                reduce(lambda x, y: x + entity_var ** y, range(1, multiplicity + 1), 0), 1
            )
            if isinstance(self.first, BagInit):
                first_entity_mul = self.first.multiplicity(entity)
            else:
                first_entity_mul = context.get_entity_var(self.first, entity)
            if isinstance(self.second, BagInit):
                second_entity_mul = self.second.multiplicity(entity)
            else:
                second_entity_mul = context.get_entity_var(self.second, entity)
            context.validator.append(
                Eq(entity_var, Max(first_entity_mul - second_entity_mul, 0))
            )
        return context


# =======================================================
# Bag Constraints
# =======================================================


class BagMultiplicity(SizedObject, MockObject):
    """
    Used for unifying the size constraint and multiplicity constraint
    """
    def __init__(self, obj: Bag, entity: Entity) -> None:
        super().__init__(obj, entity)

    def _assign_args(self) -> None:
        self.obj, self.entity = self.args

    def body_str(self) -> str:
        return f"{self.obj.name}.count({self.entity.name})"

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", RingElement]:
        return context, context.get_entity_var(self.obj, self.entity)


class BagConstraint(AtomicConstraint):
    pass


class SizeConstraint(BagConstraint):
    """
    This constraint is used to constraint the size of bags, sets, the ultiplicity of entities in bags
    as well as the count of entities in tuples
    """
    def __init__(self, expr: list[tuple[SizedObject, int]],
                 comp: str, param: int) -> None:
        super().__init__(expr, comp, param)

    def _assign_args(self) -> None:
        self.expr, self.comp, self.param = self.args

    def _build_dependences(self) -> None:
        # NOTE: the dependences of a size constraint are the objects that are involved in the expression
        expr = self.args[0]
        self.dependences = set(item[0] for item in expr)
        for dep in self.dependences:
            dep.descendants.add(self)

    def negate(self):
        super().negate()
        self.comp = invert_comparator(self.comp)

    def __str__(self) -> str:
        s = ''
        strs = []
        for obj, coef in self.expr:
            strs.append(f'{coef} * ({obj.name})')
        s += ' + '.join(strs)
        s += ' {} {}'.format(self.comp, self.param)
        return s

    def subs_obj(self, old_obj: CombinatoricsObject,
                 new_obj: CombinatoricsObject):
        expr = list(
            (new_obj if obj == old_obj else obj, coef) for obj, coef in self.expr
        )
        self.subs_args(expr, self.comp, self.param)

    def contains(self, bag: Bag) -> bool:
        """
        Whether the bag is contained in the constraint. For inferring the size of bag

        :param bag: the bag
        :return: True if the bag is contained in the constraint
        """
        return any(obj == bag for obj, _ in self.expr)

    def encode(self, context: "Context") -> "Context":
        left = 0
        for obj, coef in self.expr:
            context, var = obj.encode_size_var(context)
            left = left + coef * var
        context.validator.append(parse_comparator(self.comp)(left, self.param))
        return context


# class BagMembershipConstraint(BagConstraint):
#     def __init__(self, obj: Bag, member: Entity) -> None:
#         super().__init__(obj, member)
#         self.obj: Bag
#         self.member: Entity
#
#     def build(self) -> None:
#         super().build()
#         self.obj, self.member = self.args
#
#     def __str__(self) -> str:
#         return f"{self.member.name} ∈ {self.obj.name}"
#
#     def encode(self, context: "Context") -> "Context":
#         obj_pred = context.get_pred(self.obj)
#         context.unary_evidence.add(
#             obj_pred(Const(self.member.name))
#         )
#         return context


# TODO
class BagSubsetConstraint(BagConstraint):
    def __init__(self, sup: Bag, sub: Bag) -> None:
        super().__init__(sup, sub)
        self.sup = sup
        self.sub = sub

    def __str__(self) -> str:
        return f"{self.sub.name} ⊆ {self.sup.name}"

    def encode(self, context: "Context") -> "Context":
        sub_pred = context.get_pred(self.sub)
        sup_pred = context.get_pred(self.sup)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({sub_pred}(X) -> {sup_pred}(X))"
        )
        return context


class BagEqConstraint(BagConstraint):
    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__(first, second)
        self.first = first
        self.second = second

    def __str__(self) -> str:
        return f"{self.first.name} == {self.second.name}"

    def encode(self, context: "Context") -> "Context":
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({first_pred}(X) <-> {second_pred}(X))"
        )
        return context
