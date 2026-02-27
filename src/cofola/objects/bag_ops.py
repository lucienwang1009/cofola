"""Bag operations and binary operators."""
from __future__ import annotations

from functools import reduce

from sympy import Eq, Min
from wfomc import fol_parse as parse, Const

from cofola.objects.base import AtomicConstraint, Bag, CombinatoricsObject, Entity, Set
from cofola.objects.utils import invert_comparator, parse_comparator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cofola.context import Context


class BagBinaryOp(Bag):
    """Base class for bag binary operations."""
    _fields = ("first", "second")

    def __init__(self, op_name: str, first: Bag, second: Bag) -> None:
        self.op_name: str = op_name
        super().__init__(first, second)

    def body_str(self) -> str:
        return f"({self.first.name} {self.op_name} {self.second.name})"


class BagUnion(BagBinaryOp):
    """Bag union (max multiplicity per element)."""

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
    """Bag additive union (sum multiplicities)."""

    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("⊎", first, second)

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
    """Bag intersection (min multiplicity per element)."""

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


class BagDifference(BagBinaryOp):
    """Bag difference."""

    def __init__(self, first: Bag, second: Bag) -> None:
        super().__init__("\\", first, second)

    def inherit(self) -> None:
        self.update(
            self.first.p_entities_multiplicity,
            self.first.max_size,
        )
        self.dis_entities = self.first.dis_entities & self.second.dis_entities
        self.indis_entities = set()

    def combinatorially_eq(self, o):
        return type(o) is BagDifference and \
            ((self.first == o.first and self.second == o.second) or \
             (self.first == o.second and self.second == o.first))

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) & ~{second_pred}(X)))"
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
                Eq(entity_var, first_entity_mul - second_entity_mul)
            )
        return context


class BagSupport(Set):
    """Support set of a bag (set of distinct elements)."""
    _fields = ("obj_from",)

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


# =======================================================
# Bag Constraints
# =======================================================


class BagConstraint(AtomicConstraint):
    """Base class for bag constraints."""
    pass


class BagSubsetConstraint(BagConstraint):
    """Bag subset constraint."""

    def __init__(self, sup: Bag, sub: Bag) -> None:
        raise NotImplementedError("Bag subset constraint is not implemented yet.")
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
    """Bag equality constraint."""

    def __init__(self, first: Bag, second: Bag) -> None:
        raise NotImplementedError("Bag equality constraint is not implemented yet.")
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


# Re-export for backwards compatibility
__all__ = [
    "BagBinaryOp",
    "BagUnion",
    "BagAdditiveUnion",
    "BagIntersection",
    "BagDifference",
    "BagSupport",
    "BagConstraint",
    "BagSubsetConstraint",
    "BagEqConstraint",
]