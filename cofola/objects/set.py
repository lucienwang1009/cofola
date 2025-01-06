from __future__ import annotations
from decimal import Context
from functools import reduce
from wfomc import fol_parse as parse
from wfomc import Const, Pred
from symengine import Eq

from cofola.objects.base import AtomicConstraint, Set, Bag, Entity
from cofola.objects.bag import Bag

from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from cofola.context import Context


class SetInit(Set):
    def __init__(self, iterable: set[Entity]) -> None:
        super().__init__(iterable)

    def _assign_args(self) -> None:
        # the potential entities here are exactly the entities in the set
        self.p_entities = set(self.args[0])
        self.size = len(self.p_entities)
        self.max_size = self.size

    def combinatorially_eq(self, o):
        return type(o) is SetInit and \
            frozenset(self.p_entities) == frozenset(o.p_entities)

    def body_str(self) -> str:
        return "{" + ", ".join(str(e) for e in self) + "}"

    def __len__(self) -> int:
        return len(self.p_entities)

    def __iter__(self):
        return iter(self.p_entities)

    def __contains__(self, item):
        return item in self.p_entities

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        context.unary_evidence.update(
            obj_pred(Const(entity.name)) for entity in self
        )
        context.unary_evidence.update(
            ~obj_pred(Const(entity.name)) for entity
            in context.problem.entities
            if entity not in self
        )
        return context


class SetChoose(Set):
    def __init__(self, obj_from: Set, size: int = None) -> None:
        super().__init__(obj_from, size)

    def _assign_args(self) -> None:
        self.obj_from, self.size = self.args

    def inherit(self) -> None:
        self.update(self.obj_from.p_entities,
                    max_size=self.obj_from.max_size)

    def body_str(self) -> str:
        if self.size is None:
            return f"choose({self.obj_from.name})"
        else:
            return f"choose({self.obj_from.name}, {self.size})"

    def is_uncertain(self) -> bool:
        return True

    def encode(self, context: Context) -> Context:
        obj_pred = context.get_pred(self, create=True, use=False)
        from_pred = context.get_pred(self.obj_from)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) -> {from_pred}(X))"
        )
        if self.size is not None:
            context.cardinality_constraint.add_simple_constraint(
                obj_pred, "==", self.size
            )
            # var = context.get_obj_var(self)
            # context.validator.append(Eq(var, self.size))
        return context


class SetChooseReplace(Bag):
    def __init__(self, obj_from: Set, size: int = None) -> None:
        super().__init__(obj_from, size)

    def _assign_args(self) -> None:
        self.obj_from, self.size = self.args
        if self.size is not None:
            self.max_size = self.size
        self.p_entities_multiplicity = dict(
            (entity, self.max_size)
            for entity in self.obj_from.p_entities
        )
        # all entities in the set are indistinguishable
        # as they share the same multiplicity (the size)
        self.dis_entities = set()
        self.indis_entities = {self.max_size: self.obj_from.p_entities}

    def inherit(self) -> None:
        # all entities in the set are indistinguishable
        # as they share the same multiplicity (the size)
        self.p_entities_multiplicity = dict(
            (entity, self.max_size)
            for entity in self.obj_from.p_entities
        )
        self.indis_entities = {self.max_size: self.obj_from.p_entities}

    def body_str(self) -> str:
        return f"chooseReplace({self.obj_from.name}, {self.size})"

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
            if multiplicity == float('inf'):
                raise ValueError("SetChooseReplace with infinite size is not supported")
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
        return context


class SetBinaryOp(Set):
    def __init__(self, op_name: str, first: Set, second: Set) -> None:
        self.op_name = op_name
        super().__init__(first, second)

    def _assign_args(self) -> None:
        self.first, self.second = self.args

    def body_str(self) -> str:
        return f"({self.first.name} {self.op_name} {self.second.name})"


class SetUnion(SetBinaryOp):
    def __init__(self, first: Set, second: Set) -> None:
        super().__init__("∪", first, second)

    def inherit(self) -> None:
        if self.first.size is not None and self.second.size is not None:
            if self.first.p_entities.isdisjoint(self.second.p_entities):
                self.size = self.first.size + self.second.size
            if isinstance(self.first, SetInit) and \
                    self.second.p_entities.issubset(self.first.p_entities):
                self.size = self.first.size
            if isinstance(self.second, SetInit) and \
                    self.first.p_entities.issubset(self.second.p_entities):
                self.size = self.second.size
        if self.size is not None:
            self.max_size = self.size
        self.update(self.first.p_entities | self.second.p_entities,
                    max_size=self.first.max_size + self.second.max_size)

    def combinatorially_eq(self, o):
        return type(o) is SetUnion and \
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


class SetIntersection(SetBinaryOp):
    def __init__(self, first: Set, second: Set) -> None:
        super().__init__("∩", first, second)

    def inherit(self) -> None:
        if self.first.size is not None and self.second.size is not None:
            if self.first.p_entities.isdisjoint(self.second.p_entities):
                self.size = 0
            if isinstance(self.first, SetInit) and \
                    self.second.p_entities.issubset(self.first.p_entities):
                self.size = self.second.size
            if isinstance(self.second, SetInit) and \
                    self.first.p_entities.issubset(self.second.p_entities):
                self.size = self.first.size
        if self.size is not None:
            self.max_size = self.size
        self.update(self.first.p_entities & self.second.p_entities,
                    min(self.first.max_size, self.second.max_size))

    def combinatorially_eq(self, o):
        return type(o) is SetIntersection and \
            ((self.first == o.first and self.second == o.second) or \
             (self.first == o.second and self.second == o.first))

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) & {second_pred}(X)))"
        )
        return context


class SetDifference(SetBinaryOp):
    def __init__(self, first: Set, second: Set) -> None:
        super().__init__("\\", first, second)

    def inherit(self) -> None:
        if self.first.size is not None and self.second.size is not None:
            if self.first.p_entities.isdisjoint(self.second.p_entities):
                self.size = self.first.size
            if isinstance(self.first, SetInit) and \
                    self.second.p_entities.issubset(self.first.p_entities):
                self.size = self.first.size - self.second.size
            if isinstance(self.second, SetInit) and \
                    self.first.p_entities.issubset(self.second.p_entities):
                self.size = 0
        if self.size is not None:
            self.max_size = self.size
        self.update(self.first.p_entities,
                    self.first.max_size)

    def combinatorially_eq(self, o):
        return type(o) is SetDifference and self.first == o.first and self.second == o.second

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        context.sentence = context.sentence & parse(
            f"\\forall X: ({obj_pred}(X) <-> ({first_pred}(X) & ~{second_pred}(X)))"
        )
        return context


# =======================================================
# Set Constraints
# =======================================================

class SetConstraint(AtomicConstraint):
    pass


# NOTE: membership constraint is for both set and bag
class MembershipConstraint(SetConstraint):
    def __init__(self, obj: Union[Set, Bag],
                 member: Entity) -> None:
        super().__init__(obj, member)

    def _assign_args(self) -> None:
        self.obj, self.member = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.member.name} ∈ {self.obj.name}"
        else:
            return f"{self.member.name} ∉ {self.obj.name}"

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self.obj)
        if self.positive:
            context.unary_evidence.add(
                obj_pred(Const(self.member.name))
            )
        else:
            context.unary_evidence.add(
                ~obj_pred(Const(self.member.name))
            )
        return context


class SubsetConstraint(SetConstraint):
    """
    A constraint that the first set or bag is a subset of the second set or bag.
    The multiplicities of the entities in the bags (if any) are not considered.
    """
    def __init__(self, sub: Union[Set, Bag], sup: Union[Set, Bag]) -> None:
        super().__init__(sub, sup)

    def _assign_args(self) -> None:
        self.sub, self.sup = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.sub.name} ⊆ {self.sup.name}"
        else:
            return f"{self.sub.name} ⊈ {self.sup.name}"

    def encode(self, context: "Context") -> "Context":
        sub_pred = context.get_pred(self.sub)
        sup_pred = context.get_pred(self.sup)
        if self.positive:
            context.sentence = context.sentence & parse(
                f"\\forall X: ({sub_pred}(X) -> {sup_pred}(X))"
            )
        else:
            context.sentence = context.sentence & parse(
                f"\\exists X: ({sub_pred}(X) & ~{sup_pred}(X))"
            )
        return context


class DisjointConstraint(SetConstraint):
    def __init__(self, first: Set, second: Set) -> None:
        super().__init__(first, second)

    def _assign_args(self) -> None:
        self.first, self.second = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.first.name} ∩ {self.second.name} = ∅"
        else:
            return f"{self.first.name} ∩ {self.second.name} ≠ ∅"

    def encode(self, context: "Context") -> "Context":
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        if self.positive:
            context.sentence = context.sentence & parse(
                f"\\forall X: (~({first_pred}(X) & {second_pred}(X)))"
            )
        else:
            context.sentence = context.sentence & parse(
                f"\\exists X: ({first_pred}(X) & {second_pred}(X))"
            )
        return context


class SetEqConstraint(SetConstraint):
    def __init__(self, first: Set, second: Set) -> None:
        super().__init__(first, second)

    def _assign_args(self) -> None:
        self.first, self.second = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.first.name} = {self.second.name}"
        else:
            return f"{self.first.name} ≠ {self.second.name}"

    def encode(self, context: "Context") -> "Context":
        first_pred = context.get_pred(self.first)
        second_pred = context.get_pred(self.second)
        if self.positive:
            context.sentence = context.sentence & parse(
                f"\\forall X: ({first_pred}(X) <-> {second_pred}(X))"
            )
        else:
            context.sentence = context.sentence & parse(
                f"\\exists X: (({first_pred}(X) & ~{second_pred}(X)) | (~{first_pred}(X) & {second_pred}(X)))"
            )
        return context
