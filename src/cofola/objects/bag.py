"""Bag objects and operations."""
from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Generator

from sympy import Eq
from wfomc import fol_parse as parse, Const

from cofola.objects.base import (
    AtomicConstraint,
    Bag,
    CombinatoricsObject,
    Entity,
    MockObject,
    Set,
    SizedObject,
)
from cofola.objects.utils import invert_comparator, parse_comparator

# Import bag operations for backwards compatibility
from cofola.objects.bag_ops import (
    BagAdditiveUnion,
    BagBinaryOp,
    BagConstraint,
    BagDifference,
    BagEqConstraint,
    BagIntersection,
    BagSubsetConstraint,
    BagSupport,
    BagUnion,
)

if TYPE_CHECKING:
    from cofola.context import Context


class BagInit(Bag):
    """Bag initialized with entity-multiplicity pairs."""
    _fields = ("_entities_multiplicity_data",)

    def _assign_fields(self) -> None:
        self._entities_multiplicity_data = self._init_args[0] if self._init_args else {}
        self.p_entities_multiplicity = self._entities_multiplicity_data
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
            if entity not in o.p_entities_multiplicity or \
                    o.p_entities_multiplicity[entity] != multiplicity:
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
    """Bag formed by choosing a subset of elements."""
    _fields = ("obj_from", "size")

    def _assign_fields(self) -> None:
        super()._assign_fields()
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


# =======================================================
# Bag Multiplicity
# =======================================================


class BagMultiplicity(SizedObject, MockObject):
    """
    Used for unifying the size constraint and multiplicity constraint
    """
    _fields = ("obj", "entity")

    def _assign_fields(self) -> None:
        super()._assign_fields()
        if type(self.obj) is BagInit:
            self.size = self.obj.multiplicity(self.entity)
            self.max_size = self.size

    def body_str(self) -> str:
        return f"{self.obj.name}.count({self.entity.name})"

    def combinatorially_eq(self, o):
        return type(o) is BagMultiplicity and \
            self.obj == o.obj and self.entity == o.entity

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", "Expr"]:
        return context, context.get_entity_var(self.obj, self.entity)


# =======================================================
# Size Constraint
# =======================================================


class SizeConstraint(BagConstraint):
    """
    This constraint is used to constraint the size of bags, sets, the multiplicity of entities in bags
    as well as the count of entities in tuples
    """
    _fields = ("expr", "comp", "param")

    def _build_dependences(self) -> None:
        # NOTE: the dependences of a size constraint are the objects that are involved in the expression
        expr = self.expr
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


# Re-export for backwards compatibility
__all__ = [
    "BagInit",
    "BagChoose",
    "BagMultiplicity",
    "SizeConstraint",
    # Re-exported from bag_ops
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