from wfomc import fol_parse as parse
from sympy import Eq

from cofola.objects.bag import BagInit
from cofola.objects.base import Bag, Partition, Set, Part

from typing import TYPE_CHECKING

from cofola.utils import ListLessThan


if TYPE_CHECKING:
    from cofola.context import Context


class SetPart(Part, Set):
    def inherit(self) -> None:
        obj_from = self.obj_from.obj_from
        self.update(
            obj_from.p_entities,
            obj_from.max_size
        )


class BagPart(Part, Bag):
    def inherit(self) -> None:
        obj_from = self.obj_from.obj_from
        self.update(
            obj_from.p_entities_multiplicity,
            obj_from.max_size
        )
        self.dis_entities = obj_from.dis_entities.copy()
        self.indis_entities = obj_from.indis_entities.copy()


class SetPartition(Partition):
    def __init__(self, obj_from: Set, size: int, ordered: bool) -> None:
        super().__init__(obj_from, size, ordered)
        self.partitioned_objs: list[SetPart] = [
            SetPart(self, i) for i in range(self.size)
        ]

    def encode(self, context: "Context") -> "Context":
        obj_from_pred = context.get_pred(self.obj_from)
        partitioned_preds = list(
            context.get_pred(obj, create=True)
            for obj in self.partitioned_objs
        )
        context.sentence = context.sentence & parse(
            "\\forall X: ({}(X) <-> ({}))".format(
                obj_from_pred, ' | '.join(
                    f"{pred}(X)" for pred in partitioned_preds
                )
            )
        )
        for i, obj1 in enumerate(self.partitioned_objs):
            for j, obj2 in enumerate(self.partitioned_objs):
                if i >= j:
                    continue
                pred1 = context.get_pred(obj1)
                pred2 = context.get_pred(obj2)
                context.sentence = context.sentence & parse(
                    "\\forall X: (~({}(X) & {}(X)))".format(
                        pred1, pred2
                    )
                )
        if not self.ordered:
            indis_vars = list()
            pre_var = None
            for obj in self.partitioned_objs:
                var = context.get_obj_var(obj)
                # make the variables non-decreasing to avoid duplications of partition
                if pre_var is not None:
                    context.validator.append(
                        pre_var <= var
                    )
                pre_var = var
                context.weighting[context.get_pred(obj)] = (var, 1)
                indis_vars.append([var])
            # see `decoder.py` for the meaning of `indis_vars`
            context.indis_vars.append(indis_vars)
        return context


class BagPartition(Partition):
    def __init__(self, obj_from: Bag, size: int, ordered: bool) -> None:
        super().__init__(obj_from, size, ordered)
        self.partitioned_objs: list[BagPart] = [
            BagPart(self, i) for i in range(self.size)
        ]

    def encode(self, context: "Context") -> "Context":
        obj_from_pred = context.get_pred(self.obj_from)
        partitioned_preds = list(
            context.get_pred(obj, create=True)
            for obj in self.partitioned_objs
        )
        context.sentence = context.sentence & parse(
            "\\forall X: ({}(X) <-> ({}))".format(
                obj_from_pred, ' | '.join(
                    f"{pred}(X)" for pred in partitioned_preds
                )
            )
        )
        # TODO: manually construct the weight for partition
        bag = self.obj_from
        ordered_vars = list([] for _ in range(len(self.partitioned_objs)))
        for entity in bag.dis_entities:
            multi = bag.p_entities_multiplicity[entity]
            context, entity_pred = entity.encode(context)
            if isinstance(bag, BagInit):
                multi_var = multi
            else:
                multi_var = context.get_entity_var(bag, entity)
            partitioned_vars = []
            for idx, obj in enumerate(self.partitioned_objs):
                pred = context.get_entity_pred(
                    obj, entity
                )
                obj_pred = context.get_pred(obj)
                context.sentence = context.sentence & parse(
                    "\\forall X: ({}(X) <-> ({}(X) & {}(X)))".format(
                        pred, obj_pred, entity_pred
                    )
                )
                var = context.get_entity_var(obj, entity)
                context.weighting[pred] = (
                    sum(var ** i for i in range(1, multi + 1)), 1
                )
                partitioned_vars.append(var)
                if not self.ordered:
                    ordered_vars[idx].append(var)
            context.validator.append(
                Eq(sum(partitioned_vars), multi_var)
            )
        if not self.ordered:
            for i in range(len(ordered_vars) - 1):
                context.validator.append(
                    ListLessThan(ordered_vars[i], ordered_vars[i + 1])
                )
        return context
