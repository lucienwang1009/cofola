from __future__ import annotations

from logzero import logger
from cofola.objects.bag import BagAdditiveUnion, BagChoose, BagDifference, BagInit, BagIntersection, BagMultiplicity, BagSupport, BagUnion, SizeConstraint
from cofola.objects.base import Bag, CombinatoricsBase, CombinatoricsObject, \
    CombinatoricsConstraint, Entity, Function, Part, Partition, Sequence, Set, Tuple
from cofola.objects.set import DisjointConstraint, MembershipConstraint, SetChoose, SetDifference, SetEqConstraint, SetInit, \
    SetChooseReplace, SetIntersection, SetUnion, SubsetConstraint
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, FuncPairConstraint
from cofola.objects.tuple import TupleCount, TupleImpl, TupleIndex, \
    TupleIndexEqConstraint, TupleMembershipConstraint
from cofola.objects.sequence import SequenceImpl, SequenceConstraint, SequencePattern
from cofola.objects.utils import IDX_PREFIX


class CofolaProblem(object):
    def __init__(
        self, objs: list[CombinatoricsObject] = None,
        constraints: list[CombinatoricsConstraint] = None
    ) -> None:
        super().__init__()
        self.objects: list[CombinatoricsObject] = objs
        self.constraints: list[CombinatoricsConstraint] = constraints
        if self.objects is None:
            self.objects = list()
        if self.constraints is None:
            self.constraints = list()
        self.entities: set[Entity] = set()
        # the entities that are always singletons in the problem
        self.singletons: set[Entity] = set()

    def update_entities(self) -> None:
        for obj in self.objects:
            if isinstance(obj, SetInit):
                self.entities.update(obj)
            if isinstance(obj, BagInit):
                self.entities.update(obj.keys())

    def update_singletons(self) -> None:
        singletons = set()
        for obj in self.objects:
            if isinstance(obj, SetChooseReplace):
                return
            if isinstance(obj, BagInit):
                singletons.update(
                    set(e for e, m in obj.items() if m == 1)
                )
        for obj in self.objects:
            # singletons in BagInit might become non-singletons through BagAdditiveUnion
            if isinstance(obj, BagAdditiveUnion):
                singletons = singletons.intersection(
                    set(e for e, m in obj.p_entities_multiplicity.items() if m == 1)
                )
        for cst in self.constraints:
            if isinstance(cst, SizeConstraint):
                for size_obj, _ in cst.expr:
                    if isinstance(size_obj, BagMultiplicity):
                        singletons = singletons.difference(set([size_obj.entity]))
        self.singletons = singletons

    def contains_entity(self, entity: Entity) -> bool:
        """
        Check if the entity is in the problem

        :param entity: the entity
        :return: True if the entity is in the problem, False otherwise
        """
        return entity in self.entities

    def propagate(self):
        """
        Propogate the properties of the objects
        """
        for obj in self.objects:
            obj.inherit()
        for constraint in self.constraints:
            if isinstance(constraint, SizeConstraint):
                if len(constraint.expr) == 1 and constraint.expr[0][1] == 1 \
                        and constraint.comp == "==" and constraint.expr[0][0].size is None:
                    constraint.expr[0][0].size = constraint.param

    def is_unsat(self) -> bool:
        for constraint in self.constraints:
            if isinstance(constraint, SizeConstraint):
                if len(constraint.expr) == 1 and constraint.expr[0][1] == 1:
                    if constraint.expr[0][0].size is not None:
                        if (constraint.comp == "=="
                            and constraint.expr[0][0].size != constraint.param) or \
                            (constraint.comp == '<='
                            and constraint.expr[0][0].size > constraint.param) or \
                            (constraint.comp == '>='
                            and constraint.expr[0][0].size < constraint.param) or \
                            (constraint.comp == '<'
                            and constraint.expr[0][0].size >= constraint.param) or \
                            (constraint.comp == '>'
                            and constraint.expr[0][0].size <= constraint.param):
                            logger.info(
                                f"Size constraint {constraint} is inconsistent with the size of {constraint.expr[0][0]}"
                            )
                            return True

    def topological_sort(self):
        """
        Topological sort the objects in the problem
        """
        # sort the objects by their dependences using Kahn’s Algorithm
        objs = self.objects
        in_degree = {obj: 0 for obj in objs}
        for obj in objs:
            for desc in obj.descendants:
                if isinstance(desc, CombinatoricsObject) and \
                        desc in objs:
                    in_degree[desc] += 1
        queue = [obj for obj in objs if in_degree[obj] == 0]
        sorted_objs = []
        while queue:
            obj = queue.pop()
            sorted_objs.append(obj)
            for desc in obj.descendants:
                if isinstance(desc, CombinatoricsObject) and \
                        desc in objs:
                    in_degree[desc] -= 1
                    if in_degree[desc] == 0:
                        queue.append(desc)
        if len(sorted_objs) != len(objs):
            raise ValueError("The problem contains a cycle")
        self.objects = sorted_objs

    def build(self):
        """
        Post-process the problem
        """
        self.topological_sort()
        self.update_entities()
        self.propagate()
        self.update_singletons()

    def add_object(self, obj: CombinatoricsObject) -> CombinatoricsObject:
        """
        Check if the object has already been defined. If so, return the existing object.
        Otherwise, add the object to the list of objects.

        :param obj: The object to be added.
        """
        existing_obj = list(
            filter(lambda x: obj.combinatorially_eq(x), self.objects)
        )
        if len(existing_obj) > 0:
            logger.info(f"Found an existing object {existing_obj[0]} that is equivalent to {obj}.")
            obj = existing_obj[0]
        else:
            if obj not in self.objects:
                self.objects.append(obj)
                if isinstance(obj, SetInit):
                    self.entities.update(obj)
                if isinstance(obj, BagInit):
                    self.entities.update(obj.keys())
            else:
                logger.info(f"Object {obj} already exists in the problem.")
        return obj

    def add_constraint(self, constraint: CombinatoricsConstraint):
        """
        Add a constraint to the problem

        :param constraint: the constraint
        """
        if constraint not in self.constraints:
            self.constraints.append(constraint)
        else:
            logger.info(f"Constraint {constraint} already exists in the problem.")

    def remove(self, item: CombinatoricsBase):
        """
        Remove an object or constraint from the problem

        :param item: the item to be removed
        """
        if isinstance(item, CombinatoricsObject):
            if item in self.objects:
                self.objects.remove(item)
            else:
                raise RuntimeError(
                    f"Object {item} not found in the problem."
                )
        elif isinstance(item, CombinatoricsConstraint):
            if item in self.constraints:
                self.constraints.remove(item)
            else:
                raise RuntimeError(
                    f"Constraint {item} not found in the problem."
                )
        else:
            raise ValueError(f"Unknown item type {item}")
        for dependence in item.dependences:
            if dependence in self.objects:
                dependence.descendants.remove(item)

    def __str__(self) -> str:
        s = "Entities:\n"
        s += "\t" + ", ".join(str(e) for e in self.entities) + "\n"
        s += '\n'
        s += "Objects:\n"
        for obj in self.objects:
            obj_name = type(obj).__name__
            s += f"\t{obj_name} {obj}\n"
        s += '\n'
        s += "Constraints:\n"
        for constraint in self.constraints:
            s += f"\t{constraint}\n"
        return s

    def __repr__(self) -> str:
        return str(self)

    def get_uncertain_objs(self) -> set[CombinatoricsObject]:
        """
        Find the objects that introduce uncertainty to the problem.

        :return: the uncertain objects
        """
        uncertain_objs = set(obj for obj in self.objects if obj.is_uncertain())
        return uncertain_objs

    def get_all_dependences(self, *base_items: CombinatoricsBase) \
            -> set[CombinatoricsObject]:
        """
        Get all the depended objects of the base items

        :param base_items: the base items
        :return: the depended objects
        """
        dependences = set()
        def helper(*objs: CombinatoricsBase) -> set[CombinatoricsObject]:
            # existing_objs = set(obj for obj in objs if obj in self.objects)
            existing_objs = objs
            dependences.update(existing_objs)
            for obj in existing_objs:
                helper(*obj.dependences)
        for base_item in base_items:
            helper(*base_item.dependences)
        return dependences

    def get_all_descendants(self, *src_objs: CombinatoricsObject) \
            -> set[CombinatoricsObject]:
        """
        Get all the objects that depend on the given objects, note only **objects** are returned

        :param base_items: the base items
        :return: the depended objects
        """
        descendants = set()
        def helper(*objs: CombinatoricsObject) -> set[CombinatoricsObject]:
            # existing_objs = set(obj for obj in objs if obj in self.objects)
            existing_objs = objs
            descendants.update(existing_objs)
            for obj in existing_objs:
                helper(*obj.descendants)
        for obj in src_objs:
            helper(*obj.descendants)
        return descendants

    def replace(self, old: CombinatoricsBase,
                new: CombinatoricsBase):
        """
        Replace the old object or constraint with the new one

        :param old: the old object or constraint
        :param new: the new object or constraint
        """
        ret = new
        if old in self.constraints:
            self.remove(old)
            self.add_constraint(new)
        elif old in self.objects:
            self.remove(old)
            # NOTE: in case the new object already exists in the problem
            ret = self.add_object(new)
            for descendant in old.descendants:
                descendant.subs_obj(old, ret)
        else:
            raise ValueError(f"Object or constraint {old} to replaced not found in the problem")
        return ret



