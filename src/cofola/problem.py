from __future__ import annotations
from collections import defaultdict
from os import replace

import numpy as np
from scipy.optimize import linprog
from functools import reduce

from logzero import logger
from cofola.objects.bag import BagAdditiveUnion, BagChoose, BagDifference, BagInit, BagIntersection, BagMultiplicity, BagSupport, BagUnion, SizeConstraint
from cofola.objects.base import Bag, CombinatoricsBase, CombinatoricsObject, \
    CombinatoricsConstraint, Entity, Function, Part, Partition, Sequence, Set, Tuple
from cofola.objects.set import DisjointConstraint, MembershipConstraint, SetChoose, SetDifference, SetEqConstraint, SetInit, \
    SetChooseReplace, SetIntersection, SetUnion, SubsetConstraint
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, FuncPairConstraint
from cofola.objects.tuple import TupleCount, TupleIndex, \
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
        self.singletons = singletons

    def contains_entity(self, entity: Entity) -> bool:
        """
        Check if the entity is in the problem

        :param entity: the entity
        :return: True if the entity is in the problem, False otherwise
        """
        return entity in self.entities

    def propogate(self):
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
        self.propogate()
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


def sanity_check(problem: CofolaProblem):
    """
    Check if the problem is well-defined
    """
    # 1. check there is at least one object
    if len(problem.objects) == 0:
        raise RuntimeError("No object found in the problem")

    for obj in problem.objects:
        # 2. check if all chooseReplace have finite multiplicity
        if isinstance(obj, SetChooseReplace):
            if any(m == float('inf') for m in obj.p_entities_multiplicity.values()):
                raise ValueError(f"The size of bag {obj.name} is not specified and unable to be inferred")
        # 3. check if the size of tuple is given
        if (isinstance(obj, Tuple) and obj.size is None) or \
                (isinstance(obj, Sequence) and obj.size is None):
            raise ValueError(f"The size of tuple {obj.name} is not specified and unable to be inferred")
        # 4. check if the index of a tuple is within the range
        if isinstance(obj, TupleIndex) and obj.index >= obj.obj_from.size:
            raise ValueError(f"Index out of bound: {obj.index} >= {obj.obj_from.size}")


def fold_constants(problem: CofolaProblem) -> None:
    """
    Fold the constant objects

    :param problem: the problem
    """
    # TODO: fold the constant objects
    ret = False
    for obj in problem.objects:
        if isinstance(obj, BagSupport) and isinstance(obj.obj_from, BagInit):
            new_obj = SetInit(obj.obj_from.keys())
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, SetUnion) and \
                all(isinstance(o, SetInit) for o in [obj.first, obj.second]):
            new_obj = SetInit(set.union(
                *[o.p_entities for o in [obj.first, obj.second]]
            ))
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, SetIntersection) and \
                all(isinstance(o, SetInit) for o in [obj.first, obj.second]):
            new_obj = SetInit(set.intersection(
                *[o.p_entities for o in [obj.first, obj.second]]
            ))
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, SetDifference) and \
                all(isinstance(o, SetInit) for o in [obj.first, obj.second]):
            new_obj = SetInit(set.difference(
                obj.first.p_entities, obj.second.p_entities
            ))
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, BagAdditiveUnion) and \
                all(isinstance(o, BagInit) for o in [obj.first, obj.second]):
            new_obj = BagInit(
                {
                    e: obj.first.p_entities_multiplicity.get(e, 0) +
                       obj.second.p_entities_multiplicity.get(e, 0)
                    for e in set(obj.first.p_entities_multiplicity.keys()).union(
                        set(obj.second.p_entities_multiplicity.keys())
                    )
                }
            )
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, BagIntersection) and \
                all(isinstance(o, BagInit) for o in [obj.first, obj.second]):
            new_obj = BagInit(
                {
                    e: min(
                        obj.first.p_entities_multiplicity.get(e, 0),
                        obj.second.p_entities_multiplicity.get(e, 0)
                    )
                    for e in set(obj.first.p_entities_multiplicity.keys()).intersection(
                        set(obj.second.p_entities_multiplicity.keys())
                    )
                }
            )
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
        if isinstance(obj, BagDifference) and \
                all(isinstance(o, BagInit) for o in [obj.first, obj.second]):
            new_obj = BagInit(
                {
                    e: obj.first.p_entities_multiplicity.get(e, 0) -
                        obj.second.p_entities_multiplicity.get(e, 0)
                    for e in set(obj.first.p_entities_multiplicity.keys())
                    if obj.first.p_entities_multiplicity.get(e, 0) -
                       obj.second.p_entities_multiplicity.get(e, 0) > 0
                }
            )
            problem.replace(obj, new_obj)
            logger.info(f"Folded {obj} to {new_obj}")
            ret = True
    # fold constraints
    for constraint in problem.constraints:
        # substitute the constant objects in size constraints
        if isinstance(constraint, SizeConstraint):
            folded = False
            remaining_expr = []
            param = constraint.param
            for obj, coef in constraint.expr:
                size = obj.size
                if size is not None:
                    param -= coef * size
                    folded = True
                else:
                    remaining_expr.append((obj, coef))
            if folded and len(remaining_expr) > 0:
                problem.replace(
                    constraint,
                    SizeConstraint(remaining_expr, constraint.comp, param)
                )
                logger.info(f"Folded constant objects in {constraint}")
                ret = True
    # TODO: remove the constraint if it is always satisfied
    problem.build()
    return ret


def add_disjoint_constraints(problem: CofolaProblem) -> None:
    """
    Add disjoint constraints to the problem

    :param problem: the problem
    """
    func_images = defaultdict(list)
    for obj in problem.objects:
        if isinstance(obj, FuncImage) and isinstance(obj.set_or_entity, Entity):
            func_images[obj.func.name].append(obj)
    # add pairwise disjoint constraints
    for func_name, images in func_images.items():
        if len(images) > 1:
            for i, img1 in enumerate(images):
                for j, img2 in enumerate(images):
                    if i < j:
                        disjoint_constraint = DisjointConstraint(img1, img2)
                        problem.add_constraint(disjoint_constraint)
                        logger.info(f"Added disjoint constraint {disjoint_constraint}")

def infer_max_size(problem: CofolaProblem):
    """
    Infer the maximum size of basic objects, i.e., sets and bags from the size constraints

    :param problem: the problem
    """
    constrained_objs = set()
    n_size_constraints = 0
    for constraint in problem.constraints:
        if isinstance(constraint, SizeConstraint):
            n_size_constraints += 1
            for obj, coef in constraint.expr:
                constrained_objs.add(obj)
    constrained_objs = list(constrained_objs)
    logger.info(f"Constrained objects: {constrained_objs}")
    A_u = np.zeros((n_size_constraints, len(constrained_objs)))
    b_u = np.zeros(n_size_constraints)
    A_e = np.zeros((n_size_constraints, len(constrained_objs)))
    b_e = np.zeros(n_size_constraints)
    i = 0
    for constraint in problem.constraints:
        if isinstance(constraint, SizeConstraint):
            comp = constraint.comp
            if comp == '==':
                b_e[i] = constraint.param
            elif comp == '<=':
                b_u[i] = constraint.param
            elif comp == '>=':
                b_u[i] = -constraint.param
            elif comp == '<':
                b_u[i] = constraint.param - 1
            elif comp == '>':
                b_u[i] = -constraint.param + 1
            else:
                continue
            for obj, coef in constraint.expr:
                index = constrained_objs.index(obj)
                if comp == '==':
                    A_e[i, index] = coef
                elif comp in ('<=', '<'):
                    A_u[i, index] = coef
                else:
                    A_u[i, index] = -coef
            i += 1
    for obj in constrained_objs:
        index = constrained_objs.index(obj)
        c = np.zeros(len(constrained_objs))
        c[index] = -1
        ret = linprog(c, A_u, b_u, A_e, b_e)
        if ret.success:
            size = int(ret.x[index])
            logger.info(f"Inferred maximum size of {obj.name} = {size}")
            obj.max_size = size
    # propogate the inferred maximum sizes
    problem.propogate()


def transform_tuples(problem: CofolaProblem) -> bool:
    for obj in problem.objects:
        if isinstance(obj, Tuple):
            if obj.choose:
                indices = SetInit(
                    Entity(f"{IDX_PREFIX}{i}") for i in range(obj.size)
                )
                obj.indices = problem.add_object(indices)
                obj_from = obj.obj_from
                if isinstance(obj_from, Set):
                    if obj.replace:
                        obj.mapping = problem.add_object(
                            FuncInit(obj.indices, obj_from)
                        )
                    else:
                        obj.mapping = problem.add_object(
                            FuncInit(obj.indices, obj_from, injective=True)
                        )
                if isinstance(obj_from, Bag):
                    if obj.replace:
                        raise RuntimeError(
                            "Choose from a bag with replacement is not supported, "
                            "please convert it to a set by support operation and then choose "
                            "from the set instead"
                        )
                    # decompose to choose and tuple
                    choosing_obj = problem.add_object(
                        BagChoose(obj_from, obj.size)
                    )
                    obj.subs_args(
                        choosing_obj, False, False, None,
                    )
                    return True
                    support = problem.add_object(BagSupport(obj_from))
                    obj.mapping = problem.add_object(
                        FuncInit(obj.indices, support)
                    )
                    singletons = set()
                    reverse_images = list()
                    for entity in obj_from.p_entities_multiplicity.keys():
                        if entity in problem.singletons:
                            singletons.add(entity)
                        else:
                            reverse_image = problem.add_object(
                                FuncInverseImage(obj.mapping, entity)
                            )
                            if isinstance(obj_from, BagInit):
                                injectiveness_constraint = SizeConstraint(
                                    [(reverse_image, 1)], "<=",
                                    obj_from.p_entities_multiplicity[entity]
                                )
                            else:
                                # TODO: if the bag is chosen from another bag, the multiplicity is unknown
                                raise NotImplementedError(
                                    "Constructing a tuple from a bag chosen from another bag is not supported"
                                )
                            problem.add_constraint(injectiveness_constraint)
                            reverse_images.append(reverse_image)
                    for i, image1 in enumerate(reverse_images):
                        for j, image2 in enumerate(reverse_images):
                            if i < j:
                                problem.add_constraint(
                                    DisjointConstraint(image1, image2)
                                )
                    if len(singletons) > 0:
                        singletons_set = problem.add_object(
                            SetInit(singletons)
                        )
                        func_image = problem.add_object(
                            FuncImage(obj.mapping, obj.indices)
                        )
                        singletons_img = problem.add_object(
                            SetIntersection(func_image, singletons_set)
                        )
                        injectiveness_constraint = SizeConstraint(
                            [(singletons_img, 1)] + list(
                                (reverse_image, 1) for reverse_image in reverse_images
                            ), "==", obj.size
                        )
                        problem.add_constraint(injectiveness_constraint)
            else:
                obj_from = obj.obj_from
                indices = SetInit(
                    Entity(f"{IDX_PREFIX}{i}") for i in range(obj_from.size)
                )
                obj.indices = problem.add_object(indices)
                if isinstance(obj_from, Set):
                    obj.mapping = problem.add_object(
                        FuncInit(obj.indices, obj_from, surjective=True)
                    )

                if isinstance(obj_from, Bag):
                    support = problem.add_object(BagSupport(obj_from))
                    obj.mapping = problem.add_object(
                        FuncInit(obj.indices, support)
                    )
                    singletons = set()
                    reverse_images = list()
                    for entity in obj_from.p_entities_multiplicity.keys():
                        if entity in problem.singletons:
                            singletons.add(entity)
                        else:
                            reverse_image = problem.add_object(
                                FuncInverseImage(obj.mapping, entity)
                            )
                            reverse_images.append(reverse_image)
                            if isinstance(obj_from, BagInit):
                                injectiveness_constraint = SizeConstraint(
                                    [(reverse_image, 1)], "==",
                                    obj_from.p_entities_multiplicity[entity]
                                )
                            else:
                                # TODO: if the bag is chosen from another bag, the multiplicity is unknown
                                entity_multiplicity = problem.add_object(BagMultiplicity(
                                    obj_from, entity
                                ))
                                injectiveness_constraint = SizeConstraint(
                                    [
                                        (reverse_image, 1),
                                        (entity_multiplicity, -1)
                                    ],
                                    "==", 0
                                )
                            problem.add_constraint(injectiveness_constraint)
                    for i, image1 in enumerate(reverse_images):
                        for j, image2 in enumerate(reverse_images):
                            if i < j:
                                problem.add_constraint(
                                    DisjointConstraint(image1, image2)
                                )
                    if len(singletons) > 0:
                        image = problem.add_object(
                            FuncImage(obj.mapping, obj.indices)
                        )
                        injectiveness_constraint = SetEqConstraint(
                            image, support
                        )
                        problem.add_constraint(injectiveness_constraint)
            logger.info(
                f"Transformed {obj} to {obj.indices} and {obj.mapping}"
            )
            problem.remove(obj)
            return True
        # if isinstance(obj, TupleIndex):
        #     func_img = FuncImage(obj.obj_from.mapping,
        #                          Entity(f"{IDX_PREFIX}{obj.index}"))
        #     problem.replace(obj, func_img)
        #     logger.info(f"Transformed {obj} to {func_img}")
        #     return True

    for constraint in problem.constraints:
        if isinstance(constraint, TupleIndexEqConstraint):
            obj, entity = constraint.obj, constraint.entity
            index = Entity(f"{IDX_PREFIX}{obj.index}")
            new_constraint = FuncPairConstraint(
                obj.obj_from.mapping, index, entity
            )
            new_constraint.positive = constraint.positive
            problem.replace(constraint, new_constraint)
            return True
            # membership_constraint = MembershipConstraint(
            #     constraint.obj, constraint.entity
            # )
            # membership_constraint.positive = constraint.positive
            # problem.add_constraint(membership_constraint)
            # problem.remove(constraint)
            # return True
        if isinstance(constraint, TupleMembershipConstraint):
            # tuple[index] in set/bag/tuple or entity in tuple
            entity_or_tuple_index, obj = constraint.member, constraint.obj
            if isinstance(obj, Tuple):
                if not obj.choose:
                    obj = obj.obj_from
                else:
                    obj = problem.add_object(
                        FuncImage(obj.mapping, obj.indices)
                    )
            if isinstance(entity_or_tuple_index, Entity):
                # entity in tuple
                new_constraint = MembershipConstraint(
                    obj, entity_or_tuple_index
                )
            else:
                # tuple[index] in tuple
                index = Entity(f"{IDX_PREFIX}{entity_or_tuple_index.index}")
                new_constraint = FuncPairConstraint(
                    entity_or_tuple_index.obj_from.mapping,
                    index, obj
                )
            new_constraint.positive = constraint.positive
            problem.replace(constraint, new_constraint)
            logger.info(f"Transformed {constraint} to {new_constraint}")
            return True
    return False


def transform_sequences(problem: CofolaProblem) -> bool:
    for obj in problem.objects:
        if isinstance(obj, Sequence):
            obj_from, choose, replace, size, \
                circular, reflection, flatten = obj.args
            if choose and not replace:
                if isinstance(obj_from, Set):
                    chosen_obj = problem.add_object(
                        SetChoose(obj_from, size)
                    )
                else:
                    chosen_obj = problem.add_object(
                        BagChoose(obj_from, size)
                    )
                obj.subs_args(
                    chosen_obj, False, False, None,
                    circular, reflection, flatten
                )
                return True
            if (isinstance(obj_from, Bag) or (choose and replace)) and \
                    obj.flatten_obj is None:
                flatten = problem.add_object(
                    SetInit(
                        Entity(f"{IDX_PREFIX}{i}") for i in range(obj.size)
                    )
                )
                obj.subs_args(
                    *(obj.args[:-1] + (flatten, ))
                )
                return True
    return False


def transform_functions(problem: CofolaProblem) -> bool:
    for obj in problem.objects:
        if isinstance(obj, FuncInit) and obj.injective:
            obj = problem.replace(
                obj, FuncInit(obj.domain, obj.codomain, False, obj.surjective)
            )
            img_obj = FuncImage(obj, obj.domain)
            img_obj = problem.add_object(img_obj)
            if obj.domain.size is not None:
                injectiveness_constraint = SizeConstraint(
                    [(img_obj, 1)], "==", obj.domain.size
                )
            else:
                injectiveness_constraint = SizeConstraint(
                    [(img_obj, 1), (obj.domain, -1)], "==", 0
                )
            problem.add_constraint(injectiveness_constraint)
            logger.info(f"Transformed function {obj} to {obj} with size constraint {injectiveness_constraint}")
            return True
    return False


def transform_size_constraint(problem: CofolaProblem) -> bool:
    for constraint in problem.constraints:
        # size and count constraint on tuples -> size constraint on function inverse image
        if isinstance(constraint, SizeConstraint):
            transformed = False
            new_expr = []
            for obj, coef in constraint.expr:
                if isinstance(obj, Tuple):
                    new_expr.append((obj.indices, coef))
                    transformed = True
                elif isinstance(obj, TupleCount):
                    tuple_obj, count_obj = obj.tuple_obj, obj.count_obj
                    if not tuple_obj.choose and obj.deduplicate:
                        # Note the count_obj is always an entity or a set
                        transformed_obj = problem.add_object(
                            SetIntersection(
                                tuple_obj.mapping.codomain, count_obj
                            )
                        )
                    else:
                        if not obj.deduplicate:
                            func_inverse_img = FuncInverseImage(
                                obj.tuple_obj.mapping, obj.count_obj
                            )
                            transformed_obj = problem.add_object(
                                func_inverse_img)
                        else:
                            func_img = FuncImage(
                                obj.tuple_obj.mapping, obj.tuple_obj.indices
                            )
                            func_img = problem.add_object(
                                func_img)
                            transformed_obj = problem.add_object(
                                SetIntersection(func_img, obj.count_obj)
                            )
                    new_expr.append((transformed_obj, coef))
                    transformed = True
                else:
                    new_expr.append((obj, coef))
            if transformed:
                problem.replace(
                    constraint,
                    SizeConstraint(new_expr, constraint.comp, constraint.param)
                )
                return True
    return False


def transform_once(problem: CofolaProblem) -> bool:
    if transform_tuples(problem):
        return True
    if transform_sequences(problem):
        return True
    if transform_functions(problem):
        return True
    if transform_size_constraint(problem):
        return True
    return False


def transform(problem: CofolaProblem) -> CofolaProblem:
    """
    Transform the problem such that some constraints, e.g., injective are replaced
    by set size constraints, and tuples are replaced by functions.
    """
    while transform_once(problem):
        problem.update_entities()
        problem.propogate()
        problem.topological_sort()
    return problem


def workaround(problem: CofolaProblem):
    # NOTE: workaround for lifting BagChoose with size constraint
    # as now the encoding for distinguishable and indistinguishable entities are separated,
    # the size constraint is postponed to the encoding of indistinguishable entities
    for obj in problem.objects:
        if isinstance(obj, (SetChooseReplace, BagChoose)) and \
                obj.size is not None:
            # NOTE: here the number 1 in `(self, 1)` should match the type of `self.size` due to
            # the equality in symengine respects the term types
            size_constraint = SizeConstraint([(obj, 1)], "==", obj.size)
            problem.add_constraint(size_constraint)
    problem.build()


def optimize(problem: CofolaProblem) -> None:
    """
    Optimize the problem by applying the following techniques:
    1. constant folding

    :param problem: the original problem
    :return: the optimized problem
    """
    optimizing = True
    while optimizing:
        optimizing = fold_constants(problem)
    # add_disjoint_constraints(problem)


def simplify(problem: CofolaProblem) -> CofolaProblem:
    """
    Simplify the problem by removing redundant constraints and objects.

    :param problem: the original problem
    :return: the simplified problem
    """
    ret_problem = CofolaProblem()
    # find all uncertain objects, which are necessary for the problem
    necessary_objs: set[str] = problem.get_uncertain_objs()
    logger.info(f"Uncertain objs: {list(obj.name for obj in necessary_objs)}")
    # remove all constraints that are not dependent on the necessary objs
    for constraint in problem.constraints:
        # NOTE: alway keep the sequence constraint
        if necessary_objs.intersection(
            problem.get_all_dependences(constraint)
        ) or isinstance(constraint, SequenceConstraint):
            ret_problem.add_constraint(constraint)
    logger.info(f"Simplified constraints: {ret_problem.constraints}")
    # the dependent objects of the simplified constraints are also necessary
    for constraint in ret_problem.constraints:
        necessary_objs.update(
            [obj for obj in constraint.dependences]
        )
    logger.info(f"Necessary objs: {list(obj.name for obj in necessary_objs)}")
    # the dependent objects of the necessary objects are also necessary
    necessary_objs.update(
        problem.get_all_dependences(*necessary_objs)
    )
    for obj in problem.objects:
        if obj in necessary_objs:
            ret_problem.add_object(obj)
    logger.info(f"Simplied objs: {list(obj.name for obj in ret_problem.objects)}")
    ret_problem.build()
    return ret_problem


def decompose(problem: CofolaProblem) -> list[CofolaProblem]:
    """
    Decompose the problem into sub-problems, which are independent of each other.
    If the problem cannot be decomposed, return the original problem.

    :param problem: the original problem
    :return: the list of sub-problems
    """
    # Sub-problems are the sets of objects and constraints that form a connected component
    # of the dependency graph of the original problem.
    # We can use the dfs algorithm to find the connected components.
    pass
