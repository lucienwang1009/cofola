from __future__ import annotations

from logzero import logger

from cofola.objects.bag import BagChoose, BagInit, BagMultiplicity, BagSupport, SizeConstraint
from cofola.objects.base import Bag, Entity, Sequence, Set, Tuple
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, FuncPairConstraint
from cofola.objects.set import (
    DisjointConstraint, MembershipConstraint, SetChoose, SetEqConstraint, SetInit, SetIntersection
)
from cofola.objects.tuple import (
    TupleCount, TupleImpl,
    TupleIndexEqConstraint, TupleMembershipConstraint
)
from cofola.objects.utils import IDX_PREFIX
from cofola.problem import CofolaProblem


def transform_tuples(problem: CofolaProblem) -> bool:
    for obj in problem.objects:
        if isinstance(obj, Tuple) and obj.mapping is None:
            obj_from = obj.obj_from
            if obj.choose:
                indices = problem.add_object(SetInit(
                    Entity(f"{IDX_PREFIX}{i}") for i in range(obj.size)
                ))
                if isinstance(obj_from, Set):
                    if obj.replace:
                        mapping = problem.add_object(
                            FuncInit(indices, obj_from)
                        )
                    else:
                        mapping = problem.add_object(
                            FuncInit(indices, obj_from, injective=True)
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
            else:
                indices = problem.add_object(SetInit(
                    Entity(f"{IDX_PREFIX}{i}") for i in range(obj_from.size)
                ))
                if isinstance(obj_from, Set):
                    mapping = problem.add_object(
                        FuncInit(indices, obj_from, surjective=True)
                    )

                if isinstance(obj_from, Bag):
                    support = problem.add_object(BagSupport(obj_from))
                    mapping = problem.add_object(
                        FuncInit(indices, support)
                    )
                    singletons = set()
                    reverse_images = list()
                    for entity in obj_from.p_entities_multiplicity.keys():
                        if entity in problem.singletons:
                            singletons.add(entity)
                        else:
                            reverse_image = problem.add_object(
                                FuncInverseImage(mapping, entity)
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
                            FuncImage(mapping, indices)
                        )
                        injectiveness_constraint = SetEqConstraint(
                            image, support
                        )
                        problem.add_constraint(injectiveness_constraint)
            logger.info(
                f"Transformed {obj} to {mapping}"
            )
            problem.replace(obj, TupleImpl(
                obj.obj_from, obj.choose, obj.replace, obj.size,
                indices, mapping))
            return True

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
                logger.info(f"Transformed size constraint {constraint} to {new_expr}")
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
