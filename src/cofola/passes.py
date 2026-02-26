from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

from logzero import logger
from cofola.objects.bag import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagInit,
    BagIntersection,
    BagMultiplicity,
    BagSupport,
    BagUnion,
    SizeConstraint,
)
from cofola.objects.base import (
    CombinatoricsBase,
    CombinatoricsConstraint,
    CombinatoricsObject,
    Entity,
    Sequence,
    Tuple,
)
from cofola.objects.set import (
    DisjointConstraint,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetUnion,
)
from cofola.objects.function import FuncImage
from cofola.objects.tuple import TupleIndex
from cofola.objects.sequence import SequenceConstraint
from cofola.problem import CofolaProblem


def sanity_check(problem: CofolaProblem) -> None:
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


def fold_constants(problem: CofolaProblem) -> bool:
    """
    Fold the constant objects

    :param problem: the problem
    :return: True if any constants were folded, False otherwise
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


def infer_max_size(problem: CofolaProblem) -> CofolaProblem:
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
    # propagate the inferred maximum sizes
    problem.propagate()
    return problem


def workaround(problem: CofolaProblem) -> CofolaProblem:
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
    return problem


def optimize(problem: CofolaProblem) -> CofolaProblem:
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
    return problem


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


def decompose_problem(problem: CofolaProblem) -> list[CofolaProblem]:
    """
    Decompose a combinatorics problem into disjoint sub-problems, which is achieved by finding the connected components in the "problem graph".

    :param problem: the combinatorics problem
    :return: the list of sub-problems
    """
    problems = list()
    visited = set()
    components = list()
    items = problem.objects + problem.constraints

    def visit(item: CombinatoricsBase):
        if item in visited:
            return []
        visited.add(item)
        ret = [item]
        for i in item.dependences.union(item.descendants):
            if i in items:
                ret.extend(visit(i))
        return ret

    for item in items:
        component = visit(item)
        if len(component) > 0:
            components.append(component)
    for obj_constraint in components:
        objs = list(o for o in obj_constraint if isinstance(o, CombinatoricsObject))
        constraints = list(o for o in obj_constraint if isinstance(o, CombinatoricsConstraint))
        decomposed_problem = CofolaProblem(objs, constraints)
        problems.append(decomposed_problem)
    return problems
