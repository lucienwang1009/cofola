from collections import defaultdict
import logging

from logzero import logger
from functools import reduce
import argparse
from wfomc import Algo
from symengine import expand
from sympy import O, Symbol, comp, satisfiable
from copy import deepcopy

from cofola.encoder import encode
from cofola.objects.bag import SizeConstraint
from cofola.objects.base import AtomicConstraint, CombinatoricsBase, CombinatoricsConstraint, CombinatoricsObject, Negation, And, Or, Tuple
from cofola.wfomc_solver import solve as solve_wfomc
from cofola.parser.parser import parse
from cofola.problem import CofolaProblem, infer_max_size, optimize, \
    sanity_check, simplify, transform, workaround

def parse_args():
    parser = argparse.ArgumentParser(description='Solve a combinatorics math problem using weighted first-order model counting')
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    # parser.add_argument('--wfomc_algo', '-a', type=Algo,
    #                     choices=list(Algo), default=Algo.FASTv2)
    # parser.add_argument('--use_partition_constraint', '-p', action='store_true',
    #                     help='use partition constraint to speed up the solver, '
    #                          'it would override the algorithm choice to FASTv2')
    # parser.add_argument('--lifted', '-l', action='store_true',
    #                     help='use lifted encoding for bags')
    return parser.parse_args()


def solve_single_problem(problem: CofolaProblem, wfomc_algo: Algo,
                         use_partition_constraint: bool = False,
                         lifted: bool = True) -> int:
    """
    Solve a single combinatorics problem that contains no compound constraints, i.e.,
    the constraints are all atomic constraints and formed by conjunctions.

    :param problem: the combinatorics problem
    :param wfomc_algo: the algorithm to solve the problem
    :param use_partition_constraint: whether to use partition constraint
    :param lifted: whether to use lifted encoding for bags
    :return: the answer
    """
    problem.build()
    problem = simplify(problem)
    final = 1
    for p in decompose_problem(problem):
        print('=====================')
        print(p)
        print('=====================')
        p.build()
        logger.info(f'Solving a decomposed problem')
        logger.info(p)
        if p.is_unsat():
            logger.info('The problem is unsatisfiable')
            return 0
        logger.info('Simplifying the problem...')
        p = simplify(p)
        logger.info(p)
        logger.info("Optimizing the problem...")
        optimize(p)
        logger.info("Inferring the maximum size of sized objects...")
        infer_max_size(p)
        logger.info("Transforming the problem...")
        sanity_check(p)
        p = transform(p)
        logger.info(p)
        logger.info("Optimizing the problem...")
        optimize(p)
        workaround(p)
        logger.info('Simplifying the problem...')
        p = simplify(p)
        logger.info(p)
        sanity_check(p)
        logger.info(f'The problem for encoding: \n{p}')
        wfomc_problem, decoder, full_circle = encode(p, lifted)
        logger.info(f'Encoded WFOMC problem: \n{wfomc_problem}')
        logger.info(f'Result decoder: \n{decoder}')
        if wfomc_problem.contain_linear_order_axiom() and \
                wfomc_algo != Algo.INCREMENTAL and wfomc_algo != Algo.RECURSIVE:
            logger.warning(
                'Linear order axiom with the predicate LEQ is found, '
                'while the algorithm is not INCREMENTAL or RECURSIVE. '
                'Switching to INCREMENTAL algorithm...'
            )
            wfomc_algo = Algo.INCREMENTAL
            use_partition_constraint = True
        ret = None
        # ret = expand(solve_wfomc(wfomc_problem, wfomc_algo,
        #                          use_partition_constraint))
        # logger.debug(f'WFOMC solver result: {ret}')
        # ret = decoder.decode_result(ret)
        if ret is None:
            logger.info('The problem is unsatisfiable')
            return 0, wfomc_problem, full_circle
        logger.info(f'Answer for the decomposed problem: {ret}')
        final = final * ret
    return final, wfomc_problem, full_circle


def decompose_problem(problem: CofolaProblem) -> list[CofolaProblem]:
    """
    Decompose a combinatorics problem into disjoint sub-problems, which is achieve by finding the connected components in the "problem graph".

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


def solve(problem: CofolaProblem,
          wfomc_algo: Algo = Algo.FASTv2,
          use_partition_constraint: bool = False,
          lifted: bool = True) -> int:
    """
    Solve a combinatorics problem that may contain compound constraints

    :param problem: the combinatorics problem
    :param wfomc_algo: the algorithm to solve the problem
    :param use_partition_constraint: whether to use partition constraint
    :param lifted: whether to use lifted encoding for bags
    :return: the answer
    """
    # ============================================
    # NOTE: workaround for unknown length tuples
    for constraint in problem.constraints:
        if isinstance(constraint, SizeConstraint) and \
                len(constraint.expr) == 1 and \
                isinstance(constraint.expr[0][0], Tuple) and \
                constraint.expr[0][1] == 1 and \
                constraint.comp in ['<', '<=']:
            param = constraint.param
            if constraint.comp == '<':
                param -= 1
            problem.remove(constraint)
            problem.add_constraint(
                reduce(
                    lambda x, y: x | y,
                    (SizeConstraint([(constraint.expr[0][0], 1)], '==', i)
                     for i in range(2, constraint.param + 1)),
                    SizeConstraint([(constraint.expr[0][0], 1)], '==', 1)
                )
            )
    # ============================================
    logger.info("The original problem:")
    logger.info(problem)
    if len(problem.constraints) == 0:
        answer, wfomc_problem, full_circle = solve_single_problem(
            problem, wfomc_algo, use_partition_constraint, lifted
        )
        return answer, wfomc_problem, full_circle
    constraints = list()
    constraintidx2atom = dict()
    atom2constraintidx = dict()
    def build_formula(constraint):
        if isinstance(constraint, AtomicConstraint):
            if constraint in constraints:
                idx = constraints.index(constraint)
                return constraintidx2atom[idx]
            else:
                atom = Symbol(f'c_{len(constraints)}')
                constraintidx2atom[len(constraints)] = atom
                atom2constraintidx[atom] = len(constraints)
                constraints.append(constraint)
                return atom
        elif isinstance(constraint, Negation):
            return ~build_formula(constraint.sub_constraint)
        elif isinstance(constraint, And):
            return build_formula(constraint.first_constraint) & \
                build_formula(constraint.second_constraint)
        elif isinstance(constraint, Or):
            return build_formula(constraint.first_constraint) | \
                build_formula(constraint.second_constraint)
    formula = True
    for constraint in problem.constraints:
        formula = formula & build_formula(constraint)
    logger.info('Building the formula for constraints:')
    logger.info(formula)
    logger.info('Constraint to atom mapping:')
    logger.info(list(
        (constraints[i], atom)
        for i, atom in constraintidx2atom.items()
    ))
    answer = 0
    for model in satisfiable(formula, all_models=True):
        logger.info('Sovling a sub-problem under the constraint:')
        # NOTE: solve the sub-problem with deep copied objects and constraints
        # to avoid the side effect on the original problem
        sub_problem = deepcopy(
            CofolaProblem(
                problem.objects, constraints
            )
        )
        for idx, constraint in enumerate(sub_problem.constraints):
            if not model[constraintidx2atom[idx]]:
                constraint.negate()
        logger.info(sub_problem.constraints)
        sub_answer, wfomc_problem, full_circle = solve_single_problem(
            sub_problem, wfomc_algo, use_partition_constraint, lifted
        )
        logger.info(f'Answer for the sub-problem: {sub_answer}')
        answer += sub_answer
    return answer, wfomc_problem, full_circle


def main(args):
    input_file = args.input_file
    logger.info(f'Input file: {input_file}')
    with open(input_file, 'r') as f:
        problem: CofolaProblem = parse(f.read())
    logger.info(f'Problem: \n{problem}')
    res, wfomc_problem, full_circle = solve(
        problem,
        # args.wfomc_algo,
        # args.use_partition_constraint,
        # args.lifted
    )
    return res, wfomc_problem, full_circle


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    res = main(args)
    logger.info(f'Answer: {res}')
