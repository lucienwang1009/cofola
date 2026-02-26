"""
Pipeline for solving combinatorics problems via WFOMC.

The SolvePipeline class encapsulates the full solve flow:
    build → simplify → decompose → per-sub: build → is_unsat → simplify → optimize →
    infer_max_size → sanity_check → transform → optimize → workaround → simplify →
    sanity_check → encode → solve_wfomc
"""
from __future__ import annotations

from copy import deepcopy
from functools import reduce

from logzero import logger
from sympy import Symbol, satisfiable
from wfomc import Algo

from cofola.encoder import encode
from cofola.objects.bag import SizeConstraint
from cofola.objects.base import (
    And,
    AtomicConstraint,
    CombinatoricsConstraint,
    CombinatoricsObject,
    Negation,
    Or,
    Tuple,
)
from cofola.passes import (
    decompose_problem,
    infer_max_size,
    optimize,
    sanity_check,
    simplify,
    workaround,
)
from cofola.problem import CofolaProblem
from cofola.transforms import transform
from cofola.wfomc_solver import solve as solve_wfomc


class SolvePipeline(object):
    """
    Pipeline for solving combinatorics problems via WFOMC.
    """

    def __init__(
        self,
        wfomc_algo: Algo = Algo.FASTv2,
        use_partition_constraint: bool = True,
        lifted: bool = False,
    ) -> None:
        """
        Initialize the solve pipeline.

        :param wfomc_algo: WFOMC algorithm to use
        :param use_partition_constraint: whether to use partition constraint
        :param lifted: whether to use lifted encoding for bags
        """
        self.wfomc_algo = wfomc_algo
        self.use_partition_constraint = use_partition_constraint
        self.lifted = lifted

    def run(self, problem: CofolaProblem) -> int:
        """
        Solve a combinatorics problem that may contain compound constraints.

        :param problem: the combinatorics problem
        :return: the answer
        """
        # Workaround for unknown length tuples
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

        logger.info("The original problem:")
        logger.info(problem)

        if len(problem.constraints) == 0:
            return self._solve_sub(problem)

        # Handle compound constraints via SAT enumeration
        constraints = list()
        constraintidx2atom = dict()
        atom2constraintidx = dict()

        def build_formula(constraint: CombinatoricsConstraint):
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
            logger.info('Solving a sub-problem under the constraint:')
            # Deep copy objects and constraints to avoid side effects
            sub_problem = deepcopy(
                CofolaProblem(problem.objects, constraints)
            )
            for idx, constraint in enumerate(sub_problem.constraints):
                if not model[constraintidx2atom[idx]]:
                    constraint.negate()
            logger.info(sub_problem.constraints)
            sub_answer = self._solve_sub(sub_problem)
            logger.info(f'Answer for the sub-problem: {sub_answer}')
            answer += sub_answer
        return answer

    def _solve_sub(self, problem: CofolaProblem) -> int:
        """
        Solve a single sub-problem with no compound constraints.

        :param problem: the sub-problem (atomic constraints only)
        :return: the answer
        """
        problem.build()
        problem = simplify(problem)

        final = 1
        for p in decompose_problem(problem):
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
            p = optimize(p)

            logger.info("Inferring the maximum size of sized objects...")
            p = infer_max_size(p)

            logger.info("Transforming the problem...")
            sanity_check(p)
            p = transform(p)
            logger.info(p)

            logger.info("Optimizing the problem...")
            p = optimize(p)
            p = workaround(p)

            logger.info('Simplifying the problem...')
            p = simplify(p)
            logger.info(p)
            sanity_check(p)

            logger.info(f'The problem for encoding: \n{p}')
            wfomc_problem, decoder = encode(p, self.lifted)
            logger.info(f'Encoded WFOMC problem: \n{wfomc_problem}')
            logger.info(f'Result decoder: \n{decoder}')

            wfomc_algo = self.wfomc_algo
            use_partition_constraint = self.use_partition_constraint
            if wfomc_problem.contain_linear_order_axiom() and \
                    wfomc_algo != Algo.INCREMENTAL and wfomc_algo != Algo.RECURSIVE:
                logger.warning(
                    'Linear order axiom with the predicate LEQ is found, '
                    'while the algorithm is not INCREMENTAL or RECURSIVE. '
                    'Switching to INCREMENTAL algorithm...'
                )
                wfomc_algo = Algo.INCREMENTAL
                use_partition_constraint = True

            ret = solve_wfomc(wfomc_problem, wfomc_algo, use_partition_constraint)
            logger.debug(f'WFOMC solver result: {ret}')
            ret = decoder.decode_result(ret)

            if ret is None:
                logger.info('The problem is unsatisfiable')
                return 0

            logger.info(f'Answer for the decomposed problem: {ret}')
            final = final * ret

        return final