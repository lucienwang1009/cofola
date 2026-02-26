from __future__ import annotations

import argparse
import logging

from logzero import logger
from wfomc import Algo

from cofola.parser.parser import parse
from cofola.pipeline import SolvePipeline
from cofola.problem import CofolaProblem


def solve(
    problem: CofolaProblem,
    wfomc_algo: Algo = Algo.FASTv2,
    use_partition_constraint: bool = True,
    lifted: bool = True,
) -> int:
    """
    Solve a combinatorics problem via WFOMC.

    :param problem: the combinatorics problem
    :param wfomc_algo: the algorithm to solve the problem
    :param use_partition_constraint: whether to use partition constraint
    :param lifted: whether to use lifted encoding for bags
    :return: the answer
    """
    pipeline = SolvePipeline(wfomc_algo, use_partition_constraint, lifted)
    return pipeline.run(problem)


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


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    input_file = args.input_file
    logger.info(f'Input file: {input_file}')
    with open(input_file, 'r') as f:
        problem: CofolaProblem = parse(f.read())
    logger.info(f'Problem: \n{problem}')
    res: int = solve(
        problem,
        # args.wfomc_algo,
        # use_partition_constraint=args.use_partition_constraint,
        lifted=False
        # args.lifted
    )
    logger.info(f'Answer: {res}')
