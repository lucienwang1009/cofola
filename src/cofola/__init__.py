import argparse

import logging
from logzero import logger
from cofola.problem import CofolaProblem
from cofola.parser.parser import parse
from cofola.solver import solve


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
        # args.use_partition_constraint,
        # args.lifted
    )
    logger.info(f'Answer: {res}')
