from __future__ import annotations

import argparse
import math

from loguru import logger

from cofola.frontend.problem import Problem
from cofola.log import setup_logging
from cofola.ir.pipeline import IRPipeline
from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.parser.parser import parse


def solve(problem: Problem, debug: bool = False) -> int:
    """Solve a combinatorics problem.

    :param problem: A cofola.frontend.Problem instance.
    :param debug: Enable debug logging.
    :return: the answer
    """
    setup_logging(debug)
    logger.info("Solving problem with {} objects, {} constraints",
                len(problem.defs), len(problem.constraints))
    schedule = IRPipeline().process(problem)
    backend = WFOMCBackend(lifted=False)
    return sum(
        math.prod(backend.solve(p, a) for p, a in branch.components)
        for branch in schedule.branches
    )


def parse_and_solve(text: str, debug: bool = False) -> int:
    """Parse .cfl source text and solve the combinatorics problem.

    :param text: the .cofola source text
    :param debug: Enable debug logging.
    :return: the answer
    """
    setup_logging(debug)
    logger.debug("Parsing input text ({} chars)", len(text))
    return solve(parse(text, debug=debug), debug=debug)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Solve a combinatorics math problem using weighted first-order model counting'
    )
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug)
    input_file = args.input_file
    logger.info('Input file: {}', input_file)
    with open(input_file, 'r') as f:
        text = f.read()
    res: int = parse_and_solve(text, debug=args.debug)
    logger.info('Answer: {}', res)
