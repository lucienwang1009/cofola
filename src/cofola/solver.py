from __future__ import annotations

import argparse
import math

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.coso.backend import CoSoBackend
from cofola.backend.coso.planning import COSO_LOCAL_PASSES
from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.frontend import validate_problem
from cofola.frontend.problem import Problem
from cofola.log import setup_logging
from cofola.planing.pipeline import PlaningPipeline
from cofola.parser.parser import parse


BackendChoice = str | Backend


def _make_backend(backend: BackendChoice, *, debug: bool = False) -> Backend:
    if isinstance(backend, Backend):
        return backend
    normalized = backend.lower()
    if normalized == "wfomc":
        return WFOMCBackend(lifted=False)
    if normalized == "coso":
        return CoSoBackend(debug=debug)
    raise ValueError(f"Unknown backend {backend!r}. Expected 'wfomc' or 'coso'.")


def _make_pipeline(backend: Backend) -> PlaningPipeline:
    if isinstance(backend, CoSoBackend):
        return PlaningPipeline(local_passes=COSO_LOCAL_PASSES)
    return PlaningPipeline()


def solve(
    problem: Problem,
    debug: bool = False,
    validate: bool = True,
    backend: BackendChoice = "wfomc",
) -> int:
    """Solve a combinatorics problem.

    :param problem: A cofola.frontend.Problem instance.
    :param debug: Enable debug logging.
    :param validate: Run frontend type validation before solving.
    :param backend: Backend name or Backend instance.
    :return: the answer
    """
    setup_logging(debug)
    if validate:
        validate_problem(problem)
    logger.info("Solving problem with {} objects, {} constraints",
                len(problem.defs), len(problem.constraints))
    solver_backend = _make_backend(backend, debug=debug)
    schedule = _make_pipeline(solver_backend).process(problem)
    return sum(
        math.prod(solver_backend.solve(p, a) for p, a in branch.components)
        for branch in schedule.branches
    )


def parse_and_solve(
    text: str,
    debug: bool = False,
    backend: BackendChoice = "wfomc",
) -> int:
    """Parse .cfl source text and solve the combinatorics problem.

    :param text: the .cofola source text
    :param debug: Enable debug logging.
    :param backend: Backend name or Backend instance.
    :return: the answer
    """
    setup_logging(debug)
    logger.debug("Parsing input text ({} chars)", len(text))
    return solve(parse(text, debug=debug), debug=debug, validate=False, backend=backend)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Solve a combinatorics math problem using weighted first-order model counting'
    )
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    parser.add_argument(
        '--backend',
        choices=('wfomc', 'coso'),
        default='wfomc',
        help='solver backend to use',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug)
    input_file = args.input_file
    logger.info('Input file: {}', input_file)
    with open(input_file, 'r') as f:
        text = f.read()
    res: int = parse_and_solve(text, debug=args.debug, backend=args.backend)
    logger.info('Answer: {}', res)
