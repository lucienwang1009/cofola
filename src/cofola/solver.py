from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from loguru import logger

from cofola.frontend.problem import Problem
from cofola.log import setup_logging
from cofola.ir.pipeline import IRPipeline
from cofola.parser.parser import parse

if TYPE_CHECKING:
    from cofola.backend.base import Backend


def _resolve_backend(backend: str | Backend | None) -> Backend:
    if backend is None or backend == "wfomc":
        from cofola.backend.wfomc.backend import WFOMCBackend

        return WFOMCBackend(lifted=False)
    if backend == "unaryfomc":
        from cofola.backend.unaryfomc.backend import UnaryFOMCBackend

        return UnaryFOMCBackend()
    if isinstance(backend, str):
        raise ValueError(f"Unknown Cofola backend {backend!r}")
    return backend


def solve(
    problem: Problem,
    debug: bool = False,
    *,
    backend: str | Backend | None = None,
) -> int:
    """Solve a combinatorics problem.

    :param problem: A cofola.frontend.Problem instance.
    :param debug: Enable debug logging.
    :param backend: A backend instance or the name ``"wfomc"`` or
        ``"unaryfomc"``. The default remains ``"wfomc"``.
    :return: the answer
    """
    setup_logging(debug)
    logger.info("Solving problem with {} objects, {} constraints",
                len(problem.defs), len(problem.constraints))
    schedule = IRPipeline().process(problem)
    selected_backend = _resolve_backend(backend)
    return sum(
        math.prod(selected_backend.solve(p, a) for p, a in branch.components)
        for branch in schedule.branches
    )


def parse_and_solve(
    text: str,
    debug: bool = False,
    *,
    backend: str | Backend | None = None,
) -> int:
    """Parse .cfl source text and solve the combinatorics problem.

    :param text: the .cofola source text
    :param debug: Enable debug logging.
    :param backend: A backend instance or the name ``"wfomc"`` or
        ``"unaryfomc"``. The default remains ``"wfomc"``.
    :return: the answer
    """
    setup_logging(debug)
    logger.debug("Parsing input text ({} chars)", len(text))
    return solve(parse(text, debug=debug), debug=debug, backend=backend)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Solve a combinatorics math problem using weighted first-order model counting'
    )
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    parser.add_argument(
        '--backend',
        choices=('wfomc', 'unaryfomc'),
        default='wfomc',
        help='model-counting backend',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    setup_logging(args.debug)
    input_file = args.input_file
    logger.info('Input file: {}', input_file)
    with open(input_file, 'r') as f:
        text = f.read()
    res: int = parse_and_solve(text, debug=args.debug, backend=args.backend)
    logger.info('Answer: {}', res)
