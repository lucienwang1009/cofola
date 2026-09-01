from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Union

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.asp.backend import ASPBackend
from cofola.backend.coso.backend import CoSoBackend
from cofola.backend.essence.backend import EssenceBackend
from cofola.backend.wfomc.api import Algo, LinearOrderEncoding
from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.frontend import validate_problem
from cofola.frontend.problem import Problem
from cofola.log import setup_logging
from cofola.planing.pipeline import PlaningPipeline
from cofola.parser.parser import parse


BackendChoice = str | Backend
AlgoChoice = Union[Algo, str, None]
LinearOrderEncodingChoice = Union[LinearOrderEncoding, str, None]


def _resolve_algo(algo: AlgoChoice) -> Algo:
    if algo is None:
        return Algo.FASTv2
    if isinstance(algo, Algo):
        return algo
    return Algo(algo)


def _resolve_linear_order_encoding(
    encoding: LinearOrderEncodingChoice,
) -> LinearOrderEncodingChoice:
    """Pass-through resolver: wfomc accepts None/str/enum so we just normalise None."""
    if encoding is None or isinstance(encoding, LinearOrderEncoding):
        return encoding
    return LinearOrderEncoding(encoding)


def _make_backend(
    backend: BackendChoice,
    *,
    debug: bool = False,
    algo: AlgoChoice = None,
    linear_order_encoding: LinearOrderEncodingChoice = None,
    conjure_dir: str | Path | None = None,
    java_bin: str | Path | None = None,
) -> Backend:
    if isinstance(backend, Backend):
        return backend
    normalized = backend.lower()
    if normalized == "wfomc":
        return WFOMCBackend(
            algo=_resolve_algo(algo),
            lifted=False,
            linear_order_encoding=_resolve_linear_order_encoding(linear_order_encoding),
        )
    if normalized == "coso":
        return CoSoBackend(debug=debug)
    if normalized == "asp":
        return ASPBackend(debug=debug)
    if normalized == "essence":
        return EssenceBackend(
            conjure_dir=conjure_dir,
            java_bin=java_bin,
            debug=debug,
        )
    raise ValueError(
        f"Unknown backend {backend!r}. Expected 'wfomc', 'coso', 'asp', or 'essence'."
    )


def solve(
    problem: Problem,
    debug: bool = False,
    validate: bool = True,
    backend: BackendChoice = "wfomc",
    algo: AlgoChoice = None,
    linear_order_encoding: LinearOrderEncodingChoice = None,
    conjure_dir: str | Path | None = None,
    java_bin: str | Path | None = None,
) -> int:
    """Solve a combinatorics problem.

    :param problem: A cofola.frontend.Problem instance.
    :param debug: Enable debug logging.
    :param validate: Run frontend type validation before solving.
    :param backend: Backend name or Backend instance.
    :param algo: WFOMC algorithm to use (only honoured for the wfomc backend).
        Accepts an Algo enum value or its string form ("fastv2", "incremental",
        "recursive", "propositional"). Defaults to FASTv2.
    :param linear_order_encoding: Only consulted when algo == PROPOSITIONAL;
        picks how the order axioms are encoded ("pin" or "axioms").
    :param conjure_dir: Directory containing Conjure/Savile Row tools for
        the Essence backend. If omitted, COFOLA_CONJURE_DIR and PATH are used.
    :param java_bin: Java executable for the Essence backend. If omitted,
        COFOLA_JAVA_BIN and PATH are used.
    :return: the answer
    """
    setup_logging(debug)
    if validate:
        validate_problem(problem)
    logger.info("Solving problem with {} objects, {} constraints",
                len(problem.defs), len(problem.constraints))
    solver_backend = _make_backend(
        backend,
        debug=debug,
        algo=algo,
        linear_order_encoding=linear_order_encoding,
        conjure_dir=conjure_dir,
        java_bin=java_bin,
    )
    schedule = PlaningPipeline(solver_backend.planning_profile()).process(problem)
    return sum(
        math.prod(solver_backend.solve(p, a) for p, a in branch.components)
        for branch in schedule.branches
    )


def parse_and_solve(
    text: str,
    debug: bool = False,
    backend: BackendChoice = "wfomc",
    algo: AlgoChoice = None,
    linear_order_encoding: LinearOrderEncodingChoice = None,
    conjure_dir: str | Path | None = None,
    java_bin: str | Path | None = None,
) -> int:
    """Parse .cfl source text and solve the combinatorics problem.

    :param text: the .cofola source text
    :param debug: Enable debug logging.
    :param backend: Backend name or Backend instance.
    :param algo: see :func:`solve`.
    :param linear_order_encoding: see :func:`solve`.
    :return: the answer
    """
    setup_logging(debug)
    logger.debug("Parsing input text ({} chars)", len(text))
    return solve(
        parse(text, debug=debug),
        debug=debug,
        validate=False,
        backend=backend,
        algo=algo,
        linear_order_encoding=linear_order_encoding,
        conjure_dir=conjure_dir,
        java_bin=java_bin,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Solve a combinatorics math problem using weighted first-order model counting'
    )
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    parser.add_argument(
        '--backend',
        choices=('wfomc', 'coso', 'asp', 'essence'),
        default='wfomc',
        help='solver backend to use',
    )
    parser.add_argument(
        '--algo', '-a',
        type=Algo,
        choices=list(Algo),
        default=Algo.FASTv2,
        help='WFOMC algorithm (wfomc backend only). Default: fastv2. '
             'Use "propositional" to ground the sentence and count with ganak '
             '(requires an external ganak binary; see wfomc README).',
    )
    parser.add_argument(
        '--linear-order-encoding', '-l',
        type=LinearOrderEncoding,
        choices=list(LinearOrderEncoding),
        default=None,
        help='How the propositional counter encodes order axioms. '
             'Ignored by other algorithms. Default: pin.',
    )
    parser.add_argument(
        '--conjure-dir',
        type=Path,
        default=None,
        help='Directory containing Conjure/Savile Row tools for --backend essence.',
    )
    parser.add_argument(
        '--java-bin',
        type=Path,
        default=None,
        help='Java executable for --backend essence. Defaults to java on PATH.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug)
    input_file = args.input_file
    logger.info('Input file: {}', input_file)
    with open(input_file, 'r') as f:
        text = f.read()
    res: int = parse_and_solve(
        text,
        debug=args.debug,
        backend=args.backend,
        algo=args.algo,
        linear_order_encoding=args.linear_order_encoding,
        conjure_dir=args.conjure_dir,
        java_bin=args.java_bin,
    )
    logger.info('Answer: {}', res)
