"""Cofola parser - main entry point for parsing .cfl files."""
from __future__ import annotations

from functools import lru_cache

from lark import Lark
from lark.exceptions import VisitError
from loguru import logger

from cofola.frontend.pretty import fmt_problem
from cofola.frontend.problem import Problem
from cofola.frontend.type_check import CofolaTypeError, validate_problem
from cofola.parser.utils import (
    CofolaParsingError,
    RESERVED_KEYWORDS,
    RESERVED_PREFIXES,
)
from cofola.parser.grammar import grammar
from cofola.parser.transformer import CofolaTransformer

__all__ = [
    "CofolaParsingError",
    "RESERVED_KEYWORDS",
    "RESERVED_PREFIXES",
    "parse",
]


@lru_cache(maxsize=1)
def _parser() -> Lark:
    """Return the shared Cofola grammar parser.

    Building a Lark parser is relatively expensive and the grammar is static.
    """

    return Lark(grammar, start="cofola", propagate_positions=True)


def parse(text: str, debug: bool = False) -> Problem:
    """Parse Cofola source text into a frontend Problem.

    :param text: The text to parse.
    :param debug: Enable debug logging (passed from caller; logging must be
                  configured via cofola.log.setup_logging before calling).
    :return: The parsed frontend Problem.
    """
    tree = _parser().parse(text)
    if debug:
        logger.debug(
            "Grammar parsed successfully; tree has {} values",
            sum(1 for _ in tree.scan_values(lambda _: True)),
        )
    try:
        problem = CofolaTransformer().transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, (CofolaParsingError, CofolaTypeError)):
            raise exc.orig_exc from None
        raise
    logger.info(
        "Parsed problem: {} objects, {} constraints",
        len(problem.defs),
        len(problem.constraints),
    )
    logger.debug("\n{}", fmt_problem(problem, stage="[Parser] Raw Parsed Problem"))
    validate_problem(problem)
    return problem
