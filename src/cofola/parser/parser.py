"""Cofola parser - main entry point for parsing .cfl files."""
from __future__ import annotations

from lark import Lark
from loguru import logger

from cofola.frontend.pretty import fmt_problem
from cofola.parser.errors import CofolaParsingError, CofolaTypeMismatchError
from cofola.parser.grammar import grammar
from cofola.parser.transformer import CofolaTransfomer

__all__ = [
    "CofolaParsingError",
    "CofolaTypeMismatchError",
    "RESERVED_KEYWORDS",
    "RESERVED_PREFIXES",
    "parse",
]


RESERVED_KEYWORDS = [
    "set",
    "bag",
    "choose",
    "choose_replace",
    "count",
    "in",
    "subset",
    "disjoint",
    "supp",
    "compose",
    "partition",
    "tuple",
    "choose_tuple",
    "choose_replace_tuple",
    "sequence",
    "choose_sequence",
    "choose_replace_sequence",
    "together",
    "not",
    "and",
    "or",
]

RESERVED_PREFIXES = [
    "AUX_",
    "IDX_",
]


def parse(text: str, debug: bool = False):
    """Parse a Cofola problem from text, returning an ir.Problem directly.

    :param text: The text to parse.
    :param debug: Enable debug logging (passed from caller; logging must be
                  configured via cofola.log.setup_logging before calling).
    :return: The parsed ir.Problem.
    """
    parser = Lark(grammar, start="cofola")
    tree = parser.parse(text)
    logger.debug(
        "Grammar parsed successfully; tree has {} tokens",
        len(list(tree.scan_values(lambda _: True))),
    )
    problem = CofolaTransfomer().transform(tree)
    logger.info(
        "Parsed problem: {} objects, {} constraints",
        len(problem.defs),
        len(problem.constraints),
    )
    logger.debug("\n{}", fmt_problem(problem, stage="[Parser] Raw Parsed Problem"))
    return problem
