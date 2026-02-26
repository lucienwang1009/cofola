"""Cofola parser - main entry point for parsing .cfl files."""
from __future__ import annotations

from lark import Lark

from cofola.parser.grammar import grammar
from cofola.parser.transformer import CofolaTransfomer
from cofola.problem import CofolaProblem


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
    'compose',
    'partition',
    'tuple',
    'choose_tuple',
    'choose_replace_tuple',
    'sequence',
    'choose_sequence',
    'choose_replace_sequence',
    'together',
    'not',
    'and',
    'or',
]

RESERVED_PREFIXES = [
    'AUX_',
    'IDX_'
]


class CofolaParsingError(Exception):
    """Base exception for Cofola parsing errors."""
    pass


class CofolaTypeMismatchError(CofolaParsingError):
    """Exception raised when an object has the wrong type during parsing."""

    def __init__(self, expected_types, actual):
        if isinstance(expected_types, tuple):
            expected_types = [t.__name__ for t in expected_types]
            expected_type = " or ".join(expected_types)
        else:
            expected_type = expected_types.__name__
        super().__init__(f"Expect a {expected_type} object, but got {actual} of type {type(actual)}.")


def parse(text: str) -> CofolaProblem:
    """
    Parse a Cofola problem from text.

    :param text: The text to parse.
    :return: The parsed CofolaProblem.
    """
    parser = Lark(grammar, start='cofola')
    tree = parser.parse(text)
    return CofolaTransfomer().transform(tree)


if __name__ == '__main__':
    text = r"""
# declare the sets
nondefective_tvs = set(nondef1...10)
defective_tvs = set(def10...13)
# set tvs = nondefective_tvs + defective_tvs

# perform choose operation
purchase = choose(nondefective_tvs+defective_tvs, 5)

# specify the constraints
# |(purchase & defective_tvs)| >= 2
    """
    problem = parse(text)
    print(problem)
    from cofola.passes import simplify
    problem = simplify(problem)
    print(problem)