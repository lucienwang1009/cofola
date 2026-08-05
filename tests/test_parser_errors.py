"""Parser error boundary tests."""
from __future__ import annotations

import pytest
from lark.exceptions import UnexpectedInput, VisitError

from cofola.parser import CofolaParsingError
from cofola.parser.parser import parse
from cofola.frontend import Entity, SizeConstraint, TupleCountAtom


PARSING_ERROR_CASES: list[tuple[str, str, str]] = [
    (
        "undefined_object_in_count",
        """
S = set(a, b)
T = tuple(S)
T.count(unknown) == 1
""",
        "Object unknown has not been defined",
    ),
    (
        "entity_left_equality",
        """
S = set(a, b)
T = tuple(S)
a == T[0]
""",
        "equivalence_constraint: unsupported types",
    ),
    (
        "duplicate_set_entity",
        """
S = set(a, a)
""",
        "Duplicate entities are not allowed",
    ),
    (
        "inline_for_partition_expression",
        """
S = set(a, b)
|p| == 1 for p in partition(S, 2)
""",
        "requires a named partition",
    ),
    (
        "duplicate_object_name",
        """
S = set(a, b)
S = set(c, d)
""",
        "already been defined",
    ),
    (
        "part_name_shadows_object",
        """
S = set(a, b)
P = partition(S, 2)
|S| == 1 for S in P
""",
        "part name S has been used as an object name",
    ),
    (
        "part_name_shadows_entity",
        """
S = set(p, q)
P = partition(S, 2)
|p| == 1 for p in P
""",
        "part name p has been used as an Entity",
    ),
]


@pytest.mark.parametrize(
    "label,program,expected_substring",
    PARSING_ERROR_CASES,
    ids=[label for label, _, _ in PARSING_ERROR_CASES],
)
def test_parser_unwraps_cofola_parsing_errors(
    label: str,
    program: str,
    expected_substring: str,
) -> None:
    """Transformer-raised CofolaParsingError should not leak as VisitError."""
    with pytest.raises(CofolaParsingError) as exc_info:
        parse(program)

    assert not isinstance(exc_info.value, VisitError)
    msg = str(exc_info.value)
    assert expected_substring in msg, (
        f"[{label}] expected substring {expected_substring!r} in error message, "
        f"got:\n{msg}"
    )


def test_part_constraint_accepts_parenthesized_partition_name() -> None:
    """Parentheses around the partition name should not crash the transformer."""
    program = """
S = set(a, b)
P = partition(S, 2)
|p| == 1 for p in (P)
"""

    parse(program)


def test_circle_is_available_as_an_identifier() -> None:
    """The removed object operator must not leave a stale reserved keyword."""
    problem = parse("circle = set(a, b)")

    assert any(name == "circle" for _, name in problem.names)


@pytest.mark.parametrize(
    "program",
    [
        "S = set(a, b)\nC = circle(S)",
        "S = set(a, b)\nC = choose_circle(S, 1)",
        "S = set(a, b)\nC = choose_replace_circle(S, 2)",
        "S = set(a, b)\nC = sequence(S, reflection=True)",
    ],
)
def test_circle_syntax_is_rejected(program: str) -> None:
    """Circle constructors and their reflection option are no longer syntax."""
    with pytest.raises(UnexpectedInput):
        parse(program)


def test_tuple_membership_workaround_uses_tuple_count() -> None:
    """Tuple membership syntax is parsed as tuple count constraints."""
    problem = parse("""
S = set(a, b)
T = tuple(S)
a in T
b not in T
""")
    tuple_ref = next(ref for ref, name in problem.names if name == "T")

    assert problem.constraints == (
        SizeConstraint(
            terms=((TupleCountAtom(tuple_ref=tuple_ref, count_obj=Entity("a")), 1),),
            comparator=">",
            rhs=0,
        ),
        SizeConstraint(
            terms=((TupleCountAtom(tuple_ref=tuple_ref, count_obj=Entity("b")), 1),),
            comparator="==",
            rhs=0,
        ),
    )
