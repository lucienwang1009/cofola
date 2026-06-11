"""Parser error boundary tests."""
from __future__ import annotations

import pytest
from lark.exceptions import UnexpectedInput, VisitError

from cofola.parser import CofolaParsingError
from cofola.parser.parser import parse
from cofola.frontend import Entity, MembershipConstraint


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
        "reserved_circle_identifier",
        """
circle = set(a, b)
""",
        "reserved keyword",
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


def test_tuple_membership_parses_as_membership_constraint() -> None:
    """Tuple membership syntax should lower directly instead of using count atoms."""
    problem = parse("""
S = set(a, b)
T = tuple(S)
a in T
b not in T
""")
    tuple_ref = next(ref for ref, name in problem.names if name == "T")

    assert problem.constraints == (
        MembershipConstraint(
            entity=Entity("a"),
            container=tuple_ref,
            positive=True,
        ),
        MembershipConstraint(
            entity=Entity("b"),
            container=tuple_ref,
            positive=False,
        ),
    )


def test_pattern_not_in_does_not_accept_coverage_qualifier() -> None:
    """Use explicit boolean not for negated coverage constraints."""
    with pytest.raises(UnexpectedInput):
        parse("""
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
(A, b) not in row for each b
""")


def _capture_warnings(source: str) -> list[str]:
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        parse(source)
    finally:
        logger.remove(handler_id)
    return messages


@pytest.mark.parametrize(
    "program",
    [
        "S = set(a, b)\nrow = sequence(S)\na < b in row\n",
        "S = set(a, b)\nrow = sequence(S)\ntogether(set(a, b)) in row\n",
    ],
    ids=["less_than", "together"],
)
def test_global_pattern_in_form_warns_deprecated(program: str) -> None:
    """`together(...) in seq` and `a < b in seq` are deprecated legacy spellings."""
    messages = _capture_warnings(program)
    assert any("deprecated" in m for m in messages)


@pytest.mark.parametrize(
    "program",
    [
        "S = set(a, b)\nrow = sequence(S)\nrow.before(a, b)\n",
        "S = set(a, b)\nrow = sequence(S)\nrow.together(set(a, b))\n",
        "S = set(a, b, c)\nrow = sequence(S)\n(a, b) in row\n",
    ],
    ids=["before_method", "together_method", "local_pattern"],
)
def test_method_and_local_forms_do_not_warn(program: str) -> None:
    """The documented method form and local patterns emit no deprecation warning."""
    messages = _capture_warnings(program)
    assert not any("deprecated" in m for m in messages)
