"""Planning regressions for negating numerical constraints."""
from __future__ import annotations

from dataclasses import replace

import pytest

from cofola.frontend import ObjRef, OrConstraint, SizeConstraint
from cofola.planing.pipeline import _negate_constraint


@pytest.mark.parametrize(
    ("comparator", "complement"),
    [("<", ">="), ("<=", ">"), (">", "<="), (">=", "<"), ("!=", "==")],
)
def test_size_constraint_negation_preserves_linear_terms(
    comparator: str, complement: str,
) -> None:
    constraint = SizeConstraint(
        terms=((ObjRef(1), 2), (ObjRef(2), -1)), comparator=comparator, rhs=3,
    )

    assert _negate_constraint(constraint) == replace(constraint, comparator=complement)
    assert constraint.comparator == comparator


def test_size_constraint_negation_splits_equality_into_disjoint_ranges() -> None:
    constraint = SizeConstraint(terms=((ObjRef(1), 1),), comparator="==", rhs=1)

    assert _negate_constraint(constraint) == OrConstraint(
        left=replace(constraint, comparator="<=", rhs=0),
        right=replace(constraint, comparator=">=", rhs=2),
    )
