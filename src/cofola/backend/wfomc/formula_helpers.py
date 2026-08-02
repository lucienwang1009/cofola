"""Formula helpers with semantics owned by the Cofola encoder."""
from __future__ import annotations

from wfomc.fol import X, conjunction, disjunction, forall

from cofola.backend.wfomc.api import Formula, Pred, top


def exactly_one_qf(predicates: list[Pred]) -> Formula:
    """Require exactly one predicate to hold for the free variable ``X``."""

    if not predicates:
        raise ValueError("exactly_one_qf requires at least one predicate")
    literals = tuple(predicate(X) for predicate in predicates)
    if len(literals) == 1:
        return literals[0]
    pairwise = tuple(
        ~(left & right)
        for index, left in enumerate(literals)
        for right in literals[index + 1 :]
    )
    return disjunction(*literals) & conjunction(*pairwise)


def exclusive(predicates: list[Pred]) -> Formula:
    """Require that no two predicates hold for the same domain element."""

    if not predicates:
        raise ValueError("exclusive requires at least one predicate")
    if len(predicates) == 1:
        return top
    literals = tuple(predicate(X) for predicate in predicates)
    return forall(
        X,
        conjunction(
            *(
                ~(left & right)
                for index, left in enumerate(literals)
                for right in literals[index + 1 :]
            )
        ),
    )


__all__ = ["exactly_one_qf", "exclusive"]
