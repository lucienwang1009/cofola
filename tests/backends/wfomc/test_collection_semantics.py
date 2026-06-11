"""WFOMC set, bag, and choice collection semantics."""
from __future__ import annotations

from cofola.solver import parse_and_solve


class TestWFOMCCollectionSemantics(object):
    """Set, bag, tuple, and choice semantics at the WFOMC boundary."""

    def test_full_choose_subset_is_trivially_satisfiable(self) -> None:
        """`set subset choose(set, |set|)` collapses to a tautology (answer 1).

        Regression for two WFOMC crashes when counting an atomless `\\forall X: True`
        formula: the lifted algorithms crashed building the cell graph, and the
        propositional backend produced an empty CNF that aborted ganak. The
        propositional case short-circuits to 1 and needs no ganak binary.
        """
        program = """
S = set(a, b, c)
T = choose(S, 3)
S subset T
"""
        assert parse_and_solve(program) == 1
        assert parse_and_solve(program, algo="propositional") == 1


    def test_bag_difference_counts_leftover_multiplicities(self) -> None:
        """Bag difference should use max(left - right, 0), not support difference."""
        assert parse_and_solve(
            """
B = bag(a: 2, b: 1)
C = bag(a: 1, c: 1)
E = bag(b: 1)
D = B - C
|D| == 2
"""
        ) == 1


    def test_bag_difference_subtracts_singleton_entities(self) -> None:
        """A singleton entity in a bag difference must still subtract the RHS.

        Regression: singletons were skipped in the per-entity loop and only the
        loose ``obj -> left`` bound was emitted, so ``a in (X - Y)`` ignored Y
        for singleton ``a``.
        """
        # a is a singleton: a in (X - Y) iff a in X and a not in Y.
        # X,Y range over sub-bags of bag(a:1); pairs with X_a=1,Y_a=0 -> 1.
        assert parse_and_solve(
            """
B = bag(a: 1)
X = choose(B)
Y = choose(B)
Z = X - Y
a in Z
"""
        ) == 1

    def test_bag_union_preserves_max_multiplicity_for_dynamic_sources(self) -> None:
        """Bag union should constrain multiplicities with max(left, right)."""
        assert parse_and_solve(
            """
B = bag(a: 2)
C = choose(B)
D = C + B
|D| == 2
"""
        ) == 3


    def test_bag_union_count_atom_uses_encoded_multiplicity(self) -> None:
        """Bag count atoms should read the resolved bag multiplicity expression."""
        assert parse_and_solve(
            """
B = bag(a: 2)
C = choose(B)
D = C + B
D.count(a) == 2
"""
        ) == 3


    def test_bag_count_atom_on_base_bag_uses_constant_multiplicity(self) -> None:
        """A base bag count is a fixed integer, not a fresh symbolic variable."""
        assert parse_and_solve(
            """
B = bag(a: 2)
B.count(a) == 2
"""
        ) == 1
        assert parse_and_solve(
            """
B = bag(a: 2)
B.count(a) == 1
"""
        ) == 0
