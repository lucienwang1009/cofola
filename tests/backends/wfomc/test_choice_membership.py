"""WFOMC choice-object, bag-negation, and tuple-membership semantics."""
from __future__ import annotations

from cofola.solver import parse_and_solve


class TestWFOMCChoiceAndMembershipSemantics(object):
    """Choice objects, bag negation, and tuple membership semantics."""

    def test_negative_bag_equality_is_any_multiplicity_difference(self) -> None:
        """B != C means at least one multiplicity differs."""
        assert parse_and_solve(
            """
S = bag(a: 2, b: 2)
B = choose(S)
C = choose(S)
B != C
"""
        ) == 72


    def test_negative_bag_subset_is_any_multiplicity_violation(self) -> None:
        """not (B subset C) means at least one entity has a larger multiplicity."""
        assert parse_and_solve(
            """
S = bag(a: 2, b: 2)
B = choose(S)
C = choose(S)
not B subset C
"""
        ) == 45


    def test_bag_subset_constrains_singleton_entities(self) -> None:
        """Singleton entities (multiplicity <= 1) must still obey bag subset.

        Regression: singletons were skipped in the multiplicity loop and never
        constrained over the singleton subdomain, so ``a in sub`` was allowed
        even when ``a not in sup``.
        """
        # `a` is a singleton; pairs over a in {0,1}: (0,0),(0,1),(1,1) = 3.
        assert parse_and_solve(
            """
B = bag(a: 1)
sub = choose(B)
sup = choose(B)
sub subset sup
"""
        ) == 3
        assert parse_and_solve(
            """
B = bag(a: 1)
sub = choose(B)
sup = choose(B)
not sub subset sup
"""
        ) == 1
        # `a` singleton, `b` multiplicity 2: 3 (a) * 6 (b) = 18.
        assert parse_and_solve(
            """
B = bag(a: 1, b: 2)
sub = choose(B)
sup = choose(B)
sub subset sup
"""
        ) == 18

    def test_bag_equality_constrains_singleton_entities(self) -> None:
        """Bag equality must hold per singleton entity, not only non-singletons."""
        # equal pairs: a (2) * b (3) = 6; unequal = 36 - 6 = 30.
        assert parse_and_solve(
            """
B = bag(a: 1, b: 2)
sub = choose(B)
sup = choose(B)
sub == sup
"""
        ) == 6
        assert parse_and_solve(
            """
B = bag(a: 1, b: 2)
sub = choose(B)
sup = choose(B)
sub != sup
"""
        ) == 30

    def test_choose_replace_sequence_uses_dynamic_chosen_source(self) -> None:
        """choose_replace_sequence over a chosen set must respect the chosen source."""
        assert parse_and_solve(
            """
S = set(a, b, c)
C = choose(S, 2)
Q = choose_replace_sequence(C, 4)
"""
        ) == 48


    def test_sequence_of_fixed_size_choice_uses_analysis_exact_size(self) -> None:
        """Full sequence size should come from the chosen source's analysis facts."""
        assert parse_and_solve(
            """
U = set(a, b, c, d)
S = choose(U)
|S| == 2
row = sequence(S)
"""
        ) == 12


    def test_sequence_of_variable_choice_branches_by_actual_source_size(self) -> None:
        """Each full-sequence size branch should constrain the chosen source too."""
        assert parse_and_solve(
            """
U = set(a, b)
S = choose(U)
row = sequence(S)
"""
        ) == 5


    def test_tuple_of_composition_part_constrains_part_size(self) -> None:
        """Full tuple over a dynamic composition part must fix the part size."""
        assert parse_and_solve(
            """
S = set(a, b)
P = compose(S, 2)
T = tuple(P[0])
"""
        ) == 5


    def test_full_size_bag_choose_tuple_matches_full_tuple(self) -> None:
        """Full-size bag tuple choices should share the compact tuple encoding."""
        full_tuple = """
S = bag(A: 5, B: 4, C: 2)
T = tuple(S)
"""
        full_choose_tuple = """
S = bag(A: 5, B: 4, C: 2)
T = choose_tuple(S, 11)
"""
        inferred_full_choose_tuple = """
S = bag(A: 5, B: 4, C: 2)
T = choose_tuple(S)
|T| == 11
"""

        assert (
            parse_and_solve(full_choose_tuple)
            == parse_and_solve(inferred_full_choose_tuple)
            == parse_and_solve(full_tuple)
            == 6930
        )


    def test_full_size_choose_is_identity_but_unsized_choose_stays_variable(self) -> None:
        """choose(S, |S|) is an alias; choose(S) still ranges over subsets."""
        assert parse_and_solve(
            """
S = set(a, b, c)
T = choose(S, 3)
"""
        ) == 1
        assert parse_and_solve(
            """
S = set(a, b, c)
T = choose(S)
"""
        ) == 8
        assert parse_and_solve(
            """
B = bag(a: 2, b: 1)
C = choose(B, 3)
"""
        ) == 1


    def test_tuple_membership_uses_tuple_image_semantics(self) -> None:
        """Tuple membership should constrain whether an entity appears anywhere."""
        assert parse_and_solve(
            """
S = set(a, b)
T = choose_tuple(S, 1)
a in T
"""
        ) == 1
        assert parse_and_solve(
            """
S = set(a, b)
T = choose_tuple(S, 1)
a not in T
"""
        ) == 1
        assert parse_and_solve(
            """
S = set(a, b)
T = tuple(S)
b not in T
"""
        ) == 0


    def test_choose_replace_sequence_from_chosen_set_respects_source_choice(self) -> None:
        """Repeated sequence entries must come from the chosen source subset."""
        assert parse_and_solve(
            """
set_0 = set(e_1, e_2, e_3)
choose_0 = choose(set_0, 2)
choose_replace_sequence_0 = choose_replace_sequence(choose_0, 3)
"""
        ) == 24
