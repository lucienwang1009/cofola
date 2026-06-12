"""Direct ASP backend semantic smoke tests."""
from __future__ import annotations

from cofola.solver import parse_and_solve


def _solve(source: str) -> int:
    return parse_and_solve(source, backend="asp")


class TestASPBackend(object):
    """Coverage for cases that do not pass through CoLa cleanly."""

    def test_basic_set_choice(self) -> None:
        assert _solve(
            """
S = set(a, b, c)
B = choose(S)
|B| == 2
"""
        ) == 3

    def test_multiple_configurations_with_constraint(self) -> None:
        assert _solve(
            """
A = set(a, b, c)
B = choose(A, 1)
C = choose(A, 1)
B disjoint C
"""
        ) == 6

    def test_nested_choice_preserves_upstream_count(self) -> None:
        assert _solve(
            """
team = set(p0...6)
lineup = choose(team, 4)
captains = choose(lineup, 2)
"""
        ) == 90

    def test_sequence_before_pattern(self) -> None:
        assert _solve(
            """
A = set(a, b, c)
row = sequence(A)
row.before(a, b)
"""
        ) == 3

    def test_sequence_next_to_patterns(self) -> None:
        assert _solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
next_to(A, b) in row
"""
        ) == 6
        assert _solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
next_to(A, b) in row for each A
"""
        ) == 2

    def test_sequence_predecessor_pattern_count(self) -> None:
        assert _solve(
            """
digits = set(1, 2)
integer = choose_replace_sequence(digits, 10)
integer.count((1, 1)) > 0
"""
        ) == 880

    def test_sequence_group_predecessor_pattern_count(self) -> None:
        assert _solve(
            """
sons = set(son0...4)
daughters = set(daughter0...3)
family = sons + daughters
row = sequence(family)
row.count((sons, sons)) >= 1
"""
        ) == 4896

    def test_sequence_transition_counts(self) -> None:
        assert _solve(
            """
coin = set(H, T)
seq = choose_replace_sequence(coin, 15)
seq.count((H, H)) == 2
seq.count((H, T)) == 3
seq.count((T, H)) == 4
seq.count((T, T)) == 5
"""
        ) == 560

    def test_choose_replace_sequence_from_dynamic_source(self) -> None:
        assert _solve(
            """
S = set(a, b, c)
C = choose(S, 2)
Q = choose_replace_sequence(C, 4)
"""
        ) == 48

    def test_circle_quotients_rotations(self) -> None:
        assert _solve(
            """
A = set(a, b, c, d)
C = circle(A)
"""
        ) == 6

    def test_circle_next_to_wraps_around_boundary(self) -> None:
        assert _solve(
            """
people = set(John, Sam, other0...4)
table = circle(people)
next_to(John, Sam) in table
"""
        ) == 48

    def test_circle_together_wraps_around_boundary(self) -> None:
        assert _solve(
            """
together_people = set(Pierre, Rosa, Thomas)
people = together_people + set(other0...5)
table = circle(people)
table.together(together_people)
"""
        ) == 720

    def test_circle_negative_next_to_uses_circular_adjacency(self) -> None:
        assert _solve(
            """
people = set(Alice, Bob, other0...6)
table = circle(people)
not (next_to(Alice, Bob) in table)
"""
        ) == 3600

    def test_circle_multiple_next_to_can_share_center(self) -> None:
        assert _solve(
            """
tribe = set(mother, wife, chief, other0...5)
arrangement = circle(tribe)
next_to(mother, chief) in arrangement
next_to(chief, wife) in arrangement
"""
        ) == 240

    def test_tuple_dedup_count(self) -> None:
        assert _solve(
            """
digits = set(0...3)
number = choose_replace_tuple(digits, 3)
number.dedup_count(digits) == 2
"""
        ) == 18

    def test_bag_subset_and_equality_use_multiplicity(self) -> None:
        assert _solve(
            """
B = bag(a: 1, b: 2)
sub = choose(B)
sup = choose(B)
sub subset sup
"""
        ) == 18
        assert _solve(
            """
B = bag(a: 1, b: 2)
sub = choose(B)
sup = choose(B)
sub == sup
"""
        ) == 6

    def test_unsized_choose_replace_respects_inferred_total_size_bound(self) -> None:
        assert _solve(
            """
oreos = set(oreo0...5)
milk = set(m0...3)
products = oreos + milk
Alpha = choose(products)
Beta = choose_replace(oreos)
|Alpha| + |Beta| == 3
"""
        ) == 351

    def test_composition_allows_empty_labeled_groups(self) -> None:
        assert _solve(
            """
group = set(friend0...6)
C = compose(group, 3)
"""
        ) == 729

    def test_composition_for_all_part_constraints(self) -> None:
        assert _solve(
            """
S = set(0...6)
C = compose(S, 3)
|part| > 0 for part in C
"""
        ) == 540

    def test_indexed_composition_constraints(self) -> None:
        assert _solve(
            """
people = set(0...11) + set(Henry)
groups = compose(people, 3)
|groups[0]| == 3
|groups[1]| == 4
|groups[2]| == 5
Henry in groups[1]
"""
        ) == 9240

    def test_partition_quotients_part_order(self) -> None:
        assert _solve(
            """
group = set(friend0...6)
P = partition(group, 3)
"""
        ) == 122

    def test_bag_partition_allows_empty_unlabeled_groups(self) -> None:
        assert _solve(
            """
B = bag(orange: 4)
P = partition(B, 3)
"""
        ) == 4
