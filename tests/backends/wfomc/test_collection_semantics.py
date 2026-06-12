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

    def test_empty_derived_bag_size_is_satisfiable(self) -> None:
        """`|X op Y| == 0` must count the empty models, not collapse to 0.

        Regression (decoder): when `|Z| == 0` collapses Z, its multiplicity
        variables drop out of the WFOMC polynomial, but the validators still
        reference them. Absent variables must be read as degree 0 rather than
        dangling free symbols.
        """
        # bag(a:2): X,Y range over a in {0,1,2}. Empty results, by operation:
        #   X+Y: both empty -> 1;  X&Y: X=0 or Y=0 -> 5;  X-Y: X<=Y -> 6.
        expected = {"X + Y": 1, "X & Y": 5, "X - Y": 6}
        for op, exp in expected.items():
            assert parse_and_solve(
                f"B = bag(a: 2)\nX = choose(B)\nY = choose(B)\nZ = {op}\n|Z| == 0\n"
            ) == exp, op

    def test_derived_bag_tracks_equal_multiplicity_entities(self) -> None:
        """Union/intersection/difference must keep entities the sources hold as
        indistinguishable (equal multiplicities).

        Regression (bag classification): a derived bag took only the sources'
        `dis_entities`, dropping entities kept in `indis_entities`, so its size
        expression became 0 and `|Z| == k` folded to a constant.
        """
        # bag(a:2, b:2): a,b indistinguishable. |X+Y| == 1 -> 6 (one unit in a or b).
        assert parse_and_solve(
            "B = bag(a: 2, b: 2)\nX = choose(B)\nY = choose(B)\nZ = X + Y\n|Z| == 1\n"
        ) == 6
        # Intersection size 4 (both full): 1.
        assert parse_and_solve(
            "B = bag(a: 2, b: 2)\nX = choose(B)\nY = choose(B)\nZ = X & Y\n|Z| == 4\n"
        ) == 1

    def test_partition_of_bag_with_variable_singletons(self) -> None:
        """Partitioning a non-base bag whose singleton entities have variable
        membership must not force the singletons out of every part.

        Regression: the partition multiplicity loop tied each singleton's part
        counts to the source's multiplicity variable via `_bag_entity_expr`,
        which is unbound (read as 0) for singletons in a chosen/derived bag —
        contradicting the coverage/exactly-one constraints and collapsing the
        count to 0. Singleton membership is already pinned by those constraints.
        """
        # choose 2 of 3 singletons, partition into 2 parts.
        # sum over size-2 sub-bags (3 of them) of 2-partitions (2 each) = 6.
        assert parse_and_solve(
            "B = bag(a, b, c)\nC = choose(B, 2)\nP = partition(C, 2)\n"
        ) == 6
        # Same through an intersection (the original failing benchmark shape).
        assert parse_and_solve(
            "B = bag(a, b, c)\nC = choose(B, 2)\nI = C & B\nP = partition(I, 2)\n"
        ) == 6

    def test_partition_singletons_kept_in_symmetry_breaking(self) -> None:
        """Parts distinguished only by which singleton they hold are one
        unordered partition — the singleton vars must stay in the part-symmetry
        comparison, so e.g. {a, c:2} | {b, c:2} is not double counted.
        """
        # bag(a:1, b:1, c:4) into 2 parts: 10 unordered (the ordered count is 20).
        assert parse_and_solve("B = bag(a: 1, b: 1, c: 4)\nP = partition(B, 2)\n") == 10

    def test_full_source_choice_alias_keeps_derived_consumers_valid(self) -> None:
        """Aliasing a full-source choice must not leave a dangling reference.

        `choose(B, |B|)` is aliased to `B`, turning a derived consumer into a
        constant-foldable object (`B & B`). Regression: the fold/merge that
        removes it ran before the alias, so the object was dropped while a
        constraint still referenced it -> `No predicate for ref` at encode time.
        """
        # |choose(B, |B|) & B| == 2 reduces to |B| == 2 -> 1 (no crash).
        assert parse_and_solve(
            "B = bag(a: 1, b: 1)\nC = choose(B, 2)\nI = C & B\n|I| == 2\n"
        ) == 1
        # partition over the same full-choice intersection reduces to partition(B, 2).
        assert parse_and_solve(
            "B = bag(a: 1, b: 1)\nC = choose(B, 2)\nI = C & B\nP = partition(I, 2)\n"
        ) == 2

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
