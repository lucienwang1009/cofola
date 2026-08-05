"""WFOMC sequence pattern syntax and semantic regressions."""
from __future__ import annotations

from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.backend.wfomc.encoder import encode
from cofola.planing.pipeline import PlaningPipeline
from cofola.parser.parser import parse
from cofola.solver import parse_and_solve


def _encode_single_component(source: str) -> tuple[object, object]:
    problem = parse(source)
    backend = WFOMCBackend(lifted=False)
    schedule = PlaningPipeline(backend.planning_profile()).process(problem)
    assert len(schedule.branches) == 1
    assert len(schedule.branches[0].components) == 1
    component, analysis = schedule.branches[0].components[0]
    return encode(component, analysis, lifted=False)

class TestWFOMCSequencePatterns(object):
    """Sequence pattern syntax and semantic regressions."""

    def test_group_less_than_pattern_uses_universal_semantics(self) -> None:
        """A < B requires every A entity to precede every B entity."""
        assert parse_and_solve(
            """
A = set(a0, a1)
B = set(b0, b1)
row = sequence(A + B)
A < B in row
"""
        ) == 4


    def test_before_method_uses_universal_occurrence_semantics(self) -> None:
        """seq.before(A, B) requires every A occurrence to precede every B occurrence."""
        assert parse_and_solve(
            """
S = set(a, b)
row = sequence(S)
row.before(a, b)
"""
        ) == 1
        assert parse_and_solve(
            """
A = set(a0, a1)
B = set(b0, b1)
row = sequence(A + B)
row.before(A, B)
"""
        ) == 4


    def test_before_method_quantifies_only_sequence_occurrences(self) -> None:
        """Absent A/B occurrences make before vacuously true and its negation false."""
        source = """
A = set(a)
B = set(b)
S = set(c)
row = sequence(S)
"""
        assert parse_and_solve(source + "row.before(A, B)\n") == 1
        assert parse_and_solve(source + "not row.before(A, B)\n") == 0


    def test_negative_before_method_requires_sequence_counterexample(self) -> None:
        """not seq.before(A, B) means some in-sequence A/B pair violates before."""
        assert parse_and_solve(
            """
A = set(a)
B = set(b)
row = sequence(A + B)
not row.before(A, B)
"""
        ) == 1


    def test_less_than_pattern_uses_strict_order(self) -> None:
        """a < a is false because pattern before semantics are strict."""
        assert parse_and_solve(
            """
S = set(a)
row = sequence(S)
a < a in row
"""
        ) == 0


    def test_together_method_uses_projection_block_semantics(self) -> None:
        """Only group occurrences that actually appear in seq need to form one block."""
        source = """
s = set(e1, e2)
seq = choose_replace_sequence(s, 3)
"""
        assert parse_and_solve(source + "seq.together(set(e2, e3))\n") == 7
        assert parse_and_solve(source + "not seq.together(set(e2, e3))\n") == 1


    def test_group_next_to_pattern_uses_occurrence_semantics(self) -> None:
        """next_to(A, b) means some A entity is adjacent to b."""
        assert parse_and_solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
next_to(A, b) in row
"""
        ) == 6


    def test_group_next_to_pattern_for_each_left_uses_coverage_semantics(self) -> None:
        """next_to(A, b) in row for each A requires every A occurrence to be adjacent to b."""
        assert parse_and_solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
next_to(A, b) in row for each A
"""
        ) == 2


    def test_predecessor_pattern_for_each_right_uses_coverage_semantics(self) -> None:
        """(A, b) in row for each b requires b to have an A predecessor."""
        assert parse_and_solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
(A, b) in row for each b
"""
        ) == 4


    def test_predecessor_pattern_for_each_left_can_be_unsatisfiable(self) -> None:
        """The coverage anchor is semantic, not always the first argument."""
        assert parse_and_solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
(A, b) in row for each A
"""
        ) == 0


    def test_sequence_of_composition_part_constrains_part_size(self) -> None:
        """Full sequence over a dynamic composition part must fix the part size."""
        assert parse_and_solve(
            """
set_0 = set(e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8)
compose_0 = compose(set_0, 3)
sequence_0 = sequence(compose_0[2])
|compose_0[2]| > 0
(e_8, e_6) in sequence_0
"""
        ) == 26608


    def test_local_coverage_quantifies_only_sequence_occurrences(self) -> None:
        """for each A ranges over A occurrences in the sequence, not all A entities."""
        source = """
A = set(a)
S = set(c)
row = sequence(S)
"""
        assert parse_and_solve(source + "(A, c) in row for each A\n") == 1
        assert parse_and_solve(source + "not ((A, c) in row for each A)\n") == 0


    def test_negative_predecessor_pattern_for_each_uses_boolean_negation(self) -> None:
        """not (... for each b) means some b has no matching predecessor."""
        assert parse_and_solve(
            """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
not ((A, b) in row for each b)
"""
        ) == 2


    def test_predecessor_pattern_not_in_aliases_boolean_negation(self) -> None:
        """(a, b) not in seq is accepted as an alias for not ((a, b) in seq)."""
        source = """
S = set(a, b, c)
row = sequence(S)
(a, b) not in row
"""
        canonical = """
S = set(a, b, c)
row = sequence(S)
not ((a, b) in row)
"""
        assert parse_and_solve(source) == parse_and_solve(canonical) == 4


    def test_less_than_pattern_not_in_aliases_boolean_negation(self) -> None:
        """a < b not in seq is accepted as an alias for not (a < b in seq)."""
        source = """
S = set(a, b)
row = sequence(S)
a < b not in row
"""
        canonical = """
S = set(a, b)
row = sequence(S)
not (a < b in row)
"""
        assert parse_and_solve(source) == parse_and_solve(canonical) == 1


    def test_negative_predecessor_pattern_forbids_all_occurrences(self) -> None:
        """not ((a, b) in seq) means no matching predecessor pair occurs."""
        assert parse_and_solve(
            """
creatures = bag(crocodile: 4, catfish, squid: 2)
order = sequence(creatures)
not ((crocodile, crocodile) in order)
"""
        ) == 3


    def test_negative_local_pattern_uses_direct_fo_encoding(self) -> None:
        """Negative local patterns should not allocate a count predicate/validator."""
        positive = _encode_single_component(
            """
S = set(a, b, c)
row = sequence(S)
(a, b) in row
"""
        )
        negative = _encode_single_component(
            """
S = set(a, b, c)
row = sequence(S)
not ((a, b) in row)
"""
        )

        positive_problem, positive_decoder = positive
        negative_problem, negative_decoder = negative
        assert len(positive_problem.problem.weights) == 1
        assert len(positive_decoder.validator) == 1
        assert len(negative_problem.problem.weights) == 0
        assert len(negative_decoder.validator) == 0
