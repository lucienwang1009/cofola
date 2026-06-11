"""WFOMC backend boundary and semantic regression tests."""
from __future__ import annotations

import pytest
from flint import fmpq
from sympy import Eq, var

from cofola.backend.wfomc.api import (
    Algo,
    Pred,
    WFOMCResult,
    parse as parse_formula,
    top,
)
from cofola.backend.wfomc.backend import (
    WFOMC_GLOBAL_PASSES,
    WFOMC_LOCAL_PASSES,
    WFOMCBackend,
)
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.decoder import Decoder
from cofola.backend.wfomc.encoder import encode
from cofola.backend.wfomc.formula_helpers import exactly_one_qf
from cofola.frontend import (
    BagEqConstraint,
    BagInit,
    BagSubsetConstraint,
    Entity,
    ObjRef,
    Problem,
    SequenceDef,
    SequencePatternConstraint,
    SetInit,
    TupleIndexEq,
)
from cofola.planing.analysis.entities import AnalysisResult, BagInfo, SetInfo
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import LoweringPass
from cofola.planing.pipeline import PlaningPipeline
from cofola.parser.parser import parse
from cofola.solver import parse_and_solve


def test_constant_result_treats_absent_weight_generators_as_zero() -> None:
    generator = var("v_absent")
    accepted = Decoder(1, [generator], [Eq(generator, 0)], [])
    rejected = Decoder(1, [generator], [Eq(generator, 1)], [])

    result = WFOMCResult(fmpq(1))
    assert accepted.decode_result(result) == 1
    assert rejected.decode_result(result) == 0


def test_exactly_one_qf_requires_its_only_predicate() -> None:
    predicate = Pred("P", 1)

    assert exactly_one_qf([predicate]) == parse_formula("P(X)")
    assert exactly_one_qf([predicate]) != top


def test_exactly_one_qf_preserves_multi_predicate_semantics() -> None:
    left = Pred("P", 1)
    right = Pred("Q", 1)

    assert exactly_one_qf([left, right]) == parse_formula(
        "(P(X) | Q(X)) & ~(P(X) & Q(X))"
    )
    with pytest.raises(ValueError, match="at least one predicate"):
        exactly_one_qf([])


def _encode_single_component(source: str) -> tuple[object, object]:
    problem = parse(source)
    backend = WFOMCBackend(lifted=False)
    schedule = PlaningPipeline(backend.planning_profile()).process(problem)
    assert len(schedule.branches) == 1
    assert len(schedule.branches[0].components) == 1
    component, analysis = schedule.branches[0].components[0]
    return encode(component, analysis, lifted=False)


class TestWFOMCBackendProfile(object):
    """Backend profile and planner integration."""

    def test_wfomc_backend_declares_default_planning_profile(self) -> None:
        profile = WFOMCBackend().planning_profile()

        assert profile.global_passes == WFOMC_GLOBAL_PASSES
        assert profile.local_passes == WFOMC_LOCAL_PASSES
        assert profile.local_passes is not None
        assert any(
            isinstance(pass_spec, FixedPointPass) and pass_spec.pass_cls is LoweringPass
            for pass_spec in profile.local_passes
        )


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
        assert len(positive_problem.weights) == 1
        assert len(positive_decoder.validator) == 1
        assert len(negative_problem.weights) == 0
        assert len(negative_decoder.validator) == 0


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
        unsized_choose_tuple = """
S = bag(A: 5, B: 4, C: 2)
T = choose_tuple(S)
"""

        assert (
            parse_and_solve(full_choose_tuple)
            == parse_and_solve(unsized_choose_tuple)
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


class TestWFOMCEncodingBoundaries(object):
    """Backend error boundaries and encoding invariants."""

    def test_encode_does_not_mutate_analysis_for_unlifted_mode(self) -> None:
        """Encoding should not rewrite cached analysis facts in-place."""
        a = Entity("a")
        b = Entity("b")
        ref = ObjRef(0)
        problem = Problem(
            defs=((ref, BagInit(entity_multiplicity=((a, 2), (b, 2)))),),
            constraints=(),
            names=((ref, "B"),),
        )
        analysis = AnalysisResult(
            set_info={},
            bag_info={
                ref: BagInfo(
                    p_entities_multiplicity={a: 2, b: 2},
                    max_size=4,
                    dis_entities=set(),
                    indis_entities={2: {a, b}},
                    exact_size=4,
                )
            },
            all_entities={a, b},
            singletons=set(),
        )

        encode(problem, analysis, lifted=False)

        assert analysis.bag_info[ref].dis_entities == set()
        assert analysis.bag_info[ref].indis_entities == {2: {a, b}}


    def test_context_rejects_multiple_sequences_in_one_component(self) -> None:
        """The WFOMC backend currently has one global sequence-order context."""
        a = Entity("a")
        source = ObjRef(0)
        first = ObjRef(1)
        second = ObjRef(2)
        problem = Problem(
            defs=(
                (source, SetInit(entities=frozenset({a}))),
                (first, SequenceDef(source=source)),
                (second, SequenceDef(source=source)),
            ),
            constraints=(),
            names=((source, "S"), (first, "A"), (second, "B")),
        )
        analysis = AnalysisResult(
            set_info={
                source: SetInfo(p_entities={a}, max_size=1, exact_size=1),
                first: SetInfo(p_entities={a}, max_size=1, exact_size=1),
                second: SetInfo(p_entities={a}, max_size=1, exact_size=1),
            },
            bag_info={},
            all_entities={a},
            singletons={a},
        )

        with pytest.raises(ValueError, match="at most one sequence object"):
            Context(problem, analysis)


    def test_tuple_index_constraints_reaching_backend_are_errors(self) -> None:
        """Lowering owns tuple index constraints; the backend should not ignore leaks."""
        a = Entity("a")
        source = ObjRef(0)
        problem = Problem(
            defs=((source, SetInit(entities=frozenset({a}))),),
            constraints=(TupleIndexEq(tuple_ref=ObjRef(99), index=0, entity=a),),
            names=((source, "S"),),
        )
        analysis = AnalysisResult(
            set_info={source: SetInfo(p_entities={a}, max_size=1, exact_size=1)},
            bag_info={},
            all_entities={a},
            singletons={a},
        )

        with pytest.raises(NotImplementedError, match="TupleIndexEq reached encoder"):
            encode(problem, analysis)


    def test_bag_subset_constraint_rejects_non_bag_refs(self) -> None:
        """Malformed public-builder input should fail visibly at the backend."""
        a = Entity("a")
        left = ObjRef(0)
        right = ObjRef(1)
        problem = Problem(
            defs=(
                (left, SetInit(entities=frozenset({a}))),
                (right, SetInit(entities=frozenset({a}))),
            ),
            constraints=(BagSubsetConstraint(sub=left, sup=right, positive=True),),
            names=((left, "A"), (right, "B")),
        )
        analysis = AnalysisResult(
            set_info={
                left: SetInfo(p_entities={a}, max_size=1, exact_size=1),
                right: SetInfo(p_entities={a}, max_size=1, exact_size=1),
            },
            bag_info={},
            all_entities={a},
            singletons={a},
        )

        with pytest.raises(TypeError, match="BagSubsetConstraint requires"):
            encode(problem, analysis)


    def test_bag_equality_constraint_rejects_non_bag_refs(self) -> None:
        """Bag equality should not silently disappear for invalid refs."""
        a = Entity("a")
        left = ObjRef(0)
        right = ObjRef(1)
        problem = Problem(
            defs=(
                (left, SetInit(entities=frozenset({a}))),
                (right, SetInit(entities=frozenset({a}))),
            ),
            constraints=(BagEqConstraint(left=left, right=right, positive=True),),
            names=((left, "A"), (right, "B")),
        )
        analysis = AnalysisResult(
            set_info={
                left: SetInfo(p_entities={a}, max_size=1, exact_size=1),
                right: SetInfo(p_entities={a}, max_size=1, exact_size=1),
            },
            bag_info={},
            all_entities={a},
            singletons={a},
        )

        with pytest.raises(TypeError, match="BagEqConstraint requires"):
            encode(problem, analysis)


    def test_unknown_sequence_pattern_reaching_backend_is_an_error(self) -> None:
        """Unknown sequence patterns should not become no-op constraints."""
        a = Entity("a")
        source = ObjRef(0)
        seq = ObjRef(1)
        problem = Problem(
            defs=(
                (source, SetInit(entities=frozenset({a}))),
                (seq, SequenceDef(source=source)),
            ),
            constraints=(SequencePatternConstraint(seq=seq, pattern=object(), positive=True),),
            names=((source, "S"), (seq, "T")),
        )
        analysis = AnalysisResult(
            set_info={
                source: SetInfo(p_entities={a}, max_size=1, exact_size=1),
                seq: SetInfo(p_entities={a}, max_size=1, exact_size=1),
            },
            bag_info={},
            all_entities={a},
            singletons={a},
        )

        with pytest.raises(TypeError, match="Unknown sequence pattern type"):
            encode(problem, analysis)


    def test_backend_does_not_convert_unexpected_solver_errors_to_zero(self, monkeypatch) -> None:
        """Only known WFOMC degenerate IndexError cases should become count 0."""

        class FakeInputProblem(object):
            constraints = ()

            def iter_objects(self):
                return iter(())

        class FakeProblem(object):
            sentence = top

        class FakeDecoder(object):
            def decode_result(self, result: object) -> int:
                return 1

        def fake_encode(problem: object, analysis: object, lifted: bool):
            return FakeProblem(), FakeDecoder()

        def fake_solve_wfomc(problem: object, algo: Algo, unary_evidence_strategy: object,
                             *, linear_order_encoding=None):
            raise ValueError("backend bug")

        import cofola.backend.wfomc.backend as backend_module

        monkeypatch.setattr(backend_module, "encode", fake_encode)
        monkeypatch.setattr(backend_module, "solve_wfomc", fake_solve_wfomc)

        with pytest.raises(ValueError, match="backend bug"):
            WFOMCBackend().solve(FakeInputProblem(), object())
