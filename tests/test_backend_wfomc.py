"""WFOMC backend boundary and semantic regression tests."""
from __future__ import annotations

import pytest
from flint import fmpq
from sympy import Eq, var

from cofola.backend.wfomc.api import (
    Algo,
    Pred,
    WFOMCResult,
    parse,
    top,
)
from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.backend.wfomc.constraint_encoders import _count_singleton_violations
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

    assert exactly_one_qf([predicate]) == parse("P(X)")
    assert exactly_one_qf([predicate]) != top


def test_exactly_one_qf_preserves_multi_predicate_semantics() -> None:
    left = Pred("P", 1)
    right = Pred("Q", 1)

    assert exactly_one_qf([left, right]) == parse(
        "(P(X) | Q(X)) & ~(P(X) & Q(X))"
    )
    with pytest.raises(ValueError, match="at least one predicate"):
        exactly_one_qf([])


def test_full_choice_keeps_weighted_source_evidence_for_decoder() -> None:
    """Constant folding must not prune evidence for weighted predicates.

    The full-size choice is folded into its source. The source predicates then
    disappear from the sentence, but the decoder still consumes their
    polynomial degrees to recover the unique full selection.
    """
    assert parse_and_solve(
        """
defective = set(d0...2)
working = set(w0...3)
purchase = choose(defective + working, 5)
|purchase & defective| >= 2
"""
    ) == 1


def test_bag_difference_counts_leftover_multiplicities() -> None:
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


def test_small_choice_from_high_multiplicity_bag() -> None:
    """Choice size should cap irrelevant polynomial multiplicity degrees."""
    assert parse_and_solve(
        "B = bag(a: 100, b: 100)\nC = choose(B, 2)\n"
    ) == 3


@pytest.mark.parametrize(
    ("source", "constraint", "expected"),
    [
        ("bag(a: 1)", "sub subset sup", 3),
        ("bag(a: 1)", "not sub subset sup", 1),
        ("bag(a: 1)", "sub == sup", 2),
        ("bag(a: 1)", "sub != sup", 2),
        ("bag(a: 1, b: 2)", "sub subset sup", 18),
        ("bag(a: 1, b: 2)", "not sub subset sup", 18),
        ("bag(a: 1, b: 2)", "sub == sup", 6),
        ("bag(a: 1, b: 2)", "sub != sup", 30),
    ],
)
def test_bag_relations_constrain_singleton_entities(
    source: str, constraint: str, expected: int,
) -> None:
    """Bag relations must compare singleton membership as well as multiplicities."""
    assert parse_and_solve(
        f"B = {source}\nsub = choose(B)\nsup = choose(B)\n{constraint}\n"
    ) == expected


def test_singleton_violation_counts_use_distinct_variables() -> None:
    """Repeated helper calls must not share a generating variable."""
    context = Context(
        Problem(defs=(), constraints=(), names=()),
        AnalysisResult(set_info={}, bag_info={}, all_entities=set(), singletons=set()),
    )

    first = _count_singleton_violations("bag_eq_viol_1_2", "P(X)", context)
    second = _count_singleton_violations("bag_eq_viol_1_2", "P(X)", context)

    assert first != second
    assert len(context.weighting) == 2
    assert {positive for positive, _ in context.weighting.values()} == {first, second}


@pytest.mark.parametrize(
    ("object_name", "constraint"),
    [
        ("bag_subset_viol_1_2", "not sub subset sup"),
        ("bag_eq_viol_1_2", "sub != sup"),
    ],
)
def test_singleton_violation_counts_do_not_collide_with_object_names(
    object_name: str, constraint: str,
) -> None:
    """Renaming an object to a violation-counter prefix must not change counts."""
    for name in ("selection", object_name):
        assert parse_and_solve(
            "B = bag(a: 1)\nsub = choose(B)\nsup = choose(B)\n"
            f"{name} = choose(supp(sub), 1)\n{constraint}\n"
        ) == 1


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_repeated_bag_inequality_preserves_count(repetitions: int) -> None:
    """Repeating the same inequality must not affect the number of solutions."""
    assert parse_and_solve(
        "B = bag(a: 1, b: 2)\nsub = choose(B)\nsup = choose(B)\n"
        + "sub != sup\n" * repetitions
    ) == 30


@pytest.mark.parametrize(("membership", "expected"), [("in", 1), ("not in", 3)])
def test_bag_difference_subtracts_singleton_entities(
    membership: str, expected: int,
) -> None:
    """For a singleton, membership in X - Y means membership in X and not Y."""
    assert parse_and_solve(
        "B = bag(a: 1)\nX = choose(B)\nY = choose(B)\nZ = X - Y\n"
        f"a {membership} Z\n"
    ) == expected


@pytest.mark.parametrize(
    ("operation", "size", "expected"), [("X + Y", 1, 6), ("X & Y", 4, 1)]
)
def test_derived_bag_tracks_equal_multiplicity_entities(
    operation: str, size: int, expected: int,
) -> None:
    """A derived bag must keep entities its sources classify as indistinguishable."""
    assert parse_and_solve(
        "B = bag(a: 2, b: 2)\nX = choose(B)\nY = choose(B)\n"
        f"Z = {operation}\n|Z| == {size}\n"
    ) == expected


@pytest.mark.parametrize(
    ("operation", "expected"), [("X + Y", 1), ("X & Y", 5), ("X - Y", 6)]
)
def test_empty_derived_bag_size_is_satisfiable(operation: str, expected: int) -> None:
    """Empty derived bags still count models with zero multiplicity degrees."""
    assert parse_and_solve(
        "B = bag(a: 2)\nX = choose(B)\nY = choose(B)\n"
        f"Z = {operation}\n|Z| == 0\n"
    ) == expected


@pytest.mark.parametrize("derived", ["I = C", "I = C & B"])
def test_partition_of_bag_with_variable_singletons(derived: str) -> None:
    """Part counts must not force a chosen source's singletons out of every part."""
    assert parse_and_solve(
        f"B = bag(a, b, c)\nC = choose(B, 2)\n{derived}\nP = partition(I, 2)\n"
    ) == 6


def test_partition_singletons_kept_in_symmetry_breaking() -> None:
    """Parts that differ only in singleton content still need symmetry breaking."""
    assert parse_and_solve(
        "B = bag(a: 1, b: 1, c: 4)\nP = partition(B, 2)\n"
    ) == 10


def test_bag_union_preserves_max_multiplicity_for_dynamic_sources() -> None:
    """Bag union should constrain multiplicities with max(left, right)."""
    assert parse_and_solve(
        """
B = bag(a: 2)
C = choose(B)
D = C + B
|D| == 2
"""
    ) == 3


def test_bag_union_count_atom_uses_encoded_multiplicity() -> None:
    """Bag count atoms should read the resolved bag multiplicity expression."""
    assert parse_and_solve(
        """
B = bag(a: 2)
C = choose(B)
D = C + B
D.count(a) == 2
"""
    ) == 3


def test_bag_count_atom_on_base_bag_uses_constant_multiplicity() -> None:
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


def test_group_less_than_pattern_uses_universal_semantics() -> None:
    """A < B requires every A entity to precede every B entity."""
    assert parse_and_solve(
        """
A = set(a0, a1)
B = set(b0, b1)
row = sequence(A + B)
A < B in row
"""
    ) == 4


def test_group_next_to_pattern_uses_universal_semantics() -> None:
    """next_to(A, b) requires every A entity to be adjacent to b."""
    assert parse_and_solve(
        """
A = set(a0, a1)
S = A + set(b)
row = sequence(S)
next_to(A, b) in row
"""
    ) == 2


def test_negative_predecessor_pattern_forbids_all_occurrences() -> None:
    """(a, b) not in seq means no matching predecessor pair occurs."""
    assert parse_and_solve(
        """
creatures = bag(crocodile: 4, catfish, squid: 2)
order = sequence(creatures)
(crocodile, crocodile) not in order
"""
    ) == 3


def test_negative_bag_equality_is_any_multiplicity_difference() -> None:
    """B != C means at least one multiplicity differs."""
    assert parse_and_solve(
        """
S = bag(a: 2, b: 2)
B = choose(S)
C = choose(S)
B != C
"""
    ) == 72


def test_negative_bag_subset_is_any_multiplicity_violation() -> None:
    """not (B subset C) means at least one entity has a larger multiplicity."""
    assert parse_and_solve(
        """
S = bag(a: 2, b: 2)
B = choose(S)
C = choose(S)
not B subset C
"""
    ) == 45


def test_encode_does_not_mutate_analysis_for_unlifted_mode() -> None:
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


def test_context_rejects_multiple_sequences_in_one_component() -> None:
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


def test_tuple_index_constraints_reaching_backend_are_errors() -> None:
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


def test_bag_subset_constraint_rejects_non_bag_refs() -> None:
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


def test_bag_equality_constraint_rejects_non_bag_refs() -> None:
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


def test_unknown_sequence_pattern_reaching_backend_is_an_error() -> None:
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


def test_backend_does_not_convert_unexpected_solver_errors_to_zero(monkeypatch) -> None:
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

    def fake_solve_wfomc(
        problem: object,
        algo: Algo,
        unary_evidence_strategy: object,
        linear_order_encoding: object = None,
    ):
        raise ValueError("backend bug")

    import cofola.backend.wfomc.backend as backend_module

    monkeypatch.setattr(backend_module, "encode", fake_encode)
    monkeypatch.setattr(backend_module, "solve_wfomc", fake_solve_wfomc)

    with pytest.raises(ValueError, match="backend bug"):
        WFOMCBackend().solve(FakeInputProblem(), object())
