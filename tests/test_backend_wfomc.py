"""WFOMC backend boundary and semantic regression tests."""
from __future__ import annotations

import pytest
from wfomc import Algo

from cofola.backend.wfomc.backend import WFOMCBackend
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.encoder import encode
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

    with pytest.raises(ValueError, match="at most one sequence-like object"):
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
        def contain_linear_order_axiom(self) -> bool:
            return False

    class FakeDecoder(object):
        def decode_result(self, result: object) -> int:
            return 1

    def fake_encode(problem: object, analysis: object, lifted: bool):
        return FakeProblem(), FakeDecoder()

    def fake_solve_wfomc(problem: object, algo: Algo, use_partition_constraint: bool):
        raise ValueError("backend bug")

    import cofola.backend.wfomc.backend as backend_module

    monkeypatch.setattr(backend_module, "encode", fake_encode)
    monkeypatch.setattr(backend_module, "solve_wfomc", fake_solve_wfomc)

    with pytest.raises(ValueError, match="backend bug"):
        WFOMCBackend().solve(FakeInputProblem(), object())
