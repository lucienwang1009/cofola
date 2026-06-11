"""Planning analysis boundary cases and policy edges."""
from __future__ import annotations

import pytest

from cofola.frontend import (
    BagInit,
    Entity,
    ObjRef,
    ProblemBuilder,
    SetPartDef,
)
from cofola.planing.analysis.bag_classify import BagClassification
from cofola.planing.analysis.entities import EntityAnalysis
from cofola.planing.analysis.merged import MergedAnalysis
from cofola.planing.pass_manager import AnalysisManager
from cofola.parser.parser import parse
from tests.helpers import _ref_named


class TestPlanningAnalysisBoundaries(object):
    """Unsatisfiable and malformed-input analysis boundaries."""

    def test_set_choose_size_larger_than_source_is_unsatisfiable(self) -> None:
        """A no-replacement set choice cannot request more elements than its source."""
        problem = parse("""
S = set(a, b)
T = choose(S, 3)
""")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.unsatisfiable


    def test_bag_choose_size_larger_than_source_is_unsatisfiable(self) -> None:
        """A no-replacement bag choice cannot request more items than source capacity."""
        problem = parse("""
B = bag(a: 2)
C = choose(B, 3)
""")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.unsatisfiable


    def test_entity_analysis_set_difference_keeps_conservative_capacity(self) -> None:
        """A disjoint RHS does not shrink the possible size of a set difference."""
        problem = parse("""
A = set(a)
B = set(b)
C = A - B
""")

        analysis = AnalysisManager(problem).get(EntityAnalysis)

        assert analysis.set_info[_ref_named(problem, "C")].max_size == 1


    def test_entity_analysis_bag_support_caps_by_distinct_entities(self) -> None:
        """Support size is bounded by distinct entities, not total multiplicity."""
        problem = parse("""
B = bag(a: 2, b: 2)
S = supp(B)
T = choose(S, 3)
""")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.set_info[_ref_named(problem, "S")].max_size == 2
        assert analysis.unsatisfiable


    def test_ordered_collection_size_larger_than_source_is_unsatisfiable(self) -> None:
        """A no-replacement ordered choice cannot exceed source capacity."""
        problem = parse("""
S = set(a, b)
T = choose_tuple(S, 3)
""")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.unsatisfiable


    def test_entity_analysis_infos_satisfy_size_invariants(self) -> None:
        """EntityAnalysis should centrally normalize SetInfo and BagInfo facts."""
        problem = parse("""
A = set(a, b)
B = set(b, c)
C = A + B
M = bag(a: 2, b: 2)
S = supp(M)
T = choose_tuple(A, 3)
""")

        analysis = AnalysisManager(problem).get(EntityAnalysis)

        assert analysis.unsatisfiable
        for info in analysis.set_info.values():
            assert 0 <= info.max_size <= len(info.p_entities)
            if info.exact_size is not None:
                assert 0 <= info.exact_size <= info.max_size
        for info in analysis.bag_info.values():
            assert all(mult >= 0 for mult in info.p_entities_multiplicity.values())
            assert 0 <= info.max_size <= sum(info.p_entities_multiplicity.values())
            if info.exact_size is not None:
                assert 0 <= info.exact_size <= info.max_size


    def test_entity_analysis_rejects_negative_bag_multiplicity(self) -> None:
        """Malformed public-builder input should not create negative BagInfo facts."""
        a = Entity("a")
        builder = ProblemBuilder()
        bag = builder.add(BagInit(entity_multiplicity=((a, -1),)), name="B")
        problem = builder.build()

        analysis = AnalysisManager(problem).get(EntityAnalysis)

        assert analysis.unsatisfiable
        assert analysis.bag_info[bag].p_entities_multiplicity[a] == 0
        assert analysis.bag_info[bag].max_size == 0
        assert analysis.bag_info[bag].exact_size == 0


    def test_entity_analysis_reports_invalid_part_partition_ref(self) -> None:
        """Malformed hand-built PartDef refs should fail with a controlled error."""
        builder = ProblemBuilder()
        builder.add(SetPartDef(partition=ObjRef(999), index=0), name="bad")
        problem = builder.build()

        with pytest.raises(ValueError, match="partition ref=999"):
            AnalysisManager(problem).get(EntityAnalysis)


    def test_bag_classification_reports_invalid_part_partition_ref(self) -> None:
        """The public bag-classification boundary should not leak AttributeError."""
        builder = ProblemBuilder()
        builder.add(SetPartDef(partition=ObjRef(999), index=0), name="bad")
        problem = builder.build()

        with pytest.raises(ValueError, match="partition ref=999"):
            AnalysisManager(problem).get(BagClassification)
