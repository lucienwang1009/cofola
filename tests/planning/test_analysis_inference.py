"""Planning analysis inference: entity, max-size, and merged analyses."""
from __future__ import annotations

from cofola.frontend import (
    BagChoose,
    BagCountAtom,
    BagInit,
    Entity,
    ProblemBuilder,
    SetChoose,
    SetInit,
    SizeConstraint,
)
from cofola.planing.analysis.entities import EntityAnalysis
from cofola.planing.analysis.bag_classify import BagClassification
from cofola.planing.analysis.merged import MergedAnalysis
from cofola.planing.analysis.max_size import MaxSizeInference
from cofola.planing.pass_manager import AnalysisManager
from cofola.parser.parser import parse
from tests.helpers import _ref_named


class TestPlanningAnalysisInference(object):
    """Planner analysis facts, LP bounds, and merged analysis behavior."""

    def test_max_size_inference_skips_size_atom_constraints(self) -> None:
        """LP inference only reasons about raw object-cardinality terms."""
        builder = ProblemBuilder()
        a = Entity("a")
        source = builder.add(BagInit(entity_multiplicity=((a, 2),)), name="B")
        chosen = builder.add(BagChoose(source=source), name="C")
        builder.add_constraint(
            SizeConstraint(
                terms=((BagCountAtom(bag=chosen, entity=a), 1),),
                comparator="==",
                rhs=1,
            )
        )
        problem = builder.build()

        result = AnalysisManager(problem).get(MaxSizeInference)

        assert result.exact_sizes == {}
        assert result.max_sizes == {}
        assert not result.unsatisfiable


    def test_max_size_inference_keeps_raw_ref_constraints(self) -> None:
        """Raw ObjRef size constraints remain LP-compatible."""
        builder = ProblemBuilder()
        source = builder.add(
            SetInit(entities=frozenset({Entity("a"), Entity("b"), Entity("c")})),
            name="S",
        )
        chosen = builder.add(SetChoose(source=source), name="T")
        builder.add_constraint(
            SizeConstraint(terms=((chosen, 1),), comparator="==", rhs=2)
        )
        problem = builder.build()

        result = AnalysisManager(problem).get(MaxSizeInference)

        assert result.exact_sizes[chosen] == 2


    def test_max_size_inference_uses_entity_capacity_bounds(self) -> None:
        """LP inference should reject constraints exceeding known object capacity."""
        problem = parse("""
S = set(a, b)
T = choose(S)
|T| >= 3
""")

        result = AnalysisManager(problem).get(MaxSizeInference)

        assert result.unsatisfiable


    def test_max_size_inference_rejects_exact_size_above_capacity(self) -> None:
        """Known exact objects should be fixed by LP bounds."""
        problem = parse("""
S = set(a, b)
|S| == 3
""")

        result = AnalysisManager(problem).get(MaxSizeInference)

        assert result.unsatisfiable


    def test_max_size_inference_tightens_upper_bound_from_inequality(self) -> None:
        """Upper-bound constraints should refine max_sizes through bounded LP."""
        problem = parse("""
S = set(a, b, c)
T = choose(S)
|T| <= 1
""")
        chosen = _ref_named(problem, "T")

        result = AnalysisManager(problem).get(MaxSizeInference)

        assert not result.unsatisfiable
        assert result.max_sizes[chosen] == 1


    def test_merged_analysis_rejects_exact_size_above_tightened_max(self) -> None:
        """Merging LP max bounds must not leave exact_size > max_size."""
        problem = parse("""
S = set(a, b)
|S| <= 1
""")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.unsatisfiable


    def test_merged_analysis_does_not_mutate_entity_analysis_cache(self) -> None:
        """MergedAnalysis should refine a deep copy of EntityAnalysis facts."""
        problem = parse("""
S = set(a, b, c)
T = choose(S)
|T| <= 1
""")
        chosen = _ref_named(problem, "T")
        am = AnalysisManager(problem)

        base = am.get(EntityAnalysis)
        merged = am.get(MergedAnalysis)

        assert base.set_info[chosen].max_size == 3
        assert merged.set_info[chosen].max_size == 1


    def test_merged_analysis_propagates_full_ordered_source_exact_size(self) -> None:
        """A full ordered collection has the same exact size as its source."""
        problem = parse("""
U = set(a, b, c, d)
S = choose(U)
|S| == 2
row = sequence(S)
""")
        chosen = _ref_named(problem, "S")
        row = _ref_named(problem, "row")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert analysis.set_info[chosen].exact_size == 2
        assert analysis.set_info[row].exact_size == 2


    def test_entity_analysis_caps_bag_choice_multiplicities_by_size(self) -> None:
        """A fixed choice size bounds every per-entity multiplicity."""
        problem = parse("""
B = bag(a: 100, b: 100)
C = choose(B, 2)
""")
        source = _ref_named(problem, "B")
        chosen = _ref_named(problem, "C")

        analysis = AnalysisManager(problem).get(EntityAnalysis)

        assert set(analysis.bag_info[source].p_entities_multiplicity.values()) == {100}
        assert analysis.bag_info[chosen].max_size == 2
        assert analysis.bag_info[chosen].exact_size == 2
        assert set(analysis.bag_info[chosen].p_entities_multiplicity.values()) == {2}


    def test_merged_analysis_caps_multiplicities_by_inferred_max_size(self) -> None:
        """An LP-derived bag-size bound also bounds every multiplicity."""
        problem = parse("""
B = bag(a: 100, b: 100)
C = choose(B)
|C| <= 2
""")
        chosen = _ref_named(problem, "C")

        analysis = AnalysisManager(problem).get(MergedAnalysis)

        assert not analysis.unsatisfiable
        assert analysis.bag_info[chosen].max_size == 2
        assert set(analysis.bag_info[chosen].p_entities_multiplicity.values()) == {2}


    def test_single_item_bag_choice_has_unit_multiplicity_bounds(self) -> None:
        """Choosing one item leaves no entity with multiplicity above one."""
        problem = parse("""
B = bag(a: 100, b: 100)
C = choose(B, 1)
""")
        chosen = _ref_named(problem, "C")

        info = AnalysisManager(problem).get(EntityAnalysis).bag_info[chosen]

        assert info.max_size == 1
        assert info.exact_size == 1
        assert set(info.p_entities_multiplicity.values()) == {1}


    def test_bag_classification_uses_tightened_child_multiplicities(self) -> None:
        """A chosen bag must not retain its source's stale multiplicity class."""
        problem = parse("""
B = bag(a: 100, b: 100)
C = choose(B, 2)
""")
        chosen = _ref_named(problem, "C")

        info = AnalysisManager(problem).get(BagClassification).bag_info[chosen]

        assert set(info.p_entities_multiplicity.values()) == {2}
        assert info.indis_entities == {2: set(info.p_entities_multiplicity)}
        assert info.dis_entities == set()
