"""ASP program construction for the direct ASP backend."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable

from cofola.frontend.constraints import (
    BagCountAtom,
    BagEqConstraint,
    BagSubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    FuncPairConstraint,
    LessThanPattern,
    MembershipConstraint,
    NextToPattern,
    PredecessorPattern,
    SeqPatternCountAtom,
    SequencePatternConstraint,
    SizeConstraint,
    SubsetConstraint,
    TogetherPattern,
    TupleCountAtom,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagInit,
    BagIntersection,
    BagPartDef,
    BagSupport,
    BagUnion,
    CompositionDef,
    Entity,
    FuncDef,
    FuncImage,
    FuncInverse,
    FuncInverseImage,
    ObjRef,
    PartDef,
    PartitionDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetPartDef,
    SetUnion,
    TupleDef,
)
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult, BagInfo, SetInfo

__all__ = ["ASPEncoder", "ASPEncodingError"]


class ASPEncodingError(Exception):
    """Raised when a Cofola construct is outside the direct ASP prototype."""


@dataclass(frozen=True)
class _ValueExpr(object):
    """An integer value represented by a variable and the body that binds it."""

    var: str
    body: tuple[str, ...]


class ASPEncoder(object):
    """Translate one planning component to a clingo program."""

    def __init__(self, problem: Problem, analysis: AnalysisResult) -> None:
        self.problem = problem
        self.analysis = analysis
        self.entities = sorted(analysis.all_entities, key=lambda entity: entity.name)
        self.entity_ids = {entity: idx for idx, entity in enumerate(self.entities)}
        self.lines: list[str] = []
        self.aux_counter = 0
        self.global_max = self._global_max()

    def encode(self) -> str:
        """Return an ASP program whose stable models are Cofola solutions."""

        if not self.problem.defs:
            return "#program base.\n"
        self._emit_preamble()
        for ref, defn in self.problem.defs:
            self._emit_object(ref, defn)
        self._emit_size_definitions()
        self._emit_size_bounds()
        for constraint in self.problem.constraints:
            self._emit_constraint(constraint)
        return "\n".join(self.lines) + "\n"

    def _global_max(self) -> int:
        max_value = 0
        for info in self.analysis.set_info.values():
            max_value = max(max_value, info.max_size)
        for info in self.analysis.bag_info.values():
            max_value = max(
                max_value,
                info.max_size,
                *(info.p_entities_multiplicity.values() or [0]),
            )
        if max_value >= sys.maxsize // 4:
            raise ASPEncodingError(
                "ASP backend needs finite multiplicity bounds; add a size constraint."
            )
        return max_value

    def _emit_preamble(self) -> None:
        self.lines.append(f"num(0..{self.global_max}).")
        for entity, idx in self.entity_ids.items():
            self.lines.append(f"entity({idx}).")
            self.lines.append(f'entity_label({idx},"{self._quote(entity.name)}").')
        for ref, _ in self.problem.defs:
            self.lines.append(f"object({ref.id}).")

    def _emit_object(self, ref: ObjRef, defn: object) -> None:
        if isinstance(defn, SetInit):
            self._emit_constant_multiplicities(
                ref,
                {entity: int(entity in defn.entities) for entity in self.entities},
            )
        elif isinstance(defn, BagInit):
            self._emit_constant_multiplicities(ref, dict(defn.entity_multiplicity))
        elif isinstance(defn, SetChoose):
            self._emit_set_choose(ref, defn)
        elif isinstance(defn, SetChooseReplace):
            self._emit_set_choose_replace(ref, defn)
        elif isinstance(defn, BagChoose):
            self._emit_bag_choose(ref, defn)
        elif isinstance(defn, SetUnion):
            self._emit_set_union(ref, defn.left, defn.right)
        elif isinstance(defn, SetIntersection):
            self._emit_set_intersection(ref, defn.left, defn.right)
        elif isinstance(defn, SetDifference):
            self._emit_set_difference(ref, defn.left, defn.right)
        elif isinstance(defn, BagUnion):
            self._emit_bag_binary(ref, defn.left, defn.right, "max")
        elif isinstance(defn, BagAdditiveUnion):
            self._emit_bag_binary(ref, defn.left, defn.right, "add")
        elif isinstance(defn, BagIntersection):
            self._emit_bag_binary(ref, defn.left, defn.right, "min")
        elif isinstance(defn, BagDifference):
            self._emit_bag_binary(ref, defn.left, defn.right, "sub")
        elif isinstance(defn, BagSupport):
            self._emit_bag_support(ref, defn.source)
        elif isinstance(defn, (TupleDef, SequenceDef)):
            self._emit_ordered(ref, defn)
        elif isinstance(defn, (PartitionDef, CompositionDef)):
            self._emit_grouped(ref, defn)
        elif isinstance(defn, PartDef):
            # Part multiplicities are generated by the owning partition/composition.
            return
        elif isinstance(defn, (FuncDef, FuncImage, FuncInverseImage, FuncInverse)):
            raise ASPEncodingError(
                f"ASP backend does not yet support {type(defn).__name__}."
            )
        else:
            raise ASPEncodingError(f"Unsupported object for ASP: {type(defn).__name__}.")

    def _emit_constant_multiplicities(
        self,
        ref: ObjRef,
        multiplicities: dict[Entity, int],
    ) -> None:
        for entity in self.entities:
            self.lines.append(
                f"mult({ref.id},{self._entity_id(entity)},{multiplicities.get(entity, 0)})."
            )

    def _emit_set_choose(self, ref: ObjRef, defn: SetChoose) -> None:
        source = defn.source.id
        for entity in self.entities:
            eid = self._entity_id(entity)
            if self._max_mult(ref, entity) == 0:
                self.lines.append(f"mult({ref.id},{eid},0).")
                continue
            self.lines.append(f"1 {{ mult({ref.id},{eid},0); mult({ref.id},{eid},1) }} 1.")
            self.lines.append(
                f":- mult({ref.id},{eid},1), mult({source},{eid},0)."
            )

    def _emit_set_choose_replace(self, ref: ObjRef, defn: SetChooseReplace) -> None:
        source = defn.source.id
        for entity in self.entities:
            eid = self._entity_id(entity)
            ub = self._max_mult(ref, entity)
            self._emit_multiplicity_choice(ref, entity, ub)
            self.lines.append(
                f":- mult({ref.id},{eid},N), N > 0, mult({source},{eid},0)."
            )

    def _emit_bag_choose(self, ref: ObjRef, defn: BagChoose) -> None:
        source = defn.source.id
        for entity in self.entities:
            eid = self._entity_id(entity)
            ub = self._max_mult(ref, entity)
            self._emit_multiplicity_choice(ref, entity, ub)
            self.lines.append(
                f":- mult({ref.id},{eid},N), mult({source},{eid},SN), N > SN."
            )

    def _emit_set_union(self, ref: ObjRef, left: ObjRef, right: ObjRef) -> None:
        for entity in self.entities:
            eid = self._entity_id(entity)
            self.lines.append(f"mult({ref.id},{eid},1) :- mult({left.id},{eid},1).")
            self.lines.append(f"mult({ref.id},{eid},1) :- mult({right.id},{eid},1).")
            self.lines.append(
                f"mult({ref.id},{eid},0) :- mult({left.id},{eid},0), mult({right.id},{eid},0)."
            )

    def _emit_set_intersection(self, ref: ObjRef, left: ObjRef, right: ObjRef) -> None:
        for entity in self.entities:
            eid = self._entity_id(entity)
            self.lines.append(
                f"mult({ref.id},{eid},1) :- mult({left.id},{eid},1), mult({right.id},{eid},1)."
            )
            self.lines.append(f"mult({ref.id},{eid},0) :- mult({left.id},{eid},0).")
            self.lines.append(f"mult({ref.id},{eid},0) :- mult({right.id},{eid},0).")

    def _emit_set_difference(self, ref: ObjRef, left: ObjRef, right: ObjRef) -> None:
        for entity in self.entities:
            eid = self._entity_id(entity)
            self.lines.append(
                f"mult({ref.id},{eid},1) :- mult({left.id},{eid},1), mult({right.id},{eid},0)."
            )
            self.lines.append(f"mult({ref.id},{eid},0) :- mult({left.id},{eid},0).")
            self.lines.append(f"mult({ref.id},{eid},0) :- mult({right.id},{eid},1).")

    def _emit_bag_binary(
        self,
        ref: ObjRef,
        left: ObjRef,
        right: ObjRef,
        op: str,
    ) -> None:
        for entity in self.entities:
            eid = self._entity_id(entity)
            self._emit_multiplicity_choice(ref, entity, self._max_mult(ref, entity))
            base = (
                f"mult({ref.id},{eid},N), "
                f"mult({left.id},{eid},A), mult({right.id},{eid},B)"
            )
            if op == "add":
                self.lines.append(f":- {base}, N != A + B.")
            elif op == "max":
                self.lines.append(f":- {base}, N < A.")
                self.lines.append(f":- {base}, N < B.")
                self.lines.append(f":- {base}, N > A, N > B.")
            elif op == "min":
                self.lines.append(f":- {base}, N > A.")
                self.lines.append(f":- {base}, N > B.")
                self.lines.append(f":- {base}, N < A, N < B.")
            elif op == "sub":
                self.lines.append(f":- {base}, A >= B, N != A - B.")
                self.lines.append(f":- {base}, A < B, N != 0.")
            else:
                raise AssertionError(op)

    def _emit_bag_support(self, ref: ObjRef, source: ObjRef) -> None:
        for entity in self.entities:
            eid = self._entity_id(entity)
            self.lines.append(
                f"mult({ref.id},{eid},1) :- mult({source.id},{eid},N), N > 0."
            )
            self.lines.append(f"mult({ref.id},{eid},0) :- mult({source.id},{eid},0).")

    def _emit_ordered(
        self,
        ref: ObjRef,
        defn: TupleDef | SequenceDef,
    ) -> None:
        size = self._exact_size(ref)
        if size is None:
            raise ASPEncodingError(
                f"{type(defn).__name__} ref={ref.id} needs an exact size before ASP encoding."
            )
        for pos in range(size):
            self.lines.append(f"pos({ref.id},{pos}).")
        self.lines.append(
            f"can({ref.id},E) :- entity(E), mult({defn.source.id},E,N), N > 0."
        )
        if size > 0:
            self.lines.append(
                f"1 {{ at({ref.id},P,E) : can({ref.id},E) }} 1 :- pos({ref.id},P)."
            )
        self.lines.append(
            f"mult({ref.id},E,N) :- entity(E), num(N), "
            f"N = #count {{ P : at({ref.id},P,E) }}."
        )
        if not defn.replace:
            comparator = "!=" if not defn.choose else ">"
            self.lines.append(
                f":- mult({ref.id},E,N), mult({defn.source.id},E,SN), N {comparator} SN."
            )
    def _emit_grouped(
        self,
        ref: ObjRef,
        defn: PartitionDef | CompositionDef,
    ) -> None:
        parts = self._part_refs(ref)
        if len(parts) != defn.num_parts:
            raise ASPEncodingError(
                f"{type(defn).__name__} ref={ref.id} expected {defn.num_parts} "
                f"parts, found {len(parts)}."
            )

        for index, part_ref in enumerate(parts):
            self.lines.append(f"part({ref.id},{index},{part_ref.id}).")
            for entity in self.entities:
                self._emit_multiplicity_choice(
                    part_ref,
                    entity,
                    self._max_mult(part_ref, entity),
                )

        for entity in self.entities:
            eid = self._entity_id(entity)
            terms = " + ".join(f"N{idx}" for idx in range(len(parts))) or "0"
            body = [f"mult({defn.source.id},{eid},SN)"]
            body.extend(
                f"mult({part_ref.id},{eid},N{idx})"
                for idx, part_ref in enumerate(parts)
            )
            body.append(f"SN != {terms}")
            self.lines.append(f":- {', '.join(body)}.")

        if isinstance(defn, PartitionDef):
            self._emit_partition_canonical(parts)

    def _emit_partition_canonical(self, parts: list[ObjRef]) -> None:
        for left, right in zip(parts, parts[1:], strict=False):
            self._emit_part_not_lex_smaller(left, right)

    def _emit_part_not_lex_smaller(self, left: ObjRef, right: ObjRef) -> None:
        tag = self._new_aux_id()
        self.lines.append(f"same_part_prefix({tag},0).")
        for pos, entity in enumerate(self.entities):
            eid = self._entity_id(entity)
            self.lines.append(
                f"part_lex_smaller({tag}) :- same_part_prefix({tag},{pos}), "
                f"mult({left.id},{eid},A), mult({right.id},{eid},B), A < B."
            )
            self.lines.append(
                f"same_part_prefix({tag},{pos + 1}) :- "
                f"same_part_prefix({tag},{pos}), "
                f"mult({left.id},{eid},N), mult({right.id},{eid},N)."
            )
        self.lines.append(f":- part_lex_smaller({tag}).")

    def _emit_multiplicity_choice(
        self,
        ref: ObjRef,
        entity: Entity,
        upper: int,
    ) -> None:
        eid = self._entity_id(entity)
        if upper == 0:
            self.lines.append(f"mult({ref.id},{eid},0).")
            return
        self.lines.append(
            f"1 {{ mult({ref.id},{eid},N) : num(N), N <= {upper} }} 1."
        )

    def _emit_size_definitions(self) -> None:
        for ref, _ in self.problem.defs:
            self.lines.append(
                f"size({ref.id},N) :- num(N), "
                f"N = #sum {{ K,E : mult({ref.id},E,K) }}."
            )

    def _emit_size_bounds(self) -> None:
        for ref, defn in self.problem.defs:
            # Base objects have constant multiplicities, so their size is fixed
            # by construction — no bound is needed.
            if isinstance(defn, (SetInit, BagInit)):
                continue
            exact = self._exact_size(ref)
            if exact is not None:
                # An exact size already pins the total; the max-size bound
                # (max_size >= exact) would be redundant with `!= exact`.
                self.lines.append(
                    f":- #sum {{ K,E : mult({ref.id},E,K) }} != {exact}."
                )
                continue
            info = self._info(ref)
            if info is not None and info.max_size < sys.maxsize // 4:
                self.lines.append(
                    f":- #sum {{ K,E : mult({ref.id},E,K) }} > {info.max_size}."
                )

    def _emit_constraint(self, constraint: object) -> None:
        if isinstance(constraint, SizeConstraint):
            self._emit_size_constraint(constraint)
        elif isinstance(constraint, MembershipConstraint):
            self._emit_membership_constraint(constraint)
        elif isinstance(constraint, (SubsetConstraint, BagSubsetConstraint)):
            self._emit_multiplicity_subset(
                constraint.sub, constraint.sup, positive=constraint.positive
            )
        elif isinstance(constraint, DisjointConstraint):
            self._emit_disjoint_constraint(constraint)
        elif isinstance(constraint, (EqualityConstraint, BagEqConstraint)):
            self._emit_multiplicity_equality(
                constraint.left, constraint.right, positive=constraint.positive
            )
        elif isinstance(constraint, TupleIndexEq):
            self._emit_tuple_index_eq(constraint)
        elif isinstance(constraint, TupleIndexMembership):
            self._emit_tuple_index_membership(constraint)
        elif isinstance(constraint, SequencePatternConstraint):
            self._emit_sequence_pattern_constraint(constraint)
        elif isinstance(constraint, FuncPairConstraint):
            raise ASPEncodingError("ASP backend does not yet support function constraints.")
        else:
            raise ASPEncodingError(
                f"Unsupported constraint for ASP: {type(constraint).__name__}."
            )

    def _emit_size_constraint(self, constraint: SizeConstraint) -> None:
        values = [
            self._value_expr(atom, f"V{idx}")
            for idx, (atom, _coeff) in enumerate(constraint.terms)
        ]
        body = [part for value in values for part in value.body]
        expr = " + ".join(
            self._term_expr(value.var, coeff)
            for value, (_atom, coeff) in zip(values, constraint.terms, strict=True)
            if coeff != 0
        ) or "0"
        invalid = self._invalid_comparison(expr, constraint.comparator, constraint.rhs)
        self.lines.append(f":- {', '.join(body + [invalid])}.")

    def _emit_membership_constraint(self, constraint: MembershipConstraint) -> None:
        eid = self._entity_id(constraint.entity)
        if constraint.positive:
            self.lines.append(f":- mult({constraint.container.id},{eid},0).")
        else:
            self.lines.append(f":- mult({constraint.container.id},{eid},N), N > 0.")

    def _emit_multiplicity_subset(
        self,
        sub: ObjRef,
        sup: ObjRef,
        *,
        positive: bool,
    ) -> None:
        # Works for both set and bag subset: a violation is any entity whose
        # multiplicity in ``sub`` exceeds its multiplicity in ``sup``. For sets
        # (multiplicities in {0, 1}) this reduces to "in sub but not in sup".
        pred = self._violation_predicate()
        self.lines.append(
            f"{pred} :- entity(E), mult({sub.id},E,N), mult({sup.id},E,M), N > M."
        )
        self._constrain_violation(pred, positive=positive)

    def _emit_disjoint_constraint(self, constraint: DisjointConstraint) -> None:
        pred = self._violation_predicate()
        self.lines.append(
            f"{pred} :- entity(E), mult({constraint.left.id},E,N), N > 0, "
            f"mult({constraint.right.id},E,M), M > 0."
        )
        self._constrain_violation(pred, positive=constraint.positive)

    def _emit_multiplicity_equality(
        self,
        left: ObjRef,
        right: ObjRef,
        *,
        positive: bool,
    ) -> None:
        # Works for both set and bag equality: a violation is any entity whose
        # multiplicity differs between the two objects. For sets (multiplicities
        # in {0, 1}) this reduces to "differ in membership".
        pred = self._violation_predicate()
        self.lines.append(
            f"{pred} :- entity(E), mult({left.id},E,N), mult({right.id},E,M), N != M."
        )
        self._constrain_violation(pred, positive=positive)

    def _emit_tuple_index_eq(self, constraint: TupleIndexEq) -> None:
        if constraint.positive:
            self.lines.append(
                f":- not at({constraint.tuple_ref.id},{constraint.index},"
                f"{self._entity_id(constraint.entity)})."
            )
        else:
            self.lines.append(
                f":- at({constraint.tuple_ref.id},{constraint.index},"
                f"{self._entity_id(constraint.entity)})."
            )

    def _emit_tuple_index_membership(self, constraint: TupleIndexMembership) -> None:
        pred = self._violation_predicate()
        self.lines.append(
            f"{pred} :- at({constraint.tuple_ref.id},{constraint.index},E), "
            f"mult({constraint.container.id},E,N), N > 0."
        )
        # pred means the indexed value is in the container.
        if constraint.positive:
            self.lines.append(f":- not {pred}.")
        else:
            self.lines.append(f":- {pred}.")

    def _emit_sequence_pattern_constraint(
        self,
        constraint: SequencePatternConstraint,
    ) -> None:
        pattern = constraint.pattern
        if isinstance(pattern, LessThanPattern):
            self._emit_less_than_pattern(constraint)
        elif isinstance(pattern, TogetherPattern):
            self._emit_together_pattern(constraint)
        elif isinstance(pattern, (PredecessorPattern, NextToPattern)):
            self._emit_local_pattern(constraint)
        else:
            raise ASPEncodingError(f"Unsupported sequence pattern {type(pattern).__name__}.")

    def _emit_less_than_pattern(self, constraint: SequencePatternConstraint) -> None:
        if constraint.coverage is not None:
            raise ASPEncodingError("ASP backend does not support coverage on before patterns.")
        pattern = constraint.pattern
        if not isinstance(pattern, LessThanPattern):
            raise AssertionError(type(pattern).__name__)
        pred = self._violation_predicate()
        body = (
            self._arg_at_body(constraint.seq, "P1", pattern.left, "E1")
            + self._arg_at_body(constraint.seq, "P2", pattern.right, "E2")
            + ["P1 >= P2"]
        )
        self.lines.append(f"{pred} :- {', '.join(body)}.")
        self._constrain_violation(pred, positive=constraint.positive)

    def _emit_together_pattern(self, constraint: SequencePatternConstraint) -> None:
        if constraint.coverage is not None:
            raise ASPEncodingError("ASP backend does not support coverage on together patterns.")
        pattern = constraint.pattern
        if not isinstance(pattern, TogetherPattern):
            raise AssertionError(type(pattern).__name__)
        tag = self._new_aux_id()
        group = f"group_{tag}"
        gap = f"gap_{tag}"
        for body in self._arg_entity_bodies(pattern.group, "E"):
            self.lines.append(f"{group}(E) :- {', '.join(body)}.")
        self.lines.append(
            f"{gap} :- at({constraint.seq.id},P1,E1), {group}(E1), "
            f"at({constraint.seq.id},P2,E2), {group}(E2), P1 < PM, PM < P2, "
            f"at({constraint.seq.id},PM,EM), not {group}(EM)."
        )
        if constraint.positive:
            self.lines.append(f":- {gap}.")
        else:
            self.lines.append(f":- not {gap}.")

    def _emit_local_pattern(self, constraint: SequencePatternConstraint) -> None:
        pattern = constraint.pattern
        if not isinstance(pattern, (PredecessorPattern, NextToPattern)):
            raise AssertionError(type(pattern).__name__)
        match = self._emit_local_match_pred(constraint.seq, pattern)
        tag = self._new_aux_id()
        has_match = f"has_match_{tag}"

        first = pattern.first
        second = pattern.second

        self.lines.append(f"{has_match} :- {match}(P,Q).")
        if constraint.coverage is None:
            if constraint.positive:
                self.lines.append(f":- not {has_match}.")
            else:
                self.lines.append(f":- {has_match}.")
            return

        cov = f"coverage_{tag}"
        covered = f"covered_{tag}"
        uncovered = f"uncovered_{tag}"
        body = self._arg_at_body(constraint.seq, "P", constraint.coverage, "CE")
        self.lines.append(f"{cov}(P) :- {', '.join(body)}.")
        if self._same_pattern_arg(constraint.coverage, first):
            self.lines.append(f"{covered}(P) :- {cov}(P), {match}(P,_).")
        if self._same_pattern_arg(constraint.coverage, second):
            self.lines.append(f"{covered}(P) :- {cov}(P), {match}(_,P).")
        if not (
            self._same_pattern_arg(constraint.coverage, first)
            or self._same_pattern_arg(constraint.coverage, second)
        ):
            self.lines.append(f"{covered}(P) :- {cov}(P), {match}(P,_).")
            self.lines.append(f"{covered}(P) :- {cov}(P), {match}(_,P).")
        if constraint.positive:
            self.lines.append(f":- {cov}(P), not {covered}(P).")
        else:
            self.lines.append(f"{uncovered} :- {cov}(P), not {covered}(P).")
            self.lines.append(f":- not {uncovered}.")

    def _emit_local_match_pred(
        self,
        seq: ObjRef,
        pattern: PredecessorPattern | NextToPattern,
    ) -> str:
        tag = self._new_aux_id()
        match = f"match_{tag}"
        if isinstance(pattern, PredecessorPattern):
            first = pattern.first
            second = pattern.second
            relation_bodies = self._successor_bodies("P", "Q")
        elif isinstance(pattern, NextToPattern):
            first = pattern.first
            second = pattern.second
            relation_bodies = (
                self._successor_bodies("P", "Q")
                + self._successor_bodies("Q", "P")
            )
        else:
            raise AssertionError(type(pattern).__name__)

        for relation in relation_bodies:
            body = (
                self._arg_at_body(seq, "P", first, "E1")
                + self._arg_at_body(seq, "Q", second, "E2")
                + list(relation)
            )
            self.lines.append(f"{match}(P,Q) :- {', '.join(body)}.")
        return match

    def _successor_bodies(self, left_pos: str, right_pos: str) -> list[list[str]]:
        return [[f"{right_pos} = {left_pos} + 1"]]

    def _value_expr(self, atom: object, var: str) -> _ValueExpr:
        if isinstance(atom, ObjRef):
            return _ValueExpr(var=var, body=(f"size({atom.id},{var})",))
        if isinstance(atom, BagCountAtom):
            return _ValueExpr(
                var=var,
                body=(f"mult({atom.bag.id},{self._entity_id(atom.entity)},{var})",),
            )
        if isinstance(atom, TupleCountAtom):
            return self._tuple_count_value(atom, var)
        if isinstance(atom, SeqPatternCountAtom):
            return self._seq_pattern_count_value(atom, var)
        raise ASPEncodingError(f"Unsupported size atom for ASP: {type(atom).__name__}.")

    def _seq_pattern_count_value(
        self,
        atom: SeqPatternCountAtom,
        var: str,
    ) -> _ValueExpr:
        if not isinstance(atom.pattern, (PredecessorPattern, NextToPattern)):
            raise ASPEncodingError(
                f"ASP backend cannot count {type(atom.pattern).__name__} patterns."
            )
        match = self._emit_local_match_pred(atom.seq, atom.pattern)
        aux = f"seq_count_{self._new_aux_id()}"
        self.lines.append(
            f"{aux}({var}) :- num({var}), "
            f"{var} = #count {{ P,Q : {match}(P,Q) }}."
        )
        return _ValueExpr(var=var, body=(f"{aux}({var})",))

    def _tuple_count_value(self, atom: TupleCountAtom, var: str) -> _ValueExpr:
        if isinstance(atom.count_obj, Entity):
            return _ValueExpr(
                var=var,
                body=(f"mult({atom.tuple_ref.id},{self._entity_id(atom.count_obj)},{var})",),
            )
        aux = f"tuple_count_{self._new_aux_id()}"
        if atom.deduplicate:
            self.lines.append(
                f"{aux}({var}) :- num({var}), {var} = #count {{ E : "
                f"mult({atom.tuple_ref.id},E,N), N > 0, "
                f"mult({atom.count_obj.id},E,CN), CN > 0 }}."
            )
        else:
            self.lines.append(
                f"{aux}({var}) :- num({var}), {var} = #sum {{ N,E : "
                f"mult({atom.tuple_ref.id},E,N), "
                f"mult({atom.count_obj.id},E,CN), CN > 0 }}."
            )
        return _ValueExpr(var=var, body=(f"{aux}({var})",))

    def _arg_at_body(
        self,
        seq: ObjRef,
        pos_var: str,
        arg: ObjRef | Entity,
        entity_var: str,
    ) -> list[str]:
        if isinstance(arg, Entity):
            return [f"at({seq.id},{pos_var},{self._entity_id(arg)})"]
        return [
            f"at({seq.id},{pos_var},{entity_var})",
            f"mult({arg.id},{entity_var},N_{entity_var})",
            f"N_{entity_var} > 0",
        ]

    def _arg_entity_bodies(self, arg: ObjRef | Entity, entity_var: str) -> Iterable[list[str]]:
        if isinstance(arg, Entity):
            yield [f"{entity_var} = {self._entity_id(arg)}", f"entity({entity_var})"]
            return
        yield [f"entity({entity_var})", f"mult({arg.id},{entity_var},N)", "N > 0"]

    @staticmethod
    def _same_pattern_arg(left: ObjRef | Entity, right: ObjRef | Entity) -> bool:
        return left == right

    def _constrain_violation(self, pred: str, *, positive: bool) -> None:
        if positive:
            self.lines.append(f":- {pred}.")
        else:
            self.lines.append(f":- not {pred}.")

    def _violation_predicate(self) -> str:
        return f"viol_{self._new_aux_id()}"

    def _part_refs(self, partition: ObjRef) -> list[ObjRef]:
        parts: list[tuple[int, ObjRef]] = []
        for ref, defn in self.problem.defs:
            if isinstance(defn, (SetPartDef, BagPartDef)) and defn.partition == partition:
                parts.append((defn.index, ref))
        return [ref for _index, ref in sorted(parts)]

    def _new_aux_id(self) -> int:
        value = self.aux_counter
        self.aux_counter += 1
        return value

    def _exact_size(self, ref: ObjRef) -> int | None:
        info = self._info(ref)
        return info.exact_size if info is not None else None

    def _max_mult(self, ref: ObjRef, entity: Entity) -> int:
        info = self._info(ref)
        if isinstance(info, SetInfo):
            return 1 if entity in info.p_entities else 0
        if isinstance(info, BagInfo):
            return info.p_entities_multiplicity.get(entity, 0)
        return 0

    def _info(self, ref: ObjRef) -> SetInfo | BagInfo | None:
        return self.analysis.set_info.get(ref) or self.analysis.bag_info.get(ref)

    def _entity_id(self, entity: Entity) -> int:
        try:
            return self.entity_ids[entity]
        except KeyError as exc:
            raise ASPEncodingError(f"Unknown entity {entity.name!r}.") from exc

    @staticmethod
    def _term_expr(var: str, coefficient: int) -> str:
        if coefficient == 1:
            return var
        if coefficient == -1:
            return f"-{var}"
        return f"{coefficient}*{var}"

    @staticmethod
    def _invalid_comparison(expr: str, comparator: str, rhs: int) -> str:
        invalid = {
            "==": "!=",
            "!=": "==",
            "<": ">=",
            "<=": ">",
            ">": "<=",
            ">=": "<",
        }
        try:
            op = invalid[comparator]
        except KeyError as exc:
            raise ASPEncodingError(f"Unsupported comparator {comparator!r}.") from exc
        return f"{expr} {op} {rhs}"

    @staticmethod
    def _quote(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')
