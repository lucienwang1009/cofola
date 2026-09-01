"""Essence model construction for the direct Conjure backend."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, Literal

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

__all__ = ["EssenceEncoder", "EssenceEncodingError"]


class EssenceEncodingError(Exception):
    """Raised when a Cofola construct is outside the direct Essence prototype."""


_Kind = Literal["set", "mset", "sequence", "none"]


@dataclass(frozen=True)
class _ValueExpr(object):
    """Integer expression used in linear size constraints."""

    expr: str


class EssenceEncoder(object):
    """Translate one planning component to an Essence model."""

    def __init__(self, problem: Problem, analysis: AnalysisResult) -> None:
        self.problem = problem
        self.analysis = analysis
        self.entities = sorted(analysis.all_entities, key=lambda entity: entity.name)
        self.entity_ids = {entity: idx for idx, entity in enumerate(self.entities)}
        self.decls: list[str] = []
        self.constraints: list[str] = []
        self.kinds: dict[ObjRef, _Kind] = {}
        self.global_max = self._global_max()

    def encode(self) -> str:
        """Return an Essence model whose solutions are Cofola solutions."""

        if not self.problem.defs:
            return "find cofola_unit : bool\nsuch that\n  cofola_unit = true\n"

        self._emit_preamble()
        for ref, defn in self.problem.defs:
            self._emit_object(ref, defn)
        for constraint in self.problem.constraints:
            self._emit_constraint(constraint)
        return self._render()

    def _emit_preamble(self) -> None:
        upper = max(0, len(self.entities) - 1)
        self.decls.append(f"letting Entity be domain int(0..{upper})")
        for entity, idx in self.entity_ids.items():
            self.decls.append(f"$ entity {idx} = {entity.name}")

    def _render(self) -> str:
        lines = list(self.decls)
        if self.constraints:
            lines.append("such that")
            for index, constraint in enumerate(self.constraints):
                suffix = "," if index < len(self.constraints) - 1 else ""
                lines.append(f"  {constraint}{suffix}")
        return "\n".join(lines) + "\n"

    def _emit_object(self, ref: ObjRef, defn: object) -> None:
        if isinstance(defn, SetInit):
            self._declare_set(ref, exact=len(defn.entities))
            self._add(f"{self._name(ref)} = {self._set_literal(defn.entities)}")
        elif isinstance(defn, BagInit):
            multiplicities = dict(defn.entity_multiplicity)
            self._declare_mset(ref, exact=sum(multiplicities.values()))
            for entity in self.entities:
                self._add(
                    f"freq({self._name(ref)}, {self._entity_id(entity)}) = "
                    f"{multiplicities.get(entity, 0)}"
                )
        elif isinstance(defn, SetChoose):
            self._declare_set(ref)
            self._add(f"{self._name(ref)} subsetEq {self._name(defn.source)}")
        elif isinstance(defn, SetChooseReplace):
            self._declare_mset(ref)
            self._add(
                f"forAll e : Entity . "
                f"(freq({self._name(ref)}, e) > 0 -> e in {self._name(defn.source)})"
            )
        elif isinstance(defn, BagChoose):
            self._declare_mset(ref)
            self._for_all_entities(
                lambda e: (
                    f"freq({self._name(ref)}, {e}) <= "
                    f"freq({self._name(defn.source)}, {e})"
                )
            )
        elif isinstance(defn, SetUnion):
            self._declare_set(ref)
            self._add(f"{self._name(ref)} = {self._name(defn.left)} union {self._name(defn.right)}")
        elif isinstance(defn, SetIntersection):
            self._declare_set(ref)
            self._add(
                f"{self._name(ref)} = {self._name(defn.left)} intersect {self._name(defn.right)}"
            )
        elif isinstance(defn, SetDifference):
            self._declare_set(ref)
            self._add(f"{self._name(ref)} = {self._name(defn.left)} - {self._name(defn.right)}")
        elif isinstance(defn, BagUnion):
            self._emit_bag_binary(ref, defn.left, defn.right, "max")
        elif isinstance(defn, BagAdditiveUnion):
            self._emit_bag_binary(ref, defn.left, defn.right, "add")
        elif isinstance(defn, BagIntersection):
            self._emit_bag_binary(ref, defn.left, defn.right, "min")
        elif isinstance(defn, BagDifference):
            self._emit_bag_binary(ref, defn.left, defn.right, "sub")
        elif isinstance(defn, BagSupport):
            self._declare_set(ref)
            self._for_all_entities(
                lambda e: (
                    f"({e} in {self._name(ref)}) <-> "
                    f"(freq({self._name(defn.source)}, {e}) > 0)"
                )
            )
        elif isinstance(defn, (TupleDef, SequenceDef)):
            self._emit_ordered(ref, defn)
        elif isinstance(defn, (PartitionDef, CompositionDef)):
            self._emit_grouped(ref, defn)
        elif isinstance(defn, PartDef):
            # Part variables are declared by the owning partition/composition.
            self.kinds.setdefault(ref, self._collection_kind(ref))
        elif isinstance(defn, (FuncDef, FuncImage, FuncInverseImage, FuncInverse)):
            raise EssenceEncodingError(
                f"Essence backend does not yet support {type(defn).__name__}."
            )
        else:
            raise EssenceEncodingError(
                f"Unsupported object for Essence: {type(defn).__name__}."
            )

    def _emit_bag_binary(
        self,
        ref: ObjRef,
        left: ObjRef,
        right: ObjRef,
        op: str,
    ) -> None:
        self._declare_mset(ref)
        left_name = self._name(left)
        right_name = self._name(right)
        own_name = self._name(ref)
        if op == "add":
            rhs = lambda e: f"freq({left_name}, {e}) + freq({right_name}, {e})"
        elif op == "max":
            rhs = lambda e: f"max([freq({left_name}, {e}), freq({right_name}, {e})])"
        elif op == "min":
            rhs = lambda e: f"min([freq({left_name}, {e}), freq({right_name}, {e})])"
        elif op == "sub":
            rhs = lambda e: f"max([freq({left_name}, {e}) - freq({right_name}, {e}), 0])"
        else:
            raise AssertionError(op)
        self._for_all_entities(
            lambda e: f"freq({own_name}, {e}) = {rhs(e)}"
        )

    def _emit_ordered(
        self,
        ref: ObjRef,
        defn: TupleDef | SequenceDef,
    ) -> None:
        size = self._exact_size(ref)
        if size is None:
            raise EssenceEncodingError(
                f"{type(defn).__name__} ref={ref.id} needs an exact size before Essence encoding."
            )
        self._declare_sequence(ref, exact=size)
        own = self._name(ref)
        source = self._name(defn.source)

        if size > 0:
            for pos in range(1, size + 1):
                if self._kind(defn.source) == "set":
                    self._add(f"{own}({pos}) in {source}")
                elif self._kind(defn.source) == "mset":
                    self._add(f"freq({source}, {own}({pos})) > 0")
                else:
                    raise EssenceEncodingError(
                        f"{type(defn).__name__} source must be a set or mset."
                    )

        if not defn.replace:
            comparator = "=" if not defn.choose else "<="
            self._for_all_entities(
                lambda e: (
                    f"{self._mult(ref, e)} <= {self._mult(defn.source, e)}"
                    if comparator == "<="
                    else f"{self._mult(ref, e)} = {self._mult(defn.source, e)}"
                )
            )

    def _emit_grouped(
        self,
        ref: ObjRef,
        defn: PartitionDef | CompositionDef,
    ) -> None:
        self.kinds[ref] = "none"
        parts = self._part_refs(ref)
        if len(parts) != defn.num_parts:
            raise EssenceEncodingError(
                f"{type(defn).__name__} ref={ref.id} expected {defn.num_parts} "
                f"parts, found {len(parts)}."
            )
        for part in parts:
            kind = self._collection_kind(part)
            if kind == "set":
                self._declare_set(part)
            elif kind == "mset":
                self._declare_mset(part)
            else:
                raise EssenceEncodingError(
                    f"{type(self.problem.get_object(part)).__name__} must be set or mset."
                )

        self._for_all_entities(
            lambda e: (
                f"sum([{', '.join(self._mult(part, e) for part in parts)}]) = "
                f"{self._mult(defn.source, e)}"
            )
        )

        if isinstance(defn, PartitionDef):
            self._emit_partition_canonical(parts)

    def _emit_partition_canonical(self, parts: list[ObjRef]) -> None:
        for left, right in zip(parts, parts[1:], strict=False):
            self._add(self._lex_geq(self._part_lex_terms(left), self._part_lex_terms(right)))

    def _emit_constraint(self, constraint: object) -> None:
        if isinstance(constraint, SizeConstraint):
            self._emit_size_constraint(constraint)
        elif isinstance(constraint, MembershipConstraint):
            eid = self._entity_id_opt(constraint.entity)
            if eid is None:
                # Entity cannot occur here: `e in C` is false, `e not in C` true.
                self._add("false" if constraint.positive else "true")
            else:
                atom = self._contains(constraint.container, eid)
                self._add(atom if constraint.positive else f"!({atom})")
        elif isinstance(constraint, SubsetConstraint):
            self._emit_subset_constraint(constraint)
        elif isinstance(constraint, BagSubsetConstraint):
            self._emit_bag_subset_constraint(constraint)
        elif isinstance(constraint, DisjointConstraint):
            self._emit_disjoint_constraint(constraint)
        elif isinstance(constraint, EqualityConstraint):
            self._emit_equality_constraint(constraint)
        elif isinstance(constraint, BagEqConstraint):
            self._emit_bag_equality_constraint(constraint)
        elif isinstance(constraint, TupleIndexEq):
            self._emit_tuple_index_eq(constraint)
        elif isinstance(constraint, TupleIndexMembership):
            self._emit_tuple_index_membership(constraint)
        elif isinstance(constraint, SequencePatternConstraint):
            self._emit_sequence_pattern_constraint(constraint)
        elif isinstance(constraint, FuncPairConstraint):
            raise EssenceEncodingError(
                "Essence backend does not yet support function constraints."
            )
        else:
            raise EssenceEncodingError(
                f"Unsupported constraint for Essence: {type(constraint).__name__}."
            )

    def _emit_size_constraint(self, constraint: SizeConstraint) -> None:
        terms = [
            self._term_expr(self._value_expr(atom).expr, coefficient)
            for atom, coefficient in constraint.terms
            if coefficient != 0
        ]
        expr = " + ".join(terms) or "0"
        self._add(f"{expr} {self._essence_comparator(constraint.comparator)} {constraint.rhs}")

    def _emit_subset_constraint(self, constraint: SubsetConstraint) -> None:
        body = self._forall_entity_expr(
            lambda e: f"{self._contains(constraint.sub, e)} -> {self._contains(constraint.sup, e)}"
        )
        self._add(body if constraint.positive else f"!({body})")

    def _emit_bag_subset_constraint(self, constraint: BagSubsetConstraint) -> None:
        body = self._forall_entity_expr(
            lambda e: f"{self._mult(constraint.sub, e)} <= {self._mult(constraint.sup, e)}"
        )
        self._add(body if constraint.positive else f"!({body})")

    def _emit_disjoint_constraint(self, constraint: DisjointConstraint) -> None:
        body = self._forall_entity_expr(
            lambda e: f"!({self._contains(constraint.left, e)} /\\ {self._contains(constraint.right, e)})"
        )
        self._add(body if constraint.positive else f"!({body})")

    def _emit_equality_constraint(self, constraint: EqualityConstraint) -> None:
        body = self._forall_entity_expr(
            lambda e: f"{self._mult(constraint.left, e)} = {self._mult(constraint.right, e)}"
        )
        self._add(body if constraint.positive else f"!({body})")

    def _emit_bag_equality_constraint(self, constraint: BagEqConstraint) -> None:
        self._emit_equality_constraint(
            EqualityConstraint(
                left=constraint.left,
                right=constraint.right,
                positive=constraint.positive,
            )
        )

    def _emit_tuple_index_eq(self, constraint: TupleIndexEq) -> None:
        eid = self._entity_id_opt(constraint.entity)
        if eid is None:
            # The position can never hold an entity absent from this component.
            self._add("false" if constraint.positive else "true")
            return
        expr = f"{self._name(constraint.tuple_ref)}({constraint.index + 1}) = {eid}"
        self._add(expr if constraint.positive else f"!({expr})")

    def _emit_tuple_index_membership(self, constraint: TupleIndexMembership) -> None:
        value = f"{self._name(constraint.tuple_ref)}({constraint.index + 1})"
        expr = self._contains(constraint.container, value)
        self._add(expr if constraint.positive else f"!({expr})")

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
            raise EssenceEncodingError(
                f"Unsupported sequence pattern {type(pattern).__name__}."
            )

    def _emit_less_than_pattern(self, constraint: SequencePatternConstraint) -> None:
        if constraint.coverage is not None:
            raise EssenceEncodingError(
                "Essence backend does not support coverage on before patterns."
            )
        pattern = constraint.pattern
        if not isinstance(pattern, LessThanPattern):
            raise AssertionError(type(pattern).__name__)
        size = self._required_exact_size(constraint.seq)
        violations = [
            f"({self._arg_at(constraint.seq, left, pattern.left)} /\\ "
            f"{self._arg_at(constraint.seq, right, pattern.right)})"
            for left in range(1, size + 1)
            for right in range(1, size + 1)
            if left >= right
        ]
        violation = self._or(violations)
        self._add(f"!({violation})" if constraint.positive else violation)

    def _emit_together_pattern(self, constraint: SequencePatternConstraint) -> None:
        if constraint.coverage is not None:
            raise EssenceEncodingError(
                "Essence backend does not support coverage on together patterns."
            )
        pattern = constraint.pattern
        if not isinstance(pattern, TogetherPattern):
            raise AssertionError(type(pattern).__name__)
        size = self._required_exact_size(constraint.seq)
        group_at = lambda pos: self._arg_at(constraint.seq, pos, pattern.group)
        gaps = [
            f"({group_at(left)} /\\ {group_at(right)} /\\ !({group_at(mid)}))"
            for left in range(1, size + 1)
            for mid in range(left + 1, size + 1)
            for right in range(mid + 1, size + 1)
        ]
        gap = self._or(gaps)
        self._add(f"!({gap})" if constraint.positive else gap)

    def _emit_local_pattern(self, constraint: SequencePatternConstraint) -> None:
        pattern = constraint.pattern
        if not isinstance(pattern, (PredecessorPattern, NextToPattern)):
            raise AssertionError(type(pattern).__name__)
        matches = self._local_match_exprs(constraint.seq, pattern)
        has_match = self._or(matches)
        if constraint.coverage is None:
            self._add(has_match if constraint.positive else f"!({has_match})")
            return

        size = self._required_exact_size(constraint.seq)
        obligations = []
        for pos in range(1, size + 1):
            cov = self._arg_at(constraint.seq, pos, constraint.coverage)
            covered = self._covered_expr(constraint.seq, pattern, constraint.coverage, pos)
            obligations.append(f"({cov} -> {covered})")
        body = self._and(obligations)
        self._add(body if constraint.positive else f"!({body})")

    def _value_expr(self, atom: object) -> _ValueExpr:
        if isinstance(atom, ObjRef):
            return _ValueExpr(expr=f"|{self._name(atom)}|")
        if isinstance(atom, BagCountAtom):
            eid = self._entity_id_opt(atom.entity)
            if eid is None:
                return _ValueExpr(expr="0")
            return _ValueExpr(expr=self._mult(atom.bag, eid))
        if isinstance(atom, TupleCountAtom):
            return _ValueExpr(expr=self._tuple_count(atom))
        if isinstance(atom, SeqPatternCountAtom):
            return _ValueExpr(expr=self._seq_pattern_count(atom))
        raise EssenceEncodingError(
            f"Unsupported size atom for Essence: {type(atom).__name__}."
        )

    def _tuple_count(self, atom: TupleCountAtom) -> str:
        if isinstance(atom.count_obj, Entity):
            eid = self._entity_id_opt(atom.count_obj)
            if eid is None:
                return "0"
            return self._mult(atom.tuple_ref, eid)
        terms = []
        for entity in self.entities:
            eid = self._entity_id(entity)
            if atom.deduplicate:
                terms.append(
                    f"toInt({self._contains(atom.tuple_ref, eid)} /\\ "
                    f"{self._contains(atom.count_obj, eid)})"
                )
            else:
                terms.append(
                    f"{self._mult(atom.tuple_ref, eid)} * "
                    f"toInt({self._contains(atom.count_obj, eid)})"
                )
        return self._sum(terms)

    def _seq_pattern_count(self, atom: SeqPatternCountAtom) -> str:
        if not isinstance(atom.pattern, (PredecessorPattern, NextToPattern)):
            raise EssenceEncodingError(
                f"Essence backend cannot count {type(atom.pattern).__name__} patterns."
            )
        return self._sum(f"toInt({expr})" for expr in self._local_match_exprs(atom.seq, atom.pattern))

    def _local_match_exprs(
        self,
        seq: ObjRef,
        pattern: PredecessorPattern | NextToPattern,
    ) -> list[str]:
        pairs = self._successor_pairs(seq)
        if isinstance(pattern, PredecessorPattern):
            first = pattern.first
            second = pattern.second
            oriented = pairs
        elif isinstance(pattern, NextToPattern):
            first = pattern.first
            second = pattern.second
            oriented = pairs + [(right, left) for left, right in pairs]
        else:
            raise AssertionError(type(pattern).__name__)
        return [
            f"({self._arg_at(seq, left, first)} /\\ {self._arg_at(seq, right, second)})"
            for left, right in oriented
        ]

    def _covered_expr(
        self,
        seq: ObjRef,
        pattern: PredecessorPattern | NextToPattern,
        coverage: ObjRef | Entity,
        pos: int,
    ) -> str:
        covered = []
        for expr, left, right in self._local_match_exprs_with_pairs(seq, pattern):
            if self._same_pattern_arg(coverage, pattern.first) and left == pos:
                covered.append(expr)
            if self._same_pattern_arg(coverage, pattern.second) and right == pos:
                covered.append(expr)
            if (
                not self._same_pattern_arg(coverage, pattern.first)
                and not self._same_pattern_arg(coverage, pattern.second)
                and (left == pos or right == pos)
            ):
                covered.append(expr)
        return self._or(covered)

    def _local_match_exprs_with_pairs(
        self,
        seq: ObjRef,
        pattern: PredecessorPattern | NextToPattern,
    ) -> list[tuple[str, int, int]]:
        pairs = self._successor_pairs(seq)
        if isinstance(pattern, PredecessorPattern):
            first = pattern.first
            second = pattern.second
            oriented = pairs
        elif isinstance(pattern, NextToPattern):
            first = pattern.first
            second = pattern.second
            oriented = pairs + [(right, left) for left, right in pairs]
        else:
            raise AssertionError(type(pattern).__name__)
        return [
            (
                f"({self._arg_at(seq, left, first)} /\\ {self._arg_at(seq, right, second)})",
                left,
                right,
            )
            for left, right in oriented
        ]

    def _successor_pairs(self, seq: ObjRef) -> list[tuple[int, int]]:
        size = self._required_exact_size(seq)
        return [(pos, pos + 1) for pos in range(1, size)]

    def _arg_at(self, seq: ObjRef, pos: int, arg: ObjRef | Entity) -> str:
        value = f"{self._name(seq)}({pos})"
        if isinstance(arg, Entity):
            eid = self._entity_id_opt(arg)
            return "false" if eid is None else f"{value} = {eid}"
        return self._contains(arg, value)

    @staticmethod
    def _same_pattern_arg(left: ObjRef | Entity, right: ObjRef | Entity) -> bool:
        return left == right

    def _declare_set(self, ref: ObjRef, *, exact: int | None = None) -> None:
        info = self._set_info(ref)
        exact = info.exact_size if exact is None else exact
        if exact is not None:
            domain = f"set (size {exact}) of Entity"
        else:
            domain = f"set (maxSize {info.max_size}) of Entity"
        self.decls.append(f"find {self._name(ref)} : {domain}")
        self.kinds[ref] = "set"

    def _declare_mset(self, ref: ObjRef, *, exact: int | None = None) -> None:
        info = self._bag_info(ref)
        exact = info.exact_size if exact is None else exact
        max_occur = max(info.p_entities_multiplicity.values() or [0, info.max_size, self.global_max])
        if exact is not None:
            attrs = f"size {exact}, maxOccur {max_occur}"
        else:
            attrs = f"maxSize {info.max_size}, maxOccur {max_occur}"
        self.decls.append(f"find {self._name(ref)} : mset ({attrs}) of Entity")
        self.kinds[ref] = "mset"

    def _declare_sequence(self, ref: ObjRef, *, exact: int) -> None:
        self.decls.append(f"find {self._name(ref)} : sequence (size {exact}) of Entity")
        self.kinds[ref] = "sequence"

    def _kind(self, ref: ObjRef) -> _Kind:
        try:
            return self.kinds[ref]
        except KeyError as exc:
            raise EssenceEncodingError(f"No Essence variable kind for ref={ref.id}.") from exc

    def _collection_kind(self, ref: ObjRef) -> _Kind:
        if ref in self.analysis.set_info:
            return "set"
        if ref in self.analysis.bag_info:
            return "mset"
        return "none"

    def _contains(self, ref: ObjRef, entity_expr: int | str) -> str:
        name = self._name(ref)
        kind = self._kind(ref)
        if kind == "set":
            return f"{entity_expr} in {name}"
        if kind == "mset":
            return f"freq({name}, {entity_expr}) > 0"
        if kind == "sequence":
            return f"{self._mult(ref, entity_expr)} > 0"
        raise EssenceEncodingError(f"ref={ref.id} is not a container.")

    def _mult(self, ref: ObjRef, entity_expr: int | str) -> str:
        name = self._name(ref)
        kind = self._kind(ref)
        if kind == "set":
            return f"toInt({entity_expr} in {name})"
        if kind == "mset":
            return f"freq({name}, {entity_expr})"
        if kind == "sequence":
            size = self._required_exact_size(ref)
            if size == 0:
                return "0"
            return (
                f"sum([toInt({name}(i) = {entity_expr}) | "
                f"i : int(1..{size})])"
            )
        raise EssenceEncodingError(f"ref={ref.id} has no multiplicity view.")

    def _part_lex_terms(self, ref: ObjRef) -> list[str]:
        return [self._mult(ref, self._entity_id(entity)) for entity in self.entities]

    def _lex_geq(self, left: list[str], right: list[str]) -> str:
        return self._lex_leq(right, left)

    def _lex_leq(self, left: list[str], right: list[str]) -> str:
        if len(left) != len(right):
            raise EssenceEncodingError("Lexicographic operands must have the same length.")
        strict_terms = []
        for index, (left_term, right_term) in enumerate(zip(left, right, strict=True)):
            prefix_equal = [
                f"{left[prev]} = {right[prev]}"
                for prev in range(index)
            ]
            strict_terms.append(
                self._and([*prefix_equal, f"{left_term} < {right_term}"])
            )
        all_equal = self._and(
            f"{left_term} = {right_term}"
            for left_term, right_term in zip(left, right, strict=True)
        )
        return self._or([*strict_terms, all_equal])

    def _for_all_entities(self, body_factory) -> None:
        self._add(f"forAll e : Entity . {body_factory('e')}")

    def _forall_entity_expr(self, body_factory) -> str:
        return f"(forAll e : Entity . {body_factory('e')})"

    def _set_literal(self, entities: Iterable[Entity]) -> str:
        values = sorted(self._entity_id(entity) for entity in entities)
        return "{" + ", ".join(str(value) for value in values) + "}"

    def _part_refs(self, partition: ObjRef) -> list[ObjRef]:
        parts: list[tuple[int, ObjRef]] = []
        for ref, defn in self.problem.defs:
            if isinstance(defn, (SetPartDef, BagPartDef)) and defn.partition == partition:
                parts.append((defn.index, ref))
        return [ref for _index, ref in sorted(parts)]

    def _required_exact_size(self, ref: ObjRef) -> int:
        exact = self._exact_size(ref)
        if exact is None:
            raise EssenceEncodingError(f"ref={ref.id} needs an exact size.")
        return exact

    def _exact_size(self, ref: ObjRef) -> int | None:
        info = self._info(ref)
        return info.exact_size if info is not None else None

    def _info(self, ref: ObjRef) -> SetInfo | BagInfo | None:
        return self.analysis.set_info.get(ref) or self.analysis.bag_info.get(ref)

    def _set_info(self, ref: ObjRef) -> SetInfo:
        try:
            return self.analysis.set_info[ref]
        except KeyError as exc:
            raise EssenceEncodingError(f"ref={ref.id} is not analysed as a set.") from exc

    def _bag_info(self, ref: ObjRef) -> BagInfo:
        try:
            return self.analysis.bag_info[ref]
        except KeyError as exc:
            raise EssenceEncodingError(f"ref={ref.id} is not analysed as a bag.") from exc

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
            raise EssenceEncodingError(
                "Essence backend needs finite multiplicity bounds; add a size constraint."
            )
        return max_value

    def _entity_id(self, entity: Entity) -> int:
        try:
            return self.entity_ids[entity]
        except KeyError as exc:
            raise EssenceEncodingError(f"Unknown entity {entity.name!r}.") from exc

    def _entity_id_opt(self, entity: Entity) -> int | None:
        """Entity id, or ``None`` if the entity cannot occur in this component.

        Constant folding can drop an entity from a component's domain while a
        constraint still references it (e.g. ``a in (S & T)`` once ``S & T``
        folds to a constant not containing ``a``). Such an entity is provably
        absent, so membership is ``false`` and any count/multiplicity is ``0``.
        """

        return self.entity_ids.get(entity)

    def _name(self, ref: ObjRef) -> str:
        return f"o_{ref.id}"

    def _add(self, constraint: str) -> None:
        self.constraints.append(constraint)

    @staticmethod
    def _sum(terms: Iterable[str]) -> str:
        materialized = [term for term in terms if term != "0"]
        if not materialized:
            return "0"
        return f"sum([{', '.join(materialized)}])"

    @staticmethod
    def _and(terms: Iterable[str]) -> str:
        materialized = list(terms)
        if not materialized:
            return "true"
        if len(materialized) == 1:
            return materialized[0]
        return f"and([{', '.join(materialized)}])"

    @staticmethod
    def _or(terms: Iterable[str]) -> str:
        materialized = list(terms)
        if not materialized:
            return "false"
        if len(materialized) == 1:
            return materialized[0]
        return f"or([{', '.join(materialized)}])"

    @staticmethod
    def _term_expr(expr: str, coefficient: int) -> str:
        if coefficient == 1:
            return expr
        if coefficient == -1:
            return f"-({expr})"
        return f"{coefficient}*({expr})"

    @staticmethod
    def _essence_comparator(comparator: str) -> str:
        if comparator == "==":
            return "="
        return comparator
