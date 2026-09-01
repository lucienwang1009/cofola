"""Translate a lowered Cofola component to a single CoSo/CoLa program."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from cofola.frontend.constraints import (
    BagCountAtom,
    BagEqConstraint,
    BagSubsetConstraint,
    Constraint,
    DisjointConstraint,
    EqualityConstraint,
    FuncPairConstraint,
    MembershipConstraint,
    SeqPatternCountAtom,
    SequencePatternConstraint,
    SizeConstraint,
    SubsetConstraint,
    TupleCountAtom,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import (
    BagDifference,
    BagIntersection,
    BagChoose,
    CompositionDef,
    Entity,
    ObjRef,
    BagPartDef,
    PartitionDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetIntersection,
    SetPartDef,
    TupleDef,
)
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult

__all__ = ["CoSoEncodingError", "CoSoProgram", "encode"]


class CoSoEncodingError(Exception):
    """Raised when a Cofola component is outside the CoSo backend fragment."""


@dataclass(frozen=True)
class CoSoProgram:
    """A generated CoLa program, or a trivial count for constant components."""

    cola: str
    target: ObjRef | None = None
    is_trivial: bool = False
    trivial_count: int = 1
    count_divisor: int = 1


@dataclass(frozen=True)
class _ConfigSpec:
    target: ObjRef
    kind: str
    source: ObjRef
    size: int | None


@dataclass
class _EncodingState:
    problem: Problem
    analysis: AnalysisResult
    target: ObjRef
    entity_labels: dict[Entity, str]
    property_names: dict[ObjRef, str] = field(default_factory=dict)
    property_lines: list[str] = field(default_factory=list)

    def ref_name(self, ref: ObjRef) -> str:
        if ref == self.target:
            return "cfg"
        if ref not in self.property_names:
            self.property_names[ref] = f"p_{ref.id}"
        return self.property_names[ref]


_CONFIG_TYPES = (
    SetChoose,
    SetChooseReplace,
    BagChoose,
    TupleDef,
    SequenceDef,
    PartitionDef,
    CompositionDef,
)
_RESERVED = {"in", "property", "universe", "part", "repeated", "labelled"}


def encode(problem: Problem, analysis: AnalysisResult) -> CoSoProgram:
    """Encode a backend-ready Cofola component into CoLa."""

    spec = _find_config(problem, analysis)
    if spec is None:
        if problem.constraints:
            raise CoSoEncodingError("CoSo backend cannot solve constraint-only components.")
        return CoSoProgram(cola="", is_trivial=True, trivial_count=1)

    entities = _source_multiplicities(spec.source, analysis)
    if not entities:
        count = 1 if spec.size in (None, 0) else 0
        return CoSoProgram(cola="", target=spec.target, is_trivial=True, trivial_count=count)

    entity_labels = _entity_labels(entities)
    state = _EncodingState(
        problem=problem,
        analysis=analysis,
        target=spec.target,
        entity_labels=entity_labels,
    )

    lines: list[str] = []
    universe_items = [
        entity_labels[entity]
        for entity, multiplicity in sorted(entities.items(), key=lambda item: item[0].name)
        for _ in range(multiplicity)
    ]
    lines.append(f"universe u={{{','.join(universe_items)}}};")
    lines.append(f"cfg in {_config_expr(spec.kind, 'u')};")

    constraints = list(_size_lines(spec))
    constraint_lines, count_divisor = _constraint_lines(problem.constraints, state, spec)
    constraints.extend(constraint_lines)

    lines.extend(state.property_lines)
    lines.extend(constraints)
    return CoSoProgram(
        cola="\n".join(lines) + "\n",
        target=spec.target,
        count_divisor=count_divisor,
    )


def _find_config(problem: Problem, analysis: AnalysisResult) -> _ConfigSpec | None:
    candidates = [
        ref
        for ref, defn in problem.defs
        if isinstance(defn, _CONFIG_TYPES)
    ]
    if not candidates:
        return None

    referenced_by_config: set[ObjRef] = set()
    for ref in candidates:
        defn = problem.get_object(ref)
        if defn is None:
            continue
        for dep in problem.get_refs(defn):
            dep_defn = problem.get_object(dep)
            if isinstance(dep_defn, _CONFIG_TYPES):
                referenced_by_config.add(dep)

    roots = [ref for ref in candidates if ref not in referenced_by_config]
    if len(roots) != 1:
        names = ", ".join(_display_ref(problem, ref) for ref in roots or candidates)
        raise CoSoEncodingError(
            "CoSo backend supports exactly one target configuration per component; "
            f"found {len(roots) or len(candidates)} ({names})."
        )

    _reject_configuration_source_dependency(roots[0], problem)
    return _config_spec(roots[0], problem, analysis)


def _reject_configuration_source_dependency(ref: ObjRef, problem: Problem) -> None:
    defn = problem.get_object(ref)
    source = getattr(defn, "source", None)
    if not isinstance(source, ObjRef):
        return
    source_defn = problem.get_object(source)
    if isinstance(source_defn, _CONFIG_TYPES):
        raise CoSoEncodingError(
            "CoSo backend cannot encode a target that depends on another "
            f"configuration ({_display_ref(problem, ref)} depends on "
            f"{_display_ref(problem, source)}); translating only the target "
            "would drop the upstream count."
        )


def _config_spec(ref: ObjRef, problem: Problem, analysis: AnalysisResult) -> _ConfigSpec:
    defn = problem.get_object(ref)
    if isinstance(defn, SetChoose):
        return _ConfigSpec(ref, "subset", defn.source, _exact_size(ref, analysis, defn.size))
    if isinstance(defn, SetChooseReplace):
        return _ConfigSpec(ref, "multisubset", defn.source, _exact_size(ref, analysis, defn.size))
    if isinstance(defn, BagChoose):
        # CoSo represents bounded bag selection as an ordinary subset over a
        # universe that may contain repeated indistinguishable labels. Its
        # ``{repeated u}`` multisubset form is unbounded choose-with-replacement
        # and is not the semantics of Cofola's choose(Bag, k).
        return _ConfigSpec(ref, "subset", defn.source, _exact_size(ref, analysis, defn.size))
    if isinstance(defn, TupleDef):
        kind = "sequence" if defn.replace else "permutation"
        return _ConfigSpec(ref, kind, defn.source, _exact_size(ref, analysis, defn.size))
    if isinstance(defn, SequenceDef):
        raise CoSoEncodingError(
            "CoSo backend does not support Cofola sequence objects; "
            "use TupleDef/tuple(...) for CoSo-compatible ordered arrangements."
        )
    if isinstance(defn, PartitionDef):
        return _ConfigSpec(ref, "partition", defn.source, defn.num_parts)
    if isinstance(defn, CompositionDef):
        return _ConfigSpec(ref, "composition", defn.source, defn.num_parts)
    raise CoSoEncodingError(f"Unsupported CoSo target: {type(defn).__name__}.")


def _exact_size(ref: ObjRef, analysis: AnalysisResult, fallback: int | None = None) -> int | None:
    if fallback is not None:
        return fallback
    info = analysis.set_info.get(ref) or analysis.bag_info.get(ref)
    return info.exact_size if info is not None else None


def _source_multiplicities(ref: ObjRef, analysis: AnalysisResult) -> dict[Entity, int]:
    bag_info = analysis.bag_info.get(ref)
    if bag_info is not None:
        return dict(bag_info.p_entities_multiplicity)
    set_info = analysis.set_info.get(ref)
    if set_info is not None:
        return {entity: 1 for entity in set_info.p_entities}
    raise CoSoEncodingError(f"CoSo backend cannot determine source domain for ref {ref.id}.")


def _size_lines(spec: _ConfigSpec) -> list[str]:
    if spec.size is None:
        return []
    return [f"#cfg={spec.size};"]


def _constraint_lines(
    constraints: Sequence[Constraint],
    state: _EncodingState,
    spec: _ConfigSpec,
) -> tuple[list[str], int]:
    part_lines, skip = _part_constraint_lines(constraints, state, spec)
    lines: list[str] = []

    lines.extend(part_lines)

    for idx, c in enumerate(constraints):
        if idx in skip:
            continue
        line = _constraint_line(c, state, spec)
        if line is not None:
            lines.append(line)

    return lines, 1


def _part_constraint_lines(
    constraints: Sequence[Constraint],
    state: _EncodingState,
    spec: _ConfigSpec,
) -> tuple[list[str], set[int]]:
    if spec.kind not in {"composition", "partition"}:
        return [], set()

    grouped: dict[tuple[str, str, int], int] = {}
    lines: list[str] = []
    skip: set[int] = set()
    for idx, c in enumerate(constraints):
        part_line = _indexed_part_constraint_line(c, state, spec)
        if part_line is not None:
            lines.append(part_line)
            skip.add(idx)
            continue

        grouped_key = _grouped_part_constraint_key(c, state, spec)
        if grouped_key is not None:
            grouped[grouped_key] = grouped.get(grouped_key, 0) + 1
            skip.add(idx)

    lines.extend(
        f"#( #{expr}{comparator}{rhs} )={count};"
        for (expr, comparator, rhs), count in sorted(grouped.items())
    )
    return lines, skip


def _indexed_part_constraint_line(
    c: Constraint,
    state: _EncodingState,
    spec: _ConfigSpec,
) -> str | None:
    if spec.kind != "composition":
        return None
    if isinstance(c, SizeConstraint):
        try:
            atom = _single_unit_size_atom(c)
        except CoSoEncodingError:
            return None
        expr = _part_atom_expr(atom, state, indexed=True)
        if expr is not None:
            return f"#{expr}{_comparator(c.comparator)}{c.rhs};"
    if isinstance(c, MembershipConstraint):
        expr = _part_ref_expr(c.container, state, indexed=True)
        if expr is not None:
            op = ">" if c.positive else "="
            return f"#{expr}&{_entity_term(c.entity, state)}{op}0;"
    return None


def _grouped_part_constraint_key(
    c: Constraint,
    state: _EncodingState,
    spec: _ConfigSpec,
) -> tuple[str, str, int] | None:
    if spec.kind != "partition":
        return None
    if isinstance(c, SizeConstraint):
        try:
            atom = _single_unit_size_atom(c)
        except CoSoEncodingError:
            return None
        expr = _part_atom_expr(atom, state, indexed=False)
        if expr is not None:
            return expr, _comparator(c.comparator), c.rhs
    if isinstance(c, MembershipConstraint):
        expr = _part_ref_expr(c.container, state, indexed=False)
        if expr is not None:
            comparator = ">" if c.positive else "="
            return f"{expr}&{_entity_term(c.entity, state)}", comparator, 0
    return None


def _part_atom_expr(atom: object, state: _EncodingState, *, indexed: bool) -> str | None:
    if isinstance(atom, ObjRef):
        part_expr = _part_ref_expr(atom, state, indexed=indexed)
        if part_expr is not None:
            return part_expr
        defn = state.problem.get_object(atom)
        if isinstance(defn, (SetIntersection, BagIntersection)):
            left_part = _part_ref_expr(defn.left, state, indexed=indexed)
            if left_part is not None:
                return f"{left_part}&{_property(defn.right, state)}"
            right_part = _part_ref_expr(defn.right, state, indexed=indexed)
            if right_part is not None:
                return f"{right_part}&{_property(defn.left, state)}"
    return None


def _part_ref_expr(ref: ObjRef, state: _EncodingState, *, indexed: bool) -> str | None:
    defn = state.problem.get_object(ref)
    if not isinstance(defn, (SetPartDef, BagPartDef)) or defn.partition != state.target:
        return None
    if indexed:
        return f"cfg[{defn.index + 1}]"
    return "part"


def _constraint_line(c: Constraint, state: _EncodingState, spec: _ConfigSpec) -> str | None:
    if isinstance(c, SizeConstraint):
        return _size_constraint_line(c, state, spec)
    if isinstance(c, MembershipConstraint):
        return _membership_line(c, state, spec)
    if isinstance(c, SubsetConstraint):
        return _subset_line(c, state)
    if isinstance(c, DisjointConstraint):
        return _disjoint_line(c, state)
    if isinstance(c, EqualityConstraint):
        return _equality_line(c, state)
    if isinstance(c, (TupleIndexEq, TupleIndexMembership)):
        return _tuple_index_line(c, state)
    if isinstance(c, SequencePatternConstraint):
        raise CoSoEncodingError("CoSo backend does not support sequence pattern constraints.")
    if isinstance(c, (FuncPairConstraint, BagSubsetConstraint, BagEqConstraint)):
        raise CoSoEncodingError(f"CoSo backend does not support {type(c).__name__}.")
    raise CoSoEncodingError(f"Unknown constraint type for CoSo backend: {type(c).__name__}.")


def _size_constraint_line(c: SizeConstraint, state: _EncodingState, spec: _ConfigSpec) -> str:
    try:
        atom = _single_unit_size_atom(c)
    except CoSoEncodingError:
        line = _linear_partition_count_line(c, state, spec)
        if line is not None:
            return line
        raise

    comparator = _comparator(c.comparator)
    if isinstance(atom, ObjRef):
        counted = _counted_ref_expr(atom, state, spec)
        return f"#{counted}{comparator}{c.rhs};"
    if isinstance(atom, BagCountAtom):
        if atom.bag != state.target:
            raise CoSoEncodingError("CoSo backend only supports count atoms on the target.")
        return f"#cfg&{_entity_term(atom.entity, state)}{comparator}{c.rhs};"
    if isinstance(atom, TupleCountAtom):
        if atom.tuple_ref != state.target:
            raise CoSoEncodingError("CoSo backend only supports tuple count atoms on the target.")
        if atom.deduplicate:
            raise CoSoEncodingError("CoSo backend does not support tuple dedup_count atoms.")
        return f"#cfg&{_count_term(atom.count_obj, state)}{comparator}{c.rhs};"
    if isinstance(atom, SeqPatternCountAtom):
        raise CoSoEncodingError("CoSo backend does not support sequence pattern count atoms.")
    raise CoSoEncodingError(f"Unsupported CoSo size atom: {type(atom).__name__}.")


def _single_unit_size_atom(c: SizeConstraint) -> object:
    if len(c.terms) != 1:
        raise CoSoEncodingError("CoSo backend only supports single-term size constraints.")
    atom, coeff = c.terms[0]
    if coeff != 1:
        raise CoSoEncodingError("CoSo backend only supports unit-coefficient size constraints.")
    return atom


def _linear_partition_count_line(
    c: SizeConstraint,
    state: _EncodingState,
    spec: _ConfigSpec,
) -> str | None:
    if spec.size is None or len(c.terms) != 2:
        return None
    (left_atom, left_coef), (right_atom, right_coef) = c.terms
    left_prop = _target_intersection_property(left_atom, state)
    right_prop = _target_intersection_property(right_atom, state)
    if left_prop is None or right_prop is None:
        return None

    left_entities = _positive_entity_set(left_prop, state)
    right_entities = _positive_entity_set(right_prop, state)
    target_entities = _positive_entity_set(spec.source, state)
    if left_entities & right_entities or left_entities | right_entities != target_entities:
        return None

    coefficient = left_coef - right_coef
    constant = right_coef * spec.size
    valid = [
        value
        for value in range(spec.size + 1)
        if _compare_linear(coefficient * value + constant, c.comparator, c.rhs)
    ]
    if not valid:
        return f"#cfg&{_property(left_prop, state)}<0;"
    if valid == list(range(spec.size + 1)):
        return None
    lower = min(valid)
    upper = max(valid)
    if valid != list(range(lower, upper + 1)):
        return None

    counted = f"cfg&{_property(left_prop, state)}"
    if lower == upper:
        return f"#{counted}={lower};"
    if lower == 0:
        return f"#{counted}<={upper};"
    if upper == spec.size:
        return f"#{counted}>={lower};"
    return f"#{counted}>={lower};\n#{counted}<={upper};"


def _target_intersection_property(atom: object, state: _EncodingState) -> ObjRef | None:
    if not isinstance(atom, ObjRef):
        return None
    defn = state.problem.get_object(atom)
    if isinstance(defn, (SetIntersection, BagIntersection)):
        return _binary_ref_other_side(defn.left, defn.right, state.target)
    return None


def _positive_entity_set(ref: ObjRef, state: _EncodingState) -> set[Entity]:
    return {
        entity
        for entity, multiplicity in _source_multiplicities(ref, state.analysis).items()
        if multiplicity > 0
    }


def _counted_ref_expr(ref: ObjRef, state: _EncodingState, spec: _ConfigSpec) -> str:
    if ref == state.target:
        return "cfg"
    part_expr = _part_ref_expr(ref, state, indexed=spec.kind == "composition")
    if part_expr is not None:
        return part_expr
    defn = state.problem.get_object(ref)
    if isinstance(defn, SetIntersection):
        other = _binary_ref_other_side(defn.left, defn.right, state.target)
        if other is not None:
            return f"cfg&{_property(other, state)}"
        part_other = _binary_ref_part_other_side(defn.left, defn.right, state, spec)
        if part_other is not None:
            part_expr, other_ref = part_other
            return f"{part_expr}&{_property(other_ref, state)}"
    if isinstance(defn, BagIntersection):
        other = _binary_ref_other_side(defn.left, defn.right, state.target)
        if other is not None:
            return f"cfg&{_property(other, state)}"
        part_other = _binary_ref_part_other_side(defn.left, defn.right, state, spec)
        if part_other is not None:
            part_expr, other_ref = part_other
            return f"{part_expr}&{_property(other_ref, state)}"
    if isinstance(defn, SetDifference) and defn.left == state.target:
        return f"cfg&{_complement_property(defn.right, state)}"
    if isinstance(defn, BagDifference) and defn.left == state.target:
        return f"cfg&{_complement_property(defn.right, state)}"
    raise CoSoEncodingError(
        "CoSo backend supports size constraints on the target, "
        "target/property intersections, target differences, and target parts."
    )


def _binary_ref_other_side(left: ObjRef, right: ObjRef, target: ObjRef) -> ObjRef | None:
    if left == target and right != target:
        return right
    if right == target and left != target:
        return left
    return None


def _binary_ref_part_other_side(
    left: ObjRef,
    right: ObjRef,
    state: _EncodingState,
    spec: _ConfigSpec,
) -> tuple[str, ObjRef] | None:
    left_part = _part_ref_expr(left, state, indexed=spec.kind == "composition")
    if left_part is not None:
        return left_part, right
    right_part = _part_ref_expr(right, state, indexed=spec.kind == "composition")
    if right_part is not None:
        return right_part, left
    return None


def _membership_line(c: MembershipConstraint, state: _EncodingState, spec: _ConfigSpec) -> str | None:
    part_expr = _part_ref_expr(c.container, state, indexed=spec.kind == "composition")
    if part_expr is not None:
        op = ">" if c.positive else "="
        return f"#{part_expr}&{_entity_term(c.entity, state)}{op}0;"
    if c.container != state.target:
        return _constant_membership_line(c, state)
    op = ">" if c.positive else "="
    rhs = 0
    return f"#cfg&{_entity_term(c.entity, state)}{op}{rhs};"


def _subset_line(c: SubsetConstraint, state: _EncodingState) -> str:
    if c.sub == state.target and c.sup != state.target:
        complement = _complement_property(c.sup, state)
        op = "=" if c.positive else ">"
        return f"#cfg&{complement}{op}0;"
    if c.sup == state.target and c.sub != state.target:
        size = _ref_cardinality(c.sub, state)
        op = "=" if c.positive else "<"
        return f"#cfg&{_property(c.sub, state)}{op}{size};"
    raise CoSoEncodingError("CoSo backend supports subset constraints only between target and a constant set.")


def _disjoint_line(c: DisjointConstraint, state: _EncodingState) -> str:
    if c.left == state.target and c.right != state.target:
        prop = _property(c.right, state)
    elif c.right == state.target and c.left != state.target:
        prop = _property(c.left, state)
    else:
        raise CoSoEncodingError("CoSo backend supports disjoint constraints only with the target.")
    op = "=" if c.positive else ">"
    return f"#cfg&{prop}{op}0;"


def _equality_line(c: EqualityConstraint, state: _EncodingState) -> str:
    if not c.positive:
        raise CoSoEncodingError("CoSo backend does not support negative equality constraints.")
    other = c.right if c.left == state.target else c.left if c.right == state.target else None
    if other is None:
        raise CoSoEncodingError("CoSo backend supports equality constraints only with the target.")
    size = _ref_cardinality(other, state)
    prop = _property(other, state)
    return f"#cfg={size};\n#cfg&{prop}={size};"


def _tuple_index_line(c: TupleIndexEq | TupleIndexMembership, state: _EncodingState) -> str:
    tuple_ref = c.tuple_ref
    if tuple_ref != state.target:
        raise CoSoEncodingError("CoSo backend only supports positional constraints on the target.")
    pos = c.index + 1
    if isinstance(c, TupleIndexEq):
        term = (
            _entity_term(c.entity, state)
            if c.positive
            else _entity_complement_property(c.entity, state)
        )
    else:
        term = _property(c.container, state) if c.positive else _complement_property(c.container, state)
    return f"cfg[{pos}]={term};"


def _constant_membership_line(c: MembershipConstraint, state: _EncodingState) -> str | None:
    multiplicities = _source_multiplicities(c.container, state.analysis)
    present = c.entity in multiplicities and multiplicities[c.entity] > 0
    if present == c.positive:
        return None
    raise CoSoEncodingError("Constant membership constraint is unsatisfiable before CoSo encoding.")


def _property(ref: ObjRef, state: _EncodingState) -> str:
    name = state.ref_name(ref)
    if name == "cfg":
        return name
    if any(line.startswith(f"property {name}=") for line in state.property_lines):
        return name
    multiplicities = _source_multiplicities(ref, state.analysis)
    labels = [
        state.entity_labels[entity]
        for entity in sorted(multiplicities, key=lambda item: item.name)
        if multiplicities[entity] > 0 and entity in state.entity_labels
    ]
    state.property_lines.append(f"property {name}={{{','.join(labels)}}};")
    return name


def _complement_property(ref: ObjRef, state: _EncodingState) -> str:
    base_name = state.ref_name(ref)
    name = f"{base_name}_comp"
    if any(line.startswith(f"property {name}=") for line in state.property_lines):
        return name
    ref_entities = {
        entity
        for entity, multiplicity in _source_multiplicities(ref, state.analysis).items()
        if multiplicity > 0
    }
    labels = [
        label
        for entity, label in sorted(state.entity_labels.items(), key=lambda item: item[0].name)
        if entity not in ref_entities
    ]
    state.property_lines.append(f"property {name}={{{','.join(labels)}}};")
    return name


def _entity_complement_property(entity: Entity, state: _EncodingState) -> str:
    label = _entity_term(entity, state)
    name = f"{label}_comp"
    if any(line.startswith(f"property {name}=") for line in state.property_lines):
        return name
    labels = [
        other_label
        for other_entity, other_label in sorted(state.entity_labels.items(), key=lambda item: item[0].name)
        if other_entity != entity
    ]
    state.property_lines.append(f"property {name}={{{','.join(labels)}}};")
    return name


def _ref_cardinality(ref: ObjRef, state: _EncodingState) -> int:
    info = state.analysis.set_info.get(ref)
    if info is not None:
        if info.exact_size is not None:
            return info.exact_size
        return len(info.p_entities)
    bag_info = state.analysis.bag_info.get(ref)
    if bag_info is not None:
        if bag_info.exact_size is not None:
            return bag_info.exact_size
        return sum(bag_info.p_entities_multiplicity.values())
    raise CoSoEncodingError(f"CoSo backend cannot determine cardinality for ref {ref.id}.")


def _config_expr(kind: str, domain: str) -> str:
    match kind:
        case "permutation":
            return f"[{domain}]"
        case "sequence":
            return f"[repeated {domain}]"
        case "subset":
            return f"{{{domain}}}"
        case "multisubset":
            return f"{{repeated {domain}}}"
        case "partition":
            return f"{{{{{domain}}}}}"
        case "composition":
            return f"[{{{domain}}}]"
        case _:
            raise CoSoEncodingError(f"Unknown CoSo configuration kind: {kind}.")


def _entity_labels(multiplicities: dict[Entity, int]) -> dict[Entity, str]:
    used: set[str] = set()
    labels: dict[Entity, str] = {}
    for entity in sorted(multiplicities, key=lambda item: item.name):
        base = _sanitize_label(entity.name)
        label = base
        i = 2
        while label in used:
            label = f"{base}_{i}"
            i += 1
        used.add(label)
        labels[entity] = label
    return labels


def _sanitize_label(label: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    if not clean or not clean[0].islower():
        clean = f"e_{clean}"
    if clean in _RESERVED:
        clean = f"{clean}_entity"
    return clean


def _entity_term(entity: Entity, state: _EncodingState) -> str:
    label = state.entity_labels.get(entity)
    if label is None:
        raise CoSoEncodingError(f"Entity {entity.name!r} is outside the CoSo target universe.")
    return label


def _count_term(term: ObjRef | Entity, state: _EncodingState) -> str:
    if isinstance(term, Entity):
        return _entity_term(term, state)
    return _property(term, state)


def _comparator(comparator: str) -> str:
    if comparator == "==":
        return "="
    return comparator


def _compare_linear(lhs: int, comparator: str, rhs: int) -> bool:
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[comparator]


def _display_ref(problem: Problem, ref: ObjRef) -> str:
    return problem.get_name(ref) or f"#{ref.id}"
