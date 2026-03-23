"""Pretty-printing for cofola frontend Problem objects.

Provides human-readable formatting of Problem state for debug logging.
Each object type is rendered in concise mathematical notation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cofola.frontend.constraints import (
    AndConstraint,
    BagCountAtom,
    BagEqConstraint,
    BagSubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    ForAllParts,
    FuncPairConstraint,
    LessThanPattern,
    MembershipConstraint,
    NextToPattern,
    NotConstraint,
    OrConstraint,
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
    BagSupport,
    BagUnion,
    FuncDef,
    FuncImage,
    FuncInverse,
    FuncInverseImage,
    PartitionDef,
    PartRef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetUnion,
    TupleDef,
)
from cofola.frontend.types import Entity, ObjRef

if TYPE_CHECKING:
    from cofola.frontend.problem import Problem
    from cofola.ir.analysis.entities import AnalysisResult


_LINE_WIDTH = 64
_HR = "─" * _LINE_WIDTH


def _name(ref: ObjRef, problem: Problem) -> str:
    """Return display name for an ObjRef (name or '#N' fallback)."""
    n = problem.get_name(ref)
    return n if n else f"#{ref.id}"


def _ref_or_entity(x: ObjRef | Entity, problem: Problem) -> str:
    if isinstance(x, Entity):
        return x.name
    return _name(x, problem)


def _fmt_pattern(pattern: object, problem: Problem) -> str:
    match pattern:
        case TogetherPattern(group=g):
            return f"together({_name(g, problem)})"
        case LessThanPattern(left=l, right=r):
            return f"{_ref_or_entity(l, problem)} < {_ref_or_entity(r, problem)}"
        case PredecessorPattern(first=f, second=s):
            return f"{_ref_or_entity(f, problem)} ≺ {_ref_or_entity(s, problem)}"
        case NextToPattern(first=f, second=s):
            return f"{_ref_or_entity(f, problem)} ~ {_ref_or_entity(s, problem)}"
        case _:
            return repr(pattern)


def _fmt_size_atom(atom: object, problem: Problem) -> str:
    match atom:
        case ObjRef():
            return f"|{_name(atom, problem)}|"
        case TupleCountAtom(tuple_ref=tr, count_obj=co, deduplicate=dedup):
            dedup_s = " (dedup)" if dedup else ""
            return f"{_name(tr, problem)}.count({_ref_or_entity(co, problem)}){dedup_s}"
        case SeqPatternCountAtom(seq=s, pattern=p):
            return f"{_name(s, problem)}.count({_fmt_pattern(p, problem)})"
        case BagCountAtom(bag=b, entity=e):
            return f"{_name(b, problem)}.count({e.name})"
        case _:
            return repr(atom)


def _fmt_defn(defn: object, problem: Problem) -> str:
    """Render an object definition as a concise mathematical expression."""
    def n(ref: ObjRef) -> str:
        return _name(ref, problem)

    match defn:
        case SetInit(entities=ents):
            items = ", ".join(sorted(e.name for e in ents))
            return f"Set{{{items}}}"
        case SetChoose(source=src, size=sz):
            sz_s = f"{sz} " if sz is not None else ""
            return f"choose {sz_s}from {n(src)}"
        case SetChooseReplace(source=src, size=sz):
            sz_s = f"{sz} " if sz is not None else ""
            return f"choose {sz_s}from {n(src)} (replace)"
        case SetUnion(left=l, right=r):
            return f"{n(l)} ∪ {n(r)}"
        case SetIntersection(left=l, right=r):
            return f"{n(l)} ∩ {n(r)}"
        case SetDifference(left=l, right=r):
            return f"{n(l)} − {n(r)}"
        case BagInit(entity_multiplicity=em):
            items = ", ".join(f"{e.name}:{m}" for e, m in em)
            return f"Bag{{{items}}}"
        case BagChoose(source=src, size=sz):
            sz_s = f"{sz} " if sz is not None else ""
            return f"choose {sz_s}from {n(src)} (bag)"
        case BagUnion(left=l, right=r):
            return f"{n(l)} ∪₊ {n(r)}"
        case BagAdditiveUnion(left=l, right=r):
            return f"{n(l)} ⊕ {n(r)}"
        case BagIntersection(left=l, right=r):
            return f"{n(l)} ∩₊ {n(r)}"
        case BagDifference(left=l, right=r):
            return f"{n(l)} −₊ {n(r)}"
        case BagSupport(source=src):
            return f"support({n(src)})"
        case FuncDef(domain=dom, codomain=cod, injective=inj, surjective=sur):
            quals = []
            if inj:
                quals.append("inj")
            if sur:
                quals.append("sur")
            q = f" [{', '.join(quals)}]" if quals else ""
            return f"Func[{n(dom)} → {n(cod)}]{q}"
        case FuncImage(func=f, argument=arg):
            arg_s = arg.name if isinstance(arg, Entity) else n(arg)
            return f"{n(f)}({arg_s})"
        case FuncInverseImage(func=f, argument=arg):
            arg_s = arg.name if isinstance(arg, Entity) else n(arg)
            return f"{n(f)}⁻¹({arg_s})"
        case FuncInverse(func=f):
            return f"{n(f)}⁻¹"
        case TupleDef(source=src, choose=ch, replace=rep, size=sz):
            if ch:
                sz_s = f"{sz}-" if sz is not None else ""
                rep_s = " (replace)" if rep else ""
                return f"choose {sz_s}tuple from {n(src)}{rep_s}"
            return f"tuple({n(src)})"
        case SequenceDef(source=src, choose=ch, replace=rep, size=sz,
                         circular=circ, reflection=refl):
            flags = []
            if circ:
                flags.append("circle")
            if refl:
                flags.append("reflect")
            if ch:
                sz_s = f"{sz} " if sz is not None else ""
                rep_s = " (replace)" if rep else ""
                base = f"choose {sz_s}seq from {n(src)}{rep_s}"
            else:
                base = f"seq({n(src)})"
            return f"{base} [{', '.join(flags)}]" if flags else base
        case PartitionDef(source=src, num_parts=np_, ordered=ord_):
            kind = "compose" if ord_ else "partition"
            return f"{kind}({n(src)}, into={np_})"
        case PartRef(partition=p, index=i):
            return f"{n(p)}[{i}]"
        case _:
            return repr(defn)


def _fmt_constraint(c: object, problem: Problem) -> str:
    """Render a constraint as a concise mathematical expression."""
    def n(ref: ObjRef) -> str:
        return _name(ref, problem)

    def roe(x: ObjRef | Entity) -> str:
        return _ref_or_entity(x, problem)

    match c:
        case SizeConstraint(terms=terms, comparator=cmp, rhs=rhs):
            def _term(atom: object, coeff: int) -> str:
                s = _fmt_size_atom(atom, problem)
                return s if coeff == 1 else f"{coeff}·{s}"
            expr = " + ".join(_term(a, coeff) for a, coeff in terms)
            return f"{expr} {cmp} {rhs}"
        case MembershipConstraint(entity=e, container=cont, positive=pos):
            sym = "∈" if pos else "∉"
            return f"{e.name} {sym} {n(cont)}"
        case SubsetConstraint(sub=sub, sup=sup, positive=pos):
            sym = "⊆" if pos else "⊄"
            return f"{n(sub)} {sym} {n(sup)}"
        case DisjointConstraint(left=l, right=r, positive=pos):
            sym = "= ∅" if pos else "≠ ∅"
            return f"{n(l)} ∩ {n(r)} {sym}"
        case EqualityConstraint(left=l, right=r, positive=pos):
            sym = "=" if pos else "≠"
            return f"{n(l)} {sym} {n(r)}"
        case TupleIndexEq(tuple_ref=tr, index=i, entity=e, positive=pos):
            sym = "=" if pos else "≠"
            return f"{n(tr)}[{i}] {sym} {e.name}"
        case TupleIndexMembership(tuple_ref=tr, index=i, container=cont, positive=pos):
            sym = "∈" if pos else "∉"
            return f"{n(tr)}[{i}] {sym} {n(cont)}"
        case SequencePatternConstraint(seq=s, pattern=p, positive=pos):
            neg = "¬" if not pos else ""
            return f"{neg}{_fmt_pattern(p, problem)} in {n(s)}"
        case FuncPairConstraint(func=f, arg_entity=arg, result=res, positive=pos):
            sym = "=" if pos else "≠"
            return f"{n(f)}({arg.name}) {sym} {roe(res)}"
        case BagSubsetConstraint(sub=sub, sup=sup, positive=pos):
            sym = "⊆₊" if pos else "⊄₊"
            return f"{n(sub)} {sym} {n(sup)}"
        case BagEqConstraint(left=l, right=r, positive=pos):
            sym = "=₊" if pos else "≠₊"
            return f"{n(l)} {sym} {n(r)}"
        case NotConstraint(sub=sub):
            return f"¬({_fmt_constraint(sub, problem)})"
        case AndConstraint(left=l, right=r):
            return f"({_fmt_constraint(l, problem)}) ∧ ({_fmt_constraint(r, problem)})"
        case OrConstraint(left=l, right=r):
            return f"({_fmt_constraint(l, problem)}) ∨ ({_fmt_constraint(r, problem)})"
        case ForAllParts(partition=p, constraint_template=tmpl):
            return f"∀ part ∈ {n(p)}: {_fmt_constraint(tmpl, problem)}"
        case _:
            return repr(c)


def fmt_problem(problem: Problem, stage: str | None = None) -> str:
    """Format a Problem as a beautiful, readable multi-line string.

    Suitable for passing directly to logger.debug() to show pipeline state.

    Args:
        problem: The Problem to format.
        stage: Optional stage label shown in the header line.

    Returns:
        A multi-line string with Unicode decorations.
    """
    lines: list[str] = []

    n_objs = len(problem.defs)
    n_cons = len(problem.constraints)

    # Header
    stage_s = f"  {stage}" if stage else "  Problem"
    stats_s = (
        f"  {n_objs} object{'s' if n_objs != 1 else ''}"
        f" · {n_cons} constraint{'s' if n_cons != 1 else ''}"
    )
    lines.append(_HR)
    lines.append(stage_s)
    lines.append(stats_s)
    lines.append(_HR)

    # Objects section
    lines.append("  Objects:")
    defs_list = list(problem.defs)
    if defs_list:
        name_col = [_name(ref, problem) for ref, _ in defs_list]
        pad = max(len(nm) for nm in name_col)
        for (ref, defn), nm in zip(defs_list, name_col):
            defn_s = _fmt_defn(defn, problem)
            lines.append(f"    {nm:<{pad}}  =  {defn_s}")
    else:
        lines.append("    (none)")

    # Constraints section
    lines.append("")
    lines.append("  Constraints:")
    if problem.constraints:
        for c in problem.constraints:
            lines.append(f"    {_fmt_constraint(c, problem)}")
    else:
        lines.append("    (none)")

    lines.append(_HR)
    return "\n".join(lines)


def fmt_analysis(
    analysis: AnalysisResult,
    problem: Problem,
    stage: str | None = None,
) -> str:
    """Format an AnalysisResult as a beautiful, readable multi-line string.

    Shows entities, singletons, per-object set/bag analysis, and (if present)
    the dis/indis classification from BagClassification.

    Args:
        analysis: The AnalysisResult to format.
        problem: The associated Problem (used for name lookup).
        stage: Optional stage label shown in the header line.

    Returns:
        A multi-line string with Unicode decorations.
    """
    lines: list[str] = []

    n_sets = len(analysis.set_info)
    n_bags = len(analysis.bag_info)
    n_ents = len(analysis.all_entities)
    n_sing = len(analysis.singletons)

    # Header
    stage_s = f"  {stage}" if stage else "  Analysis"
    stats_s = (
        f"  {n_sets} set{'s' if n_sets != 1 else ''}"
        f" · {n_bags} bag{'s' if n_bags != 1 else ''}"
        f" · {n_ents} entit{'ies' if n_ents != 1 else 'y'}"
        f" · {n_sing} singleton{'s' if n_sing != 1 else ''}"
    )
    lines.append(_HR)
    lines.append(stage_s)
    lines.append(stats_s)
    lines.append(_HR)

    # Global entity summary
    all_names = sorted(e.name for e in analysis.all_entities)
    sing_names = sorted(e.name for e in analysis.singletons)
    lines.append(f"  Entities:   {{{', '.join(all_names)}}}")
    lines.append(f"  Singletons: {{{', '.join(sing_names)}}}" if sing_names else "  Singletons: {}")

    # Set analysis
    if analysis.set_info:
        lines.append("")
        lines.append("  Set Analysis:")
        refs_sorted = sorted(analysis.set_info.keys())
        name_col = [_name(r, problem) for r in refs_sorted]
        pad = max(len(nm) for nm in name_col)
        for ref, nm in zip(refs_sorted, name_col):
            info = analysis.set_info[ref]
            ents = ", ".join(sorted(e.name for e in info.p_entities))
            lines.append(
                f"    {nm:<{pad}}  max={info.max_size:<4}  p={{{ents}}}"
            )

    # Bag analysis
    if analysis.bag_info:
        lines.append("")
        lines.append("  Bag Analysis:")
        refs_sorted = sorted(analysis.bag_info.keys())
        name_col = [_name(r, problem) for r in refs_sorted]
        pad = max(len(nm) for nm in name_col)
        for ref, nm in zip(refs_sorted, name_col):
            info = analysis.bag_info[ref]
            # p_entities_multiplicity
            pm = ", ".join(
                f"{e.name}:{m}"
                for e, m in sorted(info.p_entities_multiplicity.items(), key=lambda x: x[0].name)
            )
            row = f"    {nm:<{pad}}  max={info.max_size:<4}  p={{{pm}}}"
            # dis/indis (only present after BagClassification)
            if info.dis_entities or info.indis_entities:
                dis = "{" + ", ".join(sorted(e.name for e in info.dis_entities)) + "}"
                indis_parts = []
                for mult in sorted(info.indis_entities):
                    grp = "{" + ", ".join(sorted(e.name for e in info.indis_entities[mult])) + "}"
                    indis_parts.append(f"{mult}×{grp}")
                indis = "[" + ", ".join(indis_parts) + "]" if indis_parts else "[]"
                row += f"  dis={dis}  indis={indis}"
            lines.append(row)

    lines.append(_HR)
    return "\n".join(lines)
