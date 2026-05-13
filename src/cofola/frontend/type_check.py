"""TypeChecker — validates operator/constraint type signatures on the frontend Problem.

Walks the `Problem` once, looks up each frontend node's signature in
`SIGNATURES`, and validates positional parameter types against the spec
type lattice. Collects all errors (does not fail-fast) and finally raises
a single `CofolaTypeError` that lists every issue with its source location.

Spec source: `~/Sync/overleaf/comb_aij/{basic,grouped,ordered}_objects.tex`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from cofola.frontend.constraints import (
    AndConstraint,
    BagCountAtom,
    Constraint,
    ForAllParts,
    NotConstraint,
    OrConstraint,
    SeqPatternCountAtom,
    SequencePatternConstraint,
    SizeConstraint,
    TupleCountAtom,
)
from cofola.frontend.objects import (
    BagObjDef,
    BagPartDef,
    CircleDef,
    CompositionDef,
    FuncObjDef,
    Grouped,
    Linear,
    Ordered,
    PartitionDef,
    PartPlaceholderDef,
    SequenceDef,
    SetLike,
    SetObjDef,
    SetPartDef,
    TupleDef,
)
from cofola.frontend.problem import Problem, SourceLoc
from cofola.frontend.objects import Entity, ObjRef
from cofola.frontend.utils import constraint_refs
from cofola.frontend.op_signatures import (
    PATTERN_FIELD_EXPECT,
    SIGNATURES,
    Signature,
)


@dataclass(frozen=True)
class TypeCheckError:
    """One type-check error.

    Attributes:
        loc: Source (line, col) of the offending node, or None.
        message: Human-readable description.
        node: The offending frontend node (for debugging / programmatic access).
    """

    loc: Optional[SourceLoc]
    message: str
    node: object


class CofolaTypeError(Exception):
    """Raised when one or more type errors are found in a Cofola frontend problem.

    Carries the full list of `TypeCheckError` records so callers can
    inspect them individually if needed; the formatted str() is multi-line
    and includes the source location of each error when available.
    """

    def __init__(self, errors: Iterable[TypeCheckError]) -> None:
        self.errors = list(errors)
        if not self.errors:
            super().__init__("CofolaTypeError raised with no errors.")
            return
        lines: list[str] = [
            f"{len(self.errors)} type error(s) found:"
        ]
        for i, err in enumerate(self.errors, 1):
            if err.loc is not None:
                line, col = err.loc
                lines.append(f"  [{i}] line {line}, col {col}: {err.message}")
            else:
                lines.append(f"  [{i}] {err.message}")
        super().__init__("\n".join(lines))


# Friendly spec-type names for diagnostic messages — group ABCs expand
# into the user-facing leaf types they cover, leaf classes drop their
# `Def` / `ObjDef` suffix.
_FRIENDLY_NAME: dict[type, str] = {
    SetObjDef: "Set",
    BagObjDef: "Bag",
    TupleDef: "Tuple",
    SequenceDef: "Sequence",
    CircleDef: "Circle",
    PartitionDef: "Partition",
    CompositionDef: "Composition",
    FuncObjDef: "Function",
    Entity: "Entity",
    int: "Int",
    SetLike: "Bag or Set",
    Linear: "Circle or Sequence",
    Grouped: "Composition or Partition",
    Ordered: "Circle or Sequence or Tuple",
}


def _name_of(t: type) -> str:
    return _FRIENDLY_NAME.get(t, t.__name__)


def _format_expected(expected: type | tuple[type, ...]) -> str:
    """Render a class or tuple of classes as 'A or B or C' using friendly names."""
    if isinstance(expected, tuple):
        return " or ".join(sorted(_name_of(t) for t in expected))
    return _name_of(expected)


class TypeChecker:
    """Validate operator/constraint type signatures over a frontend Problem."""

    def __init__(self) -> None:
        self.errors: list[TypeCheckError] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, problem: Problem) -> Problem:
        """Run the type check on `problem`. Returns the same problem on success.

        Raises:
            CofolaTypeError: if any type errors are found.
        """
        self.errors = []
        self._problem = problem

        # 1) Check object definitions.
        for ref, defn in problem.iter_objects():
            self._check_objdef(ref, defn)

        # 2) Check the partition-indexing-forbidden rule.
        self._check_partition_indexing()

        # 3) Check constraints (recursing into compound constraints).
        for c in problem.constraints:
            self._check_constraint(c)

        if self.errors:
            raise CofolaTypeError(self.errors)
        return problem

    # ------------------------------------------------------------------
    # Per-node dispatch helpers
    # ------------------------------------------------------------------

    def _record(self, node: object, message: str, ref: ObjRef | None = None) -> None:
        loc = self._problem.get_loc(ref) if ref is not None else None
        self.errors.append(TypeCheckError(loc=loc, message=message, node=node))

    def _check_objdef(self, ref: ObjRef, defn: object) -> None:
        sig = SIGNATURES.get(type(defn))
        if sig is not None:
            self._check_signature(defn, sig, owning_ref=ref)

        self._check_ordered_semantics(ref, defn)

    def _check_ordered_semantics(self, ref: ObjRef, defn: object) -> None:
        """Validate ordered-object rules that depend on multiple fields."""

        if not isinstance(defn, (TupleDef, SequenceDef, CircleDef)):
            return
        if not (defn.choose and defn.replace):
            return

        source_defn = self._problem.get_effective_object(defn.source)
        if isinstance(source_defn, BagObjDef):
            self._record(
                defn,
                f"{type(defn).__name__}: choose-with-replacement from a Bag "
                "source is not supported",
                ref=ref,
            )

    def _check_constraint(self, c: Constraint) -> None:
        # Recurse into compound constraints first.
        if isinstance(c, NotConstraint):
            self._check_constraint(c.sub)
            return
        if isinstance(c, AndConstraint):
            self._check_constraint(c.left)
            self._check_constraint(c.right)
            return
        if isinstance(c, OrConstraint):
            self._check_constraint(c.left)
            self._check_constraint(c.right)
            return
        if isinstance(c, ForAllParts):
            # `partition` field must reference a PartitionDef or CompositionDef.
            partition_defn = self._problem.get_object(c.partition)
            if not isinstance(partition_defn, Grouped):
                kind = (
                    type(partition_defn).__name__
                    if partition_defn is not None
                    else "unknown"
                )
                self._record(
                    c,
                    f"`for ... in <partition>` requires a Partition or "
                    f"Composition, got {kind}",
                    ref=c.partition,
                )
            if c.part_ref is None:
                self._record(
                    c,
                    "`for ... in <partition>` is missing its scoped part placeholder",
                    ref=c.partition,
                )
            elif not isinstance(self._problem.get_object(c.part_ref), PartPlaceholderDef):
                self._record(
                    c,
                    "`for ... in <partition>` part_ref must reference a PartPlaceholderDef",
                    ref=c.part_ref,
                )
            self._check_constraint(c.constraint_template)
            return
        if isinstance(c, SizeConstraint):
            # SizeConstraint terms reference either ObjRefs (any cardinal
            # object) or SizeAtom dataclasses.
            for entry, _coef in c.terms:
                if isinstance(entry, ObjRef):
                    resolved = self._problem.get_object(entry)
                    if resolved is None:
                        self._record(
                            c,
                            "SizeConstraint: term references an unknown object",
                            ref=entry,
                        )
                    elif isinstance(resolved, FuncObjDef):
                        self._record(
                            c,
                            "SizeConstraint: |X| terms require a cardinality "
                            f"object, got {_name_of(type(resolved))}",
                            ref=entry,
                        )
                elif isinstance(entry, (TupleCountAtom, BagCountAtom, SeqPatternCountAtom)):
                    self._check_size_atom(entry)
                else:
                    self._record(
                        c,
                        "SizeConstraint: terms must contain ObjRef or SizeAtom "
                        f"entries, got {type(entry).__name__}",
                    )
            return

        sig = SIGNATURES.get(type(c))
        if sig is None:
            return
        self._check_signature(c, sig)

        # Pattern fields: SequencePatternConstraint and SeqPatternCountAtom
        # carry a `pattern` whose own ObjRef fields must be SetLike.
        if isinstance(c, SequencePatternConstraint):
            self._check_pattern_fields(c.pattern, owning_constraint=c)

    def _check_size_atom(self, atom: object) -> None:
        """Validate a size atom (TupleCountAtom / BagCountAtom / SeqPatternCountAtom)."""
        sig = SIGNATURES.get(type(atom))
        if sig is None:
            return
        self._check_signature(atom, sig)
        if isinstance(atom, SeqPatternCountAtom):
            self._check_pattern_fields(atom.pattern, owning_constraint=atom)

    # ------------------------------------------------------------------
    # Signature application
    # ------------------------------------------------------------------

    def _check_signature(
        self,
        node: object,
        sig: Signature,
        owning_ref: ObjRef | None = None,
    ) -> None:
        """Validate `node` against `sig`. Errors are appended to self.errors."""
        for param in sig.params:
            field_name = param.field
            if not field_name:
                continue
            value = getattr(node, field_name, None)
            if value is None:
                continue

            # Resolve ObjRef to its underlying definition; pass Entity/int
            # values through directly so isinstance can match them.
            err_ref: ObjRef | None = owning_ref
            if isinstance(value, ObjRef):
                err_ref = value
                resolved = self._problem.get_effective_object(value)
                if resolved is None:
                    self._record(
                        node,
                        f"{type(node).__name__}: parameter `{param.name}` "
                        f"references an unknown object",
                        ref=value,
                    )
                    continue
                check_value: object = resolved
            elif isinstance(value, (Entity, int)):
                check_value = value
            else:
                # Other types (e.g. patterns) are validated elsewhere.
                continue

            if not isinstance(check_value, param.accepts):
                expected = _format_expected(param.accepts)
                got = _name_of(type(check_value))
                self._record(
                    node,
                    f"{type(node).__name__}: parameter `{param.name}` "
                    f"requires {expected}, got {got}",
                    ref=err_ref,
                )

        # Run extra spec-prescribed predicates after positional checks.
        for predicate in sig.extra:
            try:
                msg = predicate(node, self._problem)
            except Exception as exc:  # pragma: no cover - defensive
                msg = f"internal error in predicate: {exc}"
            if msg:
                self._record(node, f"{type(node).__name__}: {msg}", ref=owning_ref)

    # ------------------------------------------------------------------
    # Pattern field checks (ObjRef | Entity unions)
    # ------------------------------------------------------------------

    def _check_pattern_fields(
        self,
        pattern: object,
        owning_constraint: object,
    ) -> None:
        """Validate ObjRef/Entity fields inside a sequence pattern.

        Per spec, some patterns (TogetherPattern.group) require SetLike
        and reject Entity; others (LessThanPattern, NextToPattern,
        PredecessorPattern) accept either an Entity or a SetLike ref.
        """
        spec = PATTERN_FIELD_EXPECT.get(type(pattern))
        if spec is None:
            return
        for field_name, expect, entity_ok in spec:
            value = getattr(pattern, field_name, None)
            if value is None:
                continue
            if isinstance(value, Entity):
                if not entity_ok:
                    self._record(
                        owning_constraint,
                        f"{type(pattern).__name__}: field `{field_name}` "
                        f"requires {_format_expected(expect)}, got Entity",
                    )
                continue
            if isinstance(value, ObjRef):
                resolved = self._problem.get_effective_object(value)
                if resolved is None:
                    self._record(
                        owning_constraint,
                        f"{type(pattern).__name__}: field `{field_name}` "
                        f"references an unknown object",
                        ref=value,
                    )
                    continue
                if not isinstance(resolved, expect):
                    suffix = " or Entity" if entity_ok else ""
                    self._record(
                        owning_constraint,
                        f"{type(pattern).__name__}: field `{field_name}` "
                        f"requires {_format_expected(expect)}{suffix}, "
                        f"got {_name_of(type(resolved))}",
                        ref=value,
                    )

    # ------------------------------------------------------------------
    # Partition indexing
    # ------------------------------------------------------------------

    def _check_partition_indexing(self) -> None:
        """Indexing into an unordered partition (P[i]) is forbidden per spec.

        Approach: iterate over all concrete PartDefs in the problem. The
        transformer eagerly creates `k` PartDefs for every PartitionDef, so
        the bare existence of a PartDef does not mean the user wrote `P[i]` —
        only PartDefs that are referenced elsewhere indicate user indexing.
        """
        problem = self._problem
        # 1) Collect every PartDef and its source partition kind.
        part_refs: dict[ObjRef, tuple[object, ObjRef]] = {}
        for ref, defn in problem.iter_objects():
            if isinstance(defn, (SetPartDef, BagPartDef)):
                part_refs[ref] = (defn, defn.partition)

        if not part_refs:
            return

        # 2) Build a set of refs that are referenced by some other defn or
        #    by some constraint (anywhere — i.e. used by the user).
        used: set[ObjRef] = set()
        for ref, defn in problem.iter_objects():
            if ref in part_refs:
                continue  # the PartDef itself doesn't count
            for r in problem.get_refs(defn):
                used.add(r)
        for c in problem.constraints:
            used.update(constraint_refs(c))

        # 3) Flag every used PartDef whose source partition is unordered.
        for pref, (part_defn, partition_ref) in part_refs.items():
            if pref not in used:
                continue
            partition_defn = problem.get_object(partition_ref)
            if not isinstance(partition_defn, Grouped):
                continue
            if not isinstance(partition_defn, CompositionDef):
                self._record(
                    part_defn,
                    "Indexing into an unordered Partition is not supported "
                    "(use Composition for ordered access)",
                    ref=pref,
                )


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def validate_problem(problem: Problem) -> Problem:
    """Validate a hand-built or parsed frontend problem.

    This is the public validation boundary for callers that construct a
    ``Problem`` directly with ``ProblemBuilder``.
    """

    return TypeChecker().run(problem)
