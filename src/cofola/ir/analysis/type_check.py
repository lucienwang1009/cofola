"""TypeCheckPass — validates operator/constraint type signatures on the IR.

Walks the IR `Problem` once, looks up each IR node's signature in
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
    BagEqConstraint,
    BagSubsetConstraint,
    Constraint,
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
    BagPartRef,
    PartitionDef,
    SetPartRef,
)
from cofola.frontend.problem import Problem, SourceLoc
from cofola.frontend.types import Entity, ObjRef
from cofola.ir.analysis.op_signatures import (
    PATTERN_FIELD_EXPECT,
    SIGNATURES,
    Signature,
)
from cofola.ir.analysis.type_lattice import (
    CofolaType,
    is_subtype,
    type_of,
)


@dataclass(frozen=True)
class TypeCheckError:
    """One type-check error.

    Attributes:
        loc: Source (line, col) of the offending node, or None.
        message: Human-readable description.
        node: The offending IR node (for debugging / programmatic access).
    """

    loc: Optional[SourceLoc]
    message: str
    node: object


class CofolaTypeError(Exception):
    """Raised when one or more type errors are found in a Cofola IR.

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


def _format_expected(expected) -> str:
    """Render a single CofolaType or a frozenset of them as 'A or B or C'."""
    if isinstance(expected, CofolaType):
        return expected.value
    return " or ".join(sorted(t.value for t in expected))


class TypeCheckPass:
    """Validate operator/constraint type signatures over an IR Problem."""

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
        if sig is None:
            # No signature registered (e.g. SetInit, BagInit, FuncDef, FuncInverse).
            return
        self._check_signature(defn, sig, owning_ref=ref)

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
            # `partition` field must reference a PartitionDef (any kind).
            partition_defn = self._problem.get_object(c.partition)
            if not isinstance(partition_defn, PartitionDef):
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
            self._check_constraint(c.constraint_template)
            return
        if isinstance(c, SizeConstraint):
            # SizeConstraint terms reference either ObjRefs (any cardinal
            # object) or SizeAtom dataclasses. We don't restrict the ObjRef
            # case (sets, bags, tuples, sequences, partitions all have
            # |X|), but we DO validate any SizeAtom carried as a term.
            for entry, _coef in c.terms:
                if isinstance(entry, (TupleCountAtom, BagCountAtom, SeqPatternCountAtom)):
                    self._check_size_atom(entry)
            return

        sig = SIGNATURES.get(type(c))
        if sig is None:
            return
        self._check_signature(c, sig)

        # Pattern fields: SequencePatternConstraint and SeqPatternCountAtom
        # carry a `pattern` whose own ObjRef fields must be SET_LIKE.
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

            # Resolve the value to a CofolaType.
            actual: CofolaType | None
            err_ref: ObjRef | None = owning_ref
            if isinstance(value, ObjRef):
                err_ref = value
                resolved = self._problem.get_object(value)
                if resolved is None:
                    self._record(
                        node,
                        f"{type(node).__name__}: parameter `{param.name}` "
                        f"references an unknown object",
                        ref=value,
                    )
                    continue
                try:
                    actual = type_of(resolved)
                except TypeError:
                    self._record(
                        node,
                        f"{type(node).__name__}: parameter `{param.name}` "
                        f"references an object of unknown type "
                        f"({type(resolved).__name__})",
                        ref=value,
                    )
                    continue
            elif isinstance(value, Entity):
                actual = CofolaType.ENTITY
            elif isinstance(value, int):
                actual = CofolaType.INT
            else:
                # Other types (e.g. patterns) are validated elsewhere.
                continue

            if not is_subtype(actual, param.accepts):
                expected = _format_expected(param.accepts)
                self._record(
                    node,
                    f"{type(node).__name__}: parameter `{param.name}` "
                    f"requires {expected}, got {actual.value}",
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

        Per spec, some patterns (TogetherPattern.group) require SET_LIKE
        and reject Entity; others (LessThanPattern, NextToPattern,
        PredecessorPattern) accept either an Entity or a SET_LIKE ref.
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
                resolved = self._problem.get_object(value)
                if resolved is None:
                    self._record(
                        owning_constraint,
                        f"{type(pattern).__name__}: field `{field_name}` "
                        f"references an unknown object",
                        ref=value,
                    )
                    continue
                try:
                    actual = type_of(resolved)
                except TypeError:
                    continue
                if not is_subtype(actual, expect):
                    suffix = " or Entity" if entity_ok else ""
                    self._record(
                        owning_constraint,
                        f"{type(pattern).__name__}: field `{field_name}` "
                        f"requires {_format_expected(expect)}{suffix}, "
                        f"got {actual.value}",
                        ref=value,
                    )

    # ------------------------------------------------------------------
    # Partition indexing
    # ------------------------------------------------------------------

    def _check_partition_indexing(self) -> None:
        """Indexing into an unordered partition (P[i]) is forbidden per spec.

        Approach: iterate over all PartRefs in the problem. The transformer
        eagerly creates `k` PartRefs for every PartitionDef, so the bare
        existence of a PartRef does not mean the user wrote `P[i]` — only
        PartRefs that are *referenced* elsewhere indicate user indexing.
        """
        problem = self._problem
        # 1) Collect every PartRef and its source partition kind.
        #    Skip sentinel PartRefs (index == -1) created for `for part in P`
        #    forall-parts; those are not user indexing and are valid on
        #    unordered partitions per the spec.
        part_refs: dict[ObjRef, tuple[object, ObjRef]] = {}
        for ref, defn in problem.iter_objects():
            if isinstance(defn, (SetPartRef, BagPartRef)):
                if defn.index < 0:
                    continue
                part_refs[ref] = (defn, defn.partition)

        if not part_refs:
            return

        # 2) Build a set of refs that are referenced by some other defn or
        #    by some constraint (anywhere — i.e. used by the user).
        used: set[ObjRef] = set()
        for ref, defn in problem.iter_objects():
            if ref in part_refs:
                continue  # the PartRef itself doesn't count
            for r in problem.get_refs(defn):
                used.add(r)
        for c in problem.constraints:
            for r in _constraint_refs(c):
                used.add(r)

        # 3) Flag every used PartRef whose source PartitionDef is unordered.
        for pref, (part_defn, partition_ref) in part_refs.items():
            if pref not in used:
                continue
            partition_defn = problem.get_object(partition_ref)
            if not isinstance(partition_defn, PartitionDef):
                continue
            if not partition_defn.ordered:
                self._record(
                    part_defn,
                    "Indexing into an unordered Partition is not supported "
                    "(use Composition for ordered access)",
                    ref=pref,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constraint_refs(c: Constraint) -> list[ObjRef]:
    """Collect every ObjRef appearing inside a constraint (recursive)."""
    from dataclasses import fields, is_dataclass

    refs: list[ObjRef] = []

    def visit(val: object) -> None:
        if isinstance(val, ObjRef):
            refs.append(val)
        elif isinstance(val, tuple):
            for x in val:
                visit(x)
        elif is_dataclass(val) and not isinstance(val, type):
            for f in fields(val):
                visit(getattr(val, f.name))

    if isinstance(c, NotConstraint):
        refs.extend(_constraint_refs(c.sub))
    elif isinstance(c, (AndConstraint, OrConstraint)):
        refs.extend(_constraint_refs(c.left))
        refs.extend(_constraint_refs(c.right))
    elif isinstance(c, ForAllParts):
        refs.append(c.partition)
        refs.extend(_constraint_refs(c.constraint_template))
    else:
        from dataclasses import fields as _fields

        for f in _fields(c):
            visit(getattr(c, f.name))
    return refs
