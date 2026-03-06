"""IR pipeline — the sole IR-layer interface for solving combinatorics problems.

IRPipeline.process(problem) is the single entry point.  It performs all IR
work — passes, Shannon decomposition, connected-component decomposition — and
returns a SolveSchedule describing exactly which (Problem, Analysis) pairs to
hand to the WFOMC backend and how to combine their counts.

The caller (solver.py) only needs to:
    schedule = IRPipeline().process(problem)
    total = sum(prod(backend.solve(p,a) for p,a in b.components) for b in schedule.branches)

Data types
----------
SolveBranch
    One Shannon truth-assignment.  Its components are independent sub-problems
    whose WFOMC counts must be *multiplied* together.

SolveSchedule
    Collection of SolveBranches.  Results must be *summed* across branches.
    An empty schedule means the problem is unsatisfiable (answer = 0).

Pass lists
----------
GLOBAL_PASSES  — run once on the full problem (structural simplification).
LOCAL_PASSES   — run per atomic sub-problem, after Shannon decomposition, so
                 that SizeConstraintFolder's LP sees only atomic constraints.

Design invariant: no pass in LOCAL_PASSES may create new compound constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from dataclasses import replace as dc_replace

from loguru import logger
from sympy import Symbol, satisfiable

from cofola.frontend.constraints import (
    AndConstraint,
    BagEqConstraint,
    BagSubsetConstraint,
    Constraint,
    DisjointConstraint,
    EqualityConstraint,
    FuncPairConstraint,
    MembershipConstraint,
    NotConstraint,
    OrConstraint,
    SequencePatternConstraint,
    SizeConstraint,
    SubsetConstraint,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import BagInit, SetInit
from cofola.frontend.pretty import fmt_analysis, fmt_problem
from cofola.frontend.problem import Problem
from cofola.frontend.types import ObjRef
from cofola.ir.analysis.bag_classify import BagClassification
from cofola.ir.analysis.entities import AnalysisResult
from cofola.ir.pass_manager import AnalysisManager
from cofola.ir.passes.lowering import LoweringPass
from cofola.ir.passes.merge_identical import MergeIdenticalObjects
from cofola.ir.passes.optimize import ConstantFolder, SizeConstraintFolder, UnsatisfiableConstraint
from cofola.ir.passes.simplify import SimplifyPass


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SolveBranch:
    """One Shannon truth-assignment.

    ``components`` is a list of (Problem, BagClassification) pairs that are
    *independent* of each other.  Their WFOMC counts must be multiplied.
    """
    components: list[tuple[Problem, BagClassification]]


@dataclass
class SolveSchedule:
    """Complete solve plan returned by IRPipeline.process().

    Each branch corresponds to one satisfying Shannon assignment.
    Their WFOMC counts must be summed.  An empty schedule means the
    problem is unsatisfiable (answer = 0).
    """
    branches: list[SolveBranch]

    def is_unsatisfiable(self) -> bool:
        return len(self.branches) == 0


# ---------------------------------------------------------------------------
# Constraint helpers (internal)
# ---------------------------------------------------------------------------

_HAS_POSITIVE = (
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    SequencePatternConstraint,
    FuncPairConstraint,
    BagSubsetConstraint,
    BagEqConstraint,
)
_COMPOUND = (NotConstraint, AndConstraint, OrConstraint)


def _negate_constraint(c: Constraint) -> Constraint:
    if isinstance(c, _HAS_POSITIVE):
        return dc_replace(c, positive=not c.positive)
    if isinstance(c, SizeConstraint):
        flip = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}
        if c.comparator in flip:
            return dc_replace(c, comparator=flip[c.comparator])
        return OrConstraint(
            left=dc_replace(c, comparator="<=", rhs=c.rhs - 1),
            right=dc_replace(c, comparator=">=", rhs=c.rhs + 1),
        )
    if isinstance(c, NotConstraint):
        return c.sub
    if isinstance(c, AndConstraint):
        return OrConstraint(
            left=_negate_constraint(c.left),
            right=_negate_constraint(c.right),
        )
    if isinstance(c, OrConstraint):
        return AndConstraint(
            left=_negate_constraint(c.left),
            right=_negate_constraint(c.right),
        )
    raise TypeError(f"Cannot negate constraint of type {type(c).__name__}")


def _extract_refs_from_value(val: object) -> list[ObjRef]:
    """Recursively extract all ObjRef instances from a field value."""
    if isinstance(val, ObjRef):
        return [val]
    if isinstance(val, tuple):
        result = []
        for item in val:
            result.extend(_extract_refs_from_value(item))
        return result
    if hasattr(val, "__dataclass_fields__"):
        result = []
        for fname in val.__dataclass_fields__:
            result.extend(_extract_refs_from_value(getattr(val, fname)))
        return result
    return []


def _constraint_refs(c: Constraint) -> list[ObjRef]:
    refs = []
    if isinstance(c, (AndConstraint, OrConstraint)):
        refs.extend(_constraint_refs(c.left))
        refs.extend(_constraint_refs(c.right))
    elif isinstance(c, NotConstraint):
        refs.extend(_constraint_refs(c.sub))
    else:
        for f in fields(c):
            refs.extend(_extract_refs_from_value(getattr(c, f.name)))
    return refs


def _decompose_into_components(problem: Problem) -> list[Problem]:
    """Split a problem into independent connected-component sub-problems."""
    all_refs = set(problem.refs())
    if not all_refs:
        return [problem]

    parent: dict[ObjRef, ObjRef] = {r: r for r in all_refs}

    def find(x: ObjRef) -> ObjRef:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: ObjRef, b: ObjRef) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ref, defn in problem.defs:
        for dep_ref in problem.get_refs(defn):
            if dep_ref in all_refs:
                union(ref, dep_ref)

    for c in problem.constraints:
        c_refs = [r for r in _constraint_refs(c) if r in all_refs]
        for i in range(1, len(c_refs)):
            union(c_refs[0], c_refs[i])

    component_of = {r: find(r) for r in all_refs}
    roots = set(component_of.values())

    if len(roots) <= 1:
        return [problem]

    logger.debug("_decompose: {} connected components found", len(roots))

    sub_problems = []
    for root in roots:
        comp_refs = frozenset(r for r, rt in component_of.items() if rt == root)
        sub_defs = tuple((r, d) for r, d in problem.defs if r in comp_refs)
        sub_constraints = tuple(
            c for c in problem.constraints
            if any(r in comp_refs for r in _constraint_refs(c))
        )
        if not sub_constraints and all(
            isinstance(d, (SetInit, BagInit)) for _, d in sub_defs
        ):
            logger.debug("  skipping trivial constant-only component (root={})", root)
            continue
        sub_names = tuple((r, n) for r, n in problem.names if r in comp_refs)
        sub_problems.append(Problem(
            defs=sub_defs,
            constraints=sub_constraints,
            names=sub_names,
        ))

    return sub_problems if sub_problems else [problem]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IRPipeline:
    """IR pipeline — the sole IR-layer interface.

    Call process(problem) to get a SolveSchedule.  All pass execution,
    Shannon decomposition, and connected-component decomposition happen here.
    The caller only needs to evaluate the schedule against a WFOMC backend.
    """

    GLOBAL_PASSES = [
        ConstantFolder,
        MergeIdenticalObjects,
    ]

    LOCAL_PASSES = [
        SizeConstraintFolder,
        LoweringPass,
        MergeIdenticalObjects,
        SimplifyPass,
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, problem: Problem) -> SolveSchedule:
        """Transform and decompose a problem into a ready-to-solve schedule.

        Steps:
        1. Run GLOBAL_PASSES once on the full problem.
        2. Shannon-decompose compound constraints (recursively flattened).
        3. For each Shannon branch: run LOCAL_PASSES + component decomposition.

        Returns:
            SolveSchedule with one SolveBranch per satisfying Shannon
            assignment.  Empty schedule means unsatisfiable (answer = 0).
        """
        logger.debug("\n{}", fmt_problem(problem, stage="[Input] Parsed Problem"))

        # Phase 1: global structural passes
        try:
            am = self.run_passes(problem, self.GLOBAL_PASSES)
        except UnsatisfiableConstraint as exc:
            logger.info("IRPipeline: unsatisfiable after global passes → 0 ({})", exc)
            return SolveSchedule(branches=[])

        logger.debug("\n{}", fmt_problem(am.problem, stage="[After Global Passes]"))

        # Phase 2+3: Shannon expansion → local passes → component decomp
        branches = self._collect_branches(am.problem)
        return SolveSchedule(branches=branches)

    # ------------------------------------------------------------------
    # Pass runner (reusable helper)
    # ------------------------------------------------------------------

    @classmethod
    def run_passes(
        cls,
        problem: Problem,
        pass_classes: list,
        *,
        check_compound_invariant: bool = False,
    ) -> AnalysisManager:
        """Run a list of transform passes and return the final AnalysisManager.

        Raises:
            UnsatisfiableConstraint: propagated from SizeConstraintFolder.
        """
        am = AnalysisManager(problem)
        for pass_cls in pass_classes:
            pass_ = pass_cls()
            pass_name = pass_cls.__name__
            logger.info("[Pass] {}", pass_name)

            before_compound = (
                sum(1 for c in am.problem.constraints if isinstance(c, _COMPOUND))
                if check_compound_invariant else 0
            )

            new_problem = pass_.run(am.problem, am)
            am.update(new_problem)
            logger.debug("\n{}", fmt_problem(am.problem, stage=f"[After] {pass_name}"))

            if check_compound_invariant:
                after_compound = sum(
                    1 for c in am.problem.constraints if isinstance(c, _COMPOUND)
                )
                assert after_compound <= before_compound, (
                    f"{pass_name} introduced new compound constraints "
                    f"({before_compound} → {after_compound})"
                )
        return am

    # ------------------------------------------------------------------
    # Internal: Shannon decomposition
    # ------------------------------------------------------------------

    def _collect_branches(self, problem: Problem) -> list[SolveBranch]:
        """Recursively expand compound constraints into a flat branch list.

        If no compound constraints exist, delegates to _make_branch().
        Otherwise enumerates all satisfying Shannon assignments and recurses.
        The recursion handles the case where negating a SizeConstraint(==)
        produces a new OrConstraint.
        """
        if not any(isinstance(c, _COMPOUND) for c in problem.constraints):
            return self._make_branch(problem)

        # Build propositional formula over atomic constraint symbols
        atomic_constraints: list[Constraint] = []
        idx_to_sym: dict[int, Symbol] = {}

        def _get_sym(c: Constraint) -> Symbol:
            for i, existing in enumerate(atomic_constraints):
                if existing == c:
                    return idx_to_sym[i]
            idx = len(atomic_constraints)
            atomic_constraints.append(c)
            sym = Symbol(f"c_{idx}")
            idx_to_sym[idx] = sym
            return sym

        def _build_formula(c: Constraint):
            if isinstance(c, NotConstraint):
                return ~_build_formula(c.sub)
            if isinstance(c, AndConstraint):
                return _build_formula(c.left) & _build_formula(c.right)
            if isinstance(c, OrConstraint):
                return _build_formula(c.left) | _build_formula(c.right)
            return _get_sym(c)

        formula = True
        for c in problem.constraints:
            formula = formula & _build_formula(c)

        logger.info("Shannon: {} atoms, formula={}", len(atomic_constraints), formula)

        branches: list[SolveBranch] = []
        for model in satisfiable(formula, all_models=True):
            if model is False:
                break
            sub_constraints: list[Constraint] = []
            for idx, atomic in enumerate(atomic_constraints):
                sym = idx_to_sym[idx]
                if model.get(sym, True):
                    sub_constraints.append(atomic)
                else:
                    sub_constraints.append(_negate_constraint(atomic))

            logger.debug(
                "  model={} → {} constraints",
                {str(k): v for k, v in model.items()},
                len(sub_constraints),
            )
            sub_prob = dc_replace(problem, constraints=tuple(sub_constraints))
            # Recurse: negated SizeConstraint(==) may introduce a new OrConstraint
            branches.extend(self._collect_branches(sub_prob))

        return branches

    # ------------------------------------------------------------------
    # Internal: local passes + component decomposition
    # ------------------------------------------------------------------

    def _make_branch(self, problem: Problem) -> list[SolveBranch]:
        """Run LOCAL_PASSES and component decomposition for one atomic problem.

        Returns a list with one SolveBranch on success, or an empty list if
        the problem is unsatisfiable (so the branch contributes 0 to the sum).
        """
        # Local passes: SizeConstraintFolder (LP is most accurate here, since
        # all constraints are atomic), LoweringPass, SimplifyPass
        try:
            am = self.run_passes(problem, self.LOCAL_PASSES)
        except UnsatisfiableConstraint as exc:
            logger.info("IRPipeline: unsatisfiable after local passes → 0 ({})", exc)
            return []

        final_analysis = am.get(BagClassification)
        logger.debug(
            "\n{}",
            fmt_analysis(final_analysis, am.problem, stage="[Final] BagClassification"),
        )

        if final_analysis.unsatisfiable:
            logger.info("IRPipeline: unsatisfiable constraints → 0")
            return []

        logger.info(
            "[Decompose] {} objects, {} constraints",
            len(am.problem.defs), len(am.problem.constraints),
        )

        # Connected-component decomposition
        components = _decompose_into_components(am.problem)
        if len(components) > 1:
            logger.info("IRPipeline: decomposed into {} independent components", len(components))

        branch_components: list[tuple[Problem, BagClassification]] = []
        for comp in components:
            sub_am = AnalysisManager(comp)
            sub_analysis = sub_am.get(BagClassification)
            if sub_analysis.unsatisfiable:
                logger.info("IRPipeline: component is unsatisfiable → branch contributes 0")
                return []
            branch_components.append((comp, sub_analysis))
            logger.debug("  component: {} objects", len(comp.defs))

        return [SolveBranch(components=branch_components)]
