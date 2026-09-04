"""Lifted symbolic encoding for exchangeable bag entities.

This module is deliberately backend-local: it compiles a conservative subset
of bag object graphs into one generating function per exchangeable entity
class. The first implementation supports fixed ``BagInit`` roots followed by
``BagChoose`` and ``BagIntersection`` nodes. Anything outside that fragment is
left to the existing per-entity encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, cast

from sympy import Integer

import cofola.frontend.constraints as ir_cst
import cofola.frontend.objects as ir_obj
from cofola.backend.wfomc.api import Const, symbolic_ganak_count
from cofola.backend.wfomc.context import Context
from cofola.frontend.objects import Entity, ObjRef
from cofola.frontend.problem import Problem
from cofola.frontend.utils import constraint_refs
from cofola.planing.analysis.entities import AnalysisResult
from loguru import logger


# Cumulative upper bound on finite-domain assignments inspected while planning
# the local propositional compilation. The check happens before AnalysisResult
# is changed, so an expensive orbit can fall back completely to the existing
# per-entity encoding.
MAX_FACTOR_STATES = 100_000


_BooleanTerm = int | bool


class _CNFBuilder(object):
    """Build a functional binary circuit without duplicating model counts."""

    def __init__(self) -> None:
        self.next_var = 1
        self.clauses: list[frozenset[int]] = []

    @property
    def n_vars(self) -> int:
        return self.next_var - 1

    def fresh(self) -> int:
        variable = self.next_var
        self.next_var += 1
        return variable

    @staticmethod
    def negate(term: _BooleanTerm) -> _BooleanTerm:
        return not term if isinstance(term, bool) else -term

    def add_clause(self, *terms: _BooleanTerm) -> None:
        literals: set[int] = set()
        for term in terms:
            if term is True:
                return
            if term is False:
                continue
            if -term in literals:
                return
            literals.add(term)
        self.clauses.append(frozenset(literals))

    def and_(self, left: _BooleanTerm, right: _BooleanTerm) -> _BooleanTerm:
        if left is False or right is False:
            return False
        if left is True:
            return right
        if right is True:
            return left
        if left == right:
            return left
        if left == -right:
            return False
        result = self.fresh()
        self.add_clause(-result, left)
        self.add_clause(-result, right)
        self.add_clause(result, -left, -right)
        return result

    def or_(self, left: _BooleanTerm, right: _BooleanTerm) -> _BooleanTerm:
        if left is True or right is True:
            return True
        if left is False:
            return right
        if right is False:
            return left
        if left == right:
            return left
        if left == -right:
            return True
        result = self.fresh()
        self.add_clause(result, -left)
        self.add_clause(result, -right)
        self.add_clause(-result, left, right)
        return result

    def equal(self, left: _BooleanTerm, right: _BooleanTerm) -> _BooleanTerm:
        if isinstance(left, bool):
            return right if left else self.negate(right)
        if isinstance(right, bool):
            return left if right else self.negate(left)
        if left == right:
            return True
        if left == -right:
            return False
        result = self.fresh()
        self.add_clause(left, right, result)
        self.add_clause(left, -right, -result)
        self.add_clause(-left, right, -result)
        self.add_clause(-left, -right, result)
        return result

    def leq(
        self,
        left: tuple[_BooleanTerm, ...],
        right: tuple[_BooleanTerm, ...],
    ) -> _BooleanTerm:
        """Return a literal equivalent to unsigned ``left <= right``."""
        width = max(len(left), len(right))
        left = left + (False,) * (width - len(left))
        right = right + (False,) * (width - len(right))
        equal_prefix: _BooleanTerm = True
        less_prefix: _BooleanTerm = False
        for left_bit, right_bit in zip(reversed(left), reversed(right)):
            first_less = self.and_(
                equal_prefix,
                self.and_(self.negate(left_bit), right_bit),
            )
            less_prefix = self.or_(less_prefix, first_less)
            equal_prefix = self.and_(
                equal_prefix,
                self.equal(left_bit, right_bit),
            )
        return self.or_(less_prefix, equal_prefix)

    def constrain_min(
        self,
        result: tuple[_BooleanTerm, ...],
        left: tuple[_BooleanTerm, ...],
        right: tuple[_BooleanTerm, ...],
    ) -> None:
        """Add a functional encoding of ``result == min(left, right)``."""
        width = max(len(result), len(left), len(right))
        result = result + (False,) * (width - len(result))
        left = left + (False,) * (width - len(left))
        right = right + (False,) * (width - len(right))
        select_left = self.leq(left, right)
        for result_bit, left_bit, right_bit in zip(result, left, right):
            self.add_clause(
                self.negate(select_left),
                self.equal(result_bit, left_bit),
            )
            self.add_clause(
                select_left,
                self.equal(result_bit, right_bit),
            )


@dataclass(frozen=True, slots=True)
class LiftedBagOrbit:
    """An exchangeable entity class and its bounded Ganak compilation plan."""

    root: ObjRef
    multiplicity: int
    entities: frozenset[Entity]
    refs: tuple[ObjRef, ...]
    upper_bounds: tuple[tuple[ObjRef, int], ...] = ()
    tracked_refs: frozenset[ObjRef] = frozenset()
    estimated_work: int = 0

    @property
    def dynamic_refs(self) -> tuple[ObjRef, ...]:
        """Return the non-constant bag states in the local 1-type encoding."""
        return tuple(ref for ref in self.refs if ref != self.root)

    @property
    def bounds(self) -> dict[ObjRef, int]:
        """Return each dynamic bag state's finite multiplicity upper bound."""
        return dict(self.upper_bounds)


class BagLiftPlan(object):
    """A conservative lifted-encoding plan for one WFOMC component."""

    def __init__(
        self,
        problem: Problem,
        analysis: AnalysisResult,
        orbits: tuple[LiftedBagOrbit, ...],
    ) -> None:
        self.problem = problem
        self.analysis = analysis
        self.orbits = orbits

    @classmethod
    def empty(cls, problem: Problem, analysis: AnalysisResult) -> "BagLiftPlan":
        """Return a plan that delegates every entity to the existing encoder."""
        cls._make_all_entities_distinguishable(analysis)
        return cls(problem, analysis, ())

    @classmethod
    def build(cls, problem: Problem, analysis: AnalysisResult) -> "BagLiftPlan":
        """Build a plan, falling back unless an entire orbit is safe and cheap."""
        candidates: list[LiftedBagOrbit] = []
        planned_work = 0
        topo_order = tuple(problem.topological_order())

        for root in topo_order:
            root_defn = problem.get_object(root)
            root_info = analysis.bag_info.get(root)
            if not isinstance(root_defn, ir_obj.BagInit) or root_info is None:
                continue

            for multiplicity, entities in root_info.indis_entities.items():
                if multiplicity <= 1 or len(entities) <= 1:
                    continue
                orbit = cls._propagate_orbit(
                    problem,
                    analysis,
                    topo_order,
                    root,
                    multiplicity,
                    frozenset(entities),
                )
                if orbit is None or not cls._is_safe(problem, orbit):
                    continue
                planned_orbit = cls._plan_orbit(problem, analysis, orbit)
                if planned_orbit is None:
                    continue
                if planned_work + planned_orbit.estimated_work > MAX_FACTOR_STATES:
                    logger.debug(
                        "BagLiftPlan: orbit rooted at {} exceeds remaining plan budget",
                        orbit.root.id,
                    )
                    continue
                candidates.append(planned_orbit)
                planned_work += planned_orbit.estimated_work

        # Start from the known-correct encoding. Valid orbits are then removed
        # from its per-entity sets below. This makes fallback total rather than
        # relying on every future object/constraint to remember lifting rules.
        cls._make_all_entities_distinguishable(analysis)
        for orbit in candidates:
            for ref in orbit.refs:
                info = analysis.bag_info[ref]
                info.dis_entities.difference_update(orbit.entities)
                info.indis_entities.setdefault(orbit.multiplicity, set()).update(
                    orbit.entities
                )

        logger.debug(
            "BagLiftPlan: {} lifted orbit(s): {}",
            len(candidates),
            [
                {
                    "root": orbit.root.id,
                    "multiplicity": orbit.multiplicity,
                    "entities": sorted(entity.name for entity in orbit.entities),
                    "refs": [ref.id for ref in orbit.refs],
                    "tracked_refs": [ref.id for ref in orbit.tracked_refs],
                    "estimated_work": orbit.estimated_work,
                }
                for orbit in candidates
            ],
        )
        return cls(problem, analysis, tuple(candidates))

    @staticmethod
    def _make_all_entities_distinguishable(analysis: AnalysisResult) -> None:
        for info in analysis.bag_info.values():
            for entities in info.indis_entities.values():
                info.dis_entities.update(entities)
            info.indis_entities = {}

    @classmethod
    def _propagate_orbit(
        cls,
        problem: Problem,
        analysis: AnalysisResult,
        topo_order: tuple[ObjRef, ...],
        root: ObjRef,
        multiplicity: int,
        entities: frozenset[Entity],
    ) -> LiftedBagOrbit | None:
        refs: set[ObjRef] = {root}

        for ref in topo_order:
            defn = problem.get_object(ref)
            info = analysis.bag_info.get(ref)
            if info is None:
                continue

            if isinstance(defn, ir_obj.BagChoose) and defn.source in refs:
                if all(entity in info.p_entities_multiplicity for entity in entities):
                    refs.add(ref)
            elif (
                isinstance(defn, ir_obj.BagIntersection)
                and defn.left in refs
                and defn.right in refs
                and all(entity in info.p_entities_multiplicity for entity in entities)
            ):
                refs.add(ref)

        if refs == {root}:
            return None
        ordered_refs = tuple(ref for ref in topo_order if ref in refs)
        return LiftedBagOrbit(root, multiplicity, entities, ordered_refs)

    @classmethod
    def _is_safe(cls, problem: Problem, orbit: LiftedBagOrbit) -> bool:
        refs = set(orbit.refs)

        # A consumer outside the supported subgraph could observe the hidden
        # support or ask for a multiplicity variable that no longer exists.
        for consumer, defn in problem.iter_objects():
            dependencies = set(problem.get_refs(defn))
            if dependencies & refs and consumer not in refs:
                logger.debug(
                    "BagLiftPlan: orbit rooted at {} falls back for consumer {} ({})",
                    orbit.root.id,
                    consumer.id,
                    type(defn).__name__,
                )
                return False

        for constraint in problem.constraints:
            used_refs = set(constraint_refs(constraint)) & refs
            if not used_refs:
                continue
            if not isinstance(constraint, ir_cst.SizeConstraint):
                return False
            for term, _ in constraint.terms:
                if (
                    isinstance(term, ir_cst.BagCountAtom)
                    and term.bag in refs
                    and term.entity in orbit.entities
                ):
                    return False

        return True

    @classmethod
    def _plan_orbit(
        cls,
        problem: Problem,
        analysis: AnalysisResult,
        orbit: LiftedBagOrbit,
    ) -> LiftedBagOrbit | None:
        """Attach finite domains and a bounded local-CNF compilation plan."""
        upper_bounds: dict[ObjRef, int] = {}
        for ref in orbit.dynamic_refs:
            info = analysis.bag_info[ref]
            ref_bounds = {
                info.p_entities_multiplicity[entity] for entity in orbit.entities
            }
            if len(ref_bounds) != 1:
                logger.debug(
                    "BagLiftPlan: orbit rooted at {} has inconsistent bounds at {}",
                    orbit.root.id,
                    ref.id,
                )
                return None
            upper_bounds[ref] = ref_bounds.pop()

        tracked_refs = cls._tracked_refs(problem, orbit)
        estimated_work = cls._estimate_compilation_work(
            problem,
            orbit,
            upper_bounds,
            tracked_refs,
        )
        if estimated_work is None:
            logger.debug(
                "BagLiftPlan: orbit rooted at {} exceeds factor budget {}",
                orbit.root.id,
                MAX_FACTOR_STATES,
            )
            return None
        return LiftedBagOrbit(
            root=orbit.root,
            multiplicity=orbit.multiplicity,
            entities=orbit.entities,
            refs=orbit.refs,
            upper_bounds=tuple(
                (ref, upper_bounds[ref]) for ref in orbit.dynamic_refs
            ),
            tracked_refs=tracked_refs,
            estimated_work=estimated_work,
        )

    @staticmethod
    def _tracked_refs(
        problem: Problem,
        orbit: LiftedBagOrbit,
    ) -> frozenset[ObjRef]:
        """Return bag sizes that must remain visible to decoder validators."""
        dynamic_refs = set(orbit.dynamic_refs)
        tracked: set[ObjRef] = set()
        for ref in orbit.dynamic_refs:
            defn = problem.get_object(ref)
            if isinstance(defn, ir_obj.BagChoose) and defn.size is not None:
                tracked.add(ref)
        for constraint in problem.constraints:
            if not isinstance(constraint, ir_cst.SizeConstraint):
                continue
            tracked.update(
                term
                for term, _ in constraint.terms
                if isinstance(term, ObjRef) and term in dynamic_refs
            )
        return frozenset(tracked)

    @classmethod
    def _estimate_compilation_work(
        cls,
        problem: Problem,
        orbit: LiftedBagOrbit,
        upper_bounds: dict[ObjRef, int],
        tracked_refs: frozenset[ObjRef],
    ) -> int | None:
        """Bound binary-circuit size and the observable polynomial size."""
        domain_sizes = {ref: bound + 1 for ref, bound in upper_bounds.items()}
        bit_widths = {ref: bound.bit_length() for ref, bound in upper_bounds.items()}
        # Every tracked state can produce a distinct exponent vector in the
        # final polynomial, so include the output-space bound as well.
        output_states = prod(domain_sizes[ref] for ref in tracked_refs)
        circuit_work = sum(bit_widths.values())
        for ref in orbit.dynamic_refs:
            scope = cls._operation_scope(problem, orbit, ref)
            # Comparators, equality gates, and the min multiplexer all have
            # size linear in the largest operand width. The constant is a
            # conservative gate/clause allowance, not an asymptotic factor.
            circuit_work += 32 * max((bit_widths[item] for item in scope), default=1)
        work = circuit_work + output_states
        return work if work <= MAX_FACTOR_STATES else None

    @staticmethod
    def _operation_scope(
        problem: Problem,
        orbit: LiftedBagOrbit,
        ref: ObjRef,
    ) -> tuple[ObjRef, ...]:
        """Return the dynamic multiplicities used by one local operation."""
        defn = problem.get_object(ref)
        if isinstance(defn, ir_obj.BagChoose):
            dependencies = (ref, defn.source)
        elif isinstance(defn, ir_obj.BagIntersection):
            dependencies = (ref, defn.left, defn.right)
        else:
            raise TypeError(f"Unsupported lifted bag operation: {type(defn).__name__}")

        dynamic_refs = set(orbit.dynamic_refs)
        scope: list[ObjRef] = []
        for item in dependencies:
            if item in dynamic_refs and item not in scope:
                scope.append(item)
        return tuple(scope)

    def prepare(self, context: Context) -> None:
        """Create shared variables before object encoders build size expressions."""
        for orbit in self.orbits:
            for ref in orbit.dynamic_refs:
                if ref in orbit.tracked_refs:
                    context.get_indis_entity_var(ref, orbit.multiplicity)

    def finalize(self, context: Context) -> None:
        """Attach each orbit polynomial once and suppress duplicate support models."""
        for index, orbit in enumerate(self.orbits):
            polynomial = self._build_polynomial(orbit, context)
            group_pred = context.create_pred(
                f"lifted_bag_group_{orbit.root.id}_{orbit.multiplicity}_{index}",
                1,
            )
            for entity in self.analysis.all_entities:
                atom = cast(Any, group_pred(Const(entity.name)))
                context.unary_evidence.add(
                    atom if entity in orbit.entities else ~atom
                )
            context.weighting[group_pred] = (polynomial, 1)

            # Multiplicity and support choices for these ground atoms have been
            # summed into ``polynomial``. Fixing the ordinary bag predicates to
            # false prevents WFOMC from enumerating the same choices again.
            for ref in orbit.dynamic_refs:
                bag_pred = cast(Any, context.get_pred(ref))
                for entity in orbit.entities:
                    context.unary_evidence.add(~bag_pred(Const(entity.name)))

    def _build_polynomial(self, orbit: LiftedBagOrbit, context: Context) -> object:
        """Compile one representative's 1-types and sum them with Ganak.

        A multiplicity in ``[0, u]`` is represented by ``ceil(log2(u+1))``
        Boolean variables. A complete assignment to all such variables is a
        1-type for the representative entity. Local CNF clauses reject the
        assignments that violate bag operations, while bit weights encode
        ``x_ref ** multiplicity`` for observable bags. Ganak then returns the
        exact symbolic sum of all valid 1-type weights.
        """
        builder = _CNFBuilder()
        state_bits: dict[ObjRef, tuple[int, ...]] = {}
        for ref in orbit.dynamic_refs:
            width = orbit.bounds[ref].bit_length()
            state_bits[ref] = tuple(builder.fresh() for _ in range(width))

        weights: dict[int, tuple[object, object]] = {
            variable: (Integer(1), Integer(1))
            for bits in state_bits.values()
            for variable in bits
        }

        # Exclude unused bit patterns above each inferred multiplicity bound.
        for ref, bits in state_bits.items():
            bound = orbit.bounds[ref]
            if bound != (1 << len(bits)) - 1:
                builder.add_clause(
                    builder.leq(bits, self._constant_bits(bound, len(bits)))
                )

        for ref in orbit.dynamic_refs:
            if ref in orbit.tracked_refs:
                marker = cast(
                    Any,
                    context.get_indis_entity_var(ref, orbit.multiplicity),
                )
                for bit, variable in enumerate(state_bits[ref]):
                    weights[variable] = (marker ** (1 << bit), Integer(1))

            self._encode_operation(builder, orbit, ref, state_bits)

        for variable in range(1, builder.n_vars + 1):
            weights.setdefault(variable, (Integer(1), Integer(1)))
        return symbolic_ganak_count(builder.n_vars, builder.clauses, weights)

    def _encode_operation(
        self,
        builder: _CNFBuilder,
        orbit: LiftedBagOrbit,
        ref: ObjRef,
        state_bits: dict[ObjRef, tuple[int, ...]],
    ) -> None:
        """Add the binary circuit for one representative bag operation."""
        defn = self.problem.get_object(ref)

        def bits(item: ObjRef) -> tuple[_BooleanTerm, ...]:
            if item == orbit.root:
                width = orbit.multiplicity.bit_length()
                return self._constant_bits(orbit.multiplicity, width)
            return state_bits[item]

        if isinstance(defn, ir_obj.BagChoose):
            builder.add_clause(builder.leq(bits(ref), bits(defn.source)))
        elif isinstance(defn, ir_obj.BagIntersection):
            builder.constrain_min(bits(ref), bits(defn.left), bits(defn.right))
        else:
            raise TypeError(f"Unsupported lifted bag operation: {type(defn).__name__}")

    @staticmethod
    def _constant_bits(value: int, width: int) -> tuple[bool, ...]:
        return tuple(
            bool(value & (1 << bit))
            for bit in range(width)
        )
