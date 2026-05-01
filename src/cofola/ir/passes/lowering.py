"""Lowering pass for the immutable IR.

This module implements LoweringPass, which lowers high-level constructs
to primitive objects that the WFOMC encoder can handle:

- TupleDef → FuncDef + SetInit(indices)
- SequenceDef → (various transformations)
- FuncDef with injective → FuncDef + SizeConstraint

Ports the legacy transform functions to work with the new IR.
"""

from __future__ import annotations

from loguru import logger

from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import (
    ObjDef,
    TupleDef,
    SequenceDef,
    FuncDef,
    SetInit,
    BagInit,
    BagChoose,
    BagSupport,
    FuncImage,
    FuncInverseImage,
    SetChoose,
    SetIntersection,
    BagObjDef,
    PartRef,
)
from cofola.frontend.constraints import (
    SizeConstraint,
    DisjointConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    MembershipConstraint,
    FuncPairConstraint,
    TupleCountAtom,
    BagCountAtom,
    ForAllParts,
)
from cofola.ir.pass_manager import TransformPass
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult
from cofola.ir.analysis.merged import MergedAnalysis
from cofola.ir.passes.optimize import UnsatisfiableConstraint


# Prefix for generated index entities
IDX_PREFIX = "idx_"


from dataclasses import dataclass as _dataclass, fields


@_dataclass
class _TupleInfo:
    """Lowering metadata for a single TupleDef."""
    mapping_ref: ObjRef
    indices_ref: ObjRef
    choose: bool


class LoweringPass(TransformPass):
    """Lowers high-level constructs to primitive objects.

    This pass transforms:
    - TupleDef → indices SetInit + FuncDef(indices → source)
    - SequenceDef → various transformations based on properties
    - FuncDef(injective=True) → FuncDef + SizeConstraint

    The lowering process creates new objects and constraints, and updates
    existing constraints to reference the new objects.
    """

    required_analyses = [MergedAnalysis]

    def __init__(self) -> None:
        self._next_id: int = 10000  # Start high to avoid collision
        self._tuple_info: dict[ObjRef, _TupleInfo] = {}  # tuple ref -> choose flag

    def run(self, problem: Problem, am=None) -> Problem:
        """Run lowering on a Problem.

        Args:
            problem: The Problem to lower.
            am: AnalysisManager for accessing MergedAnalysis result.

        Returns:
            A new Problem with high-level constructs lowered.
        """
        analysis: AnalysisResult = am.get(MergedAnalysis)

        logger.info("LoweringPass.run: {} objects before", len(problem.defs))
        current = problem
        changed = True
        iteration = 0

        while changed:
            iteration += 1
            logger.debug("LoweringPass iteration {}", iteration)
            current, changed = self._lower_once(current, analysis)

        logger.info("LoweringPass done: {} iterations, {} objects after",
                    iteration, len(current.defs))
        return current

    def _lower_once(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Run one round of lowering.

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        # Try lowering in order: ForAllParts, tuples, sequences, functions
        result, changed = self._try_lower_for_all_parts(problem)
        if changed:
            return result, True

        result, changed = self._try_lower_tuples(problem, analysis)
        if changed:
            return result, True

        result, changed = self._try_lower_sequences(problem, analysis)
        if changed:
            return result, True

        result, changed = self._try_lower_functions(problem, analysis)
        if changed:
            return result, True

        result, changed = self._try_lower_size_constraints(problem, analysis)
        if changed:
            return result, True

        return problem, False

    def _new_ref(self) -> ObjRef:
        """Create a new unique reference."""
        ref = ObjRef(self._next_id)
        self._next_id += 1
        return ref

    @staticmethod
    def _sentinel_dep_defs(
        problem: Problem, sentinel_ref: ObjRef
    ) -> list[tuple[ObjRef, object]]:
        """Return all defs (in topological order) that transitively reference sentinel_ref.

        The sentinel itself is NOT included.  Defs are ordered so that
        dependencies come before dependents — safe for sequential cloning.
        """
        sentinel_dep_set: set[ObjRef] = {sentinel_ref}
        ordered: list[tuple[ObjRef, object]] = []
        for ref, defn in problem.defs:
            if ref == sentinel_ref:
                continue
            dep_refs = set(problem.get_refs(defn))
            if dep_refs & sentinel_dep_set:
                sentinel_dep_set.add(ref)
                ordered.append((ref, defn))
        return ordered

    @staticmethod
    def _apply_ref_map_to_field(val: object, ref_map: dict) -> object:
        """Recursively replace ObjRefs in a field value using ref_map."""
        if isinstance(val, ObjRef):
            return ref_map.get(val, val)
        if isinstance(val, tuple):
            return tuple(LoweringPass._apply_ref_map_to_field(v, ref_map) for v in val)
        return val

    def _clone_sentinel_graph(
        self,
        problem: Problem,
        sentinel_dep_defs: list[tuple[ObjRef, object]],
        sentinel_ref: ObjRef,
        part_ref: ObjRef,
    ) -> tuple[list[tuple[ObjRef, object]], dict[ObjRef, ObjRef]]:
        """Clone sentinel-dependent defs for one real part.

        Returns:
            (new_defs_to_add, ref_map)  where ref_map maps sentinel and all
            cloned old refs to their per-part replacements.
        """
        ref_map: dict[ObjRef, ObjRef] = {sentinel_ref: part_ref}
        new_defs: list[tuple[ObjRef, object]] = []
        for old_ref, old_defn in sentinel_dep_defs:
            new_ref = self._new_ref()
            ref_map[old_ref] = new_ref
            new_fields = {
                f.name: self._apply_ref_map_to_field(getattr(old_defn, f.name), ref_map)
                for f in fields(old_defn)
            }
            new_defs.append((new_ref, type(old_defn)(**new_fields)))
        return new_defs, ref_map

    def _try_lower_for_all_parts(
        self, problem: Problem
    ) -> tuple[Problem, bool]:
        """Expand all ForAllParts constraints into concrete per-part constraints.

        Sentinel-dependent defs (e.g. SetIntersection(sentinel, S)) are cloned
        once per real part so each concrete constraint references its own
        independent copy.  Multiple ForAllParts sharing the same sentinel are all
        expanded in one pass before the sentinel and its deps are removed.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        for_all = [(i, c) for i, c in enumerate(problem.constraints) if isinstance(c, ForAllParts)]
        if not for_all:
            return problem, False

        # Build per-partition lookup tables from defs (done once).
        sentinels: dict[ObjRef, ObjRef] = {}
        real_parts: dict[ObjRef, list[tuple[ObjRef, int]]] = {}
        for ref, defn in problem.defs:
            if not isinstance(defn, PartRef):
                continue
            p = defn.partition
            if defn.index == -1:
                sentinels[p] = ref
            else:
                real_parts.setdefault(p, []).append((ref, defn.index))

        # Precompute sentinel-dependent def lists per partition (topological order).
        sentinel_dep_cache: dict[ObjRef, list] = {}
        for p, sentinel_ref in sentinels.items():
            sentinel_dep_cache[p] = self._sentinel_dep_defs(problem, sentinel_ref)

        kept_constraints: list = []
        new_constraints: list = []
        extra_defs: list[tuple[ObjRef, object]] = []
        refs_to_remove: set[ObjRef] = set()  # sentinels + their deps

        for c in problem.constraints:
            if not isinstance(c, ForAllParts):
                kept_constraints.append(c)
                continue

            p = c.partition
            sentinel_ref = sentinels.get(p)
            parts = real_parts.get(p, [])

            if sentinel_ref is None:
                raise ValueError(
                    f"ForAllParts: no sentinel PartRef (index=-1) found for partition {p}"
                )
            if not parts:
                raise ValueError(
                    f"ForAllParts: no real parts found for partition {p}"
                )

            dep_defs = sentinel_dep_cache[p]
            refs_to_remove.add(sentinel_ref)
            refs_to_remove.update(old_ref for old_ref, _ in dep_defs)

            parts_sorted = sorted(parts, key=lambda x: x[1])
            for part_ref, _ in parts_sorted:
                cloned_defs, ref_map = self._clone_sentinel_graph(
                    problem, dep_defs, sentinel_ref, part_ref
                )
                extra_defs.extend(cloned_defs)
                # Apply ref_map to the constraint template
                concrete = c.constraint_template
                for old_r, new_r in ref_map.items():
                    concrete = problem._sub_constraint(concrete, old_r, new_r)
                new_constraints.append(concrete)

        logger.info(
            "LoweringPass: expanded {} ForAllParts → {} concrete constraints",
            len(for_all), len(new_constraints),
        )

        base_defs = [(r, d) for r, d in problem.defs if r not in refs_to_remove]
        return Problem(
            defs=tuple(base_defs) + tuple(extra_defs),
            constraints=tuple(kept_constraints + new_constraints),
            names=problem.names,
        ), True

    def _is_bag_like(self, source_defn: ObjDef | None, problem: Problem) -> bool:
        """Return True if source_defn is a bag-like object.

        A BagObjDef is directly bag-like.
        A PartRef is bag-like iff its partition's source is a BagObjDef.
        """
        if isinstance(source_defn, BagObjDef):
            return True
        if isinstance(source_defn, PartRef):
            partition_defn = problem.get_object(source_defn.partition)
            if partition_defn is not None:
                partition_source_defn = problem.get_object(partition_defn.source)
                return isinstance(partition_source_defn, BagObjDef)
        return False

    def _try_lower_tuples(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Try to lower one TupleDef.

        Lowering strategy:
        - choose=True  + Bag source → split into BagChoose + TupleDef(choose=False)
          (Bug 1 fix: explicit decomposition so next iteration handles choose=False)
        - choose=False + Bag source → FuncDef(surjective=True) + FuncInverseImage per entity
          with exact multiplicity constraints (== not <=) and no singleton special-casing
          (Bug 2 fix)
          - BagInit source: fixed integer RHS
          - any other Bag source (e.g. BagChoose): BagCountAtom RHS for dynamic multiplicity
          (Bug 3 fix)
        - Set source (choose or not) → FuncDef(injective/surjective per choose/replace flags)

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if not isinstance(defn, TupleDef):
                continue
            if ref in self._tuple_info:
                continue

            source = defn.source
            source_defn = problem.get_object(source)
            size = defn.size

            # Resolve size: EntityAnalysis propagates source.exact_size into
            # set_info/bag_info[ref], and MaxSizeInference may refine it further
            # via |T| == k constraints.
            if size is None:
                own_info = analysis.set_info.get(ref) or analysis.bag_info.get(ref)
                if own_info is not None and own_info.exact_size is not None:
                    size = own_info.exact_size
                else:
                    raise ValueError(
                        f"TupleDef {ref.id}: tuple size must be specified explicitly "
                        f"or implied by its source (or a size constraint on the tuple itself)"
                    )

            # is_bag_source: must check the SOURCE type, not the tuple itself — a tuple with
            # replace=True from a set source has bag_info[ref] but must still take the set path.
            is_bag_source = self._is_bag_like(source_defn, problem)
            new_defs = list(problem.defs)
            new_constraints = list(problem.constraints)

            # ── choose=True + Bag → BagChoose + TupleDef(choose=False) ──────
            if defn.choose and is_bag_source:
                chosen_ref = self._new_ref()
                new_defs.append((chosen_ref, BagChoose(source=source, size=size)))
                new_tuple_defn = TupleDef(
                    source=chosen_ref, choose=False, replace=False, size=size
                )
                new_defs = [
                    (r, d) if r != ref else (ref, new_tuple_defn)
                    for r, d in new_defs
                ]
                logger.info(
                    "LoweringPass: TupleDef {} choose+bag → BagChoose {} + non-choose tuple",
                    ref.id, chosen_ref.id,
                )
                return Problem(
                    defs=tuple(new_defs),
                    constraints=tuple(new_constraints),
                    names=problem.names,
                ), True

            # ── Create shared index set ────────────────────────────────────────────
            idx_entities = frozenset(Entity(f"{IDX_PREFIX}{i}") for i in range(size))
            indices_ref = self._new_ref()
            indices_defn = SetInit(entities=idx_entities)
            mapping_ref = self._new_ref()

            if is_bag_source:
                # ── choose=False + Bag ──────────────────────────────────────
                support_ref = self._new_ref()
                new_defs.append((support_ref, BagSupport(source=source)))

                # surjective=True: every element of the support must be hit
                mapping_defn = FuncDef(
                    domain=indices_ref,
                    codomain=support_ref,
                    injective=False,
                    surjective=True,
                )

                # Use the tuple's own bag_info (populated from the original source at
                # analysis time), which remains valid even when source was replaced by
                # a freshly-created BagChoose in a prior lowering iteration.
                bag_info = analysis.bag_info.get(ref)
                if bag_info is None:
                    raise ValueError(
                        f"TupleDef {ref.id}: no bag_info for tuple ref {ref.id}; "
                        f"entity analysis must run before lowering"
                    )
                inv_img_refs: list[ObjRef] = []
                for entity, max_mult in bag_info.p_entities_multiplicity.items():
                    if entity in analysis.singletons:
                        continue
                    inv_img_ref = self._new_ref()
                    new_defs.append(
                        (inv_img_ref, FuncInverseImage(func=mapping_ref, argument=entity))
                    )
                    inv_img_refs.append(inv_img_ref)

                    # BagInit → fixed int; derived bags → BagCountAtom
                    if isinstance(source_defn, BagInit):
                        mult_constraint = SizeConstraint(
                            terms=((inv_img_ref, 1),),
                            comparator="==",
                            rhs=max_mult,
                        )
                    else:
                        mult_constraint = SizeConstraint(
                            terms=(
                                (inv_img_ref, 1),
                                (BagCountAtom(bag=source, entity=entity), -1),
                            ),
                            comparator="==",
                            rhs=0,
                        )
                    new_constraints.append(mult_constraint)

                # Disjoint constraints between every pair of inverse images
                for i, ref_i in enumerate(inv_img_refs):
                    for j, ref_j in enumerate(inv_img_refs):
                        if i < j:
                            new_constraints.append(
                                DisjointConstraint(left=ref_i, right=ref_j, positive=True)
                            )

            else:
                # ── Set source (SetObjDef) ─────────────────────────────────────────
                injective = not defn.replace
                surjective = not defn.choose
                if surjective:
                    injective = False  # surjective + full coverage → can't be injective
                mapping_defn = FuncDef(
                    domain=indices_ref,
                    codomain=source,
                    injective=injective,
                    surjective=surjective,
                )

            # Store lowering metadata
            self._tuple_info[ref] = _TupleInfo(
                mapping_ref=mapping_ref,
                indices_ref=indices_ref,
                choose=defn.choose,
            )

            # Add index set and mapping; remove the TupleDef
            new_defs.append((indices_ref, indices_defn))
            new_defs.append((mapping_ref, mapping_defn))
            new_defs = [(r, d) for r, d in new_defs if r != ref]

            # Rewrite constraints referencing this TupleDef
            new_constraints, extra_defs = self._lower_tuple_constraints(
                tuple(new_constraints), ref, mapping_ref, indices_ref, size
            )
            new_defs.extend(extra_defs)

            logger.info(
                "LoweringPass: TupleDef {} → indices {} + mapping {}",
                ref.id, indices_ref.id, mapping_ref.id,
            )
            return Problem(
                defs=tuple(new_defs),
                constraints=new_constraints,
                names=problem.names,
            ), True

        return problem, False

    def _lower_one_constraint(
        self,
        c: object,
        tuple_ref: ObjRef,
        mapping_ref: ObjRef,
        indices_ref: ObjRef,
        size: int,
        extra_defs: list,
    ) -> object | None:
        """Lower a single atomic constraint that may reference tuple_ref.

        Handles TupleIndexEq, MembershipConstraint, TupleIndexMembership.
        Compound constraints (Or/And/Not) never reach LoweringPass — they are
        Shannon-expanded before LOCAL_PASSES run.

        Returns ``None`` for trivially-true constraints (out-of-range index with
        ``positive=False``); the caller drops these. An out-of-range index with
        ``positive=True`` is unsatisfiable and raises ``UnsatisfiableConstraint``.
        """
        if isinstance(c, TupleIndexEq) and c.tuple_ref == tuple_ref:
            if not 0 <= c.index < size:
                return self._handle_out_of_range_index(c, tuple_ref, size)
            idx_entity = Entity(f"{IDX_PREFIX}{c.index}")
            return FuncPairConstraint(
                func=mapping_ref,
                arg_entity=idx_entity,
                result=c.entity,
                positive=c.positive,
            )
        elif isinstance(c, MembershipConstraint) and c.container == tuple_ref:
            new_image_ref = self._new_ref()
            extra_defs.append((new_image_ref, FuncImage(func=mapping_ref, argument=indices_ref)))
            return MembershipConstraint(
                entity=c.entity,
                container=new_image_ref,
                positive=c.positive,
            )
        elif isinstance(c, TupleIndexMembership) and c.tuple_ref == tuple_ref:
            if not 0 <= c.index < size:
                return self._handle_out_of_range_index(c, tuple_ref, size)
            idx_entity = Entity(f"{IDX_PREFIX}{c.index}")
            return FuncPairConstraint(
                func=mapping_ref,
                arg_entity=idx_entity,
                result=c.container,
                positive=c.positive,
            )
        return c

    @staticmethod
    def _handle_out_of_range_index(c, tuple_ref: ObjRef, size: int) -> None:
        """Resolve a tuple-index constraint whose index is outside ``[0, size)``.

        Positive form is unsatisfiable; negative form is trivially true so we
        drop it by returning ``None``.
        """
        if c.positive:
            raise UnsatisfiableConstraint(
                f"Tuple {tuple_ref.id} index {c.index} out of range [0, {size}): "
                f"constraint {c} is unsatisfiable"
            )
        logger.info(
            "LoweringPass: dropping trivially-true tuple {} index {} (size {}) constraint",
            tuple_ref.id, c.index, size,
        )
        return None

    def _lower_tuple_constraints(
        self,
        constraints: tuple,
        tuple_ref: ObjRef,
        mapping_ref: ObjRef,
        indices_ref: ObjRef,
        size: int,
    ) -> tuple[tuple, list]:
        """Lower atomic constraints that reference a just-lowered TupleDef.

        Replaces:
        - TupleIndexEq(tuple_ref=T, index=i, entity=e) →
            FuncPairConstraint(mapping, Entity("idx_i"), e)
        - MembershipConstraint(entity=e, container=T) →
            MembershipConstraint(entity=e, container=FuncImage(mapping, indices))
        - TupleIndexMembership(tuple_ref=T, index=i, container=C) →
            FuncPairConstraint(mapping, Entity("idx_i"), C)

        Compound constraints (Or/And/Not) are never present here: LoweringPass
        runs inside LOCAL_PASSES, after Shannon expansion has flattened all
        compound constraints to atomic form.

        Duplicate FuncImage defs created here are later merged by
        MergeIdenticalObjects, which runs after LoweringPass in LOCAL_PASSES.

        Returns:
            (new_constraints, extra_defs_to_add)
        """
        extra_defs: list = []

        new_constraints = tuple(
            lowered for c in constraints
            if (lowered := self._lower_one_constraint(
                c, tuple_ref, mapping_ref, indices_ref, size, extra_defs
            )) is not None
        )
        return new_constraints, extra_defs

    def _try_lower_sequences(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Try to lower one SequenceDef.

        SequenceDef is lowered based on its properties:
        - choose + not replace → convert to SetChoose/BagChoose first
        - other cases → create flatten object

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        # Find a SequenceDef that needs lowering
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if not isinstance(defn, SequenceDef):
                continue

            # Reject choose-with-replacement from a bag source.
            # Check the SOURCE's bag_info (not the sequence's own, which would be BagInfo
            # even for set sources when replace=True).
            if defn.choose and defn.replace and analysis.bag_info.get(defn.source) is not None:
                raise ValueError(
                    f"SequenceDef {ref.id}: choose-with-replacement from a bag source is not supported."
                )

            # Case 1: choose without replace → convert to choose first
            if defn.choose and not defn.replace:
                if analysis.bag_info.get(ref) is None:
                    chosen_ref = self._new_ref()
                    chosen_defn = SetChoose(
                        source=defn.source,
                        size=defn.size,
                    )
                else:
                    # Bag source (or PartRef backed by a bag)
                    chosen_ref = self._new_ref()
                    chosen_defn = BagChoose(
                        source=defn.source,
                        size=defn.size,
                    )

                # Update the SequenceDef to not choose
                new_seq_defn = SequenceDef(
                    source=chosen_ref,
                    choose=False,
                    replace=False,
                    size=defn.size,
                    circular=defn.circular,
                    reflection=defn.reflection,
                )

                new_defs = list(problem.defs)
                new_defs.append((chosen_ref, chosen_defn))
                new_defs = [(r, d) if r != ref else (ref, new_seq_defn) for r, d in new_defs]

                logger.info(f"Lowered SequenceDef {ref}: added choose object {chosen_ref}")

                return Problem(
                    defs=tuple(new_defs),
                    constraints=problem.constraints,
                    names=problem.names,
                ), True

            # Case 2: bag-like sequence OR choose-with-replace → create position-index flatten domain.
            # A sequence is bag-like iff its own bag_info was populated by EntityAnalysis
            # (replace=True, or replace=False with bag source).
            seq_is_bag = analysis.bag_info.get(ref) is not None
            choose_replace = defn.choose and defn.replace
            if (seq_is_bag or choose_replace) and defn.flatten is None:
                size = defn.size
                if size is None:
                    own_info = analysis.set_info.get(ref) or analysis.bag_info.get(ref)
                    if own_info is not None and own_info.exact_size is not None:
                        size = own_info.exact_size
                    else:
                        raise ValueError(
                            f"SequenceDef {ref.id}: sequence size must be specified explicitly. "
                            "Use 'choose k sequence from <source>' with an explicit k, "
                            "or add a size constraint '|<seq>| == k' or '|<source>| == k'."
                        )
                idx_entities = frozenset(Entity(f"{IDX_PREFIX}{i}") for i in range(size))
                flatten_ref = self._new_ref()
                flatten_defn = SetInit(entities=idx_entities)
                new_seq_defn = SequenceDef(
                    source=defn.source,
                    choose=defn.choose,
                    replace=defn.replace,
                    size=size,
                    circular=defn.circular,
                    reflection=defn.reflection,
                    flatten=flatten_ref,
                )
                new_defs = list(problem.defs)
                new_defs.append((flatten_ref, flatten_defn))
                new_defs = [(r, d) if r != ref else (ref, new_seq_defn) for r, d in new_defs]
                logger.info(
                    f"Lowered SequenceDef {ref}: added flatten object {flatten_ref} "
                    f"with {size} position entities"
                )
                return Problem(
                    defs=tuple(new_defs),
                    constraints=problem.constraints,
                    names=problem.names,
                ), True

        return problem, False

    def _try_lower_functions(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Try to lower injective FuncDef.

        FuncDef with injective=True is converted to:
        - FuncDef with injective=False
        - SizeConstraint ensuring injectivity

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if not isinstance(defn, FuncDef):
                continue

            if not defn.injective:
                continue

            # Create new FuncDef without injective
            new_func_defn = FuncDef(
                domain=defn.domain,
                codomain=defn.codomain,
                injective=False,
                surjective=defn.surjective,
            )

            # Create FuncImage for the domain
            img_defn = FuncImage(func=ref, argument=defn.domain)
            img_ref = self._new_ref()

            # Create SizeConstraint for injectivity
            # |f(domain)| == |domain|
            domain_info = analysis.set_info.get(defn.domain)
            if domain_info is not None:
                constraint = SizeConstraint(
                    terms=((img_ref, 1),),
                    comparator="==",
                    rhs=domain_info.max_size,
                )
            else:
                constraint = SizeConstraint(
                    terms=((img_ref, 1), (defn.domain, -1)),
                    comparator="==",
                    rhs=0,
                )

            new_defs = list(problem.defs)
            new_defs = [(r, d) if r != ref else (ref, new_func_defn) for r, d in new_defs]
            new_defs.append((img_ref, img_defn))

            new_constraints = list(problem.constraints)
            new_constraints.append(constraint)

            logger.info(f"Lowered injective FuncDef {ref}: added size constraint")

            return Problem(
                defs=tuple(new_defs),
                constraints=tuple(new_constraints),
                names=problem.names,
            ), True

        return problem, False

    def _try_lower_size_constraints(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Try to lower size constraints involving TupleCountAtom.

        SizeConstraint terms containing TupleCountAtom are converted to
        FuncInverseImage or SetIntersection based on the tuple properties:

        - deduplicate=False → FuncInverseImage(mapping, count_obj)
        - deduplicate=True, choose=False → SetIntersection(codomain, count_obj)
        - deduplicate=True, choose=True → SetIntersection(FuncImage(mapping, indices), count_obj)

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        constraints = list(problem.constraints)
        defs = list(problem.defs)
        changed = False

        for i, c in enumerate(constraints):
            if not isinstance(c, SizeConstraint):
                continue

            new_terms = []
            constraint_changed = False
            extra_defs = []

            for atom, coef in c.terms:
                if not isinstance(atom, TupleCountAtom):
                    new_terms.append((atom, coef))
                    continue

                tuple_ref = atom.tuple_ref
                if tuple_ref not in self._tuple_info:
                    # Tuple not yet lowered — keep as-is for next round
                    new_terms.append((atom, coef))
                    continue

                info = self._tuple_info[tuple_ref]
                mapping_ref = info.mapping_ref
                indices_ref = info.indices_ref
                choose = info.choose

                if not choose and atom.deduplicate:
                    # SetIntersection(mapping.codomain, count_obj)
                    mapping_defn = problem.get_object(mapping_ref)
                    codomain_ref = mapping_defn.codomain
                    new_ref = self._new_ref()
                    extra_defs.append(
                        (new_ref, SetIntersection(left=codomain_ref, right=atom.count_obj))
                    )
                elif not atom.deduplicate:
                    # FuncInverseImage(mapping, count_obj)
                    new_ref = self._new_ref()
                    extra_defs.append((new_ref, FuncInverseImage(func=mapping_ref, argument=atom.count_obj)))
                else:
                    # SetIntersection(FuncImage(mapping, indices), count_obj)
                    img_ref = self._new_ref()
                    extra_defs.append(
                        (img_ref, FuncImage(func=mapping_ref, argument=indices_ref))
                    )
                    new_ref = self._new_ref()
                    extra_defs.append(
                        (new_ref, SetIntersection(left=img_ref, right=atom.count_obj))
                    )

                new_terms.append((new_ref, coef))
                constraint_changed = True

            if constraint_changed:
                defs.extend(extra_defs)
                constraints[i] = SizeConstraint(
                    terms=tuple(new_terms),
                    comparator=c.comparator,
                    rhs=c.rhs,
                )
                changed = True

        if changed:
            return Problem(
                defs=tuple(defs),
                constraints=tuple(constraints),
                names=problem.names,
            ), True

        return problem, False
