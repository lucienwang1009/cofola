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
)
from cofola.frontend.constraints import (
    SizeConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    MembershipConstraint,
    FuncPairConstraint,
    TupleCountAtom,
)
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult


# Prefix for generated index entities
IDX_PREFIX = "idx_"


class LoweringPass:
    """Lowers high-level constructs to primitive objects.

    This pass transforms:
    - TupleDef → indices SetInit + FuncDef(indices → source)
    - SequenceDef → various transformations based on properties
    - FuncDef(injective=True) → FuncDef + SizeConstraint

    The lowering process creates new objects and constraints, and updates
    existing constraints to reference the new objects.

    Ports the legacy transform function to work with the new IR.
    """

    def __init__(self) -> None:
        self._next_id: int = 10000  # Start high to avoid collision
        self._tuple_to_indices: dict[ObjRef, ObjRef] = {}  # tuple ref -> indices ref
        self._tuple_to_mapping: dict[ObjRef, ObjRef] = {}  # tuple ref -> mapping ref
        self._tuple_choose: dict[ObjRef, bool] = {}  # tuple ref -> choose flag

    def run(self, problem: Problem, analysis: AnalysisResult) -> Problem:
        """Run lowering on a Problem.

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            A new Problem with high-level constructs lowered.
        """
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
        # Try lowering in order: tuples, sequences, functions
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

    def _try_lower_tuples(
        self, problem: Problem, analysis: AnalysisResult
    ) -> tuple[Problem, bool]:
        """Try to lower one TupleDef.

        TupleDef is lowered to:
        1. A SetInit of index entities (idx_0, idx_1, ...)
        2. A FuncDef mapping indices to the source

        Args:
            problem: The Problem to lower.
            analysis: Entity analysis result.

        Returns:
            Tuple of (new Problem, whether any changes were made).
        """
        # Find an unlowered TupleDef
        for ref in problem.refs():
            defn = problem.get_object(ref)
            if not isinstance(defn, TupleDef):
                continue

            # Check if already lowered (would have mapping info)
            if ref in self._tuple_to_mapping:
                continue

            # Lower this tuple
            source = defn.source
            source_defn = problem.get_object(source)
            size = defn.size

            # Get size from analysis if not specified
            if size is None:
                # Try to get from source
                if source in analysis.set_info:
                    size = analysis.set_info[source].max_size
                elif source in analysis.bag_info:
                    size = analysis.bag_info[source].max_size
                else:
                    # Can't determine size, skip
                    continue

            # Create index entities
            idx_entities = frozenset(
                Entity(f"{IDX_PREFIX}{i}") for i in range(size)
            )
            indices_ref = self._new_ref()
            indices_defn = SetInit(entities=idx_entities)

            # Create mapping function
            mapping_ref = self._new_ref()

            # Determine if injective/surjective based on tuple properties.
            # For bag sources: never injective — multiple indices can map to the same
            # entity (e.g., B×2 means idx_0 and idx_1 both map to B). Injectivity would
            # require |image| == |domain|, but |support| < |domain| when any mult > 1.
            # Multiplicity is already enforced via FuncInverseImage constraints below.
            is_bag_source = isinstance(source_defn, (BagInit, BagChoose))
            injective = not defn.replace and not is_bag_source
            surjective = not defn.choose  # If not choosing, must cover all elements

            # For bags, need to go through support
            codomain = source
            new_constraints = list(problem.constraints)

            if source_defn is not None and isinstance(source_defn, (BagInit, BagChoose)):
                # Need to create BagSupport first
                support_ref = self._new_ref()
                support_defn = BagSupport(source=source)
                codomain = support_ref

                # Add support to definitions
                new_defs = list(problem.defs)
                new_defs.append((support_ref, support_defn))

                # For bags, we need to create inverse images with multiplicity constraints
                # to ensure each entity appears the correct number of times
                if source in analysis.bag_info:
                    bag_info = analysis.bag_info[source]
                    singletons_in_bag = set()
                    inverse_image_refs = []

                    for entity, multiplicity in bag_info.p_entities_multiplicity.items():
                        if entity in analysis.singletons:
                            singletons_in_bag.add(entity)
                        else:
                            # Create FuncInverseImage for this entity
                            inv_img_ref = self._new_ref()
                            inv_img_defn = FuncInverseImage(func=mapping_ref, argument=entity)
                            new_defs.append((inv_img_ref, inv_img_defn))
                            inverse_image_refs.append(inv_img_ref)

                            # Add multiplicity constraint: |inverse_image| == multiplicity
                            mult_constraint = SizeConstraint(
                                terms=((inv_img_ref, 1),),
                                comparator="==",
                                rhs=multiplicity,
                            )
                            new_constraints.append(mult_constraint)

                    # Add disjoint constraints between all pairs of inverse images
                    for i, ref_i in enumerate(inverse_image_refs):
                        for j, ref_j in enumerate(inverse_image_refs):
                            if i < j:
                                disj_constraint = DisjointConstraint(
                                    left=ref_i,
                                    right=ref_j,
                                    positive=True,
                                )
                                new_constraints.append(disj_constraint)

                    # For singletons, create FuncImage with SetEqConstraint
                    if singletons_in_bag:
                        # For now, we'll handle singletons by adding them to the constraints
                        # Create FuncImage and add SetEqConstraint
                        pass  # Singletons are handled separately in the encoder

                problem = Problem(
                    defs=tuple(new_defs),
                    constraints=tuple(new_constraints),
                    names=problem.names,
                )
                new_defs = list(problem.defs)
                new_constraints = list(problem.constraints)

            mapping_defn = FuncDef(
                domain=indices_ref,
                codomain=codomain,
                injective=injective,
                surjective=surjective if defn.choose else False,
            )

            # Store mappings
            self._tuple_to_indices[ref] = indices_ref
            self._tuple_to_mapping[ref] = mapping_ref
            self._tuple_choose[ref] = defn.choose

            # Add new objects
            new_defs = list(problem.defs)
            new_defs.append((indices_ref, indices_defn))
            new_defs.append((mapping_ref, mapping_defn))

            # Remove the TupleDef
            new_defs = [(r, d) for r, d in new_defs if r != ref]

            logger.info(f"Lowered TupleDef {ref} to indices {indices_ref} and mapping {mapping_ref}")

            # Lower constraints that reference this TupleDef
            new_constraints, extra_defs = self._lower_tuple_constraints(
                tuple(new_constraints), ref, mapping_ref, indices_ref
            )
            new_defs.extend(extra_defs)

            return Problem(
                defs=tuple(new_defs),
                constraints=new_constraints,
                names=problem.names,
            ), True

        return problem, False

    def _lower_tuple_constraints(
        self,
        constraints: tuple,
        tuple_ref: ObjRef,
        mapping_ref: ObjRef,
        indices_ref: ObjRef,
    ) -> tuple[tuple, list]:
        """Lower constraints that reference a just-lowered TupleDef.

        Replaces:
        - TupleIndexEq(tuple_ref=T, index=i, entity=e) →
            FuncPairConstraint(mapping, Entity("idx_i"), e)
        - MembershipConstraint(entity=e, container=T) →
            MembershipConstraint(entity=e, container=FuncImage(mapping, indices))
        - TupleIndexMembership(tuple_ref=T, index=i, container=C) →
            FuncPairConstraint(mapping, Entity("idx_i"), C)

        Returns:
            (new_constraints, extra_defs_to_add)
        """
        new_constraints = list(constraints)
        extra_defs: list = []
        image_ref = None  # lazily created FuncImage ref

        for i, c in enumerate(new_constraints):
            if isinstance(c, TupleIndexEq) and c.tuple_ref == tuple_ref:
                idx_entity = Entity(f"{IDX_PREFIX}{c.index}")
                new_constraints[i] = FuncPairConstraint(
                    func=mapping_ref,
                    arg_entity=idx_entity,
                    result=c.entity,
                    positive=c.positive,
                )
            elif isinstance(c, MembershipConstraint) and c.container == tuple_ref:
                if image_ref is None:
                    image_ref = self._new_ref()
                    extra_defs.append((image_ref, FuncImage(func=mapping_ref, argument=indices_ref)))
                new_constraints[i] = MembershipConstraint(
                    entity=c.entity,
                    container=image_ref,
                    positive=c.positive,
                )
            elif isinstance(c, TupleIndexMembership) and c.tuple_ref == tuple_ref:
                idx_entity = Entity(f"{IDX_PREFIX}{c.index}")
                new_constraints[i] = FuncPairConstraint(
                    func=mapping_ref,
                    arg_entity=idx_entity,
                    result=c.container,
                    positive=c.positive,
                )

        return tuple(new_constraints), extra_defs

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

            source_defn = problem.get_object(defn.source)

            # Case 1: choose without replace → convert to choose first
            if defn.choose and not defn.replace:
                if isinstance(source_defn, (SetInit, SetChoose)):
                    chosen_ref = self._new_ref()
                    chosen_defn = SetChoose(
                        source=defn.source,
                        size=defn.size,
                    )
                else:
                    # Bag source
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
            img_ref = self._new_ref()
            img_defn = FuncImage(func=ref, argument=defn.domain)

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
                if tuple_ref not in self._tuple_to_mapping:
                    # Tuple not yet lowered — keep as-is for next round
                    new_terms.append((atom, coef))
                    continue

                mapping_ref = self._tuple_to_mapping[tuple_ref]
                indices_ref = self._tuple_to_indices[tuple_ref]
                choose = self._tuple_choose.get(tuple_ref, True)

                if not choose and atom.deduplicate:
                    # SetIntersection(mapping.codomain, count_obj)
                    mapping_defn = problem.get_object(mapping_ref)
                    codomain_ref = mapping_defn.codomain
                    new_ref = self._new_ref()
                    extra_defs.append((new_ref, SetIntersection(left=codomain_ref, right=atom.count_obj)))
                elif not atom.deduplicate:
                    # FuncInverseImage(mapping, count_obj)
                    new_ref = self._new_ref()
                    extra_defs.append((new_ref, FuncInverseImage(func=mapping_ref, argument=atom.count_obj)))
                else:
                    # SetIntersection(FuncImage(mapping, indices), count_obj)
                    img_ref = self._new_ref()
                    extra_defs.append((img_ref, FuncImage(func=mapping_ref, argument=indices_ref)))
                    new_ref = self._new_ref()
                    extra_defs.append((new_ref, SetIntersection(left=img_ref, right=atom.count_obj)))

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