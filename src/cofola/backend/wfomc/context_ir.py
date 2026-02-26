"""IR-native Context for encoding combinatorics problems to WFOMC.

This module provides ContextIR, which replaces the legacy Context class.
Instead of holding a CofolaProblem with legacy mutable objects, it holds
an immutable ir.Problem and ir.AnalysisResult, using ObjRef as dictionary keys.

IMPLEMENTATION NOTES FOR THE IMPLEMENTER
=========================================

This class mirrors context.py but with the following key differences:

1. **Keys**: Use ObjRef (or tuple[ObjRef, Entity]) instead of legacy object instances.
2. **Data source**: Read analysis data from self.analysis (SetInfo/BagInfo) instead
   of from object attributes (.dis_entities, .p_entities_multiplicity, etc.).
3. **Name resolution**: Use self.problem.get_name(ref) instead of obj.name.
4. **Sequence detection**: Scan problem.refs() for SequenceDef instead of checking
   isinstance(obj, Sequence) in problem.objects.
5. **pred creation**: Use create_cofola_pred(name, arity) from cofola.utils.
6. **var creation**: Use create_cofola_var(name) from cofola.utils.

TYPE ALIASES (for implementer convenience):
    IREntity = cofola.ir.types.Entity
    ObjRef   = cofola.ir.types.ObjRef

WFOMC types (from wfomc package):
    Pred, Const, Formula, AtomicFormula, Expr, Rational, WFOMCProblem
    fol_parse as parse, to_sc2, top
"""
from __future__ import annotations

from wfomc import (
    AtomicFormula,
    Const,
    Formula,
    Pred,
    Rational,
    WFOMCProblem,
    fol_parse as parse,
    to_sc2,
    top,
)
from wfomc import Expr  # type: ignore[attr-defined]

from loguru import logger

from cofola.utils import create_cofola_pred, create_cofola_var, create_aux_pred
from cofola.backend.wfomc.decoder import Decoder

from cofola.frontend.problem import Problem
from cofola.frontend.objects import SequenceDef, PartRef
from cofola.ir.analysis.entities import AnalysisResult
from cofola.frontend.types import ObjRef, Entity as IREntity


class ContextIR:
    """Context for encoding the new IR to a WFOMC problem.

    Holds all state built up during object/constraint encoding,
    then produces a (WFOMCProblem, Decoder) pair via build().

    Mirrors the legacy Context class (context.py) but uses ObjRef
    as dictionary keys and reads analysis data from AnalysisResult.

    Attributes:
        problem: The immutable IR Problem being encoded.
        analysis: Analysis result (SetInfo/BagInfo per ObjRef).
        singletons: Entities that are always singletons (from analysis).
        domain: WFOMC domain constants (one Const per entity).
        sentence: The accumulated FO sentence (conjunction of all sub-sentences).
        weighting: WFOMC weighting map (Pred → (pos_weight, neg_weight)).
        unary_evidence: Unary ground atoms (e.g., p_A(c_a)).
        overcount: Rational overcounting correction factor (for partitions etc.).
        validator: List of (Expr, Expr) tuples for Decoder validation.
        indis_vars: List of symbolic vars for indistinguishable-entity decoding.
        gen_vars: List of all generated symbolic vars (for Decoder).
        ref2pred: Maps ObjRef → WFOMC Pred (the predicate representing that object).
        ref_entity2pred: Maps (ObjRef, IREntity) → Pred (entity-specific predicates).
        ref2var: Maps ObjRef → symbolic Expr (for size constraints).
        ref_entity2var: Maps (ObjRef, IREntity) → symbolic Expr (bag entity vars).
        ref_mul2var: Maps (ObjRef, int) → symbolic Expr (indis-entity multiplicity vars).
        used_refs: Set of ObjRef whose predicates are actually used in the sentence.
        sequence_ref: ObjRef of the unique SequenceDef, or None if no sequence.
        leq_pred: Global 'LEQ' binary predicate (linear order).
        pred_pred: Global 'PRED' binary predicate (predecessor for linear sequences).
        circular_pred: Global 'CIRCULAR_PRED' binary predicate (predecessor for circles).
        circle_len: Number of elements in the domain (used for circular sequences).
    """

    def __init__(self, problem: Problem, analysis: AnalysisResult) -> None:
        """Initialise the encoding context.

        Args:
            problem: The immutable IR Problem to encode.
            analysis: The entity analysis result for this problem.
        """
        self.problem: Problem = problem
        self.analysis: AnalysisResult = analysis

        self.singletons: frozenset[IREntity] = frozenset(analysis.singletons)
        self.domain: set[Const] = {Const(e.name) for e in analysis.all_entities}

        # Accumulated WFOMC components
        self.sentence: Formula = top
        self.weighting: dict[Pred, tuple] = {}
        self.unary_evidence: set[AtomicFormula] = set()
        self.overcount: Rational = Rational(1, 1)
        self.validator: list = []
        self.indis_vars: list = []
        self.gen_vars: list = []

        # ObjRef → WFOMC Pred (main predicate for each object)
        self.ref2pred: dict[ObjRef, Pred] = {}
        # (ObjRef, IREntity) → Pred  (entity-specific predicates for bags/sequences)
        self.ref_entity2pred: dict[tuple[ObjRef, IREntity], Pred] = {}
        # ObjRef → symbolic Expr  (for size-constraint variables)
        self.ref2var: dict[ObjRef, Expr] = {}
        # (ObjRef, IREntity) → Expr  (per-entity multiplicity variables)
        self.ref_entity2var: dict[tuple[ObjRef, IREntity], Expr] = {}
        # (ObjRef, int multiplicity) → Expr  (indistinguishable-entity variables)
        self.ref_mul2var: dict[tuple[ObjRef, int], Expr] = {}
        # Set of ObjRef whose predicates are referenced in self.sentence
        self.used_refs: set[ObjRef] = set()

        # Find the unique SequenceDef, if any (at most one per problem after lowering)
        self.sequence_ref: ObjRef | None = self._find_sequence_ref()

        # Global linear-order / predecessor predicates (shared by all sequences)
        self.leq_pred: Pred = Pred('LEQ', 2)
        self.pred_pred: Pred = Pred('PRED', 2)
        self.circular_pred: Pred = Pred('CIRCULAR_PRED', 2)
        self.circle_len: int = len(self.domain)

        # Singletons predicate (set during _encode_singleton if singletons exist)
        self.singletons_pred: Pred | None = None

    # =========================================================================
    # Setup helpers
    # =========================================================================

    def _find_sequence_ref(self) -> ObjRef | None:
        """Return the ObjRef of the first SequenceDef in the problem, or None.

        IMPLEMENTATION: iterate self.problem.refs(), call self.problem.get_object(ref),
        check isinstance(defn, SequenceDef).
        """
        for ref in self.problem.refs():
            defn = self.problem.get_object(ref)
            if isinstance(defn, SequenceDef):
                return ref
        return None

    # =========================================================================
    # Pred management
    # =========================================================================

    def get_pred(
        self,
        ref: ObjRef,
        *,
        create: bool = False,
        use: bool = True,
    ) -> Pred:
        """Get (or optionally create) the WFOMC predicate for an object reference.

        The predicate name is derived from problem.get_name(ref) via _obj_pred_name().
        Arity is always 1 (unary predicate over the domain).

        Args:
            ref: The object reference.
            create: If True, create the predicate if it doesn't exist yet.
            use: If True, mark this ref as used (affects evidence pruning).

        Returns:
            The Pred for this object.

        Raises:
            ValueError: If not found and create=False.

        IMPLEMENTATION:
            - If ref not in self.ref2pred and create: create with _obj_pred_name(ref), arity=1
            - If ref not in self.ref2pred and not create: raise ValueError
            - If use: self.used_refs.add(ref)
            - Return self.ref2pred[ref]
        """
        if ref not in self.ref2pred:
            if create:
                self.ref2pred[ref] = create_cofola_pred(self._obj_pred_name(ref), 1)
            else:
                raise ValueError(f"No predicate for ref {ref} (id={ref.id})")
        if use:
            self.used_refs.add(ref)
        return self.ref2pred[ref]

    def set_pred(self, ref: ObjRef, pred: Pred) -> None:
        """Explicitly assign a Pred to a ref (used when encoder creates preds directly).

        Args:
            ref: The object reference.
            pred: The predicate to assign.
        """
        self.ref2pred[ref] = pred

    def get_entity_pred(self, ref: ObjRef, entity: IREntity) -> Pred:
        """Get (or create) the entity-specific predicate for (ref, entity).

        Used for bag encoding where each entity in a bag gets its own predicate.
        Name: '<obj_name>_<entity_name>' with cofola prefix.

        Args:
            ref: The bag object reference.
            entity: The entity.

        Returns:
            The entity-specific Pred.

        IMPLEMENTATION:
            key = (ref, entity)
            If not in ref_entity2pred: create Pred with name _entity_pred_name(ref, entity), arity=1
            Return ref_entity2pred[key]
        """
        key = (ref, entity)
        if key not in self.ref_entity2pred:
            self.ref_entity2pred[key] = create_cofola_pred(
                self._entity_pred_name(ref, entity), 1
            )
        return self.ref_entity2pred[key]

    # =========================================================================
    # Var management
    # =========================================================================

    def get_obj_var(self, ref: ObjRef, *, set_weight: bool = True) -> Expr:
        """Get (or create) the symbolic variable for an object (for size constraints).

        If set_weight=True, also configures self.weighting[pred] = (var, 1).

        Args:
            ref: The object reference.
            set_weight: Whether to register the variable as a WFOMC weight.

        Returns:
            The symbolic Expr for this object's size variable.

        IMPLEMENTATION:
            If ref not in ref2var: create var with name from _obj_var_name(ref)
            If set_weight: pred = get_pred(ref); self.weighting[pred] = (ref2var[ref], 1)
            Return ref2var[ref]
        """
        if ref not in self.ref2var:
            self.ref2var[ref] = self.create_var(self._obj_var_name(ref))
        if set_weight:
            pred = self.get_pred(ref)
            self.weighting[pred] = (self.ref2var[ref], 1)
        return self.ref2var[ref]

    def get_entity_var(
        self,
        ref: ObjRef,
        entity: IREntity | None = None,
    ) -> Expr | dict[IREntity, Expr]:
        """Get (or create) the symbolic variable for an entity within a bag object.

        Used for bag multiplicity encoding. If entity is None, returns the full
        per-entity dict for this ref.

        Args:
            ref: The bag object reference.
            entity: The entity, or None to get the full dict.

        Returns:
            The symbolic Expr if entity is given, else dict[IREntity, Expr].

        IMPLEMENTATION:
            If entity is None: return {e: v for (r,e),v in ref_entity2var.items() if r==ref}
            key = (ref, entity)
            If not in ref_entity2var: create var with _entity_var_name(ref, entity)
            Return ref_entity2var[key]
        """
        if entity is None:
            return {e: v for (r, e), v in self.ref_entity2var.items() if r == ref}
        key = (ref, entity)
        if key not in self.ref_entity2var:
            self.ref_entity2var[key] = self.create_var(self._entity_var_name(ref, entity))
        return self.ref_entity2var[key]

    def get_indis_entity_var(
        self,
        ref: ObjRef,
        multiplicity: int | None = None,
    ) -> Expr | dict[int, Expr]:
        """Get (or create) the symbolic variable for indistinguishable entities.

        Used when multiple entities in a bag are indistinguishable (same multiplicity).
        If multiplicity is None, returns the full dict for this ref.

        Args:
            ref: The bag object reference.
            multiplicity: The shared multiplicity, or None for the full dict.

        Returns:
            The symbolic Expr if multiplicity given, else dict[int, Expr].

        IMPLEMENTATION:
            If multiplicity is None: return {m:v for (r,m),v in ref_mul2var.items() if r==ref}
            key = (ref, multiplicity)
            If not in ref_mul2var: create var with _indis_var_name(ref, multiplicity)
            Return ref_mul2var[key]
        """
        if multiplicity is None:
            return {m: v for (r, m), v in self.ref_mul2var.items() if r == ref}
        key = (ref, multiplicity)
        if key not in self.ref_mul2var:
            self.ref_mul2var[key] = self.create_var(self._indis_var_name(ref, multiplicity))
        return self.ref_mul2var[key]

    # =========================================================================
    # Sequence-specific predicates
    # =========================================================================

    def get_leq_pred(self, seq_ref: ObjRef) -> Pred:
        """Get the sequence-restricted LEQ predicate for a SequenceDef.

        Creates a new binary predicate `<seq_name>_LEQ` and adds a sentence:
            ∀X∀Y: (seq_LEQ(X,Y) ↔ (source_pred(X) ∧ source_pred(Y) ∧ LEQ(X,Y)))
        where source_pred is the predicate for the sequence's source.

        Args:
            seq_ref: ObjRef of the SequenceDef.

        Returns:
            The sequence-restricted LEQ Pred.

        IMPLEMENTATION:
            defn = problem.get_object(seq_ref)  # SequenceDef
            source_pred = get_pred(defn.source)
            leq_name = f"{_get_name(seq_ref)}_LEQ"
            leq_pred = create_pred(leq_name, 2)
            Add to self.sentence: ∀X∀Y: (leq_pred(X,Y) <-> (source_pred(X) & source_pred(Y) & self.leq_pred(X,Y)))
            Return leq_pred
        """
        defn = self.problem.get_object(seq_ref)
        source_pred = self.get_pred(defn.source)
        leq_name = f"{self._get_name(seq_ref)}_LEQ"
        leq_pred = self.create_pred(leq_name, 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({leq_pred}(X,Y) <-> "
            f"({source_pred}(X) & {source_pred}(Y) & {self.leq_pred}(X,Y))))"
        )
        return leq_pred

    def get_predecessor_pred(self, seq_ref: ObjRef) -> Pred:
        """Get the sequence-restricted PRED predicate for a SequenceDef.

        For circular sequences uses self.circular_pred; otherwise self.pred_pred.

        Creates:
            ∀X∀Y: (seq_PRED(X,Y) ↔ (source_pred(X) ∧ source_pred(Y) ∧ PRED(X,Y)))

        Args:
            seq_ref: ObjRef of the SequenceDef.

        Returns:
            The sequence-restricted predecessor Pred.

        IMPLEMENTATION:
            defn = problem.get_object(seq_ref)  # SequenceDef
            source_pred = get_pred(defn.source)
            pred_pred = self.circular_pred if defn.circular else self.pred_pred
            seq_pred_name = f"{_get_name(seq_ref)}_PRED"
            Create pred and add sentence as with get_leq_pred but using pred_pred.
        """
        defn = self.problem.get_object(seq_ref)
        source_pred = self.get_pred(defn.source)
        pred_pred = self.circular_pred if defn.circular else self.pred_pred
        seq_pred_name = f"{self._get_name(seq_ref)}_PRED"
        seq_pred = self.create_pred(seq_pred_name, 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({seq_pred}(X,Y) <-> "
            f"({source_pred}(X) & {source_pred}(Y) & {pred_pred}(X,Y))))"
        )
        return seq_pred

    def get_next_to_pred(self, seq_ref: ObjRef) -> Pred:
        """Get the NEXT_TO predicate for a SequenceDef.

        next_to(X,Y) ↔ pred(X,Y) ∨ pred(Y,X)

        Args:
            seq_ref: ObjRef of the SequenceDef.

        Returns:
            The next-to Pred.

        IMPLEMENTATION:
            pred_pred = get_predecessor_pred(seq_ref)
            next_to_pred = create_pred(f"{_get_name(seq_ref)}_NEXT_TO", 2)
            Add sentence: ∀X∀Y: (next_to(X,Y) <-> (pred(X,Y) | pred(Y,X)))
            Return next_to_pred
        """
        pred_pred = self.get_predecessor_pred(seq_ref)
        next_to_name = f"{self._get_name(seq_ref)}_NEXT_TO"
        next_to_pred = self.create_pred(next_to_name, 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({next_to_pred}(X,Y) <-> "
            f"({pred_pred}(X,Y) | {pred_pred}(Y,X))))"
        )
        return next_to_pred

    # =========================================================================
    # Utility
    # =========================================================================

    def get_parts_of(self, partition_ref: ObjRef) -> list[ObjRef]:
        """Return all PartRef ObjRefs belonging to a PartitionDef, sorted by index.

        Args:
            partition_ref: ObjRef of the PartitionDef.

        Returns:
            List of PartRef ObjRefs in ascending index order.

        IMPLEMENTATION:
            from cofola.frontend.objects import PartRef
            parts = [(r, defn.index) for r in problem.refs()
                     if isinstance((defn := problem.get_object(r)), PartRef)
                     and defn.partition == partition_ref]
            Return [r for r, _ in sorted(parts, key=lambda x: x[1])]
        """
        parts = []
        for r in self.problem.refs():
            defn = self.problem.get_object(r)
            if isinstance(defn, PartRef) and defn.partition == partition_ref:
                parts.append((r, defn.index))
        return [r for r, _ in sorted(parts, key=lambda x: x[1])]

    def _get_name(self, ref: ObjRef) -> str:
        """Return the user name for ref, or a default 'obj_<id>'.

        Args:
            ref: The object reference.

        Returns:
            Name string.
        """
        return self.problem.get_name(ref) or f"obj_{ref.id}"

    def _obj_pred_name(self, ref: ObjRef) -> str:
        """Derive the WFOMC predicate name for an object reference.

        Format: '<obj_name>' (create_cofola_pred will add the 'p_' prefix).

        Args:
            ref: The object reference.

        Returns:
            Name string for the predicate.
        """
        return self._get_name(ref)

    def _entity_pred_name(self, ref: ObjRef, entity: IREntity) -> str:
        """Derive the entity-specific predicate name.

        Format: '<obj_name>_<entity_name>'

        Args:
            ref: The bag object reference.
            entity: The entity.

        Returns:
            Name string.
        """
        return f"{self._get_name(ref)}_{entity.name}"

    def _obj_var_name(self, ref: ObjRef) -> str:
        """Derive the symbolic variable name for an object.

        Format: '<obj_name>' (create_cofola_var will add the 'v_' prefix).

        Args:
            ref: The object reference.

        Returns:
            Name string.
        """
        return self._get_name(ref)

    def _entity_var_name(self, ref: ObjRef, entity: IREntity) -> str:
        """Derive the per-entity variable name for a bag object.

        Format: '<obj_name>_<entity_name>'

        Args:
            ref: The bag object reference.
            entity: The entity.

        Returns:
            Name string.
        """
        return f"{self._get_name(ref)}_{entity.name}"

    def _indis_var_name(self, ref: ObjRef, multiplicity: int) -> str:
        """Derive the indistinguishable-entity variable name.

        Format: '<obj_name>#<multiplicity>'

        Args:
            ref: The bag object reference.
            multiplicity: The shared multiplicity.

        Returns:
            Name string.
        """
        return f"{self._get_name(ref)}#{multiplicity}"

    def create_pred(self, name: str, arity: int) -> Pred:
        """Create a new WFOMC predicate with the cofola naming convention.

        Args:
            name: The predicate name (will have 'p_' prepended if not present).
            arity: The predicate arity.

        Returns:
            A new Pred.
        """
        return create_cofola_pred(name, arity)

    def create_aux_pred(self, arity: int) -> Pred:
        """Create a unique auxiliary predicate.

        Args:
            arity: The predicate arity.

        Returns:
            A new uniquely-named auxiliary Pred.
        """
        return create_aux_pred(arity)

    def create_var(self, name: str, *, use_gen: bool = True) -> Expr:
        """Create a new symbolic variable with the cofola naming convention.

        Args:
            name: The variable name (will have 'v_' prepended if not present).
            use_gen: If True, append to self.gen_vars (for Decoder).

        Returns:
            A new symbolic Expr.
        """
        var = create_cofola_var(name)
        if use_gen:
            self.gen_vars.append(var)
        return var

    def prune_evidence(self) -> None:
        """Remove unary evidence atoms whose predicates are not in the sentence.

        Should be called during build() before constructing WFOMCProblem.

        IMPLEMENTATION:
            used_preds = self.sentence.preds()
            self.unary_evidence = {e for e in self.unary_evidence if e.pred in used_preds}
        """
        used_preds = self.sentence.preds()
        self.unary_evidence = {e for e in self.unary_evidence if e.pred in used_preds}

    def build(self) -> tuple[WFOMCProblem, Decoder]:
        """Finalise encoding and return the (WFOMCProblem, Decoder) pair.

        Steps:
        1. Convert sentence to SC2 form: self.sentence = to_sc2(self.sentence)
        2. Prune unused evidence: self.prune_evidence()
        3. Rename domain constants with 'c_' prefix (WFOMC convention)
        4. Rename evidence atom arguments with 'c_' prefix
        5. Construct WFOMCProblem(sentence, new_domain, weighting,
                                   unary_evidence=new_evidence,
                                   circle_len=self.circle_len)
        6. Construct Decoder(self.overcount, self.gen_vars,
                              self.validator, self.indis_vars)
        7. Return (wfomc_problem, decoder)

        Returns:
            Tuple of (WFOMCProblem, Decoder).
        """
        logger.debug(
            "ContextIR.build: sentence (pre-sc2):\n  {}",
            self.sentence,
        )
        logger.debug(
            "ContextIR.build: domain: {}",
            sorted(str(c) for c in self.domain),
        )
        logger.debug(
            "ContextIR.build: weighting:\n  {}",
            {str(p): (str(w[0]), str(w[1])) for p, w in self.weighting.items()},
        )
        logger.debug(
            "ContextIR.build: evidence:\n  {}",
            sorted(str(a) for a in self.unary_evidence),
        )
        self.sentence = to_sc2(self.sentence)
        logger.debug(
            "ContextIR.build: sentence (post-sc2):\n  {}",
            self.sentence,
        )
        self.prune_evidence()
        logger.debug(
            "ContextIR.build: evidence (post-prune):\n  {}",
            sorted(str(a) for a in self.unary_evidence),
        )
        new_domain = set()
        new_unary_evidence = set()
        for const in self.domain:
            new_domain.add(Const(f"c_{const.name}"))
        for atom in self.unary_evidence:
            new_unary_evidence.add(
                AtomicFormula(atom.pred, (Const(f"c_{atom.args[0].name}"),), atom.positive)
            )
        logger.debug(
            "ContextIR.build: WFOMCProblem:\n"
            "  sentence: {}\n"
            "  domain: {}\n"
            "  weighting: {}\n"
            "  evidence: {}\n"
            "  circle_len: {}",
            self.sentence,
            sorted(str(c) for c in new_domain),
            {str(p): (str(w[0]), str(w[1])) for p, w in self.weighting.items()},
            sorted(str(a) for a in new_unary_evidence),
            self.circle_len,
        )
        wfomc_problem = WFOMCProblem(
            self.sentence,
            new_domain,
            self.weighting,
            unary_evidence=new_unary_evidence,
            circle_len=self.circle_len,
        )
        decoder = Decoder(
            self.overcount,
            self.gen_vars,
            self.validator,
            self.indis_vars,
        )
        logger.debug(
            "ContextIR.build: Decoder — overcount={}, gen_vars={}, validator={}, indis_vars={}",
            self.overcount,
            self.gen_vars,
            self.validator,
            self.indis_vars,
        )
        return wfomc_problem, decoder
