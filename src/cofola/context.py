
from typing import Union
from wfomc import CardinalityConstraint, Formula, AtomicFormula, top, Const, Pred, to_sc2, \
    WFOMCProblem, fol_parse as parse, Expr

from cofola.decoder import Decoder
from cofola.objects.base import Bag, Entity, Sequence
from cofola.problem import CofolaProblem
from cofola.utils import create_aux_pred, create_cofola_pred, create_pred_for_object, create_cofola_var

from cofola.objects.base import CombinatoricsObject


class Context(object):
    """
    Context for encoding the combinatorics problem to the WFOMC problem
    """
    def __init__(self, problem: CofolaProblem,
                 old_pred_encoding: bool = False):
        self.problem: CofolaProblem = problem
        self.old_pred_encoding: bool = old_pred_encoding
        self.singletons: set[Entity] = problem.singletons
        # components of the WFOMC problem
        self.domain: set[Const] = set(
            Const(entity.name) for entity in problem.entities
        )
        self.sentence: Formula = top
        self.weighting: dict[Pred, tuple] = dict()
        self.unary_evidence: set[AtomicFormula] = set()

        # auxiliary variables
        self.obj2pred: dict[CombinatoricsObject, Pred] = dict()
        self.obj_entity2pred: dict[CombinatoricsObject, dict[Entity, Pred]] = dict()
        self.overcount: int = 1
        self.used_objs: set[CombinatoricsObject] = set()

        # for size constraint
        self.obj2var: dict[CombinatoricsObject, Expr] = dict()
        # for encoding bags
        self.entity2var: dict[CombinatoricsObject, dict[Entity, Expr]] = dict()
        # for encoding indistinguishable entities with the same multiplicity
        self.mul2var: dict[CombinatoricsObject, dict[int, Expr]] = dict()
        self.validator = list()
        # for deduplicate the encoding of partition
        self.indis_vars = list()
        self.gen_vars = list()

        # for encoding sequence
        # the only possible sequence object
        self.sequence_obj = None
        for obj in problem.objects:
            if isinstance(obj, Sequence):
                self.sequence_obj = obj
                break
        self.leq_pred = Pred('LEQ', 2)
        self.pred_pred = None
        self.circular_pred = None
        self.circle_len: int = len(self.domain)

        self.cardinality_constraint = CardinalityConstraint()

    def build(self) -> tuple[WFOMCProblem, Decoder]:
        # do some post encoding work
        self.sentence = to_sc2(self.sentence)
        self.prune_evidence()
        new_domain = set()
        new_unary_evidence = set()
        for const in self.domain:
            new_domain.add(Const(f"c_{const.name}"))
        for atom in self.unary_evidence:
            new_unary_evidence.add(
                AtomicFormula(atom.pred,
                              (Const(f"c_{atom.args[0].name}"), ),
                              atom.positive)
            )
        problem = WFOMCProblem(
            self.sentence,
            new_domain,
            self.weighting,
            unary_evidence=new_unary_evidence,
            circle_len=self.circle_len,
            cardinality_constraint=self.cardinality_constraint
        )
        decoder = Decoder(
            self.overcount,
            self.gen_vars,
            self.validator,
            self.indis_vars
        )
        return problem, decoder

    def prune_evidence(self):
        # prune the unused evidence
        # used_preds = set(pred for obj, pred in self.obj2pred.items()
        #                  if obj in self.used_objs)
        used_preds = self.sentence.preds()
        self.unary_evidence = set(
            evi for evi in self.unary_evidence if evi.pred in used_preds
        )

    def get_pred(self, obj: Union["CombinatoricsObject", set],
                 create: bool = False,
                 use: bool = True) -> Pred:
        """
        Get or create the predicate for the object (entity) or the set of objects (entities).
        If use is set to False, which means the predicate is not used, and thus all its evidence will be pruned

        :param obj: the object or the set of objects (for lifting bags)
        :param create: whether to create the predicate if not found
        :param use: whether to use the predicate
        """
        if isinstance(obj, set):
            obj = frozenset(obj)
        if obj not in self.obj2pred:
            if create:
                if isinstance(obj, (CombinatoricsObject, Entity)):
                    pred = create_pred_for_object(obj)
                else:
                    pred = create_aux_pred(1, 'entities')
                self.obj2pred[obj] = pred
            else:
                raise ValueError(f'Object {obj.name} not found')
        else:
            pred = self.obj2pred[obj]
        if use:
            self.used_objs.add(obj)
        return pred

    def get_entity_pred(self, obj: "CombinatoricsObject",
                        entity: Entity) -> Pred:
        """
        Get the predicate for the entity in the object

        :param obj: the object
        :param entity: the entity
        :return: the predicate
        """
        if obj not in self.obj_entity2pred:
            self.obj_entity2pred[obj] = dict()
        obj_entity_preds = self.obj_entity2pred[obj]
        if entity not in obj_entity_preds:
            obj_entity_preds[entity] = self.create_pred(f'{obj.name}_{entity.name}', 1)
        return obj_entity_preds[entity]

    def get_obj_var(self, obj: "CombinatoricsObject",
                    set_weight: bool = True) -> Expr:
        """
        Get a symbolic variable for the object used to encode size constraint
        :param obj: the object
        :param set_weight: whether to set the weight for the object
        :return: the symbolic variable
        """
        if obj not in self.obj2var:
            self.obj2var[obj] = self.create_var(f"{obj.name}")
        if set_weight:
            pred = self.get_pred(obj)
            self.weighting[pred] = (self.obj2var[obj], 1)
        return self.obj2var[obj]

    def get_entity_var(self, obj: Bag, entity: Entity = None) -> Expr:
        """
        Get a symbolic variable for the entity in the bag used to encode the multiplicity constraint or the size constraint

        :param obj: the bag object
        :param entity: the entity in the bag
        :return: the symbolic variable
        """
        if obj not in self.entity2var:
            self.entity2var[obj] = dict()
        obj_entities = self.entity2var[obj]
        if entity is None:
            return obj_entities
        if entity not in obj_entities:
            obj_entities[entity] = self.create_var(f'{obj.name}_{entity.name}')
        return obj_entities[entity]

    def get_indis_entity_var(self, obj: Bag, multiplicity: int = None) \
            -> Expr:
        """
        Get a symbolic variable for the indistinguishable entities in the bag

        :param obj: the bag object
        :param multiplicity: the multiplicity
        :return: the symbolic variable
        """
        if obj not in self.mul2var:
            self.mul2var[obj] = dict()
        obj_multiplicities = self.mul2var[obj]
        if multiplicity is None:
            return obj_multiplicities
        if multiplicity not in obj_multiplicities:
            obj_multiplicities[multiplicity] = self.create_var(f'{obj.name}#{multiplicity}')
        return obj_multiplicities[multiplicity]

    def get_leq_pred(self, seq: Sequence) -> Pred:
        obj_from = seq.obj_from
        if seq.flatten_obj is not None:
            obj_from = seq.flatten_obj
        obj_from_pred = self.get_pred(obj_from)
        seq_leq_pred = self.create_pred(f'{seq.name}_LEQ', 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({seq_leq_pred}(X,Y) <-> ({obj_from_pred}(X) & {obj_from_pred}(Y) & {self.leq_pred}(X,Y))))"
        )
        return seq_leq_pred

    def get_predecessor_pred(self, seq: Sequence) -> Pred:
        """
        Get the predecessor predicate for encoding the sequence

        :param seq: the sequence object
        :return: the predecessor predicate
        """
        if self.old_pred_encoding:
            return self.get_predecessor_pred_old(seq)

        if self.pred_pred is None:
            self.pred_pred = Pred('PRED', 2)
        if self.circular_pred is None:
            self.circular_pred = Pred('CIRCULAR_PRED', 2)

        obj_from = seq.obj_from
        if seq.flatten_obj is not None:
            obj_from = seq.flatten_obj
        obj_from_pred = self.get_pred(obj_from)
        if not seq.circular:
            pred_pred = self.pred_pred
        else:
            pred_pred = self.circular_pred
        seq_pred_pred = self.create_pred(f'{seq.name}_PRED', 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({seq_pred_pred}(X,Y) <-> ({obj_from_pred}(X) & {obj_from_pred}(Y) & {pred_pred}(X,Y))))"
        )
        return seq_pred_pred

    def get_predecessor_pred_old(self, seq: Sequence) -> Pred:
        """
        Get the predecessor predicate for encoding the sequence

        :param seq: the sequence object
        :return: the predecessor predicate
        """
        if self.pred_pred is None:
            self.pred_pred = Pred('Pred', 2)
            self.sentence = self.sentence & parse(
                f"\\forall X: (\\exists Y: (PERM(X,Y))) & \\forall X: (\\exists Y: (PERM(Y,X))) & \\forall X: (~PERM(X,X)) & \\forall X: (\\forall Y: ({self.pred_pred}(X,Y) -> PERM(X,Y))) & \\forall X: (\\forall Y: ({self.pred_pred}(X,Y) -> LEQ(X,Y)))"
            )
            self.cardinality_constraint.add_simple_constraint(
                self.pred_pred, "=", len(self.domain) - 1
            )
            self.cardinality_constraint.add_simple_constraint(
                Pred('PERM', 2), "=", len(self.domain)
            )
        obj_from = seq.obj_from
        if seq.flatten_obj is not None:
            obj_from = seq.flatten_obj
        obj_from_pred = self.get_pred(obj_from)
        if not seq.circular:
            pred_pred = self.pred_pred
        else:
            if self.circular_pred is None:
                self.circular_pred = Pred('Circular_Pred', 2)
                self.sentence = self.sentence & parse(
                    f"\\forall X: (First(X) <-> (\\forall Y: (~{self.pred_pred}(Y,X)))) & \\forall X: (Last(X) <-> (\\forall Y: (~{self.pred_pred}(X,Y)))) & \\forall X: (\\forall Y: ({self.circular_pred}(X,Y) <-> ({self.pred_pred}(X,Y) | (Last(X) & First(Y)))))"
                )
            pred_pred = self.circular_pred
        seq_pred_pred = self.create_pred(f'{seq.name}_PRED', 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({seq_pred_pred}(X,Y) <-> ({obj_from_pred}(X) & {obj_from_pred}(Y) & {pred_pred}(X,Y))))"
        )
        return seq_pred_pred

    def get_next_to_pred(self, seq: Sequence) -> Pred:
        """
        Get the next-to predicate for encoding the sequence

        :return: the next-to predicate
        """
        pred_pred = self.get_predecessor_pred(seq)
        next_to_pred = self.create_pred(f'{seq.name}_NEXT_TO', 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({next_to_pred}(X,Y) <-> ({pred_pred}(X,Y) | {pred_pred}(Y,X))))"
        )
        return next_to_pred

    def create_pred(self, name: str, arity: int) -> Pred:
        return create_cofola_pred(name, arity)

    def create_var(self, name: str, use_gen: bool = True) -> Expr:
        var = create_cofola_var(name)
        if use_gen:
            self.gen_vars.append(var)
        return var
