import math

from sympy import Eq
from wfomc import Pred, Expr, fol_parse as parse, exclusive

from cofola.objects.bag import BagInit
from cofola.objects.base import AtomicConstraint, Entity, MockObject, Sequence, Set, Bag, SizedObject
from typing import TYPE_CHECKING, Union

from cofola.objects.set import SetInit
from cofola.objects.utils import Quantifier
from cofola.utils import create_aux_pred


if TYPE_CHECKING:
    from cofola.context import Context


class SequenceImpl(Sequence):
    def __init__(self, obj_from: Union[Set, Bag], choose: bool = True,
                 replace: bool = True, size: int = None,
                 circular: bool = False, reflection: bool = False,
                 flatten_obj: Set = None) -> None:
        """
        A sequence formed by permuting a set.
        It is distinguished from a tuple by the ability to support constraints on relative positions.

        :param obj_from: the set
        :param choose: whether the tuple is formed by choosing elements from the set and permuting them
        :param replace: whether the elements are chosen with replacement
        :param size: the size of the tuple
        :param flatten_obj: the flatten set of the input bag (if any) to be used for forming the sequence; if the input is a set, this should be None
        """
        super().__init__(obj_from, choose, replace, size,
                         circular, reflection, flatten_obj)

    def _assign_args(self) -> None:
        (
            self.obj_from, self.choose,
            self.replace, self.size,
            self.circular, self.reflection,
            self.flatten_obj
        ) = self.args
        if not self.choose and self.replace:
            raise ValueError(
                f"A sequence is formed with replacement but not by choosing: {self}"
            )

    def inherit(self) -> None:
        if not self.choose and self.obj_from.size is not None:
            self.size = self.obj_from.size
            self.max_size = self.size
        else:
            if not self.replace:
                self.max_size = min(self.max_size, self.obj_from.max_size)

    def body_str(self) -> str:
        if self.choose:
            # if formed by choosing, its size must be specified
            if self.replace:
                if not self.circular:
                    return f"choose_replace_sequence({self.obj_from.name}, {self.size})"
                else:
                    if self.reflection:
                        return f"choose_replace_circle({self.obj_from.name}, {self.size}, reflection=True)"
                    else:
                        return f"choose_replace_circle({self.obj_from.name}, {self.size})"
            else:
                if not self.circular:
                    return f"choose_sequence({self.obj_from.name}, {self.size})"
                else:
                    if self.reflection:
                        return f"choose_circle({self.obj_from.name}, {self.size}, reflection=True)"
                    else:
                        return f"choose_circle({self.obj_from.name}, {self.size})"
        else:
            if not self.circular:
                return f"sequence({self.obj_from.name})"
            else:
                if self.reflection:
                    return f"circle({self.obj_from.name}, reflection=True)"
                else:
                    return f"circle({self.obj_from.name})"

    def encode(self, context: "Context") -> "Context":
        domain_size = len(context.domain)
        if isinstance(self.obj_from, Set):
            if self.choose:
                if self.replace:
                    # use the flatten set to form the sequence
                    flatten_obj_pred = context.get_pred(self.flatten_obj)
                    context.sentence = context.sentence & parse(
                        f'\\forall X: (\\forall Y: (({flatten_obj_pred}(X) & ~{flatten_obj_pred}(Y)) -> {context.leq_pred}(X,Y)))'
                    )
                    entity_preds = list()
                    for entity in self.obj_from:
                        if isinstance(self.obj_from, SetInit):
                            entity_pred = context.get_entity_pred(self, entity)
                            entity_preds.append(entity_pred)
                    or_formula = ' | '.join(f'{pred}(X)' for pred in entity_preds)
                    context.sentence = context.sentence & parse(
                        f'\\forall X: (({or_formula}) <-> {flatten_obj_pred}(X))'
                    )
                    context.sentence = context.sentence & exclusive(entity_preds)
                    # handle overcount (and undercount) introduced by the permutation
                    context.overcount = (
                        context.overcount *
                        math.factorial(domain_size - self.size) *
                        math.factorial(self.size)
                    )
                else:
                    raise RuntimeError(
                        f"choose_sequence should have been transformed to choose and sequence: {self}"
                    )
            else:
                obj_pred = context.get_pred(self.obj_from)
                context.sentence = context.sentence & parse(
                    f'\\forall X: (\\forall Y: (({obj_pred}(X) & ~{obj_pred}(Y)) -> {context.leq_pred}(X,Y)))'
                )
                # handle overcount (and undercount) introduced by the permutation
                context.overcount = (
                    context.overcount *
                    math.factorial(domain_size - self.size)
                )
        else:
            if self.choose:
                raise RuntimeError(
                    f"choose_sequence from a bag should have been transformed to choose and sequence: {self}"
                )
            # use the flatten set to form the sequence
            flatten_obj_pred = context.get_pred(self.flatten_obj)
            context.sentence = context.sentence & parse(
                f'\\forall X: (\\forall Y: (({flatten_obj_pred}(X) & ~{flatten_obj_pred}(Y)) -> {context.leq_pred}(X,Y)))'
            )
            entity_preds = list()
            for entity in self.obj_from.dis_entities:
                entity_pred = context.get_entity_pred(self, entity)
                entity_preds.append(entity_pred)
                entity_var = context.get_entity_var(self, entity)
                context.weighting[entity_pred] = (
                    entity_var, 1
                )
                if isinstance(self.obj_from, BagInit):
                    multi = self.obj_from.p_entities_multiplicity[entity]
                    context.validator.append(
                        Eq(entity_var, multi)
                    )
                else:
                    bag_entity_var = context.get_entity_var(self.obj_from, entity)
                    context.validator.append(
                        Eq(entity_var, bag_entity_var)
                    )
            or_formula = ' | '.join(f'{pred}(X)' for pred in entity_preds)
            context.sentence = context.sentence & parse(
                f'\\forall X: (({or_formula}) <-> {flatten_obj_pred}(X))'
            )
            context.sentence = context.sentence & exclusive(entity_preds)
            # handle overcount (and undercount) introduced by the permutation
            context.overcount = (
                context.overcount *
                math.factorial(domain_size - self.size) *
                math.factorial(self.size)
            )
        if self.circular:
            context.circle_len = self.size
            context.overcount = context.overcount * self.size
            if self.reflection:
                context.overcount = context.overcount * 2
        return context


class SequencePattern(MockObject):
    def encode_for_seq(self, context: "Context", seq: Sequence) -> "Context":
        raise NotImplementedError


# TODO: consider together(set) in a bag sequence
class TogetherPattern(SequencePattern):
    def __init__(self, obj: Union[Set, Entity]) -> None:
        super().__init__(obj)

    def _assign_args(self) -> None:
        self.obj = self.args[0]

    def body_str(self) -> str:
        return f"together({self.obj.name})"

    def encode_for_seq(self, context: "Context", seq: Sequence, positive: bool) -> "Context":
        if isinstance(self.obj, Entity):
            if isinstance(seq.obj_from, Bag) or (
                seq.choose and seq.replace
            ):
                obj_pred = context.get_entity_pred(seq, self.obj)
            else:
                context, obj_pred = self.obj.encode(context)
        else:
            obj_pred = context.get_pred(self.obj)
        pred_pred = context.get_predecessor_pred(seq)
        first_pred = context.create_pred(f"{self.obj.name}_first", 1)
        context.sentence = context.sentence & parse(
f"""
\\forall X: ({first_pred}(X) <-> ({obj_pred}(X) & \\forall Y: ({obj_pred}(Y) -> ~{pred_pred}(Y,X))))
"""
        )
        first_var = context.create_var(f"{self.obj.name}_first")
        context.weighting[first_pred] = (first_var, 1)
        if positive:
            context.validator.append(
                first_var <= 1
            )
        else:
            context.validator.append(
                first_var > 1
            )
        return context


class SequenceSizedPattern(SequencePattern):
    def __init__(self, entity_or_set1: Union[Entity, Set],
                 entity_or_set2: Union[Entity, Set],
                 quantifier1: Quantifier = Quantifier.FORALL,
                 quantifier2: Quantifier = Quantifier.FORALL) -> None:
        super().__init__(
            entity_or_set1, entity_or_set2, quantifier1, quantifier2)

    def _assign_args(self) -> None:
        (
            self.entity_or_set1, self.entity_or_set2,
            self.quantifier1, self.quantifier2
        ) = self.args

    def _has_size(self, seq: Sequence) -> bool:
        if isinstance(self.entity_or_set1, Set) or (
            isinstance(self.entity_or_set1, Entity) and
            (
                isinstance(seq.obj_from, Bag) or
                seq.choose and seq.replace
            )
        ):
            return True
        if isinstance(self.entity_or_set2, Set) or (
            isinstance(self.entity_or_set2, Entity) and
            (
                isinstance(seq.obj_from, Bag) or
                seq.choose and seq.replace
            )
        ):
            return True
        return False

    def _get_preds(self, context: "Context", seq: Sequence) \
            -> tuple["Context", Pred, Pred]:
        if isinstance(self.entity_or_set1, Entity):
            if isinstance(seq.obj_from, Bag) or (
                seq.choose and seq.replace
            ):
                obj_pred1 = context.get_entity_pred(seq, self.entity_or_set1)
            else:
                context, obj_pred1 = self.entity_or_set1.encode(context)
        else:
            obj_pred1 = context.get_pred(self.entity_or_set1)
        if isinstance(self.entity_or_set2, Entity):
            if isinstance(seq.obj_from, Bag) or (
                seq.choose and seq.replace
            ):
                obj_pred2 = context.get_entity_pred(seq, self.entity_or_set2)
            else:
                context, obj_pred2 = self.entity_or_set2.encode(context)
        else:
            obj_pred2 = context.get_pred(self.entity_or_set2)
        return context, obj_pred1, obj_pred2

    def encode_size_var(self, context: "Context", seq: Sequence) \
            -> tuple["Context", Expr]:
        raise NotImplementedError


class LessThanPattern(SequenceSizedPattern):
    def body_str(self) -> str:
        return f"{self.entity_or_set1.name} < {self.entity_or_set2.name}"

    def encode_for_seq(self, context: "Context", seq: Sequence, positive: bool) -> "Context":
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        leq_pred = context.get_leq_pred(seq)
        obj_leq_pred = create_aux_pred(
            2, f"{self.entity_or_set1.name}_leq_{self.entity_or_set2.name}"
        )
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {leq_pred}(X,Y)) <-> {obj_leq_pred}(X,Y)))"
        )
        leq_var = context.create_var(
            obj_leq_pred.name
        )
        context.weighting[obj_leq_pred] = (leq_var, 1)
        if positive:
            context.validator.append(leq_var > 0)
        else:
            context.validator.append(Eq(leq_var, 0))
        return context

    def encode_size_var(self, context: "Context", seq: Sequence) \
            -> tuple["Context", Expr]:
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        leq_pred = context.get_leq_pred(seq)
        obj_leq_pred = context.create_pred(
            f"{self.entity_or_set1.name}_leq_{self.entity_or_set2.name}", 2
        )
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {leq_pred}(X,Y)) <-> {obj_leq_pred}(X,Y)))"
        )
        obj_leq_var = context.create_var(
            f"{self.entity_or_set1.name}_leq_{self.entity_or_set2.name}"
        )
        context.weighting[obj_leq_pred] = (obj_leq_var, 1)
        return context, obj_leq_var


class PredecessorPattern(SequenceSizedPattern):
    def body_str(self) -> str:
        return f"({self.entity_or_set1.name}, {self.entity_or_set2.name})"

    def encode_for_seq(self, context: "Context", seq: Sequence, positive: bool) -> "Context":
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        pred_pred = context.get_predecessor_pred(seq)
        obj_pred_pred = create_aux_pred(
            2, f"{self.entity_or_set1.name}_pred_{self.entity_or_set2.name}"
        )
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {pred_pred}(X,Y)) <-> {obj_pred_pred}(X,Y)))"
        )
        pred_var = context.create_var(
            obj_pred_pred.name
        )
        context.weighting[obj_pred_pred] = (pred_var, 1)
        if positive:
            context.validator.append(pred_var > 0)
        else:
            context.validator.append(Eq(pred_var, 0))
        return context

    def encode_size_var(self, context: "Context", seq: Sequence) \
            -> tuple["Context", Expr]:
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        obj_pred_pred = context.create_pred(
            f"{self.entity_or_set1.name}_pred_{self.entity_or_set2.name}", 2
        )
        pred_pred = context.get_predecessor_pred(seq)
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {pred_pred}(X,Y)) <-> {obj_pred_pred}(X,Y)))"
        )
        obj_var = context.create_var(
            f"{self.entity_or_set1.name}_pred_{self.entity_or_set2.name}"
        )
        context.weighting[obj_pred_pred] = (obj_var, 1)
        return context, obj_var


class NextToPattern(SequenceSizedPattern):
    def body_str(self) -> str:
        return f"next_to({self.entity_or_set1.name}, {self.entity_or_set2.name})"

    def encode_for_seq(self, context: "Context", seq: Sequence, positive: bool) -> "Context":
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        next_to_pred = context.get_next_to_pred(seq)
        obj_next_to_pred = create_aux_pred(
            2, f"{self.entity_or_set1.name}_next_to_{self.entity_or_set2.name}"
        )
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {next_to_pred}(X,Y)) <-> {obj_next_to_pred}(X,Y)))"
        )
        next_to_var = context.create_var(
            obj_next_to_pred.name
        )
        context.weighting[obj_next_to_pred] = (next_to_var, 1)
        if positive:
            # TODO: care about semantics here! exists/forall/...?
            context.validator.append(next_to_var > 0)
        else:
            context.validator.append(Eq(next_to_var, 0))
        return context

    def encode_size_var(self, context: "Context", seq: Sequence) \
            -> tuple["Context", Expr]:
        context, obj_pred1, obj_pred2 = self._get_preds(context, seq)
        next_to_pred = context.get_next_to_pred(seq)
        obj_next_to_pred = context.create_pred(
            f"{self.entity_or_set1.name}_next_to_{self.entity_or_set2.name}", 2
        )
        context.sentence = context.sentence & parse(
            f"\\forall X: (\\forall Y: (({obj_pred1}(X) & {obj_pred2}(Y) & {next_to_pred}(X,Y)) <-> {obj_next_to_pred}(X,Y)))"
        )
        obj_var = context.create_var(
            f"{self.entity_or_set1.name}_next_to_{self.entity_or_set2.name}"
        )
        context.weighting[obj_next_to_pred] = (obj_var, 1)
        return context, obj_var


class SequencePatternCount(SizedObject, MockObject):
    def __init__(self, obj: Sequence, pattern: SequencePattern) -> None:
        super().__init__(obj, pattern)

    def _assign_args(self) -> None:
        self.obj, self.pattern = self.args

    def body_str(self) -> str:
        return f"{self.obj.name}.count({self.pattern})"

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", Expr]:
        return self.pattern.encode_size_var(context, self.obj)


class SequenceConstraint(AtomicConstraint):
    def __init__(self, seq: Sequence,
                 pattern: SequencePattern) -> None:
        super().__init__(seq, pattern)

    def _assign_args(self) -> None:
        self.seq, self.pattern = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.pattern} in {self.seq.name}"
        else:
            return f"{self.pattern} not in {self.seq.name}"

    def encode(self, context: "Context") -> "Context":
        return self.pattern.encode_for_seq(
            context, self.seq, self.positive
        )
