"""Sequence patterns for sequence constraints."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sympy import Eq
from wfomc import Pred, fol_parse as parse

from cofola.objects.base import Entity, MockObject, Sequence, Set, SizedObject
from cofola.objects.utils import Quantifier
from cofola.utils import create_aux_pred

if TYPE_CHECKING:
    from cofola.context import Context


class SequencePattern(MockObject):
    """Base class for sequence patterns."""

    def encode_for_seq(self, context: "Context", seq: Sequence) -> "Context":
        raise NotImplementedError


class TogetherPattern(SequencePattern):
    """Pattern for elements that must appear together."""
    _fields = ("obj",)

    def body_str(self) -> str:
        return f"together({self.obj.name})"

    def encode_for_seq(self, context: "Context", seq: Sequence, positive: bool) -> "Context":
        from cofola.objects.bag import Bag

        if isinstance(self.obj, Entity):
            if isinstance(seq.obj_from, Bag) or (
                seq.choose and seq.replace
            ):
                obj_pred = context.get_entity_pred(seq, self.obj)
            else:
                context, obj_pred = self.obj.encode(context)
        else:
            if isinstance(seq.obj_from, Bag) or (
                seq.choose and seq.replace
            ):
                raise RuntimeError(
                    "Not supported: together(set) in a bag sequence"
                )
            obj_pred = context.get_pred(self.obj)
        seq_obj_from_pred = context.get_pred(seq.obj_from)
        pred_pred = context.get_predecessor_pred(seq)
        first_pred = context.create_pred(f"{self.obj.name}_first", 1)
        if isinstance(seq.obj_from, Bag) or (
            seq.choose and seq.replace
        ):
            context.sentence = context.sentence & parse(
f"""
\\forall X: ({first_pred}(X) <-> ({obj_pred}(X) & \\forall Y: ({obj_pred}(Y) -> ~{pred_pred}(Y,X))))
"""
            )
        else:
            context.sentence = context.sentence & parse(
f"""
\\forall     X: ({first_pred}(X) <-> ({obj_pred}(X) & {seq_obj_from_pred}(X) & \\forall Y: (({obj_pred}(Y) & {seq_obj_from_pred}(Y)) -> ~{pred_pred}(Y,X))))
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
    """Base class for patterns that have a size/count."""
    _fields = ("entity_or_set1", "entity_or_set2", "quantifier1", "quantifier2")

    def __init__(self, entity_or_set1: Union[Entity, Set],
                 entity_or_set2: Union[Entity, Set],
                 quantifier1: Quantifier = Quantifier.FORALL,
                 quantifier2: Quantifier = Quantifier.FORALL) -> None:
        super().__init__(entity_or_set1, entity_or_set2, quantifier1, quantifier2)

    def _has_size(self, seq: Sequence) -> bool:
        from cofola.objects.bag import Bag

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
        from cofola.objects.bag import Bag

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
            -> tuple["Context", "Expr"]:
        raise NotImplementedError


class LessThanPattern(SequenceSizedPattern):
    """Pattern for element X < Y in sequence."""

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
            -> tuple["Context", "Expr"]:
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
    """Pattern for X immediately before Y in sequence."""

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
            -> tuple["Context", "Expr"]:
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
    """Pattern for X next to Y in sequence."""

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
            -> tuple["Context", "Expr"]:
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
    """Count of a pattern in a sequence (used as SizedObject)."""
    _fields = ("obj", "pattern")

    def body_str(self) -> str:
        return f"{self.obj.name}.count({self.pattern})"

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", "Expr"]:
        return self.pattern.encode_size_var(context, self.obj)


__all__ = [
    "SequencePattern",
    "TogetherPattern",
    "SequenceSizedPattern",
    "LessThanPattern",
    "PredecessorPattern",
    "NextToPattern",
    "SequencePatternCount",
]