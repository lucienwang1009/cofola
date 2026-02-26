"""Sequence objects and constraints."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from sympy import Eq
from wfomc import exclusive, fol_parse as parse

from cofola.objects.bag import Bag, BagInit
from cofola.objects.base import AtomicConstraint, Entity, Sequence, Set
from cofola.objects.set import SetInit

# Import pattern classes for backwards compatibility
from cofola.objects.sequence_patterns import (
    LessThanPattern,
    NextToPattern,
    PredecessorPattern,
    SequencePattern,
    SequencePatternCount,
    SequenceSizedPattern,
    TogetherPattern,
)

if TYPE_CHECKING:
    from cofola.context import Context


class SequenceImpl(Sequence):
    """A sequence formed by permuting a set or bag."""

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


class SequenceConstraint(AtomicConstraint):
    """Constraint on a sequence pattern."""

    def __init__(self, seq: Sequence, pattern: SequencePattern) -> None:
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


# Re-export for backwards compatibility
__all__ = [
    "SequenceImpl",
    "SequenceConstraint",
    # Pattern classes re-exported
    "SequencePattern",
    "TogetherPattern",
    "SequenceSizedPattern",
    "LessThanPattern",
    "PredecessorPattern",
    "NextToPattern",
    "SequencePatternCount",
]