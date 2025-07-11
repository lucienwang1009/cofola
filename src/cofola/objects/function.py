from __future__ import annotations

from symengine import Eq
from typing import Union
from wfomc import fol_parse as parse

from cofola.objects.base import AtomicConstraint, Bag, Entity, Set, Function

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cofola.context import Context


class FuncInit(Function):
    def __init__(self, domain: Set, codomain: Set,
                 injective: bool = False,
                 surjective: bool = False) -> None:
        super().__init__(domain, codomain, injective, surjective)

    def _assign_args(self) -> None:
        self.domain, self.codomain, self.injective, self.surjective = \
            self.args

    @property
    def bijective(self) -> bool:
        return self.injective and self.surjective

    def body_str(self) -> str:
        if self.bijective:
            return f"{self.domain.name} ↣↠ {self.codomain.name}"
        elif self.injective:
            return f"{self.domain.name} ↣ {self.codomain.name}"
        elif self.surjective:
            return f"{self.domain.name} ↠ {self.codomain.name}"
        else:
            return f"{self.domain.name} → {self.codomain.name}"

    def is_uncertain(self) -> bool:
        return True

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        domain, codomain = self.domain, self.codomain
        domain_pred = context.get_pred(domain)
        codomain_pred = context.get_pred(codomain)
        context.sentence &= parse(
            f"\\forall X: (\\exists Y: ({domain_pred}(X) -> {obj_pred}(X, Y)))"
            f"& \\forall X: (\\forall Y: ({obj_pred}(X, Y) -> ({domain_pred}(X) & {codomain_pred}(Y))))"
        )
        obj_var = context.get_obj_var(self)
        if domain.size is not None:
            domain_size = domain.size
        else:
            domain_size = context.get_obj_var(domain)
        context.validator.append(Eq(obj_var, domain_size))
        if self.surjective or self.bijective:
            context.sentence &= parse(
                f"\\forall Y: (\\exists X: ({codomain_pred}(Y) -> ({obj_pred}(X, Y))))"
            )
        elif self.injective:
            pass
        return context


class FuncInverse(Function):
    def __init__(self, func: Function) -> None:
        super().__init__(func)

    def _assign_args(self) -> None:
        self.func = self.args[0]
        if not self.func.bijective:
            raise ValueError(
                "Only bijective function can have an inverse function!"
            )

    def body_str(self) -> str:
        return f"{self.func.name}⁻¹"

    def combinatorially_eq(self, o):
        return type(o) is FuncInverse and self.func == o.func

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        context.sentence &= parse(
            f"\\forall X: (\\forall Y: ({obj_pred}(X, Y) <-> {func_pred}(Y, X)))"
        )
        return context


# Set objects obtained from function operations
class FuncImage(Set):
    def __init__(self, func: Function, set_or_entity: Union[Set, Entity]) -> None:
        super().__init__(func, set_or_entity)

    def _assign_args(self) -> None:
        self.func, self.set_or_entity = self.args

    def inherit(self) -> None:
        self.update(self.func.codomain.p_entities,
                    max_size=self.func.codomain.max_size)

    def body_str(self) -> str:
        return f"{self.func.name}({self.set_or_entity.name})"

    def combinatorially_eq(self, o):
        return type(o) is FuncImage and \
            self.func == o.func and self.set_or_entity == o.set_or_entity

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        set_or_entity = self.set_or_entity
        if isinstance(set_or_entity, Entity):
            context, pred = set_or_entity.encode(context)
            context.sentence &= parse(
                f"\\forall X: (\\forall Y: ({pred}(X) & {func_pred}(X, Y) -> {obj_pred}(Y))) "
                f"& \\forall X: (\\forall Y: ({pred}(X) & ~{func_pred}(X, Y) -> ~{obj_pred}(Y)))"
            )
        else:
            pred = context.get_pred(set_or_entity)
            context.sentence &= parse(
                f"\\forall Y: ({obj_pred}(Y) <-> (\\exists X: ({pred}(X) & {func_pred}(X, Y))))"
            )
        return context


class FuncInverseImage(Set):
    def __init__(self, func: Function, set_or_entity: Union[Set, Entity]) -> None:
        super().__init__(func, set_or_entity)

    def _assign_args(self) -> None:
        self.func, self.set_or_entity = self.args

    def inherit(self) -> None:
        self.update(self.func.domain.p_entities,
                    max_size=self.func.domain.max_size)

    def body_str(self) -> str:
        return f"{self.func.name}⁻¹({self.set_or_entity.name})"

    def combinatorially_eq(self, o):
        return type(o) is FuncInverseImage and \
            self.func == o.func and self.set_or_entity == o.set_or_entity

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        set_or_entity = self.set_or_entity
        if isinstance(set_or_entity, Entity):
            context, pred = set_or_entity.encode(context)
            context.sentence &= parse(
                f"\\forall X: (\\forall Y: ({pred}(Y) & {func_pred}(X, Y) -> {obj_pred}(X))) "
                f"& \\forall X: (\\forall Y: ({pred}(Y) & ~{func_pred}(X, Y) -> ~{obj_pred}(X)))"
            )
        else:
            pred = context.get_pred(set_or_entity)
            context.sentence &= parse(
                f"\\forall X: ({obj_pred}(X) <-> (\\exists Y: ({pred}(Y) & {func_pred}(X, Y))))"
            )
        return context


class FuncSizeConstrainedImage(Set):
    def __init__(self, func: Function, comparator: str, size: int) -> None:
        super().__init__(func, comparator, size)

    def _assign_args(self) -> None:
        self.func, self.comparator, self.size = self.args

    def body_str(self) -> str:
        return f"{{y | |{self.func.name}⁻¹(y)| {self.comparator} {self.size}}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncSizeConstrainedImage and \
            self.func == o.func and self.comparator == o.comparator and \
            self.size == o.size


class FuncMembershipConstraintedImage(Set):
    def __init__(self, func: Function, member: Entity) -> None:
        super().__init__(func, member)

    def _assign_args(self) -> None:
        self.func, self.member = self.args

    def body_str(self) -> str:
        return f"{{y | {self.member} ∈ {self.func.name}⁻¹(y)}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncMembershipConstraintedImage and \
            self.func == o.func and self.member == o.member

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        context, member_pred = self.member.encode()
        context.sentence &= parse(
            f"\\forall Y: ({obj_pred}(Y) <-> (\\exists X: ({member_pred}(X) & {func_pred}(X, Y))))"
        )
        return context


class FuncSubsetConstrainedImage(Set):
    def __init__(self, func: Function, subset: Set, inverse: bool = False) -> None:
        super().__init__(func, subset, inverse)

    def _assign_args(self) -> None:
        self.func, self.subset, self.inverse = self.args

    def body_str(self) -> str:
        if self.inverse:
            return f"{{y | {self.subset.name} ⊆ {self.func.name}⁻¹(y)}}"
        else:
            return f"{{y | {self.func.name}⁻¹(y) ⊆ {self.subset.name}}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncSubsetConstrainedImage and \
            self.func == o.func and self.subset == o.subset and \
            self.inverse == o.inverse

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        set_pred = context.get_pred(self.subset)
        if not self.inverse:
            # f-1 subset S
            context.sentence &= parse(
                f"\\forall Y: ({obj_pred}(Y) <-> (\\forall X: ({func_pred}(X, Y) -> {set_pred}(X))))"
            )
        else:
            context.sentence &= parse(
                f"\\forall Y: ({obj_pred}(Y) <-> (\\forall X: ({set_pred}(X) -> {func_pred}(X, Y))))"
            )
        return context


class FuncDisjointConstrainedImage(Set):
    def __init__(self, func: Function, disjoint_set: Set) -> None:
        super().__init__(func, disjoint_set)

    def _assign_args(self) -> None:
        self.func, self.disjoint_set = self.args

    def body_str(self) -> str:
        return f"{{y | {self.func.name}⁻¹(y) ∩ {self.disjoint_set.name} = ∅}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncDisjointConstrainedImage and \
            self.func == o.func and self.disjoint_set == o.disjoint_set

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        set_pred = context.get_pred(self.disjoint_set)
        context.sentence &= parse(
            f"\\forall Y: ({obj_pred}(Y) <-> (\\forall X: ({func_pred}(X, Y) -> ~{set_pred}(X))))"
        )
        return context


class FuncEqConstrainedImage(Set):
    def __init__(self, func: Function, eq_set: Set) -> None:
        super().__init__(func, eq_set)

    def _assign_args(self) -> None:
        self.func, self.eq_set = self.args

    def body_str(self) -> str:
        return f"{{y | {self.func.name}⁻¹(y) = {self.eq_set.name}}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncEqConstrainedImage and \
            self.func == o.func and self.eq_set == o.eq_set

    def encode(self, context: "Context") -> "Context":
        obj_pred = context.get_pred(self, create=True, use=False)
        func_pred = context.get_pred(self.func)
        set_pred = context.get_pred(self.eq_set)
        context.sentence &= parse(
            f"\\forall Y: ({obj_pred}(Y) <-> (\\forall X: ({set_pred}(X) <-> {func_pred}(X, Y))))"
        )
        return context


class FuncPairConstraint(AtomicConstraint):
    def __init__(self, func: Function,
                 entities1: Union[Entity, Set, Bag],
                 entities2: Union[Entity, Set, Bag]) -> None:
        """
        Construct a pair constraint for a function:
        1. if entities1 and entities2 are both entities, then func(entities1) = entities2
        2. if entities1 is an entity and entities2 is a set, then func(entities1) ∈ entities2
        3. if entities1 is a set and entities2 is an entity, then for all x in entities1, func(x) = entities2
        4. if entities1 and entities2 are both sets, then for all x in entities1, func(x) ∈ entities2

        :param func: the function
        :param entities1: the first set of entities
        :param entities2: the second set of entities
        """
        super().__init__(func, entities1, entities2)

    def _assign_args(self) -> None:
        self.func, self.entities1, self.entities2 = self.args

    def __str__(self) -> str:
        if isinstance(self.entities2, Entity):
            if self.positive:
                return f"{self.func.name}({self.entities1.name}) = {self.entities2.name}"
            else:
                return f"{self.func.name}({self.entities1.name}) != {self.entities2.name}"
        else:
            if self.positive:
                return f"{self.func.name}({self.entities1.name}) ∈ {self.entities2.name}"
            else:
                return f"{self.func.name}({self.entities1.name}) ∉ {self.entities2.name}"

    def encode(self, context: "Context") -> "Context":
        func_pred = context.get_pred(self.func)
        if isinstance(self.entities1, Entity):
            context, pred1 = self.entities1.encode(context)
        else:
            pred1 = context.get_pred(self.entities1)
        if isinstance(self.entities2, Entity):
            context, pred2 = self.entities2.encode(context)
        else:
            pred2 = context.get_pred(self.entities2)
        if self.positive:
            context.sentence &= parse(
                f"\\forall X: (\\forall Y: ({func_pred}(X, Y) & {pred1}(X) -> {pred2}(Y)))"
            )
        else:
            context.sentence &= parse(
                f"\\forall X: (\\forall Y: ({func_pred}(X, Y) & {pred1}(X) -> ~{pred2}(Y)))"
            )
        return context
