from __future__ import annotations

from sympy import Eq
from typing import TYPE_CHECKING, Union
from wfomc import fol_parse as parse

from cofola.objects.base import AtomicConstraint, Bag, Entity, Set, Function

if TYPE_CHECKING:
    from cofola.context import Context


class FuncInit(Function):
    _fields = ("domain", "codomain", "injective", "surjective")

    def __init__(self, domain: Set, codomain: Set,
                 injective: bool = False,
                 surjective: bool = False) -> None:
        super().__init__(domain, codomain, injective, surjective)

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
    _fields = ("func",)

    def _assign_fields(self) -> None:
        super()._assign_fields()
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
    _fields = ("func", "set_or_entity")

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
    _fields = ("func", "set_or_entity")

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
    _fields = ("func", "comparator", "size")

    def body_str(self) -> str:
        return f"{{y | |{self.func.name}⁻¹(y)| {self.comparator} {self.size}}}"

    def combinatorially_eq(self, o):
        return type(o) is FuncSizeConstrainedImage and \
            self.func == o.func and self.comparator == o.comparator and \
            self.size == o.size


class FuncMembershipConstraintedImage(Set):
    _fields = ("func", "member")

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
    _fields = ("func", "subset", "inverse")

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
    _fields = ("func", "disjoint_set")

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
    _fields = ("func", "eq_set")

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
    _fields = ("func", "entities1", "entities2")

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
