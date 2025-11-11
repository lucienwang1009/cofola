from __future__ import annotations

from typing import TYPE_CHECKING, Union
from wfomc import Const, Pred, fol_parse as parse

from cofola.objects.utils import aux_obj_name

if TYPE_CHECKING:
    from cofola.context import Context


# Entity is NOT a combinatorics object, but the base element of all combinatorics objects
class Entity(object):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Entity):
            return self.name == o.name
        return False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def encode(self, context: "Context") -> tuple[Context, Pred]:
        # Though Entity is not a combinatorics object, we might still need to encode it
        if self in context.obj2pred:
            return context, context.obj2pred[self]
        pred = context.get_pred(self, create=True)
        context.unary_evidence.add(pred(Const(self.name)))
        context.unary_evidence.update(
            ~pred(Const(e.name)) for e in context.problem.entities if e != self
        )
        return context, pred


"""
NOTE: for combinatorics objects and constraints, we allow them to be defined duplicate times, e.g.,
set S = {1, 2, 3}
set S1 = {1, 2, 3}
func f = S -> S1
set img = f(S)
set img1 = f(S)

However, to avoid duplicate encoding of the same object, we need to make sure that the objects that don't introduce additional combinatoric solutions are encoded only once.
For example, if we have the above problem, we should only encode S and S1 once, and also encode img and img1 only once. This is because when in a model, if the func f is determined, img and img1 are identical.

We use the `combinatorially_eq` method to determine if two objects are the same in terms of combinatorics.
"""
class CombinatoricsBase(object):
    def __init__(self, *args) -> None:
        """
        Base class for all combinatorics objects and constraints

        :param args: the args of the object or constraint
        """
        super().__init__()
        self.args: tuple = args
        self.dependences: set[CombinatoricsObject] = set()
        self.descendants: set[CombinatoricsBase] = set()
        self._assign_args()
        self._build_dependences()

    def _assign_args(self) -> None:
        """
        Assign the object or constraint to its dependences
        """
        pass

    def _build_dependences(self) -> None:
        # NOTE: all args that are CombinatoricsObject are dependences
        self.dependences = set(
            arg for arg in self.args if isinstance(arg, CombinatoricsObject)
        )
        for dep in self.dependences:
            dep.descendants.add(self)

    # def __hash__(self) -> int:
    #     """
    #     The hash value of the object or constraint
    #     """
    #     return id(self)

    # def __eq__(self, o: object) -> bool:
    #     if isinstance(o, CombinatoricsBase):
    #         # return self.name == o.name
    #         return id(self) == id(o)
    #     return False

    def combinatorially_eq(self, o: CombinatoricsBase) -> bool:
        """
        Default implementation of combinatorially_eq, i.e., always return False
        Implement this method in subclasses
        """
        return False

    def inherit(self) -> None:
        """
        Inherit the properties from its dependences
        """
        pass

    def subs_args(self, *args):
        """
        Substitute the args of the object or constraint

        :param args: the new args
        """
        self.args = args + self.args[len(args):]
        self._assign_args()
        self._build_dependences()

    def subs_obj(self, old_obj: CombinatoricsObject,
                 new_obj: CombinatoricsObject):
        """
        Substitute the object in the object or constraint

        :param old_obj: the old object
        :param new_obj: the new object
        """
        args = (
            new_obj if arg == old_obj else arg
            for arg in self.args
        )
        self.subs_args(*args)

    def encode(self, context: "Context") -> "Context":
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)


class CombinatoricsObject(CombinatoricsBase):
    def __init__(self, *args) -> None:
        # the name is for debugging
        self.name = aux_obj_name()
        super().__init__(*args)

    def body_str(self) -> str:
        """
        The string representation of the object without the name
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name} = {self.body_str()}"

    def is_uncertain(self) -> bool:
        """
        Whether the object introduces uncertainty to the problem
        """
        return False


class MockObject(CombinatoricsObject):
    """
    A mock object. This object is used to represent the object that is in the problem but not combinatorial.
    Thus, we don't need to encode it.
    """
    def encode(self, context: "Context") -> "Context":
        return context


class SizedObject(CombinatoricsObject):
    """
    Base class for objects that have a size, i.e. we can put a constraint on the size of the object
    """
    def __init__(self, *args) -> None:
        # the `size` of the object is the exact size of the object
        # while the `max_size` is the maximum possible size of the object
        # the max size is useless if the size is not None
        self.size: int = None
        self.max_size: int = float("inf")
        super().__init__(*args)

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", Expr]:
        raise NotImplementedError


class Set(SizedObject):
    def __init__(self, *args) -> None:
        self.p_entities: set[Entity] = None
        super().__init__(*args)

    def update(self, p_entities: set[Entity],
               max_size: int = float("inf")):
        if self.p_entities is None:
            self.p_entities = p_entities
        else:
            self.p_entities.update(p_entities)
        if self.size is None:
            self.max_size = min(self.max_size, max_size, len(self.p_entities))
        else:
            self.max_size = self.size

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", Expr]:
        return context, context.get_obj_var(self)


class Bag(SizedObject):
    def __init__(self, *args) -> None:
        self.p_entities_multiplicity: dict[Entity, int] = None
        # distinguishable entities are those entities that cannot be lifted
        self.dis_entities: set[Entity] = None
        # indistinguishable entities are those entities whose multiplicity is the same
        # indistinguishable entities are used to lift bags
        self.indis_entities: dict[int, set[Entity]] = None
        super().__init__(*args)

    def multiplicity(self, entity: Entity) -> int:
        """
        Get the potential multiplicity of the entity in the bag

        :param entity: the entity
        :return: the potential multiplicity
        """
        return self.p_entities_multiplicity.get(entity, 0)

    def update(self, p_entities_multiplicity: dict[Entity, int],
               max_size: int = float("inf")):
        if self.p_entities_multiplicity is None:
            self.p_entities_multiplicity = p_entities_multiplicity
        if self.size is None:
            self.max_size = min(self.max_size, max_size, sum(self.p_entities_multiplicity.values()))
        else:
            self.max_size = self.size
        self.p_entities_multiplicity = dict(
            (e, min(m, self.max_size, p_entities_multiplicity.get(e, float("inf"))))
            for e, m in self.p_entities_multiplicity.items()
        )

    def encode_size_var(self, context: "Context") \
            -> tuple["Context", Expr]:
        term = 0
        vars = context.get_entity_var(self)
        term = term + sum(var for var in vars.values())
        if len(context.singletons) > 0:
            obj_pred = context.get_pred(self)
            singletons_pred = context.get_pred(context.singletons)
            bag_singletons_pred = context.create_pred(f"{obj_pred.name}_singletons", 1)
            context.sentence = context.sentence & parse(
                f"\\forall X: ({bag_singletons_pred}(X) <-> {obj_pred}(X) & {singletons_pred}(X))"
            )
            singleton_var = context.get_obj_var(self, set_weight=False)
            context.weighting[bag_singletons_pred] = (singleton_var, 1)
            term = term + singleton_var
        # handle indistinguishable entities
        vars = context.get_indis_entity_var(self)
        term = term + sum(var for var in vars.values())
        return context, term


class Function(CombinatoricsObject):
    pass


class Partition(CombinatoricsObject):
    def __init__(self, obj_from: Union[Set, Bag], size: int, ordered: bool) -> None:
        super().__init__(obj_from, size, ordered)
        self.partitioned_objs: list

    def _assign_args(self) -> None:
        self.obj_from, self.size, self.ordered = self.args

    def body_str(self) -> str:
        if self.ordered:
            return f"compose({self.obj_from.name}, {self.size})"
        else:
            return f"partition({self.obj_from.name}, {self.size})"

    def is_uncertain(self) -> bool:
        return True

    def subs_obj(self, old_obj: CombinatoricsObject,
                 new_obj: CombinatoricsObject):
        super().subs_obj(old_obj, new_obj)
        for obj in self.partitioned_objs:
            obj.subs_obj(old_obj, new_obj)


class Part(SizedObject):
    def __init__(self, obj_from: Partition, index: int) -> None:
        super().__init__(obj_from, index)

    def _assign_args(self) -> None:
        self.obj_from, self.index = self.args

    def combinatorially_eq(self, o: CombinatoricsBase) -> bool:
        return isinstance(o, Part) and self.obj_from == o.obj_from \
            and self.index == o.index

    def body_str(self) -> str:
        return f"{self.obj_from.name}[{self.index}]"

    def encode(self, context: "Context") -> "Context":
        return context


class Tuple(SizedObject):
    def is_uncertain(self) -> bool:
        return True


class Sequence(SizedObject):
    def is_uncertain(self) -> bool:
        return True


class Circle(SizedObject):
    def is_uncertain(self) -> bool:
        return True


class CombinatoricsObjectsGroup(CombinatoricsObject):
    pass


class CombinatoricsConstraint(CombinatoricsBase):
    def __invert__(self):
        return Negation(self)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __repr__(self) -> str:
        return str(self)


class AtomicConstraint(CombinatoricsConstraint):
    def __init__(self, *args) -> None:
        # whether the constraint is positive, only used for encoding
        # not used for constructing the constraint, the negation of
        # the constraint should be handled by the negation class below
        self.positive: bool = True
        super().__init__(*args)

    def negate(self):
        self.positive = not self.positive


class ComplexConstraint(CombinatoricsConstraint):
    pass


class Negation(ComplexConstraint):
    def __init__(self, constraint: CombinatoricsConstraint):
        super().__init__(constraint)

    def _assign_args(self) -> None:
        self.sub_constraint = self.args[0]

    def __str__(self):
        return f"~({self.sub_constraint})"


class BinaryConstraint(ComplexConstraint):
    def __init__(self, op_name: str,
                 first_constraint: CombinatoricsConstraint,
                 second_constraint: CombinatoricsConstraint):
        self.op_name: str = op_name
        super().__init__(first_constraint, second_constraint)

    def _assign_args(self) -> None:
        self.first_constraint, self.second_constraint = self.args

    def __str__(self):
        return f"({self.first_constraint} {self.op_name} {self.second_constraint})"


class And(BinaryConstraint):
    def __init__(self, first_constraint: CombinatoricsConstraint,
                 second_constraint: CombinatoricsConstraint):
        super().__init__("&", first_constraint, second_constraint)


class Or(BinaryConstraint):
    def __init__(self, first_constraint: CombinatoricsConstraint,
                 second_constraint: CombinatoricsConstraint):
        super().__init__("|", first_constraint, second_constraint)
