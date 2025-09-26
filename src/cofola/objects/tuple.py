from logzero import logger

from cofola.objects.base import AtomicConstraint, Bag, CombinatoricsObject, Entity, MockObject, Set, SizedObject, Tuple
from typing import Union
from cofola.objects.function import FuncInit

from cofola.objects.set import SetInit


class TupleImpl(Tuple):
    def __init__(self, obj_from: Union[Set, Bag], choose: bool = True,
                 replace: bool = True, size: int = None,
                 indices: SetInit = None, mapping: FuncInit = None) -> None:
        """
        A tuple formed by permuting a set. Tuples are realized by functions (the mapping arg)

        :param obj_from: the set
        :param choose: whether the tuple is formed by choosing elements from the set and permuting them
        :param replace: whether the elements are chosen with replacement
        :param size: the size of the tuple
        :param indices: the indices for functions
        :param mapping: the mapping function from indices to obj_from, realizing the tuple
        """
        super().__init__(obj_from, choose, replace, size, indices, mapping)

    def _assign_args(self) -> None:
        self.obj_from, self.choose, self.replace, self.size, self.indices, self.mapping = \
            self.args
        if not self.choose and self.replace:
            raise ValueError(
                f"A tuple is formed with replacement but not by choosing: {self}"
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
                return f"choose_replace_tuple({self.obj_from.name}, {self.size})"
            else:
                return f"choose_tuple({self.obj_from.name}, {self.size})"
        else:
            return f"tuple({self.obj_from.name})"

    def encode(self, context: "Context") -> "Context":
        return context


class TupleIndex(MockObject):
    def __init__(self, obj_from: Tuple, index: int) -> None:
        super().__init__(obj_from, index)

    def _assign_args(self) -> None:
        self.obj_from, self.index = self.args

    def combinatorially_eq(self, o: CombinatoricsObject) -> bool:
        if isinstance(o, TupleIndex):
            return self.obj_from == o.obj_from and self.index == o.index
        return False

    def body_str(self) -> str:
        return f"{self.obj_from.name}[{self.index}]"


class TupleCount(SizedObject, MockObject):
    def __init__(self, tuple_obj: Tuple,
                 count_obj: Union[Set, Entity],
                 deduplicate: bool = False) -> None:
        super().__init__(tuple_obj, count_obj, deduplicate)

    def _assign_args(self) -> None:
        self.tuple_obj, self.count_obj, self.deduplicate = self.args

    def body_str(self) -> str:
        if self.deduplicate:
            return f"{self.tuple_obj.name}.dedup_count({self.count_obj.name})"
        else:
            return f"{self.tuple_obj.name}.count({self.count_obj.name})"


# =======================================
# Constraints
# =======================================
class TupleConstraint(AtomicConstraint):
    pass


class TupleIndexEqConstraint(TupleConstraint):
    def __init__(self, obj: TupleIndex,
                 entity: Entity) -> None:
        super().__init__(obj, entity)

    def _assign_args(self) -> None:
        self.obj, self.entity = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.obj} = {self.entity}"
        else:
            return f"{self.obj} != {self.entity}"


class TupleMembershipConstraint(TupleConstraint):
    def __init__(self, obj: Union[Tuple, Set, Bag],
                 member: Union[Entity, TupleIndex]) -> None:
        super().__init__(obj, member)

    def _assign_args(self) -> None:
        self.obj, self.member = self.args

    def __str__(self) -> str:
        if self.positive:
            return f"{self.member} ∈ {self.obj.name}"
        else:
            return f"{self.member} ∉ {self.obj.name}"
