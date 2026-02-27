from cofola.objects.base import AtomicConstraint, Bag, CombinatoricsObject, Entity, MockObject, Set, SizedObject, Tuple
from typing import Union, TYPE_CHECKING
from cofola.objects.function import FuncInit

from cofola.objects.set import SetInit

if TYPE_CHECKING:
    from cofola.context import Context


class TupleImpl(Tuple):
    _fields = ("obj_from", "choose", "replace", "size", "indices", "mapping")

    def __init__(self, obj_from: Union[Set, Bag], choose: bool = True,
                 replace: bool = True, size: int = None,
                 indices: SetInit = None, mapping: FuncInit = None) -> None:
        super().__init__(obj_from, choose, replace, size, indices, mapping)

    def _assign_fields(self) -> None:
        super()._assign_fields()
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
    _fields = ("obj_from", "index")

    def combinatorially_eq(self, o: CombinatoricsObject) -> bool:
        if isinstance(o, TupleIndex):
            return self.obj_from == o.obj_from and self.index == o.index
        return False

    def body_str(self) -> str:
        return f"{self.obj_from.name}[{self.index}]"


class TupleCount(SizedObject, MockObject):
    _fields = ("tuple_obj", "count_obj", "deduplicate")

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
    _fields = ("obj", "entity")

    def __str__(self) -> str:
        if self.positive:
            return f"{self.obj} = {self.entity}"
        else:
            return f"{self.obj} != {self.entity}"


class TupleMembershipConstraint(TupleConstraint):
    _fields = ("obj", "member")

    def __str__(self) -> str:
        if self.positive:
            return f"{self.member} ∈ {self.obj.name}"
        else:
            return f"{self.member} ∉ {self.obj.name}"
