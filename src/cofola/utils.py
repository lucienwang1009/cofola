import re

from wfomc import Pred, Expr
from sympy import Eq, var

from cofola.objects.base import Bag, CombinatoricsObject, Entity, Partition, Set, Function


OBJ_PRE_PREFIX_ARITY = {
    Entity: ('p_entity_', 1),
    Set: ('p_set_', 1),
    Bag: ('p_bag_', 1),
    Function: ('p_func_', 2),
    Partition: ('p_partition_', 1),
}

AUX_PRED_PREFIX = '$cofola_aux_'
AUX_PRED_CNT = 0


def create_cofola_pred(name: str, arity: int) -> Pred:
    if not name.startswith('p_'):
        name = 'p_' + name
    return Pred(name, arity)

def create_cofola_var(name: str) -> Expr:
    if not name.startswith('v_'):
        name = 'v_' + name
    return var(name)

def get_prefix_arity(obj: CombinatoricsObject) -> str:
    for t in OBJ_PRE_PREFIX_ARITY:
        if isinstance(obj, t):
            return OBJ_PRE_PREFIX_ARITY[t]

def create_pred_for_object(obj: CombinatoricsObject) -> Pred:
    pre, arity = get_prefix_arity(obj)
    return create_cofola_pred(pre + obj.name, arity)

def reset_aux_pred_cnt(start_from: int = 0):
    global AUX_PRED_CNT
    AUX_PRED_CNT = start_from

def create_aux_pred(arity: int, aux_pred_prefix: str = AUX_PRED_PREFIX) -> Pred:
    global AUX_PRED_CNT
    cnt = AUX_PRED_CNT
    AUX_PRED_CNT += 1
    return create_cofola_pred(f"{aux_pred_prefix}_" + str(cnt), arity)

def get_type_name(obj: object):
    return to_snake_case(type(obj).__name__)

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


class ListLessThan(object):
    def __init__(self, left: list[Expr], right: list[Expr]) -> None:
        super().__init__()
        self.left: list[Expr] = left
        self.right: list[Expr] = right
        self.less_than_comparators = [left[i] < right[i] for i in range(len(left))]
        self.equal_comparators = [Eq(left[i], right[i]) for i in range(len(left))]

    def __str__(self) -> str:
        return f"{self.left} < {self.right}"

    def __repr__(self) -> str:
        return str(self)

    def subs(self, mappings) -> bool:
        for i, less_than in enumerate(self.less_than_comparators):
            if less_than.subs(mappings):
                return True
            if self.equal_comparators[i].subs(mappings):
                if i == len(self.left) - 1:
                    return True
                else:
                    continue
            else:
                return False


class CofolaUnSATProblemError(Exception):
    pass
