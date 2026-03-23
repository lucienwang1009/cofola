import re

from wfomc import Pred, Expr
from sympy import Eq, var


AUX_PRED_PREFIX = '$cofola_aux_'


class _AuxPredCounter:
    """Encapsulated counter for generating unique auxiliary predicate names."""
    _count: int = 0

    @classmethod
    def next(cls) -> int:
        """Get the current counter value and increment it."""
        cnt = cls._count
        cls._count += 1
        return cnt

    @classmethod
    def reset(cls, start_from: int = 0) -> None:
        """Reset the counter to a specific starting value."""
        cls._count = start_from


def create_cofola_pred(name: str, arity: int) -> Pred:
    if not name.startswith('p_'):
        name = 'p_' + name
    return Pred(name, arity)

def create_cofola_var(name: str) -> Expr:
    if not name.startswith('v_'):
        name = 'v_' + name
    return var(name)

def reset_aux_pred_cnt(start_from: int = 0):
    """Reset the auxiliary predicate counter."""
    _AuxPredCounter.reset(start_from)

def create_aux_pred(arity: int, aux_pred_prefix: str = AUX_PRED_PREFIX) -> Pred:
    """Create a unique auxiliary predicate."""
    cnt = _AuxPredCounter.next()
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
