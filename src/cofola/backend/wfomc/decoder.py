import sympy
from collections import defaultdict
from math import factorial, prod

from cofola.backend.wfomc.api import Expr, WFOMCResult

from cofola.backend.wfomc.utils import ListLessThan


class Decoder(object):
    def __init__(self, overcount: int,
                 var_gens: list[Expr],
                 validator: list, indis_vars: list):
        self.overcount: int = overcount
        self.gens = var_gens
        self.validator = validator
        # A list representing the indistinguishable variables, used for deduplicating the encoding of partition
        # Each item in the list is a list of tuples of symbolic variables, in which the tuples are indisguishable
        # For example, if one item is [[(x1, x2), (x3, x4)]], it means that (x1, x2) and (x3, x4) are indistinguishable
        # I.e., if the degrees of x1 and x2 are the same as the degrees of x3 and x4, they are considered the same
        self.indis_vars: list = indis_vars

    def __str__(self) -> str:
        s = ''
        s += f'Overcount: {self.overcount}\n'
        s += 'Variables: \n'
        s += '\t' + str(self.gens) + '\n'
        s += 'Validator: \n'
        for v in self.validator:
            s += '\t' + str(v) + '\n'
        return s

    def __repr__(self) -> str:
        return str(self)

    def decode_result(self, result: WFOMCResult) -> int:
        if result.is_zero():
            return 0

        if result.is_constant():
            zero_degrees = {generator: 0 for generator in self.gens}
            if not self._constant_validators_accept(zero_degrees):
                return 0
            constant = result.constant_value()
            if constant is None:
                return 0
            return int(constant / self.overcount)

        if not result.is_polynomial():
            return 0

        ret = 0
        ret_gens = result.variable_names()
        # Symbols (from self.gens) that appear in the result polynomial, ordered
        # to match the degree tuples returned by result.terms().
        reordered_gens = list()
        for v_name in ret_gens:
            for v in self.gens:
                if str(v) == v_name:
                    reordered_gens.append(v)
                    break

        # A gen absent from the result polynomial has degree 0 in every term
        # (its weighted predicate is unsatisfiable / was optimized away). A
        # validator referencing it would otherwise carry a dangling free symbol
        # and misevaluate. Substitute those gens with 0 once, up front, so the
        # per-term loop stays as cheap as evaluating over the raw degree tuple.
        present = set(reordered_gens)
        absent = {g: 0 for g in self.gens if g not in present}
        lambdified_validator = [
            sympy.lambdify(reordered_gens, v.subs(absent) if absent else v, 'math')
            for v in self.validator if not isinstance(v, ListLessThan)
        ]
        list_less_than_validator = [
            v for v in self.validator if isinstance(v, ListLessThan)
        ]
        for degrees, coeff in result.terms():
            # if all(v.subs(var2degree) for v in self.validator):
            if all(v(*degrees) for v in lambdified_validator):
                var2degree = dict(
                    zip(reordered_gens, list(int(d) for d in degrees))
                )
                var2degree.update(absent)  # absent gens have degree 0
                if len(list_less_than_validator) > 0 and any(
                    not v.subs(var2degree)
                    for v in list_less_than_validator
                ):
                    continue
                # handle the overcount for partition
                overcount = 1
                for indis_vars in self.indis_vars:
                    n_confs = defaultdict(lambda : 0)
                    for vars_ in indis_vars:
                        n_confs[tuple(var2degree[v] for v in vars_)] += 1
                    # overcount *= (factorial(len(indis_vars)) / prod(
                    #     factorial(n) for n in n_confs.values()
                    # ))
                    # if the degrees of k tuples of variables are the same, we need to divide the overcount by k!
                    # note that here we need to be careful about the degrees of 0
                    overcount *= prod(
                        factorial(n) for k, n in n_confs.items() if sum(k) > 0
                    )
                ret = ret + coeff / overcount
        return int(ret / self.overcount)

    def _constant_validators_accept(self, degrees: dict | None = None) -> bool:
        """Evaluate validators when WFOMC produced a constant result."""
        degrees = degrees or {}
        for validator in self.validator:
            if isinstance(validator, ListLessThan):
                if not validator.subs(degrees):
                    return False
            elif not bool(validator.subs(degrees)):
                return False
        return True
