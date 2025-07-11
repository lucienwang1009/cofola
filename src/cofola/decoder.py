from collections import defaultdict
from math import factorial, prod

from wfomc import coeff_dict, RingElement


class Decoder(object):
    def __init__(self, overcount: int,
                 var_gens: list[RingElement],
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

    def decode_result(self, result: RingElement) -> int:
        if result == 0:
            return 0

        if len(self.gens) == 0:
            return int(result / self.overcount)

        ret = 0
        for degrees, coeff in coeff_dict(result, self.gens):
            sat = True
            var2degree = dict(zip(self.gens, degrees))
            for v in self.validator:
                if not v.subs(var2degree):
                    sat = False
                    break
            if sat:
                # handle the overcount for partition
                overcount = 1
                for indis_vars in self.indis_vars:
                    n_confs = defaultdict(lambda : 0)
                    for vars_ in indis_vars:
                        n_confs[tuple(var2degree[v] for v in vars_)] += 1
                    # overcount *= (factorial(len(indis_vars)) / prod(
                    #     factorial(n) for n in n_confs.values()
                    # ))
                    # print(overcount)
                    # if the degrees of k tuples of variables are the same, we need to divide the overcount by k!
                    # note that here we need to be careful about the degrees of 0
                    overcount *= prod(
                        factorial(n) for k, n in n_confs.items() if sum(k) > 0
                    )
                ret = ret + coeff / overcount
        return int(ret / self.overcount)
