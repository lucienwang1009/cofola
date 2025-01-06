import pickle
from sympy.logic import And, Or, Not, Implies, Equivalent

from wfomc import QFFormula, QuantifiedFormula


with open('./wfomc_problems.pkl', 'rb') as f:
    problems = pickle.load(f)


def to_str(formula):
    if isinstance(formula, QuantifiedFormula):
        return f"{formula.quantifier_scope}: ({to_str(formula.quantified_formula)})"
    if isinstance(formula, QFFormula):
        return to_str(formula.expr)
    if isinstance(formula, And):
        return "(" + " & ".join(to_str(arg) for arg in formula.args) + ")"
    if isinstance(formula, Or):
        return "(" + " | ".join(to_str(arg) for arg in formula.args) + ")"
    if isinstance(formula, Not):
        return f"~({to_str(formula.args[0])})"
    if isinstance(formula, Implies):
        return f"({to_str(formula.args[0])} -> {to_str(formula.args[1])})"
    if isinstance(formula, Equivalent):
        return f"({to_str(formula.args[0])} <-> {to_str(formula.args[1])})"
    return str(formula)


def problem_to_file(problem, filename):
    domain = problem.domain
    sentence = problem.sentence
    uni_formula = sentence.uni_formula
    ext_formula = sentence.ext_formulas
    cardinality_constraint = problem.cardinality_constraint
    unary_evidence = problem.unary_evidence
    s = f"{to_str(uni_formula)}\n"
    for f in ext_formula:
        s += f"& {to_str(f)}\n"
    s += "\n"
    s += f"domain = {domain}\n\n"
    for expr, comp, param in cardinality_constraint.constraints:
        s += " ".join(f"{coef} |{pred.name}|" for pred, coef in expr.items())
        s += f" {comp} {param}\n"
    s += "\n"
    s += ", ".join(str(atom) for atom in unary_evidence)
    with open(filename, 'w') as f:
        f.write(s)


for i, problem in problems.items():
    problem = problem
    problem_to_file(problem, f"./wfomc_problems/new_pred/{i}.wfomcs")

