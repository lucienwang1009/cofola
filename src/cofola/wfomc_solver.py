from sympy import Expr, Add, Mul, Number, Symbol, expand
from sympy.core.relational import Relational, Equality, StrictLessThan, LessThan, StrictGreaterThan, GreaterThan, Ne
from collections import defaultdict

from wfomc import CardinalityConstraint, Pred, wfomc, Algo, WFOMCProblem, UnaryEvidenceEncoding
from wfomc.utils import RingElement

from cofola.decoder import Decoder


def sympy_expr_to_cardinality_constraint(
    expr: Expr | Relational,
    sym2pred: dict[Symbol, str]
) -> tuple[dict[Pred, float], str, float]:
    """
    Convert a SymPy expression or relational to a cardinality constraint.
    Supports both forms:
    1. expr, comparator, rhs (e.g., 2*P + Q, '<=', 10)
    2. relational expression (e.g., 2*P + Q <= 10 or P <= Q)
    """
    if isinstance(expr, Relational):
        lhs = expr.lhs
        rhs = expr.rhs

        if isinstance(expr, Equality):
            comparator = '='
        elif isinstance(expr, Ne):
            comparator = '!='
        elif isinstance(expr, StrictLessThan):
            comparator = '<'
        elif isinstance(expr, LessThan):
            comparator = '<='
        elif isinstance(expr, StrictGreaterThan):
            comparator = '>'
        elif isinstance(expr, GreaterThan):
            comparator = '>='
        else:
            raise ValueError(f"Unsupported relational type: {type(expr)}")

        combined_expr = lhs - rhs
        return sympy_to_cardinality_constraint(
            combined_expr, comparator, 0.0, sym2pred)
    else:
        raise ValueError("Expected a SymPy Relational expression (e.g., x <= y)")


def sympy_to_cardinality_constraint(
    expr: Expr,
    comparator: str,
    rhs: float | Expr,
    sym2pred: dict[Symbol, str]
) -> tuple[dict[str, float], str, float]:
    """Convert a SymPy expression to a cardinality constraint."""

    def _get_pred_name(symbol: Symbol) -> str:
        """Helper function to get predicate name from symbol."""
        if symbol not in sym2pred:
            raise ValueError(f"Symbol {symbol} not found in sym2pred mapping.")
        return sym2pred[symbol]

    def _extract_term(term: Expr, coef_dict: dict[str, float]) -> float:
        """Helper function to extract coefficient and predicate name from a term."""
        if isinstance(term, Symbol):
            coef_dict[_get_pred_name(term)] += 1.0
            return None
        elif isinstance(term, Number) or term.is_number:
            return float(term)
        elif isinstance(term, Mul):
            coef = 1.0
            symbol = None
            for factor in term.args:
                if isinstance(factor, Number) or factor.is_number:
                    coef *= float(factor)
                elif isinstance(factor, Symbol):
                    if symbol is not None:
                        raise ValueError(f"Non-linear term detected: {term}. "
                                       "Cardinality constraints must be linear.")
                    symbol = _get_pred_name(factor)
                else:
                    raise ValueError(f"Unsupported factor type in term {term}: {type(factor)}")

            if symbol is not None:
                coef_dict[symbol] += coef
                return None
            else:
                return coef
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")

    expr = expand(expr)
    coef_dict = defaultdict(float)
    constant = 0.0

    if isinstance(expr, Add):
        for term in expr.args:
            const_term = _extract_term(term, coef_dict)
            if const_term is not None:
                constant += const_term
    elif isinstance(expr, (Mul, Symbol)):
        const_term = _extract_term(expr, coef_dict)
        if const_term is not None:
            constant += const_term
    elif isinstance(expr, Number) or expr.is_number:
        constant = float(expr)
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    rhs = float(rhs - constant)
    result_dict = {pred: coef for pred, coef in coef_dict.items() if coef != 0}
    return (result_dict, comparator, rhs)


def export(problem: WFOMCProblem, decoder: Decoder, filename: str) -> None:
    """Export the given problem to a file.

    Args:
        problem (WFOMCProblem): The problem to export.
        filename (str): The name of the file to export to.
    """
    weight2pred = dict()
    weights = dict()
    for pred, weight in problem.weights.items():
        if isinstance(weight[0], Symbol):
            weight2pred[weight[0]] = pred
            weights[pred] = (1, weight[1])
        else:
            weights[pred] = weight
    cc = list()
    for expr in decoder.validator:
        constraint = sympy_expr_to_cardinality_constraint(expr, weight2pred)
        cc.append(constraint)
    cc = CardinalityConstraint(cc)
    print(cc)
    problem = WFOMCProblem(
        domain=problem.domain,
        sentence=problem.sentence,
        weights=weights,
        cardinality_constraint=cc,
        unary_evidence=problem.unary_evidence,
        circle_len=problem.circle_len
    )
    problem.to_file(filename)


def solve(problem: WFOMCProblem, algo: Algo,
          use_partition_constraint: bool = False) -> RingElement:
    """Solve the given problem using the given algorithm.

    Args:
        problem (WFOMCProblem): The problem to solve.

    Returns:
        int: The number of models of the given problem.
    """
    if use_partition_constraint:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.PC)
    else:
        ret = wfomc(problem, algo, UnaryEvidenceEncoding.CCS)
    return ret
