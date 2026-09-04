"""The dependency boundary between Cofola and the supported WFOMC API.

Cofola tracks the typed API from the WFOMC ``devel`` branch.  Keeping its
construction and solver calls here prevents dependency changes from leaking
through the encoder, decoder, and public solver modules without pretending to
support older, incompatible WFOMC releases.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, TypeAlias, TypeVar, cast

from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx
from sympy import Expr, Integer, Poly, Rational as SympyRational, sympify
from wfomc import (
    AlgoName as _NativeAlgoName,
    AlgoOptions as _NativeAlgoOptions,
    Domain as _NativeDomain,
    Evidence as _NativeEvidence,
    EvidenceStrategy as _NativeEvidenceStrategy,
    GroundUnaryLiteral as _NativeGroundUnaryLiteral,
    LinearOrderEncoding,
    Problem as _NativeProblem,
    ProblemInstance as _NativeProblemInstance,
    UnaryEvidence as _NativeUnaryEvidence,
    WFOMCResult,
    parse_formula as parse,
    solve as _native_solve,
)
from wfomc.fol import (
    Atom as _NativeAtom,
    Constant as Const,
    Formula,
    Not as _NativeNot,
    Predicate as Pred,
    true,
)


EncodedProblem: TypeAlias = _NativeProblemInstance
EvidenceFormula: TypeAlias = _NativeAtom | _NativeNot
Rational = Fraction
top = true()
_WeightKey = TypeVar("_WeightKey")


class Algo(Enum):
    """Algorithms exposed through Cofola's solver options."""

    STANDARD = "standard"
    FAST = "fast"
    FASTv2 = "fastv2"
    FASTV2 = "fastv2"
    INCREMENTAL = "incremental"
    INCREMENTAL3 = "incremental3"
    RECURSIVE = "recursive"
    PROPOSITIONAL = "propositional"

    def __str__(self) -> str:
        return self.value


class UnaryEvidenceStrategy(Enum):
    """Evidence choices exposed through Cofola's solver options."""

    AUTO = "auto"
    CCS = "ccs"

    def __str__(self) -> str:
        return self.value


def normalize_sentence(sentence: Formula) -> Formula:
    """Leave normalization to WFOMC's typed reduction pipeline."""

    return sentence


def evidence_parts(literal: EvidenceFormula) -> tuple[Pred, Const, bool]:
    """Return the predicate, constant, and polarity of unary evidence."""

    if isinstance(literal, _NativeAtom):
        atom = literal
        positive = True
    elif isinstance(literal, _NativeNot) and isinstance(literal.body, _NativeAtom):
        atom = literal.body
        positive = False
    else:
        raise TypeError(f"Expected a ground unary literal, got {literal!r}")
    if len(atom.terms) != 1 or not isinstance(atom.terms[0], Const):
        raise TypeError(f"Expected a ground unary literal, got {literal!r}")
    return atom.predicate, atom.terms[0], positive


def _coefficient(value: object) -> fmpq:
    coefficient = sympify(value)
    if coefficient.is_Rational is not True:
        raise TypeError(f"WFOMC weights must have rational coefficients: {value!r}")
    return fmpq(int(coefficient.p), int(coefficient.q))


def _native_weights(
    weights: Mapping[_WeightKey, tuple[object, object]],
) -> dict[_WeightKey, tuple[object, object]]:
    expressions = [sympify(value) for pair in weights.values() for value in pair]
    symbols = sorted(
        {symbol for expression in expressions for symbol in expression.free_symbols},
        key=str,
    )
    if not symbols:
        return {
            predicate: (
                Fraction(int(sympify(positive).p), int(sympify(positive).q)),
                Fraction(int(sympify(negative).p), int(sympify(negative).q)),
            )
            for predicate, (positive, negative) in weights.items()
        }

    arithmetic = fmpq_mpoly_ctx.get(tuple(map(str, symbols)), "lex")

    def convert(value: object) -> object:
        polynomial = Poly(sympify(value), *symbols)
        return arithmetic.from_dict(
            {
                tuple(monomial): _coefficient(coefficient)
                for monomial, coefficient in polynomial.terms()
            }
        )

    return {
        predicate: (convert(positive), convert(negative))
        for predicate, (positive, negative) in weights.items()
    }


def symbolic_ganak_count(
    n_vars: int,
    clauses: Iterable[Iterable[int]],
    weights: Mapping[int, tuple[object, object]],
) -> object:
    """Return the exact symbolic WMC of a local propositional encoding.

    Lifted bag compilation uses this adapter instead of depending on WFOMC's
    FLINT and Ganak integration directly. The caller can describe literal
    weights with the same SymPy expressions used by the outer Cofola encoding
    and receives a SymPy polynomial back.
    """
    from wfomc.ganak import ganak_count

    expressions = [sympify(value) for pair in weights.values() for value in pair]
    symbols = sorted(
        {symbol for expression in expressions for symbol in expression.free_symbols},
        key=str,
    )
    native_weights = _native_weights(weights)
    native_result = ganak_count(
        n_vars,
        clauses,
        cast(Any, native_weights),
        symbolic=bool(symbols),
        npolyvars=len(symbols),
        poly_ctx=(
            fmpq_mpoly_ctx.get(tuple(map(str, symbols)), "lex")
            if symbols
            else None
        ),
    )
    if isinstance(native_result, fmpq):
        return SympyRational(int(native_result.p), int(native_result.q))
    if not isinstance(native_result, fmpq_mpoly):
        raise TypeError(
            "Ganak returned an unsupported symbolic value: "
            f"{type(native_result).__name__}"
        )

    result = Integer(0)
    for monomial, coefficient in native_result.to_dict().items():
        term = SympyRational(int(coefficient.p), int(coefficient.q))
        for symbol, exponent in zip(symbols, monomial):
            term *= cast(Any, symbol) ** exponent
        result += term
    return result


def build_problem(
    sentence: Formula,
    domain: set[Const],
    weights: Mapping[Pred, tuple[object, object]],
    unary_evidence: set[EvidenceFormula],
) -> EncodedProblem:
    """Translate Cofola's encoded state to WFOMC's typed input model."""

    literals = tuple(
        _NativeGroundUnaryLiteral(predicate, constant, positive)
        for predicate, constant, positive in (
            evidence_parts(literal)
            for literal in sorted(unary_evidence, key=str)
        )
    )
    return _NativeProblemInstance(
        problem=_NativeProblem(
            sentence=sentence,
            weights=_native_weights(weights),
            evidence=_NativeEvidence(unary=_NativeUnaryEvidence(literals)),
        ),
        domain=_NativeDomain(elements=frozenset(domain)),
    )


def solve_problem(
    problem: EncodedProblem,
    algo: Algo,
    unary_evidence_strategy: UnaryEvidenceStrategy,
    linear_order_encoding: LinearOrderEncoding | str | None,
) -> WFOMCResult:
    evidence_strategy = (
        None
        if unary_evidence_strategy is UnaryEvidenceStrategy.AUTO
        else _NativeEvidenceStrategy.CCS
    )
    if isinstance(linear_order_encoding, str):
        linear_order_encoding = LinearOrderEncoding(linear_order_encoding)
    return _native_solve(
        problem,
        algo=_NativeAlgoName(algo.value),
        options=_NativeAlgoOptions(
            evidence_strategy=evidence_strategy,
            linear_order_encoding=linear_order_encoding,
        ),
    )


def contains_linear_order_axiom(problem: EncodedProblem) -> bool:
    """Whether Cofola encoded the distinguished ``LEQ`` predicate."""

    logical_problem = getattr(problem, "problem", problem)
    return any(
        str(predicate) == "LEQ"
        for predicate in logical_problem.sentence.preds()
    )


__all__ = [
    "Algo",
    "Const",
    "EncodedProblem",
    "EvidenceFormula",
    "Expr",
    "Formula",
    "LinearOrderEncoding",
    "Pred",
    "Rational",
    "UnaryEvidenceStrategy",
    "WFOMCResult",
    "build_problem",
    "contains_linear_order_axiom",
    "evidence_parts",
    "normalize_sentence",
    "parse",
    "solve_problem",
    "symbolic_ganak_count",
    "top",
]
