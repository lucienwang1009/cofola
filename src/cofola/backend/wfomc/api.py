"""The single compatibility boundary between Cofola and the WFOMC package.

Cofola builds formulas and symbolic weights, but it should not depend on the
internal staging types of WFOMC.  This module translates that small, stable
surface to either the legacy API pinned in older ``uv.lock`` files or the
typed API exposed by the current WFOMC ``devel`` branch.
"""
from __future__ import annotations

from enum import Enum
from fractions import Fraction
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, TypeAlias, cast

from sympy import Expr, Poly, sympify

if TYPE_CHECKING:
    from wfomc import LinearOrderEncoding
else:
    try:
        from wfomc import LinearOrderEncoding
    except ImportError:
        try:
            LinearOrderEncoding = import_module("wfomc.algo").LinearOrderEncoding
        except AttributeError:
            class LinearOrderEncoding(Enum):
                """Order encodings unavailable in pre-propositional WFOMC."""

                PIN = "pin"
                AXIOMS = "axioms"


EncodedProblem: TypeAlias = Any
EvidenceFormula: TypeAlias = Any


def _numeric_value(value: object) -> Fraction | float | None:
    """Convert a scalar returned by any supported WFOMC version."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return value

    numerator = getattr(value, "p", None)
    denominator = getattr(value, "q", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))
    return None


class WFOMCResult(object):
    """Version-independent view of a WFOMC solver result.

    Typed WFOMC releases return their own ``WFOMCResult`` wrapper, while
    legacy releases return a FLINT scalar or multivariate polynomial directly.
    Cofola wraps either representation here so the decoder never imports or
    inspects version-specific result classes.
    """

    def __init__(self, value: object) -> None:
        self._value = value.raw if isinstance(value, WFOMCResult) else value

    @property
    def raw(self) -> object:
        return self._value

    def is_zero(self) -> bool:
        check = getattr(self._value, "is_zero", None)
        return bool(check()) if callable(check) else self._value == 0

    def is_polynomial(self) -> bool:
        check = getattr(self._value, "is_polynomial", None)
        if callable(check):
            return bool(check())
        return callable(getattr(self._value, "context", None)) and callable(
            getattr(self._value, "terms", None)
        )

    def is_constant(self) -> bool:
        check = getattr(self._value, "is_constant", None)
        if callable(check):
            return bool(check())
        return _numeric_value(self._value) is not None

    def constant_value(self) -> Fraction | float | None:
        get_constant = getattr(self._value, "constant_value", None)
        if callable(get_constant):
            constant = get_constant()
            if constant is None:
                return None
            normalized = _numeric_value(constant)
            if normalized is None:
                raise TypeError(f"Unsupported constant type: {type(constant)}")
            return normalized

        scalar = _numeric_value(self._value)
        if scalar is not None:
            return scalar
        if self.is_polynomial() and self.is_constant():
            leading_coefficient = getattr(self._value, "leading_coefficient", None)
            if callable(leading_coefficient):
                return _numeric_value(leading_coefficient())
        return None

    def variable_names(self) -> tuple[str, ...]:
        get_names = getattr(self._value, "variable_names", None)
        if callable(get_names):
            return tuple(str(name) for name in cast(Iterable[object], get_names()))
        if not self.is_polynomial():
            return ()
        get_context = getattr(self._value, "context")
        return tuple(get_context().names())

    def terms(self) -> Iterator[tuple[tuple[int, ...], Fraction | float]]:
        get_terms = getattr(self._value, "terms", None)
        if not callable(get_terms):
            raise TypeError(f"Unsupported result type: {type(self._value)}")
        raw_terms = cast(Iterable[tuple[Iterable[int], object]], get_terms())
        for degrees, coefficient in raw_terms:
            normalized = _numeric_value(coefficient)
            if normalized is None:
                raise TypeError(f"Unsupported coefficient type: {type(coefficient)}")
            yield tuple(int(degree) for degree in degrees), normalized

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"WFOMCResult({self._value!r})"


class Algo(Enum):
    """Algorithms Cofola is prepared to request from WFOMC."""

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
    """Evidence choices exposed by Cofola's public solver options."""

    AUTO = "auto"
    CCS = "ccs"

    def __str__(self) -> str:
        return self.value


try:
    from wfomc import (
        AlgoName as _NativeAlgoName,
        AlgoOptions as _NativeAlgoOptions,
        Evidence as _NativeEvidence,
        EvidenceStrategy as _NativeEvidenceStrategy,
        GroundUnaryLiteral as _NativeGroundUnaryLiteral,
        Problem as _NativeProblem,
        UnaryEvidence as _NativeUnaryEvidence,
        parse_formula as parse,
        solve as _native_solve,
    )
except ImportError:
    from wfomc import (
        Algo as _LegacyAlgo,
        AtomicFormula as _LegacyAtomicFormula,
        Const,
        Formula,
        Pred,
        WFOMCProblem as _LegacyProblem,
        X,
        exactly_one_qf as _legacy_exactly_one_qf,
        exclusive,
        fol_parse as parse,
        to_sc2 as _legacy_to_sc2,
        top,
        wfomc as _legacy_solve,
    )

    try:
        from wfomc import UnaryEvidenceStrategy as _LegacyUnaryEvidenceStrategy
    except ImportError:
        from wfomc import UnaryEvidenceEncoding as _LegacyUnaryEvidenceEncoding

        _LegacyUnaryEvidenceStrategy = None
    else:
        _LegacyUnaryEvidenceEncoding = None

    USING_NATIVE_API = False
else:
    from flint import fmpq, fmpq_mpoly_ctx
    from wfomc.fol import (
        Atom as _NativeAtom,
        Constant as Const,
        Formula,
        Not as _NativeNot,
        Predicate as Pred,
        X,
        conjunction,
        disjunction,
        forall,
        true,
    )

    try:
        from wfomc import (
            Domain as _NativeDomain,
            ProblemInstance as _NativeProblemInstance,
        )
    except ImportError:
        _NativeDomain = None
        _NativeProblemInstance = None

    USING_NATIVE_API = True
    top = true()


Rational = Fraction


if USING_NATIVE_API:

    def normalize_sentence(sentence: Formula) -> Formula:
        """Leave normalization to the typed WFOMC reduction pipeline."""

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
        weights: Mapping[Pred, tuple[object, object]],
    ) -> dict[Pred, tuple[object, object]]:
        expressions = [sympify(value) for pair in weights.values() for value in pair]
        symbols = sorted(
            {symbol for expression in expressions for symbol in expression.free_symbols},
            key=str,
        )
        if not symbols:
            return {
                predicate: tuple(
                    Fraction(int(expression.p), int(expression.q))
                    for expression in map(sympify, pair)
                )
                for predicate, pair in weights.items()
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


    def build_problem(
        sentence: Formula,
        domain: set[Const],
        weights: Mapping[Pred, tuple[object, object]],
        unary_evidence: set[EvidenceFormula],
        *,
        circle_len: int,
    ) -> EncodedProblem:
        """Translate Cofola's encoded state to the typed WFOMC input model."""

        literals = tuple(
            _NativeGroundUnaryLiteral(predicate, constant, positive)
            for predicate, constant, positive in (
                evidence_parts(literal)
                for literal in sorted(unary_evidence, key=str)
            )
        )
        native_weights = _native_weights(weights)
        native_evidence = _NativeEvidence(unary=_NativeUnaryEvidence(literals))
        if _NativeDomain is not None and _NativeProblemInstance is not None:
            return _NativeProblemInstance(
                problem=_NativeProblem(
                    sentence=sentence,
                    weights=native_weights,
                    evidence=native_evidence,
                ),
                domain=_NativeDomain(
                    elements=frozenset(domain),
                    circular_order_size=circle_len,
                ),
            )
        return cast(Any, _NativeProblem)(
            sentence=sentence,
            domain=frozenset(domain),
            weights=native_weights,
            evidence=native_evidence,
            circular_order_size=circle_len,
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
        return WFOMCResult(
            _native_solve(
                problem,
                algo=_NativeAlgoName(algo.value),
                options=_NativeAlgoOptions(
                    evidence_strategy=evidence_strategy,
                    linear_order_encoding=linear_order_encoding,
                ),
            )
        )


    def exactly_one_qf(predicates: list[Pred]) -> Formula:
        if not predicates:
            raise ValueError("exactly_one_qf requires at least one predicate")
        literals = tuple(predicate(X) for predicate in predicates)
        if len(literals) == 1:
            return literals[0]
        pairwise = tuple(
            ~(left & right)
            for index, left in enumerate(literals)
            for right in literals[index + 1 :]
        )
        return disjunction(*literals) & conjunction(*pairwise)


    def exclusive(predicates: list[Pred]) -> Formula:
        if len(predicates) == 1:
            return top
        if not predicates:
            raise ValueError("exclusive requires at least one predicate")
        literals = tuple(predicate(X) for predicate in predicates)
        return forall(
            X,
            conjunction(
                *(
                    ~(left & right)
                    for index, left in enumerate(literals)
                    for right in literals[index + 1 :]
                )
            ),
        )

else:

    def exactly_one_qf(predicates: list[Pred]) -> Formula:
        if not predicates:
            raise ValueError("exactly_one_qf requires at least one predicate")
        if len(predicates) == 1:
            return predicates[0](X)
        return _legacy_exactly_one_qf(predicates)


    def normalize_sentence(sentence: Formula) -> Formula:
        return _legacy_to_sc2(sentence)


    def evidence_parts(literal: EvidenceFormula) -> tuple[Pred, Const, bool]:
        if not isinstance(literal, _LegacyAtomicFormula) or len(literal.args) != 1:
            raise TypeError(f"Expected a ground unary literal, got {literal!r}")
        return literal.pred, literal.args[0], literal.positive


    def build_problem(
        sentence: Formula,
        domain: set[Const],
        weights: Mapping[Pred, tuple[object, object]],
        unary_evidence: set[EvidenceFormula],
        *,
        circle_len: int,
    ) -> EncodedProblem:
        return _LegacyProblem(
            sentence,
            domain,
            weights,
            unary_evidence=unary_evidence,
            circle_len=circle_len,
        )


    def solve_problem(
        problem: EncodedProblem,
        algo: Algo,
        unary_evidence_strategy: UnaryEvidenceStrategy,
        linear_order_encoding: LinearOrderEncoding | str | None,
    ) -> WFOMCResult:
        legacy_algo = _LegacyAlgo(algo.value)
        if _LegacyUnaryEvidenceStrategy is None:
            encoding_name = (
                "PC"
                if unary_evidence_strategy is UnaryEvidenceStrategy.AUTO
                and algo in (Algo.FASTv2, Algo.INCREMENTAL)
                else "CCS"
            )
            assert _LegacyUnaryEvidenceEncoding is not None
            return WFOMCResult(
                _legacy_solve(
                    problem,
                    legacy_algo,
                    getattr(_LegacyUnaryEvidenceEncoding, encoding_name),
                )
            )

        return WFOMCResult(
            _legacy_solve(
                problem,
                legacy_algo,
                unary_evidence_strategy=_LegacyUnaryEvidenceStrategy(
                    unary_evidence_strategy.value
                ),
                linear_order_encoding=linear_order_encoding,
            )
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
    "USING_NATIVE_API",
    "WFOMCResult",
    "build_problem",
    "contains_linear_order_axiom",
    "evidence_parts",
    "exactly_one_qf",
    "exclusive",
    "normalize_sentence",
    "parse",
    "solve_problem",
    "top",
]
