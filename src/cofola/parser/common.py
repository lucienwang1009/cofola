from __future__ import annotations

from lark import Transformer


common_grammar = r"""
    left_square_bracket: "["
    right_square_bracket: "]"
    left_parenthesis: "("
    right_parenthesis: ")"
    quantifier_variable: quantifier variable
    ?quantifier: universal_quantifier | existential_quantifier | counting_quantifier
    universal_quantifier: "\\forall"
    existential_quantifier: "\\exists"
    counting_quantifier: "\\exists_{" comparator count_parameter "}"
    constant: LCASE_CNAME
    variable: UCASE_LETTER
    predicate: CNAME
    NOT: "not"
    AND: "and"
    OR: "or"
    implies: "->"
    iff: "<->"
    ?comparator: EQUALITY | LE | GE | LT | GT | NEQUALITY
    EQUALITY: "=="
    NEQUALITY: "!="
    LE: "<="
    GE: ">="
    LT: "<"
    GT: ">"
    LCASE_CNAME: LCASE_LETTER ("_"|LCASE_LETTER|DIGIT)*
    ?bool: true | false
    true: "true" | "True"
    false: "false" | "False"

    %import common.LCASE_LETTER
    %import common.UCASE_LETTER
    %import common.CNAME
    %import common.LETTER
    %import common.DIGIT
    %import common.FLOAT
    %import common.INT
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.WS
    %import common.SH_COMMENT
    %ignore WS
    %ignore SH_COMMENT
"""


class CommonTransformer(Transformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)

    def parenthesis(self, args):
        return args[1]

    def equality(self, args):
          return '=='

    def nequality(self, args):
        return '!='

    def le(self, args):
        return '<='

    def ge(self, args):
        return '>='

    def lt(self, args):
        return '<'

    def gt(self, args):
        return '>'

    def true(self, args):
        return True

    def false(self, args):
        return False

    def INT(self, args):
        return int(args)
