from cofola.parser.common import common_grammar


cofola_grammar = r"""
    cofola: statement*
    ?statement: object_declaration | constraint

    object_declaration: left_identity object
    ?object: left_parenthesis object right_parenthesis -> parenthesis
        | base_object_init
        | identity
        | operations

    base_object_init: BASE_OBJ_TYPE left_parenthesis (base_object_args | object) right_parenthesis
    ?base_object_args: entities_body
    entities_body: entity_atom ("," entity_atom)*
    ?entity_atom: entity | slicing_entities | duplicate_entities
    slicing_entities: CNAME_LETTER_END? INT "..." INT
    duplicate_entities: entity ":" INT

    operations: common_operations
        | binary_operations
        | indexing
    common_operations: (CHOOSE
        | CHOOSE_REPLACE
        | SUPP
        | COMPOSE
        | PARTITION
        | PERMUTE
        | CHOOSE_PERMUTE
        | CHOOSE_REPLACE_PERMUTE) left_parenthesis object ("," INT)? ("," "reflection" "=" bool)? right_parenthesis
    binary_operations: object (UNION_OP | ADDITIVE_UNION_OP | INTERSECTION_OP | DIFFERENCE_OP) object
    indexing: object left_square_bracket INT right_square_bracket

    ?constraint: left_parenthesis constraint right_parenthesis -> parenthesis
        | atomic_constraint
        | compound_constraint
        | part_constraint
    part_constraint: atomic_constraint FOR CNAME IN object
    ?atomic_constraint: size_constraint
        | membership_constraint
        | subset_constraint
        | disjoint_constraint
        | equivalence_constraint
        | seq_constraint
    ?compound_constraint: negation_constraint | binary_constraint
    negation_constraint: NOT constraint
    binary_constraint: constraint (AND | OR) constraint

    size_constraint: size_expr comparator NUMBER
    ?size_expr: left_parenthesis size_expr right_parenthesis -> parenthesis
        | size_atom -> size_atomic_expr
        | size_expr "+" size_atom -> size_add
        | size_expr "-" size_atom -> size_sub
    size_atom: "|" object "|"
        | NUMBER "|" object "|"
        | count
    count: object "." (COUNT | COUNT_DEDUPLICATE) left_parenthesis (object | entity | seq_pattern) right_parenthesis
    membership_constraint: (object | entity) in_or_not object
    subset_constraint: object SUBSET object
    disjoint_constraint: object DISJOINT object
    equivalence_constraint: object (NEQUALITY | EQUALITY) (object | entity)
    seq_constraint: seq_pattern in_or_not object
    seq_pattern: left_parenthesis seq_pattern right_parenthesis -> parenthesis
        | together
        | less_than
        | next_to
        | predecessor
    together: TOGETHER left_parenthesis object right_parenthesis
    less_than: (object | entity) LT (object | entity)
    next_to: NEXT_TO left_parenthesis (object | entity) "," (object | entity) right_parenthesis
    predecessor: left_parenthesis (object | entity) "," (object | entity) right_parenthesis

    BASE_OBJ_TYPE: SET | BAG
    SET: "set"
    BAG: "bag"
    COMPOSE: "compose"
    PARTITION: "partition"
    CNAME_LETTER_END: CNAME* LETTER
    left_identity: CNAME "="
    identity: CNAME
    CHOOSE: "choose"
    CHOOSE_REPLACE: "choose_replace"
    PERMUTE: "tuple" | "sequence" | "circle"
    CHOOSE_PERMUTE: "choose_tuple" | "choose_sequence" | "choose_circle"
    CHOOSE_REPLACE_PERMUTE: "choose_replace_tuple" | "choose_replace_sequence" | "choose_replace_circle"
    SUPP: "supp"
    UNION_OP: "+"
    ADDITIVE_UNION_OP: "++"
    INTERSECTION_OP: "&"
    DIFFERENCE_OP: "-"
    entity: INT | CNAME
    SUBSET: "subset"
    FOR: "for"
    in_or_not: IN | NOT IN
    IN: "in"
    TOGETHER: "together"
    DISJOINT: "disjoint"
    NEXT_TO: "next_to"
    COUNT: "count"
    COUNT_DEDUPLICATE: "dedup_count"
    count_parameter: INT
"""

grammar = cofola_grammar + common_grammar
