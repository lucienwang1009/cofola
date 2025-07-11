set_grammar = r"""
    set_declaration: SET left_identity set_bag_object
    set_bag_object: left_parenthesis set_object right_parenthesis -> parenthesis
        | identity
        | set_init
        | operation_producing_set

    set_init: "{" set_body "}"
    ?set_body: list_entities | slicing_entities
    list_entities: entity ("," entity)*
    slicing_entities: CNAME_LETTER_END? INT colon INT

    operation_producing_set: set_operations
        | func_operation_set
        | bag_operations_set

    ?set_operations: set_choose
        | set_choose_replace
        | set_union
        | set_intersection
        | set_difference
        | set_cartesian_product
    set_choose: choose left_parenthesis set_object ("," INT)? right_parenthesis
    set_union: set_object union_op set_object
    set_intersection: set_object intersection_op set_object
    set_difference: set_object difference_op set_object
    set_cartesian_product: set_object cartesian_product_op set_object

    set_operations_bag: set_choose_replace
    set_choose_replace: choose_replace left_parenthesis set_object ("," INT)? right_parenthesis

    ?set_constraint: set_size_constraint
        | set_membership_constraint
        | subset_constraint
        | disjoint_constraint
        | set_equivalence_constraint
    set_size_constraint: set_size_expr comparator NUMBER
    ?set_size_expr: left_parenthesis set_size_expr right_parenthesis -> parenthesis
        | set_size_atom -> set_size_atomic_expr
        | set_size_expr "+" set_size_atom -> set_size_add
        | set_size_expr "-" set_size_atom -> set_size_sub
    set_size_atom: "|" set_object "|"
        | NUMBER "|" set_object "|"
    set_membership_constraint: entity "in" set_object
    subset_constraint: set_object "subset" set_object
    disjoint_constraint: set_object "disjoint" set_object
    set_equivalence_constraint: set_object "==" set_object

    SET: "set"
    CNAME_LETTER_END: CNAME* LETTER
"""
