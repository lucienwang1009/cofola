bag_grammar = r"""
    bag_declaration: BAG left_identity bag_object
    bag_object: left_parenthesis bag_object right_parenthesis -> parenthesis
        | identity
        | bag_init
        | operation_producing_bag

    bag_init: "{" bag_body "}"
    bag_body: entity_multiplicity ("," entity_multiplicity)*
    entity_multiplicity: entity colon INT

    operation_producing_bag: bag_operations
        | set_operations_bag

    ?bag_operations: bag_choose
        | bag_union
        | bag_additive_union
        | bag_intersection
        | bag_difference
    bag_choose: choose left_parenthesis bag_object ("," INT)? right_parenthesis
    bag_union: bag_object union_op bag_object
    bag_additive_union: bag_object additive_union_op bag_object
    bag_intersection: bag_object intersection_op bag_object
    bag_difference: bag_object difference_op bag_object

    ?bag_operations_set: bag_support
    bag_support: "support" left_parenthesis bag_object right_parenthesis

    ?bag_constraint: bag_size_constraint
        | bag_multiplicity_constraint
        | bag_membership_constraint
    bag_size_constraint: bag_size_expr comparator NUMBER
    ?bag_size_expr: left_parenthesis bag_size_expr right_parenthesis -> parenthesis
        | bag_size_atom -> bag_size_atomic_expr
        | bag_size_expr "+" bag_size_atom -> bag_size_add
        | bag_size_expr "-" bag_size_atom -> bag_size_sub
    bag_size_atom: "|" bag_object "|"
        | NUMBER "|" bag_object "|"
    bag_multiplicity_constraint: bag_object dot "count" left_parenthesis entity right_parenthesis comparator NUMBER
    bag_membership_constraint: entity "in" bag_object

    BAG: "bag"
"""
