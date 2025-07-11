function_grammar = r"""
    func_declaration: FUNC left_identity func_object
    func_object: left_parenthesis func_object right_parenthesis -> parenthesis
        | identity
        | func_init
        | operation_producing_func

    func_init: general_func_init
        | injective_func_init
        | surjective_func_init
        | bijective_func_init
    general_func_init: set_object mapping set_object
    injective_func_init: set_object injection set_object
    surjective_func_init: set_object surjection set_object
    bijective_func_init: set_object bijection set_object

    operation_producing_func: func_operation_func
    ?func_operation_func: func_composition
    func_composition: func_object composition_op func_object

    ?func_operation_set: func_image
        | func_inverse_image
        | func_constrained_image
    ?func_image: func_object left_parenthesis entity right_parenthesis -> func_image_entity
        | func_object left_square_bracket set_object right_square_bracket -> func_image_set
    ?func_inverse_image: func_object inverse_op left_parenthesis entity right_parenthesis -> func_inverse_image_entity
        | func_object inverse_op left_square_bracket set_object right_square_bracket -> func_inverse_image_set
    ?func_constrained_image: func_size_constrained_image
        | func_membership_constrained_image
        | func_subset_constrained_image
        | func_disjoint_constrained_image
        | func_equivalence_constrained_image
    func_size_constrained_image: "|" func_object inverse_op "|" comparator count_parameter
    func_membership_constrained_image: entity "in" func_object inverse_op
    ?func_subset_constrained_image: func_object inverse_op "subset" set_object -> func_subset_constrained_image_1
        | set_object "subset" func_object inverse_op -> func_subset_constrained_image_2
    func_disjoint_constrained_image: func_object inverse_op "disjoint" set_object
    func_equivalence_constrained_image: func_object inverse_op "==" set_object

    FUNC: "func"
    mapping: "->"
    injection: "|->"
    surjection: "->|"
    bijection: "|->|"
    inverse_op: "-1"
    composition_op: "."
"""
