
from collections import defaultdict
from lark import Lark
from logzero import logger
from cofola.objects.bag import Bag, BagAdditiveUnion, BagChoose, \
    BagDifference, BagEqConstraint, BagInit, BagIntersection, BagMultiplicity, BagSubsetConstraint, \
    SizeConstraint, BagSupport

from cofola.objects.base import CombinatoricsConstraint, CombinatoricsObject, Entity, Partition, Sequence, SizedObject, Tuple
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, Function
from cofola.objects.partition import BagPartition, SetPartition
from cofola.objects.sequence import LessThanPattern, NextToPattern, PredecessorPattern, SequenceConstraint, SequenceImpl, SequencePattern, SequencePatternCount, SequenceSizedPattern, TogetherPattern
from cofola.objects.set import MembershipConstraint, SetChoose, SetDifference, SetChooseReplace, \
    DisjointConstraint, SetEqConstraint, SetInit, SetIntersection, \
    SubsetConstraint, SetUnion, Set
from cofola.objects.tuple import TupleCount, TupleImpl, TupleIndexEqConstraint, TupleMembershipConstraint, TupleIndex
from cofola.parser.common import CommonTransformer
from cofola.problem import CofolaProblem, simplify
from cofola.parser.grammar import grammar


RESERVED_KEYWORDS = [
    "set",
    "bag",
    "choose",
    "choose_replace",
    "count",
    "in",
    "subset",
    "disjoint",
    "supp",
    'compose',
    'partition',
    'tuple',
    'choose_tuple',
    'choose_replace_tuple',
    'sequence',
    'choose_sequence',
    'choose_replace_sequence',
    'together',
    'not',
    'and',
    'or',
]

RESERVED_PREFIXES = [
    'AUX_',
    'IDX_'
]


class CofolaParsingError(Exception):
    pass


class CofolaTypeMismatchError(CofolaParsingError):
    def __init__(self, expected_types, actual):
        if isinstance(expected_types, tuple):
            expected_types = [t.__name__ for t in expected_types]
            expected_type = " or ".join(expected_types)
        else:
            expected_type = expected_types.__name__
        super().__init__(f"Expect a {expected_type} object, but got {actual} of type {type(actual)}.")


class CofolaTransfomer(CommonTransformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        # Here, `objs` contains all the objects that have been defined.
        # `id2obj` contains the mapping from object id to object reference.
        # You can think of objs as a list of object specifications, and id2obj as a dictionary of objects references.
        self.problem: CofolaProblem = CofolaProblem()
        self.id2obj: dict[str, CombinatoricsObject] = dict()
        self._processing_partition: Partition = None
        self._processing_part_name: str = None

    def left_identity(self, args):
        obj_id = str(args[0].value)
        self._check_id(obj_id)
        return obj_id

    def object_declaration(self, args):
        obj_id, obj = args
        self._attach_obj(obj_id, obj)

    def base_object_init(self, args):
        obj_type, _, obj_init, _ = args
        if obj_type == "set":
            entities = []
            for atom in obj_init:
                self._check_obj_type(atom, Entity)
                entities.append(atom)
            # check if entities are duplicated
            if len(entities) != len(set(entities)):
                raise CofolaParsingError(f"Duplicate entities are not allowed in a set object.")
            obj = SetInit(entities)
        if obj_type == "bag":
            entity_multiplicity = defaultdict(lambda: 0)
            for atom in obj_init:
                if isinstance(atom, tuple):
                    entity, multiplicity = atom
                    entity_multiplicity[entity] = multiplicity
                else:
                    entity_multiplicity[atom] += 1
            obj = BagInit(entity_multiplicity)
        obj = self.problem.add_object(obj)
        return obj

    def entities_body(self, args):
        entities = []
        for atom in args:
            if isinstance(atom, Entity):
                entities.append(atom)
            elif isinstance(atom, list):
                entities.extend(atom)
            else:
                entities.append(atom)
        return entities

    def slicing_entities(self, args):
        name = ""
        start = 0
        end = 0
        if len(args) == 2:
            start = args[0]
            end = args[1]
        elif len(args) == 3:
            name = args[0].value
            start = args[1]
            end = args[2]
        entities = list(
            Entity(f'{name}{i}') for i in range(start, end)
        )
        return entities

    def duplicate_entities(self, args):
        entity, count = args
        return entity, int(count)

    def entity(self, args):
        item = args[0]
        entity_name = str(item)
        if entity_name in self.id2obj:
            raise CofolaParsingError(f"Entity {entity_name} has already used as an object name.")
        ret = Entity(entity_name)
        return ret

    def func_init(self, args):
        domain, func_type, codomain = args
        injection = ((func_type == "|->") or (func_type == "|->|"))
        surjection = ((func_type == "->|") or (func_type == "|->|"))
        obj = FuncInit(domain, codomain, injection, surjection)
        return self.problem.add_object(obj)

    def identity(self, args):
        obj_id = str(args[0].value)
        # if the object is a part of a partition,
        # return the partitioned objects
        if self._processing_part_name is not None and \
                obj_id == self._processing_part_name:
            ret_objs = []
            for part in self._processing_partition.partitioned_objs:
                ret_objs.append(self.problem.add_object(part))
            return ret_objs
        if self.problem.contains_entity(Entity(obj_id)):
            return Entity(obj_id)
        return self._get_obj_by_id(obj_id)

    def operations(self, args):
        objs = args[-1]
        ret_objs = []
        if isinstance(objs, list):
            for obj in objs:
                ret_objs.append(self.problem.add_object(obj))
            return ret_objs
        return self.problem.add_object(objs)

    def common_operations(self, args):
        op_type = args[0]
        obj = args[2]
        self._check_obj_type(obj, Set, Bag)
        if len(args) == 4:
            size = None
            op_arg = False
        elif len(args) == 5:
            if isinstance(args[3], bool):
                size = None
                op_arg = args[3]
            else:
                size = args[3]
                op_arg = False
        else:
            size = args[3]
            op_arg = args[4]

        if op_type == "choose":
            if isinstance(obj, Set):
                return SetChoose(obj, size)
            else:
                return BagChoose(obj, size)
        if op_type == 'choose_replace':
            # operation `choose_replace` only supports Set object
            self._check_obj_type(obj, Set)
            return SetChooseReplace(obj, size)
        if op_type == 'supp':
            return BagSupport(obj)
        if op_type == 'compose':
            if size is None:
                raise CofolaParsingError(f"The size of a composition must be specified.")
            if isinstance(obj, Set):
                return SetPartition(obj, size, True)
            else:
                return BagPartition(obj, size, True)
        if op_type == 'partition':
            if size is None:
                raise CofolaParsingError(f"The size of a partition must be specified.")
            if isinstance(obj, Set):
                return SetPartition(obj, size, False)
            else:
                return BagPartition(obj, size, False)
        if op_type == 'tuple':
            return TupleImpl(
                obj, size=size, choose=False, replace=False
            )
        if op_type == 'choose_tuple':
            return TupleImpl(
                obj, size=size, choose=True, replace=False
            )
        if op_type == 'choose_replace_tuple':
            self._check_obj_type(obj, Set)
            return TupleImpl(
                obj, size=size, choose=True, replace=True
            )
        if op_type == 'sequence':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=False, replace=False
            )
        if op_type == 'choose_sequence':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=True, replace=False
            )
        if op_type == 'choose_replace_sequence':
            self._check_obj_type(obj, Set)
            return SequenceImpl(
                obj, size=size, choose=True, replace=True
            )
        if op_type == 'circle':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=False, replace=False,
                circular=True, reflection=op_arg
            )
        if op_type == 'choose_circle':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=True, replace=False,
                circular=True, reflection=op_arg
            )
        if op_type == 'choose_replace_circle':
            self._check_obj_type(obj, Set)
            return SequenceImpl(
                obj, size=size, choose=True, replace=True,
                circular=True, reflection=op_arg
            )

    def binary_operations(self, args):
        objs1, op, objs2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, Bag)
            self._check_obj_type(obj2, Set, Bag)
            if (isinstance(obj1, Set) and isinstance(obj2, Bag)) or \
                    (isinstance(obj1, Bag) and isinstance(obj2, Set)):
                raise CofolaParsingError(f"Set and Bag objects cannot be operated together")
            if isinstance(obj1, Set):
                if op == "+":
                    return SetUnion(obj1, obj2)
                elif op == "++":
                    raise CofolaParsingError(f"Additive union operation is not supported for set objects.")
                elif op == "&":
                    return SetIntersection(obj1, obj2)
                elif op == "-":
                    return SetDifference(obj1, obj2)
                else:
                    raise CofolaParsingError(f"Unknown set operation {op}.")
            else:
                if op == "+":
                    return BagAdditiveUnion(obj1, obj2)
                # elif op == "++":
                #     return BagAdditiveUnion(obj1, obj2)
                elif op == "&":
                    return BagIntersection(obj1, obj2)
                elif op == "-":
                    return BagDifference(obj1, obj2)
                else:
                    raise CofolaParsingError(f"Unknown bag operation {op}.")
        return self._op_or_constraint_on_list(
            single_operation, objs1, objs2)

    def indexing(self, args):
        obj, index = args[0], args[2]
        self._check_obj_type(obj, Partition, Tuple)
        if isinstance(obj, Partition):
            if index == '*':
                return obj.partitioned_objs
            # if not obj.ordered:
            #     raise CofolaParsingError("Indexing operation is only supported for ordered partitions.")
            index_obj = obj.partitioned_objs[int(index)]
        else:
            index_obj = TupleIndex(obj, int(index))
        return self.problem.add_object(index_obj)

    def inverse_object(self, args):
        obj = args[0]
        return obj

    def image(self, args):
        obj, set_or_entity = args[0], args[2]
        self._check_obj_type(obj, Function)
        return FuncImage(obj, set_or_entity)

    def inverse_image(self, args):
        obj, set_or_entity = args[0], args[2]
        self._check_obj_type(obj, Function)
        return FuncInverseImage(obj, set_or_entity)

    def size_constraint(self, args):
        expr, comparator, param = args
        expr = tuple(expr)
        param = int(param)
        if any(
            isinstance(obj, list) for obj, _ in expr
        ):
            if len(expr) > 1:
                raise CofolaParsingError(
                    "Only support simple size constraints on parts of partitions"
                )
            self._check_obj_type(expr[0][0][0], SizedObject)
            return list(
                SizeConstraint([(part, expr[0][1])], comparator, param)
                for part in expr[0][0]
            )
        return SizeConstraint(expr, comparator, param)

    def size_atom(self, args):
        if len(args) == 1:
            coef, obj = 1, args[0]
        else:
            coef, obj = args[0][0], args[1]
        self._check_obj_type(obj, SizedObject, list)
        return coef, obj

    def size_atomic_expr(self, args):
        coef, obj = args[0]
        expr = [(obj, int(coef))]
        return expr

    def size_add(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, coef))
        return expr

    def size_sub(self, args):
        expr, atom = args
        coef, set_obj = atom
        expr.append((set_obj, -coef))
        return expr

    def count(self, args):
        objs, count_name, _, entity_or_obj, _ = args
        deduplicate = count_name == "dedup_count"
        def single_operation(obj):
            if deduplicate:
                self._check_obj_type(obj, Tuple)
            if isinstance(obj, Bag) and isinstance(entity_or_obj, Entity):
                count_obj = BagMultiplicity(obj, entity_or_obj)
            elif isinstance(obj, Tuple) and (
                isinstance(entity_or_obj, Entity) or isinstance(entity_or_obj, Set)
            ):
                count_obj = TupleCount(obj, entity_or_obj, deduplicate)
            elif isinstance(obj, Sequence) and isinstance(entity_or_obj, SequenceSizedPattern):
                if not entity_or_obj._has_size(obj):
                    raise CofolaParsingError(
                        f"Count constraint must be applied to a sequence pattern with a size."
                    )
                count_obj = SequencePatternCount(obj, entity_or_obj)
            else:
                raise CofolaParsingError(
                    f"Count constraint is not supported for the given objects: {obj}, {entity_or_obj}"
                )
            return self.problem.add_object(count_obj)
        return self._op_or_constraint_on_list(
            single_operation, objs)

    def in_or_not(self, args):
        return True if len(args) == 1 else False

    def membership_constraint(self, args):
        entity_or_tuple_index, in_or_not, objs = args
        def single_operation(obj):
            self._check_obj_type(obj, Set, Bag, Tuple)
            if not isinstance(entity_or_tuple_index, TupleIndex) and \
                    not self.problem.contains_entity(entity_or_tuple_index):
                raise CofolaParsingError(
                    f"Entity {entity_or_tuple_index} not found."
                )
            if isinstance(obj, Tuple) or \
                    isinstance(entity_or_tuple_index, TupleIndex):
                ret = TupleMembershipConstraint(obj, entity_or_tuple_index)
            else:
                ret = MembershipConstraint(obj, entity_or_tuple_index)
            if not in_or_not:
                ret.negate()
            return ret
        return self._op_or_constraint_on_list(
            single_operation, objs)

    def subset_constraint(self, args):
        objs1, _, objs2 = args
        def single_operation(obj1, obj2):
            if isinstance(obj1, Set) or isinstance(obj2, Set):
                self._check_obj_type(obj1, Set, Bag)
                self._check_obj_type(obj2, Set, Bag)
                return SubsetConstraint(obj1, obj2)
            self._check_obj_type(obj1, Bag)
            self._check_obj_type(obj2, Bag)
            return BagSubsetConstraint(obj1, obj2)
        return self._op_or_constraint_on_list(
            single_operation, objs1, objs2)

    def disjoint_constraint(self, args):
        obj1, _, obj2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, Bag)
            self._check_obj_type(obj2, Set, Bag)
            return DisjointConstraint(obj1, obj2)
        return self._op_or_constraint_on_list(
            single_operation, obj1, obj2)

    def equivalence_constraint(self, args):
        obj1, symbol, obj2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, TupleIndex)
            self._check_obj_type(obj2, Set, Entity)
            if isinstance(obj1, TupleIndex) and isinstance(obj2, Entity):
                ret = TupleIndexEqConstraint(obj1, obj2)
            elif isinstance(obj1, Set) and isinstance(obj2, Set):
                ret = SetEqConstraint(obj1, obj2)
            elif isinstance(obj1, Bag) and isinstance(obj2, Bag):
                ret = BagEqConstraint(obj1, obj2)
            else:
                raise CofolaParsingError(f"Equivalence constraint is not supported for the given objects: {obj1}, {obj2}")
            if symbol == '!=':
                ret.negate()
            return ret
        return self._op_or_constraint_on_list(
            single_operation, obj1, obj2)

    def count_parameter(self, args):
        count = int(args[0])
        if count < 0:
            raise CofolaParsingError(f"Count parameter must be non-negative.")
        return count

    def seq_constraint(self, args):
        pattern, is_in, obj = args
        self._check_obj_type(obj, SequenceImpl)
        constraint = SequenceConstraint(
            obj, pattern
        )
        if not is_in:
            constraint.negate()
        return constraint

    def seq_pattern(self, args):
        pattern = args[0]
        return self.problem.add_object(pattern)

    def together(self, args):
        obj = args[2]
        return TogetherPattern(obj)

    def less_than(self, args):
        entity_or_obj1, _, entity_or_obj2 = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return LessThanPattern(entity_or_obj1, entity_or_obj2)

    def next_to(self, args):
        _, _, entity_or_obj1, entity_or_obj2, _ = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return NextToPattern(entity_or_obj1, entity_or_obj2)

    def predecessor(self, args):
        _, entity_or_obj1, entity_or_obj2, _ = args
        self._check_obj_type(entity_or_obj1, Entity, Set)
        self._check_obj_type(entity_or_obj2, Entity, Set)
        return PredecessorPattern(entity_or_obj1, entity_or_obj2)

    def _op_or_constraint_on_list(self, op_or_constraint, *args):
        assert len(args) <= 2
        if len(args) == 1:
            if isinstance(args[0], list):
                return list(
                    op_or_constraint(obj)
                    for obj in args[0]
                )
            return op_or_constraint(args[0])
        if len(args) == 2:
            if all(isinstance(obj, list) for obj in args):
                raise CofolaParsingError(f"Operation {op_or_constraint} is not supported for parts of two partitions")
            if isinstance(args[1], list):
                args[0], args[1] = args[1], args[0]
            if isinstance(args[0], list):
                return list(
                    op_or_constraint(obj, args[1])
                    for obj in args[0]
                )
            return op_or_constraint(*args)

    def _get_obj_by_id(self, obj_id: str) -> CombinatoricsObject:
        if obj_id in self.id2obj:
            return self.id2obj[obj_id]
        else:
            raise CofolaParsingError(f"Object {obj_id} has not been defined.")

    def _attach_obj(self, name: str, obj: CombinatoricsObject):
        """
        Attach the object to the dictionary of objects for the late reference.
        :param name: The name of the object.
        :param obj: The object to be attached.
        """
        if self.problem.contains_entity(Entity(name)):
            raise CofolaParsingError(f"The name {obj} has used as an Entity. Please use another name.")
        if name in self.id2obj:
            logger.warning(f"Object {obj.name} has already been defined. It will be overwritten.")
        self.id2obj[name] = obj

    def _check_id(self, id: str):
        """
        Check if the identifier is a reserved keyword.
        """
        if id in RESERVED_KEYWORDS:
            raise CofolaParsingError(f"Identifier {id} is a reserved keyword. Please use another name.")
        for prefix in RESERVED_PREFIXES:
            if id.startswith(prefix):
                raise CofolaParsingError(f"Identifier {id} starts with reserved prefix {prefix}. Please use another name.")

    def _check_obj_type(self, obj: object, *expected_types: type):
        """
        Check if the object is of the expected type.

        :param obj: The object to be checked.
        :param expected_types: The expected types.
        :raise CofolaTypeMismatchError: If the object is not of the expected type.
        """
        if not isinstance(obj, expected_types):
            raise CofolaTypeMismatchError(expected_types, obj)

    def cofola(self, args):
        # NOTE: use list to keep the defining order of the objects and constraints
        statements = list(args)
        for statement in statements:
            if isinstance(statement, CombinatoricsConstraint):
                self.problem.add_constraint(statement)
            if isinstance(statement, list):
                for sub_statement in statement:
                    if isinstance(sub_statement, CombinatoricsConstraint):
                        self.problem.add_constraint(sub_statement)
        # set the name of objs to be the name defined in the problem,
        # it's just for debugging purpose
        already_set = set()
        for name, obj in self.id2obj.items():
            if obj.name in already_set:
                continue
            obj.name = name
            already_set.add(name)
        # NOTE: propogate the properties (especially the size) of the objects
        self.problem.build()
        return self.problem

    def negation_constraint(self, args):
        constraint = args[1]
        self._check_obj_type(
            constraint, CombinatoricsConstraint
        )
        return ~args[1]

    def binary_constraint(self, args):
        arg1, op, arg2 = args
        self._check_obj_type(
            arg1, CombinatoricsConstraint
        )
        self._check_obj_type(
            arg2, CombinatoricsConstraint
        )
        if op == 'and':
            return arg1 & arg2
        if op == 'or':
            return arg1 | arg2

    def part_constraint(self, args):
        return args[0]

    def _transform_tree(self, tree):
        # NOTE: process the constraint over a partition
        if tree.data == 'part_constraint':
            partition_id = tree.children[-1].children[0].value
            self._processing_partition = self._get_obj_by_id(partition_id)
            self._processing_part_name = tree.children[-3].value
            logger.info(f"Processing partition {partition_id} with the part name {self._processing_part_name}")
        return super()._transform_tree(tree)

    # def size_constrained_image(self, args):
    #     obj, comparator, count = args
    #     self._check_obj_type(obj, Function)
    #     return FuncSizeConstrainedImage(obj, comparator, count)

    # def membership_constrained_image(self, args):
    #     entity, obj = args[0], args[2]
    #     self._check_obj_type(obj, Function)
    #     return FuncMembershipConstraintedImage(obj, entity)

    # def subset_constrained_image(self, args):
    #     obj1, obj2 = args[0], args[2]
    #     self._check_obj_type(obj1, Function)
    #     self._check_obj_type(obj2, Set)
    #     return FuncSubsetConstrainedImage(obj1, obj2)

    # def subset_constrained_image_reverse(self, args):
    #     obj1, obj2 = args[0], args[2]
    #     self._check_obj_type(obj1, Set)
    #     self._check_obj_type(obj2, Function)
    #     return FuncSubsetConstrainedImage(obj2, obj1, True)

    # def disjoint_constrained_image(self, args):
    #     obj1, obj2 = args[0], args[2]
    #     self._check_obj_type(obj1, Function)
    #     self._check_obj_type(obj2, Set)
    #     return FuncDisjointConstrainedImage(obj1, obj2)

    # def equivalence_constrained_image(self, args):
    #     obj1, obj2 = args[0], args[2]
    #     self._check_obj_type(obj1, Function)
    #     self._check_obj_type(obj2, Set)
    #     return FuncEqConstrainedImage(obj1, obj2)


def parse(text: str) -> CofolaProblem:
    set_parser = Lark(grammar, start='cofola')
    tree = set_parser.parse(text)
    return CofolaTransfomer().transform(tree)


if __name__ == '__main__':
    text = r"""
# declare the sets
nondefective_tvs = set(nondef1...10)
defective_tvs = set(def10...13)
# set tvs = nondefective_tvs + defective_tvs

# perform choose operation
purchase = choose(nondefective_tvs+defective_tvs, 5)

# specify the constraints
# |(purchase & defective_tvs)| >= 2
    """
#     text = r"""
# set students = {1:100}
# set male_students = {1:50}
# set female_students = {51:100}
#
# set class_one = choose(students, 20) + male_students
# set class_two = choose(students, 20) + female_students
# set class_three = choose(students, 20) - class_one - class_two
# (1 in class_one) & (|class_one & male_students| >= 10)
# 1 in class_one <-> ~(1 in class_three)
# """
    text = r"""
indices = set(idx1...3)
S = set(val1...10)

composition = (S -> indices)
T = |composition-1| = 2
T1 = composition-1(idx1)
T2 = composition-1(set(idx1...10))
# |composition-1(val1)| = 2
# val9 in composition-1(idx1)
#
# |(|composition-1| = 3)| = 1
"""
    text = """
items = bag(nickel: 10, ball0...5, a, b, c)
payment = items ++ choose(items, 5)
|payment| > 0
"""
    problem = parse(text)
    print(problem)
    problem = simplify(problem)
    print(problem)
