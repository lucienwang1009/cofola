"""Cofola transformer combining core, objects, and constraints mixins."""
from __future__ import annotations

from logzero import logger

from cofola.objects.base import CombinatoricsConstraint, CombinatoricsObject, Entity, Partition, Tuple
from cofola.parser.common import CommonTransformer
from cofola.parser.transformer_objects import ObjectTransformerMixin
from cofola.parser.transformer_constraints import ConstraintTransformerMixin
from cofola.problem import CofolaProblem
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cofola.parser.parser import CofolaParsingError


class CofolaTransfomer(
    CommonTransformer,
    ObjectTransformerMixin,
    ConstraintTransformerMixin
):
    """Main transformer that processes the parse tree into a CofolaProblem."""

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
        from cofola.parser.parser import RESERVED_KEYWORDS, RESERVED_PREFIXES, CofolaParsingError

        obj_id = str(args[0].value)
        self._check_id(obj_id, RESERVED_KEYWORDS, RESERVED_PREFIXES, CofolaParsingError)
        return obj_id

    def object_declaration(self, args):
        obj_id, obj = args
        self._attach_obj(obj_id, obj)

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
                from cofola.parser.parser import CofolaParsingError
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
        from cofola.parser.parser import CofolaParsingError

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
        from cofola.parser.parser import CofolaParsingError

        if self.problem.contains_entity(Entity(name)):
            raise CofolaParsingError(f"The name {obj} has used as an Entity. Please use another name.")
        if name in self.id2obj:
            logger.warning(f"Object {obj.name} has already been defined. It will be overwritten.")
        self.id2obj[name] = obj

    def _check_id(self, id: str, reserved_keywords: list, reserved_prefixes: list, error_class):
        """
        Check if the identifier is a reserved keyword.
        """
        if id in reserved_keywords:
            raise error_class(f"Identifier {id} is a reserved keyword. Please use another name.")
        for prefix in reserved_prefixes:
            if id.startswith(prefix):
                raise error_class(f"Identifier {id} starts with reserved prefix {prefix}. Please use another name.")

    def _check_obj_type(self, obj: object, *expected_types: type):
        """
        Check if the object is of the expected type.

        :param obj: The object to be checked.
        :param expected_types: The expected types.
        :raise CofolaTypeMismatchError: If the object is not of the expected type.
        """
        from cofola.parser.parser import CofolaTypeMismatchError

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
        # NOTE: propagate the properties (especially the size) of the objects
        self.problem.build()
        return self.problem

    def _transform_tree(self, tree):
        # NOTE: process the constraint over a partition
        if tree.data == 'part_constraint':
            partition_id = tree.children[-1].children[0].value
            self._processing_partition = self._get_obj_by_id(partition_id)
            self._processing_part_name = tree.children[-3].value
            logger.info(f"Processing partition {partition_id} with the part name {self._processing_part_name}")
        return super()._transform_tree(tree)