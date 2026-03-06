"""Cofola transformer - directly produces IR Problem via ProblemBuilder."""
from __future__ import annotations

from loguru import logger

from cofola.frontend.objects import PartRef as IRPartRef
from cofola.frontend.problem import ProblemBuilder
from cofola.frontend.types import Entity, ObjRef
from cofola.parser.common import CommonTransformer
from cofola.parser.transformer_objects import ObjectTransformerMixin
from cofola.parser.transformer_constraints import ConstraintTransformerMixin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cofola.parser.parser import CofolaParsingError


class TupleIndexSentinel:
    """Sentinel for tuple index expressions (T[i]) used in constraints.

    Not a real IR object — only lives during parsing to carry tuple_ref and index
    until a constraint is built.
    """

    __slots__ = ('tuple_ref', 'index')

    def __init__(self, tuple_ref: ObjRef, index: int) -> None:
        self.tuple_ref = tuple_ref
        self.index = index

    def __repr__(self) -> str:
        return f"TupleIndexSentinel({self.tuple_ref}, {self.index})"


class CofolaTransfomer(
    CommonTransformer,
    ObjectTransformerMixin,
    ConstraintTransformerMixin,
):
    """Main transformer that processes the parse tree into an IR Problem."""

    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self.builder = ProblemBuilder()
        self.id2ref: dict[str, ObjRef] = {}
        self.entities: set[str] = set()  # known entity names

    def contains_entity(self, name: str) -> bool:
        return name in self.entities

    def left_identity(self, args):
        from cofola.parser.parser import RESERVED_KEYWORDS, RESERVED_PREFIXES, CofolaParsingError

        obj_id = str(args[0].value)
        self._check_id(obj_id, RESERVED_KEYWORDS, RESERVED_PREFIXES, CofolaParsingError)
        return obj_id

    def object_declaration(self, args):
        obj_id, ref = args
        if isinstance(ref, list):
            # partition expansion produces a list — no top-level registration needed
            return
        self._attach_ref(obj_id, ref)

    def identity(self, args):
        obj_id = str(args[0].value)
        if obj_id in self.entities:
            return Entity(obj_id)
        return self._get_ref_by_id(obj_id)

    def _op_or_constraint_on_list(self, op_or_constraint, *args):
        assert len(args) <= 2
        if len(args) == 1:
            if isinstance(args[0], list):
                return list(op_or_constraint(obj) for obj in args[0])
            return op_or_constraint(args[0])
        if len(args) == 2:
            if all(isinstance(obj, list) for obj in args):
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(
                    f"Operation {op_or_constraint} is not supported for parts of two partitions"
                )
            if isinstance(args[1], list):
                args[0], args[1] = args[1], args[0]
            if isinstance(args[0], list):
                return list(op_or_constraint(obj, args[1]) for obj in args[0])
            return op_or_constraint(*args)

    def _get_ref_by_id(self, obj_id: str) -> ObjRef:
        from cofola.parser.parser import CofolaParsingError
        if obj_id in self.id2ref:
            return self.id2ref[obj_id]
        raise CofolaParsingError(f"Object {obj_id} has not been defined.")

    def _attach_ref(self, name: str, ref: ObjRef) -> None:
        from cofola.parser.parser import CofolaParsingError
        if name in self.entities:
            raise CofolaParsingError(
                f"The name {name} has been used as an Entity. Please use another name."
            )
        if name in self.id2ref:
            logger.warning(f"Object {name} has already been defined. It will be overwritten.")
        self.id2ref[name] = ref
        self.builder.set_name(ref, name)

    def _check_id(self, id: str, reserved_keywords: list, reserved_prefixes: list, error_class):
        if id in reserved_keywords:
            raise error_class(
                f"Identifier {id} is a reserved keyword. Please use another name."
            )
        for prefix in reserved_prefixes:
            if id.startswith(prefix):
                raise error_class(
                    f"Identifier {id} starts with reserved prefix {prefix}. Please use another name."
                )

    def _check_obj_type(self, obj, *expected_types):
        from cofola.parser.parser import CofolaTypeMismatchError
        if not self._obj_matches_types(obj, expected_types):
            raise CofolaTypeMismatchError(expected_types, obj)

    def _obj_matches_types(self, obj, expected_types) -> bool:
        """Check if obj matches any of the expected sentinel types."""
        for t in expected_types:
            t_name = t.__name__ if hasattr(t, '__name__') else str(t)
            if t_name == 'Entity' and isinstance(obj, Entity):
                return True
            if t_name == 'TupleIndexSentinel' and isinstance(obj, TupleIndexSentinel):
                return True
            if t_name == 'list' and isinstance(obj, list):
                return True
            if isinstance(obj, ObjRef):
                cat = self._ref_category(obj)
                if t_name == 'Set' and cat == 'set':
                    return True
                if t_name == 'Bag' and cat == 'bag':
                    return True
                if t_name == 'Function' and cat == 'func':
                    return True
                if t_name == 'Tuple' and cat == 'tuple':
                    return True
                if t_name == 'Sequence' and cat == 'sequence':
                    return True
                if t_name == 'Partition' and cat in ('partition', 'part'):
                    return True
                if t_name == 'SizedObject' and cat in (
                    'set', 'bag', 'tuple', 'sequence', 'partition', 'part'
                ):
                    return True
        return False

    def _ref_category(self, ref: ObjRef) -> str:
        """Get the category of an ObjRef."""
        from cofola.frontend.objects import (
            SetInit, SetChoose, SetChooseReplace, SetUnion, SetIntersection, SetDifference,
            BagSupport,
            BagInit, BagChoose, BagUnion, BagAdditiveUnion, BagIntersection, BagDifference,
            FuncDef, FuncInverse, FuncImage, FuncInverseImage,
            TupleDef, SequenceDef, PartitionDef, PartRef,
        )
        defn = self.builder.get_object(ref)
        if defn is None:
            return 'unknown'
        if isinstance(defn, (
            SetInit, SetChoose, SetChooseReplace,
            SetUnion, SetIntersection, SetDifference, BagSupport
        )):
            return 'set'
        if isinstance(defn, (
            BagInit, BagChoose, BagUnion, BagAdditiveUnion,
            BagIntersection, BagDifference
        )):
            return 'bag'
        if isinstance(defn, (FuncDef, FuncInverse, FuncImage, FuncInverseImage)):
            return 'func'
        if isinstance(defn, TupleDef):
            return 'tuple'
        if isinstance(defn, SequenceDef):
            return 'sequence'
        if isinstance(defn, PartitionDef):
            return 'partition'
        if isinstance(defn, PartRef):
            return 'part'
        return 'unknown'

    def cofola(self, args):
        from cofola.frontend.constraints import (
            SizeConstraint, MembershipConstraint, SubsetConstraint, DisjointConstraint,
            EqualityConstraint, TupleIndexEq, TupleIndexMembership, SequencePatternConstraint,
            FuncPairConstraint, BagSubsetConstraint, BagEqConstraint,
            NotConstraint, AndConstraint, OrConstraint,
            ForAllParts,
        )
        _CONSTRAINT_TYPES = (
            SizeConstraint, MembershipConstraint, SubsetConstraint, DisjointConstraint,
            EqualityConstraint, TupleIndexEq, TupleIndexMembership, SequencePatternConstraint,
            FuncPairConstraint, BagSubsetConstraint, BagEqConstraint,
            NotConstraint, AndConstraint, OrConstraint,
            ForAllParts,
        )
        for statement in args:
            if isinstance(statement, _CONSTRAINT_TYPES):
                self.builder.add_constraint(statement)
            elif isinstance(statement, list):
                for sub in statement:
                    if isinstance(sub, _CONSTRAINT_TYPES):
                        self.builder.add_constraint(sub)
        return self.builder.build()

    def _transform_tree(self, tree):
        if tree.data == 'part_constraint':
            partition_id = tree.children[-1].children[0].value
            partition_ref = self._get_ref_by_id(partition_id)
            part_name = tree.children[-3].value
            sentinel_ref = self.builder.add(IRPartRef(partition=partition_ref, index=-1))
            self.id2ref[part_name] = sentinel_ref
            logger.info(f"Processing part_constraint: part={part_name}, partition={partition_id}")
            result = super()._transform_tree(tree)
            del self.id2ref[part_name]
            # sentinel_ref stays in builder._defs so LoweringPass can find it
            return result
        return super()._transform_tree(tree)
