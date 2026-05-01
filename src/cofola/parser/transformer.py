"""Cofola transformer - directly produces IR Problem via ProblemBuilder."""
from __future__ import annotations

from loguru import logger

from cofola.frontend.constraints import (
    AndConstraint,
    BagEqConstraint,
    BagSubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    ForAllParts,
    FuncPairConstraint,
    MembershipConstraint,
    NotConstraint,
    OrConstraint,
    SequencePatternConstraint,
    SizeConstraint,
    SubsetConstraint,
    TupleIndexEq,
    TupleIndexMembership,
)
from cofola.frontend.objects import (
    BagObjDef,
    BagPartRef,
    ObjDef,
    PartitionDef,
    SetObjDef,
    SetPartRef,
)
from cofola.frontend.objects import PartRef as IRPartRef
from cofola.frontend.problem import ProblemBuilder
from cofola.frontend.types import Entity, ObjRef
from cofola.parser.common import CommonTransformer
from cofola.parser.constants import RESERVED_KEYWORDS, RESERVED_PREFIXES
from cofola.parser.errors import CofolaParsingError
from cofola.parser.transformer_constraints import ConstraintTransformerMixin
from cofola.parser.transformer_objects import ObjectTransformerMixin


_CONSTRAINT_TYPES: tuple[type, ...] = (
    SizeConstraint,
    MembershipConstraint,
    SubsetConstraint,
    DisjointConstraint,
    EqualityConstraint,
    TupleIndexEq,
    TupleIndexMembership,
    SequencePatternConstraint,
    FuncPairConstraint,
    BagSubsetConstraint,
    BagEqConstraint,
    NotConstraint,
    AndConstraint,
    OrConstraint,
    ForAllParts,
)


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
        obj_id = str(args[0].value)
        self._check_id(obj_id, RESERVED_KEYWORDS, RESERVED_PREFIXES)
        return obj_id

    def object_declaration(self, args):
        obj_id, ref = args
        self._attach_ref(obj_id, ref)

    def identity(self, args):
        obj_id = str(args[0].value)
        if obj_id in self.entities:
            return Entity(obj_id)
        return self._get_ref_by_id(obj_id)

    def _get_ref_by_id(self, obj_id: str) -> ObjRef:
        if obj_id in self.id2ref:
            return self.id2ref[obj_id]
        raise CofolaParsingError(f"Object {obj_id} has not been defined.")

    def _attach_ref(self, name: str, ref: ObjRef) -> None:
        if name in self.entities:
            raise CofolaParsingError(
                f"The name {name} has been used as an Entity. Please use another name."
            )
        if name in self.id2ref:
            logger.warning(f"Object {name} has already been defined. It will be overwritten.")
        self.id2ref[name] = ref
        self.builder.set_name(ref, name)

    def _check_id(
        self,
        id: str,
        reserved_keywords: list[str],
        reserved_prefixes: list[str],
    ) -> None:
        if id in reserved_keywords:
            raise CofolaParsingError(
                f"Identifier {id} is a reserved keyword. Please use another name."
            )
        for prefix in reserved_prefixes:
            if id.startswith(prefix):
                raise CofolaParsingError(
                    f"Identifier {id} starts with reserved prefix {prefix}. Please use another name."
                )

    def _defn_of(self, obj) -> ObjDef | None:
        """Return the defn for `obj`, resolving PartRef to its source's defn.

        Returns None for non-ObjRef inputs or undefined refs. Callers use
        `isinstance(defn, SetObjDef | BagObjDef | ...)` to dispatch.
        """
        if not isinstance(obj, ObjRef):
            return None
        defn = self.builder.get_object(obj)
        if isinstance(defn, IRPartRef):
            partition_defn = self.builder.get_object(defn.partition)
            if partition_defn is None:
                return None
            return self._defn_of(partition_defn.source)
        return defn

    def cofola(self, args):
        for statement in args:
            if isinstance(statement, _CONSTRAINT_TYPES):
                self.builder.add_constraint(statement)
        return self.builder.build()

    def _transform_tree(self, tree):
        if tree.data == "part_constraint":
            partition_id = tree.children[-1].children[0].value
            partition_ref = self._get_ref_by_id(partition_id)
            part_name = tree.children[-3].value
            partition_defn = self.builder.get_object(partition_ref)
            if not isinstance(partition_defn, PartitionDef):
                raise CofolaParsingError(
                    f"`for {part_name} in {partition_id}`: "
                    f"{partition_id} is not a partition."
                )
            source_defn = self.builder.get_object(partition_defn.source)
            if isinstance(source_defn, BagObjDef):
                sentinel_cls: type = BagPartRef
            elif isinstance(source_defn, SetObjDef):
                sentinel_cls = SetPartRef
            else:
                raise CofolaParsingError(
                    f"`for {part_name} in {partition_id}`: "
                    f"partition source is neither set nor bag "
                    f"(got {type(source_defn).__name__})."
                )
            sentinel_ref = self.builder.add(sentinel_cls(partition=partition_ref, index=-1))
            self.id2ref[part_name] = sentinel_ref
            logger.info(f"Processing part_constraint: part={part_name}, partition={partition_id}")
            result = super()._transform_tree(tree)
            del self.id2ref[part_name]
            # sentinel_ref stays in the builder so LoweringPass can find it
            return result
        return super()._transform_tree(tree)
