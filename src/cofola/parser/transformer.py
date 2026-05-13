"""Cofola transformer - directly produces a frontend Problem via ProblemBuilder."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from lark import Transformer, Tree
from loguru import logger

from cofola.frontend.constraints import (
    CONSTRAINT_CLASSES,
    ForAllParts,
)
from cofola.frontend.objects import (
    Grouped,
    PartPlaceholderDef,
)
from cofola.frontend.problem import ProblemBuilder
from cofola.frontend.objects import Entity, ObjRef
from cofola.parser.utils import (
    CofolaParsingError,
    RESERVED_KEYWORDS,
    RESERVED_PREFIXES,
)
from cofola.parser.transformer_constraints import ConstraintTransformerMixin
from cofola.parser.transformer_objects import ObjectTransformerMixin


class CommonTransformer(Transformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)

    def parenthesis(self, args):
        return args[1]

    def equality(self, args):
        return "=="

    def nequality(self, args):
        return "!="

    def le(self, args):
        return "<="

    def ge(self, args):
        return ">="

    def lt(self, args):
        return "<"

    def gt(self, args):
        return ">"

    def true(self, args):
        return True

    def false(self, args):
        return False

    def INT(self, args):
        return int(args)


class CofolaTransformer(
    CommonTransformer,
    ObjectTransformerMixin,
    ConstraintTransformerMixin,
):
    """Main transformer that processes the parse tree into a frontend Problem."""

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
            raise CofolaParsingError(
                f"Object {name} has already been defined. Please use another name."
            )
        self.id2ref[name] = ref
        self.builder.set_name(ref, name)

    def _check_id(
        self,
        name: str,
        reserved_keywords: list[str],
        reserved_prefixes: list[str],
    ) -> None:
        if name in reserved_keywords:
            raise CofolaParsingError(
                f"Identifier {name} is a reserved keyword. Please use another name."
            )
        for prefix in reserved_prefixes:
            if name.startswith(prefix):
                raise CofolaParsingError(
                    f"Identifier {name} starts with reserved prefix {prefix}. Please use another name."
                )

    def _check_part_name(self, part_name: str) -> None:
        self._check_id(part_name, RESERVED_KEYWORDS, RESERVED_PREFIXES)
        if part_name in self.entities:
            raise CofolaParsingError(
                f"The part name {part_name} has been used as an Entity. "
                "Please use another name."
            )
        if part_name in self.id2ref:
            raise CofolaParsingError(
                f"The part name {part_name} has been used as an object name. "
                "Please use another name."
            )

    @contextmanager
    def _part_scope(self, part_name: str, placeholder_ref: ObjRef) -> Iterator[None]:
        """Temporarily bind a forall-part name while its template transforms."""

        self.id2ref[part_name] = placeholder_ref
        try:
            yield
        finally:
            del self.id2ref[part_name]

    def _defn_of(self, obj):
        """Return the effective definition for parser dispatch."""

        if not isinstance(obj, ObjRef):
            return None
        return self.builder.get_effective_object(obj)

    def cofola(self, args):
        for statement in args:
            if isinstance(statement, CONSTRAINT_CLASSES):
                self.builder.add_constraint(statement)
        return self.builder.build()

    def _call_userfunc(self, tree, new_children=None):
        """Override to record source locations for ObjRefs created by rule methods.

        After the user function returns, if the result is an ObjRef and the
        tree has meta with a line/column, record it on the builder. This
        side-table approach (cf. Problem.locs) keeps positions out of every
        frontend dataclass.
        """
        result = super()._call_userfunc(tree, new_children)
        meta = getattr(tree, "meta", None)
        if meta is not None and not getattr(meta, "empty", True):
            line = getattr(meta, "line", None)
            col = getattr(meta, "column", None)
            if isinstance(result, ObjRef) and line is not None and col is not None:
                self.builder.set_loc(result, line, col)
        return result

    def _transform_tree(self, tree):
        if tree.data == "part_constraint":
            partition_id = self._partition_id_from_for_object(tree.children[-1])
            partition_ref = self._get_ref_by_id(partition_id)
            part_name = tree.children[-3].value
            self._check_part_name(part_name)
            partition_defn = self.builder.get_object(partition_ref)
            if not isinstance(partition_defn, Grouped):
                raise CofolaParsingError(
                    f"`for {part_name} in {partition_id}`: "
                    f"{partition_id} is not a partition."
                )
            placeholder_ref = self.builder.add(
                PartPlaceholderDef(partition=partition_ref)
            )
            logger.debug(
                "Processing part_constraint: part={}, partition={}",
                part_name,
                partition_id,
            )
            with self._part_scope(part_name, placeholder_ref):
                result = super()._transform_tree(tree)
            if isinstance(result, ForAllParts):
                return ForAllParts(
                    constraint_template=result.constraint_template,
                    partition=result.partition,
                    part_ref=placeholder_ref,
                )
            return result
        return super()._transform_tree(tree)

    def _partition_id_from_for_object(self, node) -> str:
        """Return the named partition id used by ``for part in P``.

        The grammar accepts a general ``object`` after ``in``. For now the
        forall lowering model requires a named partition/composition before
        transforming the scoped constraint template. Parentheses around the name
        are accepted.
        """

        if isinstance(node, Tree):
            if node.data == "identity":
                return str(node.children[0].value)
            if node.data == "parenthesis":
                for child in node.children:
                    if isinstance(child, Tree):
                        try:
                            return self._partition_id_from_for_object(child)
                        except CofolaParsingError:
                            continue
            raise CofolaParsingError(
                "`for <part> in <partition>` requires a named partition or "
                "composition object."
            )
        raise CofolaParsingError(
            "`for <part> in <partition>` requires a named partition or "
            "composition object."
        )


# Compatibility alias for imports from the previous misspelled name.
CofolaTransfomer = CofolaTransformer
