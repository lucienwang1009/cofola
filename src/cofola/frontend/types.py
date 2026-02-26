"""Core types for the immutable IR.

This module defines the fundamental types used throughout the IR:
- ObjRef: A unique reference to an object in the IR graph
- Entity: An atomic entity (e.g., 'A', 'math1', 'key3')
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjRef:
    """Unique reference to an object in the IR graph.

    ObjRef acts as a stable identifier that can be used as a dict key.
    Each ObjRef corresponds to exactly one ObjDef in a Problem.

    Attributes:
        id: Unique integer identifier for this reference.
    """

    id: int

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjRef) and self.id == other.id

    def __lt__(self, other: ObjRef) -> bool:
        return self.id < other.id

    def __repr__(self) -> str:
        return f"ObjRef({self.id})"


@dataclass(frozen=True, slots=True)
class Entity:
    """An atomic entity in a combinatorics problem.

    Entities are the basic building blocks (e.g., 'A', 'math1', 'key3').
    They are used in sets, bags, and as function arguments.

    Attributes:
        name: The name/identifier of this entity.
    """

    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.name == other.name

    def __repr__(self) -> str:
        return f"Entity({self.name!r})"

    def __str__(self) -> str:
        return self.name


# Type alias for values that can be either an ObjRef or an Entity
# Used in function arguments and sequence patterns
RefOrEntity = ObjRef | Entity