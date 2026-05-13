"""Structural traversal helpers for frontend nodes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Callable, Iterator

from cofola.frontend.objects import ObjRef


def iter_refs(value: object) -> Iterator[ObjRef]:
    """Yield every ``ObjRef`` reachable from ``value``."""

    if isinstance(value, ObjRef):
        yield value
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_refs(key)
            yield from iter_refs(item)
        return

    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from iter_refs(item)
        return

    if is_dataclass(value) and not isinstance(value, type):
        for field_ in fields(value):
            yield from iter_refs(getattr(value, field_.name))


def map_refs(value: object, mapper: Callable[[ObjRef], ObjRef]) -> object:
    """Return ``value`` with every nested ``ObjRef`` replaced by ``mapper``."""

    if isinstance(value, ObjRef):
        return mapper(value)

    if isinstance(value, Mapping):
        return type(value)(
            (map_refs(key, mapper), map_refs(item, mapper))
            for key, item in value.items()
        )

    if isinstance(value, tuple):
        return tuple(map_refs(item, mapper) for item in value)
    if isinstance(value, list):
        return [map_refs(item, mapper) for item in value]
    if isinstance(value, frozenset):
        return frozenset(map_refs(item, mapper) for item in value)
    if isinstance(value, set):
        return {map_refs(item, mapper) for item in value}

    if is_dataclass(value) and not isinstance(value, type):
        new_fields = {
            field_.name: map_refs(getattr(value, field_.name), mapper)
            for field_ in fields(value)
        }
        return type(value)(**new_fields)

    return value


def object_refs(defn: object) -> list[ObjRef]:
    """Return all object refs used by an object definition."""

    return list(iter_refs(defn))


def constraint_refs(constraint: object) -> list[ObjRef]:
    """Return all object refs used by a constraint."""

    return list(iter_refs(constraint))
