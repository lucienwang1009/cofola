"""Solver backends, loaded lazily to keep optional dependencies isolated."""

from cofola.backend.base import Backend

__all__ = [
    "Backend",
    "UnaryFOMCBackend",
    "WFOMCBackend",
]


def __getattr__(name: str):
    if name == "WFOMCBackend":
        from cofola.backend.wfomc.backend import WFOMCBackend

        return WFOMCBackend
    if name == "UnaryFOMCBackend":
        from cofola.backend.unaryfomc.backend import UnaryFOMCBackend

        return UnaryFOMCBackend
    raise AttributeError(name)
