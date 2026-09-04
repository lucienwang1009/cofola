"""Cofola's optional backend for one-unary-function model counting."""

from cofola.backend.unaryfomc.backend import UnaryFOMCBackend
from cofola.backend.unaryfomc.encoder import (
    UnaryFOMCEncoding,
    UnaryFOMCUnsupportedError,
    encode_ir,
)

__all__ = [
    "UnaryFOMCBackend",
    "UnaryFOMCEncoding",
    "UnaryFOMCUnsupportedError",
    "encode_ir",
]
