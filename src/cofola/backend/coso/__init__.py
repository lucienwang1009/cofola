"""CoSo backend package."""
from __future__ import annotations

from cofola.backend.coso.backend import CoSoBackend
from cofola.backend.coso.encoder import CoSoEncodingError, CoSoProgram, encode

__all__ = [
    "CoSoBackend",
    "CoSoEncodingError",
    "CoSoProgram",
    "encode",
]
