"""CoSo backend package."""
from __future__ import annotations

from cofola.backend.coso.backend import COSO_GLOBAL_PASSES, COSO_LOCAL_PASSES, CoSoBackend
from cofola.backend.coso.encoder import CoSoEncodingError, CoSoProgram, encode

__all__ = [
    "CoSoBackend",
    "COSO_GLOBAL_PASSES",
    "CoSoEncodingError",
    "CoSoProgram",
    "COSO_LOCAL_PASSES",
    "encode",
]
