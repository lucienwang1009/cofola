"""Thin dynamic wrapper around the optional CoSo dependency."""
from __future__ import annotations

from importlib import import_module

__all__ = ["CoSoSolverError", "run_coso_program"]


class CoSoSolverError(Exception):
    """Raised when the optional CoSo solver cannot be invoked."""


def run_coso_program(cola: str, *, debug: bool = False) -> int:
    """Run CoSo on a CoLa program and return its integer count.

    CoSo is kept as an optional dependency because it brings a larger solver
    stack. Import it only when this backend is selected.
    """

    try:
        launcher = import_module("coso.launcher")
    except ImportError as exc:
        raise CoSoSolverError(
            "CoSo is not installed. Install the optional backend with "
            "`uv sync --extra coso` before using `--backend coso`."
        ) from exc

    run_coso = getattr(launcher, "run_coso", None)
    if run_coso is None:
        raise CoSoSolverError("Installed CoSo package does not expose coso.launcher.run_coso.")

    raw = run_coso(cola=cola, debug=debug)
    if raw is None:
        raise CoSoSolverError("CoSo returned no result.")

    text = str(raw).strip()
    first = text.split(maxsplit=1)[0] if text else ""
    try:
        return int(first)
    except ValueError as exc:
        raise CoSoSolverError(f"Could not parse CoSo result as an integer: {text!r}") from exc
