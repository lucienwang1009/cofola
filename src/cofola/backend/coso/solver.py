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

    _install_coso_compat_patches()
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


def _install_coso_compat_patches() -> None:
    """Patch known CoSo package bugs without modifying site-packages.

    The PyPI/GitHub snapshot used by the optional dependency has three runtime
    defects that are orthogonal to the CoLa semantics: one logging call omits
    the action argument, recursive solves sometimes wrap ``Solution`` twice,
    and a helper receives unused keyword arguments.  Keeping the fixes here
    makes the backend reproducible across clean environments.
    """

    count_module = import_module("coso.count")
    logger_module = import_module("coso.logger")
    sharp_csp_module = import_module("coso.sharpCSP")

    solution_cls = count_module.Solution
    if not getattr(solution_cls, "_cofola_compat_patched", False):
        original_init = solution_cls.__init__

        def init_compat(self, count, log):  # type: ignore[no-untyped-def]
            if isinstance(count, solution_cls):
                count = count.count
            original_init(self, count, log)

        solution_cls.__init__ = init_compat
        solution_cls._cofola_compat_patched = True

    problem_log_cls = logger_module.ProblemLog
    if not getattr(problem_log_cls, "_cofola_compat_patched", False):
        original_detail = problem_log_cls.detail

        def detail_compat(self, action, msg=None):  # type: ignore[no-untyped-def]
            if msg is None:
                if self.actions:
                    action, msg = self.actions[-1], action
                else:
                    action, msg = self.action("Detail"), action
            return original_detail(self, action, msg)

        problem_log_cls.detail = detail_compat
        problem_log_cls._cofola_compat_patched = True

    sharp_csp_cls = sharp_csp_module.SharpCSP
    if not getattr(sharp_csp_cls, "_cofola_compat_patched", False):
        original_split = sharp_csp_cls.split_on_constraints

        def split_on_constraints_compat(self, n_choices, others, id="1", **kwargs):  # type: ignore[no-untyped-def]
            return original_split(self, n_choices, others, id=id)

        sharp_csp_cls.split_on_constraints = split_on_constraints_compat
        sharp_csp_cls._cofola_compat_patched = True
