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

    raw = run_coso(cola=_normalize_cola_size_constraints(cola), debug=debug)
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
    solver_module = import_module("coso.solver")
    level_2_module = import_module("coso.level_2")
    configuration_module = import_module("coso.configuration")
    portion_module = import_module("portion")

    solution_cls = count_module.Solution
    zero_cls = count_module.Zero
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

    if not getattr(sharp_csp_cls, "_cofola_multisubsets_patched", False):
        binomial_cls = count_module.Binomial

        def count_multisubsets_exchangeable_compat(self, var_list=None):  # type: ignore[no-untyped-def]
            variables = var_list if var_list is not None else self.vars
            n = len(variables)
            domain = variables[0]
            indist = domain.elements.find(False)
            indist_sizes = self.get_sizes_indistinguishable(indist)
            indist_size = sum(indist_sizes)
            support_size = domain.size() - indist_size + len(indist_sizes)
            m = support_size + n - 1
            action = self.log.action("Counting multisubsets with all exchangeable")
            self.log.detail(action, f"With binomial coefficient ({m} {n})")
            return binomial_cls(m, n, f"Choose {n} objects out of {m} distinguishable objects")

        sharp_csp_cls.count_multisubsets_exchangeable = count_multisubsets_exchangeable_compat
        sharp_csp_cls._cofola_multisubsets_patched = True

    if not getattr(sharp_csp_cls, "_cofola_count_ranges_patched", False):
        original_apply_count_constraint = sharp_csp_cls.apply_count_constraint
        original_apply_propositional_count = sharp_csp_cls.apply_propositional_count

        def apply_count_constraint_return_inverse(self, cc, others):  # type: ignore[no-untyped-def]
            out_values = portion_module.closedopen(0, self.config.size.values.upper) - cc.values
            if not cc.values.atomic and out_values.atomic:
                action = self.log.action(f"Considering constraint {cc}")
                self.log.add_relevant_set(cc.formula)
                self.log.detail(action, f"Relax {cc} and remove unsat (={out_values})")
                count_ignore = self.split_on_constraints(1, others, op="add")
                not_cc = configuration_module.CCounting(cc.formula, out_values)
                count_not = self.split_on_constraints(1, [not_cc] + others, op="sub", id="2")
                return count_ignore - count_not
            return original_apply_count_constraint(self, cc, others)

        def apply_propositional_count_no_early_stop(self, cc, others, ub, action):  # type: ignore[no-untyped-def]
            values = cc.values.replace(upper=ub, right=portion_module.CLOSED)
            count = zero_cls()
            for i in portion_module.iterate(values, step=1):
                cc_eq = configuration_module.CCounting(cc.formula, portion_module.singleton(i))
                count_case = self.solve_subproblem(
                    self.vars,
                    self.config,
                    {},
                    [cc_eq] + others,
                    caption=f"Case with {i} {cc.formula}",
                    op="add",
                    id=str(i),
                )
                count += count_case
            return count

        sharp_csp_cls.apply_count_constraint = apply_count_constraint_return_inverse
        sharp_csp_cls.apply_propositional_count = apply_propositional_count_no_early_stop
        sharp_csp_cls._cofola_count_ranges_patched = True

    lifted_set_cls = level_2_module.LiftedSet
    if not getattr(lifted_set_cls, "_cofola_compat_patched", False):
        original_lifted_set_eq = lifted_set_cls.__eq__

        def lifted_set_eq_compat(self, rhs):  # type: ignore[no-untyped-def]
            if not isinstance(rhs, lifted_set_cls):
                return False
            return original_lifted_set_eq(self, rhs)

        lifted_set_cls.__eq__ = lifted_set_eq_compat
        lifted_set_cls._cofola_compat_patched = True

    solver_cls = solver_module.Solver
    if not getattr(solver_cls, "_cofola_empty_level2_patched", False):
        csize_cls = configuration_module.CSize
        sharp_csp_cls = sharp_csp_module.SharpCSP

        def solve_allowing_empty_level2_groups(self):  # type: ignore[no-untyped-def]
            unsat, msg = self.trivial_unsat()
            if unsat:
                return solution_cls(zero_cls(tip=msg), self.log)

            count = zero_cls()
            for n in self.size:
                if self.config.lvl1():
                    variables = [self.universe] * n
                else:
                    size = csize_cls("universe", portion_module.closed(0, self.universe.size()))
                    variables = [lifted_set_cls(self.universe, size)] * n
                csp = sharp_csp_cls(
                    variables,
                    self.config,
                    self.problem.pos_constraints,
                    self.problem.constraints,
                    self.universe,
                    caption=f"Configuration of size {n}",
                    lvl=1,
                    debug=self.debug,
                )
                size_solution = csp.solve()
                count += size_solution.count
                self.log.add_subproblem("add", size_solution.log)
            return solution_cls(count, self.log)

        solver_cls.solve = solve_allowing_empty_level2_groups
        solver_cls._cofola_empty_level2_patched = True


def _normalize_cola_size_constraints(cola: str) -> str:
    """Work around CoSo's open-lower-bound bug for non-empty config sizes."""

    import re

    match = re.search(r"(?m)^\s*([a-z][a-zA-Z\-_0-9]*)\s+in\s+", cola)
    if match is None:
        return cola

    name = re.escape(match.group(1))
    cola = re.sub(rf"(?m)(^\s*#{name})\s*>\s*0\s*;", rf"\1>=1;", cola)
    cola = re.sub(rf"(?m)(^\s*#{name})\s*!=\s*0\s*;", rf"\1>=1;", cola)
    return cola
