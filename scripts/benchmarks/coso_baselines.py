"""Self-contained CoSo external baselines for the benchmark runner.

The translators are local to Cofola and use the installed ``coso`` package only
for CoLa parsing/data structures. They intentionally do not import
``coso.src.tester`` from a source checkout.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import clingo
import portion as P

from cofola.backend.coso.backend import CoSoBackend
from cofola.backend.coso.encoder import encode
from cofola.parser.parser import parse
from cofola.planing.pipeline import PlaningPipeline
from coso.cola_parser import Parser
from coso.configuration import CCounting, CSize
from coso.util import interval_closed


DEFAULT_CONJURE_DIR = Path("/home/sunshixin/lucien/CoSo/tools/conjure")
DEFAULT_JAVA_BIN = Path("/home/sunshixin/lucien/tools/java/bin/java")


class BaselineUnsupportedError(RuntimeError):
    """Raised when a CoSo external baseline cannot encode a case."""


@dataclass(frozen=True)
class ExternalBaselineConfig:
    conjure_dir: Path = DEFAULT_CONJURE_DIR
    java_bin: Path | None = DEFAULT_JAVA_BIN


class _ClingoContext:
    def id(self, x):
        return x

    def seq(self, x, y):
        return [x, y]


def solve_external_baseline(
    source: str,
    backend: str,
    *,
    timeout: float,
    config: ExternalBaselineConfig | None = None,
) -> int:
    """Solve a Cofola program through CoSo's ASP or Essence baseline encoding."""

    config = config or ExternalBaselineConfig()
    cofola_problem = parse(source, debug=False)
    coso_backend = CoSoBackend(debug=False)
    schedule = PlaningPipeline(coso_backend.planning_profile()).process(cofola_problem)

    total = 0
    for branch in schedule.branches:
        branch_total = 1
        for component, analysis in branch.components:
            encoded = encode(component, analysis)
            if encoded.is_trivial:
                component_total = encoded.trivial_count
            else:
                parser = Parser(cola=encoded.cola)
                parser.parse()
                if backend == "asp":
                    component_total = _run_asp(_problem2asp(parser.problem))
                elif backend == "essence":
                    component_total = _run_essence(
                        _problem2essence(parser.problem),
                        timeout=timeout,
                        config=config,
                    )
                else:
                    raise ValueError(f"Unknown external CoSo baseline {backend!r}")
                if encoded.count_divisor != 1:
                    if component_total % encoded.count_divisor != 0:
                        raise ValueError(
                            "External CoSo baseline result is not divisible by indexed "
                            f"composition normalization factor {encoded.count_divisor}: "
                            f"{component_total}"
                        )
                    component_total //= encoded.count_divisor
            branch_total *= component_total
        total += branch_total
    return total


def _get_n_vars(n: int) -> list[str]:
    return [chr(c) for c in range(ord("A"), ord("Z"))][:n]


def _dom2asp(label: str, domain) -> tuple[str, int]:
    text = ""
    i = 0
    indist_intervals = domain.elements.find(False)
    for atomic_interval in indist_intervals:
        if atomic_interval != P.empty():
            e = domain.get_label(atomic_interval.lower, atomic_interval.lower)
            l, u = interval_closed(atomic_interval)
            n_copies = u - l + 1
            text += f'{label}_{i}("{e}",{n_copies}).\n'
            text += f"{label}(X) :- {label}_{i}(X, _).\n"
            i += 1
    if len(indist_intervals) == 0:
        for atomic_interval in domain.elements.find(True):
            l, r = interval_closed(atomic_interval)
            text += f"{label}({l}..{r}).\n"
    else:
        for atomic_interval in domain.elements.find(True):
            for n in P.iterate(atomic_interval, step=1):
                e = domain.get_label(n, n)
                text += f'{label}_{i}("{e}", 1).\n'
                text += f"{label}(X) :- {label}_{i}(X, _).\n"
                i += 1
    text += f"universe(X) :- {label}(X).\n"
    return text, i


def _problem2asp(problem) -> list[str]:
    asp = ""
    n_supports: dict[str, int] = {}
    for lab, dom in problem.domains.items():
        dom_text, n_support = _dom2asp(lab, dom)
        n_supports[lab] = n_support
        asp += dom_text
    if problem.configuration.size is None:
        problem.configuration.size = CSize(
            "unconstrained",
            P.closed(1, problem.universe.size()),
        )
    sizes = problem.configuration.size.values
    if sizes.upper == P.inf:
        sizes = sizes.replace(upper=problem.universe.size() + 1)

    kind = problem.configuration.type
    sequence = kind == "sequence"
    permutation = kind == "permutation"
    subset = kind == "subset"
    multiset = kind == "multisubset"
    composition = kind == "composition"
    partition = kind == "partition"
    universe_is_set = len(problem.universe.elements.find(False)) == 0
    programs: list[str] = []

    for length in P.iterate(sizes, step=1):
        asp_length = ""
        if partition:
            raise BaselineUnsupportedError("ASP baseline translator does not support partition.")
        if not composition:
            vars_ = _get_n_vars(length)
            vars_list = ",".join(vars_)
            name = f"{kind}_guess_{length}"
            domains = []
            new_props = []
            for i, var in enumerate(vars_):
                dom = ""
                for pf in problem.pos_constraints:
                    if pf.pos - 1 == i:
                        if dom:
                            raise BaselineUnsupportedError(
                                "ASP baseline cannot translate multiple position constraints "
                                "on the same position."
                            )
                        dom_text, _ = _dom2asp(f"pf_{i}", pf.formula)
                        new_props.append(dom_text)
                        dom = f"pf_{i}({var})"
                domains.append(dom or f"universe({var})")
            asp_length += f"{name}({vars_list}) :- " + ", ".join(domains)
            inequalities: list[str] = []
            if subset or multiset:
                ineq = "<" if subset and universe_is_set else "<="
                inequalities = [
                    f"{var}{ineq}{vars_[i + 1]}" for i, var in enumerate(vars_[:-1])
                ]
            elif permutation and universe_is_set:
                inequalities = [
                    f"{left}!={right}"
                    for i, left in enumerate(vars_)
                    for j, right in enumerate(vars_)
                    if i < j
                ]
            if inequalities:
                asp_length += ", " + ", ".join(inequalities)
            asp_length += ".\n"
            asp_length += "\n".join(new_props)
            asp_length += f"1{{{kind}_{length}({vars_list}):{name}({vars_list})}}1.\n"
            for i in range(length):
                pos_vars = ", ".join("_" if j != i else "X" for j in range(length))
                asp_length += f"used_{length}(X,{i}) :- {kind}_{length}({pos_vars}). \n"
            if not universe_is_set and (permutation or subset):
                for lab in n_supports:
                    for i in range(n_supports[lab]):
                        asp_length += (
                            f":- {lab}_{i}(S,SN), C = #count{{N:used_{length}(S,N)}}, "
                            "C>SN.\n"
                        )
            for i, cf in enumerate(problem.constraints):
                dlab = f"df_{i}"
                if dlab not in n_supports:
                    dom_text, n = _dom2asp(dlab, cf.formula)
                    n_supports[dlab] = n
                    asp += dom_text
                vals = P.closed(0, length) - cf.values
                for n in P.iterate(vals, step=1):
                    asp_length += (
                        f":- C = #count{{N:used_{length}(S,N),df_{i}(S)}}, C={n}.\n"
                    )
        else:
            asp += f"int(0..{problem.universe.size()}).\n"
            for i in range(length):
                asp_length += f"part({i}).\n"
            for lab in n_supports:
                for i in range(n_supports[lab]):
                    asp_length += (
                        f"1{{put(E,N,P): int(N), N<=EN}} 1 :- {lab}_{i}(E, EN), part(P).\n"
                    )
                    asp_length += (
                        f":- {lab}_{i}(E,EN), #sum{{N,P:put(E,N,P),part(P)}}!=EN.\n"
                    )
            # This follows CoSo's tester semantics: compositions have non-empty parts.
            asp_length += ":- part(P), #count{E,N:put(E,N,P), N>0}==0.\n"
            for i in range(length):
                for pf in problem.pos_constraints:
                    if pf.pos - 1 == i:
                        for j, cof in enumerate(pf.formula.ccs):
                            dlab = f"df_{i}_{j}"
                            if dlab not in n_supports:
                                dom_text, n = _dom2asp(dlab, cof.formula)
                                n_supports[dlab] = n
                                asp += dom_text
                            vals = P.closed(0, length) - cof.values
                            for n in P.iterate(vals, step=1):
                                asp_length += (
                                    f":- C=#sum{{N,E:put(E,N,{i}), {dlab}(E)}}, C={n}.\n"
                                )
                        size = pf.formula.size.values
                        if size.lower != 1 or size.upper != length:
                            vals = P.closed(0, length) - size
                            for n in P.iterate(vals, step=1):
                                asp_length += f":- C=#sum{{N,E:put(E,N,{i})}}, C={n}.\n"
            for i, outer in enumerate(problem.constraints):
                inner = outer.formula
                if isinstance(inner, CCounting):
                    dlab = f"df_{i}"
                    if dlab not in n_supports:
                        dom_text, n = _dom2asp(dlab, inner.formula)
                        n_supports[dlab] = n
                        asp += dom_text
                else:
                    dlab = "universe"
                lb = inner.values.lower if inner.values.left == P.CLOSED else inner.values.lower + 1
                ub = max(length + 1, lb + 1)
                vals = inner.values if inner.values.upper != P.inf else inner.values.replace(upper=ub)
                count_pred = []
                count_vars = []
                for n in P.iterate(vals, step=1):
                    asp_length += (
                        f"cf_{i}_{n}(P,S) :- S=#sum{{N,E:put(E,N,P), {dlab}(E)}}, "
                        f"part(P), S={n}.\n"
                    )
                    count_pred.append(f"C{n}=#count{{P:cf_{i}_{n}(P,{n})}}")
                    count_vars.append(f"C{n}")
                asp_length += f"count_{i}(C) :- " + ", ".join(count_pred)
                asp_length += ", C=" + "+".join(count_vars) + ".\n"
                vals = P.closed(0, length) - outer.values
                for n in P.iterate(vals, step=1):
                    asp_length += f":- count_{i}({n}).\n"
        programs.append(asp + asp_length)
    return programs


def _run_asp(programs: list[str]) -> int:
    total = 0
    for program in programs:
        ctl = clingo.Control()
        ctl.configuration.solve.models = 0
        ctl.add("base", [], program)
        ctl.ground([("base", [])], context=_ClingoContext())
        with ctl.solve(yield_=True, async_=True) as handle:
            total += sum(1 for _ in handle)
    return total


def _essence_name(label: str) -> str:
    if label in [str(i) for i in range(10)]:
        label = f"e_{label}"
    if label == "true":
        label = "MYtrue"
    if label == "false":
        label = "MYfalse"
    return label.replace("∧", "and").replace("¬", "not").replace("∨", "or")


def _dom2essence(label: str, domain) -> str:
    copies = {}
    for atomic_interval in domain.elements.find(False):
        entity = _essence_name(domain.get_label(atomic_interval.lower, f"e_{atomic_interval.lower}"))
        if atomic_interval != P.empty():
            lower, upper = interval_closed(atomic_interval)
            copies[entity] = upper - lower + 1
    for atomic_interval in domain.elements.find(True):
        for n in P.iterate(atomic_interval, step=1):
            copies[_essence_name(domain.get_label(n, f"e_{n}"))] = 1
    entity_list = ", ".join(copies.keys())
    label = _essence_name(label)
    if domain.universe == domain or domain.formula == "u":
        function_list = ", ".join(f"{e} --> {n}" for e, n in copies.items())
        return (
            f"letting universe be new type enum {{ {entity_list} }}\n"
            f"letting f_universe be function({function_list})\n"
        )
    return f"letting {label} be {{ {entity_list} }}\n"


def _range2essence(interval, name: str, ub: int) -> str:
    if interval.upper == P.inf:
        interval = interval.replace(upper=ub + 1)
    range_vals = ",".join(str(i) for i in P.iterate(interval, step=1))
    return f"letting {name} be {{ {range_vals} }}\n"


def _problem2essence(problem) -> list[str]:
    essence = ""
    added_doms: list[str] = []
    univ_str = _dom2essence("universe", problem.universe)
    essence += univ_str
    for lab, dom in problem.domains.items():
        dom_str = _dom2essence(lab, dom)
        if dom_str != univ_str:
            added_doms.append(lab)
            essence += dom_str
    if problem.configuration.size is None:
        problem.configuration.size = CSize(
            "unconstrained",
            P.closed(1, problem.universe.size()),
        )
    if problem.configuration.size.values.upper == P.inf:
        sizes = problem.configuration.size.values.replace(upper=problem.universe.size() + 1)
    else:
        sizes = problem.configuration.size.values

    kind = problem.configuration.type
    sequence = kind == "sequence"
    permutation = kind == "permutation"
    subset = kind == "subset"
    multiset = kind == "multisubset"
    composition = kind == "composition"
    partition = kind == "partition"
    uni = "universe"
    programs: list[str] = []
    for length in P.iterate(sizes, step=1):
        name = f"conf_{length}"
        essence_l = f"letting l_{length} be {length}\n"
        constraints: list[str] = []
        if not composition and not partition:
            if permutation or subset:
                if permutation:
                    constraints.append(
                        f"\tforAll e: {uni}.\n"
                        f"\t\tsum([1 | i: int(1..l_{length}), {name}(i)=e]) <= f_{uni}(e)\n "
                    )
                else:
                    constraints.append(
                        f"\tforAll e: universe.\n\t\tfreq({name},e) <= f_{uni}(e)"
                    )
            for i in range(length):
                for j, pf in enumerate(problem.pos_constraints):
                    if pf.pos - 1 == i:
                        dlab = f"pf_{i}_{j}"
                        if dlab not in added_doms:
                            essence += _dom2essence(dlab, pf.formula)
                            added_doms.append(dlab)
                        constraints.append(f"{name}({i + 1}) in {dlab}")
            for i, cf in enumerate(problem.constraints):
                dlab = f"df_{i}"
                if dlab not in added_doms:
                    essence += _dom2essence(dlab, cf.formula)
                    added_doms.append(dlab)
                vals = cf.values if cf.values.upper != P.inf else cf.values.replace(upper=length + 1)
                range_vals = ",".join(str(v) for v in P.iterate(vals, step=1))
                essence_l += f"letting vals_{i} be {{ {range_vals} }}\n"
                if sequence or permutation:
                    constraints.append(
                        f"sum([1 | i: int(1..l_{length}), {name}(i) in {dlab}]) in vals_{i}"
                    )
                else:
                    constraints.append(
                        f"sum([freq({name}, i) | i: {uni}, i in {dlab}]) in vals_{i}"
                    )
            if sequence or permutation:
                essence_l += f"find {name} : sequence (size l_{length}) of {uni}\n"
            if multiset or subset:
                essence_l += f"find {name} : mset (size l_{length}) of {uni}\n"
            if constraints:
                essence_l += "such that \n"
        elif composition:
            myparts = [f"p{i}" for i in range(1, length + 1)]
            essence_l += "letting myparts be new type enum {" + ",".join(myparts) + "}\n"
            essence_l += f"letting n be {problem.universe.size()}\n"
            constraints.append("forAll p: myparts.\n\t sum([put[e,p] | e:universe]) > 0")
            constraints.append(
                "forAll e: universe.\n\t sum([put[e,p] | p: myparts]) = f_universe(e)"
            )
            ub = problem.universe.size() - length + 1
            for i in range(1, length + 1):
                for j, pf in enumerate(problem.pos_constraints):
                    if pf.pos == i:
                        if pf.formula.size.values not in (P.closed(1, ub), P.closed(1, P.inf)):
                            range_name = f"s_{i}"
                            essence_l += _range2essence(pf.formula.size.values, range_name, ub)
                            constraints.append(f"sum([put[e,p{i}] | e:{uni}]) in {range_name}")
                        else:
                            for k, cof in enumerate(pf.formula.ccs):
                                dlab = f"df_{i}_{j}_{k}"
                                if dlab not in added_doms:
                                    essence += _dom2essence(dlab, cof.formula)
                                    added_doms.append(dlab)
                                range_name = f"vals_{i}_{j}_{k}"
                                essence_l += _range2essence(cof.values, range_name, ub)
                                constraints.append(
                                    f"sum([put[e,p{i}] | e<-{dlab}]) in {range_name}"
                                )
            for i, cf in enumerate(problem.constraints):
                essence_l += _range2essence(cf.values, f"vals_{i}_out", length)
                essence_l += _range2essence(cf.formula.values, f"vals_{i}_in", ub)
                dlab = f"df_{i}"
                if isinstance(cf.formula, CSize):
                    dom_str = "universe"
                else:
                    if dlab not in added_doms:
                        essence += _dom2essence(dlab, cf.formula.formula)
                        added_doms.append(dlab)
                    dom_str = dlab
                constraints.append(
                    f"|[p | p:myparts, sum([put[e,p] | e <- {dom_str}]) in vals_{i}_in]| "
                    f"in vals_{i}_out"
                )
            essence_l += "find put: matrix indexed by [universe, myparts] of int(0..n)\n"
            essence_l += "such that \n"
        else:
            # Keep CoSo tester behavior for partition; the benchmark runner will
            # mark wrong answers as unsolved.
            pass
        essence_l += "\n\t/\\ ".join(constraints)
        essence_l += "\n"
        programs.append(essence + essence_l)
    return programs


def _run_essence(
    programs: list[str],
    *,
    timeout: float,
    config: ExternalBaselineConfig,
) -> int:
    conjure = config.conjure_dir / "conjure"
    if not conjure.exists():
        raise RuntimeError(f"Essence baseline requires Conjure at {conjure}")
    env = os.environ.copy()
    env["PATH"] = str(config.conjure_dir.resolve()) + os.pathsep + env["PATH"]
    if config.java_bin is not None and config.java_bin.exists():
        env["PATH"] = str(config.java_bin.parent.resolve()) + os.pathsep + env["PATH"]
    if shutil.which("java", path=env["PATH"]) is None:
        raise RuntimeError("Essence baseline requires java on PATH for Savile Row")

    total = 0
    with tempfile.TemporaryDirectory(prefix="cofola-essence-") as tmp:
        tmp_path = Path(tmp)
        for index, program in enumerate(programs):
            model = tmp_path / f"model_{index}.essence"
            out_dir = tmp_path / f"conjure-output-{index}"
            model.write_text(program)
            proc = subprocess.Popen(
                [
                    str(conjure),
                    "solve",
                    "-ac",
                    "-o",
                    str(out_dir),
                    "--solutions-in-one-file",
                    "--number-of-solutions=all",
                    "--log-level",
                    "lognone",
                    str(model),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()
                raise TimeoutError(f"Essence timed out after {timeout:.3f}s") from exc
            if proc.returncode != 0:
                raise RuntimeError((stderr or stdout or "Conjure failed").splitlines()[0])
            solution_file = out_dir / "model000001.solutions"
            if not solution_file.exists():
                raise RuntimeError("Conjure did not produce model000001.solutions")
            solutions = solution_file.read_text()
            n_solutions = solutions.count("Count:") or solutions.count("$ Solution:")
            total += n_solutions
    return total
