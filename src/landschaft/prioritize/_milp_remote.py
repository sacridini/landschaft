# SPDX-License-Identifier: GPL-3.0-or-later
"""Compiled prioritization models solved on helpers (``on=ls.helpers()``): the model
travels as its arrays, the solver's result comes back, and the solution is made here as
on one computer."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ._milp import OptimizationProblem


def model_to(m: OptimizationProblem) -> dict:
    """A compiled model as values that travel (``_dist._pack``)."""
    A = sp.csr_matrix(m.A)
    return {"A": (A.data, A.indices, A.indptr, np.array(A.shape, dtype=np.int64)), "obj": m.obj, "lb": m.lb,
            "ub": m.ub, "rhs": m.rhs, "sense": list(m.sense), "vtype": list(m.vtype), "modelsense": m.modelsense,
            "row_ids": list(m.row_ids), "col_ids": list(m.col_ids), "connected": list(m.connected),
            "corridors": list(m.corridors)}


def model_from(d: dict) -> OptimizationProblem:
    data, indices, indptr, shape = d["A"]
    A = sp.csr_matrix((data, indices, indptr), shape=tuple(int(v) for v in shape))
    return OptimizationProblem(A=A, obj=d["obj"], lb=d["lb"], ub=d["ub"], rhs=d["rhs"], sense=tuple(d["sense"]),
                               vtype=tuple(d["vtype"]), modelsense=str(d["modelsense"]),
                               row_ids=tuple(d["row_ids"]), col_ids=tuple(d["col_ids"]),
                               connected=tuple(d["connected"]), corridors=tuple(d["corridors"]))


def options_to(options: dict) -> dict:
    out = {}
    for k, v in options.items():
        if k == "n_threads":
            continue
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, float) and math.isinf(v):
            v = None if k == "time_limit" else v
        out[k] = v
    return out


def options_from(options) -> dict:
    out = dict(options or {})
    if isinstance(out.get("spf"), list):
        out["spf"] = tuple(out["spf"])
    return out


def _model_files(m: OptimizationProblem, folder: Path, name: str) -> list:
    from .._dist import InputFile, _pack

    stored = _pack.write(model_to(m), folder / name)
    return [InputFile.of(p, f"{name}/{p.name}") for p in _pack.files(stored)]


def solve_all(problems: list, on) -> list:
    """The problems solved as tasks of ``on``: their solutions, in order (a failure raises)."""
    from .._dist import Job, Task, _pack, run_job
    from ._milp_compile import compile_problem
    from ._milp_solve import make_solution

    for q in problems:
        if q.portfolio is not None:
            raise ValueError("on= solves single problems, not portfolios")
    with tempfile.TemporaryDirectory(prefix="landschaft-prioritize-") as tmp:
        tmp = Path(tmp)
        compiled, tasks = [], []
        for i, q in enumerate(problems):
            m, targets = compile_problem(q)
            compiled.append((m, targets))
            zoned = q.units.zoned
            if zoned and q.solver[0] == "anneal":
                raise ValueError("the anneal solver does not support zones yet: use HiGHS or Gurobi")
            files = _model_files(m, tmp, f"model_{i}")
            tasks.append(Task(f"problem {i}", "prioritize.solve", {
                "model": f"model_{i}", "solver": q.solver[0], "options": options_to(q.solver[1]),
                "n_units": None if zoned else int(q.units.n), "engines": not zoned,
                "zone_units": int(q.units.n) if zoned else None}, files, root=tmp))
        found = {}

        def collect(task, folder, stats):
            found[task.name] = _pack.read(Path(folder) / "result")

        report = run_job(on, Job("prioritize", tasks, collect, threads=1))
    if report.failed:
        name, error = next(iter(report.failed.items()))
        raise RuntimeError(f"{len(report.failed)} solve(s) failed; {name}: {error}")
    return [make_solution(q.units, m, found[t.name], targets) for q, (m, targets), t in zip(problems, compiled, tasks)]


def replacement_objectives(m: OptimizationProblem, lb, ub, sel, solver: str, opts: dict, n_units: int,
                           prepared: bool, on, chunk: int | None = None) -> np.ndarray:
    """The objective without each selected unit, as tasks of ``on`` (chunks of units)."""
    from .._dist import InputFile, Job, Task, run_job

    sel = [int(i) for i in sel]
    if not sel:
        return np.zeros(0)
    if chunk is None:
        chunk = max(1, math.ceil(len(sel) / 64))
    with tempfile.TemporaryDirectory(prefix="landschaft-replacement-") as tmp:
        tmp = Path(tmp)
        files = _model_files(m, tmp, "model")
        np.save(tmp / "lb.npy", np.asarray(lb, dtype=np.float64))
        np.save(tmp / "ub.npy", np.asarray(ub, dtype=np.float64))
        files += [InputFile.of(tmp / "lb.npy"), InputFile.of(tmp / "ub.npy")]
        parts = [sel[k:k + chunk] for k in range(0, len(sel), chunk)]
        tasks = [Task(f"units {k}", "prioritize.replacement", {
            "model": "model", "lb": "lb.npy", "ub": "ub.npy", "solver": solver, "options": options_to(opts),
            "n_units": int(n_units), "prepared": bool(prepared), "units": part}, files, root=tmp)
            for k, part in enumerate(parts)]
        found = {}

        def collect(task, folder, stats):
            found[task.name] = np.load(Path(folder) / "objectives.npy", allow_pickle=False)

        report = run_job(on, Job("replacement importance", tasks, collect, threads=1))
    if report.failed:
        name, error = next(iter(report.failed.items()))
        raise RuntimeError(f"{len(report.failed)} task(s) failed; {name}: {error}")
    return np.concatenate([found[t.name] for t in tasks])
