# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark MILP solvers on prioritizr / prioriactions models (phase 0 of the MILP prioritization).

Usage::

    python tools/benchmark_optimize.py [--families minset,boundary] [--sizes 1e3,1e4] [--solvers highs,gurobi]
        [--threads N] [--gap 0.01] [--time-limit 300] [--work DIR] [--cpus 0-11] [--experiments threads,instances]

For every family and size (planning units), writes seeded synthetic inputs
(features: Gaussian random fields above a quantile, species-like; cost: a
field from 1 to 10; prioriactions tables on a grid of units), builds the
model in R with prioritizr or prioriactions (``tools/optimize_benchmark.R``,
timing ``compile()``, the baseline of landschaft's model compiler) and solves
the same matrices on every solver, each solve in its own process (HiGHS keeps
a global thread scheduler per process). Families:

- ``minset``: minimum set, 30 features, relative targets 30%, binary;
- ``boundary``: the same with boundary penalties;
- ``contiguity``: the same with contiguity constraints (small sizes only);
- ``maxutil``: maximum utility with a budget of 20% of the total cost;
- ``actions``: prioriactions minimizeCosts, 10 features, 3 threats;
- ``zones_minset``, ``zones_maxutil``, ``zones_boundary``: the same with three management zones
  (protection: the features, cost 1 to 10; sustainable use: half of them, cost 0.5 to 5; restoration:
  a gain from 0.2 to 0.8 of them, cost 1 to 6), min set targets summed over the zones (30% of the
  features' totals), max utility with one budget.

Solvers: ``highs`` (highspy, development environment only), ``builtin`` (the
HiGHS built into landschaft, patched, on the compiled model), ``zones`` (landschaft's
zones engine, zone minimum sets), ``gurobi``
(gurobipy) and ``prioritizr`` (the package's own solve on Gurobi, build +
solve). Experiments, at ``--experiment-gap`` (10% by default, prioritizr's
default): ``threads`` (HiGHS and Gurobi on 1, 2, 4 and N threads on one
model) and ``instances`` (K copies of one model solved one after the
other and at once, one HiGHS thread each: the global scheduler and the GIL).
Results go to ``<work>/results.json`` and a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

R_SCRIPT = ROOT / "tools" / "optimize_benchmark.R"
FAMILIES = ["minset", "boundary", "contiguity", "maxutil", "actions", "zones_minset", "zones_maxutil", "zones_boundary"]
# largest default size per family (the larger ones take hours or do not fit)
MAX_SIZE = {"minset": 10**6, "boundary": 10**6, "contiguity": 10**4, "maxutil": 10**6, "actions": 10**5,
            "zones_minset": 10**6, "zones_maxutil": 10**6, "zones_boundary": 10**5}
N_FEATURES = 30
OCCUPIED = 0.3


# ----------------------------------------------------------------------------- inputs

def _field(side: int, seed: int) -> np.ndarray:
    import landschaft as ls

    return ls.nlm.gaussian_field(side, side, autocorr_range=max(side / 25, 2), seed=seed).values


def _write(path: Path, values: np.ndarray) -> None:
    from optimize_cases import grid

    import landschaft as ls

    ls.write_raster(grid(values, res=1.0), path, compress="none")


def raster_inputs(work: Path, side: int) -> tuple[Path, list[Path]]:
    """Cost (1 to 10) and species-like features (field above its 70% quantile, minus the quantile)."""
    d = work / f"raster_{side}"
    d.mkdir(parents=True, exist_ok=True)
    cost = d / "cost.tif"
    if not cost.exists():
        c = _field(side, 1)
        _write(cost, 1 + 9 * (c - c.min()) / (c.max() - c.min()))
    feats = []
    for i in range(N_FEATURES):
        f = d / f"f{i}.tif"
        if not f.exists():
            v = _field(side, 1000 + i)
            cut = np.quantile(v, 1 - OCCUPIED)
            _write(f, np.where(v > cut, v - cut, 0.0))
        feats.append(f)
    return cost, feats


def zone_inputs(work: Path, side: int) -> dict[str, str]:
    """The other zones' costs (use: 0.5 to 5, restoration: 1 to 6) and restoration's gain (0.2 to 0.8)."""
    d = work / f"raster_{side}"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, seed, lo, hi in (("cost_use", 2, 0.5, 5.0), ("cost_restore", 3, 1.0, 6.0), ("gain", 4, 0.2, 0.8)):
        f = d / f"{name}.tif"
        if not f.exists():
            v = _field(side, seed)
            _write(f, lo + (hi - lo) * (v - v.min()) / (v.max() - v.min()))
        out[name] = f.as_posix()
    return out


def _table(d: Path, name: str, columns: dict[str, np.ndarray]) -> dict:
    files = []
    for k, (col, v) in enumerate(columns.items()):
        p = d / f"{name}_{k}.bin"
        np.asarray(v, dtype="<f8").tofile(p)
        files.append({"name": col, "file": p.as_posix()})
    return {"nrow": len(next(iter(columns.values()))),
            "columns": {"name": [f["name"] for f in files], "file": [f["file"] for f in files]}}


def _presence(side: int, seed: int, share: float) -> np.ndarray:
    v = _field(side, seed).ravel()
    return v > np.quantile(v, 1 - share)


def actions_inputs(work: Path, side: int) -> dict:
    """prioriactions tables on a side x side grid: 10 features and 3 threats (fields above a quantile)."""
    d = work / f"actions_{side}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n = side * side
    feat = np.stack([_presence(side, 2000 + k, 0.3) for k in range(10)])
    thr = np.stack([_presence(side, 3000 + k, 0.4) for k in range(3)])
    ids = np.arange(1.0, n + 1)
    fp, fk = np.nonzero(feat.T)
    tp, tk = np.nonzero(thr.T)
    recoverable = (feat & thr.any(axis=0)).sum(axis=1)
    sf, st = np.meshgrid(np.arange(1.0, 11.0), np.arange(1.0, 4.0))
    return {
        "pu": _table(d, "pu", {"id": ids, "monitoring_cost": rng.integers(1, 4, n), "status": np.zeros(n)}),
        "features": _table(d, "features", {"id": np.arange(1.0, 11.0),
                                           "target_recovery": np.floor(0.3 * recoverable)}),
        "dist_features": _table(d, "dist_features", {"pu": fp + 1.0, "feature": fk + 1.0, "amount": np.ones(fp.size)}),
        "threats": _table(d, "threats", {"id": np.arange(1.0, 4.0), "blm_actions": np.zeros(3)}),
        "dist_threats": _table(d, "dist_threats", {"pu": tp + 1.0, "threat": tk + 1.0, "amount": np.ones(tp.size),
                                                   "action_cost": rng.integers(1, 6, tp.size),
                                                   "status": np.zeros(tp.size)}),
        "sensitivity": _table(d, "sensitivity", {"feature": sf.ravel(), "threat": st.ravel()}),
    }


# ----------------------------------------------------------------------------- models

def build_model(work: Path, family: str, side: int, prioritizr_solve: dict | None = None) -> Path:
    """Compile the problem in R (once; cached in the work directory); with prioritizr_solve also solve it there."""
    import lsm_reference

    mdir = work / "models" / f"{family}_{side}"
    done = mdir / "result.json"
    if done.exists() and prioritizr_solve is None:
        return mdir
    mdir.mkdir(parents=True, exist_ok=True)
    if family == "actions":
        job = {"family": family, "tables": actions_inputs(work, side)}
    else:
        cost, feats = raster_inputs(work, side)
        job = {"family": family, "cost": cost.as_posix(), "features": [f.as_posix() for f in feats],
               "target": 0.3, "penalty": 0.5, "budget": 0.2 * 5.5 * side * side}
        if family.startswith("zones_"):
            job["zones"] = zone_inputs(work, side)
    out = mdir
    if prioritizr_solve is not None:
        job.update(prioritizr_solve, solve=True)
        out = Path(tempfile.mkdtemp(prefix="opt_bench_"))
    (out / "job.json").write_text(json.dumps(job), encoding="utf-8")
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    subprocess.run([str(rscript), str(R_SCRIPT), str(out / "job.json"), str(out)], check=True)
    if prioritizr_solve is not None:
        return out
    return mdir


def load_model(mdir: Path):
    from optimize_reference import Model

    code = {"L": "<=", "G": ">=", "E": "="}
    sense = [code[c] for c in (mdir / "sense.txt").read_text().strip()]
    return Model(
        A_i=np.fromfile(mdir / "A_i.bin", "<i4").astype(np.int64), A_j=np.fromfile(mdir / "A_j.bin", "<i4").astype(np.int64),
        A_x=np.fromfile(mdir / "A_x.bin", "<f8"), obj=np.fromfile(mdir / "obj.bin", "<f8"),
        lb=np.fromfile(mdir / "lb.bin", "<f8"), ub=np.fromfile(mdir / "ub.bin", "<f8"),
        rhs=np.fromfile(mdir / "rhs.bin", "<f8"), sense=sense, vtype=list((mdir / "vtype.txt").read_text().strip()),
        modelsense=(mdir / "modelsense.txt").read_text().strip())


# ----------------------------------------------------------------------------- solving (one per process)

def highs_model(m):
    import highspy
    from optimize_reference import _row_bounds

    a = m.csr()
    lo, hi = _row_bounds(m)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m.obj.size, m.rhs.size
    lp.col_cost_ = m.obj
    lp.col_lower_ = m.lb
    lp.col_upper_ = np.where(np.isinf(m.ub), highspy.kHighsInf, m.ub)
    lp.row_lower_ = np.where(np.isinf(lo), -highspy.kHighsInf, lo)
    lp.row_upper_ = np.where(np.isinf(hi), highspy.kHighsInf, hi)
    lp.sense_ = highspy.ObjSense.kMaximize if m.modelsense == "max" else highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = a.indptr
    lp.a_matrix_.index_ = a.indices
    lp.a_matrix_.value_ = a.data
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = m.obj.size, m.rhs.size
    integer = np.isin(np.asarray(m.vtype), ("B", "I"))
    lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in integer]
    return lp


MILESTONES = (0.10, 0.05, 0.01)


class Trajectory:
    """The incumbent and the bound over the solve: (seconds, primal, dual) at every change."""

    def __init__(self):
        self.points: list[tuple[float, float, float]] = []

    def add(self, t: float, primal: float, dual: float) -> None:
        if not self.points or self.points[-1][1:] != (primal, dual):
            self.points.append((t, primal, dual))

    def summary(self) -> dict:
        """Seconds until the relative gap |primal - dual| / |primal| first falls to each milestone, and the
        trajectory thinned to at most 200 points."""
        reach = {}
        for g in MILESTONES:
            hit = [t for t, p, d in self.points if math.isfinite(p) and math.isfinite(d)
                   and abs(p - d) <= g * max(abs(p), 1e-10)]
            reach[f"{g:g}"] = hit[0] if hit else None
        pts = self.points
        if len(pts) > 200:
            pts = [pts[i] for i in np.unique(np.linspace(0, len(pts) - 1, 200).round().astype(int))]
        clean = [[t, p if math.isfinite(p) else None, d if math.isfinite(d) else None] for t, p, d in pts]
        return {"seconds_to_gap": reach, "trajectory": clean}


def solve_highs(m, threads: int, gap: float, time_limit: float, log: bool = False, log_file: str = "") -> dict:
    import highspy

    lp = highs_model(m)
    h = highspy.Highs()
    h.setOptionValue("output_flag", log or bool(log_file))
    if log_file:
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", log_file)
    h.setOptionValue("threads", threads)
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("time_limit", time_limit)
    traj = Trajectory()
    h.setCallback(lambda ctype, msg, out, inp, ud: traj.add(out.running_time, out.mip_primal_bound,
                                                             out.mip_dual_bound), None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
    t0 = time.perf_counter()
    h.passModel(lp)
    t1 = time.perf_counter()
    h.run()
    t2 = time.perf_counter()
    info = h.getInfo()
    traj.add(t2 - t1, info.objective_function_value, info.mip_dual_bound)
    return {"status": h.modelStatusToString(h.getModelStatus()), "objective": info.objective_function_value,
            "bound": info.mip_dual_bound, "gap": info.mip_gap, "nodes": info.mip_node_count,
            "load_seconds": t1 - t0, "seconds": t2 - t1, "version": h.version(), **traj.summary()}


def solve_gurobi(m, threads: int, gap: float, time_limit: float, log: bool = False, log_file: str = "") -> dict:
    import gurobipy as gp

    a = m.csr()
    vtype = {"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER}
    t0 = time.perf_counter()
    params = {"OutputFlag": int(log or bool(log_file)), "LogToConsole": int(log)}
    if log_file:
        params["LogFile"] = log_file
    with gp.Env(params=params) as env, gp.Model(env=env) as g:
        x = g.addMVar(m.obj.size, lb=m.lb, ub=np.where(np.isinf(m.ub), gp.GRB.INFINITY, m.ub),
                      vtype=[vtype.get(t, gp.GRB.CONTINUOUS) for t in m.vtype])
        g.addMConstr(a, x, np.asarray([s[0] for s in m.sense]), m.rhs)
        g.setObjective(m.obj @ x, gp.GRB.MAXIMIZE if m.modelsense == "max" else gp.GRB.MINIMIZE)
        g.Params.Threads = threads
        g.Params.MIPGap = gap
        g.Params.TimeLimit = time_limit
        g.update()
        traj = Trajectory()

        def cb(model, where):
            if where == gp.GRB.Callback.MIP:
                traj.add(model.cbGet(gp.GRB.Callback.RUNTIME), model.cbGet(gp.GRB.Callback.MIP_OBJBST),
                         model.cbGet(gp.GRB.Callback.MIP_OBJBND))

        t1 = time.perf_counter()
        g.optimize(cb)
        t2 = time.perf_counter()
        ok = g.SolCount > 0
        if ok:
            traj.add(g.Runtime, g.ObjVal, g.ObjBound)
        return {"status": {gp.GRB.OPTIMAL: "Optimal", gp.GRB.TIME_LIMIT: "Time limit reached"}.get(g.Status, str(g.Status)),
                "objective": g.ObjVal if ok else math.nan, "bound": g.ObjBound, "gap": g.MIPGap if ok else math.inf,
                "nodes": g.NodeCount, "load_seconds": t1 - t0, "seconds": t2 - t1,
                "version": ".".join(map(str, gp.gurobi.version())), **traj.summary()}


def solve_builtin(m, threads: int, gap: float, time_limit: float, log: bool = False, log_file: str = "") -> dict:
    """The HiGHS built into landschaft (patched) on the compiled model, as Problem.solve calls it."""
    from landschaft import _core

    a = m.csr()
    lo, hi = _row_bounds_of(m)
    t0 = time.perf_counter()
    r = _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=m.obj, lb=m.lb,
        ub=m.ub, row_lo=lo, row_hi=hi, integer=np.isin(np.asarray(m.vtype), ("B", "I")).astype(np.uint8),
        maximize=m.modelsense == "max", offset=0.0, gap=gap, time_limit=time_limit, threads=threads,
        presolve="choose", verbose=log, relaxation=False, initial=None)
    t1 = time.perf_counter()
    return {"status": "Optimal" if r["status"] == "optimal" else r["status"], "objective": r["objective"],
            "bound": r["bound"], "gap": r["gap"], "nodes": r["nodes"], "seconds": t1 - t0}


def solve_zones_engine(m, threads: int, gap: float, time_limit: float, log: bool = False, log_file: str = "") -> dict:
    """landschaft's zones engines on a three-zone model: opt/zones.cpp (minimum sets), else opt/zones_wide.cpp."""
    from landschaft import _core

    a = m.csr()
    lo, hi = _row_bounds_of(m)
    t0 = time.perf_counter()
    kw = {"start": a.indptr.astype(np.int64), "index": a.indices.astype(np.int64), "value": a.data, "obj": m.obj,
          "lb": m.lb, "ub": m.ub, "row_lo": lo, "row_hi": hi,
          "integer": np.isin(np.asarray(m.vtype), ("B", "I")).astype(np.uint8), "maximize": m.modelsense == "max",
          "offset": 0.0, "gap": gap, "time_limit": time_limit, "threads": threads, "presolve": "choose",
          "verbose": log, "relaxation": False, "initial": None}
    units = _zone_units_of(m)
    if m.obj.size > 3 * units:  # pair columns (boundary penalties): the zones cut engine
        from landschaft.prioritize._milp import OptimizationProblem
        from landschaft.prioritize._zones import solve_zone_cut, zone_cut_model

        nnz = np.diff(a.indptr)
        ids = np.where((nnz == 3) & (hi == 1.0), "pu_zone", np.where(nnz == 2, "b1", "spp_target"))
        cols = np.array(["pu"] * (3 * units) + ["b"] * (m.obj.size - 3 * units))
        om = OptimizationProblem(A=a, obj=m.obj, lb=m.lb, ub=m.ub, rhs=m.rhs, sense=tuple(m.sense),
                                 vtype=tuple(m.vtype), modelsense=m.modelsense, row_ids=tuple(ids), col_ids=tuple(cols))
        zm = zone_cut_model(om, units, 3)
        if zm is not None:
            r = solve_zone_cut(om, zm, gap, time_limit, log, None, threads, "choose")
            t1 = time.perf_counter()
            return {"status": "Optimal" if r["status"] == "optimal" else r["status"], "objective": r["objective"],
                    "bound": r["bound"], "gap": r["gap"], "nodes": r.get("nodes", 0), "seconds": t1 - t0}
    try:
        r = _core.opt_solve_highs(**kw, method="zones", n_units=units)
    except ValueError:  # budgeted objectives
        r = _core.opt_solve_highs(**kw, method="zones_wide", n_units=units)
    t1 = time.perf_counter()
    return {"status": "Optimal" if r["status"] == "optimal" else r["status"], "objective": r["objective"],
            "bound": r["bound"], "gap": r["gap"], "nodes": r["nodes"], "seconds": t1 - t0}


def _zone_units_of(m) -> int:
    """The units of a three-zone model: the rows that hold three columns of 1 with an upper bound of 1."""
    a = m.csr()
    nnz = np.diff(a.indptr)
    _, hi = _row_bounds_of(m)
    gub = (nnz == 3) & (hi == 1.0)
    return int(gub.sum())


def _row_bounds_of(m):
    from optimize_reference import _row_bounds

    return _row_bounds(m)


def run_one(args) -> None:
    """--run-one: solve a model directory in this process and print the result as JSON."""
    m = load_model(Path(args.model))
    fn = {"highs": solve_highs, "gurobi": solve_gurobi, "builtin": solve_builtin,
          "zones": solve_zones_engine}[args.solver]
    log_file = Path(args.model) / f"{args.solver}_t{args.threads}.log"
    log_file.unlink(missing_ok=True)
    print(json.dumps(fn(m, args.threads, args.gap, args.time_limit, log_file=str(log_file))))


def solve_in_process(mdir: Path, solver: str, threads: int, gap: float, time_limit: float) -> dict:
    cmd = [sys.executable, __file__, "--run-one", "--model", str(mdir), "--solver", solver, "--threads", str(threads),
           "--gap", str(gap), "--time-limit", str(time_limit)]
    wait = time_limit * 3 + 600
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=wait, check=False)
    except subprocess.TimeoutExpired:  # the solver ignored its time limit: where it was, from its log
        log = mdir / f"{solver}_t{threads}.log"
        tail = log.read_text(errors="replace").strip().splitlines()[-8:] if log.exists() else []
        return {"status": "no response", "seconds": wait, "log_tail": tail}
    if proc.returncode != 0:
        return {"status": "error", "error": proc.stderr.strip().splitlines()[-1:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------- experiments

def instances_experiment(args) -> None:
    """--run-instances: K copies of a model on HiGHS, one after the other and at once (one HiGHS thread each)."""
    from concurrent.futures import ThreadPoolExecutor

    import highspy

    m = load_model(Path(args.model))
    lp = highs_model(m)

    def one(_=None):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("mip_rel_gap", args.gap)
        h.setOptionValue("time_limit", args.time_limit)
        h.passModel(lp)
        t0 = time.perf_counter()
        h.run()
        return time.perf_counter() - t0, h.getInfo().objective_function_value

    k = args.instances
    t0 = time.perf_counter()
    seq = [one() for _ in range(k)]
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(k) as ex:
        par = list(ex.map(one, range(k)))
    t_par = time.perf_counter() - t0
    print(json.dumps({"instances": k, "sequential_seconds": t_seq, "concurrent_seconds": t_par,
                      "speedup": t_seq / t_par, "single_seconds": [s for s, _ in seq],
                      "concurrent_each_seconds": [s for s, _ in par],
                      "same_objective": len({round(o, 9) for _, o in seq + par}) == 1}))


# ----------------------------------------------------------------------------- main

def sizes_for(family: str, sizes: list[int]) -> list[int]:
    return [n for n in sizes if n <= MAX_SIZE[family]]


def _secs(t) -> str:
    return "-" if t is None else f"{t:.1f}"


def fmt(r: dict) -> str:
    """Solve time (with the gap left at the time limit) and the seconds to a 10% and a 1% gap."""
    if r.get("status") == "error":
        return "error"
    t = r.get("seconds", math.nan)
    s = f"{t:.1f} s"
    reach = r.get("seconds_to_gap")
    if reach:
        s = f"{s} [10%: {_secs(reach['0.1'])}, 1%: {_secs(reach['0.01'])}]"
    if str(r.get("status")) not in ("Optimal", "OPTIMAL", "0") and r.get("gap") is not None:
        s += f" (gap {100 * r['gap']:.2f}%)" if math.isfinite(r["gap"]) else " (no solution)"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--sizes", default="1e3,1e4,1e5,1e6")
    ap.add_argument("--solvers", default="highs,gurobi")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--work", default=str(ROOT / "build" / "benchmark_optimize"))
    ap.add_argument("--cpus", default=None, help="processors to run on, as 0-11 or 0,2,4")
    ap.add_argument("--experiments", default="", help="threads,instances")
    ap.add_argument("--experiment-model", default="boundary_100")
    ap.add_argument("--experiment-gap", type=float, default=0.1, help="relative gap of the experiments")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--redo", action="store_true", help="solve again the runs already in results.json")
    ap.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-instances", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--solver", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.run_one:
        return run_one(args)
    if args.run_instances:
        return instances_experiment(args)

    import landschaft as ls

    if args.cpus:
        import psutil

        cpus = [int(c) for part in args.cpus.split(",") for c in (
            range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]
        psutil.Process().cpu_affinity(cpus)  # R and the solver processes inherit it
    # all the processors given (the default leaves one free: Python on Windows does not see the affinity)
    threads = args.threads or (len(cpus) if args.cpus else ls.default_threads())
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    results_file = work / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}
    results.setdefault("runs", {})
    results["settings"] = {"threads": threads, "gap": args.gap, "time_limit": args.time_limit, "cpus": args.cpus}

    def save():
        results_file.write_text(json.dumps(results, indent=1), encoding="utf-8")

    sizes = [int(float(s)) for s in args.sizes.split(",")]
    solvers = args.solvers.split(",")
    for family in filter(None, args.families.split(",")):
        for n in sizes_for(family, sizes):
            side = round(math.sqrt(n))
            key = f"{family}_{side}"
            print(f"{key}: building", flush=True)
            mdir = build_model(work, family, side)
            info = json.loads((mdir / "result.json").read_text(encoding="utf-8"))
            run = results["runs"].setdefault(key, {})
            run.update(family=family, units=side * side, rows=info["nrow"], cols=info["ncol"], nnz=info["nnz"],
                       compile_seconds=info["compile_seconds"])
            for solver in solvers:
                if solver in run and not args.redo:
                    continue
                print(f"{key}: {solver}", flush=True)
                if solver == "prioritizr":
                    out = build_model(work, family, side, {"threads": threads, "gap": args.gap,
                                                           "time_limit": args.time_limit})
                    r = json.loads((out / "result.json").read_text(encoding="utf-8"))["solve"]
                    r["status"] = str(r["status"])
                else:
                    r = solve_in_process(mdir, solver, threads, args.gap, args.time_limit)
                run[solver] = r
                save()

    exps = [e for e in args.experiments.split(",") if e]
    if "threads" in exps:
        mdir = work / "models" / args.experiment_model
        res = results.setdefault("threads", {})
        for t in sorted({1, 2, 4, threads}):
            for solver in ("highs", "gurobi"):
                print(f"threads {args.experiment_model}: {solver} on {t}", flush=True)
                res[f"{solver}_{t}"] = solve_in_process(mdir, solver, t, args.experiment_gap, args.time_limit)
                save()
    if "instances" in exps:
        mdir = work / "models" / args.experiment_model
        cmd = [sys.executable, __file__, "--run-instances", "--model", str(mdir), "--instances", str(args.instances),
               "--gap", str(args.experiment_gap), "--time-limit", str(args.time_limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        results["instances"] = json.loads(proc.stdout.strip().splitlines()[-1])
        save()

    # Markdown table
    print("\n| model | units | rows x cols (nnz) | prioritizr compile | " + " | ".join(solvers) + " |")
    print("|---|---:|---|---:|" + "---:|" * len(solvers))
    for key, run in results["runs"].items():
        cells = " | ".join(fmt(run.get(s, {"status": "error"})) for s in solvers)
        print(f"| {key} | {run['units']:,} | {run['rows']:,} x {run['cols']:,} ({run['nnz']:,}) | "
              f"{run['compile_seconds']:.1f} s | {cells} |")
    if "threads" in results:
        print("\nthreads:", {k: fmt(v) for k, v in results["threads"].items()})
    if "instances" in results:
        print("instances:", {k: v for k, v in results["instances"].items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
