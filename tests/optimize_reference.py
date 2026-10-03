# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading the references of prioritizr and prioriactions (tools/make_reference_optimize.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import optimize_cases
import pytest

REF = Path(__file__).parent / "data" / "reference" / "optimize"


@cache
def load(group: str) -> tuple[dict, dict]:
    npz, js = REF / f"{group}.npz", REF / f"{group}.json"
    if not npz.exists():
        pytest.skip(f"no reference for {group} (run tools/make_reference_optimize.py {group})")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))


def case_ids(group: str) -> list[str]:
    return optimize_cases.case_ids(group)


def result(group: str, case: str) -> dict:
    """The reference result of a case: key -> float64 array (matrices in their shape) or list of str."""
    arrays, info = load(group)
    prefix = f"{case}/"
    out = {}
    for key, meta in info["arrays"].items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if meta["kind"] == "text":
            out[name] = list(meta["values"])
        else:
            v = arrays[key]
            if meta["kind"] in ("matrix", "raster") and "dim" in meta:
                v = v.reshape(meta["dim"])
            out[name] = v
    if not out:
        pytest.skip(f"case {case} missing from the {group} reference")
    return out


@dataclass
class Model:
    """A compiled model: min/max obj @ x subject to A x (sense) rhs, lb <= x <= ub, x of type vtype."""

    A_i: np.ndarray  # int64, 0-based, in the reference's order
    A_j: np.ndarray
    A_x: np.ndarray
    obj: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    rhs: np.ndarray
    sense: list[str]  # ">=", "<=", "="
    vtype: list[str]  # "B", "C", "S" (semi-continuous), "I"
    modelsense: str  # "min" or "max"
    row_ids: list[str] | None = None
    col_ids: list[str] | None = None
    genconpow: dict | None = None  # y = x^a: xvar, yvar (0-based), a

    @property
    def shape(self) -> tuple[int, int]:
        return self.rhs.size, self.obj.size

    def csr(self):
        """The constraint matrix as a canonical scipy CSR (sorted indices, no duplicates)."""
        import scipy.sparse as sp

        a = sp.csr_matrix((self.A_x, (self.A_i, self.A_j)), shape=self.shape)
        a.sum_duplicates()
        a.sort_indices()
        return a


def model(group: str, case: str) -> Model:
    r = result(group, case)
    g = None
    if "model/genconpow/xvar" in r:
        g = {"xvar": r["model/genconpow/xvar"].astype(np.int64) - 1,
             "yvar": r["model/genconpow/yvar"].astype(np.int64) - 1, "a": r["model/genconpow/a"]}
    return Model(
        A_i=r["model/A_i"].astype(np.int64), A_j=r["model/A_j"].astype(np.int64), A_x=r["model/A_x"],
        obj=r["model/obj"], lb=r["model/lb"], ub=r["model/ub"], rhs=r["model/rhs"],
        sense=r["model/sense"], vtype=r["model/vtype"], modelsense=r["model/modelsense"][0],
        row_ids=r.get("model/row_ids"), col_ids=r.get("model/col_ids"), genconpow=g)


def assert_same(ours, ref, key: str = "", rtol: float = 0.0, atol: float = 0.0) -> None:
    """Equal values (NaN = NA matches NaN); bitwise unless a tolerance is given."""
    if isinstance(ref, list):
        assert [None if o is None else str(o) for o in ours] == ref, key
        return
    a = np.asarray(ours, dtype=float).reshape(np.shape(ref))
    b = np.asarray(ref, dtype=float)
    if rtol == 0 and atol == 0:
        same = (a == b) | (np.isnan(a) & np.isnan(b))
    else:
        same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not same.all():
        bad = np.flatnonzero(~same.ravel())
        raise AssertionError(f"{key}: {bad.size} of {same.size} differ; first at {bad[:5]}: "
                             f"ours {a.ravel()[bad[:5]]} vs reference {b.ravel()[bad[:5]]}")


# ----------------------------------------------------------------------------- solving a stored model
# Development checks only: highspy and gurobipy are not dependencies of landschaft.

def _row_bounds(m: Model) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(m.rhs.size, -np.inf)
    hi = np.full(m.rhs.size, np.inf)
    sense = np.asarray(m.sense)
    ge = (sense == ">=") | np.isin(sense, ("=", "=="))
    le = (sense == "<=") | np.isin(sense, ("=", "=="))
    lo[ge] = m.rhs[ge]
    hi[le] = m.rhs[le]
    return lo, hi


def solve_highs(m: Model, gap: float = 0.0, threads: int = 1, time_limit: float = 300.0):
    """Objective and solution of the model on HiGHS (highspy), at the given relative gap."""
    import highspy

    if m.genconpow is not None:
        raise ValueError("HiGHS has no general power constraints")
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
    if integer.any():
        lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                           for i in integer]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("time_limit", time_limit)
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS: {h.modelStatusToString(status)}")
    return h.getInfo().objective_function_value, np.asarray(h.getSolution().col_value)


def solve_gurobi(m: Model, gap: float = 0.0, threads: int = 1, time_limit: float = 300.0):
    """Objective and solution of the model on Gurobi (gurobipy), at the given relative gap."""
    import gurobipy as gp

    a = m.csr()
    vtype = np.asarray(m.vtype)
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as g:
        x = g.addMVar(m.obj.size, lb=m.lb, ub=np.where(np.isinf(m.ub), gp.GRB.INFINITY, m.ub),
                      vtype=[{"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER, "S": gp.GRB.SEMICONT}.get(t, gp.GRB.CONTINUOUS)
                             for t in vtype])
        sense = np.asarray([{"==": "="}.get(s, s)[0] for s in m.sense])
        g.addMConstr(a, x, sense, m.rhs)
        g.setObjective(m.obj @ x, gp.GRB.MAXIMIZE if m.modelsense == "max" else gp.GRB.MINIMIZE)
        if m.genconpow is not None:
            xs = x.tolist()
            for xi, yi, p in zip(m.genconpow["xvar"], m.genconpow["yvar"], m.genconpow["a"]):
                g.addGenConstrPow(xs[xi], xs[yi], p)
        g.Params.MIPGap = gap
        g.Params.Threads = threads
        g.Params.TimeLimit = time_limit
        g.optimize()
        if g.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi: status {g.Status}")
        return g.ObjVal, x.X.copy()
