# SPDX-License-Identifier: GPL-3.0-or-later
"""The operations a task may run: the only code a helper executes for the group.

Each operation gets its JSON parameters, the folder holding its input files (by the
names the task gave them), the folder to write its outputs into and its threads; it
returns a small JSON-able dict. Parameters naming files are checked to stay inside the
input folder, and tools are taken from fixed lists, never from the parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ._tasks import check_name

OPS: dict[str, Callable] = {}


def register(name: str):
    def wrap(fn):
        OPS[name] = fn
        return fn

    return wrap


def run(op: str, params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """Run a registered operation (in a child process)."""
    from .. import _threads

    fn = OPS.get(op)
    if fn is None:
        raise ValueError(f"unknown operation {op!r}")
    if not isinstance(params, dict):
        raise TypeError("the parameters of an operation are a JSON object")
    # the task's threads are the process default while it runs (its own threads too), and
    # the previous default comes back after it (a task can run in the caller's process)
    previous = _threads._default
    _threads.set_default_threads(max(1, int(threads)))
    try:
        with _threads.limit(threads):
            return fn(params, Path(inputs), Path(out), int(threads)) or {}
    finally:
        _threads.set_default_threads(previous)


def input_file(inputs: Path, name) -> Path:
    """An input file of the task: inside its input folder, existing."""
    check_name(name)
    root = Path(inputs).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"no input file {name!r}")
    return path


def _output_name(name) -> str:
    text = str(name)
    if not text or Path(text).name != text or text in (".", ".."):
        raise ValueError(f"not an output file name: {name!r}")
    return text


# ----------------------------------------------------------------------------- ls.batch

@register("batch")
def _batch(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """One raster through one of ``BATCH_TOOLS`` (``ls.batch``)."""
    from ..batch import BATCH_TOOLS, _resolve_tool, _thread_keyword, _tool_outputs

    tool = str(params["tool"])
    if tool not in BATCH_TOOLS or tool == "mspa":
        raise ValueError(f"{tool!r} is not a tool of ls.batch")
    name, fn = _resolve_tool(tool)
    src = input_file(inputs, params["input"])
    call = dict(params.get("params") or {})
    kw = _thread_keyword(fn)
    if kw:
        call[kw] = threads
    base = out / f"{_output_name(params['stem'])}_{name}"
    outputs, values = _tool_outputs(name, fn, src, base, params.get("band"), call, str(params["table_format"]))
    return {"outputs": [p.name for p in outputs], "values": values}


@register("mspa")
def _mspa(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """MSPA of one raster (``ls.mspa_batch``)."""
    from ..io import _load_map, write_raster
    from ..mspa import mspa

    src = input_file(inputs, params["input"])
    data = _load_map(src, band=params.get("band"))
    result = mspa(data, connectivity=int(params["connectivity"]), edge_width=int(params["edge_width"]),
                  transition=bool(params["transition"]), internal=bool(params["internal"]), n_threads=threads,
                  validate=False)
    write_raster(result, out / _output_name(params["output"]))
    return {}


# ----------------------------------------------------------------------------- ls.simulate

def _model(params: dict, inputs: Path):
    from ..simulate import Model

    return Model.load(input_file(inputs, params["model"]))


@register("simulate.run")
def _simulate_run(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """One run of a saved model (``ls.simulate.scenarios``): its maps and step table."""
    import numpy as np

    model, ins = _model(params, inputs)
    sim = model.run(ins["start"], ins.get("drivers"), seed=int(params["seed"]), n_threads=threads)
    np.save(out / "maps.npy", np.asarray(sim.maps.values), allow_pickle=False)
    sim.steps.to_parquet(out / "steps.parquet")
    return {}


@register("simulate.losses")
def _simulate_losses(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """The losses of a saved model against an observed map, one per seed (``ls.simulate.calibrate``)."""
    import numpy as np

    from ..simulate._calibrate import _losses, _targets

    model, ins = _model(params, inputs)
    start = np.load(input_file(inputs, params["start"]), allow_pickle=False)
    observed = np.load(input_file(inputs, params["observed"]), allow_pickle=False)
    targets = _targets(params["targets"])
    if any(fn is not None and name != "lsm" for name, (fn, _) in targets.items()):
        raise ValueError("only the built-in targets travel to helpers")
    nodata = params.get("nodata")
    parts = []
    for seed in params["seeds"]:
        sim = model.run(ins["start"], ins.get("drivers"), seed=int(seed), n_threads=threads)
        parts.append(_losses(targets, start, observed, sim.maps.isel(step=-1), nodata))
    return {"parts": parts}


# ----------------------------------------------------------------------------- registered tools

def _call(tool: str, args, kwargs: dict, threads: int):
    import inspect

    from ._registry import check_strings, resolve

    fn = resolve(tool)
    check_strings(list(args), "args")
    check_strings(kwargs, "kwargs")
    try:
        if "n_threads" in inspect.signature(fn).parameters and "n_threads" not in kwargs:
            kwargs = {**kwargs, "n_threads": threads}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kwargs)


def _spec(inputs: Path, name):
    import json

    folder = Path(inputs) / _output_name(name)
    return json.loads((folder / "value.json").read_text()), folder


def _arguments(spec, folder, params=None):
    from . import _pack

    value = _pack.decode(spec, folder, params)
    if not (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (list, tuple))
            and isinstance(value[1], dict)):
        raise ValueError("bad arguments")
    return list(value[0]), dict(value[1])


@register("call")
def _call_op(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """A registered tool on its arguments (``ls.parallel.map``): the result, stored."""
    from . import _pack

    args, kwargs = _arguments(*_spec(inputs, params["args"]))
    _pack.write(_call(str(params["tool"]), args, kwargs, threads), out / "result")
    return {}


@register("call.draws")
def _draws_op(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """A registered tool on a chunk of Monte Carlo draws: each draw's values, a row each."""
    import numpy as np

    from ..uncertainty import _values

    draws = np.load(input_file(inputs, params["draws"]), allow_pickle=False)
    spec, folder = _spec(inputs, params["args"])
    rows = []
    for d in draws:
        args, kwargs = _arguments(spec, folder, params=d)
        rows.append(np.asarray(_values(_call(str(params["tool"]), args, kwargs, threads)), dtype=np.float64).ravel())
    np.save(out / "values.npy", np.stack(rows), allow_pickle=False)
    return {}


@register("tile")
def _tile_op(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """A registered tool on one tile with its halo: the result cut to the tile's core."""
    from . import _pack
    from ..parallel import _crop

    tile = _pack.read(Path(inputs) / _output_name(params["tile"]))
    args, kwargs = _arguments(*_spec(inputs, params["args"]))
    r0, r1, c0, c1 = (int(v) for v in params["core"])
    result = _call(str(params["tool"]), [tile, *args], kwargs, threads)
    _pack.write(_crop(result, slice(r0, r1), slice(c0, c1), tuple(tile.shape[-2:])), out / "result")
    return {}


# ----------------------------------------------------------------------------- ls.sdm

def build_estimator(spec):
    """A scikit-learn estimator (or one of ls.sdm) from its class and parameters."""
    import importlib

    if isinstance(spec, list):
        return [build_estimator(x) for x in spec]
    if isinstance(spec, dict) and "$tuple" in spec:
        return tuple(build_estimator(x) for x in spec["$tuple"])
    if isinstance(spec, dict) and "$estimator" in spec:
        module, _, qualname = str(spec["$estimator"]).partition(":")
        if not (module.startswith("sklearn.") or module.startswith("landschaft.sdm")):
            raise ValueError(f"refusing to build {spec['$estimator']}: only scikit-learn and ls.sdm estimators travel")
        cls = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        if not isinstance(cls, type) or not hasattr(cls, "get_params"):
            raise ValueError(f"{spec['$estimator']} is not an estimator")
        return cls(**{k: build_estimator(v) for k, v in spec["params"].items()})
    if isinstance(spec, dict) and "$dict" in spec:
        return {k: build_estimator(v) for k, v in spec["$dict"].items()}
    if isinstance(spec, dict) and "$float" in spec:
        return float(spec["$float"])
    return spec


@register("sdm.fold")
def _sdm_fold(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """One model on one cross-validation fold (``ls.sdm.fit``): the fold's scores and test predictions."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from ..sdm._models import _fit_one, _fold_metrics, _suitability

    X = pd.read_parquet(input_file(inputs, params["X"]))
    y = np.load(input_file(inputs, params["y"]), allow_pickle=False)
    w = np.load(input_file(inputs, params["w"]), allow_pickle=False) if params.get("w") else None
    train = np.load(input_file(inputs, params["train"]), allow_pickle=False)
    test = np.load(input_file(inputs, params["test"]), allow_pickle=False)
    model = build_estimator(params["model"])
    m = _fit_one(model, X.iloc[train], y[train], None if w is None else w[train])
    pred = np.full(len(y), np.nan)
    pred[train] = _suitability(m, X.iloc[train])
    pred[test] = _suitability(m, X.iloc[test])
    scores = {}
    for k, v in _fold_metrics(pred, SimpleNamespace(y=y), train, test).items():
        v = float(v)
        scores[k] = repr(v) if not np.isfinite(v) else v
    test_pred = [float(v) if np.isfinite(v) else None for v in pred[test]]
    return {"scores": scores, "test": test_pred, "versions": versions()}


def versions() -> dict:
    """The versions of the libraries an estimator's results depend on."""
    out = {}
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    import numpy

    out["numpy"] = numpy.__version__
    return out


# ----------------------------------------------------------------------------- ls.prioritize

def _prio_model(inputs: Path, name):
    from . import _pack
    from ..prioritize._milp_remote import model_from

    return model_from(_pack.read(Path(inputs) / _output_name(name)))


@register("prioritize.solve")
def _prioritize_solve(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """A compiled prioritization model on a solver: its result (solution, objective, status)."""
    from . import _pack
    from ..prioritize._milp_solve import solve_model

    solver = str(params["solver"])
    if solver not in ("highs", "gurobi", "anneal"):
        raise ValueError(f"unknown solver {solver!r}")
    from ..prioritize._milp_remote import options_from

    options = options_from(params.get("options"))
    options["n_threads"] = threads
    r = solve_model(_prio_model(inputs, params["model"]), solver, options, None, n_units=params.get("n_units"),
                    engines=bool(params.get("engines", True)), zone_units=params.get("zone_units"))
    _pack.write(r, out / "result")
    return {}


@register("prioritize.replacement")
def _prioritize_replacement(params: dict, inputs: Path, out: Path, threads: int) -> dict:
    """The objectives of a model solved without each of some selected units (replacement importance)."""
    import math

    import numpy as np

    from ..prioritize._milp import OptimizationProblem
    from ..prioritize._milp_remote import options_from
    from ..prioritize._milp_solve import Prepared, solve_model

    m = _prio_model(inputs, params["model"])
    lb = np.load(input_file(inputs, params["lb"]), allow_pickle=False)
    ub = np.load(input_file(inputs, params["ub"]), allow_pickle=False)
    solver = str(params["solver"])
    if solver not in ("highs", "gurobi"):
        raise ValueError(f"unknown solver {solver!r}")
    one = {**options_from(params.get("options")), "n_threads": 1}
    n = int(params["n_units"])
    prep = Prepared(m, n) if params.get("prepared") else None
    objs = []
    for i in params["units"]:
        lo, hi = lb.copy(), ub.copy()
        lo[int(i)] = hi[int(i)] = 0.0
        if prep is not None:
            r = prep.solve(lo, hi, one)
        else:
            mm = OptimizationProblem(A=m.A, obj=m.obj, lb=lo, ub=hi, rhs=m.rhs, sense=m.sense, vtype=m.vtype,
                                     modelsense=m.modelsense, row_ids=m.row_ids, col_ids=m.col_ids)
            r = solve_model(mm, solver, one, None, n_units=n)
        objs.append(float(r["objective"]) if r["has_solution"] else math.inf)
    np.save(out / "objectives.npy", np.asarray(objs, dtype=np.float64), allow_pickle=False)
    return {}
