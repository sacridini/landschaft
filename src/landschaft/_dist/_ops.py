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
    _threads.set_default_threads(max(1, int(threads)))
    with _threads.limit(threads):
        return fn(params, Path(inputs), Path(out), int(threads)) or {}


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
