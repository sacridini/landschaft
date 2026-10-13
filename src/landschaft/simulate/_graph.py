# SPDX-License-Identifier: GPL-3.0-or-later
"""Models as Terraflux graphs: ``Model.save``, ``Model.load`` and running them with Terraflux.

`Terraflux <https://github.com/sacridini/terraflux>`_ runs land change models
as graphs of operators, on one machine or many, with the same C++ kernels as
``ls.simulate``. ``Model.save`` writes a fitted model as such a graph (JSON):
the fitted pieces as they are (weights of evidence, regressions, Markov
chains, demands, patch settings, rules), the data it runs on as GeoTIFF files
next to it; ``terraflux run`` then gives the same maps as ``Model.run`` with
the same seed. ``Model.load`` reads such a graph back, and graphs made by
``terraflux new`` too (fitting their pieces as Terraflux would).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

from ._change import _class_value

FORMAT = "terraflux/1"

__all__ = ["load_model", "save_model", "terraflux"]


# ----------------------------------------------------------------------------- writing

def _clean(v):
    """JSON values: NaN as null, numpy numbers as Python ones."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    return v


def _num(v):
    """A class as JSON: an int when whole."""
    return _class_value(float(v))


class _Writer:
    """Writes the data a graph reads next to it (float64 GeoTIFF, the values as the model reads them)."""

    def __init__(self, path: Path):
        self.path = path
        self.dir = path.with_name(path.stem + "_data")
        self.inputs: dict = {}

    def _write(self, name: str, layers: list, names: list | None, template):
        from ..io import write_raster

        self.dir.mkdir(parents=True, exist_ok=True)
        data = np.stack([np.asarray(x, dtype=np.float64) for x in layers])
        f = self.dir / f"{name}.tif"
        if template is not None:
            da = xr.DataArray(data, dims=("band", *template.dims),
                              coords={"band": np.arange(1, data.shape[0] + 1), **{k: template.coords[k] for k in template.dims}})
            try:
                da = da.rio.write_crs(template.rio.crs) if template.rio.crs is not None else da
                da = da.rio.write_transform(template.rio.transform())
            except Exception:  # noqa: BLE001, S110  (no georeference)
                pass
        else:
            da = xr.DataArray(data, dims=("band", "y", "x"))
        write_raster(da, f, dtype="float64", band_names=names, colormap=None, compress="deflate")
        return f"{self.dir.name}/{f.name}"

    def map(self, name: str, x, nodata=None, note=None) -> str:
        from ._maps import read_map

        if isinstance(x, (str, os.PathLike)):
            self.inputs[name] = {"kind": "map", "path": str(x), **({"note": note} if note else {})}
            return f"${name}"
        m = read_map(x, nodata)
        self.inputs[name] = {"kind": "map", "path": self._write(name, [m.as_float()], None, m.template),
                             **({"note": note} if note else {})}
        return f"${name}"

    def stack(self, name: str, maps, note=None) -> str:
        from ._maps import read_map

        if isinstance(maps, (str, os.PathLike)):
            self.inputs[name] = {"kind": "stack", "path": str(maps), **({"note": note} if note else {})}
            return f"${name}"

        names = [str(k) for k in maps]
        ms = [read_map(v) for v in maps.values()]
        self.inputs[name] = {"kind": "stack", "path": self._write(name, [m.as_float() for m in ms], names, ms[0].template),
                             **({"note": note} if note else {})}
        return f"${name}"

    def value(self, name: str, v, note=None) -> str:
        self.inputs[name] = {"kind": "value", "value": _clean(v), **({"note": note} if note else {})}
        return f"${name}"


def _node(id_, op, inputs=None, params=None, note=None, body=None) -> dict:
    n = {"id": id_, "op": op}
    if inputs:
        n["inputs"] = inputs
    if params:
        n["params"] = _clean(params)
    if note:
        n["note"] = note
    if body:
        n["body"] = body
    return n


def _woe_json(w) -> dict:
    """WeightsOfEvidence as Terraflux's woe.weights."""
    w._check()
    entries = []
    for f, t in w.transitions:
        for name, spec in w._plan[(f, t)].items():
            lay = w._layouts[(f, t, name)]
            inc = spec.increment if not isinstance(spec.increment, str) else None
            entries.append({
                "from": _num(f), "to": _num(t), "variable": name,
                "spec": {"categorical": bool(spec.categorical), "increment": inc, "min_delta": spec.min_delta,
                         "max_delta": spec.max_delta, "tolerance": spec.tolerance,
                         "breaks": None if spec.breaks is None else list(spec.breaks)},
                "layout": {"categorical": bool(lay.categorical), "classes": list(lay.classes), "start": lay.start,
                           "increment": lay.increment, "n": int(lay.n)},
                "limits": [int(x) for x in w._ranges[(f, t, name)]],
                "weights": list(w._weights[(f, t, name)]),
            })
    rows = [{"from": _num(r[0]), "to": _num(r[1]), "variable": r[2], "lower": r[3], "upper": r[4], "possible": int(r[5]),
             "executed": int(r[6]), "weight": r[7], "contrast": r[8], "significant": bool(r[9])} for r in w._rows]
    return {"transitions": [[_num(f), _num(t)] for f, t in w.transitions], "entries": entries, "rows": rows,
            "fix_abnormal": bool(w.fix_abnormal)}


def _glm_json(lr) -> list:
    lr._check()
    return [{"class": _num(k), "drivers": list(f["drivers"]), "coefficients": list(f["coefficients"]),
             "pivot": [int(x) for x in f["pivot"]], "rank": int(f["rank"]), "iterations": int(f["iterations"]),
             "converged": bool(f["converged"]), "deviance": f["deviance"], "null_deviance": f["null_deviance"],
             "cells": int(f["n"])} for k, f in lr._fits.items()]


def _drivers_of(drivers) -> dict:
    from ._woe import _drivers

    return {k: m for k, m in _drivers(drivers).items()}


def _dynamic_nodes(model, drivers_ref: str) -> tuple[list, str]:
    """The nodes of the dynamic variables in a loop: each computed from @landscape and set in the drivers."""
    from ._dynamic import distance_to, share_of

    nodes, ref = [], drivers_ref
    for k, (name, piece) in enumerate((model.dynamic or {}).items()):
        if isinstance(piece, distance_to):
            if piece.cost is not None:
                raise ValueError(f"the dynamic variable {name!r}: Terraflux has no cost distances yet")
            nodes.append(_node(f"dynamic_{k}", "dynamic.distance_to", {"landscape": "@landscape"},
                               {"classes": [_num(c) for c in piece.classes], "unit": piece.unit},
                               f"The dynamic variable {name!r}: the distance to classes {piece.classes}."))
            port = "distance"
        elif isinstance(piece, share_of):
            nodes.append(_node(f"dynamic_{k}", "dynamic.share_of", {"landscape": "@landscape"},
                               {"classes": [_num(c) for c in piece.classes], "window": piece.window},
                               f"The dynamic variable {name!r}: the share of classes {piece.classes} around."))
            port = "share"
        else:
            raise TypeError(f"the dynamic variable {name!r} is a function Terraflux does not know: use "
                             "distance_to or share_of")
        nodes.append(_node(f"drivers_{k}", "stack.set", {"stack": ref, "map": f"dynamic_{k}.{port}"}, {"name": str(name)}))
        ref = f"drivers_{k}.stack"
    return nodes, ref


def _potential_node(model, w: _Writer, drivers, landscape, kind: str):
    """The node(s) of the potential: fitted pieces as given, rasters as inputs. kind: 'class' or 'transition'."""
    from ._estimator import RasterPotential
    from ._glm import LogisticRegression
    from ._woe import WeightsOfEvidence

    p = model.potential
    if isinstance(p, WeightsOfEvidence):
        return [_node("potential", "woe.weights", params={"weights": _woe_json(p)},
                      note="Weights of evidence fitted with landschaft.")], "woe"
    if isinstance(p, LogisticRegression):
        return [_node("potential", "glm.coefficients", params={"fits": _glm_json(p)},
                      note="Logistic regressions fitted with landschaft.")], "glm"
    if isinstance(p, RasterPotential):
        names = [f"{_num(n[0])}_to_{_num(n[1])}" if isinstance(n, tuple) else str(_num(n)) for n in p._names]
        ref = w.stack("potential", dict(zip(names, p._layers)), note="the potential, one layer per " + kind)
        return [_node("potential", "stack.classes", {"stack": ref})], "raster"
    if isinstance(p, dict):
        layers = {f"{_num(f)}_to_{_num(t)}": v for (f, t), v in p.items()}
        ref = w.stack("potential", layers, note="the probability of each transition")
        return [_node("potential", "stack.classes", {"stack": ref})], "raster"
    if hasattr(p, "predict" if kind == "class" else "probability"):
        layers = _predicted(model, w, drivers, landscape, kind)
        ref = w.stack("potential", layers, note=f"the predictions of the {type(p).__name__} on the start, one "
                                                f"layer per {kind}")
        return [_node("potential", "stack.classes", {"stack": ref},
                      note=f"The {type(p).__name__} fitted with landschaft, as its predictions.")], "raster"
    raise TypeError(f"Terraflux cannot run a potential {type(p).__name__}: use WeightsOfEvidence, "
                     "LogisticRegression, RasterPotential or a piece with predict()")


def _read_given(x, w: _Writer, what: str):
    """A map or stack given to save: as is, or read from its path (relative to the model file)."""
    import landschaft as ls

    if not isinstance(x, (str, os.PathLike)):
        return x
    p = Path(x)
    da = ls.load_raster(p if p.is_absolute() else w.path.parent / p)
    if what == "map" and da.ndim == 3:
        da = da[0]
    return da


def _predicted(model, w: _Writer, drivers, landscape, kind: str) -> dict:
    """A potential without a form Terraflux knows (an estimator...), as its predictions on the start.

    The same layers ``Model.run`` reads from it when the drivers do not change: per class (``predict``) or
    per transition (``probability``, as the Expander and Patcher read it).
    """
    p = model.potential
    if model.dynamic:
        raise ValueError(f"Terraflux cannot predict with a {type(p).__name__} at every step: save it without "
                         "dynamic variables (its predictions are written as maps), or use a LogisticRegression "
                         "or WeightsOfEvidence")
    land = _read_given(landscape, w, "map")
    drv = None if drivers is None else _read_given(drivers, w, "stack")
    if kind == "class":
        out = p.predict(drv, landscape=land) if drv is not None else p.predict(landscape=land)
        names = [_class_value(c) for c in out[out.dims[0]].values]
        return {str(_num(c)): out[k] for k, c in enumerate(names)}
    out = p.probability(land, drv) if drv is not None else p.probability(land)
    layers = {}
    for k, name in enumerate(str(v) for v in out[out.dims[0]].values):
        if not name.startswith("probability_") or "_to_" not in name:
            raise ValueError(f"the {type(p).__name__} gives a layer {name!r}, not probability_<from>_to_<to>")
        f, t = name[len("probability_"):].split("_to_")
        layers[f"{_num(float(f))}_to_{_num(float(t))}"] = out[k]
    return layers


def _demand_node(demand, classes_hint=None) -> dict:
    from ._demand import AreaDemand
    from ._markov import Markov

    if isinstance(demand, AreaDemand):
        t = demand.table
        times = [float(x) for x in t.index] if np.issubdtype(np.asarray(t.index).dtype, np.number) else None
        return _node("demand", "demand.table", params={"table": {
            "classes": [_num(c) for c in t.columns], "rows": t.to_numpy(dtype=np.float64), "times": times}},
            note="The demand given (cells of each class at each step).")
    if isinstance(demand, Markov):
        return _node("demand", "markov.matrix", params={"classes": [_num(c) for c in demand.classes],
                                                       "single": demand._single, "multi": demand.multi.to_numpy(),
                                                       "steps": demand.steps},
                     note="The Markov chain fitted with landschaft (its matrix per step included).")
    raise TypeError(f"Terraflux cannot run a demand {type(demand).__name__} yet (AreaDemand or Markov)")


def to_graph(model, path, landscape, drivers=None, *, seed=None, nodata=None, name=None, description=None) -> dict:
    """The graph of a fitted model (see :meth:`Model.save`)."""
    from ._allocation import CLUE, Ordered
    from ._patches import ExpanderPatcher
    from ._plus import CARS
    from ._regional import Regional

    path = Path(path)
    w = _Writer(path)
    if isinstance(model.demand, Regional):
        raise TypeError("Terraflux runs no regional demands yet")
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    start = w.map("start", landscape, nodata, "the map the simulation starts from")
    seed_ref = w.value("seed", int(seed), "the seed of the simulation")
    if drivers is None:
        drv_ref = None
    else:
        drv = drivers if isinstance(drivers, (str, os.PathLike)) else _drivers_of(drivers)
        drv_ref = w.stack("drivers", drv, "the explanatory variables")
    rules = model.rules
    nodes, outputs = [], {}
    alloc = model.allocation

    def loop_drivers():
        if not model.dynamic:
            return [], drv_ref
        if drv_ref is None:
            raise ValueError("dynamic variables need the drivers")
        return _dynamic_nodes(model, drv_ref)

    if isinstance(alloc, (CLUE, Ordered)):
        pot, kind = _potential_node(model, w, drivers, landscape, "class")
        nodes += pot
        nodes.append(_demand_node(model.demand))
        if kind == "woe":
            raise ValueError("CLUE and the ordered allocation take suitabilities of classes, not weights of evidence")
        dyn, dref = loop_drivers()
        if kind == "glm":
            predict = _node("suitability", "glm.predict", {"model": "potential.model", "drivers": dref,
                                                           "landscape": "@landscape" if dyn else start})
            suit = "suitability.suitability"
        else:
            predict, suit = None, "potential.stack"
        if predict is not None and not dyn:
            nodes.append(predict)
        nodes.append(_node("rng", "rng.r", {"seed": seed_ref}, note="R's generator after set.seed(seed)."))
        hist_in = {"landscape": start}
        if rules.history is not None:
            hist_in["history"] = w.map("history", rules.history, note="the steps each cell has had its class")
        nodes.append(_node("history", "lulcc.history", hist_in))
        params = {"allocation": ({"method": "clue", "jitter": alloc.jitter, "scale": alloc.scale,
                                  "max_iter": alloc.max_iter, "max_diff": alloc.max_diff, "mean_diff": alloc.mean_diff}
                                 if isinstance(alloc, CLUE) else
                                 {"method": "ordered", "order": [_num(c) for c in alloc.order],
                                  "stochastic": bool(alloc.stochastic)})}
        classes = list(model.demand.classes)
        if rules.elasticity is not None:
            params["elasticity"] = list(rules.elasticities(classes))
        m = rules.matrix(classes)
        if m is not None:
            params["allowed"] = m
        if rules.neighbourhood is not None:
            nb = rules.neighbourhood
            params["neighbourhood"] = {"rules": {str(_num(c)): float(v) for c, v in nb.rules.items()},
                                       "weights": ({str(_num(c)): nb.window(c) for c in nb.rules}
                                                   if nb.weights is not None else None)}
        step_in = {"landscape": "@landscape", "history": "@history", "suitability": suit, "demand": "demand.demand",
                   "step": "@step", "rng": "rng.rng"}
        if rules.restrictions is not None:
            step_in["restrictions"] = w.map("restrictions", rules.restrictions, note="restricted cells keep their class")
        body_nodes = dyn + ([predict] if (predict is not None and dyn) else [])
        body_nodes.append(_node("allocate", "lulcc.step", step_in, params))
        nodes.append(_node("simulation", "control.repeat",
                           {"times": "demand.steps", "landscape": start, "history": "history.history"},
                           body={"nodes": body_nodes,
                                 "carry": {"landscape": "allocate.landscape", "history": "allocate.history"},
                                 "collect": {"maps": "allocate.landscape", "info": "allocate.info"}}))
    elif isinstance(alloc, ExpanderPatcher):
        from ._model import Model  # noqa: F401

        pot, kind = _potential_node(model, w, drivers, landscape, "transition")
        nodes += pot
        nodes.append(_demand_node(model.demand))
        rates = alloc._rates(model.demand)
        trs = [(f, t) for f in rates.index for t in rates.columns if f != t and rates.loc[f, t] > 0]
        from ._maps import read_map
        from ._patches import _per_transition

        land = read_map(landscape, nodata) if not isinstance(landscape, (str, os.PathLike)) else None
        if alloc.size_unit != "cells" and land is None:
            raise ValueError("patch sizes in area units need the landscape as a map, not a path")
        share = dict(zip(trs, _per_transition(alloc.expansion, trs, "expansion share")))
        e_par = dict(zip(trs, _per_transition(alloc.expander, trs, "expander parameters")))
        p_par = dict(zip(trs, _per_transition(alloc.patcher, trs, "patcher parameters")))

        def cells(v):
            return v if alloc.size_unit == "cells" else alloc._cells(v, land)

        settings = [{"from": _num(f), "to": _num(t), "expansion": float(share[(f, t)]),
                     "expander": [cells(e_par[(f, t)][0]), cells(e_par[(f, t)][1]) * cells(1), e_par[(f, t)][2]],
                     "patcher": [cells(p_par[(f, t)][0]), cells(p_par[(f, t)][1]) * cells(1), p_par[(f, t)][2]]}
                    for f, t in trs]
        nodes.append(_node("patches", "patches.settings", params={"transitions": settings}))
        if model.demand.__class__.__name__ != "Markov":
            raise ValueError("Terraflux's Expander and Patcher take a Markov chain as demand")
        steps = alloc.steps if alloc.steps is not None else model.demand.steps
        dyn, dref = loop_drivers()
        if kind == "woe":
            prob = _node("probability", "woe.probability", {"model": "potential.model", "landscape": "@landscape",
                                                             "drivers": dref})
            prob_ref = "probability.probability"
            body = dyn + [prob]
        elif kind == "raster":
            if dyn:
                raise ValueError("dynamic variables need a potential computed from drivers, not given maps")
            prob_ref, body = "potential.stack", []
        else:
            raise ValueError("the Expander and Patcher take transition probabilities (weights of evidence or maps)")
        body.append(_node("allocate", "dinamica.step", {"landscape": "@landscape", "probability": prob_ref,
                                                         "rates": "demand.markov", "patches": "patches.patches",
                                                         "step": "@step", "seed": seed_ref},
                          {"prune_factor": alloc.prune_factor, "window": list(alloc.window)}))
        nodes.append(_node("simulation", "control.repeat", {"times": int(steps), "landscape": start},
                           body={"nodes": body, "carry": {"landscape": "allocate.landscape"},
                                 "collect": {"maps": "allocate.landscape", "info": "allocate.info"}}))
    elif isinstance(alloc, CARS):
        CARS._check_rules(rules)
        pot, kind = _potential_node(model, w, drivers, landscape, "class")
        nodes += pot
        d = _demand_node(model.demand)
        nodes.append(d)
        dem_in = {"start": start, ("markov" if d["op"] == "markov.matrix" else "demand"):
                  "demand.markov" if d["op"] == "markov.matrix" else "demand.demand"}
        nodes.append(_node("cars_demand", "plus.demand", dem_in, {"steps": alloc.steps} if alloc.steps else None))
        dyn, dref = loop_drivers()
        if kind == "glm":
            predict = _node("probability", "glm.predict", {"model": "potential.model", "drivers": dref,
                                                           "landscape": "@landscape" if dyn else start})
            prob_ref = "probability.suitability"
        elif kind == "raster":
            if dyn:
                raise ValueError("dynamic variables need a potential computed from drivers, not given maps")
            predict, prob_ref = None, "potential.stack"
        else:
            raise ValueError("CARS takes growth probabilities of classes")
        if predict is not None and not dyn:
            nodes.append(predict)
        classes = list(model.demand.classes)
        params = {"weights": {str(_num(c)): v for c, v in zip(classes, _per_class_of(alloc.weights, classes, 1.0))},
                  "seeds": {str(_num(c)): v for c, v in zip(classes, _per_class_of(alloc.seeds, classes, 0.1))},
                  "decay": alloc.decay, "step_size": alloc.step_size, "window": alloc.window, "max_iter": alloc.max_iter}
        m = rules.matrix(classes) if rules is not None else None
        if m is not None:
            params["allowed"] = m
        step_in = {"landscape": "@landscape", "probability": prob_ref, "demand": "cars_demand.demand",
                   "step": "@step", "seed": seed_ref}
        if rules is not None and rules.restrictions is not None:
            step_in["restrictions"] = w.map("restrictions", rules.restrictions, note="restricted cells keep their class")
        body = dyn + ([predict] if (predict is not None and dyn) else [])
        body.append(_node("allocate", "cars.step", step_in, params))
        nodes.append(_node("simulation", "control.repeat", {"times": "cars_demand.steps", "landscape": start},
                           body={"nodes": body, "carry": {"landscape": "allocate.landscape"},
                                 "collect": {"maps": "allocate.landscape", "info": "allocate.info"}}))
    else:
        raise TypeError(f"Terraflux cannot run the allocation {type(alloc).__name__} yet "
                         "(CLUE, Ordered, ExpanderPatcher or CARS)")
    outputs = {"maps": {"from": "simulation.maps"}, "steps": {"from": "simulation.info"}}
    g = {"format": FORMAT, "name": name or f"landschaft {type(alloc).__name__} model",
         "description": description or "Written by landschaft's Model.save.", "inputs": w.inputs, "nodes": nodes,
         "outputs": outputs}
    return _clean(g)


def _per_class_of(value, classes, default):
    from ._plus import _per_class

    return list(_per_class(value, classes, "values", default))


def save_model(model, path, landscape, drivers=None, *, seed=None, nodata=None, name=None, description=None) -> Path:
    """Write a fitted model as a Terraflux graph (JSON) and the data it runs on (GeoTIFF, next to it)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("the model is written as JSON: give a path ending in .json")
    g = to_graph(model, path, landscape, drivers, seed=seed, nodata=nodata, name=name, description=description)
    path.write_text(json.dumps(g, indent=2, allow_nan=False))
    return path


# ----------------------------------------------------------------------------- reading

def _read_graph(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() == ".toml":
        import tomllib

        g = tomllib.loads(text)
    else:
        g = json.loads(text)
    if g.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Terraflux model in format {FORMAT!r}")
    return g


def _floats(v):
    return np.array([math.nan if x is None else float(x) for x in v], dtype=np.float64)


def _woe_from(j):
    from ._woe import WeightsOfEvidence, _Layout, _Variable

    w = WeightsOfEvidence()
    w.transitions = [(_class_value(f), _class_value(t)) for f, t in j["transitions"]]
    w.fix_abnormal = bool(j.get("fix_abnormal", False))
    w._plan, w._layouts, w._ranges, w._weights = {}, {}, {}, {}
    for e in j["entries"]:
        key = (_class_value(e["from"]), _class_value(e["to"]))
        s, lay = e["spec"], e["layout"]
        spec = _Variable(e["variable"], categorical=s["categorical"],
                         increment="auto" if s["increment"] is None else s["increment"], min_delta=s["min_delta"],
                         max_delta=s["max_delta"], tolerance=s["tolerance"], breaks=s["breaks"])
        w._plan.setdefault(key, {})[e["variable"]] = spec
        w._layouts[(*key, e["variable"])] = _Layout(lay["categorical"], np.asarray(lay["classes"], dtype=np.float64),
                                                    lay["start"], lay["increment"], lay["n"])
        w._ranges[(*key, e["variable"])] = list(e["limits"])
        w._weights[(*key, e["variable"])] = list(e["weights"])
    w.variables = sorted({e["variable"] for e in j["entries"]})
    w._rows = [(_class_value(r["from"]), _class_value(r["to"]), r["variable"], r["lower"], r["upper"], r["possible"],
                r["executed"], r["weight"], r["contrast"], int(r["significant"])) for r in j["rows"]]
    return w


def _glm_from(fits):
    from ._glm import LogisticRegression

    lr = LogisticRegression()
    for f in fits:
        lr._fits[_class_value(f["class"])] = {
            "coefficients": _floats(f["coefficients"]), "pivot": np.asarray(f["pivot"], dtype=np.int32),
            "rank": int(f["rank"]), "drivers": list(f["drivers"]), "n": int(f.get("cells", 0)),
            "iterations": int(f.get("iterations", 0)), "converged": bool(f.get("converged", True)),
            "deviance": math.nan if f.get("deviance") is None else f["deviance"],
            "null_deviance": math.nan if f.get("null_deviance") is None else f["null_deviance"]}
    lr.drivers = sorted({d for f in lr._fits.values() for d in f["drivers"]})
    lr.formulas = {k: list(f["drivers"]) for k, f in lr._fits.items()}
    return lr


def _load_input(spec: dict, base: Path):
    import landschaft as ls

    if spec["kind"] == "value":
        return spec["value"]
    p = Path(spec["path"])
    p = p if p.is_absolute() else base / p
    da = ls.load_raster(p)
    if spec.get("nodata") is not None:
        da = da.where(da != spec["nodata"])
    if spec["kind"] == "map":
        if da.ndim == 3:
            band = spec.get("band")
            names = [str(v) for v in da[da.dims[0]].values]
            k = names.index(band) if band in names else int(band) if band is not None else 0
            da = da[k]
        return da
    bands = spec.get("bands")
    if bands is not None and da.ndim == 3:
        names = [str(v) for v in da[da.dims[0]].values]
        da = da[[names.index(b) if b in names else int(b) for b in bands]]
    return da


def load_model(path):
    """Read a Terraflux graph of ls.simulate's pieces: ``(Model, inputs)``, inputs by name (maps read)."""
    from ._allocation import CLUE, Ordered
    from ._demand import AreaDemand
    from ._dynamic import distance_to, share_of
    from ._estimator import RasterPotential
    from ._markov import Markov
    from ._model import Model
    from ._patches import ExpanderPatcher
    from ._plus import CARS
    from ._rules import Neighbourhood, Rules

    path = Path(path)
    g = _read_graph(path)
    inputs = {k: _load_input(v, path.parent) for k, v in g.get("inputs", {}).items()}
    nodes = {n["id"]: n for n in g.get("nodes", [])}
    sim = nodes.get("simulation")
    if sim is None or sim.get("op") != "control.repeat":
        raise ValueError("not a graph of ls.simulate's pieces: run it with terraflux")
    body = {n["id"]: n for n in sim["body"]["nodes"]}
    alloc_node = body.get("allocate")
    if alloc_node is None:
        raise ValueError("not a graph of ls.simulate's pieces: run it with terraflux")

    def ref_input(r):
        return r[1:] if isinstance(r, str) and r.startswith("$") else None

    # the potential
    pn = nodes.get("potential")
    if pn is None:
        raise ValueError("the graph has no potential node: models made by `terraflux new` are fitted by terraflux")
    if pn["op"] == "woe.weights":
        potential = _woe_from(pn["params"]["weights"])
    elif pn["op"] == "glm.coefficients":
        potential = _glm_from(pn["params"]["fits"])
    elif pn["op"] == "stack.classes":
        stack = inputs[ref_input(pn["inputs"]["stack"])]
        names = [str(v) for v in stack[stack.dims[0]].values]

        def key(n):
            if "_to_" in n:
                f, t = n.split("_to_")
                return (_class_value(float(f)), _class_value(float(t)))
            return _class_value(float(n))
        keys = [key(n) for n in names]
        if isinstance(keys[0], tuple):
            potential = {k: stack[i] for i, k in enumerate(keys)}
        else:
            potential = RasterPotential({k: stack[i] for i, k in enumerate(keys)})
    else:
        raise ValueError(f"unknown potential {pn['op']}")
    # the demand
    dn = nodes["demand"]
    if dn["op"] == "demand.table":
        t = dn["params"]["table"]
        import pandas as pd

        idx = pd.Index(t["times"], name="time") if t.get("times") else None
        demand = AreaDemand(pd.DataFrame(np.asarray(t["rows"], dtype=np.float64), index=idx,
                                         columns=[_class_value(c) for c in t["classes"]]))
    elif dn["op"] == "markov.matrix":
        p = dn["params"]
        demand = Markov(np.asarray(p["single"], dtype=np.float64), p["classes"], int(p.get("steps", 1)))
        if p.get("multi") is not None:
            demand._multi = np.asarray(p["multi"], dtype=np.float64)
    else:
        raise ValueError(f"unknown demand {dn['op']}")
    # the dynamic variables
    dynamic = {}
    for n in sim["body"]["nodes"]:
        if n["op"] in ("dynamic.distance_to", "dynamic.share_of"):
            setter = next(x for x in sim["body"]["nodes"] if x["op"] == "stack.set"
                          and x["inputs"]["map"].startswith(n["id"] + "."))
            name = setter["params"]["name"]
            prm = n.get("params", {})
            dynamic[name] = (distance_to(prm["classes"], unit=prm.get("unit", "map")) if n["op"] == "dynamic.distance_to"
                             else share_of(prm["classes"], prm.get("window", 3)))
    # the allocation and rules
    op, prm, ins = alloc_node["op"], alloc_node.get("params", {}), alloc_node["inputs"]
    rules = Rules()
    if op == "lulcc.step":
        a = prm["allocation"]
        if a["method"] == "clue":
            allocation = CLUE(**{k: v for k, v in a.items() if k != "method"})
        else:
            allocation = Ordered(a["order"], a.get("stochastic", True))
        if "elasticity" in prm:
            rules.elasticity = prm["elasticity"]
        if "allowed" in prm:
            rules.allowed = np.asarray(prm["allowed"], dtype=np.int64)
        if prm.get("neighbourhood"):
            nb = prm["neighbourhood"]
            w = nb.get("weights")
            if isinstance(w, dict):
                w = {_class_value(float(k)): np.asarray(v, dtype=np.float64) for k, v in w.items()}
            rules.neighbourhood = Neighbourhood({_class_value(float(k)): v for k, v in nb["rules"].items()}, w)
        hn = nodes.get("history", {})
        if "history" in hn.get("inputs", {}):
            rules.history = inputs[ref_input(hn["inputs"]["history"])]
    elif op == "dinamica.step":
        pt = nodes["patches"]["params"]["transitions"]
        trs = [(_class_value(x["from"]), _class_value(x["to"])) for x in pt]
        allocation = ExpanderPatcher(
            expansion={tr: x["expansion"] for tr, x in zip(trs, pt)},
            expander={tr: tuple(x["expander"]) for tr, x in zip(trs, pt)},
            patcher={tr: tuple(x["patcher"]) for tr, x in zip(trs, pt)},
            prune_factor=prm.get("prune_factor", 10.0), window=tuple(prm.get("window", (3, 3))),
            steps=sim["inputs"]["times"] if isinstance(sim["inputs"]["times"], int) else None)
    elif op == "cars.step":
        steps = nodes.get("cars_demand", {}).get("params", {}).get("steps")
        allocation = CARS(weights={_class_value(float(k)): v for k, v in prm.get("weights", {}).items()} or 1.0,
                          seeds={_class_value(float(k)): v for k, v in prm.get("seeds", {}).items()} or 0.1,
                          decay=prm.get("decay", 0.9), step_size=prm.get("step_size", 500.0),
                          window=prm.get("window", 3), max_iter=prm.get("max_iter", 5000), steps=steps)
        if "allowed" in prm:
            rules.allowed = np.asarray(prm["allowed"], dtype=np.int64)
    else:
        raise ValueError(f"unknown allocation {op}")
    if "restrictions" in ins:
        rules.restrictions = inputs[ref_input(ins["restrictions"])]
    model = Model(potential, demand, allocation, rules, dynamic or None)
    return model, inputs


# ----------------------------------------------------------------------------- running

def _executable(exe=None) -> str:
    exe = exe or os.environ.get("TERRAFLUX") or shutil.which("terraflux")
    if not exe:
        raise FileNotFoundError("terraflux was not found: install it (https://github.com/sacridini/terraflux), put "
                                "it on PATH or give its path (exe= or the TERRAFLUX variable)")
    return str(exe)


def terraflux(model, *, out=None, seed=None, exe=None, quiet: bool = True) -> xr.DataArray:
    """Run a Terraflux model with the ``terraflux`` command; the simulated maps ``(step, y, x)``.

    Parameters
    ----------
    model
        The model file (``Model.save`` writes one).
    out
        The folder of the outputs (default: ``output`` next to the model).
    seed
        Replace the model's seed.
    exe
        The ``terraflux`` executable (default: the ``TERRAFLUX`` variable, else the one on PATH).

    Returns
    -------
    xarray.DataArray
        The ``maps`` output: one map per step (1, 2...), the start not included.
    """
    import landschaft as ls

    model = Path(model)
    out = Path(out) if out is not None else model.with_name("output")
    cmd = [_executable(exe), "run", str(model), "--out", str(out)]
    if seed is not None:
        cmd += ["--set", f"seed={int(seed)}"]
    if quiet:
        cmd.append("--quiet")
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"terraflux failed:\n{r.stderr}")
    maps = ls.load_raster(out / "maps.tif", masked=False)
    nodata = maps.rio.nodata
    maps = maps.astype(np.float64)
    if nodata is not None and not np.isnan(nodata):
        maps = maps.where(maps != nodata)
    if maps.ndim == 2:
        maps = maps.expand_dims("band")
    return maps.rename({maps.dims[0]: "step"}).assign_coords(step=np.arange(1, maps.shape[0] + 1))
