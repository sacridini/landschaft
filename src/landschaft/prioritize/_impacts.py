# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5's impact assessment (impacts.cpp, impactzones.cpp, cumulativeimpacts.cpp):
the loss of the features to land-use change around impact sources, and to direct habitat
loss and cumulative pressures, reported per area, impact group, source and zone; the
features are reduced accordingly before the ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core

FLT_MAX = float(np.finfo(np.float32).max)
MEASURES = ("area", "fractional_distribution", "fractional_loss", "absolute_distribution", "absolute_loss")


@dataclass(frozen=True)
class ImpactSource:
    """An impact source of an impact group (a line of Zonation's impact buffers file).

    Parameters
    ----------
    group
        Impact group (the features with this ``impact_groups`` entry).
    source
        Value of the source's cells in the impact source raster.
    footprint
        Loss (%) in the source's own cells.
    buffers
        ``(radius, loss %)`` of successive rings around the source (radius in cells,
        increasing).
    """

    group: int
    source: int
    footprint: float
    buffers: tuple = ()


@dataclass
class ImpactScenario:
    """An impact assessment scenario (Zonation's ``-s``): land-use change at sources.

    Parameters
    ----------
    sources
        Raster of the impact sources (integer values; 0: none).
    impacts
        The :class:`ImpactSource` of every group and source.
    name
        Project id (the output directory's name).
    subareas
        Raster of subareas (integers, 0: none) reported apart.
    area_unit, impact_unit, area_factor
        Names of the units of area and of the features' values, and the area of a
        cell in the area unit.
    rasters
        Write the impact rasters of the groups.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> species = np.random.default_rng(0).random((2, 30, 40))
    >>> road = np.zeros((30, 40), np.uint16)
    >>> road[15, :] = 1                                   # source 1: a planned road
    >>> sc = P.ImpactScenario(road, [P.ImpactSource(group=1, source=1, footprint=100,
    ...                                             buffers=((3, 50),))])   # 50 % loss up to 3 cells
    >>> res = P.zonation(species, impact_groups=[1, -1], impact=sc)
    >>> zones = res.impacts["zones"]
    >>> sorted(zones["zone"].unique().tolist())          # footprint and buffer ring
    [0, 1]
    """

    sources: object
    impacts: list
    name: str = "scenario"
    subareas: object = None
    area_unit: str = ""
    impact_unit: str = ""
    area_factor: float = 1.0
    rasters: bool = True


@dataclass
class Pressure:
    """A pressure raster of a cumulative impact scenario."""

    raster: object
    normalize: bool = False
    offset: float = 0.0
    divisor: float = 1.0


@dataclass
class CumulativeScenario:
    """A cumulative impact assessment scenario (Zonation's ``-e``).

    Parameters
    ----------
    name, subareas, area_unit, impact_unit, area_factor, rasters
        As :class:`ImpactScenario`.
    direct
        Direct habitat loss: ``None``, ``"sum"`` (the weighted sum of the loss rasters,
        at most 1) or ``"product"`` (1 - the product of what the rasters leave).
    direct_rasters, direct_matrix
        Loss rasters (fractions) and the weights (impact groups x rasters).
    cumulative
        Cumulative pressures: ``None``, ``"variant1"`` (the weighted sum of the
        pressures through the group's response function) or ``"variant2"`` (each
        pressure through its response function, then aggregated).
    pressures
        The :class:`Pressure` rasters.
    cumulative_matrix
        variant1: weights (impact groups x pressures).
    aggregation
        variant2: ``"add"``, ``"multiply"`` or ``"max"``.
    functions
        Response functions, ``{(group, pressure): [(x, loss), ...]}``: ``pressure``
        ``None`` for the group's default (1-based pressure numbers otherwise).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> species = np.random.default_rng(0).random((2, 30, 40))
    >>> pressure = np.linspace(0, 1, 30 * 40, dtype=np.float32).reshape(30, 40)
    >>> sc = P.CumulativeScenario(cumulative="variant1", pressures=[P.Pressure(pressure)],
    ...                           cumulative_matrix=np.array([[1.0]]),
    ...                           functions={(1, None): [(0, 0), (1, 0.5)]})   # up to 50 % loss
    >>> res = P.zonation(species, impact_groups=[1, 1], cumulative_impact=sc)
    >>> sorted(res.impacts["cumulative"]["phase"].unique().tolist())
    ['cumulative']
    """

    name: str = "cumulative"
    subareas: object = None
    area_unit: str = ""
    impact_unit: str = ""
    area_factor: float = 1.0
    rasters: bool = True
    direct: str | None = None
    direct_rasters: list = field(default_factory=list)
    direct_matrix: object = None
    cumulative: str | None = None
    pressures: list = field(default_factory=list)
    cumulative_matrix: object = None
    aggregation: str = "add"
    functions: dict = field(default_factory=dict)


# ---------------------------------------------------------------- zones


def sqr_distance(mask: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (float32) to the nearest True cell (FLT_MAX-large without any)."""
    from scipy import ndimage

    if not mask.any():
        return np.full(mask.shape, FLT_MAX, dtype=np.float32)
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    yy, xx = np.indices(mask.shape)
    return ((yy - idx[0]) ** 2 + (xx - idx[1]) ** 2).astype(np.float32)


@dataclass
class Zone:
    group: int
    source: int
    zone: int
    strength: np.float32
    y0: int
    x0: int
    mask: np.ndarray  # bool, on the rectangle from (y0, x0)


def make_zones(sources: np.ndarray, impacts: list) -> dict[int, list[Zone]]:
    """ZoneMasks: per group, its zones (source ascending, footprint then rings)."""
    ids = sorted({int(s.source) for s in impacts})
    zones: dict[int, list[Zone]] = {}
    H, W = sources.shape
    for sid in ids:
        cells = sources == sid
        if not cells.any():
            import warnings

            warnings.warn(f"source {sid} not found in the impact source raster", stacklevel=4)
            continue
        ys, xs = np.nonzero(cells)
        rmax = max((s.buffers[-1][0] for s in impacts if s.source == sid and s.buffers), default=0)
        y0, y1 = max(ys.min() - rmax, 0), min(ys.max() + rmax, H - 1)
        x0, x1 = max(xs.min() - rmax, 0), min(xs.max() + rmax, W - 1)
        sqrd = sqr_distance(cells[y0:y1 + 1, x0:x1 + 1])
        for g in sorted({s.group for s in impacts}):
            for s in impacts:
                if s.group != g or s.source != sid:
                    continue
                rings = [(-1, 0, s.footprint)] + [(prev, r, imp) for prev, (r, imp) in
                                                   zip([0] + [b[0] for b in s.buffers[:-1]], s.buffers)]
                for zn, (rmin, rmx, strength) in enumerate(rings):
                    mind = np.float32(-1) if rmin < 0 else np.float32(rmin * rmin)
                    maxd = np.float32(rmx * rmx)
                    inz = (sqrd > mind) & (sqrd <= maxd)
                    if not inz.any():
                        continue
                    zy, zx = np.nonzero(inz)
                    a, b, c, d = zy.min(), zy.max(), zx.min(), zx.max()
                    zones.setdefault(g, []).append(
                        Zone(g, sid, zn, np.float32(np.float32(strength) / np.float32(100)), y0 + a, x0 + c,
                             inz[a:b + 1, c:d + 1]))
    return zones


def zone_dense(z: Zone, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = z.mask.shape
    out[z.y0:z.y0 + h, z.x0:z.x0 + w] = z.mask
    return out


# ---------------------------------------------------------------- cumulative rasters


def pressure_value(fun, x: np.ndarray) -> np.ndarray:
    """getInterpolatedValue of a response function (lower_bound on x, float32)."""
    xs = np.asarray([p[0] for p in fun], dtype=np.float32)
    ys = np.asarray([p[1] for p in fun], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    i = np.searchsorted(xs, x, side="left")
    out = np.empty_like(x)
    first = i == 0
    last = i >= xs.size
    mid = ~(first | last)
    out[first] = ys[0]
    out[last] = ys[-1]
    im = i[mid]
    t = ((x[mid] - xs[im]) / (xs[im] - xs[im - 1])).astype(np.float32)
    out[mid] = (ys[im] + (ys[im] - ys[im - 1]) * t).astype(np.float32)
    return out


def _read_layer(read, raster) -> np.ndarray:
    v = read(raster)
    if (v < 0).any():
        import warnings

        warnings.warn("negative values in a pressure raster, clipped to zero", stacklevel=4)
        v = np.maximum(v, np.float32(0))
    return v


def direct_raster(sc: CumulativeScenario, group_row: int, read) -> np.ndarray:
    weights = np.asarray(sc.direct_matrix, dtype=np.float32)[group_row]
    multiply = sc.direct == "product"
    raster = None
    for k, r in enumerate(sc.direct_rasters):
        if weights[k] > 0:
            data = _read_layer(read, r)
            if raster is None:
                raster = np.full(data.shape, 1 if multiply else 0, dtype=np.float32)
            if multiply:
                raster = (raster * (np.float32(1) - data)).astype(np.float32)
            else:
                raster = (raster + data * weights[k]).astype(np.float32)
    if raster is None:
        raise ValueError(f"empty direct habitat loss raster for impact group row {group_row}")
    if not multiply:
        raster = (np.float32(1) - np.minimum(raster, np.float32(1))).astype(np.float32)
    return raster


def cumulative_raster(sc: CumulativeScenario, group: int, group_row: int, read) -> np.ndarray:
    variant1 = sc.cumulative == "variant1"
    agg = "add" if variant1 else sc.aggregation
    n = len(sc.pressures)
    weights = np.asarray(sc.cumulative_matrix, dtype=np.float32)[group_row] if variant1 else np.ones(n, np.float32)
    funs = {k[1]: v for k, v in sc.functions.items() if k[0] == group}
    raster = None
    for k, p in enumerate(sc.pressures):
        if weights[k] <= 0:
            continue
        data = _read_layer(read, p.raster)
        if p.normalize:
            m = data.max()
            if m > 0:
                data = (data / m).astype(np.float32)
        if p.offset != 0:
            data = np.maximum(data - np.float32(p.offset), np.float32(0)).astype(np.float32)
        if p.divisor != 1:
            if p.divisor <= 0:
                raise ValueError("zero divisor for a pressure raster")
            data = (data / np.float32(p.divisor)).astype(np.float32)
        if not variant1:
            fun = funs.get(k + 1, funs.get(None))
            if fun is None:
                raise ValueError(f"no response function for impact group {group}, pressure {k + 1}")
            data = pressure_value(fun, data)
        if raster is None:
            raster = np.full(data.shape, 1 if agg == "multiply" else 0, dtype=np.float32)
        if agg == "add":
            raster = (raster + data).astype(np.float32)
        elif agg == "multiply":
            raster = (raster * (np.float32(1) - data)).astype(np.float32)
        else:
            raster = np.maximum(raster, data)
    if raster is None:
        raise ValueError(f"empty cumulative pressure raster for impact group {group}")
    if variant1:
        if None not in funs:
            raise ValueError(f"pressure function not found for impact group {group}")
        raster = pressure_value(funs[None], raster)
    if agg != "multiply":
        raster = (np.float32(1) - np.minimum(raster, np.float32(1))).astype(np.float32)
    return raster


# ---------------------------------------------------------------- the analysis


def layer_bins(spec, max_norm_sum: float, orig_abs_sum: float):
    """getLayerBins: the value intervals (float32, of the sum-normalized values) of a feature."""
    if spec is None or (not isinstance(spec, (list, tuple, np.ndarray)) and spec < 0):
        return None
    if max_norm_sum > 0:
        if not isinstance(spec, (list, tuple, np.ndarray)):  # 0: ten automatic bins
            m = 1 / max_norm_sum
            bins = [np.float32(i * m / 10) for i in range(11)]
        else:
            bins = [np.float32(np.float32(b) / orig_abs_sum) for b in spec]
        if len(bins) == 1:
            bins.append(np.float32(FLT_MAX))
        else:
            bins[-1] = np.nextafter(np.float32(bins[-1]), np.float32(FLT_MAX), dtype=np.float32)
    else:
        bins = [np.float32(0), np.float32(FLT_MAX)]
    return [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]


def find_bin(v: np.ndarray, bins) -> np.ndarray:
    out = np.full(v.shape, -1, dtype=np.int64)
    for k in range(len(bins) - 1, -1, -1):  # the first matching interval wins
        lo, hi = bins[k]
        out[(v >= lo) & (v < hi)] = k
    return out


class Accumulator:
    """Zonation's Impact{impact, loss, area} sums per key, added in its order: by cell
    (row-major), then by the order within the cell (zone, phase)."""

    def __init__(self):
        self.parts: dict = {}
        self._done: dict | None = None

    def add(self, key, cells, sub, impact=None, loss=None):
        """Contributions at ``cells`` (packed indices) with order ``sub`` within the cell."""
        if cells.size == 0:
            return
        self.parts.setdefault(key, []).append((cells, np.broadcast_to(sub, cells.shape), impact, loss))
        self._done = None

    def _finish(self):
        done = {}
        for key, parts in self.parts.items():
            out = [0.0, 0.0, 0.0]
            for j, attr in ((0, 2), (1, 3)):
                c = [p[0] for p in parts if p[attr] is not None]
                if not c:
                    continue
                cells = np.concatenate(c)
                sub = np.concatenate([p[1] for p in parts if p[attr] is not None])
                vals = np.concatenate([np.asarray(p[attr], np.float64) for p in parts if p[attr] is not None])
                order = np.lexsort((sub, cells))
                out[j] = _seq(0.0, vals[order])
                if j == 0:
                    out[2] = float(cells.size)
            done[key] = out
        self._done = done

    def get(self, key):
        if self._done is None:
            self._finish()
        return self._done.get(key, [0.0, 0.0, 0.0])

    def keys(self):
        if self._done is None:
            self._finish()
        return self._done.keys()


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """Eigen's round (half away from zero, as std::round), exactly, in float32."""
    x = np.asarray(x, np.float32)
    t = np.trunc(x)
    frac = (x - t).astype(np.float32)  # exact
    return (t + np.where(np.abs(frac) >= np.float32(0.5), np.sign(x), np.float32(0))).astype(np.float32)


def _seq(start: float, values: np.ndarray) -> float:
    """start + v0 + v1 + ... in sequence (double)."""
    if values.size == 0:
        return start
    return float(np.bincount(np.zeros(values.size + 1, np.int64),
                             weights=np.concatenate([[start], values.astype(np.float64)]))[0])


def run(data, kept, cells, grid, impact_groups, impact_bins, scenario: ImpactScenario | None,
        cumulative: CumulativeScenario | None, read_full, read_packed):
    """Apply the scenarios to the quantized data (as Zonation's runImpactScenario) and
    return the accumulated impacts."""
    H, W = grid.shape
    groups_of = {k: int(impact_groups[i]) for k, i in enumerate(kept) if int(impact_groups[i]) >= 0}
    group_layers: dict[int, list[int]] = {}
    for k, g in groups_of.items():
        group_layers.setdefault(g, []).append(k)
    group_numbers = sorted(group_layers)
    mns = np.asarray(data.max_norm_sums)
    oas = np.asarray(data.orig_abs_sums)
    bins = {}
    if impact_bins is not None:
        for k, i in enumerate(kept):
            b = layer_bins(impact_bins[i], mns[k], oas[k])
            if b is not None:
                bins[k] = b
    rows = cells // W
    cols = cells % W
    idx = np.arange(cells.size)

    def areas_of(raster):
        if raster is None:
            return np.zeros(cells.size, np.int64), []
        a = np.clip(np.rint(read_packed(raster)), 0, 2**32 - 1).astype(np.int64)
        ids = sorted(int(x) for x in np.unique(a) if x != 0)
        if not ids:
            raise ValueError("the subareas raster has no values above 0")
        lut = {v: i + 1 for i, v in enumerate(ids)}
        return np.array([lut.get(int(x), 0) for x in a], np.int64), ids

    src_state = cum_state = None
    if scenario is not None:
        src_ids = np.clip(np.rint(np.nan_to_num(read_full(scenario.sources), nan=0)), 0, 2**32 - 1).astype(np.int64)
        zones = make_zones(src_ids, list(scenario.impacts))
        s_area, s_ids = areas_of(scenario.subareas)
        src_state = {"zones": zones, "area": s_area, "areas": s_ids, "acc": Accumulator(), "scenario": scenario,
                     "sources": {}}
    if cumulative is not None:
        if src_state is not None and cumulative.subareas is scenario.subareas:
            c_area, c_ids = src_state["area"], src_state["areas"]
        else:
            c_area, c_ids = areas_of(cumulative.subareas)
        cum_state = {"area": c_area, "areas": c_ids, "acc": Accumulator(), "maxarea": {}, "scenario": cumulative,
                     "layers": set()}
    rasters = {}
    for g in group_numbers:
        grow = group_numbers.index(g)
        direct = cum = impact_raster = None
        if cum_state is not None:
            if cumulative.direct:
                direct = direct_raster(cumulative, grow, read_full)
            if cumulative.cumulative:
                cum = cumulative_raster(cumulative, g, grow, read_full)
        calc_source = src_state is not None and g in src_state["zones"]
        if src_state is not None and not calc_source:
            import warnings

            warnings.warn(f"no zones in impact group {g}", stacklevel=3)
        zl = dense = None
        if calc_source:
            zl = src_state["zones"][g]
            impact_raster = np.ones((H, W), dtype=np.float32)
            full = [zone_dense(z, (H, W)) for z in zl]
            for z, m in zip(zl, full):
                impact_raster[m] = (impact_raster[m] * (np.float32(1) - z.strength)).astype(np.float32)
            dense = [m[rows, cols] for m in full]
            src_state["sources"][g] = sorted({z.source for z in zl})
        for k in group_layers[g]:
            scale = np.asarray(data.layer_scale)[k]
            q = np.asarray(data.layer(k, False, -1.0))
            v = np.where(q >= 0, (q.astype(np.float32).astype(np.float64) * scale).astype(np.float32),
                         np.float32(0)).astype(np.float32)
            inv = 1 / scale if scale > 0 else 0.0
            old = _core.prio_eigen_sum_double(v.astype(np.float64)) * inv
            pos = v > 0  # Zonation calls addImpact where the value is above 0
            if cum_state is not None:
                acc = cum_state["acc"]
                cum_state["layers"].add(k)
                cum_state["maxarea"][k] = cum_state["maxarea"].get(k, 0.0) + float(pos.sum())
                first = pos.copy()
                for sub, (phase, rast) in enumerate((("direct", direct), ("cumulative", cum))):
                    if rast is None:
                        continue
                    imp = rast[rows, cols]
                    sel = pos & (imp < 1)
                    loss = ((np.float32(1) - imp) * v).astype(np.float32)
                    for a_all in (True, False):
                        s = sel if a_all else sel & (cum_state["area"] > 0)
                        keys = np.zeros(cells.size, np.int64) if a_all else cum_state["area"]
                        for a in np.unique(keys[s]):
                            m = s & (keys == a)
                            f = m & first
                            acc.add(("total", int(a), k), idx[f], sub, impact=v[f])
                            acc.add(("total", int(a), k), idx[m], sub, loss=loss[m])
                            acc.add((phase, int(a), g, k), idx[m], sub, impact=v[m], loss=loss[m])
                    v = np.where(sel, (v * imp).astype(np.float32), v).astype(np.float32)
                    first = first & ~sel
            if calc_source:
                imp = impact_raster[rows, cols]
                sel = pos & (imp < 1)
                total_loss = ((np.float32(1) - imp) * v).astype(np.float32)
                tstr = np.zeros(cells.size, np.float32)
                for z, m in zip(zl, dense):
                    tstr = np.where(m, (tstr + z.strength).astype(np.float32), tstr)
                with np.errstate(divide="ignore"):
                    tr = np.where(tstr > 0, (np.float32(1) / tstr).astype(np.float32), np.float32(0))
                bk = find_bin(v, bins[k]) if k in bins else np.full(cells.size, -1, np.int64)
                first = sel.copy()
                acc = src_state["acc"]
                for sub, (z, m) in enumerate(zip(zl, dense)):
                    mz = sel & m
                    loss = ((total_loss * z.strength).astype(np.float32) * tr).astype(np.float32)
                    for a_all in (True, False):
                        ms = mz if a_all else mz & (src_state["area"] > 0)
                        keys = np.zeros(cells.size, np.int64) if a_all else src_state["area"]
                        for a in np.unique(keys[ms]):
                            m2 = ms & (keys == a)
                            fm = m2 & first
                            acc.add(("zone", int(a), g, z.source, k, -1, z.zone), idx[m2], sub, impact=v[m2],
                                    loss=loss[m2])
                            acc.add(("total", int(a), g, k, -1), idx[m2], sub, loss=loss[m2])
                            acc.add(("total", int(a), g, k, -1), idx[fm], sub, impact=v[fm])
                            for b in np.unique(bk[m2]):
                                if b < 0:
                                    continue
                                mb = m2 & (bk == b)
                                fb = mb & first
                                acc.add(("zone", int(a), g, z.source, k, int(b), z.zone), idx[mb], sub,
                                        impact=v[mb], loss=loss[mb])
                                acc.add(("total", int(a), g, k, int(b)), idx[mb], sub, loss=loss[mb])
                                acc.add(("total", int(a), g, k, int(b)), idx[fb], sub, impact=v[fb])
                    first = first & ~mz
                v = np.where(sel, (v * imp).astype(np.float32), v).astype(np.float32)
            # back to the storage units; the level scaled by what remains
            newq = _round_half_away((v * np.float32(inv)).astype(np.float32))
            if old > 0:
                new = _core.prio_eigen_sum_double(newq.astype(np.float64))
                data.scale_level(k, new / old)
            data.set_layer_values(k, np.ascontiguousarray(newq, dtype=np.float32))
        if calc_source and scenario.rasters:
            rasters[f"ig{g}_zones"] = (np.float32(1) - impact_raster).astype(np.float32)
        if cum_state is not None and cumulative.rasters:
            if direct is not None:
                rasters[f"ig{g}_directhabitatloss"] = (np.float32(1) - direct).astype(np.float32)
            if cum is not None:
                rasters[f"ig{g}_accumulativeloss"] = (np.float32(1) - cum).astype(np.float32)
    return {"source": src_state, "cumulative": cum_state, "rasters": rasters, "group_layers": group_layers,
            "bins": bins}


# ---------------------------------------------------------------- outputs


def _impact_value(measure: str, e, abs_sum: float, factor: float) -> float:
    impact, loss, area = e
    if measure == "area":
        return area * factor
    if measure == "fractional_distribution":
        return impact
    if measure == "fractional_loss":
        return loss
    if measure == "absolute_distribution":
        return impact * abs_sum
    return loss * abs_sum


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _unit(sc, measure: str) -> str:
    name = sc.area_unit if measure == "area" else (sc.impact_unit if measure.startswith("absolute") else "")
    return f"_{name}" if name else ""


def tables(res_imp, ctx) -> dict:
    """The accumulated impacts as DataFrames."""
    import pandas as pd

    out = {}
    st = res_imp["source"]
    if st is not None:
        acc = st["acc"]
        rows = []
        for key in acc.keys():
            if key[0] != "zone":
                continue
            _, a, g, s, k, b, z = key
            imp, loss, cells = acc.get(key)
            rows.append({"area": 0 if a == 0 else st["areas"][a - 1], "group": g, "source": s, "zone": z,
                         "feature": ctx["numbers"][k], "bin": b, "distribution": imp, "loss": loss, "cells": cells})
        out["zones"] = pd.DataFrame(rows)
    ct = res_imp["cumulative"]
    if ct is not None:
        acc = ct["acc"]
        rows = []
        for key in acc.keys():
            if key[0] in ("direct", "cumulative"):
                phase, a, g, k = key
                imp, loss, cells = acc.get(key)
                rows.append({"area": 0 if a == 0 else ct["areas"][a - 1], "group": g, "phase": phase,
                             "feature": ctx["numbers"][k], "distribution": imp, "loss": loss, "cells": cells})
        out["cumulative"] = pd.DataFrame(rows)
    return out


def write_source(res_imp, ctx, root: Path) -> list:
    """outputSourceAnalysisFiles: impacts/<project id>/<area>/..."""
    st = res_imp["source"]
    sc = st["scenario"]
    acc = st["acc"]
    out_dir = root / sc.name
    written = []
    abs_sums = ctx["orig_abs_sums"]
    group_layers = res_imp["group_layers"]
    bins = res_imp["bins"]
    zones = st["zones"]
    areas = [0] + list(range(1, len(st["areas"]) + 1))

    def layer_name(k):
        n = ctx["plain_names"][k]
        return n if n else Path(ctx["files"][k]).name

    for measure in MEASURES:
        unit = _unit(sc, measure)
        for a in areas:
            aname = "global" if a == 0 else f"area{st['areas'][a - 1]}"
            apath = out_dir / aname
            apath.mkdir(parents=True, exist_ok=True)
            for g in sorted(group_layers):
                if g not in zones:
                    continue
                sources = st["sources"][g]
                nzones = {s: sum(1 for z in zones[g] if z.source == s) for s in sources}

                def zone_e(s, k, b, zn, a=a, g=g):
                    return acc.get(("zone", a, g, s, k, b, zn))

                def src_total(s, k, b, nzones=nzones, zone_e=zone_e):
                    t = [0.0, 0.0, 0.0]
                    for zn in range(nzones[s]):
                        t = _add(t, zone_e(s, k, b, zn))
                    return t

                def val(e, k, measure=measure):
                    return _impact_value(measure, e, abs_sums[k], sc.area_factor)

                def group_file(path, layers, binned, a=a, g=g, sources=sources, zone_e=zone_e,
                               src_total=src_total, val=val, unit=unit):
                    kind = "bin_low" if binned else "feature"
                    head = f"{kind} total_all{unit} total_footprints{unit} total_zones{unit}"
                    head += "".join(f" s{s}total{unit} s{s}footprint{unit} s{s}zones{unit}" for s in sources)
                    if not binned:
                        head += " name"
                    lines = [head]
                    for k in layers:
                        def row(b, label, k=k):
                            tf = [0.0, 0.0, 0.0]
                            parts = ""
                            for s in sources:
                                stt = src_total(s, k, b)
                                fp = zone_e(s, k, b, 0)
                                tf = _add(tf, fp)
                                parts += f" {val(stt, k):f} {val(fp, k):f} {val(_sub(stt, fp), k):f}"
                            tot = acc.get(("total", a, g, k, b))
                            line = f"{label} {val(tot, k):f} {val(tf, k):f} {val(_sub(tot, tf), k):f}{parts}"
                            if b < 0:
                                line += f" {layer_name(k)}"
                            lines.append(line)

                        if binned:
                            for bi, (lo, _) in enumerate(bins[k]):
                                row(bi, f"{float(np.float32(float(lo) * abs_sums[k])):f}")
                        else:
                            row(-1, f"{ctx['numbers'][k]}")
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    written.append(path)

                group_file(apath / f"{aname}_ig{g}.{measure}.csv", group_layers[g], False)
                binned_layers = [k for k in group_layers[g] if k in bins]
                if binned_layers:
                    (apath / "binned").mkdir(exist_ok=True)
                    for k in binned_layers:
                        group_file(apath / "binned" / f"{aname}_ig{g}.binned.{ctx['unique_names'][k]}.{measure}.csv",
                                   [k], True)
                for s in sources:
                    spath = apath / f"source{s}"
                    spath.mkdir(exist_ok=True)

                    def source_file(path, k_binned, s=s, g=g, nzones=nzones, zone_e=zone_e, val=val, unit=unit):
                        binned = k_binned is not None
                        kind = "bin_low" if binned else "feature"
                        head = f"{kind} total_source{unit} footprint{unit} total_zones{unit}"
                        head += "".join(f" z{zn}{unit}" for zn in range(1, nzones[s]))
                        if not binned:
                            head += " name"
                        lines = [head]

                        def zrow(k, b, label, name):
                            zs = [zone_e(s, k, b, zn) for zn in range(nzones[s])]
                            t = [0.0, 0.0, 0.0]
                            for e in zs:
                                t = _add(t, e)
                            line = (f"{label} {val(t, k):f} {val(zs[0], k):f} {val(_sub(t, zs[0]), k):f}"
                                    + "".join(f" {val(e, k):f}" for e in zs[1:]))
                            if name:
                                line += f" {layer_name(k)}"
                            lines.append(line)

                        if binned:
                            k = k_binned
                            for bi, (lo, _) in enumerate(bins[k]):
                                zrow(k, bi, f"{float(np.float32(float(lo) * abs_sums[k])):f}", False)
                        else:
                            for k in group_layers[g]:
                                zrow(k, -1, f"{ctx['numbers'][k]}", True)
                        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        written.append(path)

                    source_file(spath / f"{aname}_source{s}_ig{g}.{measure}.csv", None)
                    for k in group_layers[g]:
                        if k in bins:
                            (spath / "binned").mkdir(exist_ok=True)
                            source_file(spath / "binned" /
                                        f"{aname}_source{s}_ig{g}.binned.{ctx['unique_names'][k]}.{measure}.csv", k)
    return written


def write_cumulative(res_imp, ctx, root: Path) -> list:
    """outputCumulativeAnalysisFiles: impacts/<project id>/impacts.<measure>.csv"""
    ct = res_imp["cumulative"]
    sc = ct["scenario"]
    acc = ct["acc"]
    out_dir = root / sc.name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    abs_sums = ctx["orig_abs_sums"]
    layers = sorted(ct["layers"])
    groups = sorted(res_imp["group_layers"])
    areas = [0] + list(range(1, len(ct["areas"]) + 1))
    for measure in MEASURES:
        unit = _unit(sc, measure)
        head = f"feature remaining{unit}[total] "
        for a in areas:
            label = "total" if a == 0 else f"SA{ct['areas'][a - 1]}"
            head += f"direct{unit}[{label}] cumulative{unit}[{label}] "
        lines = [head]

        def val(e, k, remaining, measure=measure):
            imp, loss, area = e
            if measure == "area":
                return ((ct["maxarea"][k] - area) if remaining else area) * sc.area_factor
            if measure == "fractional_distribution":
                return 1.0 - imp if remaining else imp
            if measure == "fractional_loss":
                return 1.0 - loss if remaining else loss
            if measure == "absolute_distribution":
                return (1.0 - imp if remaining else imp) * abs_sums[k]
            return (1.0 - loss if remaining else loss) * abs_sums[k]

        for k in layers:
            tot = acc.get(("total", 0, k))
            line = f"{ctx['numbers'][k]} {val(tot, k, True):f} "
            for a in areas:
                d = [0.0, 0.0, 0.0]
                c = [0.0, 0.0, 0.0]
                for g in groups:
                    d = _add(d, acc.get(("direct", a, g, k)))
                    c = _add(c, acc.get(("cumulative", a, g, k)))
                line += f"{val(d, k, False):f} {val(c, k, False):f} "
            lines.append(line)
        path = out_dir / f"impacts.{measure}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written
