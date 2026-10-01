# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5's post-processing analyses of a ranking (postprocessingfunctions.cpp,
flexibilitymap.cpp): statistics of areas, comparison with another solution,
histograms of the features' coverage and the flexibility map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _core
from .._save import Saveable


@dataclass
class AreaStatistics(Saveable):
    """Statistics of areas of a ranking (:meth:`ZonationResult.area_statistics`).

    Attributes
    ----------
    areas : pandas.DataFrame
        Per area (and weight class or output group, column ``features``): size in
        cells, mean rank, the sum of the features' normalized values (``ds``) and
        weighted (``wds``), relative to what the same area would hold on average
        (``rfd``, ``wfd``), and the cost.
    coverage : dict
        Per weight class / group, a DataFrame: the fraction of every feature in
        every area.
    mask : numpy.ndarray or None
        The areas (0: outside), when they come from the ranking.
    """

    areas: pd.DataFrame
    coverage: dict
    mask: np.ndarray | None = None


def _feature_sets(ctx, groups: bool = True) -> list[tuple[str, list[int]]]:
    cls = ctx["classes"]
    sets = [("pos", cls["positive"]), ("neg", cls["negative"]), ("zero", cls["zero"])]
    og = ctx["groups"]
    if groups and og is not None:
        for g in sorted({int(x) for x in og if x >= 0}):
            sets.append((str(g), [i for i, x in enumerate(og) if x == g]))
    return [(n, i) for n, i in sets if i]


def _features_float(ctx) -> np.ndarray:
    """(cells, layers) float32: Zonation's getFeatures<float> (value * layer scale, 0 without data)."""
    data = ctx["data"]
    if "features32" not in ctx:
        scale = np.asarray(data.layer_scale)
        cols = []
        for k in range(len(scale)):
            q = np.asarray(data.layer(k, False, -1.0))
            v = np.where(q >= 0, q.astype(np.float32).astype(np.float64) * scale[k], 0.0).astype(np.float32)
            cols.append(v)
        ctx["features32"] = np.stack(cols, axis=1)
    return ctx["features32"]


def area_statistics(ctx, rank: np.ndarray, mask: np.ndarray) -> AreaStatistics:
    """areaStatistics for every feature set: ``mask`` per data cell (0: none)."""
    feats = _features_float(ctx)
    w = np.asarray(ctx["weights"], dtype=np.float32)
    cost = ctx["cost"]
    N = feats.shape[0]
    areas = np.unique(mask[mask != 0])
    row = np.searchsorted(areas, mask)
    rows = []
    coverage = {}
    for fid, idx in _feature_sets(ctx, ctx["groups_on"]):
        fsum = np.zeros(N)
        fwsum = np.zeros(N)
        for i in idx:
            v = feats[:, i].astype(np.float64)
            fsum += v
            fwsum += v * float(w[i])
        full = np.bincount(np.zeros(N, np.int64), weights=fsum, minlength=1)[0]
        wfull = np.bincount(np.zeros(N, np.int64), weights=fwsum, minlength=1)[0]
        sel = mask != 0
        r = row[sel]
        na = areas.size
        size = np.bincount(r, minlength=na)
        avg = np.bincount(r, weights=rank[sel].astype(np.float64), minlength=na)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(size > 0, avg / size, avg)
        ds = np.bincount(r, weights=fsum[sel], minlength=na)
        wds = np.bincount(r, weights=fwsum[sel], minlength=na)
        ac = np.bincount(r, weights=cost[sel].astype(np.float64), minlength=na) if cost is not None else np.zeros(na)
        cov = np.stack([np.bincount(r, weights=feats[sel, i].astype(np.float64), minlength=na) for i in idx], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            es = size * full / N
            wes = size * wfull / N
            fd = ds / es
            fdw = wds / wes
        for a in range(na):
            rows.append({"features": fid, "area": int(areas[a]), "size": int(size[a]), "avgrank": avg[a],
                         "ds": ds[a], "wds": wds[a], "rfd": fd[a], "wfd": fdw[a], "cost": ac[a]})
        names = [ctx["names"][i] for i in idx]
        df = pd.DataFrame(cov, columns=names)
        df.insert(0, "area", areas.astype(np.int64))
        coverage[fid] = df
    return AreaStatistics(pd.DataFrame(rows), coverage)


def rank_range_mask(rank: np.ndarray, min_rank: float, max_rank: float) -> np.ndarray:
    _check_range(min_rank, max_rank)
    r = rank.astype(np.float64)
    return ((r >= min_rank) & (r <= max_rank)).astype(np.int32)


def _check_range(lo, hi):
    if lo > hi:
        raise ValueError("min_rank should be less than max_rank")
    if lo < 0 or lo > 1 or hi < 0 or hi > 1:
        raise ValueError("min_rank and max_rank should be between 0 and 1")


def separate_areas(rank_map: np.ndarray, min_rank: float, max_rank: float, min_size: int) -> np.ndarray:
    """Zonation's area_statistics_separate_areas labels: 8-connected areas of cells in
    the rank range, numbered in row order, those under ``min_size`` cells dropped."""
    _check_range(min_rank, max_rank)
    if min_size <= 0:
        raise ValueError("min_area_size should be greater than 0 grid cells")
    from scipy import ndimage

    r = np.where(np.isfinite(rank_map), rank_map, -1).astype(np.float64)
    inside = np.isfinite(rank_map) & (r >= min_rank) & (r <= max_rank)
    labels, n = ndimage.label(inside, structure=np.ones((3, 3), bool))
    # Zonation numbers the areas by the first cell met in row order (scipy does the same)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)


def histogram(ctx, crank: np.ndarray, levels: np.ndarray, min_rank: float, max_rank: float, bins: int) -> dict:
    """Zonation's histogram of the features' coverage of the ranks [min_rank, max_rank]."""
    if bins < 1:
        raise ValueError("Number of bars should be greater than 0 in histogram")
    if min_rank < 0 or min_rank > 1 or max_rank < 0 or max_rank > 1:
        raise ValueError("Priority rank should be between 0 and 1 in histogram")
    if max_rank < min_rank:
        raise ValueError("Priority rank maximum should not be less than priority rank minimum in histogram")
    v = (interpolated(ctx, crank, levels, np.float32(min_rank)) - interpolated(ctx, crank, levels,
                                                                             np.float32(max_rank))
         + interpolated(ctx, crank, levels, np.float32(1.0))).astype(np.float32)
    out = {}
    for fid, idx in _feature_sets(ctx, ctx["groups_on"]):
        counts = np.zeros(bins, dtype=np.int64)
        for i in idx:
            fo = min(max(v[i], np.float32(0)), np.float32(1))
            k = min(int(fo * np.float32(bins)), bins - 1)
            counts[k] += 1
        out[fid] = pd.DataFrame({"bin": np.arange(bins), "coverage_level": (np.arange(bins, dtype=np.float32)
                                                                           / np.float32(bins)),
                                 "count": counts})
    return out


def interpolated(ctx, crank: np.ndarray, levels: np.ndarray, rank: np.float32) -> np.ndarray:
    """getInterpolatedOccurrences: the curves at a rank (float32)."""
    n = crank.size
    i1 = min(int(float(rank) * ctx["curve_scalar"]), n - 1)
    i2 = min(n - 1, i1 + 1)
    r = np.float32(crank[i2] - crank[i1])
    t = np.float32((rank - crank[i1]) / r) if r > 0 else np.float32(0)
    return ((levels[i2] - levels[i1]) * t + levels[i1]).astype(np.float32)


def flexibility(ctx, order_cells: np.ndarray, crank, levels, max_error: float, up: bool, down: bool):
    if max_error <= 0.0 or max_error >= 100.0:
        raise ValueError("max error should be between 0 and 100 in flexibility_map")
    pos, neg = _core.prio_flexibility(
        ctx["data"], ctx["rule"], ctx["param"], list(map(float, ctx["weights"])), list(map(float, ctx["abf_z"])),
        ctx["cost"], ctx["cost_in_ranking"], ctx["hmask"], order_cells.astype(np.uint32), crank,
        np.ascontiguousarray(levels, dtype=np.float32), int(ctx["curve_interval"]), float(max_error) / 100.0,
        bool(up), bool(down))
    return np.asarray(pos), np.asarray(neg)


# ---------------------------------------------------------------- Zonation's post-processing file

_CALL = re.compile(r"([A-Za-z][A-Za-z0-9_]+)\s*\(([^)]*)\)")


def parse_calls(text: str) -> list[tuple[str, list]]:
    """The calls of a post-processing file: name(param, ...) with numbers, true/false and "strings"."""
    calls = []
    body = "\n".join(line.split("#", 1)[0] if '"' not in line.split("#", 1)[0] else line
                     for line in text.splitlines())
    for m in _CALL.finditer(body):
        params = []
        for tok in re.findall(r'"[^"]*"|[^,\s][^,]*', m.group(2)):
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith('"'):
                params.append(tok[1:-1])
            elif tok.lower() in ("true", "false"):
                params.append(tok.lower() == "true")
            elif re.fullmatch(r"[+-]?\d+", tok):
                params.append(int(tok))
            else:
                params.append(float(tok))
        calls.append((m.group(1), params))
    return calls


def _g(x) -> str:
    """C++ ostream's default output of a double (6 significant digits)."""
    s = f"{float(x):.6g}"
    if "e" in s:
        mant, exp = s.split("e")
        s = f"{mant}e{exp[0]}{int(exp[1:]):02d}"
    return s


def write_area_statistics(stats: AreaStatistics, ctx, directory: Path, id_: str) -> list[Path]:
    out = []
    ids = f".{id_}" if id_ else ""
    for fid, df in stats.coverage.items():
        idx = dict(_feature_sets(ctx, ctx["groups_on"]))[fid]
        heads = [ctx["headers"][i] for i in idx]
        p = directory / f"coverage_matrix.{fid}{ids}.csv"
        lines = ['"area" ' + "".join(f"{h} " for h in heads)]
        for r in df.itertuples(index=False):
            lines.append(f"{r[0]} " + "".join(f"{_g(v)} " for v in r[1:]))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.append(p)
        p = directory / f"area_stats.{fid}{ids}.csv"
        a = stats.areas[stats.areas["features"] == fid]
        lines = ['"area" "size" "avgrank" "ds" "wds" "rfd" "wfd" "cost"']
        for r in a.itertuples(index=False):
            lines.append(f"{r.area} {r.size} {_g(r.avgrank)} {_g(r.ds)} {_g(r.wds)} {_g(r.rfd)} {_g(r.wfd)} "
                         f"{_g(r.cost)}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.append(p)
        p = directory / f"highest_occurrences.{fid}{ids}.txt"
        text = []
        for r in df.itertuples(index=False):
            sums = np.asarray(r[1:], dtype=np.float64)
            order = sorted(range(sums.size), key=lambda i: -sums[i])
            block = []
            for i in order:
                pct = sums[i] * 100.0
                if pct <= 0.1:
                    break
                k = idx[i]
                block.append(f"{_g(pct)} {ctx['numbers'][k]} {ctx['quoted_names'][k]} {ctx['files'][k]}")
            if block:
                text.append(f"Area {r[0]}")
                text.extend(block)
                text.append("")
        p.write_text("\n".join(text) + ("\n" if text else ""), encoding="utf-8")
        out.append(p)
    return out


def _fmt(x) -> str:
    """Zonation's fmt {} of a double (like %g, with ".0" on integers)."""
    s = f"{float(x):g}"
    if not any(c in s for c in ".eninf"):
        s += ".0"
    return s


def _write_tif(path: Path, arr, dtype, nodata=None) -> None:
    from ..io import write_raster

    write_raster(arr, path, dtype=dtype, nodata=nodata, colormap=None)


def run_calls(result, calls, directory: Path, base: Path) -> list[Path]:
    """Run Zonation's post-processing calls on a result, writing its files into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ctx = result._ctx
    for name, params in calls:
        def arg(i, default=None):
            return params[i] if i < len(params) else default

        if name == "area_statistics_external":
            path = Path(arg(0))
            path = path if path.is_absolute() else base / path
            st = result.area_statistics(path)
            written += write_area_statistics(st, ctx, directory, arg(1, ""))
        elif name == "area_statistics_rank_range":
            id_ = arg(2, "")
            st = result.area_statistics_rank_range(float(arg(0)), float(arg(1)))
            written += write_area_statistics(st, ctx, directory, id_)
            p = directory / ("area_statistics_rank_range.mask." + (f"{id_}." if id_ else "") + "tif")
            _write_tif(p, st.mask, "uint8")
            written.append(p)
        elif name == "area_statistics_separate_areas":
            id_ = arg(3, "")
            st = result.area_statistics_separate_areas(float(arg(0)), float(arg(1)), int(arg(2)))
            written += write_area_statistics(st, ctx, directory, id_)
            p = directory / ("area_statistics_separate_areas.mask." + (f"{id_}." if id_ else "") + "tif")
            _write_tif(p, st.mask, "int32")
            written.append(p)
        elif name == "solution_comparison":
            lo, hi, path, vlo, vhi, id_ = (float(arg(0)), float(arg(1)), arg(2), float(arg(3)), float(arg(4)),
                                           arg(5, ""))
            path = Path(path)
            path = path if path.is_absolute() else base / path
            cmp = result.solution_comparison(path, lo, hi, vlo, vhi)
            n = int(ctx["cells"].size)
            ids = f".{id_}" if id_ else ""
            p = directory / f"solution_comparison{ids}.txt"
            p.write_text(
                f"Comparing present solution to {path.resolve()}\n"
                f"Priority rank range of interest in present solution is from {_fmt(lo)} to {_fmt(hi)}\n"
                f"Numeric range of interest in comparison file is from {_fmt(vlo)} to {_fmt(vhi)}\n"
                f"Number of grid cells chosen in both solutions is {cmp['both']} "
                f"({_fmt(100.0 * cmp['both'] / n)}% of analysis area)\n"
                f"Number of grid cells chosen in neither solution is {cmp['neither']} "
                f"({_fmt(100.0 * cmp['neither'] / n)}% of analysis area)\n"
                f"Number of grid cells chosen only in present solution is {cmp['only_this']} "
                f"({_fmt(100.0 * cmp['only_this'] / n)}% of analysis area)\n"
                f"Number of grid cells chosen only in comparison file is {cmp['only_other']} "
                f"({_fmt(100.0 * cmp['only_other'] / n)}% of analysis area)\n", encoding="utf-8")
            written.append(p)
            p = directory / f"solution_comparison.mask{ids}.tif"
            _write_tif(p, cmp["map"], "uint8")
            written.append(p)
        elif name == "histogram":
            lo, hi, bins, id_ = float(arg(0)), float(arg(1)), int(arg(2)), arg(3, "")
            ids = f".{id_}" if id_ else ""
            titles = {"pos": "positive weight", "neg": "negative weight", "zero": "zero weight"}
            for fid, df in result.histogram(lo, hi, bins).items():
                p = directory / f"histogram.{fid}{ids}.txt"
                lines = [f"priority rank minimum: {_g(lo)}", f"priority rank maximum: {_g(hi)}",
                         f"{titles.get(fid, 'group ' + fid)} layers", "", '"bin" "coverage_level" "count"']
                for r in df.itertuples(index=False):
                    lines.append(f"{r.bin} {float(r.coverage_level):.4f} {_fmt(r.count)}")
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                written.append(p)
        elif name == "flexibility_map":
            err, bidir, up, down, id_ = float(arg(0)), bool(arg(1)), bool(arg(2)), bool(arg(3)), arg(4, "")
            f = result.flexibility_map(err, up=up or bidir, down=down or bidir)
            ids = f".{id_}" if id_ else ""
            for key, suffix, on in (("up", ".up", up), ("down", ".down", down), ("both", "", bidir)):
                if on:
                    p = directory / f"flexibilitymap{ids}{suffix}.tif"
                    v = f["up"] + f["down"] if key == "both" and "both" not in f else f[key]
                    _write_tif(p, v.fillna(-np.inf) if hasattr(v, "fillna") else np.where(np.isnan(v), -np.inf, v),
                               "float32")
                    written.append(p)
        else:
            import warnings

            warnings.warn(f"ignoring unknown post-processing call {name!r}", stacklevel=3)
    return written
