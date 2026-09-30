# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5's output tables (main.cpp, priorityrank.cpp CurveInfo) from the curves of a ranking.

The means are Zonation's float32 sums (Eigen's AVX reduction order, through
``_core.prio_eigen_sums``), so the tables match its CSV files digit for digit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core


def _curve_info(levels: np.ndarray, weights: np.ndarray, idx: list[int]) -> dict[str, np.ndarray]:
    """CurveInfo::read of the layers ``idx``: mean, weighted mean, max, min per sample."""
    # Zonation copies the values into a fresh (aligned) array
    sub = np.ascontiguousarray(levels[:, idx], dtype=np.float32)
    every = np.arange(len(idx))
    n = np.float32(len(idx))
    w = np.ascontiguousarray(weights[idx], dtype=np.float32)
    wsum = _core.prio_eigen_sums(w[None, :], every)[0]
    mean = (_core.prio_eigen_sums(sub, every) / n).astype(np.float32)
    if wsum != 0:
        prod = np.ascontiguousarray(sub * w[None, :], dtype=np.float32)
        wmean = (_core.prio_eigen_sums(prod, every) / np.float32(wsum)).astype(np.float32)
    else:
        wmean = mean
    return {"mean": mean, "weighted_mean": wmean, "max": sub.max(axis=1), "min": sub.min(axis=1)}


def weight_classes(weights: np.ndarray, weights_on: bool) -> dict[str, list[int]]:
    """Layers with positive, negative and zero weights (all positive without weights)."""
    if not weights_on:
        return {"positive": list(range(weights.size)), "negative": [], "zero": []}
    return {"positive": [i for i, w in enumerate(weights) if w > 0],
            "negative": [i for i, w in enumerate(weights) if w < 0],
            "zero": [i for i, w in enumerate(weights) if w == 0]}


def feature_curves(rank: np.ndarray, levels: np.ndarray, names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(levels.astype(np.float32), columns=names)
    df.insert(0, "rank", rank.astype(np.float32))
    return df


def summary_curves(rank, levels, cost, weights, classes, cells: int) -> pd.DataFrame:
    rank = rank.astype(np.float32)
    levels = np.ascontiguousarray(levels, dtype=np.float32)
    L = levels.shape[1]
    area = ((np.float32(1) - rank) * np.float32(cells)).astype(np.float32)
    cols = {
        "rank": rank,
        "remaining_area": np.trunc(area.astype(np.float64)).astype(np.int64),
        "remaining_cost": np.asarray(cost, dtype=np.float64),
        "mean": (_core.prio_eigen_sums(levels) / np.float32(L)).astype(np.float32),
        "max": levels.max(axis=1),
        "min": levels.min(axis=1),
    }
    for key, suffix in (("positive", "positive"), ("zero", "zero"), ("negative", "negative")):
        idx = classes[key]
        if not idx:
            continue
        ci = _curve_info(levels, weights, idx)
        cols[f"mean_{suffix}"] = ci["mean"]
        if key != "zero":
            cols[f"weighted_mean_{suffix}"] = ci["weighted_mean"]
        cols[f"max_{suffix}"] = ci["max"]
        cols[f"min_{suffix}"] = ci["min"]
    return pd.DataFrame(cols)


def density_curves(rank, levels, weights, classes) -> pd.DataFrame:
    rank = rank.astype(np.float32)
    cols = {"rank": rank}
    for key, suffix in (("positive", "pos"), ("negative", "neg"), ("zero", "zero")):
        idx = classes[key]
        if not idx:
            continue
        ci = _curve_info(np.ascontiguousarray(levels, dtype=np.float32), weights, idx)
        r = np.zeros_like(rank)
        r[1:] = rank[1:] - rank[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fd = np.zeros(rank.size, dtype=np.float32)
            fd[1:] = (ci["mean"][:-1] - ci["mean"][1:]) / r[1:]
            wfd = np.zeros(rank.size, dtype=np.float32)
            wfd[1:] = (ci["weighted_mean"][:-1] - ci["weighted_mean"][1:]) / r[1:]
        cols[f"feature_density_{suffix}"] = fd
        if key != "zero":
            cols[f"weighted_feature_density_{suffix}"] = wfd
    return pd.DataFrame(cols)


def group_curves(rank, levels, weights, groups: np.ndarray | None) -> pd.DataFrame | None:
    if groups is None:
        return None
    ids = sorted({int(g) for g in groups if g >= 0})
    if not ids:
        return None
    cols = {"rank": rank.astype(np.float32)}
    lv = np.ascontiguousarray(levels, dtype=np.float32)
    for g in ids:
        ci = _curve_info(lv, weights, [i for i, x in enumerate(groups) if x == g])
        cols[f"mean.{g}"] = ci["mean"]
        cols[f"weighted_mean.{g}"] = ci["weighted_mean"]
        cols[f"min.{g}"] = ci["min"]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- Zonation's text files

def _g(x: float) -> str:
    """Zonation's {} of a double (its fmt: like %g, with ".0" when it looks like an integer)."""
    s = f"{float(x):g}"
    if not any(c in s for c in ".eninf"):
        s += ".0"
    return s


def format_feature_curves(df: pd.DataFrame, headers: list[str]) -> str:
    lines = ['"rank" ' + "".join(f"{h} " for h in headers)]
    vals = df.to_numpy()
    for row in vals:
        lines.append(f"{float(np.float32(row[0])):.5f} " + "".join(f"{float(np.float32(v)):.4f} " for v in row[1:]))
    return "\n".join(lines) + "\n"


def format_summary_curves(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = ['"rank"', '"remaining_area"', '"remaining_cost"', '"mean"', ' "max"', ' "min"']
    text = " ".join(head[:4]) + "".join(head[4:])
    extra = [c for c in cols if c not in ("rank", "remaining_area", "remaining_cost", "mean", "max", "min")]
    text += "".join(f' "{c}"' for c in extra) + "\n"
    out = [text]
    for _, r in df.iterrows():
        s = (f"{float(np.float32(r['rank'])):.5f} {int(r['remaining_area']):d} {_g(r['remaining_cost'])} "
             f"{float(np.float32(r['mean'])):.4f}")
        s += f" {float(np.float32(r['max'])):.4f} {float(np.float32(r['min'])):.4f}"
        for c in extra:
            s += f" {float(np.float32(r[c])):.4f}"
        out.append(s + "\n")
    return "".join(out)


def format_density_curves(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "rank"]
    out = ['"rank" ' + "".join(f'"{c}" ' for c in cols) + "\n"]
    for row in df.to_numpy():
        out.append(f"{float(np.float32(row[0])):.5f}" + "".join(f" {float(np.float32(v)):.4f}" for v in row[1:])
                   + "\n")
    return "".join(out)


def format_group_curves(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "rank"]
    groups = [c.split(".", 1)[1] for c in cols if c.startswith("mean.")]
    head = '"rank" ' + "".join(f'"mean.{g}" "weighted_mean.{g}" "min.{g}" ' for g in groups)
    out = [head + "\n"]
    for row in df.to_numpy():
        out.append(f"{float(np.float32(row[0])):.5f} " + "".join(f"{float(np.float32(v)):.4f} " for v in row[1:])
                   + "\n")
    return "".join(out)
