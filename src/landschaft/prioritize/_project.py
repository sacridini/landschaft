# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5 projects: the settings file (``.z5``), its feature list and link files.

``read_project`` turns a project and the switches of the z5 command line into the
arguments of :func:`landschaft.prioritize.zonation`; ``run_project`` runs it.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import numpy as np

# feature list columns
_COLUMNS = ("group", "weight", "wgrp", "tr_out", "threshold", "condition", "retention", "sf_conn", "ia_conn1",
            "ia_conn2", "matrix_conn", "abfz", "target", "steepness", "keep_zeros", "impact_grp", "impact_bins",
            "filename", "name")
_INT = {"group", "wgrp", "tr_out", "condition", "retention", "sf_conn", "ia_conn1", "ia_conn2", "matrix_conn",
        "keep_zeros", "impact_grp", "impact_bins"}


def _rows(path: Path) -> list[list[str]]:
    """Whitespace-separated fields ("quoted" allowed), without # comments and empty lines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(shlex.split(line, posix=True))
    return rows


def _settings(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        m = re.fullmatch(r"['\"]?(.*?)['\"]?", v)
        out[k.strip().lower()] = m.group(1) if m else v
    return out


def _resolve(name: str, base: Path) -> Path:
    p = Path(name)
    return p if p.is_absolute() else base / p


def _flags(flags) -> tuple[set[str], int]:
    if isinstance(flags, str):
        flags = flags.split()
    on: set[str] = set()
    area = 0
    for f in flags or ():
        if f.startswith("--area="):
            area = int(f.split("=", 1)[1])
        elif f.startswith("--"):
            continue
        elif f.startswith("-"):
            on.update(f[1:])
    if "W" in on:
        on.add("w")
    if "X" in on:
        on.add("x")
    return on, area


_KERNEL = re.compile(r"\s*([A-Za-z]+)\s*\(([^)]*)\)\s*")


def _kernel_links(path: Path, kind: str) -> dict:
    """A connectivity link file: id [options] kernel(parameters) [file] per line."""
    from ._connectivity import Interaction, Kernel

    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KERNEL.search(line)
        if not m:
            raise ValueError(f"{path}: no kernel in {line!r}")
        head = line[:m.start()].split()
        tail = shlex.split(line[m.end():])
        params = [float(x) for x in m.group(2).split(",") if x.strip()]
        shape = m.group(1).lower()
        kern = Kernel(shape, params[0], params[1] if len(params) > 1 else 1.0)
        kid = int(float(head[0]))
        if kind == "sf":
            out[kid] = (kern, bool(int(float(head[1]))) if len(head) > 1 else False)
        elif kind == "ia":
            positive, beta, normalize = bool(int(float(head[1]))), float(head[2]), bool(int(float(head[3])))
            out[kid] = Interaction(kern, _resolve(tail[0], path.parent), positive, beta, normalize)
        else:
            out[kid] = (kern, str(_resolve(tail[0], path.parent)))
    return out


def read_project(settings, flags="", *, mode: str | None = None) -> dict:
    """The :func:`zonation` arguments of a Zonation 5 project.

    Parameters
    ----------
    settings
        The project's settings file (``.z5``).
    flags
        The z5 command-line switches that enable its parts, as a string or list:
        ``-w`` weights, ``-W`` weight groups, ``-g`` output groups, ``-t``
        thresholds, ``-c`` conditions, ``-r`` retention, ``-x``/``-X`` cost
        (tracked / in the ranking), ``-h`` hierarchic mask, ``-a`` analysis area
        (``--area=N``), ``-b`` pre-processing maps, ``-f`` transformed layers.
    mode
        The marginal loss rule (z5's ``--mode``), e.g. ``"CAZ2"`` or ``"ABF 0.25"``.

    Returns
    -------
    dict
        Keyword arguments of :func:`zonation` (``features`` included).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np, rasterio
    >>> import landschaft as ls
    >>> d = Path(tempfile.mkdtemp())
    >>> for i in range(2):
    ...     with rasterio.open(d / f"sp{i}.tif", "w", driver="GTiff", height=5, width=6, count=1,
    ...                        dtype="float32", crs="EPSG:32633",
    ...                        transform=rasterio.transform.from_origin(0, 500, 100, 100)) as f:
    ...         f.write(np.random.default_rng(i).random((5, 6), dtype=np.float32), 1)
    >>> _ = (d / "features.txt").write_text("weight filename\\n2.0 sp0.tif\\n1.0 sp1.tif\\n")
    >>> _ = (d / "settings.z5").write_text("feature list file = features.txt\\n")
    >>> kw = ls.prioritize.read_project(d / "settings.z5", "-w", mode="ABF")
    >>> kw["weights"], kw["rule"]
    ([2.0, 1.0], 'ABF')
    """
    path = Path(settings)
    base = path.parent
    st = _settings(path)
    on, area = _flags(flags)
    fl_name = st.get("feature list file")
    if not fl_name:
        raise ValueError(f"{path}: no 'feature list file'")
    fl = _resolve(fl_name, base)
    rows = _rows(fl)
    header = [h.lower() for h in rows[0]]
    if "filename" not in header:
        raise ValueError(f"{fl}: no 'filename' column")
    cols: dict[str, list] = {h: [] for h in header}
    for r in rows[1:]:
        if len(r) != len(header):
            raise ValueError(f"{fl}: a line with {len(r)} fields for {len(header)} columns")
        for h, v in zip(header, r):
            cols[h].append(v)
    files, extract = [], []
    for name in cols["filename"]:
        p = _resolve(name, fl.parent)
        ev = None
        if not p.exists() and p.parent.is_file():  # "file.tif/3": value extraction
            ev = float(p.name)
            p = p.parent
        files.append(p)
        extract.append(ev)

    def num(col, cast=float):
        return [cast(int(float(v))) if cast is int else cast(v) for v in cols[col]] if col in cols else None

    kw: dict = {"features": files}
    if any(e is not None for e in extract):
        kw["extract"] = extract
    if "name" in cols:
        kw["names"] = cols["name"]
    if mode:
        kw["rule"] = mode
    if "w" in on:
        kw["weights"] = num("weight") or [1.0] * len(files)
    if "W" in on:
        kw["weight_groups"] = num("wgrp", int)
        gfile = st.get("weight groups file")
        if gfile:
            kw["group_weights"] = {int(float(r[0])): float(r[1]) for r in _rows(_resolve(gfile, base))}
    if "g" in on and "group" in cols:
        kw["groups"] = num("group", int)
    if "t" in on and "threshold" in cols:
        kw["thresholds"] = num("threshold")
    if "abfz" in cols:
        kw["abf_z"] = num("abfz")
    zm = st.get("zero mode")
    if zm:
        kw["zero_mode"] = zm.lower()
    if "keep_zeros" in cols:
        kw["keep_zeros"] = num("keep_zeros", int)
    if "c" in on and st.get("condition link file") and "condition" in cols:
        links = {int(float(r[0])): (bool(int(float(r[1]))), _resolve(r[2], _resolve(st["condition link file"],
                 base).parent)) for r in _rows(_resolve(st["condition link file"], base))}
        ids = num("condition", int)
        kw["condition"] = [links[i][1] if i in links else None for i in ids]
        kw["renormalize"] = [links[i][0] if i in links else False for i in ids]
    if "r" in on and st.get("retention link file") and "retention" in cols:
        rfile = _resolve(st["retention link file"], base)
        links = {}
        for r in _rows(rfile):
            rid, kind = int(float(r[0])), int(float(r[1]))
            if kind == 0:
                links[rid] = ("absolute", _resolve(r[2], rfile.parent))
            elif kind == 1:
                links[rid] = ("fraction", _resolve(r[2], rfile.parent))
            else:
                links[rid] = float(r[2])
        kw["retention"] = [links.get(i) for i in num("retention", int)]
    if "d" in on and st.get("single-feature connectivity link file") and "sf_conn" in cols:
        links = _kernel_links(_resolve(st["single-feature connectivity link file"], base), "sf")
        ids = num("sf_conn", int)
        kw["connectivity"] = [links[i][0] if i in links else None for i in ids]
        kw["self_multiply"] = [links[i][1] if i in links else False for i in ids]
    for flag, col, key in (("i", "ia_conn1", "interaction"), ("I", "ia_conn2", "interaction2")):
        if flag in on and st.get("interaction connectivity link file") and col in cols:
            links = _kernel_links(_resolve(st["interaction connectivity link file"], base), "ia")
            kw[key] = [links.get(i) for i in num(col, int)]
    if "m" in on and st.get("matrix connectivity link file") and "matrix_conn" in cols:
        links = _kernel_links(_resolve(st["matrix connectivity link file"], base), "m")
        ids = num("matrix_conn", int)
        kw["matrix_connectivity"] = [links[i] if i in links else None for i in ids]
        kw["matrices"] = {}
        for i in ids:
            if i in links:
                f = links[i][1]
                n = sum(1 for j in ids if j in links and links[j][1] == f)
                kw["matrices"][f] = np.array([[float(v) for v in r] for r in _rows(Path(f))][:n], dtype=np.float32)
    if "x" in on and st.get("cost layer"):
        kw["cost"] = _resolve(st["cost layer"], base)
        kw["cost_in_ranking"] = "X" in on
    if "h" in on and st.get("hierarchic mask layer"):
        kw["hierarchic_mask"] = _resolve(st["hierarchic mask layer"], base)
    if "a" in on and st.get("analysis area mask layer"):
        kw["analysis_area"] = _resolve(st["analysis area mask layer"], base)
        kw["area"] = area
    if st.get("external solution") and (mode or "").split()[:1] == ["LOAD"]:
        kw["solution"] = _resolve(st["external solution"], base)
    if "b" in on:
        kw["preprocessing"] = True
    if "f" in on and "tr_out" in cols:
        kw["transformed"] = [bool(v) for v in num("tr_out", int)]
    for key, arg, cast in (("max iterations", "max_iterations", int),
                           ("mean ranking error threshold", "ranking_error", float),
                           ("smoothing factor", "smoothing_factor", float),
                           ("curves output interval", "curves_interval", float)):
        if st.get(key):
            kw[arg] = cast(float(st[key])) if cast is int else cast(st[key])
    if st.get("stop on plateau"):
        kw["stop_on_plateau"] = st["stop on plateau"].strip().lower() in ("1", "true")
    return kw


def run_project(settings, flags="", *, mode: str | None = None, output=None, **overrides):
    """Run a Zonation 5 project (see :func:`read_project`).

    Parameters
    ----------
    settings, flags, mode
        As :func:`read_project`.
    output
        Directory to write Zonation's output files into (optional).
    **overrides
        Other :func:`zonation` arguments (``bits=16`` for ``z5_16bit``, ``n_threads``...).

    Returns
    -------
    ZonationResult

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np, rasterio
    >>> import landschaft as ls
    >>> d = Path(tempfile.mkdtemp())
    >>> for i in range(2):
    ...     with rasterio.open(d / f"sp{i}.tif", "w", driver="GTiff", height=5, width=6, count=1,
    ...                        dtype="float32", crs="EPSG:32633",
    ...                        transform=rasterio.transform.from_origin(0, 500, 100, 100)) as f:
    ...         f.write(np.random.default_rng(i).random((5, 6), dtype=np.float32), 1)
    >>> _ = (d / "features.txt").write_text("filename\\nsp0.tif\\nsp1.tif\\n")
    >>> _ = (d / "settings.z5").write_text("feature list file = features.txt\\n")
    >>> res = ls.prioritize.run_project(d / "settings.z5", mode="CAZ2")
    >>> res.rank.shape
    (5, 6)
    """
    from ._zonation import zonation

    kw = read_project(settings, flags, mode=mode)
    kw.update(overrides)
    res = zonation(kw.pop("features"), **kw)
    if output is not None:
        res.write(output)
    return res
