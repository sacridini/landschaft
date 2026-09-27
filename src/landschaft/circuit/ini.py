# SPDX-License-Identifier: GPL-3.0-or-later
"""Omniscape's INI files: ``run_omniscape(path)`` with landschaft's solver."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .omniscape import omniscape, read_reclass_table

TRUELIST = ("true", "True", "TRUE")
REQUIRED = ("resistance_file", "radius", "project_name")
SUPPORTED = (
    "resistance_file", "resistance_is_conductance", "source_file", "project_name", "parallelize",
    "parallel_batch_size", "block_size", "radius", "buffer", "source_threshold", "source_from_resistance",
    "r_cutoff", "precision", "connect_four_neighbors_only", "solver", "calc_flow_potential",
    "calc_normalized_current", "correct_artifacts", "write_raw_currmap", "write_as_tif", "mask_nodata",
    "suppress_cs_messages", "conditional", "n_conditions", "compare_to_future", "condition1_file",
    "condition2_file", "condition1_future_file", "condition2_future_file", "comparison1", "comparison2",
    "condition1_lower", "condition2_lower", "condition1_upper", "condition2_upper", "reclassify_resistance",
    "reclass_table", "write_reclassified_resistance", "allow_different_projections",
)
DEFAULTS = {
    "resistance_file": "", "resistance_is_conductance": "false", "source_file": "",
    "parallelize": "true", "parallel_batch_size": "10", "block_size": "1", "buffer": "0",
    "source_threshold": "0", "source_from_resistance": "false", "r_cutoff": "Inf", "precision": "double",
    "connect_four_neighbors_only": "false", "solver": "cg+amg", "calc_flow_potential": "false",
    "calc_normalized_current": "false", "correct_artifacts": "true", "write_raw_currmap": "true",
    "write_as_tif": "true", "mask_nodata": "true", "suppress_cs_messages": "true", "conditional": "false",
    "n_conditions": "1", "compare_to_future": "none", "condition1_file": "", "condition2_file": "",
    "condition1_future_file": "", "condition2_future_file": "", "comparison1": "within", "comparison2": "within",
    "condition1_lower": "0", "condition2_lower": "0", "condition1_upper": "0", "condition2_upper": "0",
    "reclassify_resistance": "false", "reclass_table": "", "write_reclassified_resistance": "false",
    "allow_different_projections": "false",
}


def parse_ini(path) -> dict[str, str]:
    """Omniscape's ``parse_cfg``: lines starting with ``[`` are skipped, every other one is
    split at its first ``=`` (key right-stripped, value stripped)."""
    cfg: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("["):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            cfg[key.rstrip()] = value.strip()
    return cfg


def _float(value: str) -> float:
    # Julia's parse(Float64, ...) accepts Inf / -Inf / NaN as well.
    return float(value.replace("Inf", "inf"))


def _resolve(name: str, base: Path) -> Path:
    # Omniscape reads paths from the working directory; failing that, next to the INI file.
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    q = base / p
    return q if q.exists() else p


def _read(path: Path, dtype):
    """A raster as Omniscape's ``read_raster``: band 1, nodata (in the band's type) and NaN missing."""
    import rasterio

    if not path.exists():
        raise FileNotFoundError(f'the file "{path}" does not exist')
    with rasterio.open(path) as ds:
        a = ds.read(1)
        nodata = ds.nodata
        profile = {"transform": ds.transform, "crs": ds.crs}
    band = a.dtype
    a = a.astype(dtype).astype(np.float64)
    if nodata is not None:
        a[a == np.array(nodata).astype(band).astype(dtype).astype(np.float64)] = np.nan
    return a, profile


def _aligned(ref, other, name1, name2, allow_projections: bool) -> None:
    (a, pa), (b, pb) = ref, other
    if pa["crs"] != pb["crs"] or pa["transform"] != pb["transform"]:
        if not allow_projections:
            warnings.warn(f"{name1} and {name2} have different projections. This could mean they are not properly "
                          "aligned. Attempting to proceed anyway.", stacklevel=3)
    if a.shape != b.shape:
        raise ValueError(f"{name1} and {name2} are different sizes.")


def _project_dir(name: str) -> Path:
    base = Path(name)
    if not base.is_dir():
        base.mkdir(parents=True, exist_ok=True)
        return base
    k = 1
    while Path(f"{name}_{k}").is_dir():
        k += 1
    out = Path(f"{name}_{k}")
    warnings.warn(f"Your specified project directory, {name}, already exists. Writing outputs to {out}.",
                  stacklevel=3)
    out.mkdir(parents=True)
    return out


def _write(path: Path, values: np.ndarray, profile: dict, tif: bool, dtype) -> None:
    import rasterio

    out = np.where(np.isnan(values), -9999, values).astype(dtype)
    ny, nx = out.shape
    if tif:
        opts = {"driver": "GTiff", "compress": "lzw", "BIGTIFF": "YES"}
        target = path.with_suffix(".tif")
    else:
        opts = {"driver": "AAIGrid"}
        target = path.with_suffix(".asc")
    with rasterio.open(target, "w", width=nx, height=ny, count=1, dtype=out.dtype, nodata=-9999,
                       transform=profile["transform"], crs=profile["crs"], **opts) as ds:
        ds.write(out, 1)


def omniscape_ini(path, *, write_outputs: bool = True, n_threads: int = 0):
    """Run an Omniscape INI file, as ``run_omniscape(path)`` does.

    Every option of Omniscape 0.7 is read with its defaults (``radius``,
    ``resistance_file`` and ``project_name`` are required); file paths are
    taken from the working directory, or else next to the INI file. With
    ``write_outputs`` the outputs go to ``project_name`` (``_1``, ``_2``…
    when it exists) as Omniscape writes them: ``config.ini``, ``cum_currmap``,
    ``flow_potential``, ``normalized_cum_currmap`` and
    ``classified_resistance``, GeoTIFF (``write_as_tif``) or ASCII grids,
    NoData -9999. ``solver``, ``parallelize``, ``parallel_batch_size`` and
    ``suppress_cs_messages`` have no effect: windows are always solved exactly,
    with ``n_threads`` threads (0: all cores but one).

    Returns
    -------
    xarray.Dataset
        As :func:`landschaft.omniscape`, plus ``classified_resistance`` when
        the resistance was reclassified.
    """
    path = Path(path)
    base = path.resolve().parent
    user = parse_ini(path)
    missing = [k for k in REQUIRED if k not in user]
    if missing:
        raise ValueError("The following arguments are missing from your .ini with no defaults: " + " ".join(missing))
    unknown = [k for k in user if k not in SUPPORTED and k != ""]
    if unknown:
        warnings.warn("The following unsupported arguments were provided and will be ignored: " + " ".join(unknown),
                      stacklevel=2)
    cfg = {**DEFAULTS, **user}
    flag = {k: cfg[k] in TRUELIST for k in cfg}
    precision = "single" if cfg["precision"] in ("single", "Single", "SINGLE") else "double"
    dtype = np.float32 if precision == "single" else np.float64
    allow = flag["allow_different_projections"]

    res = _read(_resolve(cfg["resistance_file"], base), dtype)
    resistance = res[0]
    reclass = None
    if flag["reclassify_resistance"]:
        reclass = read_reclass_table(_resolve(cfg["reclass_table"], base))
    elif cfg["reclass_table"] != "":
        warnings.warn("You provided a reclass_table, but did not set reclassify_resistance to true. Resistance will "
                      "not be reclassified.", stacklevel=2)

    source = None
    if not flag["source_from_resistance"]:
        if cfg["source_file"] == "":
            raise ValueError("You did not provide a source raster file path. Set source_from_resistance to true in "
                             "your config if you want to generate source strength from the resistance layer.")
        src = _read(_resolve(cfg["source_file"], base), dtype)
        _aligned(res, src, "resistance_file", "source_file", allow)
        source = src[0]

    conds: dict = {}
    if flag["conditional"]:
        compare = cfg["compare_to_future"].lower()
        n_cond = int(round(_float(cfg["n_conditions"])))

        def cond(key):
            c = _read(_resolve(cfg[key], base), dtype)
            _aligned(res, c, "resistance_file", key, allow)
            return c[0]

        conds["condition1"] = cond("condition1_file")
        if compare in ("1", "both"):
            conds["condition1_future"] = cond("condition1_future_file")
        if n_cond == 2:
            conds["condition2"] = cond("condition2_file")
            if compare in ("2", "both"):
                conds["condition2_future"] = cond("condition2_future_file")
        elif n_cond != 1:
            raise ValueError(f"n_conditions must be one of ['1', '2']. Got {cfg['n_conditions']}.")
        conds.update(compare_to_future=compare, comparison1=cfg["comparison1"], comparison2=cfg["comparison2"],
                     condition1_lower=_float(cfg["condition1_lower"]), condition1_upper=_float(cfg["condition1_upper"]),
                     condition2_lower=_float(cfg["condition2_lower"]), condition2_upper=_float(cfg["condition2_upper"]))
    elif cfg["condition1_file"] != "" or cfg["condition2_file"] != "":
        warnings.warn("conditional is set to false in your INI but you provided file paths to condition files.",
                      stacklevel=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # block_size: warned below, once
        ds = omniscape(
            resistance, _float(cfg["radius"]), source=source, block_size=_float(cfg["block_size"]),
            buffer=_float(cfg["buffer"]), source_threshold=_float(cfg["source_threshold"]),
            r_cutoff=_float(cfg["r_cutoff"]), resistance_is_conductance=flag["resistance_is_conductance"],
            four_neighbors=flag["connect_four_neighbors_only"], flow_potential=flag["calc_flow_potential"],
            normalized_current=flag["calc_normalized_current"], correct_artifacts=flag["correct_artifacts"],
            mask_nodata=flag["mask_nodata"], reclass_table=reclass, precision=precision, n_threads=n_threads,
            **conds)
    if int(round(_float(cfg["block_size"]))) % 2 == 0:
        warnings.warn("block_size is even, but must be odd. Using block_size + 1.", stacklevel=2)
    if reclass is not None:
        from .omniscape import reclassify

        classified = reclassify(resistance, reclass).astype(dtype)
        ds["classified_resistance"] = (("y", "x"), classified)

    if write_outputs:
        out = _project_dir(cfg["project_name"])
        with open(out / "config.ini", "a", encoding="utf-8") as f:
            for k, v in user.items():
                if k:
                    f.write(f"{k} = {v}\n")
        tif = flag["write_as_tif"]
        profile = res[1]
        if reclass is not None and flag["write_reclassified_resistance"]:
            _write(out / "classified_resistance", ds["classified_resistance"].values, profile, tif, dtype)
        if flag["write_raw_currmap"]:
            _write(out / "cum_currmap", ds["cum_currmap"].values, profile, tif, dtype)
        if flag["calc_flow_potential"]:
            _write(out / "flow_potential", ds["flow_potential"].values, profile, tif, dtype)
        if flag["calc_normalized_current"]:
            _write(out / "normalized_cum_currmap", ds["normalized_cum_currmap"].values, profile, tif, dtype)
        ds.attrs["project_dir"] = os.fspath(out)
    return ds
