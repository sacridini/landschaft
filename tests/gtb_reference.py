# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the original GuidosToolbox ``mspa`` executable on numpy arrays.

The executable ships with GTB (``MSPAstandalone/mspa_win64.exe``,
``mspa_lin64``, ``mspa_mac``). Point ``LANDSCHAFT_MSPA_EXE`` to it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def find_executable() -> Path | None:
    exe = os.environ.get("LANDSCHAFT_MSPA_EXE")
    return Path(exe) if exe and Path(exe).is_file() else None


def run_original_mspa(
    exe: os.PathLike,
    data: np.ndarray,
    *,
    connectivity: int = 8,
    edge_width: int = 1,
    internal: bool = True,
    transition: bool = True,
    timeout: float = 600,
) -> np.ndarray:
    """Run ``mspa -disk`` exactly as GuidosToolbox does and return its output."""
    exe = Path(exe).resolve()
    # mspa writes temporary files in its working directory: isolate each run.
    with tempfile.TemporaryDirectory(prefix="gtb_mspa_") as tmp:
        tmp = Path(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(
                tmp / "in.tif", "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                count=1, dtype="uint8",
            ) as dst:
                dst.write(np.ascontiguousarray(data, dtype=np.uint8), 1)
            cmd = [
                str(exe), "-graphfg", str(connectivity), "-eew", str(edge_width),
                "-internal", str(int(internal)), "-disk", "-transition", str(int(transition)),
                "-i", "in.tif", "-o", "out.tif", "-odir", "./",
            ]
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=timeout)
            out_file = tmp / "out.tif"
            if proc.returncode != 0 or not out_file.exists():
                raise RuntimeError(f"mspa failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")
            with rasterio.open(out_file) as src:
                return src.read(1)


# --- GTB tools that exist only as IDL code: run them with GDL (or IDL) ---------

IDL_DIR = Path(__file__).parent / "gtb_idl"
DEFAULT_GDL = Path(r"C:\Program Files\gnudatalanguage\bin\gdl.exe")


def find_gdl() -> Path | None:
    """GDL/IDL interpreter from ``LANDSCHAFT_GDL_EXE``, PATH or the default install."""
    import shutil

    exe = os.environ.get("LANDSCHAFT_GDL_EXE")
    if exe:
        return Path(exe) if Path(exe).is_file() else None
    found = shutil.which("gdl")
    if found:
        return Path(found)
    return DEFAULT_GDL if DEFAULT_GDL.is_file() else None


def _idl_str(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def run_gdl(exe: os.PathLike, commands: list[str], *, timeout: float = 600) -> None:
    """Run IDL statements (one per line) in GDL, compiling the scripts of tests/gtb_idl."""
    script = [f"!path = {_idl_str(IDL_DIR)} + ';' + !path", *commands, "exit"]
    proc = subprocess.run(
        [str(exe), "-quiet"], input="\n".join(script) + "\n", capture_output=True, text=True, timeout=timeout
    )
    # GDL ignores HISTOGRAM's /L64 (32-bit counts suffice for test images).
    # IDL reports 0/0 = NaN as an arithmetic error and carries on (as GTB does).
    benign = ("Compiled module", "Keyword parameter L64 not supported", "Program caused arithmetic error")
    errors = [line for line in (proc.stdout + proc.stderr).splitlines()
              if line.lstrip().startswith("%") and not any(b in line for b in benign)]
    if proc.returncode != 0 or errors:
        raise RuntimeError(f"GDL failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")


def _read_stats(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def run_gtb_accounting(
    exe: os.PathLike,
    data: np.ndarray,
    thresholds=(50, 250, 1000, 5000, 20000),
    *,
    connectivity: int = 8,
    highlight_largest: bool = False,
) -> dict:
    """Run GTB's Accounting IDL code (``tests/gtb_idl/gtb_accounting.pro``).

    Returns ``classes``, ``ids`` and ``area`` arrays and the report numbers
    as strings under ``stats``.
    """
    cls = list(thresholds) + [0] * (5 - len(thresholds))  # unused GTB slots are dropped (< 1)
    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_acc_") as tmp:
        tmp = Path(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(tmp / "in.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                               dtype="uint8") as dst:
                dst.write(np.ascontiguousarray(data, dtype=np.uint8), 1)
        args = ", ".join(str(int(c)) for c in cls)
        run_gdl(exe, [
            f"gtb_accounting, {_idl_str(tmp / 'in.tif')}, {_idl_str(tmp)} + '/', "
            f"{int(connectivity == 8)}, {args}, {int(highlight_largest)}"
        ])
        return {
            "classes": np.fromfile(tmp / "classes.raw", dtype=np.uint8).reshape(ny, nx),
            "ids": np.fromfile(tmp / "ids.raw", dtype="<i4").reshape(ny, nx),
            "area": np.fromfile(tmp / "pixels.raw", dtype="<i4").reshape(ny, nx),
            "stats": _read_stats(tmp / "stats.txt"),
        }


def run_gtb_accounting_scaled_classes(exe: os.PathLike, data: np.ndarray) -> list[int]:
    """Size classes proposed by GTB's interactive Accounting dialog (before dropping repeats)."""
    with tempfile.TemporaryDirectory(prefix="gtb_acc_") as tmp:
        tmp = Path(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(tmp / "in.tif", "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                               count=1, dtype="uint8") as dst:
                dst.write(np.ascontiguousarray(data, dtype=np.uint8), 1)
        run_gdl(exe, [f"gtb_accounting_scaled_classes, {_idl_str(tmp / 'in.tif')}, {_idl_str(tmp / 'cl.txt')}"])
        return [int(v) for v in (tmp / "cl.txt").read_text().split()]


DEFAULT_SPATCON = Path(r"C:\Users\eduar\Documents\dev\GTB\external_sources\spatcon64.exe")


def find_spatcon() -> Path | None:
    """GTB's spatcon executable from ``LANDSCHAFT_SPATCON_EXE`` or the default GTB checkout."""
    exe = os.environ.get("LANDSCHAFT_SPATCON_EXE")
    path = Path(exe) if exe else DEFAULT_SPATCON
    return path if path.is_file() else None


def _write_tif(path: Path, data: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                           dtype="uint8") as dst:
            dst.write(np.ascontiguousarray(data, dtype=np.uint8), 1)


FAD_SCALES = (7, 13, 27, 81, 243)


def _fad_mscale_into(exe, spatcon, data: np.ndarray, root: Path, name: str) -> Path:
    """Run gtb_fad_mscale.pro on ``data``; outputs go to ``root/name``."""
    import shutil

    if not (root / "spatcon" / "spatcon64.exe").exists():
        (root / "spatcon").mkdir(exist_ok=True)
        shutil.copy(spatcon, root / "spatcon" / "spatcon64.exe")
        (root / "work").mkdir(exist_ok=True)
    out = root / name
    out.mkdir()
    _write_tif(root / f"{name}.tif", data)
    run_gdl(exe, [
        f"gtb_fad_mscale, {_idl_str(root / f'{name}.tif')}, {_idl_str(root / 'work')}, "
        f"{_idl_str(out)} + '/', 'windows'"
    ])
    return out


def run_gtb_fad_mscale(exe: os.PathLike, spatcon: os.PathLike, data: np.ndarray) -> dict:
    """Run GTB's FAD-MS IDL code (``tests/gtb_idl/gtb_fad_mscale.pro``) with GTB's spatcon.

    Returns ``maps`` ({7: ..., 243: ..., "mscale": ...}), the float32 arrays
    ``fad_av`` and ``fadru_av`` (5 scales + mscale) and ``stats`` strings.
    """
    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_fadms_") as tmp:
        out = _fad_mscale_into(exe, spatcon, data, Path(tmp), "a")
        maps = {k: np.fromfile(out / f"fad_{k}.raw", dtype=np.uint8).reshape(ny, nx)
                for k in (*FAD_SCALES, "mscale")}
        av = np.fromfile(out / "fad_av.raw", dtype="<f4")
        return {"maps": maps, "fad_av": av[:6], "fadru_av": av[6:], "stats": _read_stats(out / "stats.txt")}


def run_gtb_fad_change(exe: os.PathLike, spatcon: os.PathLike, data_a: np.ndarray, data_b: np.ndarray) -> dict:
    """GTB's FAD-MS of maps A and B, then its FAD Multiscale change (``gtb_fad_change.pro``).

    Returns per scale (7, 13, 27, 81, 243, mscale): ``change`` (6, 7, 7) uint64
    [A class, B class], ``perc`` (6, 6, 6), ``sums`` (6, 8) = sum_above,
    sum_below, farea_a, farea_b, nc, forcom, diag, chpix, and the float32
    ``delta_fad_av`` / ``delta_avcon``.
    """
    ny, nx = data_a.shape
    with tempfile.TemporaryDirectory(prefix="gtb_fadch_") as tmp:
        root = Path(tmp)
        dir_a = _fad_mscale_into(exe, spatcon, data_a, root, "a")
        dir_b = _fad_mscale_into(exe, spatcon, data_b, root, "b")
        out = root / "change"
        out.mkdir()
        (out / "size.txt").write_text(f"{nx} {ny}\n")
        run_gdl(exe, [f"gtb_fad_change, {_idl_str(dir_a)} + '/', {_idl_str(dir_b)} + '/', {_idl_str(out)} + '/'"])
        delta = np.fromfile(out / "delta.raw", dtype="<f4")
        # IDL arrays are column-major: [x, y, scale] -> numpy [scale, y, x]
        return {
            "change": np.fromfile(out / "change.raw", dtype="<u8").reshape(6, 7, 7),
            "perc": np.fromfile(out / "perc.raw", dtype="<f8").reshape(6, 6, 6),
            "sums": np.fromfile(out / "sums.raw", dtype="<f8").reshape(6, 8),
            "delta_fad_av": delta[:6],
            "delta_avcon": delta[6:],
        }


def run_gtb_lm_heatmap(exe: os.PathLike, spatcon: os.PathLike, data: np.ndarray, window: int) -> dict:
    """GTB's Landscape Mosaic 103-class map and heatmap (``tests/gtb_idl/gtb_lm_heatmap.pro``).

    Returns ``im103``, the float32 heatmap vector ``hn`` (256), ``hmap`` (12, 21)
    and the text of ``heatmap.csv``.
    """
    import shutil

    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_lm_") as tmp:
        root = Path(tmp)
        (root / "spatcon").mkdir()
        shutil.copy(spatcon, root / "spatcon" / "spatcon64.exe")
        (root / "work").mkdir()
        (root / "out").mkdir()
        _write_tif(root / "in.tif", data)
        run_gdl(exe, [f"gtb_lm_heatmap, {_idl_str(root / 'in.tif')}, {_idl_str(root / 'work')}, "
                      f"{_idl_str(root / 'out')} + '/', {int(window)}"])
        out = root / "out"
        return {
            "im103": np.fromfile(out / "im103.raw", dtype=np.uint8).reshape(ny, nx),
            "hn": np.fromfile(out / "hn.raw", dtype="<f4"),
            "hmap": np.fromfile(out / "hmap.raw", dtype="<f4").reshape(12, 21),
            "csv": (out / "heatmap.csv").read_text(),
        }


def run_gtb_lm_delta(exe: os.PathLike, hn_a: np.ndarray, hn_b: np.ndarray, kdim_a: str, kdim_b: str) -> dict:
    """GTB's heatmap change B - A (``tests/gtb_idl/gtb_lm_delta.pro``): ``hn`` and ``delta_heatmap.csv``."""
    with tempfile.TemporaryDirectory(prefix="gtb_lmd_") as tmp:
        root = Path(tmp)
        for name, hn in (("a", hn_a), ("b", hn_b)):
            (root / name).mkdir()
            np.asarray(hn, dtype="<f4").tofile(root / name / "hn.raw")
        (root / "out").mkdir()
        run_gdl(exe, [f"gtb_lm_delta, {_idl_str(root / 'a')} + '/', {_idl_str(root / 'b')} + '/', "
                      f"'{kdim_a}', '{kdim_b}', {_idl_str(root / 'out')} + '/'"])
        return {"hn": np.fromfile(root / "out" / "hn.raw", dtype="<f4"),
                "csv": (root / "out" / "delta_heatmap.csv").read_text()}


DEFAULT_GGEO = Path(r"C:\GuidosToolbox\guidos_progs\spatcon\ggeo64.exe")


def find_ggeo() -> Path | None:
    """GTB's ggeo executable from ``LANDSCHAFT_GGEO_EXE`` or a GTB desktop installation."""
    exe = os.environ.get("LANDSCHAFT_GGEO_EXE")
    path = Path(exe) if exe else DEFAULT_GGEO
    return path if path.is_file() else None


def run_gtb_isochrone(exe: os.PathLike, ggeo: os.PathLike, resist: np.ndarray, marker: np.ndarray,
                      mode: str = "cost_map_aa") -> np.ndarray:
    """GTB's isochrone map A (``cost_map_aa``) or AB (``cost_map_ab``) via ``tests/gtb_idl/gtb_isochrone.pro``."""
    import shutil

    ny, nx = resist.shape
    with tempfile.TemporaryDirectory(prefix="gtb_iso_") as tmp:
        root = Path(tmp)
        (root / "spatcon").mkdir()
        shutil.copy(ggeo, root / "spatcon" / "ggeo64.exe")
        (root / "work").mkdir()
        _write_tif(root / "resist.tif", resist)
        _write_tif(root / "marker.tif", marker)
        run_gdl(exe, [f"gtb_isochrone, {_idl_str(root / 'resist.tif')}, {_idl_str(root / 'marker.tif')}, "
                      f"{_idl_str(root / 'work')} + '/', {_idl_str(root / 'out.raw')}, '{mode}'"])
        return np.fromfile(root / "out.raw", dtype=np.uint8).reshape(ny, nx)


def run_gtb_spa(exe: os.PathLike, data: np.ndarray, variant: str) -> dict:
    """GTB's SPA2/3/5/6 (``tests/gtb_idl/gtb_spa.pro``): the class map and the statistics strings."""
    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_spa_") as tmp:
        root = Path(tmp)
        _write_tif(root / "in.tif", data)
        run_gdl(exe, [f"gtb_spa, {_idl_str(root / 'in.tif')}, {_idl_str(root)} + '/', '{variant}'"])
        return {"map": np.fromfile(root / "spa.raw", dtype=np.uint8).reshape(ny, nx),
                "stats": _read_stats(root / "stats.txt")}


DEFAULT_GRAYSPATCON = Path(r"C:\Users\eduar\Documents\dev\GTB\external_sources\grayspatcon64.exe")


def find_grayspatcon() -> Path | None:
    """GTB's grayspatcon executable from ``LANDSCHAFT_GRAYSPATCON_EXE`` or the default GTB checkout."""
    exe = os.environ.get("LANDSCHAFT_GRAYSPATCON_EXE")
    path = Path(exe) if exe else DEFAULT_GRAYSPATCON
    return path if path.is_file() else None


def run_gtb_fos(exe: os.PathLike, spatcon: os.PathLike, grayspatcon: os.PathLike, data: np.ndarray,
                fosclass: str = "FAD_6class", window: int = 7, connectivity: int = 8, gray_threshold: int = 30,
                grayscale: bool | None = None) -> dict:
    """GTB's batch FOS (``tests/gtb_idl/gtb_fos.pro``): the FOS map and its statistics (strings)."""
    import shutil

    if grayscale is None:
        grayscale = int(data.max()) > 4
    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_fos_") as tmp:
        root = Path(tmp)
        (root / "spatcon").mkdir()
        shutil.copy(spatcon, root / "spatcon" / "spatcon64.exe")
        shutil.copy(grayspatcon, root / "spatcon" / "grayspatcon64.exe")
        (root / "work").mkdir()
        (root / "out").mkdir()
        _write_tif(root / "in.tif", data)
        run_gdl(exe, [f"gtb_fos, {_idl_str(root / 'in.tif')}, {_idl_str(root)} + '/', {_idl_str(root / 'out')} + '/', "
                      f"'{fosclass}', {int(window)}, '{connectivity}', {int(gray_threshold)}, {int(not grayscale)}"])
        return {"map": np.fromfile(root / "out" / "fos.raw", dtype=np.uint8).reshape(ny, nx),
                "stats": _read_stats(root / "out" / "stats.txt")}


DEFAULT_GTB_FWTOOLS = Path(r"C:\GuidosToolbox\guidos_progs\GTBtools")
GTB_PROGS = Path(r"C:\Users\eduar\Documents\dev\GTB\guidos_progs")


def find_gtb_mspa() -> Path | None:
    """GTB's ``mspa`` executable: ``LANDSCHAFT_MSPA_EXE`` or the GTB desktop install."""
    exe = find_executable()
    if exe is not None:
        return exe
    default = DEFAULT_GTB_FWTOOLS.parent / "mspa.exe"
    return default if default.is_file() else None


def find_gtb_fwtools() -> Path | None:
    """GTB's gdal tools folder (with ``setfw.bat``) from ``LANDSCHAFT_GTB_FWTOOLS`` or the default install."""
    d = Path(os.environ.get("LANDSCHAFT_GTB_FWTOOLS", DEFAULT_GTB_FWTOOLS))
    return d if (d / "setfw.bat").is_file() else None


def _gtb_colortables() -> Path | None:
    for d in (Path(os.environ.get("LANDSCHAFT_GTB_PROGS", GTB_PROGS)), DEFAULT_GTB_FWTOOLS.parent):
        if (d / "mspacolorston.sav").is_file():
            return d
    return None


def run_gtb_mspa_tile(exe: os.PathLike, mspa: os.PathLike, fwtools: os.PathLike, data: np.ndarray, *,
                      connectivity: int = 8, edge_width: int = 1, transition: bool = True, internal: bool = True,
                      tile_size: int = 6500, overlap: int = 2000) -> np.ndarray:
    """GTB's MSPA Tiling (``tests/gtb_idl/gtb_mspa_tile.pro``) with GTB's gdal tools and ``mspa`` executable."""
    import shutil

    from rasterio.transform import from_origin

    ct = _gtb_colortables()
    if ct is None:
        raise FileNotFoundError("mspacolorston.sav not found (set LANDSCHAFT_GTB_PROGS to GTB's guidos_progs)")
    import sys

    os.environ.setdefault("LANDSCHAFT_PYTHON", sys.executable)  # for tests/gtb_idl/read_tiff.pro
    ny, nx = data.shape
    with tempfile.TemporaryDirectory(prefix="gtb_tile_") as tmp:
        root = Path(tmp)
        (root / "mspatmp").mkdir()
        (root / "data").mkdir()
        shutil.copy(mspa, root / "mspa.exe")
        for name in ("mspacolorston.sav", "mspacolorstoff.sav"):
            shutil.copy(ct / name, root / name)
        with rasterio.open(root / "data" / "in.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                           dtype="uint8", crs="EPSG:3035", transform=from_origin(4_000_000, 3_000_000, 25, 25)) as dst:
            dst.write(np.ascontiguousarray(data, dtype=np.uint8), 1)
        params = [str(connectivity), str(edge_width), str(int(transition)), str(int(internal))]
        fw = str(Path(fwtools)).rstrip("\\/") + "\\"
        run_gdl(exe, [f"gtb_mspa_tile, {_idl_str(root / 'data' / 'in.tif')}, {_idl_str(root)} + '/', '{fw}', "
                      + ", ".join(f"'{p}'" for p in params) + f", {int(tile_size)}, {int(overlap)}"], timeout=3600)
        out = root / "data" / ("in_" + "_".join(params) + ".tif")
        with rasterio.open(out) as src:
            if (src.height, src.width) != (ny, nx):
                raise RuntimeError(f"GTB tiling produced a {src.height}x{src.width} map for {ny}x{nx}")
            return src.read(1)
