# SPDX-License-Identifier: GPL-3.0-or-later
"""Running the reference land change tools behind ls.simulate.

One function per engine, each taking a case's job and a scratch folder and
giving a dict ``key -> numpy array or list of str``:

- ``clumondo``: CLUMondo (github.com/VUEG/CLUMondo, GPL-3) built 32-bit with
  MSVC (``LANDSCHAFT_CLUMONDO``, default
  ``~/Documents/dev/CLUMondo/build/bin/Release/clumondo.exe``); the job holds
  the project files (text, or 2-D arrays written as ESRI ASCII grids);
- ``grass``: GRASS GIS 8 with the r.futures addon in WSL
  (``LANDSCHAFT_WSL_DISTRO`` optional); the job holds GeoTIFF inputs and a
  bash script run inside a temporary GRASS project;
- ``wsl_python``: a Python script run with WSL's python3 (e.g. calling
  GRASS's C library through ctypes), writing ``.npy`` files;
- ``dinamica``: Dinamica EGO 8's console (``LANDSCHAFT_DINAMICA``, default
  ``~/Documents/dev/dinamica_ego/install/DinamicaConsole8.exe``), one
  processor and the predefined seed; the job holds GeoTIFF inputs and an EGO
  script;
- ``molusce``: the MOLUSCE QGIS plugin's algorithms (``LANDSCHAFT_MOLUSCE``,
  default ``~/Documents/dev/qgis_molusce/src``) imported with stubs for the
  QGIS / Qt modules they name; the job is a function of the imported package.

The R packages (lulcc, OpenLand, diffeR) run through
``tools/simulate_reference.R`` in ``make_reference_simulate.py``.
"""

from __future__ import annotations

import csv
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path, PureWindowsPath

import numpy as np

DEV = Path.home() / "Documents" / "dev"
CLUMONDO = Path(os.environ.get("LANDSCHAFT_CLUMONDO", DEV / "CLUMondo" / "build" / "bin" / "Release" / "clumondo.exe"))
# CLUMondo with tools/clumondo_fixed.patch applied, built the same way
CLUMONDO_FIXED = Path(os.environ.get("LANDSCHAFT_CLUMONDO_FIXED",
                                     DEV / "CLUMondo_fixed" / "build" / "bin" / "Release" / "clumondo.exe"))
# the official executable of May 2017 (newer than the source: demand types, iteration mode 2, patches)
CLUMONDO_2017 = Path(os.environ.get("LANDSCHAFT_CLUMONDO_2017", DEV / "CLUMondo" / "ref" / "CLUMondo_may_2017.exe"))
DINAMICA = Path(os.environ.get("LANDSCHAFT_DINAMICA",
                               DEV / "dinamica_ego" / "install" / "DinamicaConsole8.exe"))
MOLUSCE = Path(os.environ.get("LANDSCHAFT_MOLUSCE", DEV / "qgis_molusce" / "src"))


# ----------------------------------------------------------------------------- files

def write_ascii_grid(path: Path, values: np.ndarray, *, xll=0.0, yll=0.0, cellsize=1.0, nodata=-9999,
                     fmt: str | None = None) -> None:
    """An ESRI ASCII grid (the Arcview format CLUMondo reads with ``arcview = 3``)."""
    v = np.asarray(values)
    if fmt is None:
        fmt = "%d" if np.issubdtype(v.dtype, np.integer) else "%.9g"
    with open(path, "w", newline="\n") as f:
        f.write(f"ncols {v.shape[1]}\nnrows {v.shape[0]}\nxllcorner {xll!r}\nyllcorner {yll!r}\n"
                f"cellsize {cellsize!r}\nNODATA_value {nodata}\n")
        f.writelines(" ".join(fmt % x for x in row) + "\n" for row in v)


def read_ascii_grid(path: Path) -> np.ndarray:
    """The values of an ESRI ASCII grid (header of six lines) as float64."""
    with open(path) as f:
        lines = f.read().split("\n")
    header = {}
    k = 0
    while k < len(lines) and lines[k].split() and lines[k].split()[0].lower() in (
            "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "nodata_value"):
        name, value = lines[k].split()[:2]
        header[name.lower()] = float(value)
        k += 1
    values = np.array(" ".join(lines[k:]).split(), dtype=np.float64)
    return values.reshape(int(header["nrows"]), int(header["ncols"]))


def read_table(path: Path) -> dict[str, np.ndarray | list[str]]:
    """A CSV table as columns: numeric columns as float64, the others as text."""
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    header = [h for h in rows[0] if h != ""]
    out: dict = {}
    for j, name in enumerate(header):
        col = [r[j] if j < len(r) else "" for r in rows[1:]]
        try:
            out[name] = np.array([float(c) for c in col], dtype=np.float64)
        except ValueError:
            out[name] = col
    return out


def write_geotiff(path: Path, da) -> None:
    import landschaft as ls

    ls.write_raster(da, path, compress="none")


def write_dinamica_raster(path: Path, da, name: str) -> None:
    """A GeoTIFF Dinamica EGO reads: the band named ``name`` (its layer is ``name/name``); float maps as float64
    with -9999 as no data (Dinamica needs a null value for NaN), categorical ones (``attrs["categorical"]``, or
    integer maps) as int32 with -1 as no data."""
    import rasterio

    v = np.asarray(da.values)
    if np.issubdtype(v.dtype, np.floating) and not da.attrs.get("categorical", False):
        out = da.copy(data=np.where(np.isnan(v), -9999.0, v)).rio.write_nodata(-9999.0)
    else:
        f = v.astype(np.float64)
        out = da.copy(data=np.where(np.isnan(f), -1, f).astype(np.int32)).rio.write_nodata(-1)
    write_geotiff(path, out)
    with rasterio.open(path, "r+") as dst:
        dst.set_band_description(1, name)


def read_geotiff(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(masked=True).astype(np.float64)
    return np.squeeze(a.filled(np.nan))


# ----------------------------------------------------------------------------- CLUMondo

def run_clumondo(job: dict, tmp: Path, exe: Path | None = None) -> dict:
    """Write the project files, run CLUMondo, read the output grids and tables.

    ``job``: ``files`` (name -> text, or 2-D array written as an ASCII grid
    with the ``grid`` header keywords), ``args`` (demand file, region file
    and the optional mode), ``outputs`` (file names or glob patterns; ASCII
    grids become arrays, other files text lines).
    """
    exe = exe or CLUMONDO
    if not exe.is_file():
        raise FileNotFoundError(f"CLUMondo not found at {exe} (set LANDSCHAFT_CLUMONDO / _FIXED / _2017)")
    grid = job.get("grid", {})
    for name, content in job["files"].items():
        if isinstance(content, str):
            (tmp / name).write_text(content, newline="\n")
        else:
            write_ascii_grid(tmp / name, content, **grid)
    proc = subprocess.run([str(exe), *map(str, job["args"])], cwd=tmp, capture_output=True, text=True,
                          timeout=job.get("timeout", 3600), check=False)
    log = (tmp / "log.fil").read_text(errors="replace") if (tmp / "log.fil").exists() else ""
    if proc.returncode != 0 or "Error" in proc.stdout:
        raise RuntimeError(f"CLUMondo failed:\n{proc.stdout[-2000:]}\n--- log.fil ---\n{log[-3000:]}")
    out: dict = {}
    for pattern in job["outputs"]:
        for path in sorted(tmp.glob(pattern)):
            if path.suffix == ".asc":
                out[path.name] = read_ascii_grid(path)
            else:
                out[path.name] = path.read_text(errors="replace").splitlines()
    return out


# ----------------------------------------------------------------------------- WSL / GRASS

def wsl_path(path: Path) -> str:
    p = PureWindowsPath(Path(path).resolve())
    return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:])


def _wsl(cmd: str, timeout: float) -> subprocess.CompletedProcess:
    distro = os.environ.get("LANDSCHAFT_WSL_DISTRO")
    args = ["wsl", *(["-d", distro] if distro else []), "-e", "bash", "-lc", cmd]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def run_grass(job: dict, tmp: Path) -> dict:
    """Run a bash script inside a temporary GRASS project in WSL.

    ``job``: ``rasters`` (name -> DataArray, written as GeoTIFF and imported
    with r.in.gdal; the first one sets the project's CRS and the region),
    ``script`` (bash, run with ``$OUT`` as the output folder), ``outputs``
    (file names in ``$OUT``: ``.tif`` read as arrays, ``.csv`` as tables,
    others as text lines).
    """
    names = list(job["rasters"])
    for name, da in job["rasters"].items():
        write_geotiff(tmp / f"{name}.tif", da)
    out_dir = tmp / "out"
    out_dir.mkdir()
    imports = "\n".join(f"r.in.gdal -o --quiet input={name}.tif output={name}" for name in names)
    body = (f"set -e\ncd {wsl_path(tmp)}\nexport OUT={wsl_path(out_dir)}\n{imports}\n"
            f"g.region raster={names[0]}\n{job['script']}\n")
    (tmp / "job.sh").write_text(body, newline="\n")
    project = f"{wsl_path(tmp)}/{names[0]}.tif"
    proc = _wsl(f"grass --tmp-location {project} --exec bash {wsl_path(tmp / 'job.sh')}", job.get("timeout", 3600))
    if proc.returncode != 0:
        raise RuntimeError(f"GRASS failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    return _read_outputs(out_dir, job["outputs"])


def run_wsl_python(job: dict, tmp: Path) -> dict:
    """Run a Python script with WSL's python3; it writes ``<key>.npy`` files to ``sys.argv[1]``."""
    out_dir = tmp / "out"
    out_dir.mkdir()
    (tmp / "job.py").write_text(job["script"], newline="\n")
    proc = _wsl(f"python3 {wsl_path(tmp / 'job.py')} {wsl_path(out_dir)}", job.get("timeout", 600))
    if proc.returncode != 0:
        raise RuntimeError(f"WSL python failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    return {p.stem: np.load(p) for p in sorted(out_dir.glob("*.npy"))}


def _read_outputs(folder: Path, outputs) -> dict:
    out: dict = {}
    for name in outputs:
        path = folder / name
        if path.suffix == ".tif":
            out[name] = read_geotiff(path)
        elif path.suffix == ".csv":
            for col, values in read_table(path).items():
                out[f"{name}/{col}"] = values
        elif path.suffix == ".asc":
            out[name] = read_ascii_grid(path)
        else:
            out[name] = path.read_text(errors="replace").splitlines()
    return out


# ----------------------------------------------------------------------------- Dinamica EGO

def run_dinamica(job: dict, tmp: Path) -> dict:
    """Run an EGO script with Dinamica EGO's console in ``tmp``.

    ``job``: ``rasters`` (name -> DataArray, written as ``<name>.tif``),
    ``script`` (the EGO script; relative paths are in ``tmp``), ``outputs``
    (files the script writes: ``.csv`` tables, ``.tif`` maps).
    """
    if not DINAMICA.is_file():
        raise FileNotFoundError(f"Dinamica EGO console not found at {DINAMICA} (set LANDSCHAFT_DINAMICA)")
    for name, da in job.get("rasters", {}).items():
        write_dinamica_raster(tmp / f"{name}.tif", da, name)
    (tmp / "model.ego").write_text(job["script"], newline="\n")
    proc = subprocess.run([str(DINAMICA), "-processors", "1", "-predefined-seed", "model.ego"], cwd=tmp,
                          capture_output=True, text=True, timeout=job.get("timeout", 3600), check=False)
    if proc.returncode != 0 or "ran successfully" not in proc.stdout.splitlines()[-1] + proc.stdout[-400:]:
        raise RuntimeError(f"Dinamica EGO failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}")
    return _read_outputs(tmp, job["outputs"])


# ----------------------------------------------------------------------------- MOLUSCE

class _Signal:
    def __init__(self, *types_):
        pass

    def __get__(self, obj, owner):
        return self

    def connect(self, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


def _qgis_stubs() -> None:
    """Stand-ins for the QGIS / Qt names MOLUSCE's algorithms import."""
    if "qgis" in sys.modules and getattr(sys.modules["qgis"], "_landschaft_stub", False):
        return

    class QObject:
        def __init__(self, *args, **kwargs):
            pass

        def tr(self, text):
            return text

    class QCoreApplication:
        @staticmethod
        def translate(context, text, *args):
            return text

    def pyqtSlot(*args, **kwargs):
        return lambda f: f

    class _Any(types.ModuleType):
        """A module whose every attribute is an empty class (names MOLUSCE imports but never calls here)."""

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, name, cls)
            return cls

    qtcore = _Any("qgis.PyQt.QtCore")
    qtcore.QObject = QObject
    qtcore.QCoreApplication = QCoreApplication
    qtcore.pyqtSignal = _Signal
    qtcore.pyqtSlot = pyqtSlot
    qtcore.__all__ = ["QObject", "QCoreApplication", "pyqtSignal", "pyqtSlot"]
    qgis = types.ModuleType("qgis")
    qgis._landschaft_stub = True
    pyqt = types.ModuleType("qgis.PyQt")
    pyqt.QtCore = qtcore
    qgis.PyQt = pyqt
    extra = {name: _Any(name) for name in ("qgis.core", "qgis.gui", "qgis.PyQt.QtWidgets", "qgis.PyQt.QtXml",
                                           "qgis.PyQt.QtGui", "osgeo.osr")}
    gdal = types.ModuleType("osgeo.gdal")
    for k, name in enumerate(["GDT_Unknown", "GDT_Byte", "GDT_UInt16", "GDT_Int16", "GDT_UInt32", "GDT_Int32",
                              "GDT_Float32", "GDT_Float64"]):
        setattr(gdal, name, k)
    gdal.DCAP_CREATE = "DCAP_CREATE"
    osgeo = types.ModuleType("osgeo")
    osgeo.gdal, osgeo.osr = gdal, extra["osgeo.osr"]
    qgis.core, qgis.gui = extra["qgis.core"], extra["qgis.gui"]
    pyqt.QtWidgets, pyqt.QtXml, pyqt.QtGui = extra["qgis.PyQt.QtWidgets"], extra["qgis.PyQt.QtXml"], extra["qgis.PyQt.QtGui"]
    sys.modules.update({"qgis": qgis, "qgis.PyQt": pyqt, "qgis.PyQt.QtCore": qtcore, "osgeo": osgeo,
                        "osgeo.gdal": gdal, **extra})


def molusce_version() -> str:
    meta = MOLUSCE / "molusce" / "metadata.txt"
    if meta.exists():
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "version":
                return value.strip()
    return "unknown"


def run_molusce(job, tmp: Path) -> dict:
    """Call ``job(import_module)`` with MOLUSCE's algorithms importable; it returns the result dict."""
    if not MOLUSCE.is_dir():
        raise FileNotFoundError(f"MOLUSCE sources not found at {MOLUSCE} (set LANDSCHAFT_MOLUSCE)")
    _qgis_stubs()
    if str(MOLUSCE) not in sys.path:
        sys.path.insert(0, str(MOLUSCE))
    return job(importlib.import_module)


def dinamica_version() -> str:
    """The installer the console came from (``SetupDinamicaEGO-<version>.exe`` beside the install folder)."""
    setups = sorted(DINAMICA.parent.parent.glob("SetupDinamicaEGO-*.exe"))
    return setups[-1].stem.removeprefix("SetupDinamicaEGO-") if setups else "unknown"


ENGINES = {
    "clumondo": run_clumondo,
    "clumondo_fixed": lambda job, tmp: run_clumondo(job, tmp, CLUMONDO_FIXED),
    "clumondo_2017": lambda job, tmp: run_clumondo(job, tmp, CLUMONDO_2017),
    "grass": run_grass,
    "wsl_python": run_wsl_python,
    "dinamica": run_dinamica,
    "molusce": run_molusce,
}
