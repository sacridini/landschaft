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
    benign = ("Compiled module", "Keyword parameter L64 not supported")
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
