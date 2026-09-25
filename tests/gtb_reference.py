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
