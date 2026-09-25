# SPDX-License-Identifier: GPL-3.0-or-later
"""guidospy: fast Python reimplementation of GuidosToolbox (GTB).

Typical use::

    import guidospy as gp

    data = gp.load_raster("forest.tif")      # checks the 0/1/2 coding
    result = gp.mspa(data, edge_width=2)      # xarray.DataArray of MSPA classes
    gp.write_raster(result, "forest_mspa.tif")
"""

from importlib import metadata as _metadata

from . import _core
from .batch import BatchResult, mspa_batch
from .io import load_raster, mspa_colormap, reclassify, write_raster
from .mspa import MSPA_CLASSES, mspa, mspa_stats
from .validation import InvalidInputError, validate_mspa_input

try:
    __version__ = _metadata.version("guidospy")  # single source: pyproject.toml
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BatchResult",
    "InvalidInputError",
    "MSPA_CLASSES",
    "load_raster",
    "max_threads",
    "mspa",
    "mspa_batch",
    "mspa_colormap",
    "mspa_stats",
    "reclassify",
    "validate_mspa_input",
    "write_raster",
]


def max_threads() -> int:
    """Number of OpenMP threads the C++ core can use (1 without OpenMP)."""
    return _core.max_threads()
