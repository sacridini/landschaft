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
from .collection import RasterCollection, RasterInfo
from .contagion import clustering_fac, contagion_pff
from .datasets import EXAMPLE_DATA, example_data
from .distance import EuclideanDistanceResult, euclidean_distance
from .entropy import SpatialEntropyResult, spatial_entropy, spatial_entropy_map
from .fad import FAD_CLASSES, FADMultiscaleResult, FADResult, fad, fad_multiscale
from .gtn import (
    BRIDGE_CLASSES,
    CORE_CLASSES,
    GTNComponentsResult,
    GTNImportanceResult,
    conefor_inputs,
    gtn_components,
    gtn_importance,
)
from .io import (
    DIST_COLORMAP,
    ENTROPY_COLORMAP,
    FAD_COLORMAP,
    GTN_COMPONENTS_COLORMAP,
    GTN_IMPORTANCE_COLORMAP,
    LM_COLORMAP,
    dist_colormap,
    entropy_colormap,
    fad_colormap,
    gtn_components_colormap,
    gtn_importance_colormap,
    lm_colormap,
    contortion_colormap,
    resistance_colormap,
    load_raster,
    mspa_colormap,
    reclassify,
    write_raster,
)
from .morphology import ParcellationResult, contortion, parcellation
from .mosaic import LM19_CLASSES, landscape_mosaic, landscape_mosaic_stats
from .mspa import MSPA_CLASSES, mspa, mspa_stats
from .restoration import (
    LeastCostPathResult,
    RestorationCorridorResult,
    RestorationStatusResult,
    cost_distance,
    least_cost_path,
    restoration_corridor,
    restoration_status,
)
from .validation import InvalidInputError, validate_mspa_input

try:
    __version__ = _metadata.version("guidospy")  # single source: pyproject.toml
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BatchResult",
    "BRIDGE_CLASSES",
    "CORE_CLASSES",
    "DIST_COLORMAP",
    "ENTROPY_COLORMAP",
    "EXAMPLE_DATA",
    "EuclideanDistanceResult",
    "FADMultiscaleResult",
    "FADResult",
    "FAD_CLASSES",
    "FAD_COLORMAP",
    "GTNComponentsResult",
    "GTNImportanceResult",
    "GTN_COMPONENTS_COLORMAP",
    "GTN_IMPORTANCE_COLORMAP",
    "InvalidInputError",
    "LM19_CLASSES",
    "LM_COLORMAP",
    "MSPA_CLASSES",
    "ParcellationResult",
    "RasterCollection",
    "RasterInfo",
    "RestorationCorridorResult",
    "RestorationStatusResult",
    "LeastCostPathResult",
    "SpatialEntropyResult",
    "clustering_fac",
    "conefor_inputs",
    "contagion_pff",
    "contortion",
    "contortion_colormap",
    "cost_distance",
    "dist_colormap",
    "entropy_colormap",
    "euclidean_distance",
    "example_data",
    "fad",
    "fad_colormap",
    "fad_multiscale",
    "gtn_components",
    "gtn_components_colormap",
    "gtn_importance",
    "gtn_importance_colormap",
    "landscape_mosaic",
    "landscape_mosaic_stats",
    "least_cost_path",
    "lm_colormap",
    "load_raster",
    "max_threads",
    "mspa",
    "mspa_batch",
    "mspa_colormap",
    "mspa_stats",
    "parcellation",
    "reclassify",
    "resistance_colormap",
    "restoration_corridor",
    "restoration_status",
    "spatial_entropy",
    "spatial_entropy_map",
    "validate_mspa_input",
    "write_raster",
]


def max_threads() -> int:
    """Number of OpenMP threads the C++ core can use (1 without OpenMP)."""
    return _core.max_threads()
