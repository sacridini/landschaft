# SPDX-License-Identifier: GPL-3.0-or-later
"""landschaft: fast Python reimplementation of GuidosToolbox (GTB).

Typical use::

    import landschaft as ls

    data = ls.load_raster("forest.tif")      # checks the 0/1/2 coding
    result = ls.mspa(data, edge_width=2)      # xarray.DataArray of MSPA classes
    ls.write_raster(result, "forest_mspa.tif")
"""

from importlib import metadata as _metadata

from . import _core
from . import _threads

_threads.install(_core, _core.set_num_threads)
from ._threads import default_threads, set_default_threads  # noqa: E402
from .accounting import ACCOUNTING_CLASSES, AccountingResult, accounting
from .batch import BATCH_TOOLS, BatchResult, batch, mspa_batch
from .change import (
    FOSChangeResult,
    MorphologicalChangeResult,
    SimpleChangeResult,
    fos_change,
    morphological_change,
    simple_change,
)
from . import bio
from . import circuit
from .circuit import omniscape
from .collection import RasterCollection, RasterInfo
from . import conn
from .conn import HabitatGraph
from . import cost
from .contagion import clustering_fac, contagion_pff
from .datasets import EXAMPLE_DATA, example_data
from .distance import EuclideanDistanceResult, euclidean_distance
from .entropy import SpatialEntropyResult, spatial_entropy, spatial_entropy_map
from .fad import FAD_CLASSES, FADMultiscaleResult, FADResult, fad, fad_multiscale
from .fad_change import FADChangeScale, FADMultiscaleChangeResult, fad_multiscale_change
from .grayspatcon import gray_spatial_convolution
from .isochrone import ISOCHRONE_CLASSES, IsochroneResult, distance_to_resistance, isochrone_map, isochrone_map_ab
from .influence import INFLUENCE_CLASSES, InfluenceZonesResult, ProximityResult, influence_zones, proximity
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
    ACCOUNTING_COLORMAP,
    CHANGE_COLORMAP,
    DIST_COLORMAP,
    ENTROPY_COLORMAP,
    FAD_COLORMAP,
    FOSCHANGE_COLORMAP,
    GTN_COMPONENTS_COLORMAP,
    GTN_IMPORTANCE_COLORMAP,
    LM_COLORMAP,
    accounting_colormap,
    change_colormap,
    contortion_colormap,
    dist_colormap,
    entropy_colormap,
    fad5_colormap,
    fad_colormap,
    foschange_colormap,
    gtn_components_colormap,
    gtn_importance_colormap,
    lm_colormap,
    load_raster,
    mspa_colormap,
    reclassify,
    resistance_colormap,
    write_raster,
)
from .morphology import ParcellationResult, contortion, parcellation
from . import nlm
from . import surface
from .lsm import (
    Landscape,
    SpatialSignatureAnalysis,
    SpatioTemporalAnalysis,
    grid_zones,
    calculate_correlation,
    check_landscape,
    construct_buffer,
    extract_metrics,
    get_unique_values,
    landscape_metrics,
    list_metrics,
    pad_raster,
    raster_boundaries,
    raster_to_points,
    sample_metrics,
    scale_sample,
    show_correlation,
    show_cores,
    show_lsm,
    show_patches,
    unpad_raster,
    window_metrics,
    write_table,
    zonal_metrics,
)
from .lm_heatmap import LMHeatmap, LMHeatmapChange, landscape_mosaic_change, landscape_mosaic_heatmap
from .mosaic import LM19_CLASSES, landscape_mosaic, landscape_mosaic_stats
from .mspa import MSPA_CLASSES, mspa, mspa_stats
from .spa import SPA_CLASSES, SPAResult, spa
from .fos import FOSResult, fos
from .mspa_tiling import mspa_tiled, tile_layout
from .restoration import (
    LeastCostPathResult,
    RestorationCorridorResult,
    RestorationStatusResult,
    cost_distance,
    least_cost_path,
    restoration_corridor,
    restoration_status,
)
from .validation import InvalidInputError, validate_fad_input, validate_mspa_input

try:
    __version__ = _metadata.version("landschaft")  # single source: pyproject.toml
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ACCOUNTING_CLASSES",
    "ACCOUNTING_COLORMAP",
    "AccountingResult",
    "BATCH_TOOLS",
    "BatchResult",
    "BRIDGE_CLASSES",
    "CORE_CLASSES",
    "DIST_COLORMAP",
    "ENTROPY_COLORMAP",
    "EXAMPLE_DATA",
    "EuclideanDistanceResult",
    "FADChangeScale",
    "FADMultiscaleChangeResult",
    "FADMultiscaleResult",
    "FADResult",
    "FAD_CLASSES",
    "FAD_COLORMAP",
    "FOSCHANGE_COLORMAP",
    "FOSChangeResult",
    "FOSResult",
    "CHANGE_COLORMAP",
    "GTNComponentsResult",
    "GTNImportanceResult",
    "GTN_COMPONENTS_COLORMAP",
    "GTN_IMPORTANCE_COLORMAP",
    "HabitatGraph",
    "INFLUENCE_CLASSES",
    "ISOCHRONE_CLASSES",
    "IsochroneResult",
    "InfluenceZonesResult",
    "InvalidInputError",
    "LM19_CLASSES",
    "LMHeatmap",
    "LMHeatmapChange",
    "LM_COLORMAP",
    "Landscape",
    "MSPA_CLASSES",
    "MorphologicalChangeResult",
    "ParcellationResult",
    "ProximityResult",
    "RasterCollection",
    "RasterInfo",
    "RestorationCorridorResult",
    "RestorationStatusResult",
    "LeastCostPathResult",
    "SpatialSignatureAnalysis",
    "SpatioTemporalAnalysis",
    "SPA_CLASSES",
    "SPAResult",
    "SimpleChangeResult",
    "SpatialEntropyResult",
    "accounting",
    "accounting_colormap",
    "change_colormap",
    "bio",
    "circuit",
    "cost",
    "nlm",
    "surface",
    "batch",
    "clustering_fac",
    "conefor_inputs",
    "contagion_pff",
    "calculate_correlation",
    "check_landscape",
    "construct_buffer",
    "contortion",
    "contortion_colormap",
    "cost_distance",
    "dist_colormap",
    "distance_to_resistance",
    "entropy_colormap",
    "euclidean_distance",
    "example_data",
    "extract_metrics",
    "get_unique_values",
    "fad",
    "fad5_colormap",
    "fad_colormap",
    "fad_multiscale",
    "fad_multiscale_change",
    "fos",
    "fos_change",
    "foschange_colormap",
    "gray_spatial_convolution",
    "grid_zones",
    "gtn_components",
    "gtn_components_colormap",
    "gtn_importance",
    "gtn_importance_colormap",
    "influence_zones",
    "isochrone_map",
    "isochrone_map_ab",
    "landscape_metrics",
    "landscape_mosaic",
    "landscape_mosaic_change",
    "landscape_mosaic_heatmap",
    "landscape_mosaic_stats",
    "least_cost_path",
    "list_metrics",
    "lm_colormap",
    "load_raster",
    "max_threads",
    "default_threads",
    "set_default_threads",
    "morphological_change",
    "mspa",
    "mspa_batch",
    "mspa_colormap",
    "mspa_stats",
    "mspa_tiled",
    "omniscape",
    "pad_raster",
    "parcellation",
    "proximity",
    "raster_boundaries",
    "raster_to_points",
    "reclassify",
    "resistance_colormap",
    "restoration_corridor",
    "restoration_status",
    "sample_metrics",
    "scale_sample",
    "show_correlation",
    "show_cores",
    "show_lsm",
    "show_patches",
    "simple_change",
    "spa",
    "tile_layout",
    "unpad_raster",
    "spatial_entropy",
    "spatial_entropy_map",
    "validate_fad_input",
    "validate_mspa_input",
    "window_metrics",
    "write_table",
    "write_raster",
    "zonal_metrics",
]


def max_threads() -> int:
    """Number of processors the compiled core can use.

    1 when landschaft was built without OpenMP. Parallel functions use
    :func:`default_threads` of them unless told otherwise.

    Returns
    -------
    int

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.max_threads() >= ls.default_threads() >= 1
    True
    """
    return _core.max_threads()
