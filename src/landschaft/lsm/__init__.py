# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics (FRAGSTATS) matching the R package landscapemetrics."""

from .analysis import (ClustergramResult, NMFResult, PCAResult, SpatialSignatureAnalysis, SpatioTemporalAnalysis,
                       grid_zones)
from .catalog import list_metrics
from .landscape import Landscape, landscape_metrics
from .export import write_table, zonal_metrics
from .helpers import construct_buffer, get_unique_values, pad_raster, raster_to_points, unpad_raster
from .plotting import show_correlation, show_cores, show_lsm, show_patches
from .sampling import sample_metrics, scale_sample, window_metrics
from .utils import calculate_correlation, check_landscape, extract_metrics, raster_boundaries

__all__ = [
    "Landscape",
    "ClustergramResult",
    "NMFResult",
    "PCAResult",
    "SpatialSignatureAnalysis",
    "SpatioTemporalAnalysis",
    "grid_zones",
    "calculate_correlation",
    "check_landscape",
    "construct_buffer",
    "extract_metrics",
    "get_unique_values",
    "landscape_metrics",
    "list_metrics",
    "pad_raster",
    "raster_boundaries",
    "raster_to_points",
    "sample_metrics",
    "scale_sample",
    "show_correlation",
    "show_cores",
    "show_lsm",
    "show_patches",
    "unpad_raster",
    "window_metrics",
    "write_table",
    "zonal_metrics",
]
