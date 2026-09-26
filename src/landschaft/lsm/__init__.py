# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics (FRAGSTATS) matching the R package landscapemetrics."""

from .analysis import PCAResult, SpatialSignatureAnalysis, SpatioTemporalAnalysis, grid_zones
from .catalog import list_metrics
from .landscape import Landscape, landscape_metrics
from .export import write_table, zonal_metrics
from .plotting import show_correlation, show_cores, show_lsm, show_patches
from .sampling import sample_metrics, scale_sample, window_metrics
from .utils import calculate_correlation, check_landscape, extract_metrics, raster_boundaries

__all__ = [
    "Landscape",
    "PCAResult",
    "SpatialSignatureAnalysis",
    "SpatioTemporalAnalysis",
    "grid_zones",
    "calculate_correlation",
    "check_landscape",
    "extract_metrics",
    "landscape_metrics",
    "list_metrics",
    "raster_boundaries",
    "sample_metrics",
    "scale_sample",
    "show_correlation",
    "show_cores",
    "show_lsm",
    "show_patches",
    "window_metrics",
    "write_table",
    "zonal_metrics",
]
