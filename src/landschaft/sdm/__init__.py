# SPDX-License-Identifier: GPL-3.0-or-later
"""Species distribution modelling: preparing occurrences and predictors, fitting, projecting.

Most of the work in a species distribution model is preparing the data:
cleaning the occurrence records, thinning them, choosing the area the species
could reach, selecting predictors, sampling the background and splitting the
data into spatially independent folds. ``ls.sdm`` does each step with one
call and sensible defaults, every step remaining a public function with all
its options; the results feed scikit-learn directly. The steps reproduce the
R packages the field relies on (CoordinateCleaner, spThin, usdm, blockCV,
ENMeval, ecospat, predicts, maxnet), in C++ with OpenMP.

Typical use::

    import landschaft as ls

    occ = ls.load_vector(ls.sdm.example_data("bradypus"), crs="EPSG:4326")
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
"""

from ._area import accessible_area
from ._autocorrelation import autocorrelation_range
from ._background import background
from ._bias import bias_surface
from ._cells import one_per_cell
from ._clean import clean_occurrences
from ._data import SDMData, dataset
from ._evaluate import Evaluation, boyce, evaluate, omission_rate, threshold
from ._examples import example_data
from ._extrapolation import exdet, mess, mop
from ._folds import (
    buffer_folds,
    checkerboard_folds,
    jackknife_folds,
    quadrant_folds,
    random_folds,
    spatial_blocks,
    spatial_clusters,
)
from ._gbif import download_gbif
from ._habitat import HabitatAreas, area_of_habitat, cell_area_km2
from ._maxnet import MaxNet
from ._models import SDMResult, fit, importance, predict, response_curves
from ._predictors import select_predictors, vif
from ._prepare import prepare
from ._refdata import reference_data
from ._thin import thin

__all__ = [
    "Evaluation",
    "HabitatAreas",
    "MaxNet",
    "SDMData",
    "SDMResult",
    "accessible_area",
    "area_of_habitat",
    "autocorrelation_range",
    "background",
    "bias_surface",
    "boyce",
    "buffer_folds",
    "cell_area_km2",
    "checkerboard_folds",
    "clean_occurrences",
    "dataset",
    "download_gbif",
    "evaluate",
    "example_data",
    "importance",
    "exdet",
    "fit",
    "jackknife_folds",
    "mess",
    "mop",
    "omission_rate",
    "one_per_cell",
    "predict",
    "prepare",
    "quadrant_folds",
    "random_folds",
    "reference_data",
    "response_curves",
    "select_predictors",
    "spatial_blocks",
    "spatial_clusters",
    "thin",
    "threshold",
    "vif",
]
