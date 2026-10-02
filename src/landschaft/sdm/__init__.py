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

from ._cells import one_per_cell
from ._examples import example_data

__all__ = [
    "example_data",
    "one_per_cell",
]
