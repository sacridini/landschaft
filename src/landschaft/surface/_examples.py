# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of the R package geodiv (MIT, Smith et al. 2021)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {
    "normforest": "normforest.tif",
    "orforest": "orforest.tif",
    "orelevation": "orelevation.tif",
}


def example_data(name: str = "normforest") -> Path:
    """Path of an example surface.

    Parameters
    ----------
    name : {"normforest", "orforest", "orelevation"}
        Three 371 x 371 rasters of about 30 m cells (longitude/latitude) over
        a part of south-western Oregon, from the geodiv package: ``"orforest"``,
        the mean NDVI of June-August 2000 (Landsat path 45, row 30);
        ``"normforest"``, the same with its best-fitting trend surface removed
        (the input the surface metrics expect); ``"orelevation"``, SRTM
        elevation (m).

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> ndvi = ls.load_raster(ls.surface.example_data("orforest"), validate=None)
    >>> ndvi.shape, round(float(ndvi.mean()), 3)
    ((371, 371), 0.682)
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return Path(str(resources.files("landschaft.surface") / "data" / EXAMPLES[name]))
