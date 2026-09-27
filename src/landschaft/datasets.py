# SPDX-License-Identifier: GPL-3.0-or-later
"""Sample rasters shipped with GuidosToolbox (GTB ``data/`` directory)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["EXAMPLE_DATA", "example_data"]

#: Name -> description of the bundled sample rasters.
EXAMPLE_DATA: dict[str, str] = {
    "input": (
        "GTB MSPA test map, 1000 x 1000, uint8 coded 0 = missing, 1 = background, "
        "2 = foreground (EPSG:4326). Ready for load_raster() + mspa()."
    ),
    "clc": (
        "CORINE Land Cover excerpt, 2000 x 2000, 25 m, EPSG:3035, CLC level-3 class codes "
        "1-44 (forests: 23 broad-leaved, 24 coniferous, 25 mixed). Use reclassify() for MSPA."
    ),
    "gscinput": (
        "GraySpatCon test map, 1990 x 1289, 2430 m, EPSG:5070, percentages 0-100 with "
        "nodata 255 (for the upcoming GraySpatCon tool)."
    ),
}


def example_data(name: str = "input") -> Path:
    """Path of an example raster shipped with the package.

    Parameters
    ----------
    name
        One of :data:`EXAMPLE_DATA`:

        - ``"input"`` (default): habitat map, 1000 x 1000 cells, coded 0 missing,
          1 background, 2 forest;
        - ``"clc"``: CORINE land cover, 2000 x 2000 cells of 25 m (EPSG:3035),
          classes 1-44 (forests: 23, 24, 25);
        - ``"gscinput"``: percentage map (0-100, 255 missing), 1990 x 1289 cells.

    Returns
    -------
    pathlib.Path
        Path of the GeoTIFF, to pass to :func:`load_raster` or any tool.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.example_data("clc").name
    'clc.tif'
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.mspa(forest)
    """
    if name not in EXAMPLE_DATA:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLE_DATA)}")
    path = resources.files("landschaft") / "data" / f"{name}.tif"
    return Path(str(path))
