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


# Example data of the modules: name -> (package, file). The module functions
# (ls.surface.example_data...) give the same files.
_MODULE_DATA: dict[str, tuple[str, str]] = {
    "normforest": ("landschaft.surface", "normforest.tif"),
    "orforest": ("landschaft.surface", "orforest.tif"),
    "orelevation": ("landschaft.surface", "orelevation.tif"),
    "maungawhau": ("landschaft.cost", "maungawhau.tif"),
    "europe": ("landschaft.cost", "europe.tif"),
    "gen_dist": ("landschaft.cost", "gen_dist.csv"),
    "pop_coord": ("landschaft.cost", "pop_coord.csv"),
    "sites": ("landschaft.bio", "example_sites.csv"),
    "tree": ("landschaft.bio", "example_tree.nex"),
    "bradypus": ("landschaft.sdm", "bradypus.csv"),
    "acaule": ("landschaft.sdm", "acaule.csv"),
    "bioclim": ("landschaft.sdm", "bioclim.tif"),
    "biome": ("landschaft.sdm", "biome.tif"),
}


def example_data(name: str = "input") -> Path:
    """Path of an example data set shipped with the package.

    One function for the examples of every module: open them with
    :func:`load_raster`, :func:`load_vector` or :func:`load_table`.

    Parameters
    ----------
    name
        Rasters of habitat and land cover (:data:`EXAMPLE_DATA`):

        - ``"input"`` (default): habitat map, 1000 x 1000 cells, coded 0 missing,
          1 background, 2 forest;
        - ``"clc"``: CORINE land cover, 2000 x 2000 cells of 25 m (EPSG:3035),
          classes 1-44 (forests: 23, 24, 25);
        - ``"gscinput"``: percentage map (0-100, 255 missing), 1990 x 1289 cells.

        Surfaces (see :func:`landschaft.surface.example_data`): ``"normforest"``,
        ``"orforest"`` (NDVI), ``"orelevation"`` (elevation), 371 x 371 cells.

        Grid distances (see :func:`landschaft.cost.example_data`):
        ``"maungawhau"`` (elevation of a volcano), ``"europe"`` (land and sea),
        ``"gen_dist"`` and ``"pop_coord"`` (tables of genetic distances and
        coordinates of European populations).

        Biodiversity (see :func:`landschaft.bio.example_data`): ``"sites"``
        (species records, CSV) and ``"tree"`` (their phylogeny, Nexus).

        Species distributions (see :func:`landschaft.sdm.example_data`):
        ``"bradypus"`` and ``"acaule"`` (occurrence records, CSV),
        ``"bioclim"`` (nine bioclimatic variables) and ``"biome"`` (biomes),
        0.5 degree over Central and South America.

    Returns
    -------
    pathlib.Path
        Path of the file.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.example_data("clc").name
    'clc.tif'
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> dem = ls.load_raster(ls.example_data("orelevation"))
    >>> sites = ls.load_table(ls.example_data("sites"))
    >>> dem.shape, len(sites)
    ((371, 371), 1795)
    """
    if name in EXAMPLE_DATA:
        return Path(str(resources.files("landschaft") / "data" / f"{name}.tif"))
    if name in _MODULE_DATA:
        package, file = _MODULE_DATA[name]
        return Path(str(resources.files(package) / "data" / file))
    names = [*EXAMPLE_DATA, *_MODULE_DATA]
    raise ValueError(f"unknown example {name!r}; available: {', '.join(names)}")
