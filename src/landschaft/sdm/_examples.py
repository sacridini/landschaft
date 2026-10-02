# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of the R packages predicts / dismo (GPL >= 3, Hijmans et al.)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {
    "bradypus": "bradypus.csv",
    "acaule": "acaule.csv",
    "bioclim": "bioclim.tif",
    "biome": "biome.tif",
}


def example_data(name: str = "bradypus") -> Path:
    """Path of an example data set for species distribution modelling.

    Parameters
    ----------
    name : {"bradypus", "acaule", "bioclim", "biome"}
        ``"bradypus"``: 116 occurrence records of the brown-throated sloth
        (*Bradypus variegatus*), a CSV table with ``species``, ``lon``, ``lat``;
        ``"acaule"``: 1366 GBIF records of the wild potato *Solanum acaule*
        (with country, locality, coordinate uncertainty, institution and basis
        of record; 284 without coordinates), the classic data for cleaning;
        ``"bioclim"``: nine WorldClim bioclimatic variables (bio1, bio5, bio6,
        bio7, bio8, bio9, bio12, bio16, bio17; temperatures in tenths of a
        degree), 192 x 186 cells of 0.5 degree over Central and South America;
        ``"biome"``: the WWF biomes (1-14) on the same grid.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_vector(ls.sdm.example_data("bradypus"), crs="EPSG:4326")
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> len(occ), env.shape
    (116, (9, 192, 186))
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return Path(str(resources.files("landschaft.sdm") / "data" / EXAMPLES[name]))
