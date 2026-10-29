# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of the R packages move2 and amt (GPL >= 3): fishers of New York and their landscape."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {
    "fishers": "fishers.csv.gz",
    "elevation": "elevation.tif",
    "landuse": "landuse.tif",
    "popden": "popden.tif",
}


def example_data(name: str = "fishers") -> Path:
    """Path of an example data set for movement ecology.

    Parameters
    ----------
    name : {"fishers", "elevation", "landuse", "popden"}
        ``"fishers"``: GPS tracks of eight fishers (*Pekania pennanti*) near
        Albany, New York, 2009-2011 (LaPoint et al. 2013), Movebank's export
        as move2 ships it: 47347 records, 14443 of them without a location;
        ``"elevation"`` (ASTER, metres), ``"landuse"`` (GlobCover classes) and
        ``"popden"`` (people per km²): the landscape of four of them, as amt
        ships it, in EPSG:5070.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> tracks = ls.load_tracks(ls.move.example_data("fishers"))
    >>> elevation = ls.load_raster(ls.move.example_data("elevation"))
    >>> len(tracks), elevation.shape
    (47347, (272, 294))
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return Path(str(resources.files("landschaft.move") / "data" / EXAMPLES[name]))
