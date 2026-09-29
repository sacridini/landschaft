# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of the R package gdistance (GPL >= 2, van Etten 2017)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {
    "maungawhau": "maungawhau.tif",
    "europe": "europe.tif",
    "gen_dist": "gen_dist.csv",
    "pop_coord": "pop_coord.csv",
}


def example_data(name: str = "maungawhau") -> Path:
    """Path of an example data set for the grid distances.

    Parameters
    ----------
    name : {"maungawhau", "europe", "gen_dist", "pop_coord"}
        ``"maungawhau"``: elevation of the Maungawhau volcano (Auckland), 87 x 61
        cells of 10 m (New Zealand Map Grid), the classic hiking example;
        ``"europe"``: land (1) and sea (missing) of Europe, 135 x 195 cells of
        1/3 degree (longitude/latitude);
        ``"gen_dist"``: genetic distances between 23 European populations of the
        Y-chromosome haplogroup R1b1b2 (Balaresque et al. 2010, PLoS Biology
        8: e1000285), a CSV table with a ``population`` column;
        ``"pop_coord"``: the longitude (``x``) and latitude (``y``) of those
        populations.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.load_raster(ls.cost.example_data("maungawhau"), validate=None)
    >>> dem.shape, int(dem.max())
    ((87, 61), 195)
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return Path(str(resources.files("landschaft.cost") / "data" / EXAMPLES[name]))
