# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of ls.prioritize: a hypothetical region (tools/make_prioritize_examples.py)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {"cost": "cost.tif", "features": "features.tif", "protected": "protected.tif", "threats": "threats.tif"}


def example_data(name: str = "features") -> Path:
    """Path of an example data set for prioritization: a hypothetical region of 40 x 60 cells of 1 km
    (UTM zone 23S), part of it sea and a reservoir (no data).

    Parameters
    ----------
    name : {"cost", "features", "protected", "threats"}
        ``"cost"``: the cost of every cell (5-100, e.g. the price of the land); ``"features"``: six
        species (``forest_bird``, ``wetland_plant``, ``grassland_butterfly``, ``large_mammal``,
        ``amphibian``, ``endemic_snail``), the probability of occurrence in every cell (0-1), from
        widespread to rare; ``"protected"``: the existing protected areas (1); ``"threats"``: the
        intensity (0-1) of two threats, ``fire`` and ``grazing``.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> features = ls.load_raster(ls.prioritize.example_data("features"))
    >>> features.shape
    (6, 40, 60)
    >>> [str(b) for b in features.band.values][:3]
    ['forest_bird', 'wetland_plant', 'grassland_butterfly']
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example data {name!r}: {sorted(EXAMPLES)}")
    return Path(str(resources.files("landschaft.prioritize").joinpath("data", EXAMPLES[name])))
