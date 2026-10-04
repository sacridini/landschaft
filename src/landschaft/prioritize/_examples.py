# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of ls.prioritize: Serra Verde, an imaginary coastal region (tools/make_prioritize_examples.py;
prioripy's tutorial landscape, by the same author)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {name: f"{name}.tif" for name in (
    "cost", "features", "protected", "urban", "condition", "regions", "carbon", "terminals", "threats", "elevation",
    "lines")}


def example_data(name: str = "features") -> Path:
    """Path of an example data set for prioritization: **Serra Verde**, an imaginary coastal region of
    60 x 90 cells of 1 km (SIRGAS 2000 / UTM 23S). Mountains rise in the north-west; the coastal plain has
    three cities, a highway and two rivers; the ocean in the south-east has no data, which leaves 4825
    planning units on land. The data are synthetic, made to behave like real planning data.

    Parameters
    ----------
    name : {"cost", "features", "protected", "urban", "condition", "regions", "carbon", "terminals", "threats", "elevation", "lines"}
        ``"cost"``: land cost (thousand USD per km²), cheap in the mountains, dear near cities and roads;
        ``"features"``: habitat suitability (0-1) of eight features, a stack with the bands
        ``jaguar`` (widespread), ``lowland_tapir`` (river valleys), ``harpy_eagle`` (remote mountain
        forest), ``maned_wolf`` (western grasslands), ``golden_lion_tamarin`` (two small lowland
        fragments: rare), ``neotropical_otter`` (the rivers), ``araucaria_forest`` (cold highlands) and
        ``mangrove`` (a thin coastal strip: rare);
        ``"protected"``: three existing protected areas (1); ``"urban"``: the cities (1);
        ``"condition"``: habitat condition (0 degraded, 1 pristine); ``"regions"``: three
        administrative regions (1-3); ``"carbon"``: carbon stock (t/ha); ``"terminals"``: the two
        mountain parks to link by a corridor (1); ``"threats"``: ``deforestation`` and ``fire``
        intensity (0-1); ``"elevation"``: relative elevation (0-1); ``"lines"``: the ``rivers`` and the
        ``roads`` (1), for maps.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> features = ls.load_raster(ls.prioritize.example_data("features"))
    >>> features.shape
    (8, 60, 90)
    >>> [str(b) for b in features.band.values][:3]
    ['jaguar', 'lowland_tapir', 'harpy_eagle']
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example data {name!r}: {sorted(EXAMPLES)}")
    return Path(str(resources.files("landschaft.prioritize").joinpath("data", EXAMPLES[name])))
