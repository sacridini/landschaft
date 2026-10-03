# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data for land change modelling.

Plum Island Ecosystems and Sibuyan Island ship with the package (from the R
package lulcc, GPL >= 2, Moulds 2015; written by
``tools/make_simulate_examples.R``). The Sao Lourenco River Basin (OpenLand,
Exavier & Zeilhofer 2020, CC-BY-4.0) is downloaded on first use and
converted to a GeoTIFF in the download cache.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

from .._download import cache_dir, fetch

EXAMPLES = {
    "pie_landuse": "pie_landuse.tif",
    "pie_drivers": "pie_drivers.tif",
    "sibuyan_landuse": "sibuyan_landuse.tif",
    "sibuyan_drivers": "sibuyan_drivers.tif",
    "sibuyan_restrictions": "sibuyan_restrictions.tif",
    "sibuyan_demand": "sibuyan_demand.csv",
    "sao_lourenco": None,
}

_SAO_LOURENCO_URL = "https://zenodo.org/records/3685230/files/SaoLourencoBasin.rda?download=1"
_SAO_LOURENCO_SHA256 = "047851f3a393ff79eeed7b7cf759566b949416330f8dd17cb871e01922c5a7cb"


def example_data(name: str = "sibuyan_landuse") -> Path:
    """Path of an example data set for land change modelling.

    Parameters
    ----------
    name : str
        Plum Island Ecosystems, Massachusetts (Pontius & Parmentier 2014;
        434 x 497 cells of ~100 m, EPSG:26986):

        - ``"pie_landuse"``: land use in 1985, 1991 and 1999 (bands named by
          year; 1 forest, 2 built, 3 other; 255 = no data);
        - ``"pie_drivers"``: elevation, slope and distance to built land in
          1985.

        Sibuyan Island, Philippines (Verburg et al. 2002, the CLUE-S
        demonstration data, modified for teaching: not for other uses;
        108 x 128 cells of 250 m, no CRS):

        - ``"sibuyan_landuse"``: land use in 1997 and 2011 (1 forest,
          2 coconut, 3 grassland, 4 rice, 5 other);
        - ``"sibuyan_drivers"``: 13 drivers (distance to sea, population
          density, diorite, ultramafic rock, sediments, no erosion, moderate
          erosion, elevation, slope, aspect, distance to roads, to urban areas
          and to streams);
        - ``"sibuyan_restrictions"``: current and proposed national park;
        - ``"sibuyan_demand"``: CSV with the demand scenarios ``demand1`` to
          ``demand4`` (cells of each land use per year).

        Sao Lourenco River Basin, Mato Grosso, Brazil (``"sao_lourenco"``;
        SOS Pantanal 2015; 6372 x 6546 cells of 30 m, UTM 21S): land use and
        cover in 2002, 2008, 2010, 2012 and 2014 (2 pasture, 3 forest, 4 park
        savannah, 5 gramineous savannah, 7 anthropized vegetation, 8 wooded
        savannah, 9 water, 10 urban, 11 crop farming, 12 reforestation,
        13 mining; 255 = no data). Downloaded (5 MB) and converted the first
        time it is asked for (the conversion needs ~2 GB of memory once).

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    >>> lu.shape
    (2, 108, 128)
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    if name == "sao_lourenco":
        return _sao_lourenco()
    return Path(str(resources.files("landschaft.simulate") / "data" / EXAMPLES[name]))


def _sao_lourenco() -> Path:
    out = cache_dir() / "simulate" / "sao_lourenco.tif"
    if out.exists():
        return out
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from .._rdata import NA_INTEGER, read_rda

    rda = fetch("simulate/SaoLourencoBasin.rda", _SAO_LOURENCO_URL, _SAO_LOURENCO_SHA256)
    stack = read_rda(rda)["SaoLourencoBasin"]
    a = stack.attributes
    nrow, ncol = int(a["nrows"].value[0]), int(a["ncols"].value[0])
    ext = a["extent"].attributes
    xmin, ymax = float(ext["xmin"].value[0]), float(ext["ymax"].value[0])
    res_x = (float(ext["xmax"].value[0]) - xmin) / ncol
    res_y = (ymax - float(ext["ymin"].value[0])) / nrow
    crs = CRS.from_proj4(a["crs"].attributes["projargs"].value[0])
    layers = a["layers"].value
    names = [str(layer.attributes["data"].attributes["names"].value[0]).removeprefix("landscape_")
             for layer in layers]
    profile = {"driver": "GTiff", "width": ncol, "height": nrow, "count": len(layers), "dtype": "uint8",
               "nodata": 255, "crs": crs, "transform": from_origin(xmin, ymax, res_x, res_y), "compress": "deflate",
               "predictor": 2, "tiled": True, "blockxsize": 512, "blockysize": 512}
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        for k, layer in enumerate(layers, start=1):
            v = layer.attributes["data"].attributes["values"].value  # by rows, as raster stores them
            band = np.where(v == NA_INTEGER, 255, v).astype(np.uint8).reshape(nrow, ncol)
            dst.write(band, k)
            dst.set_band_description(k, names[k - 1])
    del stack, layers
    tmp.replace(out)
    return out
