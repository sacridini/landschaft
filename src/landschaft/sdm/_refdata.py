# SPDX-License-Identifier: GPL-3.0-or-later
"""Geographic reference data for cleaning occurrence records, downloaded on first use.

The tables and polygons the R package CoordinateCleaner (GPL-3) ships:
country and province centroids and capitals (``countryref``), biodiversity
institutions (``institutions``), artificial hotspots (``aohi``) and the land /
sea of Natural Earth widened by one degree (``buffland``, ``buffsea``). They
are read from the package's archive on CRAN (pinned by checksum, so results
never change because a file did) and cached as Parquet / GeoPackage.
"""

from __future__ import annotations

import tarfile
from functools import lru_cache

from .. import _download, _rdata

__all__ = ["reference_data"]

_CC_VERSION = "3.0.1"
_CC_SHA256 = "2e0369fd00e36fbf69669844e16901be2ba78d8d9a49d04f1464a94242de412f"
_CC_URLS = [
    f"https://cloud.r-project.org/src/contrib/CoordinateCleaner_{_CC_VERSION}.tar.gz",
    f"https://cloud.r-project.org/src/contrib/Archive/CoordinateCleaner/CoordinateCleaner_{_CC_VERSION}.tar.gz",
    f"https://cran.r-project.org/src/contrib/Archive/CoordinateCleaner/CoordinateCleaner_{_CC_VERSION}.tar.gz",
]

#: name -> (source archive, member, R object, kind)
_DATA = {
    "countryref": ("cc", "CoordinateCleaner/data/countryref.rda", "countryref", "table"),
    "institutions": ("cc", "CoordinateCleaner/data/institutions.rda", "institutions", "table"),
    "aohi": ("cc", "CoordinateCleaner/data/aohi.rda", "aohi", "table"),
    "buffland": ("cc", "CoordinateCleaner/data/buffland.rda", "buffland", "vector"),
    "buffsea": ("cc", "CoordinateCleaner/data/buffsea.rda", "buffsea", "vector"),
}


def _archive(source: str):
    if source == "cc":
        return _download.fetch(f"sdm/CoordinateCleaner_{_CC_VERSION}.tar.gz", _CC_URLS, _CC_SHA256)
    raise KeyError(source)


@lru_cache(maxsize=None)
def reference_data(name: str, *, refresh: bool = False):
    """A geographic reference data set for cleaning occurrence records.

    Downloaded once (about 5 MB) from the CoordinateCleaner archive on CRAN
    and cached (see :func:`landschaft._download.cache_dir`).

    Parameters
    ----------
    name : {"countryref", "institutions", "aohi", "buffland", "buffsea"}
        ``"countryref"``: centroids of the countries and provinces and the
        country capitals (5305 rows: ``iso3``, ``iso2``, ``name``, ``type``,
        ``centroid.lon`` / ``.lat``, ``capital.lon`` / ``.lat``...);
        ``"institutions"``: 11 601 biodiversity institutions (herbaria,
        museums, zoos, botanical gardens, universities) with coordinates;
        ``"aohi"``: artificial hotspot occurrence inventory (Park et al. 2023);
        ``"buffland"`` / ``"buffsea"``: the land / the sea of Natural Earth
        (1:50 million) widened by one degree.
    refresh
        Rebuild the cached copy from the archive.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
    """
    import geopandas as gpd
    import pandas as pd

    if name not in _DATA:
        raise ValueError(f"unknown reference data {name!r}; available: {', '.join(_DATA)}")
    source, member, robj, kind = _DATA[name]
    cached = _download.cache_dir() / "sdm" / (f"{name}.parquet" if kind == "table" else f"{name}.gpkg")
    if cached.exists() and not refresh:
        return pd.read_parquet(cached) if kind == "table" else gpd.read_file(cached)
    with tarfile.open(_archive(source), "r:gz") as tar:
        data = tar.extractfile(member).read()
    tmp = cached.with_suffix(".rda.tmp")
    tmp.write_bytes(data)
    try:
        value = _rdata.to_python(_rdata.read_rda(tmp)[robj])
    finally:
        tmp.unlink()
    if kind == "table":
        value = pd.DataFrame(value)
        value.to_parquet(cached, index=False)
    else:
        if value.crs is None:  # Natural Earth longitude / latitude
            value = value.set_crs("EPSG:4326")
        value.to_file(cached, driver="GPKG")
    return value
