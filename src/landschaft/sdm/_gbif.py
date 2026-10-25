# SPDX-License-Identifier: GPL-3.0-or-later
"""Occurrence records from the GBIF occurrence search API."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

__all__ = ["download_gbif"]

API = "https://api.gbif.org/v1"
PAGE = 300           # the API's largest page
MAX_RECORDS = 100_000  # the search API's offset limit

# Darwin Core terms kept (search API name -> column)
_FIELDS = {
    "key": "gbifID",
    "species": "species",
    "scientificName": "scientificName",
    "taxonKey": "taxonKey",
    "decimalLongitude": "decimalLongitude",
    "decimalLatitude": "decimalLatitude",
    "coordinateUncertaintyInMeters": "coordinateUncertaintyInMeters",
    "coordinatePrecision": "coordinatePrecision",
    "countryCode": "countryCode",
    "stateProvince": "stateProvince",
    "locality": "locality",
    "elevation": "elevation",
    "year": "year",
    "eventDate": "eventDate",
    "basisOfRecord": "basisOfRecord",
    "occurrenceStatus": "occurrenceStatus",
    "institutionCode": "institutionCode",
    "collectionCode": "collectionCode",
    "catalogNumber": "catalogNumber",
    "recordedBy": "recordedBy",
    "datasetKey": "datasetKey",
    "license": "license",
}


def _get(path: str, params: dict, timeout: float = 60.0) -> dict:
    query = urllib.parse.urlencode([(k, v) for k, vs in params.items() if vs is not None
                                    for v in (vs if isinstance(vs, (list, tuple)) else [vs])])
    request = urllib.request.Request(f"{API}/{path}?{query}", headers={"User-Agent": "landschaft"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError as err:
        from .._download import connection_failed, offline_message

        if connection_failed(err):
            raise OSError(offline_message("GBIF (api.gbif.org)") + " Download the records there (download_gbif, or "
                          "a download from gbif.org) and give them to this computer as a table.") from err
        raise


def _taxon_key(species) -> tuple[int, dict]:
    if isinstance(species, (int, np.integer)):
        return int(species), {"usageKey": int(species)}
    match = _get("species/match", {"name": species, "verbose": "false"})
    if match.get("matchType") in (None, "NONE") or "usageKey" not in match:
        raise ValueError(f"GBIF found no taxon named {species!r}")
    key = match.get("acceptedUsageKey", match["usageKey"])  # synonyms -> the accepted name
    return int(key), match


def _area_wkt(area) -> str | None:
    """The bounding box of ``area`` as WKT, counter-clockwise (the order GBIF needs)."""
    if area is None:
        return None
    import geopandas as gpd

    geom = area.geometry if hasattr(area, "geometry") else gpd.GeoSeries(area)
    if getattr(geom, "crs", None) is not None:
        geom = geom.to_crs("EPSG:4326")
    x0, y0, x1, y1 = geom.total_bounds
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def download_gbif(species, *, country=None, area=None, year=None, basis_of_record=None,
                  max_uncertainty: float | None = None, limit: int | None = None,
                  has_geospatial_issue: bool = False, n_workers: int = 4) -> pd.DataFrame:
    """Occurrence records of a species from GBIF, with coordinates.

    Uses GBIF's occurrence search API: no account is needed, up to 100 000
    records per query (narrow it with ``country``, ``area`` or ``year``). For
    larger queries, and for the DOI a publication must cite, request a
    download on gbif.org and read it with :func:`landschaft.bio.read_occurrences`.

    Parameters
    ----------
    species : str or int
        Scientific name (matched to GBIF's backbone taxonomy; a synonym gives
        the records of the accepted name) or a GBIF taxon key.
    country : str or list of str, optional
        ISO 3166-1 alpha-2 country codes (``"BR"``).
    area : GeoDataFrame or geometry, optional
        Only the records in its bounding box (cut to the exact shape with
        :func:`accessible_area` or a spatial join afterwards).
    year : int, str or tuple, optional
        A year, or a range ``(1990, 2020)`` / ``"1990,2020"``.
    basis_of_record : str or list of str, optional
        E.g. ``"PRESERVED_SPECIMEN"``, ``"HUMAN_OBSERVATION"``.
    max_uncertainty : float, optional
        Drop the records whose coordinate uncertainty is above this (metres);
        records without one are kept.
    limit : int, optional
        At most this many records (default: all, up to 100 000).
    has_geospatial_issue : bool
        False (default): leave out the records GBIF flags with geospatial
        issues (zero coordinates, country mismatch...).
    n_workers : int
        Pages requested at the same time.

    Returns
    -------
    pandas.DataFrame
        One row per record: ``gbifID``, ``species``, ``scientificName``,
        ``decimalLongitude``, ``decimalLatitude``,
        ``coordinateUncertaintyInMeters``, ``countryCode``, ``year``,
        ``basisOfRecord``, ``institutionCode``, ``datasetKey``, ``issue`` and
        other Darwin Core terms. ``attrs["gbif"]`` keeps the query and the
        taxon match.

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.sdm.download_gbif("Tapirus terrestris", country="BR", limit=500)  # doctest: +SKIP
    >>> occ = ls.sdm.clean_occurrences(occ)                                        # doctest: +SKIP
    """
    key, match = _taxon_key(species)
    if isinstance(year, (tuple, list)):
        year = f"{year[0]},{year[1]}"
    params = {"taxonKey": key, "hasCoordinate": "true", "hasGeospatialIssue": str(bool(has_geospatial_issue)).lower(),
              "country": country, "geometry": _area_wkt(area), "year": year, "basisOfRecord": basis_of_record,
              "occurrenceStatus": "PRESENT"}
    first = _get("occurrence/search", {**params, "limit": PAGE, "offset": 0})
    total = min(int(first.get("count", 0)), MAX_RECORDS, limit if limit is not None else MAX_RECORDS)
    offsets = list(range(PAGE, total, PAGE))
    pages = [first]
    if offsets:
        with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
            pages += list(pool.map(lambda o: _get("occurrence/search", {**params, "limit": PAGE, "offset": o}),
                                   offsets))
    rows = [r for p in pages for r in p.get("results", [])][:total]
    table = pd.DataFrame([{col: r.get(k) for k, col in _FIELDS.items()} for r in rows],
                         columns=list(_FIELDS.values()))
    table["issue"] = [";".join(r.get("issues", [])) for r in rows]
    for col in ("decimalLongitude", "decimalLatitude", "coordinateUncertaintyInMeters", "coordinatePrecision",
                "elevation", "year"):
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if max_uncertainty is not None:
        u = table["coordinateUncertaintyInMeters"]
        table = table[u.isna() | (u <= max_uncertainty)]
    table = table.drop_duplicates(subset="gbifID").reset_index(drop=True)
    table.attrs["gbif"] = {"taxonKey": key, "match": match, "query": {k: v for k, v in params.items() if v is not None},
                           "count": int(first.get("count", 0)),
                           "truncated": int(first.get("count", 0)) > len(rows)}
    return table
