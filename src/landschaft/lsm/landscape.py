# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics of categorical rasters (FRAGSTATS, as in landscapemetrics).

The raster is analysed once by the C++ core (patch labelling, per-patch
statistics, class adjacencies) and kept in a :class:`Landscape`; every metric
is then computed from those tables. Values match the R package
landscapemetrics (``lsm_p_*``, ``lsm_c_*``, ``lsm_l_*``).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .catalog import LEVELS, add_full_name, canonical_name, select_metrics
from .utils import raster_extent

__all__ = ["Landscape", "landscape_metrics"]

_INT32 = np.iinfo(np.int32)


def _read(source, band: int | None) -> xr.DataArray:
    from ..io import load_raster

    return load_raster(source, band=band, validate=None)


def _prepare(data, res, nodata, band) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float], Any]:
    """Class values (int32), NA mask, (x, y) cell size and a template for rasters."""
    if isinstance(data, (str, os.PathLike)):
        data = _read(data, band)
    template = None
    mask = None
    if isinstance(data, xr.DataArray):
        da = data
        if "band" in da.dims:
            if da.sizes["band"] != 1 and band is None:
                raise ValueError("the raster has several bands; select one with band=")
            da = da.sel(band=band) if band is not None else da.isel(band=0)
        if da.ndim != 2:
            raise ValueError(f"expected a 2-D raster, got dimensions {da.dims}")
        template = da
        if res is None:
            try:
                rx, ry = da.rio.resolution()
                res = (abs(float(rx)), abs(float(ry)))
            except Exception:  # no georeferencing: unit cells
                res = None
            try:
                crs = da.rio.crs
            except Exception:
                crs = None
            if crs is not None and crs.is_geographic:
                warnings.warn(
                    "the raster has a geographic CRS: areas and distances are in squared degrees and degrees "
                    "(as in landscapemetrics); project it for metric units",
                    stacklevel=3,
                )
        if nodata is None:
            nodata = da.rio.nodata if hasattr(da, "rio") else None
        values = np.asarray(da.values)
    elif np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data).copy()
        values = np.ma.getdata(data)
    else:
        values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D raster, got shape {values.shape}")

    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask = values == nodata if mask is None else mask | (values == nodata)
    if np.issubdtype(values.dtype, np.floating):
        nan = np.isnan(values)
        mask = nan if mask is None else mask | nan
        valid = values[~mask]
        if valid.size and not np.array_equal(valid, np.trunc(valid)):
            warnings.warn("non-integer values are truncated to integers (as terra::as.int)", stacklevel=3)
        values = np.trunc(np.where(mask, 0, values))
    elif not (np.issubdtype(values.dtype, np.integer) or values.dtype == bool):
        raise TypeError(f"expected a numeric raster, got dtype {values.dtype}")
    if values.dtype.kind == "f" or (values.dtype.kind in "iu" and values.dtype.itemsize >= 4
                                    and not values.dtype == np.int32):
        # wider than int32 (int32 itself is checked by the C++ core)
        valid = values if mask is None else values[~mask]
        if valid.size and (valid.min() < _INT32.min + 1 or valid.max() > _INT32.max):
            raise ValueError("class values must fit in a 32-bit integer")
    values = np.ascontiguousarray(values, dtype=np.int32)
    if res is None:
        res = (1.0, 1.0)
    elif np.isscalar(res):
        res = (float(res), float(res))
    res = (float(res[0]), float(res[1]))
    if res[0] <= 0 or res[1] <= 0:
        raise ValueError(f"cell size must be positive, got {res}")
    return values, (mask.view(np.uint8) if mask is not None and mask.any() else None), res, template


# landscapemetrics' metric options and their defaults.
OPTION_DEFAULTS = {
    "directions": 8,
    "count_boundary": False,
    "consider_boundary": False,
    "edge_depth": 1,
    "cell_center": False,
    "classes_max": None,
    "neighbourhood": 4,
    "ordered": True,
    "base": "log2",
    "convention": "landscapemetrics",
    "search_radius": None,
    "connect_threshold": None,
    "contrast": None,
    "similarity": None,
    "boundary_contrast": 0.0,
}
_LEVEL_CODE = {"patch": 0, "class": 1, "landscape": 2}


def metric_options(**options) -> dict[str, Any]:
    """The metric options (defaults filled in, types checked)."""
    unknown = set(options) - set(OPTION_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}; expected {sorted(OPTION_DEFAULTS)}")
    out = {**OPTION_DEFAULTS, **options}
    out["directions"] = int(out["directions"])
    out["count_boundary"] = bool(out["count_boundary"])
    out["consider_boundary"] = bool(out["consider_boundary"])
    out["edge_depth"] = int(out["edge_depth"])
    out["cell_center"] = bool(out["cell_center"])
    out["neighbourhood"] = int(out["neighbourhood"])
    out["ordered"] = bool(out["ordered"])
    if out["convention"] not in ("landscapemetrics", "fragstats"):
        raise ValueError(f"convention must be 'landscapemetrics' or 'fragstats', got {out['convention']!r}")
    for key in ("search_radius", "connect_threshold"):
        if out[key] is not None:
            out[key] = float(out[key])
            if not out[key] >= 0:
                raise ValueError(f"{key} must be a non-negative distance")
    out["boundary_contrast"] = float(out["boundary_contrast"])
    out["contrast"] = weight_table(out["contrast"], "contrast")
    out["similarity"] = weight_table(out["similarity"], "similarity")
    return out


def weight_table(weights, what: str) -> list | None:
    """(class a, class b, weight) triplets of a class x class table: a
    DataFrame (index and columns = class values), a dict {(a, b): w} or
    {a: {b: w}}, or triplets."""
    if weights is None:
        return None
    rows = []
    if isinstance(weights, pd.DataFrame):
        for a in weights.index:
            for b in weights.columns:
                v = weights.loc[a, b]
                if pd.notna(v):
                    rows.append((float(a), float(b), float(v)))
    elif isinstance(weights, dict):
        for key, v in weights.items():
            if isinstance(v, dict):
                rows += [(float(key), float(b), float(w)) for b, w in v.items()]
            else:
                a, b = key
                rows.append((float(a), float(b), float(v)))
    else:
        for t in weights:
            a, b, v = t
            rows.append((float(a), float(b), float(v)))
    for a, b, v in rows:
        if not (a.is_integer() and b.is_integer()):
            raise ValueError(f"{what} weights need integer class values, got {(a, b)}")
    return rows


def chosen_metrics(level=None, metric=None, name=None, type=None, what=None, extras: bool = False) -> pd.DataFrame:
    """Selected catalogue rows in calculate_lsm's order (level, metric)."""
    return (
        select_metrics(level, metric, name, type, what, extras).sort_values(["level", "metric"]).reset_index(drop=True)
    )


def core_options(options: dict, n_threads: int | None) -> dict:
    """Options passed to the C++ core: the metric options and the threads."""
    if n_threads is not None:
        n_threads = int(n_threads)
        if n_threads < 0:
            raise ValueError(f"n_threads must be >= 0, got {n_threads}")
    from .._threads import resolve

    return {**options, "n_threads": resolve(n_threads)}


def metric_specs(chosen: pd.DataFrame) -> tuple[list[int], list[str]]:
    return [_LEVEL_CODE[lv] for lv in chosen["level"]], list(chosen["metric"])


def long_frame(table: dict, chosen: pd.DataFrame) -> pd.DataFrame:
    """calculate_lsm's table from the C++ long table of the metrics `chosen`."""
    level_names = sorted(set(chosen["level"]))
    metric_order = list(dict.fromkeys(chosen["metric"]))
    spec_level = np.array([level_names.index(lv) for lv in chosen["level"]], dtype=np.int8)
    spec_metric = np.array([metric_order.index(m) for m in chosen["metric"]], dtype=np.int16)
    spec = table["metric"]
    return pd.DataFrame(
        {
            "level": pd.Categorical.from_codes(spec_level[spec], level_names),
            "class": pd.arrays.IntegerArray(table["class"].astype(np.int64), table["class_na"]),
            "id": pd.arrays.IntegerArray(table["id"].astype(np.int64), table["id_na"]),
            "metric": pd.Categorical.from_codes(spec_metric[spec], metric_order),
            "value": table["value"],
        }
    )


class Landscape:
    """Categorical raster prepared for landscape metrics.

    Patches and adjacencies are computed once, on first use, and shared by
    every metric. Metric names are the landscapemetrics / FRAGSTATS
    abbreviations (see :func:`landschaft.list_metrics`).

    Parameters
    ----------
    data
        Categorical map: a file path, an ``xarray.DataArray`` (cell size, NA
        value and CRS are taken from it), a numpy array or a masked array.
        Floats are truncated to integers; NaN is NA.
    res
        Cell size ``(x, y)`` (or one number) in map units. Default: from the
        raster, else 1.
    nodata
        Value treated as NA (default: the raster's nodata value).
    directions
        Patch connectivity: 8 (queen's case, default) or 4 (rook's case).
    count_boundary
        Count the landscape boundary (and NA cells) as edge in ``te``/``ed``.
    consider_boundary, edge_depth
        Core-area options (cells at the boundary may be core; distance in
        cells from the patch edge).
    cell_center
        ``gyrate``: use the patch cell nearest to the centroid.
    classes_max
        Potential number of classes, for ``rpr``.
    neighbourhood, ordered, base
        Complexity metrics (``ent``, ``condent``, ``joinent``, ``mutinf``,
        ``relmutinf``): 4 or 8 neighbours, ordered pairs, and the logarithm
        base ("log2", "log" or "log10").
    band
        Band to read from multi-band files.
    block_rows
        Out-of-core computation for rasters larger than memory: ``data`` must
        be a file, read by blocks of this many rows (rounded up to a multiple
        of 64; ``"auto"``: about 4 million cells per block). Only the patch
        tables stay in memory. Results are identical to the in-memory ones;
        ``labels`` and ``to_raster`` are not available.
    n_threads
        Threads of the C++ core (default: all cores but one).
    convention
        ``"landscapemetrics"`` (default) or ``"fragstats"``: with non-square
        cells, left/right cell sides are as long as the cell height (the
        physical perimeter; landscapemetrics takes the width); ``dcore_*``
        summarise the sizes (ha) of the disjunct core areas instead of their
        number per patch; ``shape`` is the raster SHAPE (perimeter over the
        smallest perimeter of as many cells); ``_sd`` and ``_cv`` use the
        standard deviation over n (landscapemetrics: R's ``sd()``, over n - 1);
        the ``circle`` encloses the whole patch (landscapemetrics: the corners
        of its first and last rows and columns only). Cells that are not
        square keep ``shape`` and ``lsi`` as .25 perimeter / sqrt(area), as
        pylandstats does.
    search_radius
        ``prox`` and ``simi`` (FRAGSTATS extras): patches within this
        distance, in map units (NaN without it).
    connect_threshold
        ``connect``: patches of a class within this distance are connected.
    contrast, boundary_contrast
        Edge contrast weights (0-1) between classes for ``econ``, ``cwed`` and
        ``teci``: a DataFrame (index and columns = class values), a dict
        ``{(a, b): w}`` or ``{a: {b: w}}`` (symmetric; missing pairs give NaN),
        and the weight of edges to NA and the map edge (default 0).
    similarity
        Similarity weights between classes for ``simi``, as ``contrast``
        (default: 1 within a class, 0 between classes, i.e. ``simi = prox``).

    Notes
    -----
    Metric names are landscapemetrics' abbreviations; FRAGSTATS abbreviations
    in capitals (``"SHDI"``) and the full names of pylandstats / FRAGSTATS
    (``"shannon_diversity_index"``, ``"total_area"``) are accepted too.

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    >>> lc = ls.Landscape(landcover)
    >>> lc.n_patches
    967
    >>> lc.class_metrics(["ca", "np"]).head(2)
       class         ca     np
    0      1    174.500    2.0
    1      2  27058.125  201.0
    >>> float(lc.landscape_metrics(["shdi"])["shdi"].iloc[0])
    1.8096...
    >>> long = lc.metrics(type="aggregation metric")   # long format, like calculate_lsm
    """

    def __init__(
        self,
        data,
        *,
        res=None,
        nodata=None,
        directions: int = 8,
        count_boundary: bool = False,
        consider_boundary: bool = False,
        edge_depth: int = 1,
        cell_center: bool = False,
        classes_max: float | None = None,
        neighbourhood: int = 4,
        ordered: bool = True,
        base: str = "log2",
        band: int | None = None,
        block_rows: int | str | None = None,
        n_threads: int | None = None,
        convention: str = "landscapemetrics",
        search_radius: float | None = None,
        connect_threshold: float | None = None,
        contrast=None,
        similarity=None,
        boundary_contrast: float = 0.0,
    ):
        self.options = metric_options(
            directions=directions,
            count_boundary=count_boundary,
            consider_boundary=consider_boundary,
            edge_depth=edge_depth,
            cell_center=cell_center,
            classes_max=classes_max,
            neighbourhood=neighbourhood,
            ordered=ordered,
            base=base,
            convention=convention,
            search_radius=search_radius,
            connect_threshold=connect_threshold,
            contrast=contrast,
            similarity=similarity,
            boundary_contrast=boundary_contrast,
        )
        self.n_threads = n_threads
        self._tiled = None
        if block_rows is not None:
            from .tiled import TiledReader

            self._tiled = TiledReader(
                data, band=band, nodata=nodata, res=res, block_rows=block_rows,
                options=core_options(self.options, n_threads),
            )
            self._template = None
            self.resolution = self._tiled.res
            self.shape = self._tiled.shape
            self._extent = raster_extent(self._tiled.transform, self.shape, self.resolution)
            self._core = self._tiled.engine.landscape()
            return
        values, mask, res, template = _prepare(data, res, nodata, band)
        self._template = template
        self.resolution = res
        self.shape = values.shape
        self._extent = raster_extent(_transform(template), self.shape, res)
        self._core = _core.LsmLandscape(values, mask, res[0], res[1], core_options(self.options, n_threads))

    def __repr__(self) -> str:
        return (
            f"Landscape(shape={self.shape}, resolution={self.resolution}, classes={self.classes.tolist()}, "
            f"directions={self.options['directions']})"
        )

    # -- data ---------------------------------------------------------------

    @property
    def classes(self) -> np.ndarray:
        """Class values present in the map, in increasing order.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.classes[:5].tolist()
        [1, 2, 3, 4, 5]
        """
        return self._core.class_values

    @property
    def n_patches(self) -> int:
        """Number of patches of all classes.

        Examples
        --------
        >>> import landschaft as ls
        >>> ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None)).n_patches
        967
        """
        return self._core.n_patches()

    @functools.cached_property
    def patch_classes(self) -> np.ndarray:
        """Class value of every patch (patch ``id`` = position + 1).

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.patch_classes[:3].tolist()
        [1, 1, 2]
        """
        return self.classes[self._core.patch_class()] if self.classes.size else np.empty(0, np.int32)

    @property
    def out_of_core(self) -> bool:
        """Whether the raster is read by blocks (``block_rows``).

        Examples
        --------
        >>> import landschaft as ls
        >>> ls.Landscape(ls.example_data("clc"), block_rows=256).out_of_core
        True
        """
        return self._tiled is not None

    def labels(self) -> xr.DataArray | np.ndarray:
        """Map of patch ids.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Patch ids 1..n (the ``id`` of the patch tables), 0 where NA; same grid
            as the input. Not available out of core.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> ids = lc.labels()
        >>> int(ids.max()) == lc.n_patches
        True
        """
        self._no_raster("labels")
        return self._wrap(self._core.labels(), "patches")

    def _no_raster(self, what: str) -> None:
        if self._tiled is not None:
            raise NotImplementedError(f"{what} is not available for an out-of-core landscape (block_rows)")

    def _ensure(self, metrics) -> None:
        """Out of core: read the file again for the metrics that need it."""
        if self._tiled is not None:
            self._tiled.ensure(metrics)

    def to_raster(self, metric, level: str | None = None, path=None, **write_options):
        """Metric painted on the map (NaN where NA): a patch-level metric on
        its patches (landscapemetrics' ``spatialize_lsm``), a class-level one
        on the cells of each class.

        Parameters
        ----------
        metric
            A metric name, or a list of them (then a dict ``{name: raster}``).
        level
            "patch" or "class"; by default the patch level if the metric
            exists there, else the class level.
        path
            Write the raster to this GeoTIFF (``write_raster``; with a list, a
            directory: one ``<metric>.tif`` each), and return the path(s).

        Returns
        -------
        xarray.DataArray, numpy.ndarray, dict or path
            The raster (same form as the input), a dict of them for a list of
            metrics, or the path(s) written.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> area = lc.to_raster("area")                  # every patch painted with its area (ha)
        >>> pland = lc.to_raster("pland", level="class")
        >>> path = lc.to_raster("shape", path="shape.tif")
        """
        self._no_raster("to_raster")
        if not isinstance(metric, str):
            names = list(metric)
            if path is not None:
                folder = Path(path)
                folder.mkdir(parents=True, exist_ok=True)
                return {m: self.to_raster(m, level, folder / f"{m}.tif", **write_options) for m in names}
            return {m: self.to_raster(m, level) for m in names}
        if level is None:
            level = "patch" if canonical_name(metric, "patch") in _patch_names() else "class"
        labels = np.asarray(self._core.labels())
        if level == "patch":
            values = np.concatenate([[np.nan], self._patch_values(metric)])
            out = values[labels]
        elif level == "class":
            (name,), _ = _names(metric, "class")
            self._ensure([name])
            per_class = self._core.class_metric(name)
            pclass = np.concatenate([[-1], self._core.patch_class()])
            cell_class = pclass[labels]
            out = np.where(cell_class >= 0, per_class[np.maximum(cell_class, 0)], np.nan)
        else:
            raise ValueError("level must be 'patch' or 'class'")
        raster = self._wrap(out, metric)
        if path is None:
            return raster
        from ..io import write_raster

        if isinstance(raster, np.ndarray):
            xmin, _, _, ymax = self._extent
            from rasterio.transform import Affine

            write_options.setdefault("transform", Affine(self.resolution[0], 0, xmin, 0, -self.resolution[1], ymax))
        return write_raster(raster, path, colormap=None, **write_options)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Extent of the raster: ``(xmin, xmax, ymin, ymax)`` in map units.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> xmin, xmax, ymin, ymax = lc.extent
        >>> xmax - xmin
        50000.0
        """
        return self._extent

    def patches(self, classes=None) -> dict:
        """The patches of each class, each class on a raster of its own.

        Parameters
        ----------
        classes
            Class values to return (default: all).

        Returns
        -------
        dict
            ``{class: raster}``: patch ids on the cells of the class, NaN elsewhere.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> forests = lc.patches(classes=[23, 24])
        >>> sorted(forests)
        [23, 24]
        """
        self._no_raster("patches")
        labels = np.asarray(self._core.labels())
        pclass = np.concatenate([[np.iinfo(np.int32).min], self.patch_classes])
        cell_class = pclass[labels]
        wanted = self.classes if classes is None else np.atleast_1d(classes)
        missing = set(np.asarray(wanted).tolist()) - set(self.classes.tolist())
        if missing:
            raise ValueError(f"class(es) not present in the landscape: {sorted(missing)}")
        out = {}
        for c in wanted:
            arr = np.where((labels > 0) & (cell_class == c), labels, np.nan)
            out[int(c)] = self._wrap(arr, f"class_{int(c)}")
        return out

    def centroids(self, cell_center: bool = False) -> pd.DataFrame:
        """Centroid of every patch, like landscapemetrics' ``get_centroids``.

        The mean of the cell-centre coordinates of the patch (map
        coordinates, as terra computes them; the same bits as
        landscapemetrics). With ``cell_center`` the patch cell nearest to that
        point instead (several rows on ties, as landscapemetrics returns).

        Returns
        -------
        pandas.DataFrame
            ``class``, ``id``, ``x``, ``y``.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.centroids().columns.tolist()
        ['class', 'id', 'x', 'y']
        """
        self._no_raster("centroids")
        t = self._core.centroids(*self._extent, bool(cell_center))
        patch = t["patch"]
        return pd.DataFrame({"class": self.patch_classes[patch], "id": patch + 1, "x": t["x"], "y": t["y"]})

    def nearest_neighbours(self, return_id: bool = False) -> pd.DataFrame:
        """Distance from every patch to the nearest patch of its class, like
        landscapemetrics' ``get_nearestneighbour`` on the patches of each class
        (the distance is the ``enn`` metric: between cell centres).

        With ``return_id``, the ``id`` of the neighbour patch: one row per
        neighbour when several are equally near, found with landscapemetrics'
        own search (same ties). NaN / NA for classes with a single patch.

        Returns
        -------
        pandas.DataFrame
            ``class``, ``id``, ``dist`` (and ``id_neighbour``).

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> nn = lc.nearest_neighbours(return_id=True)
        >>> nn.columns.tolist()
        ['class', 'id', 'dist', 'id_neighbour']
        """
        self._no_raster("nearest_neighbours")
        t = self._core.nearest_neighbours(bool(return_id))
        patch = t["patch"]
        df = pd.DataFrame({"class": self.patch_classes[patch], "id": patch + 1, "dist": t["dist"]})
        if return_id:
            nb = t["neighbour"].astype(np.int64) + 1
            df["id_neighbour"] = pd.arrays.IntegerArray(nb, t["neighbour"] < 0)
        return df

    def circumscribing_circles(self, level: str = "patch") -> pd.DataFrame:
        """Smallest circle around every patch (or every class), like
        landscapemetrics' ``get_circumscribingcircle``: it encloses the corners
        of the cells in the first and last rows and columns, as for the
        ``circle`` metric (with ``convention="fragstats"``, every cell of the
        patch).

        Returns
        -------
        pandas.DataFrame
            ``class``, ``id`` (patch level), ``diameter``, ``center_x``,
            ``center_y`` (map coordinates).

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> circles = lc.circumscribing_circles()
        >>> "diameter" in circles.columns
        True
        """
        if level not in ("patch", "class"):
            raise ValueError("level must be 'patch' or 'class'")
        self._no_raster("circumscribing_circles")
        c = self._core.circles(level == "class")
        xmin, _, ymin, _ = self._extent
        ny = self.shape[0]
        center_x = c["cx"] + xmin
        center_y = (ny * self.resolution[1] - c["cy"]) + ymin
        if level == "class":
            return pd.DataFrame({"class": self.classes, "diameter": c["r"] * 2, "center_x": center_x,
                                 "center_y": center_y})
        return pd.DataFrame({"class": self.patch_classes, "id": np.arange(1, self.n_patches + 1),
                             "diameter": c["r"] * 2, "center_x": center_x, "center_y": center_y})

    def adjacencies(self, neighbourhood=4, what: str = "full", upper: bool = False) -> pd.DataFrame:
        """Cell adjacencies between classes, like landscapemetrics'
        ``get_adjacencies``: the number of (cell, neighbour) pairs, row = class
        of the cell, column = class of the neighbour (NA cells left out).

        Parameters
        ----------
        neighbourhood
            4, 8, or a matrix with 1 for the neighbours and 0 for the focal
            cell (read transposed, as landscapemetrics does).
        what
            "full" (counts), or only "like" (diagonal), "unlike" (off-diagonal)
            or "triangle" (with the diagonal) of the lower triangle (upper
            with ``upper``), NaN elsewhere.

        Returns
        -------
        pandas.DataFrame
            Class x class counts (index: class of the cell, columns: class of the
            neighbour).

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> table = lc.adjacencies(4)
        >>> table.shape == (len(lc.classes), len(lc.classes))
        True
        """
        self._no_raster("adjacencies")
        from .utils import adjacency_frame, neighbourhood_offsets

        counts = self._core.cooccurrence(neighbourhood_offsets(neighbourhood))
        return adjacency_frame(counts, self.classes, what, upper)

    def boundaries(self, consider_boundary: bool = False, edge_depth: int = 1, as_na: bool = False,
                   patch_id: bool = False):
        """Boundary cells of the patches.

        1 for the cells within ``edge_depth`` cells (4 neighbours) of another class,
        of NA or, unless ``consider_boundary``, of the map edge; 0 inside. The
        cells that are not boundary are the core cells of the core-area metrics.

        Parameters
        ----------
        consider_boundary
            Do not count the map edge as a boundary.
        edge_depth
            Width of the boundary, in cells.
        as_na
            NaN instead of 0 for the inner cells.
        patch_id
            The patch id instead of 1 on boundary cells.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Same grid as the map, NaN where it is NA.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> edges = lc.boundaries(edge_depth=2)
        >>> sorted(int(v) for v in set(edges.values[~edges.isnull().values].ravel()))
        [0, 1]
        """
        self._no_raster("boundaries")
        b = np.asarray(self._core.patch_boundaries(bool(consider_boundary), int(edge_depth))).astype(float)
        b[b < 0] = np.nan
        if patch_id:
            labels = np.asarray(self._core.labels())
            b = np.where(b == 1, labels, b)
        if as_na:
            b[b == 0] = np.nan
        return self._wrap(b, "boundaries")

    def show_patches(self, classes="global", **kwargs):
        """Draw the patches (needs matplotlib); see :func:`show_patches`.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> fig = lc.show_patches()
        """
        from .plotting import show_patches

        return show_patches(self, classes, **kwargs)

    def show_cores(self, classes="all", **kwargs):
        """Draw the core and edge cells of every class (needs matplotlib); see
        :func:`show_cores`.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None)[:300, :300])
        >>> fig = lc.show_cores(classes=[12, 23])
        """
        from .plotting import show_cores

        return show_cores(self, classes, **kwargs)

    def show_lsm(self, metric: str, classes="global", **kwargs):
        """Draw a patch metric on the patches (needs matplotlib); see :func:`show_lsm`.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> fig = lc.show_lsm("area")
        """
        from .plotting import show_lsm

        return show_lsm(self, metric, classes, **kwargs)

    def extract_metrics(self, y, *, extract_id=None, metric=None, name=None, type=None, what=None,
                        extras: bool = False, verbose: bool = True) -> pd.DataFrame:
        """Patch metrics of the patches under points; see :func:`extract_metrics`.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> xmin, xmax, ymin, ymax = lc.extent
        >>> points = [[xmin + 1000, ymin + 1000], [xmax - 1000, ymax - 1000]]
        >>> table = lc.extract_metrics(points, what=["lsm_p_area"])
        >>> table["extract_id"].tolist()
        [1, 2]
        """
        self._no_raster("extract_metrics")
        from .utils import _order, extract_frame

        level = "patch" if what is None else None
        chosen = chosen_metrics(level, metric, name, type, what, extras)
        if set(chosen["level"]) != {"patch"}:
            raise ValueError("extract_metrics only takes patch level metrics")
        return _order(extract_frame(self, y, extract_id, chosen, verbose), False)

    def _wrap(self, arr: np.ndarray, name: str):
        if self._template is None:
            return arr
        da = self._template.copy(data=arr)
        da.name = name
        da.attrs = {}
        da.encoding = {}
        if np.issubdtype(arr.dtype, np.floating):
            da = da.rio.write_nodata(np.nan, encoded=False)
        else:
            da = da.rio.write_nodata(0, encoded=False)
        return da

    # -- metrics ------------------------------------------------------------

    def _patch_values(self, metric: str) -> np.ndarray:
        (metric,), _ = _names(metric, "patch")
        self._ensure([metric])
        return self._core.patch_metric(metric)

    def patch_metrics(self, metrics: Iterable[str] | str | None = None) -> pd.DataFrame:
        """Patch-level metrics, one row per patch.

        Parameters
        ----------
        metrics
            Metric names (default: all patch metrics). Upper-case and full names
            are accepted; the columns keep the names as given.

        Returns
        -------
        pandas.DataFrame
            ``class``, ``id`` and one column per metric.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> table = lc.patch_metrics(["area", "shape"])
        >>> table.columns.tolist()
        ['class', 'id', 'area', 'shape']
        """
        metrics, columns = _names(metrics, "patch")
        self._ensure(metrics)
        df = pd.DataFrame({"class": self.patch_classes, "id": np.arange(1, self.n_patches + 1)})
        for m, col in zip(metrics, columns):
            df[col] = self._core.patch_metric(m)
        return df

    def write_patch_metrics(self, path, metrics: Iterable[str] | str | None = None, *,
                            chunk_rows: int = 1_000_000, **kwargs) -> Path:
        """Write :meth:`patch_metrics` to a file, chunk by chunk.

        The table is never built in memory: for maps with tens of millions of
        patches.

        Parameters
        ----------
        path
            ``.parquet`` (one row group per chunk; needs pyarrow), ``.csv`` or
            ``.csv.gz``.
        metrics
            As in :meth:`patch_metrics`.
        chunk_rows
            Patches per chunk.
        **kwargs
            Passed to the CSV or Parquet writer.

        Returns
        -------
        pathlib.Path
            The file written.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.write_patch_metrics("patches.csv", ["area", "perim"]).name
        'patches.csv'
        """
        from .export import TableSink

        metrics, columns = _names(metrics, "patch")
        chunk_rows = _chunk_rows(chunk_rows)
        self._ensure(metrics)
        values = [self._core.patch_metric(m) for m in metrics]
        classes = self.patch_classes
        n = self.n_patches
        with TableSink(path, **kwargs) as sink:
            for start in range(0, max(n, 1), chunk_rows):
                end = min(n, start + chunk_rows)
                frame = pd.DataFrame({"class": classes[start:end], "id": np.arange(start + 1, end + 1)})
                for v, col in zip(values, columns):
                    frame[col] = v[start:end]
                sink.write(frame)
        return sink.path

    def write_metrics(self, path, level=None, metric=None, name=None, type=None, what=None, *,
                      extras: bool = False, chunk_rows: int = 1_000_000, **kwargs) -> Path:
        """Write :meth:`metrics` (long format) to a file, without building it in memory.

        One metric at a time, the patch metrics chunk by chunk; the same rows and
        types as :meth:`metrics`.

        Parameters
        ----------
        path
            ``.parquet`` (needs pyarrow), ``.csv`` or ``.csv.gz``.
        level, metric, name, type, what, extras
            Selection, as in :meth:`metrics`.
        chunk_rows
            Patches per chunk.
        **kwargs
            Passed to the CSV or Parquet writer.

        Returns
        -------
        pathlib.Path
            The file written.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> path = lc.write_metrics("metrics.csv.gz", level=["class", "landscape"])
        """
        from .export import TableSink

        chosen = chosen_metrics(level, metric, name, type, what, extras)
        chunk_rows = _chunk_rows(chunk_rows)
        self._ensure(chosen["metric"])
        level_names = sorted(set(chosen["level"]))
        metric_order = list(dict.fromkeys(chosen["metric"]))
        n = self.n_patches
        with TableSink(path, **kwargs) as sink:
            for lv, m in zip(chosen["level"], chosen["metric"]):
                if lv == "patch" and n > 0:
                    values = self._core.patch_metric(m)
                    classes = self.patch_classes.astype(np.int64)
                    for start in range(0, n, chunk_rows):
                        end = min(n, start + chunk_rows)
                        rows = end - start
                        no_na = np.zeros(rows, dtype=bool)
                        sink.write(pd.DataFrame({
                            "level": pd.Categorical.from_codes(np.full(rows, level_names.index(lv), np.int8),
                                                               level_names),
                            "class": pd.arrays.IntegerArray(classes[start:end], no_na),
                            "id": pd.arrays.IntegerArray(np.arange(start + 1, end + 1, dtype=np.int64), no_na.copy()),
                            "metric": pd.Categorical.from_codes(np.full(rows, metric_order.index(m), np.int16),
                                                                metric_order),
                            "value": values[start:end],
                        }))
                else:
                    one = chosen[(chosen["level"] == lv) & (chosen["metric"] == m)]
                    frame = long_frame(self._core.long_table(*metric_specs(one)), one)
                    frame["level"] = pd.Categorical(frame["level"].astype(str), categories=level_names)
                    frame["metric"] = pd.Categorical(frame["metric"].astype(str), categories=metric_order)
                    sink.write(frame)
        return sink.path

    def class_metrics(self, metrics: Iterable[str] | str | None = None, *, extras: bool = False) -> pd.DataFrame:
        """Class-level metrics, one row per class.

        Parameters
        ----------
        metrics
            Metric names (default: all class metrics). Upper-case and full names
            are accepted; the columns keep the names as given.
        extras
            With the default selection, also the metrics landscapemetrics does not
            have (see :func:`list_metrics`).

        Returns
        -------
        pandas.DataFrame
            ``class`` and one column per metric.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.class_metrics(["ca", "pland", "np"]).columns.tolist()
        ['class', 'ca', 'pland', 'np']
        >>> lc.class_metrics(["CA", "area_am"]).columns.tolist()
        ['class', 'CA', 'area_am']
        """
        metrics, columns = _names(metrics, "class", extras)
        self._ensure(metrics)
        self._core.prepare([1] * len(metrics), metrics)
        df = pd.DataFrame({"class": self.classes})
        for m, col in zip(metrics, columns):
            df[col] = self._core.class_metric(m)
        return df

    def landscape_metrics(self, metrics: Iterable[str] | str | None = None, *, extras: bool = False) -> pd.DataFrame:
        """Landscape-level metrics, in a single row.

        Parameters
        ----------
        metrics, extras
            As in :meth:`class_metrics`.

        Returns
        -------
        pandas.DataFrame
            One row, one column per metric.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
        >>> lc.landscape_metrics(["ta", "np", "shdi"]).columns.tolist()
        ['ta', 'np', 'shdi']
        """
        metrics, columns = _names(metrics, "landscape", extras)
        self._ensure(metrics)
        self._core.prepare([2] * len(metrics), metrics)
        return pd.DataFrame({col: [self._core.landscape_metric(m)] for m, col in zip(metrics, columns)})

    def metrics(
        self, level=None, metric=None, name=None, type=None, what=None, *, extras: bool = False,
        full_name: bool = False,
    ) -> pd.DataFrame:
        """Selected metrics in long format, like landscapemetrics' ``calculate_lsm``.

                Selection as in :func:`landschaft.list_metrics` (all landscapemetrics
                metrics by default; ``extras=True`` adds the FRAGSTATS ones).
                ``full_name=True`` adds the columns ``name``, ``type`` and
                ``function_name`` of each metric (as ``calculate_lsm(full_name = TRUE)``).

                Returns
                -------
                pandas.DataFrame
                    Columns ``level``, ``class``, ``id``, ``metric``, ``value``
                    (``class``/``id`` are missing where they do not apply). An all-NA
                    map gives one missing value per metric, as in landscapemetrics.

            Examples
            --------
            >>> import landschaft as ls
            >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc"), validate=None))
            >>> table = lc.metrics(level="landscape", type="diversity metric")
            >>> table.columns.tolist()
            ['level', 'class', 'id', 'metric', 'value']
            >>> sorted(table["metric"].astype(str))[:3]
            ['msidi', 'msiei', 'pr']
        """
        chosen = chosen_metrics(level, metric, name, type, what, extras)
        self._ensure(chosen["metric"])
        df = long_frame(self._core.long_table(*metric_specs(chosen)), chosen)
        return add_full_name(df) if full_name else df


def _chunk_rows(chunk_rows) -> int:
    chunk_rows = int(chunk_rows)
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    return chunk_rows


def _patch_names() -> set:
    return set(select_metrics(level="patch", extras=True)["metric"])


def _transform(template):
    """The geotransform of a DataArray (None without one)."""
    if template is None:
        return None
    try:
        return template.rio.transform()
    except Exception:
        return None


def _names(metrics, level: str, extras: bool = False) -> tuple[list[str], list[str]]:
    """Canonical metric names at `level` and the column names (as asked)."""
    if metrics is None:
        names = list(select_metrics(level=level, extras=extras)["metric"])
        return names, names
    if isinstance(metrics, str):
        metrics = [metrics]
    columns = [str(m) for m in metrics]
    names = [canonical_name(m, level) for m in columns]
    _check_level(names, level, columns)
    return names, columns


def _check_level(metrics: Sequence[str], level: str, asked: Sequence[str] | None = None) -> None:
    available = set(select_metrics(level=level, extras=True)["metric"])
    asked = metrics if asked is None else asked
    unknown = [a for m, a in zip(metrics, asked) if m not in available]
    if unknown:
        other = [
            lv
            for lv in LEVELS
            if lv != level
            and {canonical_name(u, lv) for u in unknown} & set(select_metrics(level=lv, extras=True)["metric"])
        ]
        hint = f" (available at the {' and '.join(other)} level)" if other else ""
        raise ValueError(f"not {level}-level metric(s): {unknown}{hint}; see list_metrics(level={level!r})")


def landscape_metrics(
    data,
    level=None,
    metric=None,
    name=None,
    type=None,
    what=None,
    *,
    extras: bool = False,
    full_name: bool = False,
    n_threads: int | None = None,
    **options,
) -> pd.DataFrame:
    """Landscape metrics of one or several rasters, in long format.

    The one-call equivalent of landscapemetrics' ``calculate_lsm``.

    Parameters
    ----------
    data
        A raster (see :class:`Landscape`), or a list of rasters /
        :class:`~landschaft.RasterCollection` (a ``layer`` column numbers them
        from 1).
    level, metric, name, type, what
        Metric selection, as in :func:`landschaft.list_metrics`; all metrics
        by default.
    extras
        Also select the FRAGSTATS metrics missing in landscapemetrics
        (``_am``, ``_md``, ``_ra``, ...; see :func:`landschaft.list_metrics`).
    full_name
        Add the columns ``name``, ``type`` and ``function_name`` (as
        ``calculate_lsm(full_name = TRUE)``).
    n_threads
        Threads of the C++ core (default: all cores but one).
    **options
        :class:`Landscape` options (``directions``, ``count_boundary``,
        ``edge_depth``, ``res``, ``nodata``, ``block_rows`` for files larger
        than memory, ...).

    Returns
    -------
    pandas.DataFrame
        ``level``, ``class``, ``id``, ``metric``, ``value`` (plus ``layer``
        for several rasters). Several rasters are computed in parallel.

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    >>> table = ls.landscape_metrics(landcover, level="landscape", metric=["shdi", "contag"])
    >>> table[["metric", "value"]].round(2)
       metric  value
    0  contag  69.46
    1    shdi   1.81

    Several rasters (a ``layer`` column numbers them):

    >>> both = ls.landscape_metrics([landcover, landcover[:1000]], what=["lsm_l_np"])
    >>> both["layer"].tolist()
    [1, 2]
    """
    layers = as_layers(data)
    if layers is None:
        return Landscape(data, n_threads=n_threads, **options).metrics(level, metric, name, type, what, extras=extras,
                                                                       full_name=full_name)
    if options.get("block_rows") is not None:
        frames = [
            Landscape(layer, n_threads=n_threads, **options).metrics(level, metric, name, type, what, extras=extras)
            for layer in layers
        ]
        for i, df in enumerate(frames, start=1):
            df.insert(0, "layer", i)
        df = pd.concat(frames, ignore_index=True)
        return add_full_name(df) if full_name else df
    res, nodata, band = options.pop("res", None), options.pop("nodata", None), options.pop("band", None)
    opts = core_options(metric_options(**options), n_threads)
    chosen = chosen_metrics(level, metric, name, type, what, extras)
    rasters = []
    for layer in layers:
        values, mask, r, _ = _prepare(layer, res, nodata, band)
        rasters.append((values, mask, r[0], r[1]))
    table = _core.lsm_batch(rasters, opts, *metric_specs(chosen))
    df = long_frame(table, chosen)
    df.insert(0, "layer", table["item"].astype(np.int64) + 1)
    return add_full_name(df) if full_name else df


def as_layers(data) -> list | None:
    """The rasters of a list / RasterCollection; None for a single raster."""
    from ..collection import RasterCollection

    if isinstance(data, RasterCollection):
        return [data.load(i) for i in range(len(data))]
    if isinstance(data, (list, tuple)):
        return list(data)
    return None
