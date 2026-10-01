# SPDX-License-Identifier: GPL-3.0-or-later
"""landscapemetrics' utility functions: patch boundaries of raster values
(``get_boundaries``), metrics under points (``extract_lsm``), the raster
check (``check_landscape``) and metric correlations
(``calculate_correlation``). The per-patch utilities (centroids, nearest
neighbours, circumscribing circles, adjacencies, boundaries, patches) are
methods of :class:`~landschaft.Landscape`.
"""

from __future__ import annotations

from .._threads import threaded

import os
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .. import _core

__all__ = ["calculate_correlation", "check_landscape", "extract_metrics", "raster_boundaries"]

_INT32_MIN = np.iinfo(np.int32).min


# -- grid helpers ---------------------------------------------------------------


def raster_extent(transform, shape: tuple[int, int], res: tuple[float, float]) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) as terra derives it from a geotransform
    (``xmax = xmin + res * ncol``); without one, terra's extent of a matrix
    scaled by the cell size."""
    ny, nx = shape
    if transform is None:
        return 0.0, nx * res[0], 0.0, ny * res[1]
    if transform.b != 0 or transform.d != 0:
        raise ValueError("rotated rasters are not supported")
    xmin = transform.c
    xmax = xmin + transform.a * nx
    ymax = transform.f
    ymin = ymax + ny * transform.e
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return float(xmin), float(xmax), float(ymin), float(ymax)


def cell_of(extent, shape, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, column and inside-flag of points, as terra's cellFromXY (a point
    on a cell edge goes right / down, except on the last column / row)."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = shape
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(invalid="ignore"):
        row = np.floor((ymax - y) * (ny / (ymax - ymin)))
        col = np.floor((x - xmin) * (nx / (xmax - xmin)))
    row = np.where(y == ymin, ny - 1, row)
    col = np.where(x == xmax, nx - 1, col)
    inside = ~(np.isnan(row) | np.isnan(col)) & (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    return np.where(inside, row, 0).astype(np.int64), np.where(inside, col, 0).astype(np.int64), inside


def point_coords(y) -> np.ndarray:
    """(n, 2) coordinates of points: an array or point geometries."""
    from .sampling import _geometries

    if not hasattr(y, "__geo_interface__") and not isinstance(y, dict):
        try:
            arr = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    coords = []
    for g in _geometries(y):
        if g["type"] == "Point":
            coords.append(g["coordinates"][:2])
        elif g["type"] == "MultiPoint":
            coords.extend(c[:2] for c in g["coordinates"])
        else:
            raise ValueError(f"expected points, got {g['type']}")
    return np.asarray(coords, dtype=float).reshape(-1, 2)


def neighbourhood_offsets(neighbourhood) -> list[tuple[int, int]]:
    """(column, row) offsets of the neighbours: 4, 8, or a matrix with 1 for
    the neighbours and 0 for the focal cell (NA / anything else ignored), as
    landscapemetrics' rcpp_create_neighborhood reads it (a matrix is used
    transposed: rows of the matrix are columns of the raster)."""
    if np.isscalar(neighbourhood):
        if neighbourhood == 4:
            return [(0, -1), (-1, 0), (1, 0), (0, 1)]
        if neighbourhood == 8:
            return [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        raise ValueError("neighbourhood must be 4, 8 or a matrix (1 = neighbour, 0 = focal cell)")
    m = np.asarray(neighbourhood, dtype=float)
    if m.ndim != 2:
        raise ValueError("the neighbourhood matrix must be 2-D")
    centre = np.argwhere(m == 0)
    if len(centre) != 1:
        raise ValueError("the neighbourhood matrix needs exactly one 0 (the focal cell)")
    r0, c0 = centre[0]
    return [(int(r - r0), int(c - c0)) for r, c in np.argwhere(m == 1)]


def adjacency_frame(counts: np.ndarray, classes: np.ndarray, what: str, upper: bool) -> pd.DataFrame:
    """get_adjacencies' matrix: the counts, with NA outside the part `what`
    ("full", "like", "unlike" or "triangle"; upper or lower triangle)."""
    if what not in ("full", "like", "unlike", "triangle"):
        raise ValueError("what must be 'full', 'like', 'unlike' or 'triangle'")
    k = len(classes)
    rows, cols = np.indices((k, k))
    if what == "full":
        keep = np.ones((k, k), dtype=bool)
    elif what == "like":
        keep = rows == cols
    elif what == "unlike":
        keep = cols > rows if upper else rows > cols
    else:
        keep = cols >= rows if upper else rows >= cols
    values = counts.astype(np.int64) if what == "full" else np.where(keep, counts, np.nan)
    index = pd.Index(classes, name="class")
    return pd.DataFrame(values, index=index, columns=pd.Index(classes, name="neighbour"))


# -- get_boundaries of raster values ------------------------------------------------------


@threaded
def raster_boundaries(
    data,
    *,
    consider_boundary: bool = False,
    edge_depth: int = 1,
    as_na: bool = False,
    patch_id: bool = False,
    nodata=None,
    band: int | None = None,
):
    """landscapemetrics' ``get_boundaries`` on the values of a raster.

        The values are taken as they are (classes, or patch ids such as
        :meth:`Landscape.labels` or :meth:`Landscape.patches`). A cell is a
        boundary cell as in landscapemetrics' ``rcpp_get_boundaries``: the cell
        below holds another value or NA, or the cell above, left or right is NA;
        the map edge counts as NA unless ``consider_boundary``. As in
        landscapemetrics, ``patch_id`` is ignored (with a warning) when a value
        is 0. (On a map of
        several classes this marks only one side of a class change;
        :meth:`Landscape.boundaries` gives the boundaries of the patches of every
        class.) ``edge_depth`` > 1 peels the boundary off and repeats.

        Parameters
        ----------
        data
            A raster of classes or patch ids.
        consider_boundary
            Do not count the map edge as NA.
        edge_depth
            Width of the boundary, in cells.
        as_na
            NaN instead of 0 for the inner cells.
        patch_id
            The cell value instead of 1 on boundary cells.
        nodata, band
            Reading options.
        n_threads
            Threads (``None`` / 0: all cores but one).

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            1 on boundary cells (their value with ``patch_id``), 0 inside (NA with
            ``as_na``), NaN where the raster is NA.

        Examples
        --------
        >>> import landschaft as ls
        >>> lc = ls.Landscape(ls.load_raster(ls.example_data("clc")))
        >>> edges = ls.raster_boundaries(lc.labels(), edge_depth=1)
    """
    from .landscape import _prepare

    values, mask, _, template = _prepare(data, None, nodata, band)
    if patch_id and np.any(values[~mask.astype(bool)] == 0 if mask is not None else values == 0):
        warnings.warn("Not able to use original patch id because at least one id equals zero.", stacklevel=2)
        patch_id = False
    out = _core.lsm_raster_boundaries(values, mask, bool(consider_boundary), int(edge_depth), bool(patch_id))
    result = out.astype(float)
    result[out == _INT32_MIN] = np.nan
    if as_na:
        result[result == 0] = np.nan
    if template is None:
        return result
    da = template.copy(data=result)
    da.name = "boundaries"
    da.attrs = {}
    da.encoding = {}
    return da.rio.write_nodata(np.nan, encoded=False)


# -- extract_lsm --------------------------------------------------------------------------


def extract_frame(landscape, y, extract_id, chosen: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """extract_lsm on one Landscape: the patch metrics of the patches under
    the points, with the point's ``extract_id``."""
    from .landscape import long_frame, metric_specs

    xy = point_coords(y)
    n = len(xy)
    ids = np.arange(1, n + 1)
    if extract_id is not None:
        extract_id = list(extract_id)
        if len(extract_id) != n:
            if verbose:
                warnings.warn("Length of extract_id is not identical to length of y. Using 1...n as extract_id.",
                              stacklevel=3)
        else:
            ids = np.asarray(extract_id)
    rows, cols, inside = cell_of(landscape._extent, landscape.shape, xy[:, 0], xy[:, 1])
    labels = np.asarray(landscape._core.labels())
    patch = np.where(inside, labels[rows, cols], 0)
    points = pd.DataFrame({"extract_id": ids, "id": patch})
    points = points[points["id"] > 0].drop_duplicates()
    landscape._ensure(chosen["metric"])
    metrics = long_frame(landscape._core.long_table(*metric_specs(chosen)), chosen)
    metrics = metrics.assign(id=metrics["id"].astype("Int64"))
    out = metrics.merge(points.assign(id=points["id"].astype("Int64")), on="id", how="inner")
    return out


def _order(df: pd.DataFrame, with_layer: bool) -> pd.DataFrame:
    keys = (["layer"] if with_layer else []) + ["extract_id", "level", "metric", "class", "id"]
    tmp = df.assign(level=df["level"].astype(str), metric=df["metric"].astype(str))
    return df.loc[tmp.sort_values(keys, kind="stable").index].reset_index(drop=True)


def extract_metrics(
    data,
    y,
    *,
    extract_id=None,
    metric=None,
    name=None,
    type=None,
    what=None,
    extras: bool = False,
    full_name: bool = False,
    verbose: bool = True,
    n_threads: int | None = None,
    **options,
) -> pd.DataFrame:
    """Patch metrics of the patches under points, like landscapemetrics'
    ``extract_lsm``.

    Parameters
    ----------
    data
        A raster (see :class:`~landschaft.Landscape`) or a list of rasters /
        :class:`~landschaft.RasterCollection` (``layer`` column).
    y
        Points: an ``(n, 2)`` array of x/y coordinates or point geometries,
        in the raster's coordinates. A point on a cell edge belongs to the
        cell to the right / below (terra's rule); points outside the raster
        or on NA cells give no rows.
    extract_id
        Identifiers of the points (default 1..n).
    metric, name, type, what, extras
        Selection of patch-level metrics, as in
        :func:`landschaft.list_metrics` (all patch metrics by default).
    full_name
        Add the columns ``name``, ``type`` and ``function_name`` after
        ``value``.
    **options
        :class:`~landschaft.Landscape` options (``directions``, ``res``, ...).

    Returns
    -------
    pandas.DataFrame
        ``level``, ``class``, ``id``, ``metric``, ``value``, ``extract_id``
        (plus ``layer``), one row per point, patch and metric, ordered by
        (layer,) extract_id, level, metric, class and id.

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> import numpy as np
    >>> xmin, xmax, ymin, ymax = ls.Landscape(landcover).extent
    >>> points = np.array([[xmin + 5000, ymin + 5000], [xmax - 5000, ymax - 5000]])
    >>> table = ls.extract_metrics(landcover, points, what=["lsm_p_area", "lsm_p_shape"])
    >>> sorted(table["metric"].astype(str).unique())
    ['area', 'shape']
    """
    from .landscape import Landscape, as_layers, chosen_metrics

    level = "patch" if what is None else None
    chosen = chosen_metrics(level, metric, name, type, what, extras)
    if set(chosen["level"]) != {"patch"}:
        raise ValueError("extract_metrics only takes patch level metrics")
    from .catalog import add_full_name

    layers = as_layers(data)
    if layers is None:
        lm = Landscape(data, n_threads=n_threads, **options)
        out = _order(extract_frame(lm, y, extract_id, chosen, verbose), False)
    else:
        frames = []
        for i, layer in enumerate(layers, start=1):
            lm = Landscape(layer, n_threads=n_threads, **options)
            df = extract_frame(lm, y, extract_id, chosen, verbose)
            df.insert(0, "layer", i)
            frames.append(df)
        out = _order(pd.concat(frames, ignore_index=True), True)
    return add_full_name(out) if full_name else out


# -- check_landscape -----------------------------------------------------------------------

TICK = "✔"
CROSS = "✖"
QUESTION = "❓"


def _crs_info(crs) -> tuple:
    """("projected" | "geographic" | NA, units | NA), as landscapemetrics' proj_info."""
    if crs is None:
        return None, None
    from rasterio.crs import CRS

    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        return "geographic", "degrees"
    proj4 = crs.to_proj4()
    if "units=" in proj4:
        units = proj4.split("units=", 1)[1].split(" ", 1)[0]
    else:
        units = proj4.split(" ", 1)[0]
    return "projected", units


def _guess_crs(da):
    """terra's CRS of a raster file: when the file has none but a
    geotransform, and its extent fits lon/lat, terra assumes OGC:CRS84."""
    try:
        transform = da.rio.transform()
    except Exception:
        return None
    if transform.is_identity:
        return None
    xmin, xmax, ymin, ymax = raster_extent(transform, (da.sizes[da.rio.y_dim], da.sizes[da.rio.x_dim]), (1.0, 1.0))
    if xmin >= -180.1 and xmax <= 360.1 and ymin >= -90.1 and ymax <= 90.1:
        return "OGC:CRS84"
    return None


def _check_one(layer, verbose: bool, nodata, band) -> dict:
    from .landscape import _read

    crs = None
    if isinstance(layer, (str, os.PathLike)):
        layer = _read(layer, band)
    if hasattr(layer, "rio"):
        try:
            crs = layer.rio.crs
        except Exception:
            crs = None
        if crs is None:
            crs = _guess_crs(layer)
        if nodata is None:
            nodata = layer.rio.nodata
        if "band" in layer.dims:
            layer = layer.sel(band=band) if band is not None else layer.isel(band=0)
        values = np.asarray(layer.values)
        mask = np.zeros(values.shape, dtype=bool)
    elif np.ma.isMaskedArray(layer):
        values = np.ma.getdata(layer)
        mask = np.ma.getmaskarray(layer)
    else:
        values = np.asarray(layer)
        mask = np.zeros(values.shape, dtype=bool)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask = mask | (values == nodata)
    if np.issubdtype(values.dtype, np.floating):
        mask = mask | np.isnan(values)
    unique = np.unique(values[~mask])
    kind = None if unique.size == 0 else ("integer" if np.all(np.mod(unique, 1) == 0) else "non-integer")
    crs_kind, units = _crs_info(crs)
    class_ok = "ok" if kind == "integer" else "notok"
    n_ok = "ok" if unique.size <= 30 else "maybe"
    units_ok = "maybe" if units is None else ("ok" if units != "degrees" else "notok")
    if class_ok == "ok" and units_ok == "ok" and n_ok == "ok":
        ok = TICK
    elif class_ok == "notok" or units_ok == "notok":
        ok = CROSS
    else:
        ok = QUESTION
    if verbose:
        if kind is None:
            warnings.warn("All cell values are NA.", stacklevel=3)
        elif kind != "integer":
            warnings.warn("Caution: Land-cover classes must be decoded as integer values.", stacklevel=3)
        if units is None or units == "degrees":
            warnings.warn(
                "Caution: Coordinate reference system not metric - Units of results based on cellsizes and/or "
                "distances may be incorrect.",
                stacklevel=3,
            )
        if unique.size > 30:
            warnings.warn(
                "Caution: More than 30 land cover-classes - Please check if discrete land-cover classes are present.",
                stacklevel=3,
            )
    return {"crs": crs_kind, "units": units, "class": kind, "n_classes": int(unique.size), "OK": ok}


def check_landscape(data, verbose: bool = True, *, nodata=None, band: int | None = None) -> pd.DataFrame:
    """Is a raster suitable for landscape metrics? Like landscapemetrics'
        ``check_landscape``.

        Parameters
        ----------
        data
            A raster or a list of rasters.
        verbose
            Warn about the problems found.
        nodata, band
            Reading options, as in :class:`Landscape`.

        Returns
        -------
        pandas.DataFrame
            One row per raster: ``layer``, ``crs`` ("projected", "geographic" or
            NA), ``units`` (of the CRS, e.g. "m"), ``class`` ("integer",
            "non-integer", or NA when all values are NA), ``n_classes`` and
            ``OK``: ✔ (integer classes, metric CRS, at most 30 classes), ✖
            (non-integer values or degrees) or ❓ (no CRS, or more than 30
            classes). ``verbose`` warns about the problems. Like terra, a raster
            file without CRS whose extent fits longitudes and latitudes is taken
            as geographic.

        Examples
        --------
        >>> import landschaft as ls
        >>> check = ls.check_landscape(ls.example_data("clc"), verbose=False)
        >>> check[["crs", "units", "class", "n_classes"]]
                crs units    class  n_classes
        0  projected     m  integer         26
    """
    from .landscape import as_layers

    layers = as_layers(data)
    rows = [_check_one(layer, verbose, nodata, band) for layer in ([data] if layers is None else layers)]
    df = pd.DataFrame(rows, columns=["crs", "units", "class", "n_classes", "OK"])
    df.insert(0, "layer", np.arange(1, len(df) + 1))
    return df


# -- calculate_correlation -----------------------------------------------------------------


def _correlation_frame(wide: pd.DataFrame, method: str, diag: bool) -> pd.DataFrame:
    cor = _core.lsm_r_cor(np.ascontiguousarray(wide.to_numpy(dtype=float)), method)
    names = list(wide.columns)
    m = len(names)
    rows, cols = np.indices((m, m))
    keep = rows > cols if not diag else rows >= cols
    # column-major, as R's c(matrix)
    order = np.argsort(cols.ravel() * m + rows.ravel(), kind="stable")
    r, c, k, v = rows.ravel()[order], cols.ravel()[order], keep.ravel()[order], cor.ravel()[order]
    df = pd.DataFrame({"metric_1": [names[i] for i in r], "metric_2": [names[j] for j in c], "value": v})
    df = df[k & ~np.isnan(v)]
    return df.reset_index(drop=True)


def _xtabs(sub: pd.DataFrame, key: str) -> pd.DataFrame:
    """R's xtabs(value ~ key + metric): sums per key and metric (R's sum,
    NA when a value is NA), 0 where a combination is missing; rows with an NA
    key are left out but every metric keeps its column."""
    metrics = sorted(sub["metric"].unique())
    rows = sub[sub[key].notna()]

    def r_sum(v: pd.Series) -> float:
        v = v.to_numpy(dtype=float)
        if len(v) == 1 or np.isnan(v).any():
            return float(v[0]) if len(v) == 1 else np.nan
        return _core.lsm_r_statistics(list(v))[0]

    cells = rows.groupby([key, "metric"])["value"].agg(r_sum)
    wide = cells.unstack("metric").reindex(columns=metrics)
    present = rows.groupby([key, "metric"]).size().unstack("metric").reindex(columns=metrics)
    wide = wide.where(present.notna(), 0.0).sort_index()
    return wide


def calculate_correlation(metrics: pd.DataFrame, method: str = "pearson", diag: bool = True, simplify: bool = False):
    """Correlation between metrics, like landscapemetrics' ``calculate_correlation``.

    Parameters
    ----------
    metrics
        A long table of metrics (``Landscape.metrics``, ``landscape_metrics``,
        ``sample_metrics``...). Patch-level metrics are correlated over patch
        ids, class-level metrics over classes, landscape-level metrics over
        layers (``layer`` column; needs several rasters). As in R's ``xtabs``,
        values of the same id (class, layer) are summed, missing combinations
        count as 0, NA values give NA correlations for their metric (whose
        diagonal stays 1) and rows without id / class are left out.
    method
        "pearson", "spearman" or "kendall" (R's ``cor``, same values).
    diag
        Keep the diagonal (correlation 1).
    simplify
        Return the table itself when a single level is present.

    Returns
    -------
    dict or pandas.DataFrame
        ``{level: DataFrame(metric_1, metric_2, value)}`` with the lower
        triangle of the correlation matrix (column by column, as R).

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> table = ls.landscape_metrics(landcover, what=["lsm_p_area", "lsm_p_perim", "lsm_p_shape"])
    >>> corr = ls.calculate_correlation(table, method="spearman")
    >>> sorted(corr)
    ['patch']
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    df = pd.DataFrame(metrics)
    if "layer" not in df.columns:
        df = df.assign(layer=1)
    levels = [lv for lv in ("patch", "class", "landscape") if (df["level"].astype(str) == lv).any()]
    result = {}
    for level in levels:
        sub = df[df["level"].astype(str) == level]
        sub = sub.assign(metric=sub["metric"].astype(str))
        if sub["metric"].nunique() == 1:
            raise ValueError("Please provide input with more than one metric.")
        key = {"patch": "id", "class": "class", "landscape": "layer"}[level]
        if level == "landscape" and sub["layer"].nunique() == 1:
            raise ValueError("Correlation on landscape level only possible for several landscapes.")
        result[level] = _correlation_frame(_xtabs(sub, key), method, diag)
    if simplify:
        if len(result) == 1:
            return next(iter(result.values()))
        warnings.warn("Simplifying only possible if one level is present.", stacklevel=2)
    return result


def _as_list(value) -> list | None:
    if value is None or isinstance(value, str):
        return value
    return list(value) if isinstance(value, Iterable) else value
