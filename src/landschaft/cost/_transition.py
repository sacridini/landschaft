# SPDX-License-Identifier: GPL-3.0-or-later
"""Transition matrices on a raster grid (gdistance's ``TransitionLayer``)."""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import _core

__all__ = ["Transition", "transition", "geo_correction", "normalize", "sum_reciprocal"]

BUILTINS = ("mean", "min", "max", "sum", "prod", "from", "to", "diff", "harmonic", "geometric")
REDUCE_METHODS = ("NZcolMeans", "NZrowMeans", "colSums", "rowSums", "colMeans", "rowMeans")


@dataclass(frozen=True)
class GridInfo:
    """The raster a transition is defined on."""

    shape: tuple[int, int]
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    crs: object = None
    lonlat: bool = False  # CRS in longitude/latitude (raster's isLonLat)
    wrap: bool = False  # global lon/lat: the columns wrap around

    @property
    def core(self) -> tuple:
        ny, nx = self.shape
        return (ny, nx, *self.extent, self.wrap)

    @property
    def res(self) -> tuple[float, float]:
        ny, nx = self.shape
        xmin, xmax, ymin, ymax = self.extent
        return (xmax - xmin) / nx, (ymax - ymin) / ny

    def same(self, other: "GridInfo") -> bool:
        return self.shape == other.shape and self.extent == other.extent and self.lonlat == other.lonlat


def _crs_lonlat(crs) -> bool:
    if crs is None:
        return False
    if isinstance(crs, str) and "longlat" in crs:  # raster's isLonLat
        return True
    try:
        from rasterio.crs import CRS

        c = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
        return bool(c.is_geographic)
    except Exception:
        return False


def make_grid(shape, extent, crs) -> GridInfo:
    ny, nx = (int(v) for v in shape)
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("the extent must have xmax > xmin and ymax > ymin")
    lonlat = _crs_lonlat(crs)
    ext_ll = xmin > -365 and xmax < 365 and ymin > -90.1 and ymax < 90.1
    could_be_ll = lonlat or (crs is None and ext_ll)  # raster's couldBeLonLat
    xres = (xmax - xmin) / nx
    wrap = bool(could_be_ll and abs(xmin + 180) / xres <= 0.1 and abs(xmax - 180) / xres <= 0.1)
    if wrap and nx < 5:
        raise ValueError("a global longitude/latitude grid needs at least 5 columns")
    return GridInfo((ny, nx), (xmin, xmax, ymin, ymax), crs, lonlat, wrap)


def read_grid_values(x, *, extent=None, crs=None, band=None, nodata=None, layers: bool = False):
    """Values (float64, NaN = NA) and grid of a raster: path, DataArray or array.

    Arrays without ``extent`` get unit cells with the origin at 0 and the first
    row on top (as ``terra::rast(matrix)``). With ``layers``, a 3-D input (or
    several bands) is kept as (layers, rows, columns).
    """
    import xarray as xr

    template = None
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        x = _open(x)
    if isinstance(x, xr.DataArray):
        da = x
        if "band" in da.dims and (not layers or da.sizes["band"] == 1):
            da = da.isel(band=band - 1 if band else 0) if da.sizes["band"] > 1 or band else da.squeeze("band", drop=True)
        template = da
        if nodata is None:
            try:
                nodata = da.rio.nodata
            except Exception:
                nodata = None
        values = np.asarray(da.values, dtype=np.float64)
        if extent is None:
            try:
                t = da.rio.transform()
                ny, nx = values.shape[-2:]
                if t.b != 0 or t.d != 0:
                    raise ValueError("rotated rasters are not supported")
                extent = (t.c, t.c + nx * t.a, t.f + ny * t.e, t.f)
            except ValueError:
                raise
            except Exception:
                extent = None
        if crs is None:
            try:
                c = da.rio.crs
                crs = c.to_string() if c is not None else None
                if c is not None and c.is_geographic and "longlat" not in crs:
                    crs = "+proj=longlat " + crs
            except Exception:
                crs = None
    else:
        values = np.asarray(np.ma.filled(np.ma.asarray(x, dtype=np.float64), np.nan), dtype=np.float64)
    values = np.array(values, dtype=np.float64, copy=True)
    if nodata is not None and not np.isnan(nodata):
        values[values == nodata] = np.nan
    if values.ndim == 3 and not layers:
        if values.shape[0] != 1:
            raise ValueError("expected a single-band raster")
        values = values[0]
    if values.ndim not in ((2, 3) if layers else (2,)):
        raise ValueError("expected a 2-D raster")
    ny, nx = values.shape[-2:]
    if extent is None:
        extent = (0.0, float(nx), 0.0, float(ny))
    return values, make_grid((ny, nx), extent, crs), template


def _open(path):
    from ..io import load_raster

    return load_raster(path, masked=False)


def _layout_offsets(directions: int, symmetric: bool) -> list[tuple[int, int]]:
    return [tuple(o) for o in _core.cost_offsets(directions, symmetric)]


class Transition:
    """A transition matrix on a raster grid: gdistance's ``TransitionLayer``.

    The connections between neighbouring cells (4, 8 or 16 directions) with a
    conductance (or resistance). Build one with :func:`transition`; combine
    them with ``*``, ``+``, ``-``, scale them with numbers (``1 / tr`` takes
    the reciprocal of every connection), change the values with :meth:`map`,
    correct them for the distances between cells with :func:`geo_correction`.

    Attributes
    ----------
    values : numpy.ndarray
        ``(n_directions, n_cells)``: the value of the connection from each
        cell (row-major) in each direction of :attr:`offsets`; 0 = none. A
        symmetric transition keeps only the directions after ``(0, 0)`` (the
        pair's value), an asymmetric one all of them (from the cell to its
        neighbour).
    directions : int
        4, 8 or 16.
    symmetric : bool
    matrix_values : str
        ``"conductance"`` or ``"resistance"`` (gdistance's ``matrixValues``).
    """

    def __init__(self, values: np.ndarray, grid: GridInfo, directions: int, symmetric: bool,
                 matrix_values: str = "conductance", template=None):
        n_dirs = len(_layout_offsets(directions, symmetric))
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != (n_dirs, grid.shape[0] * grid.shape[1]):
            raise ValueError(f"values must have shape ({n_dirs}, {grid.shape[0] * grid.shape[1]})")
        if matrix_values not in ("conductance", "resistance"):
            raise ValueError("matrix_values must be 'conductance' or 'resistance'")
        self.values = values
        self.grid = grid
        self.directions = int(directions)
        self.symmetric = bool(symmetric)
        self.matrix_values = matrix_values
        self._template = template

    # -- description -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns of the raster."""
        return self.grid.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.grid.extent

    @property
    def crs(self):
        return self.grid.crs

    @property
    def lonlat(self) -> bool:
        """True when the CRS is longitude/latitude."""
        return self.grid.lonlat

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def offsets(self) -> list[tuple[int, int]]:
        """(row, column) offset of each direction of :attr:`values`."""
        return _layout_offsets(self.directions, self.symmetric)

    @property
    def n_connections(self) -> int:
        """Number of non-zero entries of the matrix (both triangles when symmetric)."""
        n = int(np.count_nonzero(self.values))
        return 2 * n if self.symmetric else n

    def __repr__(self) -> str:
        kind = "symmetric" if self.symmetric else "asymmetric"
        return (f"Transition({self.shape[0]} x {self.shape[1]} cells, {self.directions} directions, {kind}, "
                f"{self.matrix_values}, {self.n_connections} connections)")

    def is_symmetric(self) -> bool:
        """Whether the matrix is symmetric (gdistance's ``isSymmetric``): always for symmetric
        transitions, and for asymmetric ones whose values happen to be."""
        if self.symmetric:
            return True
        m = self.to_sparse()
        return (m != m.T).nnz == 0

    def copy(self) -> "Transition":
        return Transition(self.values.copy(), self.grid, self.directions, self.symmetric, self.matrix_values,
                          self._template)

    def _like(self, values, directions=None, symmetric=None, matrix_values=None) -> "Transition":
        return Transition(values, self.grid, self.directions if directions is None else directions,
                          self.symmetric if symmetric is None else symmetric,
                          self.matrix_values if matrix_values is None else matrix_values, self._template)

    def neighbours(self) -> np.ndarray:
        """The neighbour cell of each (direction, cell) of :attr:`values`, -1 where there is none."""
        return _core.cost_neighbours(self.grid.core, self.directions, self.symmetric)

    # -- conversions -----------------------------------------------------------

    def to_sparse(self):
        """The ``n_cells x n_cells`` matrix (scipy CSC, both triangles when symmetric)."""
        from scipy import sparse

        p, i, x = _core.cost_to_csc(self.values, self.grid.core, self.directions, self.symmetric)
        n = self.n_cells
        return sparse.csc_matrix((x, i, p), shape=(n, n))

    def adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        """Cells ``(from, to)`` of every connection (0-based), column by column.

        gdistance's ``adjacencyFromTransition`` (which is 1-based).
        """
        p, i, _ = _core.cost_to_csc(self.values, self.grid.core, self.directions, self.symmetric)
        j = np.repeat(np.arange(self.n_cells, dtype=np.int64), np.diff(p))
        return i, j

    def to_raster(self, reduce: str = "NZcolMeans"):
        """A raster summarising the connections of each cell (gdistance's ``raster()``).

        Parameters
        ----------
        reduce : {"NZcolMeans", "NZrowMeans", "colSums", "rowSums", "colMeans", "rowMeans"}
            Sums or means of the matrix's columns (connections into the cell) or
            rows (out of it); ``NZ``: the mean over the connections only (NaN
            without any).
        """
        if reduce not in REDUCE_METHODS:
            raise ValueError(f"reduce must be one of {', '.join(REDUCE_METHODS)}")
        margin = 1 if "col" in reduce else 0
        sums = self._margin(margin)
        if reduce.startswith("NZ"):
            count = _core.cost_margin_sums(self.values, self.grid.core, self.directions, self.symmetric, margin, True)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = sums / count
        elif reduce.endswith("Means"):
            out = sums / self.n_cells
        else:
            out = sums
        return self._wrap(out.reshape(self.shape), reduce)

    def _margin(self, margin: int) -> np.ndarray:
        return _core.cost_margin_sums(self.values, self.grid.core, self.directions, self.symmetric, margin, False)

    def _wrap(self, values: np.ndarray, name: str):
        if self._template is None:
            return values
        import xarray as xr

        t = self._template
        out = xr.DataArray(values, coords={k: t.coords[k] for k in t.dims}, dims=t.dims, name=name,
                           attrs={"long_name": name})
        crs = t.rio.crs if hasattr(t, "rio") else None
        if crs is not None:  # the georeferencing of the raster the transition came from
            from ..io import _with_georef

            out = _with_georef(out, crs=crs, transform=t.rio.transform())
        return out

    # -- layouts -----------------------------------------------------------------

    def _relayout(self, directions: int, symmetric: bool) -> np.ndarray:
        """The values in another layout (more directions; asymmetric from symmetric)."""
        if directions == self.directions and symmetric == self.symmetric:
            return self.values
        if symmetric and not self.symmetric:
            raise ValueError("an asymmetric transition cannot be made symmetric")
        if directions < self.directions:
            raise ValueError("cannot drop directions")
        target = _layout_offsets(directions, symmetric)
        index = {o: k for k, o in enumerate(target)}
        out = np.zeros((len(target), self.n_cells))
        src = self.offsets
        if symmetric or not self.symmetric:  # same symmetry, more directions
            for k, o in enumerate(src):
                out[index[o]] = self.values[k]
            return out
        nb = self.neighbours()
        for k, (dr, dc) in enumerate(src):
            v = self.values[k]
            m = (nb[k] >= 0) & (v != 0)
            out[index[(dr, dc)], m] = v[m]
            out[index[(-dr, -dc)], nb[k][m]] = v[m]
        return out

    def _common(self, other: "Transition"):
        if not self.grid.same(other.grid):
            raise ValueError("transitions on different grids")
        directions = max(self.directions, other.directions)
        symmetric = self.symmetric and other.symmetric
        return self._relayout(directions, symmetric), other._relayout(directions, symmetric), directions, symmetric

    # -- arithmetic --------------------------------------------------------------

    def _binary(self, other, op, name):
        if isinstance(other, Transition):
            a, b, d, s = self._common(other)
            return self._like(op(a, b), d, s)
        if isinstance(other, numbers.Number):
            if name in ("+", "-"):
                raise TypeError(f"transition {name} number would connect every cell: use map() instead")
            return self._like(op(self.values, float(other)))
        return NotImplemented

    def __mul__(self, other):
        return self._binary(other, np.multiply, "*")

    __rmul__ = __mul__

    def __add__(self, other):
        return self._binary(other, np.add, "+")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, np.subtract, "-")

    def __truediv__(self, other):
        if isinstance(other, Transition):
            raise TypeError("dividing transitions is not supported: use map() on one of them")
        return self._binary(other, np.divide, "/")

    def __rtruediv__(self, other):
        """``number / tr``: on the connections only (the reciprocal for 1)."""
        if not isinstance(other, numbers.Number):
            return NotImplemented
        out = np.zeros_like(self.values)
        m = self.values != 0
        out[m] = float(other) / self.values[m]
        return self._like(out)

    def __neg__(self):
        return self._like(-self.values)

    def __pow__(self, p):
        """``tr ** p`` on the connections, with R's ``^`` (mingw's powl)."""
        if not isinstance(p, numbers.Number):
            return NotImplemented
        return self.map(lambda v: _core.cost_rmath("pow", v, float(p)))

    def map(self, function: Callable | str, pairs: str = "edges") -> "Transition":
        """A transition with ``function`` applied to the values.

        Parameters
        ----------
        function
            A vectorised function of an array of values, or ``"sqrt"``,
            ``"exp"``, ``"log"`` for R's (bit-identical) versions.
        pairs : {"edges", "all"}
            ``"edges"``: the existing connections only; ``"all"``: every pair
            of neighbouring cells of the grid, also those without a connection
            (value 0) or with missing values, as gdistance's
            ``tr[adjacent(r, 1:ncell(r))] <- f(tr[...])``.
        """
        if isinstance(function, str):
            name = function
            if name not in ("sqrt", "exp", "log"):
                raise ValueError("function names: 'sqrt', 'exp', 'log'")
            function = lambda v: _core.cost_rmath(name, v)  # noqa: E731
        if pairs == "edges":
            m = self.values != 0
        elif pairs == "all":
            m = self.neighbours() >= 0
        else:
            raise ValueError("pairs must be 'edges' or 'all'")
        out = np.zeros_like(self.values)
        out[m] = np.asarray(function(self.values[m]), dtype=np.float64)
        return self._like(_clean(out))

    # -- methods delegating to the module functions ------------------------------

    def geo_correction(self, type: str = "c", *, factors: bool = False, scale: bool = False) -> "Transition":
        """See :func:`geo_correction`."""
        return geo_correction(self, type, factors=factors, scale=scale)

    def normalize(self, method: str = "row") -> "Transition":
        """See :func:`normalize`."""
        return normalize(self, method)


def _clean(values: np.ndarray) -> np.ndarray:
    nan = np.isnan(values)
    if nan.any():
        warnings.warn("the transition function gives NaN: those connections are left out", RuntimeWarning,
                      stacklevel=3)
        values[nan] = 0.0
    if (values < 0).any():
        warnings.warn("transition function gives negative values", RuntimeWarning, stacklevel=3)
    return values


def transition(x, function: str | Callable = "mean", directions: int = 8, symmetric: bool = True, *,
               extent=None, crs=None, band=None, nodata=None, breaks=None):
    """Transition matrix from a raster: gdistance's ``transition()``.

    Connects every pair of neighbouring cells that both have a value with
    ``function`` of their two values.

    Parameters
    ----------
    x
        Raster: path, ``xarray.DataArray`` or 2-D array (NaN = missing); for
        ``"mahal"`` several layers (3-D or a multi-band raster).
    function
        A built-in function, computed in C++ with R's arithmetic (so the
        values are gdistance's to the last bit): ``"mean"``, ``"min"``,
        ``"max"``, ``"sum"``, ``"prod"``, ``"harmonic"`` (``1/mean(1/x)``),
        ``"geometric"`` (``exp(mean(log(x)))``), ``"from"`` (``x[1]``),
        ``"to"`` (``x[2]``), ``"diff"`` (``x[2] - x[1]``); ``"mahal"``: the
        normalised reciprocal of the Mahalanobis distance between the cells'
        layers; ``"barriers"`` / ``"areas"``: lists of transitions, see
        :func:`barriers` and :func:`areas`; or a vectorised callable that receives ``x`` (``2 x n``:
        ``x[0]`` the first cell, ``x[1]`` the second) as gdistance's function
        receives ``c(x1, x2)``, e.g. ``lambda x: x[1] - x[0]``. Values of 0
        leave the cells unconnected. For a conductance from a resistance use
        ``1 / transition(r, "mean")``.
    directions : {4, 8, 16}
        Rook, queen, or queen and knight moves.
    symmetric
        True: the pair's value, ``f(x[lower cell index], x[higher])``; False:
        a value per direction, ``f(x[from], x[to])``.
    extent, crs
        ``(xmin, xmax, ymin, ymax)`` and CRS of an array (DataArrays and files
        carry their own). On a longitude/latitude grid spanning 360 degrees the
        columns wrap around, as in gdistance.
    band, nodata
        Band to read (1-based) and value to treat as missing.
    breaks
        ``"barriers"`` only: the breaks of continuous values.

    Returns
    -------
    Transition, or a list of them for ``"barriers"`` and ``"areas"``

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.array([[1.0, 2.0, 4.0], [1.0, np.nan, 2.0]])
    >>> tr = 1 / ls.cost.transition(r, "mean", 4)       # conductance = 1 / mean resistance
    >>> tr
    Transition(2 x 3 cells, 4 directions, symmetric, conductance, 8 connections)
    >>> tr.to_sparse()[0, 1]                             # cells 0 and 1: 1 / mean(1, 2)
    np.float64(0.6666666666666666)
    >>> up = ls.cost.transition(r, lambda x: x[1] - x[0], 4, symmetric=False)
    >>> float(up.to_sparse()[1, 2]), float(up.to_sparse()[2, 1])
    (2.0, -2.0)
    """
    if isinstance(function, str) and function in ("barriers", "areas"):
        from ._pathinc import areas, barriers

        kw = dict(extent=extent, crs=crs, band=band, nodata=nodata)
        if function == "barriers":
            return barriers(x, directions, breaks=breaks, symmetric=symmetric, **kw)
        return areas(x, directions, **kw)
    if isinstance(function, str) and function == "mahal":
        from ._mahal import mahal_transition

        values, grid, template = read_grid_values(x, extent=extent, crs=crs, band=band, nodata=nodata, layers=True)
        return mahal_transition(values, grid, directions, template)
    values, grid, template = read_grid_values(x, extent=extent, crs=crs, band=band, nodata=nodata)
    if isinstance(function, str):
        if function not in BUILTINS:
            raise ValueError(f"unknown function {function!r}; built-in: {', '.join(BUILTINS)}, 'mahal'")
        v = _core.cost_builtin(np.ascontiguousarray(values), grid.core, directions, symmetric, function)
        tr = Transition(v, grid, directions, symmetric, "conductance", template)
        if function != "diff" and (v < 0).any():  # differences are signed on purpose
            warnings.warn("transition function gives negative values", RuntimeWarning, stacklevel=2)
        return tr
    if not callable(function):
        raise TypeError("function must be a name or a callable")
    nb = _core.cost_neighbours(grid.core, directions, symmetric)
    flat = values.ravel()
    cell = np.broadcast_to(np.arange(flat.size), nb.shape)
    ok = nb >= 0
    ok[ok] = np.isfinite(flat[cell[ok]]) & np.isfinite(flat[nb[ok]])
    a, b = cell[ok], nb[ok]
    if symmetric:
        a, b = np.minimum(a, b), np.maximum(a, b)
    out = np.zeros(nb.shape)
    res = np.asarray(function(np.stack([flat[a], flat[b]])), dtype=np.float64)
    out[ok] = np.broadcast_to(res, a.shape)
    return Transition(_clean(out), grid, directions, symmetric, "conductance", template)


def geo_correction(x: Transition, type: str = "c", *, factors: bool = False, scale: bool = False) -> Transition:
    """Correct a transition for the distances between cell centres: gdistance's ``geoCorrection()``.

    Conductances are divided by the distance between the centres of the two
    cells (resistances multiplied): needed with more than 4 directions and on
    longitude/latitude grids, where distances are great-circle ones on the
    WGS84 ellipsoid (the R package raster's, bit for bit).

    Parameters
    ----------
    x
        A :class:`Transition`.
    type : {"c", "r"}
        On longitude/latitude grids: ``"c"`` for least-cost distances;
        ``"r"`` for random walks, which also scales north-south connections
        (conductances) by the cosine of their mean latitude.
    factors
        Return the correction factors instead of the corrected values
        (gdistance's ``multpl = TRUE``).
    scale
        Divide the distances by the west-east distance of one cell at the
        centre of the grid, to keep values near 1 (``scl = TRUE``).

    Returns
    -------
    Transition

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((3, 3)), "mean", 8, extent=(0, 30, 0, 30))
    >>> g = ls.cost.geo_correction(tr)
    >>> m = g.to_sparse()
    >>> float(m[0, 1]), round(float(m[0, 4]), 6)       # 1/10 m across, 1/(10 sqrt 2) diagonally
    (0.1, 0.070711)
    """
    if type not in ("c", "r"):
        raise ValueError("type can only be c or r")
    v = _core.cost_geo_correction(x.values, x.grid.core, x.directions, x.symmetric, x.lonlat, type,
                                  x.matrix_values == "resistance", bool(scale), bool(factors))
    return x._like(v)


def normalize(x: Transition, method: str = "row") -> Transition:
    """Normalise a transition matrix: gdistance's ``normalize()``.

    Parameters
    ----------
    x
        A :class:`Transition`.
    method : {"row", "col", "symm"}
        ``"row"``: divide each row by its sum (transition probabilities of a
        random walk, asymmetric); ``"col"``: each column; ``"symm"``: divide
        by the square roots of the row and column sums (stays symmetric).

    Returns
    -------
    Transition

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> p = ls.cost.normalize(ls.cost.transition(np.ones((3, 3)), "mean", 4), "row")
    >>> p.to_sparse()[4].toarray().round(2)            # the centre cell: 1/4 to each neighbour
    array([[0.  , 0.25, 0.  , 0.25, 0.  , 0.25, 0.  , 0.25, 0.  ]])
    """
    if method not in ("row", "col", "symm"):
        raise ValueError("invalid method argument")
    if method in ("row", "col"):
        sums = x._margin(0 if method == "row" else 1)
        with np.errstate(divide="ignore"):
            inv = 1.0 / sums
        inv[np.isinf(inv)] = 0.0
        v = x._relayout(x.directions, False)
        if method == "row":
            out = v * inv[None, :]
        else:
            nb = _core.cost_neighbours(x.grid.core, x.directions, False)
            out = np.where(nb >= 0, v * inv[np.clip(nb, 0, None)], 0.0)
        return x._like(out, symmetric=False)
    rs = _core.cost_rmath("pow", x._margin(0), -0.5)
    cs = _core.cost_rmath("pow", x._margin(1), -0.5)
    nb = x.neighbours()
    cell = np.broadcast_to(np.arange(x.n_cells), nb.shape)
    ok = (nb >= 0) & (x.values != 0)
    lo, hi = cell, np.clip(nb, 0, None)
    if x.symmetric:  # forceSymmetric: the upper triangle, (x * rs[lower]) * cs[higher]
        lo, hi = np.minimum(cell, hi), np.maximum(cell, hi)
    out = np.zeros_like(x.values)
    out[ok] = (x.values[ok] * rs[lo[ok]]) * cs[hi[ok]]
    return x._like(out)


def sum_reciprocal(x1: Transition, x2: Transition) -> Transition:
    """Combine two transitions as resistances in series: gdistance's ``sumReciprocal()``.

    Conductances are turned into resistances (``1 / x``), added, and the sum
    turned back into a conductance.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> a = ls.cost.transition(np.full((2, 2), 2.0), "mean", 4)
    >>> b = ls.cost.transition(np.full((2, 2), 3.0), "mean", 4)
    >>> round(float(ls.cost.sum_reciprocal(a, b).to_sparse()[0, 1]), 12)   # 1 / (1/2 + 1/3)
    1.2
    """
    def resist(t: Transition) -> Transition:
        if t.matrix_values != "conductance":
            return t
        out = np.zeros_like(t.values)
        m = t.values != 0
        out[m] = 1.0 / t.values[m]
        out[np.isinf(out)] = 0.0
        return t._like(out)

    s = resist(x1) + resist(x2)
    out = np.zeros_like(s.values)
    m = s.values != 0
    out[m] = 1.0 / s.values[m]
    if np.isinf(out).any():
        warnings.warn("Inf values introduced. Set to 0.", RuntimeWarning, stacklevel=2)
        out[np.isinf(out)] = 0.0
    return s._like(out, matrix_values="conductance")

