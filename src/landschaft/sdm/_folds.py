# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-validation folds for species distribution models: blockCV and ENMeval partitions."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from .. import _core
from ._thin import _seed

__all__ = ["spatial_blocks", "spatial_clusters", "buffer_folds", "quadrant_folds", "checkerboard_folds", "random_folds",
           "jackknife_folds"]

_EPS = 2.220446049250313e-16


# ------------------------------------------------------------------------------------- R arithmetic

def _round_bits(fr: Fraction, bits: int = 64) -> Fraction:
    """``fr`` rounded to ``bits`` significant bits, half to even (the x87's extended precision)."""
    if fr == 0:
        return Fraction(0)
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    scale = Fraction(2) ** (bits - 1 - e)
    m = a * scale
    q, r = divmod(m.numerator, m.denominator)
    twice = 2 * r
    if twice > m.denominator or (twice == m.denominator and q % 2 == 1):
        q += 1
    out = Fraction(q) / scale
    return out if fr > 0 else -out


def _r_fmod(x1: float, x2: float) -> float:
    """R's x1 %% x2 (arithmetic.c's myfmod, with its 80-bit intermediate steps)."""
    if abs(x2) * _EPS > 1 and math.isfinite(x1) and abs(x1) <= abs(x2):
        if abs(x1) == abs(x2):
            return 0.0
        return x1 + x2 if (x1 < 0 < x2) or (x2 < 0 < x1) else x1
    q = x1 / x2
    tmp = _round_bits(Fraction(x1) - _round_bits(Fraction(math.floor(q)) * Fraction(x2)))
    fl = math.floor(_round_bits(tmp / Fraction(x2)))
    return float(_round_bits(tmp - _round_bits(Fraction(fl) * Fraction(x2))))


def _r_seq(start: float, end: float, by: float) -> np.ndarray:
    """R's seq(from, to, by) for doubles."""
    delta = end - start
    if delta == 0 and end == 0:
        return np.array([end])
    n = delta / by
    dd = abs(delta) / max(abs(end), abs(start))
    if dd < 100 * _EPS:
        return np.array([start])
    n = int(n + 1e-10)
    x = start + np.arange(n + 1) * by
    return np.minimum(x, end) if by > 0 else np.maximum(x, end)


def _r_sd(x) -> float:
    return _core.sdm_r_sd(np.ascontiguousarray(np.asarray(x, dtype=float)))


# ----------------------------------------------------------------------------------------- inputs

def _xy_and_class(data, column):
    """Coordinates (as they are), CRS and the class of each record used to balance the folds."""
    import geopandas as gpd

    from ._data import SDMData

    if isinstance(data, SDMData):
        g = data.coords
        if column == "y":
            cls = data.y
        else:
            cls = None if column is None else g[column].to_numpy()
    elif isinstance(data, gpd.GeoDataFrame):
        g = data
        if column is not None and column not in g.columns:
            if column != "y":
                raise ValueError(f"no column {column!r}")
            column = None
        cls = None if column is None else g[column].to_numpy()
    else:
        raise TypeError("data must be SDMData or a GeoDataFrame of points")
    return g.geometry.x.to_numpy(float), g.geometry.y.to_numpy(float), g.crs, cls


def _result(data, folds: np.ndarray, info: dict):
    from ._data import SDMData

    folds = np.asarray(folds, dtype=np.int64)
    if isinstance(data, SDMData):
        out = data.with_folds(folds)
        out.settings = {**out.settings, "folds": info}
        return out
    return folds


def _is_lonlat(crs, x, y) -> bool:
    if crs is not None:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_geographic)
    return bool(np.all((x >= -180) & (x <= 180) & (y >= -180) & (y <= 180)))


# ------------------------------------------------------------------------------------- spatial blocks

def _square_blocks(x, y, size, rows_cols, offset, extend, lonlat, deg_to_metre, checkerboard):
    """blockCV's .make_blocks for squares: (cell numbers hit by each point, corners, ids)."""
    xmin0, xmax0, ymin0, ymax0 = x.min(), x.max(), y.min(), y.max()
    ext = [xmin0, xmax0, ymin0, ymax0]
    nrows, ncols = rows_cols
    if size is not None:
        if lonlat:
            size = size / deg_to_metre
        # blockCV shifts by offset * size before converting size to degrees (an error on longitude /
        # latitude data: metres added to degrees); here after, in the units of the coordinates
        off = [size * (abs(o) % 1) for o in (list(offset) + [0.0])[:2]]
    xrange, yrange = xmax0 - xmin0, ymax0 - ymin0
    ext[0] = ext[0] - (xrange * extend)
    ext[1] = ext[1] + (xrange * extend)
    ext[2] = ext[2] - (yrange * extend)
    ext[3] = ext[3] + (yrange * extend)
    if size is not None:
        xpix, ypix = math.ceil(xrange / size), math.ceil(yrange / size)
        xdif = ((xpix * size) - xrange) / 2
        ydif = ((ypix * size) - yrange) / 2
        ext[0] = ext[0] - xdif + off[0]
        ext[1] = ext[1] + xdif + off[0]
        ext[2] = ext[2] - ydif + off[1]
        ext[3] = ext[3] + ydif + off[1]
        if ext[0] > xmin0:
            ext[0] = ext[0] - size
            xpix += 1
        if ext[2] > ymin0:
            ext[2] = ext[2] - size
            ypix += 1
        ncols, nrows = xpix, ypix
    nrows, ncols = max(1, int(nrows)), max(1, int(ncols))
    gxmin, gxmax, gymin, gymax = ext
    xres = (gxmax - gxmin) / ncols
    yres = (gymax - gymin) / nrows
    cx = gxmin + (np.arange(ncols) + 0.5) * xres
    cy = gymax - (np.arange(nrows) + 0.5) * yres
    hx, hy = xres / 2, yres / 2
    left, right = cx - hx, cx + hx
    bottom, top = cy - hy, cy + hy
    # each point against the cells around the one it falls in (closed squares: edges belong to both)
    col0 = np.floor((x - gxmin) / xres).astype(np.int64)
    row0 = np.floor((gymax - y) / yres).astype(np.int64)
    pairs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row0 + dr, col0 + dc
            ok = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
            idx = np.flatnonzero(ok)
            rr, cc = r[idx], c[idx]
            hit = (x[idx] >= left[cc]) & (x[idx] <= right[cc]) & (y[idx] >= bottom[rr]) & (y[idx] <= top[rr])
            pairs.append(np.column_stack([idx[hit], rr[hit] * ncols + cc[hit]]))
    pairs = np.concatenate(pairs) if pairs else np.empty((0, 2), np.int64)
    ids = None
    if checkerboard:  # row i (1-based) even: 1, 2, 1, ...; odd: 2, 1, 2, ...
        rr, cc = np.divmod(np.arange(nrows * ncols), ncols)
        ids = np.where(((rr + 1) % 2 == 0), np.where(cc % 2 == 0, 1, 2), np.where(cc % 2 == 0, 2, 1))
    return pairs, ids, {"blocks": "squares", "nrows": nrows, "ncols": ncols, "extent": [gxmin, gxmax, gymin, gymax]}


def _hex_blocks(x, y, size, rows_cols, offset, lonlat, deg_to_metre, flat_top):
    """sf::st_make_grid(square = FALSE) as blockCV calls it: (pairs point-hexagon, polygons)."""
    import shapely

    bxmin, bxmax, bymin, bymax = x.min(), x.max(), y.min(), y.max()
    if size is not None:
        off = [1 - o if o not in (0, 1) else o for o in (list(offset) + [0.0])[:2]]
        if lonlat:
            size = size / deg_to_metre
        xy_off = [size * (abs(o) % 1) for o in off]
        hexsize = size
    else:
        xy_off = [0.0, 0.0]
        hexsize = (bxmax - bxmin) / rows_cols[0]
    pt = [bxmin - xy_off[0], bymin - xy_off[1]]
    dx = hexsize / math.sqrt(3)
    dy = math.sqrt(3) * dx / 2
    if not flat_top:  # pointy-topped: x and y swapped
        xlim, ylim, pt = (bymin, bymax), (bxmin, bxmax), pt[::-1]
    else:
        xlim, ylim = (bxmin, bxmax), (bymin, bymax)
    ox = _r_fmod(pt[0] - xlim[0], dx)
    oy = _r_fmod(pt[1] - ylim[0], 2 * dy)
    x0 = _r_seq(xlim[0] - dx, xlim[1] + 2 * dx, dx) + ox
    y0 = _r_seq(ylim[0] - 2 * dy, ylim[1] + 2 * dy, dy) + oy
    nx, ny = x0.size, y0.size
    gy = np.repeat(y0, nx)
    gx = np.resize(np.concatenate([x0, x0 + dx / 2]), gy.size)
    xi = np.resize(np.concatenate([np.arange(1, 2 * nx, 2), np.arange(2, 2 * nx + 1, 2)]), gy.size)
    yi = np.repeat(np.arange(1, ny + 1), nx)
    sel = np.isin(xi, np.arange(3, xi.max() - 2 + 1, 3)) & (yi > 1) & (yi < yi.max())
    centres = np.column_stack([xi[sel], yi[sel]])
    pattern = np.array([[-2, 0], [-1, -1], [1, -1], [2, 0], [1, 1], [-1, 1], [-2, 0]])
    polys = []
    for c in centres:
        m = c + pattern
        i = ((m[:, 0] - 1) // 2 + 1) + (m[:, 1] - 1) * nx - 1
        ring = np.column_stack([gx[i], gy[i]])
        polys.append(shapely.Polygon(ring if flat_top else ring[:, ::-1]))
    tree = shapely.STRtree(polys)
    pi, bi = tree.query(shapely.points(x, y), predicate="intersects")
    pairs = np.column_stack([pi, bi]).astype(np.int64)
    return pairs, polys, {"blocks": "hexagons", "size": hexsize}


def _fold_assign_hex(polys, k: int) -> np.ndarray:
    """blockCV's .fold_assign: hexagons numbered 1..k along rows of centroids (by their y, as text)."""
    import shapely

    cy = shapely.get_coordinates(shapely.centroid(polys))[:, 1]
    keys = [float(f"{v:.15g}") for v in cy]  # R's as.character(): 15 significant digits
    order = sorted(range(len(polys)), key=lambda i: keys[i])
    z = np.empty(len(polys), dtype=np.int64)
    z[order] = (np.arange(len(polys)) % k) + 1
    return z


def spatial_blocks(data, *, size: float | None = None, rows_cols=(10, 10), k: int = 5, hexagon: bool = True,
                   flat_top: bool = False, selection: str = "random", iteration: int = 100, column="y",
                   offset=(0.0, 0.0), extend: float = 0.0, deg_to_metre: float = 111_325.0,
                   seed: int | None = None):
    """Folds made of spatial blocks: no fold is a neighbour of the records it tests.

    Random cross-validation puts test records next to training records,
    where spatial autocorrelation makes them nearly copies: the scores
    overestimate how well the model predicts elsewhere (Roberts et al. 2017,
    *Ecography* 40: 913-929). Blocks larger than the range of
    autocorrelation are assigned to folds whole. The blocks and folds of
    ``blockCV::cv_spatial`` (Valavi et al. 2019), the same for the same seed.

    Parameters
    ----------
    data
        :class:`SDMData` (presences and background) or a GeoDataFrame of
        points.
    size
        Block width in metres (on longitude / latitude data, divided by
        ``deg_to_metre`` to degrees). Without it, ``rows_cols`` blocks over
        the points' extent.
    rows_cols
        (rows, columns) of blocks when ``size`` is not given (hexagons: the
        rows set the size).
    k
        Number of folds.
    hexagon
        Hexagons (blockCV's default) or squares.
    flat_top
        Hexagons with a flat top instead of a pointy one.
    selection : {"random", "systematic", "checkerboard"}
        How blocks go to folds: at random (the best of ``iteration`` tries:
        the most even folds), in turn, or as a checkerboard (squares, k = 2).
    iteration
        Random tries.
    column
        Classes kept balanced across folds in the random selection: ``"y"``
        (presences and background, the default for SDMData), a column of a
        GeoDataFrame, or None.
    offset
        Shift of the blocks, as fractions of ``size`` (x, y).
    extend
        Widen the extent of the blocks by this percentage (up to 5).
    deg_to_metre
        Metres per degree, to turn ``size`` into degrees.
    seed
        Seed of R's generator (default random).

    Returns
    -------
    SDMData or numpy.ndarray
        The data with ``folds`` (1..k) or the fold of each point. Points on
        no block get fold 0 (training only).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 1000, occ=occ, seed=1), env)
    >>> data = ls.sdm.spatial_blocks(data, size=500_000, k=5, seed=1)
    >>> [len(test) for _, test in data.cv]
    [235, 229, 214, 213, 225]
    """
    if selection not in ("random", "systematic", "checkerboard"):
        raise ValueError("selection must be 'random', 'systematic' or 'checkerboard'")
    x, y, crs, cls = _xy_and_class(data, column)
    n = x.size
    lonlat = _is_lonlat(crs, x, y)
    if hexagon and selection == "checkerboard":
        raise ValueError("checkerboard selection needs square blocks (hexagon=False)")
    if selection == "checkerboard":
        k = 2
    extend = min(5.0, abs(extend)) / 100
    iteration = max(1, abs(int(iteration)))
    ids = None
    if hexagon:
        pairs, polys, info = _hex_blocks(x, y, size, rows_cols, offset, lonlat, deg_to_metre, flat_top)
        nblocks_all = len(polys)
    else:
        pairs, ids, info = _square_blocks(x, y, size, rows_cols, offset, extend, lonlat, deg_to_metre,
                                          selection == "checkerboard")
        nblocks_all = info["nrows"] * info["ncols"]
    # the blocks holding points, in their order; each point's (record, block) pairs by record then block
    used = np.unique(pairs[:, 1])
    rank = np.full(nblocks_all, -1, dtype=np.int64)
    rank[used] = np.arange(used.size)
    pairs = np.column_stack([pairs[:, 0], rank[pairs[:, 1]]])
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    nb = used.size
    k = abs(int(k))
    if k > nb:
        raise ValueError(f"k is larger than the number of blocks holding points ({nb})")
    if k < 2:
        raise ValueError("k must be 2 or more")
    s = _seed(seed)
    if len(pairs) > n:  # points on edges between blocks: one of their blocks at random
        rng = _core.RRandom(s)
        perm = rng.sample_int(len(pairs), len(pairs), False, None) - 1
        p2 = pairs[perm]
        _, first = np.unique(p2[:, 0], return_index=True)
        pairs = p2[np.sort(first)]
    rec, blk = pairs[:, 0], pairs[:, 1]
    rng = _core.RRandom(s)  # cv_spatial seeds again before the folds
    classes = None if cls is None else np.sort(pd.unique(cls))  # blockCV: classes of all records

    def table(fold_of_block):
        f = fold_of_block[blk]
        if classes is None:
            cols = [np.array([(f != p).sum() for p in range(1, k + 1)]),
                    np.array([(f == p).sum() for p in range(1, k + 1)])]
        else:
            c = cls[rec]
            cols = [np.array([((f != p) & (c == cl)).sum() for p in range(1, k + 1)]) for cl in classes]
            cols += [np.array([((f == p) & (c == cl)).sum() for p in range(1, k + 1)]) for cl in classes]
        return np.concatenate(cols).astype(float)

    best = None
    if selection == "random":
        min_num, max_sd = 0.0, np.inf
        for _ in range(iteration):
            fb = np.zeros(nb, dtype=np.int64)
            num = nb // k
            reps = np.tile(np.arange(1, k + 1), num)
            fb[: num * k] = reps[rng.sample_int(num * k, num * k, False, None) - 1]
            rest = nb % k
            if rest:
                fb[num * k:] = rng.sample_int(k, rest, False, None)
            t = table(fb)
            sd = _r_sd(t)
            if t.min() >= min_num and sd < max_sd:
                min_num, max_sd, best = t.min(), sd, fb
    elif selection == "systematic":
        best = _fold_assign_hex([polys[u] for u in used], k) if hexagon else (np.arange(nb) % k) + 1
    else:
        best = ids[used]
    folds = np.zeros(n, dtype=np.int64)
    folds[rec] = best[blk]
    info.update({"method": "spatial_blocks", "k": k, "selection": selection, "seed": s, "size": size,
                 "blocks_used": int(nb)})
    return _result(data, folds, info)


# --------------------------------------------------------------------------------------- buffer LOO

def buffer_folds(data, size: float, *, presence_background: bool = True, add_background: bool = False):
    """Leave-one-out folds with a buffer: each presence is a test, nothing within ``size`` trains.

    ``blockCV::cv_buffer``: for each target record, the training set is
    every record farther than ``size`` (metres; Euclidean in projected
    coordinates, great-circle distance on longitude / latitude) and the test
    set is the record itself (and, with ``add_background``, the background
    within the buffer). With ``presence_background`` the targets are the
    presences, else every record.

    Returns
    -------
    list of (train, test) index arrays
        One fold per target; pass it to scikit-learn as ``cv``.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 300, occ=occ, seed=1), env)
    >>> cv = ls.sdm.buffer_folds(data, 200_000)
    >>> len(cv), len(cv[0][1])
    (116, 1)
    """
    from ._data import SDMData

    if not isinstance(data, SDMData):
        raise TypeError("buffer_folds needs SDMData")
    x, y, crs, _ = _xy_and_class(data, None)
    yv = data.y
    targets = np.flatnonzero(yv == 1) if presence_background else np.arange(x.size)
    lonlat = _is_lonlat(crs, x, y)
    out = []
    for i in targets:
        if lonlat:
            from pyproj import Geod

            d = Geod(ellps="WGS84").inv(np.full(x.size, x[i]), np.full(x.size, y[i]), x, y)[2]
        else:
            d = np.sqrt((x - x[i]) * (x - x[i]) + (y - y[i]) * (y - y[i]))
        train = np.flatnonzero(d > size)
        if presence_background and add_background:
            test = np.concatenate([np.flatnonzero((d <= size) & (yv == 0)), [i]])
        else:
            test = np.array([i])
        out.append((train, test.astype(np.int64)))
    return out


# --------------------------------------------------------------------------------- ENMeval partitions

def _occ_bg(data):
    from ._data import SDMData

    if not isinstance(data, SDMData):
        raise TypeError("these partitions need SDMData (presences and background)")
    g = data.coords
    xy = np.column_stack([g.geometry.x.to_numpy(float), g.geometry.y.to_numpy(float)])
    pres = np.flatnonzero(data.y == 1)
    bg = np.flatnonzero(data.y == 0)
    return xy, pres, bg


def _r_mean2(a: float, b: float) -> float:
    return _core.sdm_r_mean(np.array([a, b], dtype=float))


def quadrant_folds(data, orientation: str = "lat_lon"):
    """Four folds from the occurrences split at their median latitude and longitude (ENMeval "block").

    ``ENMeval::get.block``: the occurrences are split into two halves along
    the first axis of ``orientation``, each half into two along the second;
    the background goes to the group whose quadrant it lies in. The same
    groups as ENMeval.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 500, occ=occ, seed=1), env)
    >>> import numpy as np
    >>> np.bincount(ls.sdm.quadrant_folds(data).folds[data.y == 1])[1:].tolist()
    [29, 29, 29, 29]
    """
    if orientation not in ("lat_lon", "lon_lat", "lon_lon", "lat_lat"):
        raise ValueError("orientation must be 'lat_lon', 'lon_lat', 'lon_lon' or 'lat_lat'")
    xy, pres, bgi = _occ_bg(data)
    occs, bg = xy[pres], xy[bgi]
    n = len(occs)
    n1, n2 = math.ceil(n / 2), n // 2
    n3, n4 = math.ceil(n1 / 2), math.ceil(n2 / 2)
    d1 = 1 if orientation in ("lat_lon", "lat_lat") else 0
    d2 = 0 if orientation in ("lat_lon", "lon_lon") else 1
    o = np.argsort(occs[:, d1], kind="stable")
    grp_a = o[:n1]
    grp_b = o[::-1][:n2]  # empty when n2 = 0 (ENMeval took the first record again)
    a_sorted = grp_a[np.argsort(occs[grp_a, d2], kind="stable")]
    g1, g2 = a_sorted[:n3], grp_a[~np.isin(grp_a, a_sorted[:n3])]
    b_sorted = grp_b[np.argsort(occs[grp_b, d2], kind="stable")]
    g3, g4 = b_sorted[:n4], grp_b[~np.isin(grp_b, b_sorted[:n4])]

    def mx(g, d):
        return occs[g, d].max() if len(g) else -np.inf

    def mn(g, d):
        return occs[g, d].min() if len(g) else np.inf

    bx, by = bg[:, 0], bg[:, 1]
    bgg = np.zeros(len(bg), dtype=np.int64)
    if orientation == "lat_lon":
        bvert, tvert = _r_mean2(mx(g1, 0), mn(g2, 0)), _r_mean2(mx(g3, 0), mn(g4, 0))
        horz = _r_mean2(mx(grp_a, 1), mn(grp_b, 1))
        conds = [(by <= horz) & (bx < bvert), (by <= horz) & (bx >= bvert), (by > horz) & (bx <= tvert),
                 (by > horz) & (bx > tvert)]
    elif orientation == "lon_lat":
        bhorz, thorz = _r_mean2(mx(g1, 1), mn(g2, 1)), _r_mean2(mx(g3, 1), mn(g4, 1))
        vert = _r_mean2(mx(grp_a, 0), mn(grp_b, 0))
        conds = [(bx <= vert) & (by < bhorz), (bx <= vert) & (by >= bhorz), (bx > vert) & (by <= thorz),
                 (bx > vert) & (by > thorz)]
    elif orientation == "lon_lon":
        bvert, tvert = _r_mean2(mx(g1, 0), mn(g2, 0)), _r_mean2(mx(g3, 0), mn(g4, 0))
        vert = _r_mean2(mx(grp_a, 0), mn(grp_b, 0))
        conds = [bx <= bvert, (bx > bvert) & (bx <= vert), (bx > vert) & (bx < tvert), bx >= tvert]
    else:
        bhorz, thorz = _r_mean2(mx(g1, 1), mn(g2, 1)), _r_mean2(mx(g3, 1), mn(g4, 1))
        horz = _r_mean2(mx(grp_a, 1), mn(grp_b, 1))
        conds = [by <= bhorz, (by > bhorz) & (by <= horz), (by > horz) & (by < thorz), by >= thorz]
    for g, c in zip((1, 2, 3, 4), conds):  # rbind of the four groups, then by row: the first group wins
        bgg[(bgg == 0) & c] = g
    occg = np.zeros(n, dtype=np.int64)
    for g, idx in zip((1, 2, 3, 4), (g1, g2, g3, g4)):
        occg[idx] = g
    folds = np.zeros(len(data.y), dtype=np.int64)
    folds[pres] = occg
    folds[bgi] = bgg
    return _result(data, folds, {"method": "quadrant_folds", "orientation": orientation})


def checkerboard_folds(data, env, aggregation=2):
    """Folds from a checkerboard of aggregated cells (ENMeval "checkerboard").

    ``ENMeval::get.checkerboard``: the predictor grid aggregated by
    ``aggregation`` cells becomes a checkerboard; records on white squares
    form fold 1, on black ones fold 2. With two factors (hierarchical), the
    squares are aggregated again and the four combinations are folds 1-4.
    Presences and background follow the same grid.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 500, occ=occ, seed=1), env)
    >>> sorted(set(ls.sdm.checkerboard_folds(data, env, aggregation=(4, 2)).folds.tolist()))
    [1, 2, 3, 4]
    """
    from ..io import _transform

    xy, pres, bgi = _occ_bg(data)
    facs = [aggregation] if np.isscalar(aggregation) else list(aggregation)
    if len(facs) not in (1, 2):
        raise ValueError("aggregation: one factor (basic) or two (hierarchical)")
    tr = _transform(env)
    ny, nx = env.shape[-2:]
    xmin, ymax = tr.c, tr.f
    xres = ((xmin + nx * tr.a) - xmin) / nx
    yres = (ymax - (ymax + ny * tr.e)) / ny

    def white(level_fac):
        f = int(level_fac)
        anx, any_ = math.ceil(nx / f), math.ceil(ny / f)  # terra::aggregate keeps xmin / ymax and grows
        axres, ayres = xres * f, yres * f
        axmax, aymin = xmin + anx * axres, ymax - any_ * ayres
        x, y = xy[:, 0], xy[:, 1]
        col = np.where(x == axmax, anx - 1, np.floor((x - xmin) * (anx / (axmax - xmin))))
        row = np.where(y == aymin, any_ - 1, np.floor((ymax - y) * (any_ / (ymax - aymin))))
        inside = (x >= xmin) & (x <= axmax) & (y >= aymin) & (y <= ymax)
        return np.where(inside, ((row + col) % 2 == 0), np.nan), inside

    w1, in1 = white(facs[0])
    if len(facs) == 1:
        grp = np.where(w1 == 1, 1, 2)
        inside = in1
    else:
        w2, in2 = white(facs[0] * facs[1])
        grp = np.select([(w1 == 1) & (w2 == 1), (w1 == 1) & (w2 == 0), (w1 == 0) & (w2 == 1)], [1, 2, 3], 4)
        inside = in1 & in2
    grp = np.where(inside, grp, 0).astype(np.int64)  # outside the grid: training only
    occ_groups = set(grp[pres][grp[pres] > 0].tolist())
    bg_groups = set(grp[bgi][grp[bgi] > 0].tolist())
    if not occ_groups <= bg_groups:
        raise ValueError("some fold has occurrences but no background: use a smaller aggregation")
    if len(occ_groups) < 2:
        raise ValueError("the occurrences fall in a single square: use a smaller aggregation")
    return _result(data, grp, {"method": "checkerboard_folds", "aggregation": facs})


def random_folds(data, k: int = 5, *, seed: int | None = None):
    """Random k-fold of the presences, background always training (ENMeval "randomkfold").

    ``predicts::folds``: groups of (nearly) equal size, assigned in the
    order of ``runif()`` draws, the same as R for the same seed. Not spatial:
    use it as a non-spatial baseline.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 300, occ=occ, seed=1), env)
    >>> import numpy as np
    >>> np.bincount(ls.sdm.random_folds(data, 4, seed=3).folds[data.y == 1]).tolist()
    [0, 29, 29, 29, 29]
    """
    from ._data import SDMData

    pres = np.flatnonzero(data.y == 1) if isinstance(data, SDMData) else np.arange(len(data))
    n = pres.size
    if n / k < 1:
        raise ValueError("more folds than presences")
    s = _seed(seed)
    i = np.array([round(v) for v in [0.0, *((n / k) * np.arange(1, k)), float(n)]], dtype=np.int64)
    group = np.repeat(np.arange(1, k + 1), np.diff(i))
    u = _core.RRandom(s).runif(n, 0.0, 1.0)
    occg = group[np.argsort(u, kind="stable")]
    if not isinstance(data, SDMData):
        return occg
    folds = np.zeros(len(data.y), dtype=np.int64)
    folds[pres] = occg
    return _result(data, folds, {"method": "random_folds", "k": k, "seed": s})


def jackknife_folds(data):
    """Leave-one-out: each presence its own fold, background always training (ENMeval "jackknife").

    For very few records (fewer than about 25, Shcheglovitova & Anderson
    2013).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus")).head(10)
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 100, occ=occ, seed=1), env)
    >>> len(ls.sdm.jackknife_folds(data).cv)
    10
    """
    pres = np.flatnonzero(data.y == 1)
    folds = np.zeros(len(data.y), dtype=np.int64)
    folds[pres] = np.arange(1, pres.size + 1)
    return _result(data, folds, {"method": "jackknife_folds"})


# ------------------------------------------------------------------------------------- clusters

def _terra_scale(env):
    """terra::scale of every band: (v - mean) / sqrt(sum((v - mean)^2) / (n - 1)), sequential double sums."""
    data = np.asarray(env.values, dtype=float).reshape(env.shape[0], -1)
    nodata = env.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data)
    out = np.empty_like(data)
    for k, v in enumerate(data):
        ok = v[~np.isnan(v)]
        mean = np.cumsum(ok)[-1] / ok.size  # in order, as terra's sum_se_rm
        c = v - mean
        cok = c[~np.isnan(c)]
        rms = math.sqrt(np.cumsum(cok * cok)[-1] / (cok.size - 1))
        out[k] = c / rms
    return out


def spatial_clusters(data, k: int = 5, *, env=None, scale: bool = True, raster_cluster: bool = False,
                     num_sample: int = 10_000, algorithm: str = "Hartigan-Wong", seed: int | None = None):
    """Folds from k-means clusters of the records' coordinates (or of their environment).

    ``blockCV::cv_cluster``: k-means (R's ``kmeans``, Hartigan-Wong, 25
    random starts, up to 500 iterations, the same clusters for the same
    seed) groups the records into ``k`` spatial clusters, each a fold. With
    ``env``, the clusters are made in environmental space instead
    (predictors standardised as ``terra::scale``; with ``raster_cluster``,
    also on ``num_sample`` random cells, so that folds cover environmental
    gradients): the test folds then hold conditions the training data lack.

    Parameters
    ----------
    data
        :class:`SDMData` or a GeoDataFrame of points.
    k
        Number of folds.
    env
        Predictors to cluster in environmental space (default: coordinates).
    scale
        Standardise the predictors.
    raster_cluster
        Cluster the records together with a sample of the raster's cells.
    num_sample
        Cells sampled for ``raster_cluster``.
    algorithm : {"Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"}
        R's k-means algorithm.
    seed
        Seed of R's generator (default random).

    Returns
    -------
    SDMData or numpy.ndarray
        The data with ``folds`` (1..k), or the fold of each point.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 500, occ=occ, seed=1), env)
    >>> np.bincount(ls.sdm.spatial_clusters(data, 4, seed=2).folds).tolist()
    [0, 201, 127, 197, 91]
    """
    x, y, crs, _ = _xy_and_class(data, None)
    s = _seed(seed)
    rng = _core.RRandom(s)
    if env is None:
        table = np.column_stack([x, y])
    else:
        from ._points import grid_cells

        if env.ndim == 2:
            env = env.expand_dims(band=[env.name or "layer"])
        values = _terra_scale(env) if scale else np.asarray(env.values, dtype=float).reshape(env.shape[0], -1)
        cells = grid_cells(x, y, env)
        if (cells < 0).any():
            raise ValueError("some records are outside the predictors")
        table = values[:, cells].T
        if np.isnan(table).any():
            raise ValueError("the predictors have no value at some records")
        if raster_cluster:
            from xarray import DataArray

            from ._predictors import _sample_cells

            complete = ~np.isnan(values).any(axis=0)
            n = num_sample
            if values.shape[1] < 5 * n:
                n = min(n, int(complete.sum()))
            scaled = DataArray(values.reshape(env.shape), dims=env.dims, coords=env.coords).rio.write_crs(env.rio.crs)
            scaled = scaled.rio.write_transform(env.rio.transform())
            pick = _sample_cells(scaled, complete, n, rng)
            table = np.vstack([table, values[:, pick].T])
    res = _core.sdm_kmeans(np.ascontiguousarray(table), k, 500, 25, None, algorithm, rng)
    folds = np.asarray(res["cluster"], dtype=np.int64)[: x.size]
    return _result(data, folds, {"method": "spatial_clusters", "k": k, "seed": s,
                                 "space": "coordinates" if env is None else "environment"})
