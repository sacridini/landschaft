# SPDX-License-Identifier: GPL-3.0-or-later
"""Rasters as images for maps: reduced to the screen's resolution first, coloured by a
lookup table, reprojected to Web Mercator, encoded as PNG."""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from . import _palettes

#: Integer rasters with at most this many distinct values are drawn as classes.
MAX_CLASSES = 64
#: Floating-point rasters with at most this many whole values are drawn as classes.
MAX_FLOAT_CLASSES = 12


@dataclass
class Legend:
    """What a layer's colours mean: classes (colour, label) or a colour ramp."""

    title: str
    classes: list[tuple[str, str]] = field(default_factory=list)
    ramp: list[str] = field(default_factory=list)
    vmin: float = 0.0
    vmax: float = 1.0
    log: bool = False
    under: bool = False  # values below vmin, drawn in its colour
    over: bool = False   # values above vmax


@dataclass
class Image:
    """A rendered layer and where it goes: palette indices (0 transparent) and their
    RGBA colours, or RGBA pixels (true colour). The layer's opacity is applied by the map
    (the palette is opaque, each index in a colour of its own, so that the browser reads
    the index of a pixel back from its colour: the value under the cursor)."""

    pixels: np.ndarray          # (rows, cols) uint8 indices, or (rows, cols, 4) uint8 RGBA
    bounds_latlon: tuple        # ((south, west), (north, east)) of a Web Mercator image
    legend: Legend | None
    name: str
    extent: tuple | None = None  # (xmin, ymin, xmax, ymax) in the source CRS
    crs: object = None
    mercator_extent: tuple | None = None  # (xmin, ymin, xmax, ymax) of the Web Mercator image
    palette: np.ndarray | None = None  # (n <= 256, 4) uint8 RGBA of the indices
    opacity: float = 1.0
    lookup: dict | None = None  # the value of each index: {"labels": [...]} or {"vmin", "vmax", "log", ...}
    style: dict | None = None  # the colours chosen (render's style=): the detail of a part drawn alike

    @property
    def rgba(self) -> np.ndarray:
        return self.palette[self.pixels] if self.palette is not None else self.pixels

    def png(self, compress_level: int = 2) -> bytes:
        """PNG: 8-bit with a palette (and its transparency) when the image has one."""
        from PIL import Image as PILImage

        buf = io.BytesIO()
        if self.palette is not None:
            img = PILImage.fromarray(self.pixels, "P")
            pal = np.zeros((256, 4), np.uint8)
            pal[: len(self.palette)] = self.palette
            img.putpalette(pal[:, :3].ravel().tobytes())
            img.save(buf, "PNG", compress_level=compress_level, transparency=pal[:, 3].tobytes())
        else:
            PILImage.fromarray(self.pixels, "RGBA").save(buf, "PNG", compress_level=compress_level)
        return buf.getvalue()

    def data_url(self) -> str:
        """The PNG as a data URL (made once: the bands of a stack make theirs in parallel)."""
        url = self.__dict__.get("_url")
        if url is None:
            url = self.__dict__["_url"] = "data:image/png;base64," + base64.b64encode(self.png()).decode("ascii")
        return url


# --- reading at the screen's resolution -----------------------------------------------------------


def _source_file(da: xr.DataArray):
    src = da.attrs.get("source")
    if isinstance(src, str) and hasattr(da.data, "dask") and os.path.exists(src):
        return src
    return None


def reduce(da: xr.DataArray, max_size: int, categorical: bool):
    """The values of a 2-D raster with at most max_size cells on its long side, and their
    transform: block means for continuous values, the cell of each block for classes. A
    lazy raster read from a file is read reduced by GDAL (its overviews, if any)."""
    from ..io import _transform

    ny, nx = da.sizes["y"], da.sizes["x"]
    f = max(1, math.ceil(max(ny, nx) / max_size))
    t = _transform(da)
    if f == 1:
        return np.asarray(da.values), t
    oy, ox = ny // f, nx // f
    src = _source_file(da)
    if src is not None and da.ndim == 2:
        read = _read_reduced(src, ny, nx, f, categorical, t)
        if read is not None:
            v, (sx, sy) = read
            return v, t * t.scale(sx, sy)
    data = da.data
    if categorical:
        v = data[f // 2::f, f // 2::f][:oy, :ox]
        v = v.compute() if hasattr(v, "dask") else np.asarray(v)
        return v, t * t.scale(f)
    block = data[: oy * f, : ox * f]
    nd = da.rio.nodata
    has = nd is not None and not (isinstance(nd, float) and math.isnan(nd))
    if hasattr(block, "dask"):
        import dask.array as dsa

        block = block.astype(np.float32)
        if has:  # NoData left out of the means
            block = dsa.where(block == np.float32(nd), np.float32(np.nan), block)
        with _quiet():  # blocks of NoData only: NaN
            v = dsa.coarsen(np.nanmean, block, {0: f, 1: f}, trim_excess=True).compute()
    else:  # C++ (OpenMP): NaN and NoData left out of the means
        from .. import _core

        arr = np.ascontiguousarray(np.asarray(data))  # the whole raster: the incomplete blocks are left out
        if arr.dtype == bool:
            arr = arr.view(np.uint8)
        v = _core.io_block_mean(arr, f, has, float(nd) if has else 0.0)
    return v, t * t.scale(f)


def _read_reduced(src: str, ny: int, nx: int, f: int, categorical: bool, transform=None):
    """(values, (x scale, y scale)) of a one-band file (or of a window of it: ``transform`` that of
    the window, ny x nx its cells) read at 1/f of its resolution: through its overviews by GDAL
    when it has some, else in strips of rows over threads, each reduced at once (C++ block means,
    or the cell of each block for classes), so that the whole raster is never in memory."""
    from concurrent.futures import ThreadPoolExecutor

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    from .. import _core
    from .._threads import default_threads

    with rasterio.open(src) as ds:
        if ds.count != 1:
            return None
        r0 = c0 = 0
        if (ds.height, ds.width) != (ny, nx):  # a window of the file: where it starts
            ft = ds.transform
            if transform is None or (transform.a, transform.e, transform.b, transform.d) != (ft.a, ft.e, ft.b, ft.d):
                return None
            c, r = (transform.c - ft.c) / ft.a, (transform.f - ft.f) / ft.e
            c0, r0 = round(c), round(r)
            if abs(c - c0) > 1e-6 or abs(r - r0) > 1e-6 or c0 < 0 or r0 < 0 or c0 + nx > ds.width \
                    or r0 + ny > ds.height:
                return None
        nodata, dtype = ds.nodata, ds.dtypes[0]
        if ds.overviews(1):
            out = ds.read(1, window=Window(c0, r0, nx, ny), out_shape=(max(1, ny // f), max(1, nx // f)),
                          resampling=Resampling.nearest if categorical else Resampling.average, masked=True)
            out = out.filled(out.fill_value) if categorical else out.astype(np.float32).filled(np.nan)
            return out, (nx / out.shape[1], ny / out.shape[0])  # the whole extent
    oy, ox = ny // f, nx // f
    rows_per = max(1, (64 << 20) // max(1, nx * np.dtype(dtype).itemsize * f)) * f  # ~64 MB strips, whole blocks
    starts = list(range(0, oy * f, rows_per))
    has = nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata))
    out = np.empty((oy, ox), dtype if categorical else np.float32)

    def strip(y0):
        y1 = min(y0 + rows_per, oy * f)
        with rasterio.open(src) as d:
            if categorical:  # the centre cell of each block: read only those rows
                rows = np.arange(y0 + f // 2, y1, f)
                v = np.stack([d.read(1, window=Window(c0, r0 + int(r), nx, 1))[0, f // 2::f][:ox] for r in rows])
            else:
                block = d.read(1, window=Window(c0, r0 + y0, nx, y1 - y0))
                v = _core.io_block_mean(np.ascontiguousarray(block), f, has, float(nodata) if has else 0.0)
        out[y0 // f: y1 // f] = v[:, :ox]

    with ThreadPoolExecutor(max(1, min(default_threads(), len(starts)))) as pool:
        list(pool.map(strip, starts))
    return out, (f, f)  # whole blocks of f x f cells


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


# --- colours ---------------------------------------------------------------------------------------


def _valid(v: np.ndarray, nodata) -> np.ndarray:
    ok = np.ones(v.shape, bool)
    if np.issubdtype(v.dtype, np.floating):
        ok &= np.isfinite(v)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        ok &= v != nodata
    return ok


def float_classes(da: xr.DataArray) -> bool:
    """A floating-point raster holding a few whole numbers: a class map read with NaN for its
    NoData (``load_raster(..., masked=True)``), judged on a sample of about a million cells."""
    if not np.issubdtype(da.dtype, np.floating) or da.ndim != 2 or hasattr(da.data, "dask"):
        return False
    step = max(1, math.ceil(math.sqrt(da.size / 1_000_000)))
    sample = np.asarray(da.data[::step, ::step]).ravel()
    sample = sample[_valid(sample, da.rio.nodata)]
    if sample.size == 0:
        return False
    values = np.unique(sample)
    return values.size <= MAX_FLOAT_CLASSES and bool(np.all(values == np.round(values)))


def colour_classes(v: np.ndarray, ok: np.ndarray, table: dict | None, names: dict | None, title: str,
                   order: list | None = None):
    """Palette indices of class values (0: no data) from a colour table (or distinct colours),
    their palette (opaque, but for the classes' own transparency), the legend and the name of
    each index; None for the indices when there are over 255 classes."""
    values = np.unique(v[ok]) if ok.any() else np.array([], dtype=v.dtype)
    if v.dtype == bool:
        v = v.astype(np.uint8)
        values = values.astype(np.uint8)
    if table is None:
        colours = _palettes.distinct(len(values))
        table = {int(k): (*_palettes.hex_to_rgb(c), 255) for k, c in zip(values.tolist(), colours)}
    legend = Legend(title, classes=_legend_classes(values, table, names or {}, order))
    drawn = [int(k) for k in values.tolist() if int(k) in table]
    if len(drawn) > 255:
        return None, None, legend, None
    palette = np.zeros((len(drawn) + 1, 4), np.uint8)
    for n, k in enumerate(drawn, 1):
        c = tuple(table[k]) + (255,) * (4 - len(table[k]))
        palette[n] = c[:4]
    labels = [None] + [str((names or {}).get(k, k)) for k in drawn]
    index = np.zeros(v.shape, np.uint8)
    if drawn:
        lo, hi = min(drawn), max(drawn)
        if hi - lo <= 1 << 20:
            lut = np.zeros(hi - lo + 1, np.uint8)
            lut[np.array(drawn) - lo] = np.arange(1, len(drawn) + 1)
            inside = ok & (v >= lo) & (v <= hi)
            index[inside] = lut[(v[inside].astype(np.int64) - lo)]
        else:
            for n, k in enumerate(drawn, 1):
                index[ok & (v == k)] = n
    index[~ok] = 0
    return index, palette, legend, labels


def distinct_palette(palette: np.ndarray) -> np.ndarray:
    """The palette with each colour once (a repeated one nudged by a unit of blue or green,
    which no eye sees), so that a pixel's colour gives back its index."""
    out = palette.copy()
    seen = set()
    for i in range(1, len(out)):
        c = tuple(int(x) for x in out[i, :3])
        if c in seen:
            r, g, b = c
            c = next(x for d in range(1, 256) for x in ((r, g, min(255, b + d)), (r, g, max(0, b - d)),
                                                        (r, min(255, g + d), b), (r, max(0, g - d), b))
                     if x not in seen)
        seen.add(c)
        out[i, :3] = c
    return out


def _legend_classes(values, table: dict, names: dict, order: list | None = None) -> list[tuple[str, str]]:
    """(colour, label) of the classes present; classes of the same colour (MSPA's bridges in
    edges, internal ones...) merged under their common name, as GuidosToolbox's legends do, in
    ``order`` (labels) when given. Transparent classes are left out."""
    groups: dict[str, list[str]] = {}
    for k in values.tolist():
        if int(k) not in table or (len(table[int(k)]) > 3 and table[int(k)][3] == 0):
            continue
        colour = _palettes.rgb_to_hex(table[int(k)])
        default = str(int(k)) if float(k).is_integer() else str(k)  # 1, not 1.0, for float classes
        groups.setdefault(colour, []).append(str(names.get(int(k), default)))
    out = []
    for colour, labels in groups.items():
        base = list(dict.fromkeys(lab.replace(" (internal)", "").strip() for lab in labels))
        short = min(base, key=len)
        label = short if all(b.startswith(short) for b in base) else " / ".join(base[:2]) + (" ..." if len(base) > 2
                                                                                              else "")
        out.append((colour, label))
    if order:
        out.sort(key=lambda item: order.index(item[1]) if item[1] in order else len(order))
    return out


def colour_values(v: np.ndarray, ok: np.ndarray, cmap, vmin, vmax, log: bool, title: str):
    """Palette indices 1-255 of continuous values on a colour ramp (0: no data), the (opaque)
    palette and the legend."""
    vals = v[ok]
    if log:
        vals = vals[vals > 0]
    if vmin is None or vmax is None:
        if vals.size:
            sample = vals if vals.size <= 2_000_000 else vals[:: vals.size // 2_000_000 + 1]
            lo, hi = np.percentile(np.log10(sample) if log else sample, (2, 98))
            lo, hi = (10 ** lo, 10 ** hi) if log else (lo, hi)
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmax == vmin:
        vmax = vmin + 1.0
    table = _palettes.lut(cmap, 255)
    under = bool(vals.size) and bool(vals.min() < vmin) or (log and bool((v[ok] <= 0).any()))
    over = bool(vals.size) and bool(vals.max() > vmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        if log:
            x = (np.log10(np.maximum(v, vmin)) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        else:
            x = (v.astype(np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    index = (np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0) * 254.0 + 1.5).astype(np.uint8)
    index[~ok] = 0
    palette = np.zeros((256, 4), np.uint8)
    palette[1:, :3] = table
    palette[1:, 3] = 255
    stops = [_palettes.rgb_to_hex(c) for c in table[:: 255 // 8]] + [_palettes.rgb_to_hex(table[-1])]
    return index, palette, Legend(title, ramp=stops, vmin=float(vmin), vmax=float(vmax), log=log, under=under,
                                  over=over)


def choose_cmap(da: xr.DataArray, v: np.ndarray, ok: np.ndarray) -> tuple[str, bool, bool]:
    """(palette, log scale, centred on 0) that suit a continuous raster, from its name and values."""
    name = f"{da.name or ''} {da.attrs.get('long_name', '')} {da.attrs.get('landschaft_product', '')}".lower()
    if "rank" in name or "priority" in name:
        return "rank", False, False
    if any(k in name for k in ("current", "curmap", "currmap", "flow_potential", "voltage")):
        return "current", "voltage" not in name, False
    if any(k in name for k in ("dem", "elev", "height", "altitude", "srtm")):
        return "terrain", False, False
    if "ndvi" in name or "evi" in name:
        return "rdylgn", False, False
    vals = v[ok]
    if vals.size and vals.min() < 0 < vals.max():
        lo, hi = np.percentile(vals, (2, 98))
        if lo < 0 < hi and min(-lo, hi) / max(-lo, hi) > 0.2:
            return "rdbu_r", False, True
    return "viridis", False, False


# --- reprojection ----------------------------------------------------------------------------------


def to_web_mercator(rgba: np.ndarray, transform, crs, n_threads: int, size: int | None = None):
    """An image (palette indices, or RGBA) reprojected to EPSG:3857 (nearest cell), its lat/lon
    bounds and its Web Mercator extent; ``size`` pixels on its long side (default: about as many
    as the source: a detail at the screen's resolution takes more, so that the cells' edges, turned
    by the projection, are drawn as fine as the screen)."""
    import rasterio
    from rasterio.warp import (
        Resampling,
        calculate_default_transform,
        reproject,
        transform_bounds,
    )

    h, w = rgba.shape[:2]
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * w, top + transform.e * h
    src_bounds = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    dst = rasterio.crs.CRS.from_epsg(3857)
    src_crs = rasterio.crs.CRS.from_user_input(crs)
    if src_crs.is_geographic:  # Web Mercator stops at about 85 degrees
        src_bounds = (src_bounds[0], max(src_bounds[1], -85.0), src_bounds[2], min(src_bounds[3], 85.0))
    dt, dw, dh = calculate_default_transform(src_crs, dst, w, h, *src_bounds)
    scale = max(dw / w, dh / h, 1e-9) if size is None else max(dw, dh) / max(1, size)  # as many as the source
    dw, dh = max(1, round(dw / scale)), max(1, round(dh / scale))
    dt = dt * dt.scale(scale)
    bands = rgba[None] if rgba.ndim == 2 else np.moveaxis(rgba, -1, 0)
    out = np.zeros((bands.shape[0], dh, dw), np.uint8)
    reproject(np.ascontiguousarray(bands), out, src_transform=transform, src_crs=src_crs, dst_transform=dt,
              dst_crs=dst, resampling=Resampling.nearest, num_threads=n_threads, src_nodata=0, dst_nodata=0)  # 0: transparent
    l, t = dt.c, dt.f
    r, b = l + dt.a * dw, t + dt.e * dh
    w_, s_, e_, n_ = transform_bounds(dst, "EPSG:4326", l, b, r, t)
    img = out[0] if rgba.ndim == 2 else np.ascontiguousarray(np.moveaxis(out, 0, -1))
    return img, ((s_, w_), (n_, e_)), (l, b, r, t)


# --- a layer -------------------------------------------------------------------------------------------


def _user_table(cmap: dict) -> dict:
    """{value: colour} with colours as names, hex or (r, g, b[, a]) -> {value: (r, g, b, a)}."""
    from ..io import _rgba

    return {k: _rgba(c) for k, c in cmap.items()}


def render(da: xr.DataArray, *, max_size: int = 2048, cmap=None, vmin=None, vmax=None, log=None,
           opacity: float = 0.85, categorical: bool | None = None, labels: dict | None = None,
           title: str | None = None, mercator: bool = True, n_threads: int | None = None,
           style: dict | None = None) -> Image:
    """A 2-D georeferenced raster as a coloured image for a web map (or in its own CRS).

    The colours: ``cmap`` a palette name or a list of colours (a ramp), or a dict
    ``{value: colour}`` (classes); by default the product's colours, distinct colours for
    classes, or a ramp that suits the values. ``labels`` names the classes. ``style``: the
    choices of another image (its ``style``), so that a part of the raster drawn in detail
    has the colours of the whole."""
    from .._threads import default_threads

    title = title or (str(da.name) if da.name is not None else "value")
    if style is not None:
        return _render_styled(da, style, max_size, opacity, title, n_threads or default_threads())
    table, names = _palettes.product_style(da.attrs)
    product_names = names or {}
    if isinstance(cmap, dict):
        table, categorical = _user_table(cmap), True
    elif cmap is not None and categorical is not True:
        table = None  # a ramp asked for
    if labels:
        names = {**(names or {}), **{int(k): str(v) for k, v in labels.items()}}
    if categorical is None:
        categorical = (da.dtype == bool or table is not None
                       or (cmap is None and (np.issubdtype(da.dtype, np.integer) or float_classes(da))))
    v, t = reduce(da, max_size, categorical)
    nodata = da.rio.nodata
    ok = _valid(v, nodata)
    if categorical and table is None and np.issubdtype(v.dtype, np.integer) and np.unique(v[ok]).size > MAX_CLASSES:
        categorical = False  # many integer values (counts, ids, elevations): a colour ramp
    missing_classes = []
    if categorical and product_names:  # a product's missing class is no data: the imagery shows through
        missing_classes = [k for k, n in product_names.items() if str(n).lower().startswith("missing")]
        if missing_classes:
            ok &= ~np.isin(v, missing_classes)
    if categorical:
        if table is None and cmap is not None and not isinstance(cmap, dict):  # classes in the colours of a ramp
            values = np.unique(v[ok]) if ok.any() else np.array([])
            ramp = _palettes.lut(cmap, max(len(values), 2))
            table = {int(k): (*ramp[i], 255) for i, k in enumerate(values.tolist())}
        order = _palettes.LEGEND_ORDER.get(da.attrs.get("landschaft_product"))
        if table is None:  # distinct colours, fixed now (a part drawn later keeps them)
            values = np.unique(v[ok]) if ok.any() else np.array([], dtype=v.dtype)
            table = {int(k): (*_palettes.hex_to_rgb(c), 255)
                     for k, c in zip(values.tolist(), _palettes.distinct(len(values)))}
        pixels, palette, legend, labels = colour_classes(v, ok, table, names, title, order)
        lookup = {"labels": labels}
        chosen = {"categorical": True, "table": table, "names": names, "order": order, "missing": missing_classes}
        if pixels is None:  # over 255 classes: a ramp
            categorical = False
    if not categorical:
        auto_cmap, auto_log, centred = choose_cmap(da, v, ok)
        use_log = auto_log if log is None else bool(log)
        if centred and vmin is None and vmax is None:
            m = float(np.percentile(np.abs(v[ok]), 98)) if ok.any() else 1.0
            vmin, vmax = -m, m
        ramp = cmap if isinstance(cmap, (str, list, tuple)) else auto_cmap
        pixels, palette, legend = colour_values(v, ok, ramp, vmin, vmax, use_log, title)
        lookup = ramp_lookup(legend)
        chosen = {"categorical": False, "cmap": ramp, "vmin": legend.vmin, "vmax": legend.vmax, "log": use_log}
    return _finish(da, v, t, pixels, palette, legend, lookup, title, opacity, mercator, n_threads,
                   {**chosen, "title": title})


def _finish(da, v, t, pixels, palette, legend, lookup, title, opacity, mercator, n_threads, style,
            size=None) -> Image:
    from .._threads import default_threads

    palette = distinct_palette(palette)
    xmin, ymax = t.c, t.f
    xmax, ymin = xmin + t.a * v.shape[1], ymax + t.e * v.shape[0]
    crs = da.rio.crs
    if mercator and crs is not None:
        img, bounds, merc = to_web_mercator(pixels, t, crs, n_threads or default_threads(),
                                            None if size is None or size <= max(pixels.shape) else size)
        return Image(img, bounds, legend, title, (xmin, ymin, xmax, ymax), crs, merc, palette, opacity, lookup,
                     style)
    return Image(pixels, None, legend, title, (xmin, ymin, xmax, ymax), crs, None, palette, opacity, lookup, style)


def _render_styled(da, style: dict, max_size: int, opacity: float, title: str, n_threads: int) -> Image:
    """A raster (a part of one) drawn with the colours another image chose (its style). Classes
    the other did not hold (rare ones, left out of its overview) get colours of their own."""
    if style["categorical"]:
        v, t = reduce(da, max_size, True)
        ok = _valid(v, da.rio.nodata)
        if style.get("missing"):
            ok &= ~np.isin(v, style["missing"])
        table = dict(style["table"])
        present = np.unique(v[ok]) if ok.any() else np.array([], dtype=v.dtype)
        new = [int(k) for k in present.tolist() if int(k) not in table]
        if new:
            used = {_palettes.rgb_to_hex(c[:3]) for c in table.values()}
            spare = [c for c in _palettes.distinct(len(table) + len(new) + 8) if c not in used]
            table.update({k: (*_palettes.hex_to_rgb(c), 255) for k, c in zip(new, spare)})
        pixels, palette, legend, labels = colour_classes(v, ok, table, style.get("names"), title, style.get("order"))
        if pixels is not None:
            return _finish(da, v, t, pixels, palette, legend, {"labels": labels}, title, opacity, True, n_threads,
                           style, max_size)
        style = {"categorical": False, "cmap": "viridis", "vmin": None, "vmax": None, "log": False}
    v, t = reduce(da, max_size, False)
    ok = _valid(v, da.rio.nodata)
    pixels, palette, legend = colour_values(v, ok, style["cmap"], style["vmin"], style["vmax"], style["log"], title)
    return _finish(da, v, t, pixels, palette, legend, ramp_lookup(legend), title, opacity, True, n_threads, style,
                   max_size)


def ramp_lookup(legend: Legend) -> dict:
    """What the browser needs to give the value of a ramp's index back (index 1 + 254 x)."""
    return {"vmin": legend.vmin, "vmax": legend.vmax, "log": legend.log, "under": legend.under, "over": legend.over}


def render_rgb(da: xr.DataArray, *, max_size: int = 2048, opacity: float = 1.0, title: str | None = None,
               n_threads: int | None = None) -> Image:
    """Three bands as a true- or false-colour composite (each stretched over its 2-98 percentiles)."""
    from .._threads import default_threads

    bands = []
    t = None
    ok = None
    for k in range(3):
        v, t = reduce(da.isel({da.dims[0]: k}), max_size, False)
        valid = _valid(v, da.rio.nodata)  # NoData out of the stretch, and transparent
        ok = valid if ok is None else ok & valid
        lo, hi = np.percentile(v[valid], (2, 98)) if valid.any() else (0, 1)
        bands.append(np.clip((v.astype(np.float32) - lo) / max(hi - lo, 1e-12) * 255, 0, 255))
    rgb = np.stack(bands, -1)
    rgba = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = np.nan_to_num(rgb).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    title = title or "RGB"
    crs = da.rio.crs
    img, bounds, merc = to_web_mercator(rgba, t, crs, n_threads or default_threads())
    return Image(img, bounds, None, title, None, crs, merc, opacity=opacity)
