# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading R data files (``.rda`` / ``.RData`` / ``.rds``) without R.

R packages ship reference tables in R's own serialisation format; this reads
the subset that data use: vectors (logical, integer, double, complex,
character), lists, pairlists, attributes, S4 objects, environments and
symbols by reference, and the compact ALTREP vectors R 3.5+ writes
(``compact_intseq``, ``compact_realseq``, ``deferred_string``, ``wrap_*``),
in the XDR (big-endian) format of versions 2 and 3, compressed with gzip,
bzip2 or xz. Data frames become pandas DataFrames, factors their labels, and
terra's ``PackedSpatVector`` a GeoDataFrame.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["RObject", "read_rda", "read_rds", "to_python"]

NA_INTEGER = -(2**31)
_NA_REAL_BITS = 1954  # R's NA_real_: a NaN whose low word is 1954


@dataclass
class RObject:
    """An R value: its type, its value and its attributes."""

    type: str
    value: object = None
    attributes: dict = field(default_factory=dict)

    def attr(self, name, default=None):
        return self.attributes.get(name, default)

    @property
    def classes(self) -> list[str]:
        c = self.attributes.get("class")
        return list(c.value) if isinstance(c, RObject) and c.type == "character" else []


class _Reader:
    def __init__(self, data: bytes):
        self.b = memoryview(data)
        self.pos = 0
        self.refs: list = []

    def take(self, n: int) -> memoryview:
        out = self.b[self.pos:self.pos + n]
        if len(out) != n:
            raise ValueError("truncated R data")
        self.pos += n
        return out

    def int(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def length(self) -> int:
        n = self.int()
        if n == -1:  # long vector
            hi, lo = self.int(), self.int()
            return (hi << 32) + lo
        return n

    def string(self) -> str | None:
        n = self.int()
        if n == -1:
            return None
        return bytes(self.take(n)).decode("utf-8", errors="replace")

    def header(self) -> None:
        fmt = bytes(self.take(2))
        if fmt != b"X\n":
            raise ValueError(f"only the XDR format of R data is supported (got {fmt!r})")
        version = self.int()
        self.int()  # writer's R version
        self.int()  # minimal R version
        if version == 3:
            self.take(self.int())  # native encoding
        elif version != 2:
            raise ValueError(f"unsupported R serialisation version {version}")

    # --------------------------------------------------------------------- items

    def attributes(self) -> dict:
        attrs = {}
        pl = self.item()
        while isinstance(pl, RObject) and pl.type == "pairlist":
            for tag, value in pl.value:
                attrs[tag] = value
            break
        return attrs

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t == 254:  # NILVALUE
            return None
        if t in (253, 252, 251, 250, 242, 241):  # global/base/empty env, unbound, missing, base namespace
            return RObject("special", t)
        if t == 255:  # REFSXP
            index = flags >> 8
            if index == 0:
                index = self.int()
            return self.refs[index - 1]
        if t in (249, 248, 247):  # namespace, package, persist: a string vector
            if self.int() != 0:
                raise ValueError("bad persistent name")
            obj = RObject("namespace", [self.string() for _ in range(self.int())])
            self.refs.append(obj)
            return obj
        if t == 1:  # SYMSXP
            obj = RObject("symbol", self.item())
            self.refs.append(obj)
            return obj
        if t == 4:  # ENVSXP
            obj = RObject("environment")
            self.refs.append(obj)
            self.int()  # locked
            enclos, frame, hashtab, attrib = self.item(), self.item(), self.item(), self.item()
            obj.value = {"enclos": enclos, "frame": frame, "hashtab": hashtab}
            if isinstance(attrib, RObject) and attrib.type == "pairlist":
                obj.attributes = dict(attrib.value)
            return obj
        if t in (2, 3, 5, 6, 17, 239, 240):  # pairlist-like: LISTSXP, CLOSXP, PROMSXP, LANGSXP, DOTSXP, ATTR*
            items = []
            attrs = {}
            while True:
                if has_attr:
                    a = self.item()
                    if isinstance(a, RObject) and a.type == "pairlist":
                        attrs = dict(a.value)
                tag = None
                if has_tag:
                    s = self.item()
                    tag = s.value if isinstance(s, RObject) and s.type == "symbol" else s
                car = self.item()
                items.append((tag, car))
                # the cdr: another cell of the same list is read inline
                flags = self.int()
                t2 = flags & 0xFF
                if t2 == 254:
                    break
                if t2 not in (2, 239):
                    self.pos -= 4
                    rest = self.item()
                    items.append((None, rest))
                    break
                has_attr = bool(flags & (1 << 9))
                has_tag = bool(flags & (1 << 10))
            kind = {6: "language", 240: "language", 3: "closure", 5: "promise"}.get(t, "pairlist")
            return RObject(kind, items, attrs)
        if t == 9:  # CHARSXP
            return self.string()
        if t in (10, 13):  # LGLSXP, INTSXP
            n = self.length()
            v = np.frombuffer(self.take(4 * n), dtype=">i4").astype(np.int32)
            obj = RObject("logical" if t == 10 else "integer", v)
        elif t == 14:  # REALSXP
            n = self.length()
            obj = RObject("double", np.frombuffer(self.take(8 * n), dtype=">f8").astype(np.float64))
        elif t == 15:  # CPLXSXP
            n = self.length()
            v = np.frombuffer(self.take(16 * n), dtype=">f8").astype(np.float64)
            obj = RObject("complex", v[0::2] + 1j * v[1::2])
        elif t == 16:  # STRSXP
            n = self.length()
            obj = RObject("character", [self.item() for _ in range(n)])
        elif t in (19, 20):  # VECSXP, EXPRSXP
            n = self.length()
            obj = RObject("list", [self.item() for _ in range(n)])
        elif t == 25:  # S4SXP
            obj = RObject("S4")
        elif t == 238:  # ALTREP
            info, state, attr = self.item(), self.item(), self.item()
            obj = _altrep(info, state)
            if isinstance(attr, RObject) and attr.type == "pairlist":
                obj.attributes = dict(attr.value)
            return obj
        else:
            raise ValueError(f"unsupported R object type {t}")
        if has_attr:
            a = self.item()
            if isinstance(a, RObject) and a.type == "pairlist":
                obj.attributes = dict(a.value)
        return obj


def _altrep(info, state) -> RObject:
    cls = info.value[0][1].value if isinstance(info, RObject) else None
    if cls == "compact_intseq":
        n, start, step = state.value
        return RObject("integer", (start + step * np.arange(int(n))).astype(np.int32))
    if cls == "compact_realseq":
        n, start, step = state.value
        return RObject("double", start + step * np.arange(int(n), dtype=np.float64))
    if cls == "deferred_string":
        arg = state.value[0][1] if state.type == "pairlist" else state
        v = arg.value
        if arg.type == "integer":
            return RObject("character", [None if x == NA_INTEGER else str(int(x)) for x in v])
        return RObject("character", [None if np.isnan(x) else repr(float(x)).removesuffix(".0") for x in v])
    if cls and cls.startswith("wrap_"):
        return state.value[0]
    raise ValueError(f"unsupported ALTREP class {cls!r}")


def _decompress(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    if data[:3] == b"BZh":
        return bz2.decompress(data)
    if data[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(data)
    return data


def read_rda(path) -> dict[str, RObject]:
    """The objects of an ``.rda`` / ``.RData`` file (``save()``), by name, as R values."""
    data = _decompress(Path(path).read_bytes())
    if data[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise ValueError("not an R data file (RDX2 / RDX3)")
    r = _Reader(data[5:])
    r.header()
    top = r.item()
    return {tag: value for tag, value in top.value}


def read_rds(path) -> RObject:
    """The object of an ``.rds`` file (``saveRDS()``) as an R value."""
    r = _Reader(_decompress(Path(path).read_bytes()))
    r.header()
    return r.item()


# ------------------------------------------------------------------------- conversion

def _vector(obj: RObject):
    v = obj.value
    if obj.type == "integer":
        levels = obj.attr("levels")
        if "factor" in obj.classes and levels is not None:
            labels = np.array(list(levels.value) + [None], dtype=object)
            idx = np.where(v == NA_INTEGER, len(levels.value), v - 1)
            return labels[idx]
        if np.any(v == NA_INTEGER):
            return pd.arrays.IntegerArray(v, v == NA_INTEGER)
        return v
    if obj.type == "logical":
        return pd.array([None if x == NA_INTEGER else bool(x) for x in v], dtype="boolean") \
            if np.any(v == NA_INTEGER) else v.astype(bool)
    if obj.type == "character":
        return np.array(v, dtype=object)
    return v


def _data_frame(obj: RObject) -> pd.DataFrame:
    names = obj.attr("names").value
    cols = {name: _vector(col) if isinstance(col, RObject) and col.type != "list" else to_python(col)
            for name, col in zip(names, obj.value)}
    return pd.DataFrame(cols)


def _packed_spatvector(obj: RObject):
    import geopandas as gpd
    import shapely

    slots = obj.attributes
    kind = slots["type"].value[0]
    xy = slots["coordinates"].value.reshape(2, -1).T  # column-major matrix
    index = slots["index"].value.reshape(4, -1).T.astype(np.int64)  # geom, part, hole, start (1-based)
    crs = slots["crs"].value[0] or None
    attrs = _data_frame(slots["attributes"]) if slots["attributes"].attr("names") is not None else None
    starts = index[:, 3] - 1
    ends = np.append(starts[1:], len(xy))
    geoms = []
    n_geoms = int(index[:, 0].max()) if len(index) else 0
    for g in range(1, n_geoms + 1):
        rows = np.flatnonzero(index[:, 0] == g)
        if kind == "polygons":
            polys = []
            for p in np.unique(index[rows, 1]):
                prow = rows[index[rows, 1] == p]
                shell = None
                holes = []
                for k in prow:
                    ring = xy[starts[k]:ends[k]]
                    if index[k, 2] == 0:
                        shell = ring
                    else:
                        holes.append(ring)
                polys.append(shapely.Polygon(shell, holes))
            geoms.append(polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys))
        elif kind == "lines":
            parts = [shapely.LineString(xy[starts[k]:ends[k]]) for k in rows]
            geoms.append(parts[0] if len(parts) == 1 else shapely.MultiLineString(parts))
        else:
            pts = [xy[starts[k]:ends[k]] for k in rows]
            pts = np.concatenate(pts)
            geoms.append(shapely.Point(pts[0]) if len(pts) == 1 else shapely.MultiPoint(pts))
    return gpd.GeoDataFrame(attrs if attrs is not None and len(attrs) == len(geoms) else None,
                            geometry=geoms, crs=crs)


def to_python(obj):
    """An R value as Python: data frames as DataFrames, factors as labels, vectors as
    arrays, named lists as dicts, terra's PackedSpatVector as a GeoDataFrame."""
    if not isinstance(obj, RObject):
        return obj
    classes = obj.classes
    if "data.frame" in classes:
        return _data_frame(obj)
    if obj.type == "S4" and "PackedSpatVector" in classes:
        return _packed_spatvector(obj)
    if obj.type in ("integer", "double", "logical", "character", "complex"):
        v = _vector(obj)
        dim = obj.attr("dim")
        if dim is not None and isinstance(v, np.ndarray):
            return v.reshape(tuple(int(d) for d in dim.value[::-1])).T
        return v
    if obj.type == "list":
        names = obj.attr("names")
        values = [to_python(v) for v in obj.value]
        return dict(zip(names.value, values)) if names is not None else values
    if obj.type == "pairlist":
        return {tag: to_python(v) for tag, v in obj.value}
    if obj.type == "S4":
        return {k: to_python(v) for k, v in obj.attributes.items() if k != "class"}
    return obj
