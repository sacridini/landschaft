# SPDX-License-Identifier: GPL-3.0-or-later
"""Land systems as overlapping binary layers, and the roles of their transitions.

An extension of CLUMondo's land systems: the map is given as one binary
layer per component (forest, extensive pasture, intensive pasture, soy,
urban...), which may overlap. ``LandSystemStack`` encodes each cell's layers
as a bit mask; every combination that occurs is a land system (a mosaic when
it has several layers), so CLUMondo runs on them unchanged. ``Roles`` names
what each transition does to the layers (expansion over natural layers,
intensification, de-intensification, regeneration) so that rules, caps and
succession can be given by role instead of pair by pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr

from ._change import _class_value
from ._maps import read_map

__all__ = ["Cap", "LandSystemStack", "Roles", "Succession"]

ROLES = ("stable", "expansion", "regeneration", "intensification", "deintensification", "other")


class LandSystemStack:
    """Binary layers, possibly overlapping, as land systems (bit masks).

    Parameters
    ----------
    layers
        ``{name: map}``: 1 (or True) where the layer is present (at most 31
        layers). Cells where every layer is missing (NaN) are no data.
    overlap
        ``{(a, b): policy}`` for pairs of layers: ``"mosaic"`` (both stay: a
        system of its own), the name of the layer that wins, or ``"error"``.
    default_overlap
        The policy of the other pairs: ``"mosaic"``, ``"priority"`` (the
        layer earlier in ``priority`` wins) or ``"error"``.
    priority
        The layers from the strongest (default: the order of ``layers``).
    min_cells
        Combinations with fewer cells lose their weakest layers until they
        reach a combination with at least that many (or a single layer).
    allow_new_mosaics
        Systems absent from the map that the simulation may create: ``True``
        (every pair of layers whose overlap is ``"mosaic"``) or a list of
        tuples of layer names.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> forest = np.array([[1, 1, 0], [1, 0, 0]])
    >>> pasture = np.array([[0, 1, 1], [0, 1, 1]])
    >>> st = ls.simulate.LandSystemStack({"forest": forest, "pasture": pasture})
    >>> st.overlaps()["cells"].tolist()
    [2, 3, 1]
    >>> st.names_of(3)
    ('forest', 'pasture')
    """

    def __init__(self, layers: dict, *, overlap: dict | None = None, default_overlap: str = "mosaic",
                 priority: list | None = None, min_cells: int = 0, allow_new_mosaics=False):
        if not layers:
            raise ValueError("give at least one layer")
        if len(layers) > 31:
            raise ValueError("at most 31 layers")
        if default_overlap not in ("mosaic", "priority", "error"):
            raise ValueError("default_overlap must be 'mosaic', 'priority' or 'error'")
        self.names = [str(k) for k in layers]
        self.bits = {n: 1 << k for k, n in enumerate(self.names)}
        self.priority = [str(p) for p in (priority or self.names)]
        if set(self.priority) != set(self.names):
            raise ValueError("priority must list every layer once")
        self.overlap = {tuple(sorted(map(str, k))): v for k, v in (overlap or {}).items()}
        for (a, b), v in self.overlap.items():
            if a not in self.bits or b not in self.bits:
                raise ValueError(f"overlap for unknown layers {(a, b)}")
            if v not in ("mosaic", "error", a, b):
                raise ValueError(f"overlap policy of {(a, b)} must be 'mosaic', 'error' or one of the two layers")
        self.default_overlap = default_overlap
        maps = [read_map(m) for m in layers.values()]
        shape = maps[0].shape
        if any(m.shape != shape for m in maps):
            raise ValueError("the layers must have the same shape")
        self.template = maps[0].template
        vals = [m.as_float() for m in maps]
        nodata = np.all([np.isnan(v) for v in vals], axis=0)
        code = np.zeros(shape, dtype=np.int64)
        for n, v in zip(self.names, vals):
            code |= np.where(np.nan_to_num(v) != 0, self.bits[n], 0)
        self.raw = np.where(nodata, -1, code)
        codes = self._resolve_all(self.raw)
        codes = self._merge_rare(codes, int(min_cells))
        self.codes = codes
        present = sorted(int(c) for c in np.unique(codes[codes >= 0]))
        extra = []
        if allow_new_mosaics is True:
            extra = [self.bits[a] | self.bits[b] for a, b in combinations(self.names, 2)
                     if self._policy(a, b) == "mosaic"]
        elif allow_new_mosaics:
            extra = [self.code_of(c) for c in allow_new_mosaics]
        self.systems = sorted(set(present) | set(extra))

    # ------------------------------------------------------------------ codes

    def _policy(self, a: str, b: str) -> str:
        p = self.overlap.get(tuple(sorted((a, b))))
        if p is not None:
            return p
        if self.default_overlap == "priority":
            return a if self.priority.index(a) < self.priority.index(b) else b
        return self.default_overlap

    def _resolve(self, code: int) -> int:
        names = self.names_of(code)
        drop = set()
        for a, b in combinations(names, 2):
            if a in drop or b in drop:
                continue
            p = self._policy(a, b)
            if p == "error":
                raise ValueError(f"layers {a!r} and {b!r} overlap (policy 'error')")
            if p != "mosaic":
                drop.add(b if p == a else a)
        return int(code & ~sum(self.bits[n] for n in drop))

    def _resolve_all(self, raw: np.ndarray) -> np.ndarray:
        out = raw.copy()
        for c in np.unique(raw[raw > 0]):
            if int(c).bit_count() > 1:
                out[raw == c] = self._resolve(int(c))
        return out

    def _merge_rare(self, codes: np.ndarray, min_cells: int) -> np.ndarray:
        if min_cells <= 0:
            return codes
        codes = codes.copy()
        while True:
            vals, counts = np.unique(codes[codes > 0], return_counts=True)
            rare = [(int(v), int(n)) for v, n in zip(vals, counts) if n < min_cells and int(v).bit_count() > 1]
            if not rare:
                return codes
            v, _ = min(rare, key=lambda t: t[1])
            weakest = max(self.names_of(v), key=self.priority.index)
            codes[codes == v] = v & ~self.bits[weakest]

    def code_of(self, layers) -> int:
        """The bit mask of a set of layer names."""
        if isinstance(layers, str):
            layers = [layers]
        unknown = set(map(str, layers)) - set(self.names)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}")
        return int(sum(self.bits[str(n)] for n in layers))

    def names_of(self, code: int) -> tuple:
        """The layers of a system (bit mask)."""
        return tuple(n for n in self.names if int(code) & self.bits[n])

    def label(self, code: int) -> str:
        return "+".join(self.names_of(code)) or "none"

    # ------------------------------------------------------------------ views

    @property
    def map(self) -> xr.DataArray:
        """The systems (bit masks) as a map, NaN where no data: the landscape for :class:`Model`."""
        v = np.where(self.codes < 0, np.nan, self.codes.astype(np.float64))
        return _like(v, self.template, "land_system")

    def overlaps(self) -> pd.DataFrame:
        """The systems of the map (after the overlap policies): layers and cells, by bit mask."""
        vals, counts = np.unique(self.codes[self.codes > 0], return_counts=True)
        before = dict(zip(*np.unique(self.raw[self.raw > 0], return_counts=True)))
        rows = [{"system": int(v), "layers": self.label(int(v)), "n_layers": int(v).bit_count(),
                 "cells": int(n), "cells_before_policies": int(before.get(v, 0))} for v, n in zip(vals, counts)]
        return pd.DataFrame(rows).set_index("system")

    def layers(self, maps) -> dict:
        """Binary layers of maps of systems (``(..., y, x)``): ``{name: same shape, 1/0, NaN no data}``."""
        a = np.asarray(maps.values if isinstance(maps, xr.DataArray) else maps, dtype=np.float64)
        codes = np.nan_to_num(a, nan=0).astype(np.int64)
        out = {}
        for n in self.names:
            v = np.where(np.isnan(a), np.nan, ((codes & self.bits[n]) != 0).astype(np.float64))
            out[n] = xr.DataArray(v, dims=maps.dims, coords=maps.coords, name=n) if isinstance(maps, xr.DataArray) \
                else v
        return out

    def supply(self, per_layer: dict, *, mosaic="split"):
        """The supply of each system from the supply of its layers (a :class:`Supply`).

        Parameters
        ----------
        per_layer
            ``{layer: {service: per cell}}``.
        mosaic
            ``"split"``: a mosaic of k layers supplies 1/k of each layer's
            (each covers a share of the cell); ``"full"``: the sum of its
            layers'; ``{system or tuple of layers: {service: value}}``: given
            (the others split).
        """
        from ._clumondo import Supply

        services = sorted({s for v in per_layer.values() for s in v}, key=list(
            dict.fromkeys(s for v in per_layer.values() for s in v)).index)
        given = {}
        if isinstance(mosaic, dict):
            given = {(k if isinstance(k, (int, np.integer)) else self.code_of(k)): v for k, v in mosaic.items()}
            mode = "split"
        else:
            mode = mosaic
        if mode not in ("split", "full"):
            raise ValueError("mosaic must be 'split', 'full' or a dict")
        rows = []
        for code in self.systems:
            if code in given:
                rows.append([float(given[code].get(s, 0.0)) for s in services])
                continue
            names = self.names_of(code)
            share = 1.0 / len(names) if mode == "split" else 1.0
            rows.append([share * sum(float(per_layer.get(n, {}).get(s, 0.0)) for n in names) for s in services])
        return Supply(pd.DataFrame(rows, index=list(self.systems), columns=services))

    def suitability(self, per_layer: dict, *, combine: str = "mean"):
        """A potential for the systems from maps per layer (mosaics: the mean or max of their layers)."""
        from ._estimator import RasterPotential

        if combine not in ("mean", "max", "min"):
            raise ValueError("combine must be 'mean', 'max' or 'min'")
        arrays = {str(k): read_map(v).as_float() for k, v in per_layer.items()}
        missing = set(self.names) - set(arrays)
        if missing:
            raise ValueError(f"no suitability for layers {sorted(missing)}")
        f = getattr(np, combine)
        return RasterPotential({code: f(np.stack([arrays[n] for n in self.names_of(code)]), axis=0)
                                for code in self.systems})

    @classmethod
    def from_categorical(cls, landscape, names: dict | None = None, *, nodata=None) -> LandSystemStack:
        """A stack without overlap from a categorical map (each class a layer; ``names``: {class: name})."""
        m = read_map(landscape, nodata)
        v = m.as_float()
        classes = sorted(_class_value(c) for c in np.unique(v[~np.isnan(v)]))
        names = names or {c: str(c) for c in classes}
        layers = {names[c]: _like(np.where(np.isnan(v), np.nan, (v == c).astype(float)), m.template, names[c])
                  for c in classes}
        return cls(layers)

    def __repr__(self) -> str:
        return f"LandSystemStack({len(self.names)} layers, {len(self.systems)} systems)"


def _like(v: np.ndarray, t, name: str) -> xr.DataArray:
    if t is None:
        return xr.DataArray(v, dims=("y", "x"), name=name)
    return t.copy(data=v).rename(name)


@dataclass
class Roles:
    """What each transition between land systems does to their layers.

    Parameters
    ----------
    natural
        The natural layers (forest, native grassland...).
    families
        ``{family: [layers from the least to the most intensive]}``: the
        intensity ladders (``{"cattle": ["extensive", "intensive"]}``).

    Notes
    -----
    A transition is an ``expansion`` when it loses a natural layer, a
    ``regeneration`` when it gains one (without losing any), an
    ``intensification`` when, keeping the natural layers, it moves up a
    family's ladder or gains a productive layer, a ``deintensification``
    when it moves down or loses one; ``stable`` for no change, ``other``
    otherwise. ``set`` corrects single pairs.

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.simulate.Roles(natural=["forest"], families={"cattle": ["extensive", "intensive"]})
    >>> r.classify({"forest"}, {"extensive"}), r.classify({"extensive"}, {"intensive"})
    ('expansion', 'intensification')
    >>> r.classify({"forest", "extensive"}, {"forest", "intensive"})
    'intensification'
    """

    natural: list
    families: dict | None = None

    def __post_init__(self):
        self.natural = [str(n) for n in self.natural]
        self.families = {str(k): [str(x) for x in v] for k, v in (self.families or {}).items()}
        self._set: dict = {}

    def classify(self, before, after) -> str:
        a, b = {str(x) for x in before}, {str(x) for x in after}
        if a == b:
            return "stable"
        nat = set(self.natural)
        if (a & nat) - b:
            return "expansion"
        if (b & nat) - a:
            return "regeneration"
        up = down = False
        for ladder in self.families.values():
            la = max((ladder.index(x) for x in a if x in ladder), default=-1)
            lb = max((ladder.index(x) for x in b if x in ladder), default=-1)
            up |= lb > la
            down |= lb < la
        laddered = {x for ladder in self.families.values() for x in ladder}
        up |= bool((b - a) - nat - laddered)    # a productive layer outside the ladders gained
        down |= bool((a - b) - nat - laddered)  # or lost
        if up and not down:
            return "intensification"
        if down and not up:
            return "deintensification"
        return "other"

    def set(self, pair, role: str) -> Roles:
        """Give a pair of systems (``(from, to)``, codes or tuples of layers) its role."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}: one of {ROLES}")
        self._set[pair] = role
        return self

    def table(self, systems, layers_of=None) -> pd.DataFrame:
        """The role of every pair of systems (``layers_of``: system -> its layers; default: itself)."""
        layers_of = layers_of or (lambda s: (str(s),))
        rows = []
        for f in systems:
            for t in systems:
                role = self._lookup(f, t, layers_of)
                rows.append({"from": f, "to": t, "role": role})
        return pd.DataFrame(rows)

    def _lookup(self, f, t, layers_of) -> str:
        for key in ((f, t), (tuple(layers_of(f)), tuple(layers_of(t)))):
            if key in self._set:
                return self._set[key]
        return self.classify(layers_of(f), layers_of(t))

    def matrix(self, systems, layers_of=None) -> np.ndarray:
        """Roles as a (from, to) array of strings."""
        t = self.table(systems, layers_of)
        return t["role"].to_numpy().reshape(len(systems), len(systems))


@dataclass
class Cap:
    """At most so many cells (or so much area) a year may make a role's transitions (or given pairs).

    Parameters
    ----------
    role
        A role (``"expansion"``: deforestation) or ``None`` with ``pairs``.
    cells, area
        The limit per step, in cells, or in area units (the cells' area in
        the map's units, as ``cell_area``: ``area=50_000, cell_area=1e4 /
        1e4`` for hectares).
    pairs
        ``[(from, to), ...]`` systems instead of a role.
    cell_area
        Area of a cell for ``area`` (default: from the map's resolution).

    Notes
    -----
    When more cells would make them, those whose change scores highest over
    staying keep it; the others stay, and the demand iteration finds the
    supply elsewhere.
    """

    role: str | None = "expansion"
    cells: int | None = None
    area: float | None = None
    pairs: list | None = None
    cell_area: float | None = None

    def __post_init__(self):
        if (self.cells is None) == (self.area is None):
            raise ValueError("give either cells or area")
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")

    def limit(self, cell_area: float) -> int:
        if self.cells is not None:
            return int(self.cells)
        return int(np.floor(self.area / (self.cell_area or cell_area)))


@dataclass
class Succession:
    """Autonomous change after some years, whatever the demand (CLUMondo's code 1000 + years).

    Parameters
    ----------
    after_years
        Years a cell must have had its system.
    from_, to
        The systems (values, or tuples of layer names with a stack); or
    role
        ``"regeneration"``: each system goes to its regeneration target with
        the fewest layers (an error when that is ambiguous).
    """

    after_years: int
    from_: object = None
    to: object = None
    role: str | None = None

    def __post_init__(self):
        if (self.role is None) == (self.from_ is None or self.to is None):
            raise ValueError("give from_ and to, or a role")
        if not 1 <= int(self.after_years) < 1000:
            raise ValueError("after_years must be between 1 and 999")
