# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial conditions: which groups are neighbours of which.

Conditions are written as in Biodiverse (Perl expressions such as
``sp_circle (radius => 200000)`` or ``$D <= 3 && $nbr_y > $y``). They are
parsed here, the parts that depend on one group only (text and regular
expression matches, polygons, richness) are evaluated once per group, and
the rest runs in C++ for every candidate pair with Biodiverse's arithmetic.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _core
from .._threads import resolve
from ._perlexpr import PerlSyntaxError, parse

__all__ = ["SpatialCondition", "Neighbourhoods", "neighbours", "definition_query", "result_type"]

# Math::Complex's constants and Math::Trig::deg2rad (reduced modulo 2 pi)
_PI = 4 * math.atan2(1, 1)
_PI2 = 2 * _PI
_PIP2 = _PI / 2


def _deg2rad(deg: float) -> float:
    d = (_PI2 / 360) * deg
    return d - _PI2 * math.trunc(d / _PI2)


_OPS = {
    "Num": 0, "Var": 1, "IdxVar": 2, "Neg": 3, "Not": 4, "Add": 5, "Sub": 6, "Mul": 7, "Div": 8, "Mod": 9,
    "Pow": 10, "Lt": 11, "Le": 12, "Gt": 13, "Ge": 14, "Eq": 15, "Ne": 16, "Cmp": 17, "And": 18, "Or": 19,
    "Dor": 20, "Tern": 21, "Func": 22, "StrEq": 23, "StrNe": 24, "Call": 25, "Local": 26, "SetLocal": 27,
    "Seq": 28, "StrConst": 29, "CoordId": 30, "AxisStr": 31, "Undef": 32,
}
_VARS = {"D": 0, "C": 1, "Dsqr": 2, "Csqr": 3, "x": 4, "y": 5, "z": 6, "nbr_x": 7, "nbr_y": 8, "nbr_z": 9}
_ARRS = {"d": 0, "D": 1, "c": 2, "C": 3, "coord": 4, "nbrcoord": 5}
_FUNCS = {"abs": 0, "sqrt": 1, "sin": 2, "cos": 3, "log": 4, "exp": 5, "int": 6, "atan2": 7}
_BINOPS = {
    "+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "%": "Mod", "**": "Pow", "<": "Lt", "<=": "Le", ">": "Gt",
    ">=": "Ge", "==": "Eq", "!=": "Ne", "<=>": "Cmp", "&&": "And", "and": "And", "||": "Or", "or": "Or", "//": "Dor",
}
_CALLS = {
    "Circle": 0, "CircleCell": 1, "Rectangle": 2, "Annulus": 3, "Square": 4, "SquareCell": 5, "Ellipse": 6,
    "Block": 7, "SelfOnly": 8, "SelectAll": 9, "Side": 10, "Mask": 11, "Const": 12, "SameFeature": 13,
    "ExplicitSet": 14, "ElementIs": 15, "TextEq": 16,
}
_INF = math.inf


class _Unsupported(NotImplementedError):
    pass


def _perl_regex(pattern: str, flags: str) -> re.Pattern:
    """A Perl regex as a Python one (the common subset)."""
    f = 0
    if "i" in flags:
        f |= re.IGNORECASE
    if "x" in flags:
        f |= re.VERBOSE
    if "s" in flags:
        f |= re.DOTALL
    if "m" in flags:
        f |= re.MULTILINE
    pat = pattern.replace(r"\z", r"\Z").replace(r"\A", r"\A")
    pat = re.sub(r"\$(?=\)|\||$)", r"(?=\\n?\\Z)", pat)
    return re.compile(pat, f)


def polygon_contains(points: np.ndarray, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """Math::Polygon's ``contains`` for many points (boundary counts as inside)."""
    poly = [tuple(map(float, p[:2])) for p in polygon]
    if len(poly) < 3:
        return np.zeros(len(points), bool)
    if poly[0] != poly[-1]:
        raise ValueError("polygon must be closed: begin==end")
    x = np.asarray(points[:, 0], dtype=np.float64)
    y = np.asarray(points[:, 1], dtype=np.float64)
    inside = np.zeros(len(x), bool)
    done = np.zeros(len(x), bool)
    result = np.zeros(len(x), bool)
    px, py = poly[0]
    for nx, ny in poly[1:]:
        live = ~done
        on_h = live & (y == py) & (py == ny) & ((x >= px) | (x >= nx)) & ((x <= px) | (x <= nx))
        result |= on_h
        done |= on_h
        live = ~done
        on_v = live & (x == px) & (px == nx) & ((y >= py) | (y >= ny)) & ((y <= py) | (y <= ny))
        result |= on_v
        done |= on_v
        live = ~done
        skip = (py == ny) | ((y <= py) & (y <= ny)) | ((y > py) & (y > ny)) | ((x > px) & (x > nx))
        test = live & ~skip
        if test.any() and ny != py:
            xinters = (y - py) * (nx - px) / (ny - py) + px
            flip = test & ((px == nx) | (x <= xinters))
            inside ^= flip
        px, py = nx, ny
    return np.where(done, result, inside)


@dataclass
class _Compiled:
    nodes: list = field(default_factory=list)       # (op, a, b, c, v)
    seq: list = field(default_factory=list)
    calls: list = field(default_factory=list)       # dicts
    masks: list = field(default_factory=list)       # uint8 arrays
    sets: list = field(default_factory=list)        # (indptr, indices) or ("sequence", params)
    rotations: list = field(default_factory=list)
    locals: dict = field(default_factory=dict)
    use_euc: bool = False
    use_cell: bool = False
    euc_axes: set = field(default_factory=set)      # $d[i], $D[i]
    cell_axes: set = field(default_factory=set)     # $c[i]
    abs_cell_axes: set = field(default_factory=set)  # $C[i]
    strings: dict = field(default_factory=dict)


class SpatialCondition:
    """A Biodiverse spatial condition (Perl syntax) compiled for a BaseData.

    The supported Perl: the variables ``$D $C $Dsqr $Csqr $d[i] $D[i] $c[i]
    $C[i] $x $y $z $nbr_x $nbr_y $nbr_z $coord[i] $nbrcoord[i] $coord_id1
    $coord_id2``, ``my`` variables, arithmetic, comparisons, ``eq``/``ne``,
    ``&& || // ! and or not``, ``?:``, ``abs sqrt sin cos atan2 log exp int``
    and the functions ``sp_circle sp_circle_cell sp_rectangle sp_annulus
    sp_square sp_square_cell sp_ellipse sp_block sp_self_only sp_select_all
    sp_select_element sp_select_sequence sp_is_left_of sp_is_right_of
    sp_in_line_with sp_match_text sp_match_regex sp_group_not_empty
    sp_richness_greater_than sp_redundancy_greater_than sp_point_in_poly
    sp_point_in_poly_shape sp_points_in_same_poly_shape``.

    Parameters
    ----------
    text : str
        The condition, e.g. ``"sp_circle (radius => 200000)"``,
        ``"$D <= 3"`` or ``"sp_block (size => 500000)"``.

    Examples
    --------
    >>> import landschaft as ls
    >>> c = ls.bio.SpatialCondition("sp_circle (radius => 1.5)")
    >>> c.max_distance
    1.5
    """

    def __init__(self, text: str) -> None:
        self.text = text.strip()
        try:
            self.ast = parse(self.text)
        except PerlSyntaxError as e:
            raise ValueError(f"cannot parse spatial condition {text!r}: {e}") from None

    def __repr__(self) -> str:
        return f"SpatialCondition({self.text!r})"

    def __str__(self) -> str:
        return self.text

    # -- analysis ------------------------------------------------------------------
    @property
    def max_distance(self) -> float:
        """Largest distance (any axis) at which a neighbour can be (inf if unknown)."""
        b = _bounds(self.ast, 8)
        return max(b) if b else _INF

    @property
    def result_type(self) -> str:
        """Biodiverse's result type of the condition (see :func:`result_type`)."""
        return result_type(self.text)

    def kind(self) -> str:
        """``self_only``, ``always_true``, ``always_false``, ``always_same`` or ``complex``."""
        t = self.result_type
        if t in ("self_only", "always_true", "always_false", "always_same"):
            return t
        return "complex"

    # -- compilation -----------------------------------------------------------------
    def compile(self, bd, procs: np.ndarray | None = None, def_query: bool = False) -> dict:
        """Program for the C++ evaluator (see src/cpp/bio/conditions.hpp)."""
        comp = _Compiled()
        ctx = _Context(bd, comp, procs, def_query)
        root = ctx.node(self.ast)
        n_axes = bd.n_axes
        use_euc, use_cell = comp.use_euc, comp.use_cell
        iter_axes = sorted({*comp.euc_axes, *comp.cell_axes, *comp.abs_cell_axes}, key=str)
        bounds = _bounds(self.ast, n_axes)
        return {
            "nodes": comp.nodes,
            "seq": comp.seq,
            "root": root,
            "n_locals": len(comp.locals),
            "calls": comp.calls,
            "masks": comp.masks,
            "sets": comp.sets,
            "rotations": comp.rotations,
            "use_euc": use_euc,
            "use_cell": use_cell,
            "cell_lists": use_cell or bool(comp.cell_axes),
            "iter_axes": [int(a) for a in iter_axes],
            "bounds": [float(b) for b in bounds],
            "self_only": False,
            "always_true": False,
            "always_false": False,
            "strings": ctx.string_table(),
        }


# result types of the sp_* functions (their metadata in Biodiverse)
_SP_RESULT_TYPES = {
    "sp_circle": "circle", "sp_circle_cell": "circle", "sp_rectangle": "circle", "sp_annulus": "circle",
    "sp_square": "square", "sp_square_cell": "square", "sp_ellipse": "ellipse", "sp_block": "non_overlapping",
    "sp_self_only": "self_only", "sp_select_all": "always_true", "sp_select_element": "always_same",
    "sp_select_sequence": "subset", "sp_select_block": "complex", "sp_is_left_of": "side", "sp_is_right_of": "side",
    "sp_in_line_with": "side", "sp_match_text": "text_match_exact", "sp_match_regex": "non_overlapping",
    "sp_group_not_empty": "", "sp_richness_greater_than": "always_same", "sp_redundancy_greater_than": "always_same",
    "sp_point_in_poly": "always_same", "sp_point_in_poly_shape": "always_same",
    "sp_points_in_same_poly_shape": "non_overlapping",
}
_RE_NUMBER = r"[+-]?(?=\.?\d)\d*(?:\.\d*)?(?:[eE][+-]?\d+)?"
_BOUNDED = re.compile(r"\$nbr(?:_[xyz]|coord\[[+-]?\d+\])\s*(?:<|>|<=|>=|==)\s*(?:" + _RE_NUMBER + ")")


def _perl_looks_like_number(s: str) -> bool:
    return re.fullmatch(r"\s*" + _RE_NUMBER + r"\s*", s) is not None


def result_type(text: str) -> str:
    """Biodiverse's result type of a condition text.

    Biodiverse joins the result types of the ``sp_*`` functions the text
    calls (plus ``non_overlapping`` for each ``$nbr_x < number``-like term)
    and, when that is empty, guesses from the text (``$D <= 5`` is a circle,
    ``$D == 0`` self only...). The analysis treats ``self_only``,
    ``always_true``, ``always_false`` and ``always_same`` conditions as whole
    shortcuts, and recycles neighbour lists for ``non_overlapping``,
    ``always_true``, ``text_match_exact`` and ``always_same`` ones, whatever
    else the text contains.
    """
    cond = text.strip() + "\n"
    cond = re.sub(r"#[^\n]*\n", "", cond)
    types = " non_overlapping" * len(_BOUNDED.findall(cond))
    names = "|".join(sorted(_SP_RESULT_TYPES, key=len, reverse=True))
    pos = 0
    while pos < len(cond):
        m = re.compile(r"\b(" + names + r")\b\s*").match(cond, pos)
        if m:
            types += " " + _SP_RESULT_TYPES[m.group(1)]
            pos = m.end()
            if pos < len(cond) and cond[pos] == "(":
                depth = 0
                while pos < len(cond):
                    if cond[pos] == "(":
                        depth += 1
                    elif cond[pos] == ")":
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                    pos += 1
            continue
        pos += 1
    types = types.lstrip()
    if types and re.search(r"\S", types):
        return types
    c = re.sub(r"\n\Z", "", cond)
    if _perl_looks_like_number(c) and float(c) != 0:
        return "always_true"
    m = re.fullmatch(r"\s*\$[DC]\s*>=\s*(" + _RE_NUMBER + r")\s*(?:#.*)*", c)
    if m and float(m.group(1)) == 0:
        return "always_true"
    m = re.fullmatch(r"\s*(" + _RE_NUMBER + r")+\s*(?:#.*)*", c) or re.fullmatch(r"\$[DC]\s*<\s*(" + _RE_NUMBER + r")", c)
    if (m and float(m.group(1)) == 0) or c == "" or re.fullmatch(r"[\r\n]+", c):
        return "always_false"
    m = re.fullmatch(r"\$[DC]\s*<=+\s*(.+)", c)
    if m and _perl_looks_like_number(m.group(1)):
        return "circle"
    if re.fullmatch(r"\$[DC][<=]=0", c):
        return "self_only"
    m = re.fullmatch(
        r"\$[DC]\s*([<>])=\s*(" + _RE_NUMBER + r")\s*(?:and|&&)\s*\$[DC]\s*([<>])=\s*(" + _RE_NUMBER + r")\s*(?:#.*)*", c
    )
    if m and m.group(1) != m.group(3) and float(m.group(2)) <= float(m.group(4)):
        return "circle"
    return "complex"


_RECYCLABLE = {"non_overlapping": 0, "always_true": None, "text_match_exact": None, "always_same": None}


def _recyclable_sets(types: list[str]) -> list[bool]:
    out = []
    for i, t in enumerate(types):
        prev = out[i - 1] if i > 0 else True
        ok = prev and t in _RECYCLABLE and (_RECYCLABLE[t] is None or i <= _RECYCLABLE[t])
        out.append(bool(ok))
    return out


def _as_condition(c) -> SpatialCondition:
    return c if isinstance(c, SpatialCondition) else SpatialCondition(str(c))


# ---------------------------------------------------------------------------------------
# bounds: half-width per axis of the box holding every neighbour


def _bounds(ast, n_axes: int) -> list[float]:
    inf = [_INF] * n_axes

    def all_axes(v):
        return [v] * n_axes

    def rec(a) -> list[float]:
        kind = a[0]
        if kind == "bin":
            op = a[1]
            if op in ("&&", "and"):
                x, y = rec(a[2]), rec(a[3])
                return [min(p, q) for p, q in zip(x, y)]
            if op in ("||", "or"):
                x, y = rec(a[2]), rec(a[3])
                return [max(p, q) for p, q in zip(x, y)]
            if op in ("<=", "<") and a[2] == ("var", "D") and _is_const(a[3]):
                return all_axes(_const(a[3]))
            if op in (">=", ">") and a[3] == ("var", "D") and _is_const(a[2]):
                return all_axes(_const(a[2]))
            if op in ("<=", "<") and a[2][0] == "idx" and a[2][1] in ("D", "d") and _is_const(a[3]) and _is_const(a[2][2]):
                if a[2][1] == "D":
                    out = list(inf)
                    k = int(_const(a[2][2]))
                    if 0 <= k < n_axes:
                        out[k] = _const(a[3])
                    return out
            return inf
        if kind == "call":
            name, args = a[1], _call_args(a[2])
            try:
                if name == "sp_circle" and _is_const(args.get("radius")):
                    r = _const(args["radius"])
                    if "axes" in args:
                        out = list(inf)
                        for k in _const(args["axes"]):
                            if 0 <= int(k) < n_axes:
                                out[int(k)] = r
                        return out
                    return all_axes(r)
                if name == "sp_annulus" and _is_const(args.get("outer_radius")):
                    r = _const(args["outer_radius"])
                    if "axes" in args:
                        out = list(inf)
                        for k in _const(args["axes"]):
                            if 0 <= int(k) < n_axes:
                                out[int(k)] = r
                        return out
                    return all_axes(r)
                if name == "sp_rectangle" and _is_const(args.get("sizes")):
                    sizes = _const(args["sizes"])
                    axes = _const(args["axes"]) if "axes" in args else list(range(len(sizes)))
                    out = list(inf)
                    for k, s in zip(axes, sizes):
                        if 0 <= int(k) < n_axes:
                            out[int(k)] = float(s) / 2
                    return out
                if name == "sp_square" and _is_const(args.get("size")):
                    return all_axes(_const(args["size"]) / 2)
                if name == "sp_ellipse" and _is_const(args.get("major_radius")):
                    r = _const(args["major_radius"])
                    axes = _const(args["axes"]) if "axes" in args else [0, 1]
                    out = list(inf)
                    for k in axes:
                        if 0 <= int(k) < n_axes:
                            out[int(k)] = r
                    return out
                if name == "sp_block" and _is_const(args.get("size")):
                    size = _const(args["size"])
                    sizes = size if isinstance(size, list) else [size] * n_axes
                    return [float(s) if s is not None and k < len(sizes) else _INF for k, s in enumerate(sizes[:n_axes])] + [
                        _INF
                    ] * max(0, n_axes - len(sizes))
                if name == "sp_self_only":
                    return all_axes(0.0)
            except (TypeError, ValueError):
                return inf
            return inf
        if kind == "seq":
            return rec(a[1][-1])
        return inf

    return rec(ast)


# ---------------------------------------------------------------------------------------
# constants


def _is_const(a) -> bool:
    try:
        _const(a)
        return True
    except (_Unsupported, TypeError, KeyError, ValueError):
        return False


def _const(a):
    """Value of an expression without variables (numbers, strings, lists)."""
    if a is None:
        raise _Unsupported("missing")
    kind = a[0]
    if kind == "num":
        return a[1]
    if kind == "str":
        return a[1]
    if kind == "regex":
        return _perl_regex(a[1], a[2])
    if kind == "undef":
        return None
    if kind == "list":
        return [_const(x) for x in a[1]]
    if kind == "un":
        v = _const(a[2])
        if a[1] == "-":
            return -float(v)
        if a[1] == "!":
            return 0.0 if v else 1.0
    if kind == "bin" and a[1] in ("+", "-", "*", "/", "**", "."):
        x, y = _const(a[2]), _const(a[3])
        if a[1] == ".":
            return f"{x}{y}"
        x, y = float(x), float(y)
        return {"+": x + y, "-": x - y, "*": x * y, "/": x / y, "**": x**y}[a[1]]
    if kind == "call" and a[1] in ("deg2rad", "Math::Trig::deg2rad") and len(a[2]) == 1:
        return _deg2rad(float(_const(a[2][0])))
    if kind == "call" and a[1] in ("atan2",) and len(a[2]) == 2:
        return math.atan2(float(_const(a[2][0])), float(_const(a[2][1])))
    raise _Unsupported(f"not a constant: {a!r}")


def _call_args(items) -> dict:
    """Named arguments of a call (``key => value`` pairs or a flat list)."""
    out = {}
    flat = []
    for it in items:
        if it[0] == "pair":
            out[_const(it[1])] = it[2]
        else:
            flat.append(it)
    for k in range(0, len(flat) - 1, 2):
        out[_const(flat[k])] = flat[k + 1]
    return out


# ---------------------------------------------------------------------------------------
# compilation to C++ nodes


class _Context:
    def __init__(self, bd, comp: _Compiled, procs, def_query: bool) -> None:
        self.bd = bd
        self.c = comp
        self.procs = procs
        self.def_query = def_query
        self.strings: dict[str, int] = {}
        # group names and text-axis values come first in the string table
        for g in bd.groups:
            self.sid(g)
        self._axis_str = None

    # strings --------------------------------------------------------------------
    def sid(self, s: str) -> int:
        i = self.strings.get(s)
        if i is None:
            i = self.strings[s] = len(self.strings)
        return i

    def string_table(self) -> list[str]:
        return list(self.strings)

    # nodes ------------------------------------------------------------------------
    def add(self, op: str, a=-1, b=-1, c=-1, v=0.0) -> int:
        self.c.nodes.append((_OPS[op], int(a), int(b), int(c), float(v)))
        return len(self.c.nodes) - 1

    def num(self, v) -> int:
        return self.add("Num", v=float(v))

    def node(self, a) -> int:
        kind = a[0]
        if kind == "num":
            return self.num(a[1])
        if kind == "undef":
            return self.add("Undef")
        if kind == "str":
            return self.add("StrConst", a=self.sid(a[1]))
        if kind == "var":
            name = a[1]
            if name in self.c.locals:
                return self.add("Local", a=self.c.locals[name])
            if name in ("coord_id1", "coord_id2"):
                return self.add("CoordId", a=0 if name == "coord_id1" else 1)
            if name not in _VARS:
                raise ValueError(f"unknown variable ${name}")
            if name == "D":
                self.c.use_euc = True
            elif name == "C":
                self.c.use_cell = True
            return self.add("Var", a=_VARS[name])
        if kind == "idx":
            name = a[1]
            if name not in _ARRS:
                raise ValueError(f"unknown array @{name}")
            if name in ("d", "D", "c", "C"):
                try:
                    k = int(_const(a[2]))
                except _Unsupported:
                    raise ValueError(f"the subscript of ${name}[...] must be a constant") from None
                {"d": self.c.euc_axes, "D": self.c.euc_axes, "c": self.c.cell_axes, "C": self.c.abs_cell_axes}[name].add(k)
            return self.add("IdxVar", a=_ARRS[name], b=self.node(a[2]))
        if kind == "un":
            op = a[1]
            if op == "-":
                return self.add("Neg", a=self.node(a[2]))
            return self.add("Not", a=self.node(a[2]))
        if kind == "bin":
            op = a[1]
            if op in ("eq", "ne"):
                return self.add("StrEq" if op == "eq" else "StrNe", a=self.str_node(a[2]), b=self.str_node(a[3]))
            if op in ("=~", "!~"):
                raise _Unsupported("regex matches outside sp_match_regex are not supported")
            if op == "xor":
                x, y = self.node(a[2]), self.node(a[3])
                nx, ny = self.add("Not", a=x), self.add("Not", a=y)
                return self.add("Ne", a=nx, b=ny)
            if op not in _BINOPS:
                raise _Unsupported(f"operator {op!r} is not supported")
            return self.add(_BINOPS[op], a=self.node(a[2]), b=self.node(a[3]))
        if kind == "tern":
            return self.add("Tern", a=self.node(a[1]), b=self.node(a[2]), c=self.node(a[3]))
        if kind == "my":
            slot = self.c.locals.setdefault(a[1], len(self.c.locals))
            return self.add("SetLocal", a=slot, b=self.node(a[2]))
        if kind == "seq":
            kids = [self.node(x) for x in a[1]]
            start = len(self.c.seq)
            self.c.seq.extend(kids)
            return self.add("Seq", a=start, b=len(kids))
        if kind == "call":
            return self.call(a[1], a[2])
        raise _Unsupported(f"unsupported expression {a!r}")

    def str_node(self, a) -> int:
        kind = a[0]
        if kind == "idx" and a[1] in ("coord", "nbrcoord"):
            return self.add("AxisStr", a=0 if a[1] == "coord" else 1, b=self.node(a[2]))
        return self.node(a)

    # calls ------------------------------------------------------------------------
    def add_call(self, kind: str, **kw) -> int:
        call = {"kind": _CALLS[kind], "p": [], "axes": [], "sizes": [], "origin": [], "which": 0, "index": -1,
                "text": -1, "axis": -1}
        call.update(kw)
        self.c.calls.append(call)
        return self.add("Call", a=len(self.c.calls) - 1)

    def const_call(self, value: bool) -> int:
        return self.add_call("Const", p=[1.0 if value else 0.0])

    def mask(self, m: np.ndarray, which: int) -> int:
        self.c.masks.append(np.ascontiguousarray(m, dtype=np.uint8))
        return self.add_call("Mask", index=len(self.c.masks) - 1, which=which)

    def which(self, args) -> int:
        t = _const(args["type"]) if "type" in args else "nbr"
        if t not in ("nbr", "proc"):
            raise ValueError(f"type must be 'nbr' or 'proc', not {t!r}")
        return 1 if t == "proc" else 0

    def comparators(self, args) -> list[str]:
        """Text each group offers to sp_match_text / sp_match_regex."""
        bd = self.bd
        if "axis" in args:
            k = int(_const(args["axis"]))
            if abs(k) > bd.n_axes - 1:
                raise ValueError(f"axis argument {k} beyond array bounds")
            return [bd.group_axes(g)[k] for g in bd.groups]
        return list(bd.groups)

    def call(self, name: str, items) -> int:
        if name in _FUNCS:
            if name == "atan2":
                if len(items) != 2:
                    raise ValueError("atan2 takes two arguments")
                return self.add("Func", a=_FUNCS[name], b=self.node(items[0]), c=self.node(items[1]))
            if len(items) != 1:
                raise ValueError(f"{name} takes one argument")
            return self.add("Func", a=_FUNCS[name], b=self.node(items[0]))
        if name in ("deg2rad", "Math::Trig::deg2rad"):
            return self.num(_const(("call", "deg2rad", items)))
        if not name.startswith("sp_"):
            raise _Unsupported(f"function {name} is not supported")
        args = _call_args(items)
        bd = self.bd
        n_axes = bd.n_axes

        def axes_arg(default=None):
            if "axes" not in args:
                return default
            ax = _const(args["axes"])
            if not isinstance(ax, list):
                raise ValueError("axes argument must be an array ref")
            return [int(x) for x in ax]

        if name == "sp_circle":
            ax = axes_arg()
            if ax is None:
                self.c.use_euc = True
            else:
                self.c.euc_axes.update(ax)
            return self.add_call("Circle", p=[float(_const(args["radius"]))], axes=ax or [])
        if name == "sp_circle_cell":
            self.c.use_cell = True
            return self.add_call("CircleCell", p=[float(_const(args["radius"]))], axes=axes_arg() or [])
        if name == "sp_rectangle":
            self.c.use_euc = True
            sizes = [float(s) for s in _const(args["sizes"])]
            ax = axes_arg(list(range(len(sizes))))
            if len(ax) > len(sizes):
                raise ValueError("Too many axes in call to sp_rectangle")
            return self.add_call("Rectangle", sizes=sizes, axes=ax)
        if name == "sp_annulus":
            ax = axes_arg()
            if ax is None:
                self.c.use_euc = True
            else:
                self.c.euc_axes.update(ax)
            return self.add_call(
                "Annulus", p=[float(_const(args["inner_radius"])), float(_const(args["outer_radius"]))], axes=ax or []
            )
        if name == "sp_square":
            size = _const(args["size"])
            if not isinstance(size, (int, float)):
                raise ValueError("Size argument to sp_square is not numeric.  Did you mean to use sp_rectangle?")
            self.c.use_euc = True
            return self.add_call("Square", p=[float(size)])
        if name == "sp_square_cell":
            self.c.use_cell = True
            return self.add_call("SquareCell", p=[float(_const(args["size"]))])
        if name == "sp_ellipse":
            ax = axes_arg([0, 1])
            if len(ax) != 2:
                raise ValueError(f"sp_ellipse:  axes array needs two axes, you have given {len(ax)}")
            self.c.euc_axes.update(ax)
            if "rotate_angle" in args:
                angle = float(_const(args["rotate_angle"]))
            elif "rotate_angle_deg" in args:
                angle = _deg2rad(float(_const(args["rotate_angle_deg"])))
            else:
                angle = 0.0
            return self.add_call(
                "Ellipse", p=[float(_const(args["major_radius"])), float(_const(args["minor_radius"])), angle], axes=ax
            )
        if name == "sp_block":
            size = _const(args["size"])
            sizes = list(size) if isinstance(size, list) else [size] * n_axes
            sizes = sizes[:n_axes]
            origin = _const(args["origin"]) if "origin" in args else None
            if not origin:
                origin = [0.0] * n_axes
            elif not isinstance(origin, list):
                origin = [origin] * n_axes
            origin = [0.0 if o is None else float(o) for o in origin[:n_axes]]
            return self.add_call(
                "Block", sizes=[math.nan if s is None else float(s) for s in sizes], origin=origin
            )
        if name == "sp_self_only":
            return self.add_call("SelfOnly")
        if name == "sp_select_all":
            return self.add_call("SelectAll")
        if name in ("sp_is_left_of", "sp_is_right_of", "sp_in_line_with"):
            ax = axes_arg([0, 1])
            if len(ax) != 2:
                raise ValueError(f"_sp_side:  axes array needs two axes, you have given {len(ax)}")
            if "vector_angle" in args:
                va = float(_const(args["vector_angle"]))
            else:
                va = _deg2rad(float(_const(args["vector_angle_deg"])) if "vector_angle_deg" in args else 0.0)
            angle = _PIP2 - va
            rot = _core.bio_side_rotation(np.ascontiguousarray(bd.coords, dtype=np.float64), ax[0], ax[1], angle)
            self.c.rotations.append(np.asarray(rot, dtype=np.float64))
            want = {"sp_is_left_of": -1.0, "sp_is_right_of": 1.0, "sp_in_line_with": 0.0}[name]
            return self.add_call("Side", p=[want], axes=ax, index=len(self.c.rotations) - 1)
        if name == "sp_select_element":
            el = _const(args["element"])
            idx = bd._group_index.get(str(el), -1)
            return self.add_call("ElementIs", p=[float(idx)], which=self.which(args))
        if name == "sp_match_text":
            which = self.which(args)
            text = args["text"]
            if _is_const(text):
                t = str(_const(text))
                m = np.array([c == t for c in self.comparators(args)], dtype=bool)
                return self.mask(m, which)
            axis = int(_const(args["axis"])) if "axis" in args else -1
            if axis < -1:
                axis += n_axes
            return self.add_call("TextEq", text=self.str_node(text), axis=axis, which=which)
        if name == "sp_match_regex":
            which = self.which(args)
            rx = _const(args["re"])
            if not isinstance(rx, re.Pattern):
                rx = re.compile(str(rx))
            m = np.array([rx.search(c) is not None for c in self.comparators(args)], dtype=bool)
            return self.mask(m, which)
        if name in ("sp_group_not_empty", "sp_richness_greater_than", "sp_redundancy_greater_than"):
            rich = np.diff(bd.counts.indptr).astype(np.float64)
            if name == "sp_group_not_empty":
                values = rich > 0
            else:
                thr = float(_const(args["threshold"]))
                if name == "sp_richness_greater_than":
                    values = rich > thr
                else:
                    samples = np.asarray(bd.group_sample_count, dtype=np.float64)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        red = 1 - rich / samples
                    values = red > thr
            if "element" in args:
                el = str(_const(args["element"]))
                if el not in bd._group_index:
                    raise ValueError(f"element {el} is not in basedata")
                return self.const_call(bool(values[bd._group_index[el]]))
            return self.mask(values, 0)
        if name == "sp_point_in_poly":
            polygon = _const(args["polygon"])
            which = 0
            if "point" in args:
                pa = args["point"]
                if pa[0] == "aref" and pa[1] in ("nbrcoord", "coord"):
                    which = 1 if pa[1] == "coord" else 0
                elif _is_const(pa):
                    pt = np.array([_const(pa)[:2]], dtype=np.float64)
                    return self.const_call(bool(polygon_contains(pt, polygon)[0]))
                else:
                    raise _Unsupported("sp_point_in_poly: point must be \\@nbrcoord, \\@coord or a constant")
            m = polygon_contains(bd.coords[:, :2], polygon)
            return self.mask(m, which)
        if name == "sp_point_in_poly_shape":
            ax = axes_arg([0, 1])
            which = 0
            if "point" in args:
                pa = args["point"]
                if pa[0] == "aref" and pa[1] in ("nbrcoord", "coord"):
                    which = 1 if pa[1] == "coord" else 0
                else:
                    raise _Unsupported("sp_point_in_poly_shape: point must be \\@nbrcoord or \\@coord")
            m = _points_in_shapes(bd.coords[:, ax], str(_const(args["file"])),
                                  _const(args["field_name"]) if "field_name" in args else None,
                                  _const(args["field_val"]) if "field_val" in args else None)
            return self.mask(m, which)
        if name == "sp_points_in_same_poly_shape":
            if "point1" in args or "point2" in args:
                raise _Unsupported("sp_points_in_same_poly_shape: point1/point2 are not supported")
            ax = axes_arg([0, 1])
            indptr, indices = _features_per_point(bd.coords[:, ax], str(_const(args["file"])))
            self.c.sets.append((indptr, indices))
            return self.add_call("SameFeature", index=len(self.c.sets) - 1)
        if name == "sp_select_sequence":
            if self.procs is None:
                raise _Unsupported("sp_select_sequence needs the processing order")
            indptr, indices = _select_sequence(bd, self.procs, args)
            self.c.sets.append((indptr, indices))
            return self.add_call("ExplicitSet", index=len(self.c.sets) - 1)
        raise _Unsupported(f"{name} is not supported yet")


def _points_in_shapes(xy: np.ndarray, file: str, field_name, field_val) -> np.ndarray:
    """Points inside (or on) any feature of a vector file (optionally one attribute value)."""
    import geopandas as gpd
    import shapely

    path, layer = _dataset_layer(file)
    gdf = gpd.read_file(path, layer=layer)
    if field_name is not None:
        # SQLite: WHERE "field" = "value", with the column's numeric affinity
        col = gdf[field_name]
        if isinstance(field_val, (int, float)) and np.issubdtype(col.dtype, np.number):
            gdf = gdf[col.astype(np.float64) == float(field_val)]
        else:
            from ._names import perl_number

            text = perl_number(field_val) if isinstance(field_val, (int, float)) else str(field_val)
            gdf = gdf[col.astype(str) == text]
    elif field_val is not None:
        # a feature by its position, counted from 1
        fv = float(field_val)
        if fv != math.floor(fv):
            raise ValueError(f"field_val value must be an integer if field_name is not passed, got {field_val}")
        gdf = gdf.iloc[int(fv) - 1:int(fv)]
    pts = shapely.points(xy[:, 0], xy[:, 1])
    tree = shapely.STRtree(gdf.geometry.to_numpy())
    hit = tree.query(pts, predicate="intersects")
    m = np.zeros(len(xy), bool)
    m[hit[0]] = True
    return m


def _features_per_point(xy: np.ndarray, file: str) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the features (by identical geometry) each point falls in."""
    import geopandas as gpd
    import shapely

    path, layer = _dataset_layer(file)
    gdf = gpd.read_file(path, layer=layer)
    geoms = gdf.geometry.to_numpy()
    # identical geometries are one feature (Biodiverse hashes the WKT)
    wkts = shapely.to_wkt(geoms, rounding_precision=-1)
    _, fid = np.unique(wkts, return_inverse=True)
    pts = shapely.points(xy[:, 0], xy[:, 1])
    tree = shapely.STRtree(geoms)
    pi, gi = tree.query(pts, predicate="intersects")
    pairs = np.unique(np.stack([pi, fid[gi]], axis=1), axis=0) if len(pi) else np.zeros((0, 2), np.int64)
    indptr = np.zeros(len(xy) + 1, np.int64)
    np.add.at(indptr, pairs[:, 0] + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, pairs[:, 1].astype(np.int32)


def _dataset_layer(file: str) -> tuple[str, str | None]:
    m = re.match(r"^(.*\.(?:gpkg|gdb|sqlite)):(.+)$", file, flags=re.I)
    if m:
        return m.group(1), m.group(2)
    if not os.path.exists(file) and os.path.exists(file + ".shp"):
        return file + ".shp", None
    return file, None


def _select_sequence(bd, procs: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """sp_select_sequence: every `frequency`-th group, the offset cycling per processing group."""
    spacing = float(_const(args["frequency"]))
    first = _const(args["first_offset"]) if "first_offset" in args else 0
    first = float(first or 0)
    cycle = _const(args["cycle_offset"]) if "cycle_offset" in args else 1
    reverse = bool(_const(args["reverse_order"])) if "reverse_order" in args else False
    # get_element_list_sorted: by coordinates (text axes as strings), ties in name order
    order = sorted(
        range(bd.n_groups),
        key=lambda i: tuple(
            (bd.group_axes(bd.groups[i])[k] if bd.cell_sizes[k] < 0 else float(bd.coords[i, k])) for k in range(bd.n_axes)
        ),
    )
    if reverse:
        order = order[::-1]
    rows = []
    offset = first
    for k, _ in enumerate(procs):
        if k > 0:
            offset += 1
            if cycle and offset >= spacing:
                offset = 0
        sel = []
        last = -1
        i = offset
        while i <= len(order) - 1:
            ii = int(i)
            if ii != last:
                sel.append(order[ii])
                last = ii
            i += spacing
        rows.append(sorted(set(sel)))
    indptr = np.zeros(bd.n_groups + 1, np.int64)
    # rows are per processing group; store them by group index
    by_group = {int(g): r for g, r in zip(procs, rows)}
    data = []
    for g in range(bd.n_groups):
        r = by_group.get(g, [])
        data.extend(r)
        indptr[g + 1] = indptr[g] + len(r)
    return indptr, np.asarray(data, dtype=np.int32)


# ---------------------------------------------------------------------------------------
# public API


def _groups_payload(bd) -> dict:
    """Group table for the C++ evaluator."""
    n_axes = bd.n_axes
    strings: dict[str, int] = {}
    for g in bd.groups:
        strings.setdefault(g, len(strings))
    axis_str = np.full((bd.n_groups, n_axes), -1, dtype=np.int32)
    numeric = np.array([c >= 0 for c in bd.cell_sizes], dtype=np.uint8)
    for k in range(n_axes):
        if not numeric[k]:
            for i, g in enumerate(bd.groups):
                s = bd.group_axes(g)[k]
                axis_str[i, k] = strings.setdefault(s, len(strings))
    return {"strings": strings, "axis_str": axis_str, "numeric": numeric}


class Neighbourhoods:
    """Neighbour sets of the processing groups.

    Attributes
    ----------
    groups : numpy.ndarray
        Indices of the processing groups (those passing the definition
        query), in name order.
    sets : list of (indptr, indices)
        One CSR per condition: the neighbours of ``groups[r]`` are
        ``indices[indptr[r]:indptr[r + 1]]`` (group indices, ascending); the
        second set excludes the first.
    """

    def __init__(self, bd, groups: np.ndarray, sets: list[tuple[np.ndarray, np.ndarray]], conditions,
                 stored=None) -> None:
        self.bd = bd
        self.groups = groups
        self.sets = sets
        self.conditions = conditions
        # the lists as Biodiverse leaves them stored after its analysis
        # (recycling can overwrite lists of groups processed before)
        self._stored = stored if stored is not None else sets
        # Biodiverse copies results between groups with these neighbourhoods
        self.results_recyclable = False

    def __repr__(self) -> str:
        sizes = ", ".join(str(int(s[1].size)) for s in self.sets)
        return f"Neighbourhoods({len(self.groups)} groups, neighbour pairs per set: {sizes})"

    def of(self, group: str, set_index: int = 0, stored: bool = False) -> list[str]:
        """Neighbour names of a processing group (``stored``: as Biodiverse keeps them)."""
        r = int(np.searchsorted(self.groups, self.bd.group_index(group)))
        if r >= len(self.groups) or self.groups[r] != self.bd.group_index(group):
            raise KeyError(f"{group} is not a processing group")
        indptr, indices = (self._stored if stored else self.sets)[set_index]
        return [self.bd.groups[j] for j in indices[indptr[r]:indptr[r + 1]]]


def definition_query(bd, query: str | SpatialCondition, n_threads: int | None = None) -> np.ndarray:
    """Boolean mask of the groups passing a definition query.

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "c"], "x": [0.5, 1.5, 2.5], "y": [0.5, 0.5, 0.5]})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> ls.bio.definition_query(bd, "$x > 1")
    array([False,  True,  True])
    """
    cond = _as_condition(query)
    prog = cond.compile(bd, def_query=True)
    payload = _groups_payload(bd)
    for s in prog["strings"]:
        payload["strings"].setdefault(s, len(payload["strings"]))
    return np.asarray(
        _core.bio_definition_query(_group_arrays(bd, payload), _program_arrays(prog, payload), resolve(n_threads) > 1),
        dtype=bool,
    )


def _group_arrays(bd, payload) -> dict:
    strings = sorted(payload["strings"], key=payload["strings"].get)
    return {
        "coords": np.ascontiguousarray(bd.coords, dtype=np.float64),
        "numeric": payload["numeric"],
        "axis_str": np.ascontiguousarray(payload["axis_str"]),
        "name_str": np.arange(bd.n_groups, dtype=np.int32),
        "cell_sizes": np.asarray(bd.cell_sizes, dtype=np.float64),
        "strings": strings,
    }


def _program_arrays(prog: dict, payload) -> dict:
    # string ids of the program refer to its own table: remap to the group table
    remap = np.array([payload["strings"][s] for s in prog["strings"]], dtype=np.int32)
    nodes = np.array(prog["nodes"], dtype=np.float64).reshape(-1, 5) if prog["nodes"] else np.zeros((0, 5))
    ops = nodes[:, 0].astype(np.int32)
    abc = nodes[:, 1:4].astype(np.int32)
    str_const = ops == _OPS["StrConst"]
    abc[str_const, 0] = remap[abc[str_const, 0]]
    return {
        "ops": ops,
        "abc": np.ascontiguousarray(abc),
        "v": np.ascontiguousarray(nodes[:, 4]),
        "seq": np.asarray(prog["seq"], dtype=np.int32),
        "root": prog["root"],
        "n_locals": prog["n_locals"],
        "calls": prog["calls"],
        "masks": prog["masks"],
        "sets": prog["sets"],
        "rotations": prog["rotations"],
        "use_euc": prog["use_euc"],
        "use_cell": prog["use_cell"],
        "cell_lists": prog["cell_lists"],
        "iter_axes": prog["iter_axes"],
        "bounds": prog["bounds"],
        "self_only": prog["self_only"],
        "always_true": prog["always_true"],
        "always_false": prog["always_false"],
    }


def neighbours(
    bd,
    conditions: str | SpatialCondition | Sequence[str | SpatialCondition],
    definition_query: str | SpatialCondition | None = None,
    n_threads: int | None = None,
    *,
    recycle: bool = True,
    results_recyclable: bool | None = None,
) -> Neighbourhoods:
    """Neighbour sets of every group, as Biodiverse's spatial analysis finds them.

    Parameters
    ----------
    bd : BaseData
    conditions : condition or list of up to two conditions
        The second neighbour set leaves out the groups of the first.
    definition_query : condition, optional
        Only groups passing it are processed (all groups can be neighbours).
    n_threads : int, optional
        Threads (default: all cores but one).
    recycle : bool
        Reuse neighbour lists as Biodiverse does for non-overlapping, text
        match and "always the same" conditions (its default; it changes the
        result when such a condition is combined with a location-dependent
        term). False evaluates every group.
    results_recyclable : bool, optional
        Whether the analysis copies results between groups sharing their
        neighbours (the default: when every set is recyclable), which decides
        which groups trigger the recycling.

    Returns
    -------
    Neighbourhoods

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a"] * 4, "x": [0.5, 1.5, 2.5, 0.5], "y": [0.5, 0.5, 0.5, 1.5]})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> nb = ls.bio.neighbours(bd, ["sp_circle (radius => 1)", "sp_select_all()"])
    >>> nb.of("0.5:0.5"), nb.of("0.5:0.5", 1)
    (['0.5:0.5', '0.5:1.5', '1.5:0.5'], ['2.5:0.5'])
    """
    if isinstance(conditions, (str, SpatialCondition)):
        conditions = [conditions]
    conds = [_as_condition(c) for c in conditions]
    if not 1 <= len(conds) <= 2:
        raise ValueError("one or two spatial conditions are supported")
    threads = resolve(n_threads)
    n = bd.n_groups
    if definition_query is not None:
        passed = globals()["definition_query"](bd, definition_query, n_threads=threads)
        procs = np.flatnonzero(passed).astype(np.int32)
        if procs.size == 0:
            raise ValueError("Nothing passed the definition query")
    else:
        passed = np.ones(n, bool)
        procs = np.arange(n, dtype=np.int32)
    types = [c.result_type for c in conds]
    shortcut = {"self_only", "always_true", "always_false"}
    payload = _groups_payload(bd)
    own: list[tuple[np.ndarray, np.ndarray] | None] = []
    progs = []
    for cond, t in zip(conds, types):
        if t in shortcut:
            own.append(None)
            progs.append(None)
            continue
        prog = cond.compile(bd, procs=procs)
        for st in prog["strings"]:
            payload["strings"].setdefault(st, len(payload["strings"]))
        progs.append(prog)
    garr = _group_arrays(bd, payload)
    for i, prog in enumerate(progs):
        if prog is None:
            continue
        rows = procs
        if types[i] in ("always_same", "always_same_current_label"):
            rows = procs[:1]  # only the first processing group is ever searched
        indptr, indices = _core.bio_find_neighbours(garr, _program_arrays(prog, payload), rows, None, threads > 1)
        own.append(None)
        own[i] = (np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int32))
    recyclable = _recyclable_sets(types) if recycle else [False] * len(conds)
    if results_recyclable is None:
        results_recyclable = all(recyclable)
    results_recyclable = bool(results_recyclable and all(recyclable))
    row_of = np.full(n, -1, np.int64)
    row_of[procs] = np.arange(len(procs))
    all_groups = np.arange(n, dtype=np.int32)

    def own_list(i: int, g: int) -> np.ndarray:
        t = types[i]
        if t == "self_only":
            return np.array([g], dtype=np.int32)
        if t == "always_true":
            return all_groups
        if t == "always_false":
            return np.zeros(0, np.int32)
        indptr, indices = own[i]
        if t in ("always_same", "always_same_current_label"):
            return indices[indptr[0]:indptr[1]]
        r = row_of[g]
        return indices[indptr[r]:indptr[r + 1]]

    # Recycling and the "always same" cache only change anything when the
    # condition mixes such a function with location-dependent terms; otherwise
    # every group's own list (minus the earlier sets) is what Biodiverse uses.
    simple = all(_is_pure(c) for c in conds) or (
        not any(recyclable) and all(t not in ("always_same", "always_same_current_label") for t in types)
    )
    if simple:
        sets = []
        for i in range(len(conds)):
            if own[i] is not None and types[i] not in ("always_same", "always_same_current_label"):
                cur = own[i]
            else:
                cur = _rows_to_csr([own_list(i, int(g)) for g in procs])
            for prev in sets:
                cur = _csr_minus(cur, prev, n)
            sets.append(cur)
        out = Neighbourhoods(bd, procs, sets, conds)
        out.results_recyclable = results_recyclable
        return out

    n_sets = len(conds)
    recycled: list[dict[int, np.ndarray]] = [dict() for _ in range(n_sets)]
    stored: list[dict[int, np.ndarray]] = [dict() for _ in range(n_sets)]
    used: dict[int, list[np.ndarray]] = {}
    has_lists = np.zeros(n, bool)
    results_same = np.full(n, -1, np.int64)
    for g in procs:
        g = int(g)
        if results_recyclable and results_same[g] >= 0:
            src = int(results_same[g])
            used[g] = used[src]
            continue
        lists = []
        excl = np.zeros(0, np.int32)
        for i in range(n_sets):
            if g in recycled[i]:
                lst = recycled[i][g]
            else:
                full = own_list(i, g)
                lst = np.setdiff1d(full, excl, assume_unique=True) if excl.size else full
            lists.append(lst)
            stored[i][g] = lst
            if i != n_sets - 1:
                excl = np.union1d(excl, lst)
        used[g] = lists
        has_lists[g] = True
        nbrs1 = lists[0]
        if recyclable[0]:
            for f in nbrs1:
                if f == g or has_lists[f]:
                    continue
                targets = [int(x) for x in nbrs1 if x != g and passed[x]]
                for i in range(n_sets):
                    if not recyclable[i]:
                        break
                    for t in targets:
                        recycled[i][t] = lists[i]
                        stored[i][t] = lists[i]
                has_lists[targets] = True
                break
        if results_recyclable:
            for x in nbrs1:
                if passed[x] and results_same[x] < 0:
                    results_same[x] = g
    sets = [_rows_to_csr([used[int(g)][i] for g in procs]) for i in range(n_sets)]
    stored_sets = [_rows_to_csr([stored[i].get(int(g), np.zeros(0, np.int32)) for g in procs]) for i in range(n_sets)]
    out = Neighbourhoods(bd, procs, sets, conds, stored=stored_sets)
    out.results_recyclable = results_recyclable
    return out


_PURE_CALLS = {
    "sp_block", "sp_select_all", "sp_self_only", "sp_match_text", "sp_match_regex", "sp_select_element",
    "sp_richness_greater_than", "sp_redundancy_greater_than", "sp_group_not_empty", "sp_point_in_poly",
    "sp_point_in_poly_shape", "sp_circle", "sp_circle_cell", "sp_rectangle",
    "sp_annulus", "sp_square", "sp_square_cell", "sp_ellipse", "sp_is_left_of", "sp_is_right_of", "sp_in_line_with",
}


def _is_pure(cond: SpatialCondition) -> bool:
    """A single sp_* call whose recycled/cached lists equal every group's own list."""
    ast = cond.ast
    if ast[0] != "call" or ast[1] not in _PURE_CALLS:
        return False
    name = ast[1]
    try:
        args = _call_args(ast[2])
    except Exception:
        return False
    if name in ("sp_match_text", "sp_match_regex", "sp_select_element"):
        # neighbour-only tests (a processing-group test is recycled wrongly)
        if "type" in args and _is_const(args["type"]) and _const(args["type"]) == "proc":
            return False
        if name == "sp_match_text" and not _is_const(args.get("text")):
            return False
        return True
    if name in ("sp_point_in_poly", "sp_point_in_poly_shape"):
        pa = args.get("point")
        return pa is None or pa == ("aref", "nbrcoord")
    if name in ("sp_richness_greater_than", "sp_redundancy_greater_than", "sp_group_not_empty"):
        return True
    return cond.result_type not in ("always_same",)


def _csr_minus(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise set difference a - b of two CSR neighbour tables."""
    ai, aj = a
    bi, bj = b
    ra = np.repeat(np.arange(len(ai) - 1, dtype=np.int64), np.diff(ai))
    rb = np.repeat(np.arange(len(bi) - 1, dtype=np.int64), np.diff(bi))
    keep = ~np.isin(ra * n + aj, rb * n + bj)
    counts = np.bincount(ra[keep], minlength=len(ai) - 1)
    indptr = np.zeros(len(ai), np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, aj[keep].astype(np.int32)


def _rows_to_csr(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(rows) + 1, np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    indices = np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, np.int32)
    return indptr, indices

