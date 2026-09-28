# SPDX-License-Identifier: GPL-3.0-or-later
"""Phylogenetic trees (Newick, Nexus, Phylip) read as Biodiverse reads them.

Node names follow Biodiverse's reader: unnamed nodes get ``"0___"``,
``"1___"``... in the order their definition closes (children before parents),
duplicated names get ``__dup1``, ``__dup2``..., quoted names are unquoted, the
Nexus ``Translate`` table is applied, trees whose branches all have length zero
get length 1 everywhere, and a forest gets a new root of length 0.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ._names import QUOTE_CHAR, dequote_element, join_element, split_element

__all__ = ["Tree", "read_tree", "read_trees"]

_RE_NUMBER = re.compile(r"[+-]?(?=\.?\d)\d*(?:\.\d*)?(?:[Ee][+-]?\d+)?")
_RE_QUOTED = re.compile(r"'(?:[^']|'')*'")
_RE_NAME = re.compile(r"[^(),:'\[\]]*")
_RE_WS = re.compile(r"[\s\n\r]+")
_RE_TEXT_IN_QUOTES = re.compile(r"\A(['\"])(.+)\1\Z", re.S)


class Tree:
    """A rooted phylogenetic tree.

    Nodes are numbered in the order the reader creates them. ``names``,
    ``parent`` (-1 for the root), ``length`` and ``children`` describe the
    topology; ``node_order`` lists the nodes sorted by name, the order in which
    landschaft sums over nodes (Biodiverse's own order with sorted keys).

    Parameters
    ----------
    names, parent, length, children
        Node names, parent index per node (-1 for the root), branch lengths and
        child indices per node (in the order they were given).
    name : str, optional
        Tree name.
    bootstrap : list, optional
        The text in square brackets after each node, if any.
    """

    def __init__(
        self,
        names: Sequence[str],
        parent: Sequence[int],
        length: Sequence[float],
        children: Sequence[Sequence[int]],
        name: str = "",
        bootstrap: Sequence[str | None] | None = None,
    ) -> None:
        self.name = name
        self.names = list(names)
        self.parent = np.asarray(parent, dtype=np.int64)
        self.length = np.asarray(length, dtype=np.float64)
        self.children = [list(c) for c in children]
        self.bootstrap = list(bootstrap) if bootstrap is not None else [None] * len(self.names)
        n = len(self.names)
        if not (len(self.parent) == len(self.length) == len(self.children) == n):
            raise ValueError("names, parent, length and children must have the same length")
        self._index = {nm: i for i, nm in enumerate(self.names)}
        if len(self._index) != n:
            raise ValueError("node names must be unique")
        roots = np.flatnonzero(self.parent < 0)
        if len(roots) != 1:
            raise ValueError(f"a tree needs exactly one root, got {len(roots)}")
        self.root = int(roots[0])

    # -- basic properties -------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.names)

    @property
    def is_terminal(self) -> np.ndarray:
        return np.array([len(c) == 0 for c in self.children], dtype=bool)

    @property
    def terminal_names(self) -> list[str]:
        return [self.names[i] for i in range(self.n_nodes) if not self.children[i]]

    @property
    def node_order(self) -> np.ndarray:
        """Node indices sorted by name."""
        return np.array(sorted(range(self.n_nodes), key=self.names.__getitem__), dtype=np.int64)

    def index(self, name: str) -> int:
        """Node index of ``name``."""
        return self._index[name]

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def __len__(self) -> int:
        return self.n_nodes

    def __repr__(self) -> str:
        return f"Tree({self.name!r}, {self.n_nodes} nodes, {int(self.is_terminal.sum())} terminals)"

    @property
    def total_length(self) -> float:
        """Sum of all branch lengths (root included), summed in name order."""
        total = 0.0
        for i in self.node_order:
            total += float(self.length[i])
        return total

    def postorder(self) -> np.ndarray:
        """Nodes with every child before its parent (children in their given order)."""
        out: list[int] = []
        stack = [(self.root, 0)]
        while stack:
            node, k = stack.pop()
            ch = self.children[node]
            if k < len(ch):
                stack.append((node, k + 1))
                stack.append((ch[k], 0))
            else:
                out.append(node)
        return np.array(out, dtype=np.int64)

    def depth(self) -> np.ndarray:
        """Number of edges from the root to each node."""
        d = np.zeros(self.n_nodes, dtype=np.int64)
        for node in self.postorder()[::-1]:
            p = self.parent[node]
            if p >= 0:
                d[node] = d[p] + 1
        return d

    def to_newick(self, lengths: bool = True) -> str:
        """Newick string (names quoted when needed)."""

        def fmt(name: str) -> str:
            if re.search(r"[\s(),:;'\[\]]", name):
                return "'" + name.replace("'", "''") + "'"
            return name

        def rec(i: int) -> str:
            s = ""
            if self.children[i]:
                s = "(" + ",".join(rec(c) for c in self.children[i]) + ")"
            s += fmt(self.names[i])
            if lengths:
                s += ":" + repr(float(self.length[i]))
            return s

        return rec(self.root) + ";"


class _Builder:
    """Mutable tree used while parsing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.parent: list[int] = []
        self.length: list[float] = []
        self.children: list[list[int]] = []
        self.boot: list[str | None] = []
        self.highest_internal = -1

    def free_internal_name(self, exclude: dict[str, str]) -> str:
        values = set(exclude.values())
        while True:
            self.highest_internal += 1
            name = f"{self.highest_internal}___"
            if name not in self.index and name not in exclude and name not in values:
                return name

    def unique_name(self, prefix: str, exclude: dict[str, str]) -> str:
        i = 1
        while True:
            name = f"{prefix}__dup{i}"
            if name not in self.index and name not in exclude:
                return name
            i += 1

    def add(self, name: str, length: float | None, boot: str | None, exclude: dict[str, str]) -> int:
        if name in self.index:
            name = self.unique_name(name, exclude)
        i = len(self.names)
        self.names.append(name)
        self.index[name] = i
        self.parent.append(-1)
        self.length.append(0.0 if length is None else length)
        self.children.append([])
        self.boot.append(boot)
        return i

    def add_children(self, node: int, kids: list[int]) -> None:
        seen = set(self.children[node])
        for k in dict.fromkeys(kids):
            if k in seen:
                continue
            old = self.parent[k]
            if old >= 0 and old != node:
                self.children[old].remove(k)
            self.parent[k] = node
            self.children[node].append(k)

    def finish(self) -> Tree:
        # all-zero trees get unit lengths
        if not any(self.length):
            self.length = [1.0] * len(self.length)
        roots = [i for i, p in enumerate(self.parent) if p < 0]
        if len(roots) > 1:
            name = self.free_internal_name({})
            r = self.add(name, 0.0, None, {})
            self.add_children(r, roots)
        return Tree(self.names, self.parent, self.length, self.children, name=self.name, bootstrap=self.boot)


def _clean_name(name: str, translate: dict[str, str]) -> str:
    if name in translate:
        name = translate[name]
    if _RE_TEXT_IN_QUOTES.match(name):
        parts = split_element(name, sep=":", quote=QUOTE_CHAR)
        name = parts[0] if len(parts) == 1 else join_element(parts, sep=":", quote=QUOTE_CHAR)
        name = dequote_element(name, QUOTE_CHAR)
    return name


def _matching_paren(s: str, i: int) -> int:
    """Index of the ``)`` matching the ``(`` at ``i`` (quotes are not special)."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError(f"Tree has unbalanced parentheses (left is {s.count('(')}, right is {s.count(')')}), unable to parse")


def _parse_newick(s: str, tree: _Builder, translate: dict[str, str]) -> list[int]:
    """Port of Biodiverse's ``parse_newick`` (one level; recursion on brackets)."""
    added: list[int] = []
    name: str | None = None
    length: float | None = 0.0  # Biodiverse starts each level with length 0
    boot: str | None = None
    kids: list[int] = []
    pos = 0
    n = len(s)

    def make_node() -> int:
        nm = name if name is not None else tree.free_internal_name(translate)
        nm = _clean_name(nm, translate)
        node = tree.add(nm, length, boot, translate)
        if kids:
            tree.add_children(node, kids)
        return node

    while pos < n:
        c = s[pos]
        if c == ",":
            pos += 1
            added.append(make_node())
            name = None
            length = None
            boot = None
            kids = []
        elif c == "(":
            end = _matching_paren(s, pos)
            kids = _parse_newick(s[pos + 1:end], tree, translate)
            pos = end + 1
        elif c == "[":
            end = s.find("]", pos + 1)
            if end < 0:
                pos = n
                continue
            boot = s[pos + 1:end]
            pos = end + 1
        elif c == "'":
            m = _RE_QUOTED.match(s, pos)
            if m is None:
                raise ValueError(f"unterminated quote at position {pos}")
            name = m.group(0)
            pos = m.end()
        elif c == ":":
            m = _RE_NUMBER.match(s, pos + 1)
            if m is None or not m.group(0):
                raise ValueError(f"Length '{s[pos + 1:pos + 20]}' does not look like a number")
            length = float(m.group(0))
            pos = m.end()
        elif _RE_WS.match(s, pos):
            pos = _RE_WS.match(s, pos).end()
        else:
            m = _RE_NAME.match(s, pos)
            if m.end() == pos:
                raise ValueError(f"Unexpected character '{c}' found at position {pos + 1}")
            name = m.group(0)
            pos = m.end()
    added.append(make_node())
    return added


def _nexus_trees(text: str) -> list[tuple[str, str, dict[str, str]]]:
    lines = [ln for ln in re.split(r"[\r\n]+", text)]
    translate: dict[str, str] = {}
    newicks: list[str] = []
    in_block = False
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not in_block and re.search(r"\s*begin trees;", line, flags=re.I):
            in_block = True
            if i >= len(lines):
                break
            line = lines[i]
            i += 1
        if not in_block:
            continue
        if re.search(r"\s*end(?:block)?;", line, flags=re.I):
            break
        if re.match(r"^\s*\[", line):
            if "]" in line:
                continue
            while i < len(lines):
                c = lines[i]
                i += 1
                if "]" in c:
                    break
            continue
        if re.search(r"\s*Translate", line, flags=re.I):
            while i < len(lines):
                t = lines[i]
                i += 1
                if not t:
                    break
                m = re.search(r"\s*(\S+)\s+('(?:[^']|'')*'|\S+)", t)
                if m:
                    code, nm = m.group(1), m.group(2)
                    nm = re.sub(r"[,;]\s*\Z", "", nm)
                    translate[code] = nm
                if re.search(r";\s*\Z", t):
                    break
            continue
        if re.match(r"^\s*tree", line, flags=re.I):
            nwk = line
            if not re.search(r";\s*$", line):
                while i < len(lines):
                    t = lines[i]
                    i += 1
                    if not t:
                        break
                    nwk += t
                    if re.search(r";\s*$", t):
                        break
            newicks.append(nwk)
    if not newicks:
        raise ValueError("Data has no trees in it, or is not Nexus format")
    out = []
    for nwk in newicks:
        nwk = re.sub(r";$", "", nwk)
        m = re.search(r"\s*tree\s+(.+?)\s*=\s*(\[..\])?\s*(.*)", nwk, flags=re.I | re.S)
        name, rest = ("", "")
        if m:
            name, rest = m.group(1), m.group(3)
        out.append((name.rstrip(), rest, translate))
    return out


def _read_text(source: str | os.PathLike) -> tuple[str, bool]:
    if isinstance(source, os.PathLike) or (isinstance(source, str) and "(" not in source and Path(source).exists()):
        raw = Path(source).read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        return raw.decode("utf-8"), True
    return str(source), False


def read_trees(source: str | os.PathLike, format: str | None = None) -> list[Tree]:
    """Read every tree in a Newick, Phylip (several Newicks) or Nexus file.

    Parameters
    ----------
    source : str or path
        File path, or the text itself.
    format : {"nexus", "newick"}, optional
        Detected from the text when omitted (``#NEXUS`` header).

    Returns
    -------
    list of Tree

    Examples
    --------
    >>> import landschaft as ls
    >>> [t] = ls.bio.read_trees("((a:1,b:2):1,c:3);")
    >>> t.names
    ['a', 'b', '0___', 'c', '1___']
    >>> t.total_length
    7.0
    """
    text, _ = _read_text(source)
    if format is None:
        head = "\n".join(ln for ln in text.splitlines()[:40] if ln.strip())
        format = "nexus" if head.lstrip().upper().startswith("#NEXUS") else "newick"
    trees = []
    if format == "nexus":
        for name, rest, translate in _nexus_trees(text):
            b = _Builder(name)
            _parse_newick(rest, b, translate)
            trees.append(b.finish())
    elif format in ("newick", "phylip"):
        parts = [p for p in text.split(";") if p.strip()]
        for i, nwk in enumerate(parts):
            b = _Builder(f"anonymous_{i}")
            _parse_newick(nwk, b, {})
            trees.append(b.finish())
    else:
        raise ValueError(f"unknown tree format {format!r}")
    return trees


def read_tree(source: str | os.PathLike, format: str | None = None, index: int | str = 0) -> Tree:
    """Read one tree (the first, or the one with index/name ``index``).

    Examples
    --------
    >>> import landschaft as ls
    >>> t = ls.bio.read_tree("((a:1,b:2)ab:1,c:3)root;")
    >>> t.terminal_names
    ['a', 'b', 'c']
    """
    trees = read_trees(source, format=format)
    if isinstance(index, str):
        for t in trees:
            if t.name == index:
                return t
        raise KeyError(f"no tree named {index!r}")
    return trees[index]


def _iter_names(names: Iterable[str]) -> list[str]:
    return list(names)
