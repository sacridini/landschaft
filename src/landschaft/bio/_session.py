# SPDX-License-Identifier: GPL-3.0-or-later
"""landschaft's native format for biodiversity data: zips of Parquet tables.

A basedata is saved as its sample counts (group, label, count), groups (names
and coordinates), labels, properties and a JSON header; a session bundles
basedatas, trees (Newick), matrices and analysis results under names.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

FORMAT = "landschaft-bio"
VERSION = 1


def _write_parquet(zf: zipfile.ZipFile, name: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    zf.writestr(name, buf.getvalue())


def _read_parquet(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(zf.read(name)))


def _crs_text(crs) -> str | None:
    if crs is None:
        return None
    to_wkt = getattr(crs, "to_wkt", None)
    return to_wkt() if callable(to_wkt) else str(crs)


def _write_basedata(zf: zipfile.ZipFile, prefix: str, bd) -> None:
    c = bd.counts.tocoo()
    _write_parquet(zf, prefix + "counts.parquet", pd.DataFrame({
        "group": c.row.astype(np.int32), "label": c.col.astype(np.int32), "count": c.data.astype(np.float64)}))
    groups = pd.DataFrame({"name": bd.groups})
    for k in range(bd.coords.shape[1]):
        groups[f"axis_{k}"] = bd.coords[:, k]
    _write_parquet(zf, prefix + "groups.parquet", groups)
    _write_parquet(zf, prefix + "labels.parquet", pd.DataFrame({"name": bd.labels}))
    for kind in ("label_properties", "group_properties"):
        props = getattr(bd, kind)
        if len(props.columns):
            _write_parquet(zf, prefix + kind + ".parquet", props.astype(object).where(props.notna(), None))
    header = {
        "format": FORMAT,
        "version": VERSION,
        "cell_sizes": list(bd.cell_sizes),
        "cell_origins": list(bd.cell_origins),
        "crs": _crs_text(bd.crs),
        "grid": getattr(bd, "grid", None),
        "group_axes": getattr(bd, "_group_axes", None),
    }
    zf.writestr(prefix + "basedata.json", json.dumps(header))


def _read_basedata(zf: zipfile.ZipFile, prefix: str):
    import scipy.sparse as sp

    from ._basedata import BaseData

    header = json.loads(zf.read(prefix + "basedata.json"))
    if header.get("format") != FORMAT:
        raise ValueError("not a landschaft basedata")
    groups = _read_parquet(zf, prefix + "groups.parquet")
    labels = _read_parquet(zf, prefix + "labels.parquet")["name"].tolist()
    counts = _read_parquet(zf, prefix + "counts.parquet")
    names = groups["name"].tolist()
    coords = groups[[c for c in groups.columns if c.startswith("axis_")]].to_numpy(dtype=np.float64)
    m = sp.csr_matrix((counts["count"].to_numpy(), (counts["group"].to_numpy(), counts["label"].to_numpy())),
                      shape=(len(names), len(labels)))
    bd = BaseData(m, names, labels, coords, header["cell_sizes"], header["cell_origins"], header.get("crs"),
                  group_axes=header.get("group_axes"))
    bd.grid = header.get("grid")
    for kind, index_name in (("label_properties", "label"), ("group_properties", "group")):
        name = prefix + kind + ".parquet"
        if name in zf.namelist():
            props = _read_parquet(zf, name).astype(object)
            props = props.where(props.notna(), None)
            props.index.name = index_name
            setattr(bd, kind, props)
    return bd


def save_basedata(bd, path: str | os.PathLike) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_basedata(zf, "", bd)


def load_basedata(path: str | os.PathLike):
    with zipfile.ZipFile(path) as zf:
        return _read_basedata(zf, "")


def save_session(path: str | os.PathLike, **objects) -> None:
    """Save basedatas, trees, matrices and analysis results in one file.

    Parameters
    ----------
    path : path
        The file to write (a zip of Parquet tables and JSON).
    **objects
        Named :class:`BaseData`, :class:`Tree`, :class:`Matrix`,
        :class:`SpatialResult` or :class:`ClusterResult` objects. Results
        keep their tables (and their basedata, saved with them).

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "a"], "x": [0.5, 1.5, 1.5], "y": [0.5] * 3})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> res = ls.bio.spatial_analysis(bd, "sp_self_only()", ["richness"])
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "study.lsbio"
    >>> ls.bio.save_session(path, sites=bd, richness=res)
    >>> s = ls.bio.load_session(path)
    >>> s["richness"].table["RICHNESS_ALL"].tolist()
    [1.0, 2.0]
    """
    from ._basedata import BaseData
    from ._cluster import ClusterResult
    from ._matrix import Matrix
    from ._spatial import SpatialResult
    from ._tree import Tree

    index = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, obj in objects.items():
            prefix = f"{name}/"
            if isinstance(obj, BaseData):
                _write_basedata(zf, prefix, obj)
                index[name] = {"type": "basedata"}
            elif isinstance(obj, Tree):
                zf.writestr(prefix + "tree.nwk", obj.to_newick())
                index[name] = {"type": "tree", "name": obj.name}
            elif isinstance(obj, Matrix):
                _write_parquet(zf, prefix + "matrix.parquet", obj.to_dataframe())
                index[name] = {"type": "matrix", "self_similarity": obj.self_similarity}
            elif isinstance(obj, SpatialResult):
                _write_basedata(zf, prefix + "basedata/", obj.bd)
                _write_parquet(zf, prefix + "table.parquet", obj.table)
                for lname in obj.lists:
                    _write_parquet(zf, prefix + f"lists/{lname}.parquet", _plain(obj.lists[lname]))
                index[name] = {"type": "spatial", "calculations": list(obj.calculations),
                               "lists": list(obj.lists)}
            elif isinstance(obj, ClusterResult):
                zf.writestr(prefix + "tree.nwk", obj.tree.to_newick())
                _write_parquet(zf, prefix + "nodes.parquet", _plain(obj.nodes))
                _write_parquet(zf, prefix + "table.parquet", obj.table)
                for lname in obj.lists:
                    _write_parquet(zf, prefix + f"lists/{lname}.parquet", _plain(obj.lists[lname]))
                index[name] = {"type": "cluster", "lists": list(obj.lists)}
            else:
                raise TypeError(f"cannot save {name!r} of type {type(obj).__name__}")
        zf.writestr("session.json", json.dumps({"format": FORMAT, "version": VERSION, "objects": index}))


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns as strings or lists Parquet can store."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            if out[c].map(lambda v: isinstance(v, (list, tuple))).any():
                out[c] = out[c].map(lambda v: json.dumps(list(v)) if isinstance(v, (list, tuple)) else v)
            else:
                out[c] = out[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    return out


def load_session(path: str | os.PathLike) -> dict:
    """Load the objects saved with :func:`save_session` (a dict by name).

    Examples
    --------
    See :func:`save_session`.
    """
    from ._cluster import ClusterResult
    from ._matrix import Matrix
    from ._spatial import SpatialResult
    from ._tree import read_tree

    out = {}
    with zipfile.ZipFile(path) as zf:
        header = json.loads(zf.read("session.json"))
        if header.get("format") != FORMAT:
            raise ValueError("not a landschaft session")
        for name, info in header["objects"].items():
            prefix = f"{name}/"
            kind = info["type"]
            if kind == "basedata":
                out[name] = _read_basedata(zf, prefix)
            elif kind == "tree":
                out[name] = read_tree(zf.read(prefix + "tree.nwk").decode("utf-8"), format="newick")
            elif kind == "matrix":
                df = _read_parquet(zf, prefix + "matrix.parquet")
                out[name] = Matrix(list(df.columns), df.to_numpy(dtype=np.float64), info.get("self_similarity"))
            elif kind == "spatial":
                bd = _read_basedata(zf, prefix + "basedata/")
                table = _read_parquet(zf, prefix + "table.parquet")
                lists = {ln: _read_parquet(zf, prefix + f"lists/{ln}.parquet") for ln in info["lists"]}
                out[name] = SpatialResult(bd, table, lists, None, info["calculations"])
            elif kind == "cluster":
                tree = read_tree(zf.read(prefix + "tree.nwk").decode("utf-8"), format="newick")
                nodes = _read_parquet(zf, prefix + "nodes.parquet")
                if "children" in nodes.columns:
                    nodes["children"] = nodes["children"].map(lambda v: json.loads(v) if isinstance(v, str) else v)
                table = _read_parquet(zf, prefix + "table.parquet")
                lists = {ln: _read_parquet(zf, prefix + f"lists/{ln}.parquet") for ln in info["lists"]}
                out[name] = ClusterResult(tree, nodes, table, lists)
    return out
