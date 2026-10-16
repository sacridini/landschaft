# SPDX-License-Identifier: GPL-3.0-or-later
"""Many independent tasks on this computer and on other computers of the local network.

A computer lends itself to the group with ``landschaft worker`` (the first time
``landschaft worker --join <code>``, with the code that ``landschaft helpers invite``
shows on the computer that will use it). Functions that run many independent tasks take
``on=ls.helpers()`` and run them here and on the helpers, with the same results as on
this computer alone::

    import landschaft as ls

    with ls.helpers() as pool:
        runs = ls.batch("fad", "tiles/", "fad_out/", on=pool)
        ens = ls.simulate.scenarios(model, start, drivers, replicates=200, seed=1, on=pool)

Helpers only run landschaft's own operations, never code sent to them: a task is the
name of an operation, its parameters and its input files. They connect out to the
computer that coordinates (only that one needs a firewall rule), authenticated by the
group's secret, over TLS.
"""

from __future__ import annotations

from ._dist import _pack, _registry
from ._dist import (
    Helpers,
    JobResult,
    helpers,
    invite_helpers,
    leave_group,
    list_helpers,
    open_firewall,
    run_worker,
)

__all__ = ["TOOLS", "Call", "Helpers", "JobResult", "call", "helpers", "invite_helpers", "leave_group", "list_helpers",
           "map", "open_firewall", "param", "run_worker", "tiled"]


# --------------------------------------------------------------------------- tools by name

#: The tools ``map``, ``monte_carlo`` and ``tiled`` can run on helpers, by name: the
#: analyses of ``ls`` (``"fad"``, ``"mspa"``...), ``ls.surface`` (``"surface.tpi"``),
#: ``ls.hydro``, ``ls.services``, ``ls.nlm`` and ``ls.circuit``.
TOOLS = tuple(sorted(_registry.TOOLS))


class Call:
    """A registered tool with its arguments, some of them drawn parameters
    (:func:`param`): what :func:`landschaft.uncertainty.monte_carlo` computes for each
    draw, here or on helpers. Calling it with the parameters runs the tool."""

    def __init__(self, tool, args, kwargs):
        self.tool = _registry.name_of(tool)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def bind(self, params):
        """The arguments with the placeholders replaced by ``params``."""
        def sub(v):
            if isinstance(v, _pack.Param):
                return float(params[v.index])
            if isinstance(v, tuple):
                return tuple(sub(x) for x in v)
            if isinstance(v, list):
                return [sub(x) for x in v]
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            return v

        return [sub(a) for a in self.args], {k: sub(v) for k, v in self.kwargs.items()}

    def __call__(self, params):
        args, kwargs = self.bind(params)
        return _registry.resolve(self.tool)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Call({self.tool!r})"


def call(tool, *args, **kwargs) -> Call:
    """A registered tool with its arguments, the drawn ones marked by :func:`param`.

    Parameters
    ----------
    tool
        A name of :data:`TOOLS` (``"surface.gravity"``) or the function itself.
    *args, **kwargs
        Its arguments; ``param(i)`` stands for the i-th parameter of each draw.

    Returns
    -------
    Call

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> src = np.zeros((20, 20)); src[10, 10] = 1
    >>> f = ls.parallel.call("surface.gravity", src, exponent=ls.parallel.param(0))
    >>> mc = ls.uncertainty.monte_carlo(f, [2.0], [0.1], n=50, seed=1)      # on=ls.helpers() too
    >>> mc.mean.shape
    (20, 20)
    """
    return Call(tool, args, kwargs)


def param(index: int):
    """The ``index``-th drawn parameter, in the arguments of a :func:`call`."""
    return _pack.Param(index)


def _as_call(item) -> tuple:
    if isinstance(item, dict):
        return [], dict(item)
    if isinstance(item, tuple):
        return list(item), {}
    return [item], {}


def map(tool, items, *, on=None, output_dir=None, threads_per_task: int | None = None, n_workers: int | None = None,
        **common) -> list:
    """Run a tool on many inputs, on this computer or on helpers, in parallel.

    Parameters
    ----------
    tool
        A name of :data:`TOOLS` (``"fad"``, ``"surface.geomorphons"``, ``"nlm.random"``)
        or the function itself.
    items
        One call per item: a dict of keyword arguments, a tuple of positional ones,
        or the first argument alone (a raster).
    on
        ``ls.helpers()``: the calls run here and on the helpers of this computer's
        group (their arguments and results travel as data, never as code). Default:
        threads of this computer.
    output_dir
        Save each result there (``ls.save``: ``<output_dir>/<i>``) and return the paths
        instead of the results.
    threads_per_task
        Threads of each call (default: each computer's cores divided among the calls).
    n_workers
        Calls at once on this computer, without ``on``.
    **common
        Keyword arguments of every call.

    Returns
    -------
    list
        The results (or the paths of the saved results), in the order of ``items``.

    Examples
    --------
    >>> import landschaft as ls
    >>> maps = ls.parallel.map("nlm.random", [dict(nrow=50, ncol=50, seed=s) for s in range(4)])
    >>> [m.shape for m in maps]
    [(50, 50), (50, 50), (50, 50), (50, 50)]
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from . import _threads
    from ._dist import InputFile, Job, Task, run_job
    from ._save import save

    name = _registry.name_of(tool)
    fn = _registry.resolve(name)
    calls = []
    for item in items:
        args, kwargs = _as_call(item)
        calls.append((args, {**common, **kwargs}))
    width = len(str(max(len(calls) - 1, 0)))
    if on is None:
        total = _threads.resolve(None)
        workers = max(1, min(len(calls), n_workers or total)) if calls else 1
        inner = threads_per_task or max(1, total // workers)

        def one(c):
            args, kwargs = c
            import inspect

            if "n_threads" in inspect.signature(fn).parameters and "n_threads" not in kwargs:
                kwargs = {**kwargs, "n_threads": inner}
            return fn(*args, **kwargs)

        if workers == 1:
            results = [one(c) for c in calls]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(one, calls))
    else:
        results = [None] * len(calls)
        with tempfile.TemporaryDirectory(prefix="landschaft-map-") as tmp:
            tmp = Path(tmp)
            tasks = []
            for i, (args, kwargs) in enumerate(calls):
                _registry.check_strings(args, "args")
                _registry.check_strings(kwargs, "kwargs")
                stored = _pack.write((args, kwargs), tmp / f"a{i}")
                files = [InputFile.of(p, f"a{i}/{p.name}") for p in _pack.files(stored)]
                tasks.append(Task(f"{i}", "call", {"tool": name, "args": f"a{i}"}, files, root=tmp))

            def collect(task, folder, stats):
                results[int(task.name)] = _pack.read(Path(folder) / "result")

            report = run_job(on, Job(name, tasks, collect, threads=threads_per_task))
        if report.failed:
            first, error = next(iter(report.failed.items()))
            raise RuntimeError(f"{len(report.failed)} call(s) of {name} failed; call {first}: {error}")
    if output_dir is None:
        return results
    out = Path(output_dir)
    return [save(r, out / f"{i:0{width}d}") for i, r in enumerate(results)]


# --------------------------------------------------------------------------- tiles

def _crop(result, rows: slice, cols: slice, shape: tuple):
    """The part of a tile's result over its core: every map of the tile's shape."""
    import dataclasses

    import xarray as xr

    def cut(v):
        if isinstance(v, xr.DataArray) and tuple(v.shape[-2:]) == tuple(shape):
            return v[..., rows, cols]
        if isinstance(v, xr.Dataset):
            return xr.Dataset({k: cut(v[k]) for k in v.data_vars}, attrs=dict(v.attrs))
        return None

    if isinstance(result, (xr.DataArray, xr.Dataset)):
        out = cut(result)
        if out is None:
            raise ValueError("the tool's result is not a map of the tile")
        return out
    if isinstance(result, dict) and result and all(isinstance(v, xr.DataArray) for v in result.values()):
        maps = {str(k): cut(v) for k, v in result.items()}
        if any(v is None for v in maps.values()):
            raise ValueError("the tool's result is not a map of the tile")
        return {"$class": "", "maps": maps, "complete": False}
    if dataclasses.is_dataclass(result):
        maps = {f.name: cut(getattr(result, f.name)) for f in dataclasses.fields(result)}
        return {"$class": f"{type(result).__module__}:{type(result).__qualname__}",
                "maps": {k: v for k, v in maps.items() if v is not None},
                "complete": all(v is not None or getattr(result, k) is None for k, v in maps.items())}
    raise ValueError(f"tiles of a {type(result).__name__} cannot be put together")


def tiled(tool, raster, *args, halo: int | None = None, tile: int = 1024, on=None, output=None,
          n_workers: int | None = None, **kwargs):
    """Run a neighbourhood tool on a large raster tile by tile, on this computer or on helpers.

    Each tile is cut with a margin (``halo``) of the cells its edge cells need, the
    tool runs on it and its result over the tile proper goes into place: the same
    values as on the whole raster, for tools that see only a neighbourhood (moving
    windows, lines of sight, slopes). A raster read lazily (``ls.load_raster(path,
    chunks=...)``) is read tile by tile.

    Parameters
    ----------
    tool
        A name of :data:`TOOLS` or the function; its first argument is the raster.
    raster
        A DataArray (in memory or lazy).
    *args, **kwargs
        The tool's other arguments.
    halo
        The margin in cells (default: what the tool needs, known for ``"fad"``,
        ``"window_metrics"``, ``"surface.geomorphons"``, ``"surface.tri"``,
        ``"surface.terrain"``, ``"surface.curvature"``, ``"surface.trasp"``,
        ``"surface.dissection"`` and ``"services.carbon"``, whose tiles give the same bits
        as the whole map; required for other tools. The window means of
        ``"surface.tpi"``, ``"vrm"``, ``"srr"``, ``"hsp"`` and ``"spherical_sd"`` are
        running sums along the rows, so their tiles agree to about 1e-12).
    tile
        The side of a tile in cells (without its halo).
    on
        ``ls.helpers()``: the tiles run here and on the helpers of this computer's group.
    output
        Save the result there (``ls.save``) and return its path.
    n_workers
        Tiles at once on this computer, without ``on``.

    Returns
    -------
    The tool's result over the whole raster: a DataArray, a Dataset, or the tool's
    result class when every part of it is a map (else a dict of its maps).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = ls.nlm.fbm(120, 150, fract_dim=1.4, seed=3) * 100
    >>> a = ls.parallel.tiled("surface.tri", dem, tile=50)
    >>> bool(np.array_equal(a.values, ls.surface.tri(dem).values, equal_nan=True))
    True
    """
    import dataclasses
    import importlib
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import numpy as np
    import xarray as xr

    from ._dist import InputFile, Job, Task, run_job
    from ._save import save

    name = _registry.name_of(tool)
    fn = _registry.resolve(name)
    if not isinstance(raster, xr.DataArray):
        raise TypeError("tiled() takes a DataArray (ls.load_raster(path, chunks=...) reads a file lazily)")
    ny, nx = raster.shape[-2:]
    try:
        res = abs(float(raster.rio.resolution()[0]))
    except Exception:  # noqa: BLE001
        res = 1.0
    if halo is None:
        if name not in _registry.HALO:
            raise ValueError(f"give the halo of {name!r}: how many cells around a tile it needs")
        import inspect

        try:
            bound = dict(inspect.signature(fn).bind_partial(raster, *args, **kwargs).arguments)
        except TypeError:
            bound = dict(kwargs)
        halo = _registry.HALO[name](bound, res)
    halo, tile = int(halo), int(tile)
    if tile < 1 or halo < 0:
        raise ValueError("tile must be positive and halo not negative")
    tiles = []
    for r0 in range(0, ny, tile):
        for c0 in range(0, nx, tile):
            r1, c1 = min(ny, r0 + tile), min(nx, c0 + tile)
            rr0, rr1, cc0, cc1 = max(0, r0 - halo), min(ny, r1 + halo), max(0, c0 - halo), min(nx, c1 + halo)
            tiles.append(((r0, r1, c0, c1), (rr0, rr1, cc0, cc1)))
    parts = [None] * len(tiles)

    def window(k):
        _, (rr0, rr1, cc0, cc1) = tiles[k]
        return raster[..., rr0:rr1, cc0:cc1].load()

    def core(k):
        (r0, r1, c0, c1), (rr0, _, cc0, _) = tiles[k]
        return slice(r0 - rr0, r1 - rr0), slice(c0 - cc0, c1 - cc0)

    if on is None:
        def one(k):
            w = window(k)
            rs, cs = core(k)
            return _crop(fn(w, *args, **kwargs), rs, cs, tuple(w.shape[-2:]))

        workers = max(1, min(len(tiles), n_workers or 1))
        if workers == 1:
            parts = [one(k) for k in range(len(tiles))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(one, range(len(tiles))))
    else:
        _registry.check_strings(list(args), "args")
        _registry.check_strings(kwargs, "kwargs")
        with tempfile.TemporaryDirectory(prefix="landschaft-tiles-") as tmp:
            tmp = Path(tmp)
            stored = _pack.write((list(args), kwargs), tmp / "args")
            shared = [InputFile.of(p, f"args/{p.name}") for p in _pack.files(stored)]
            tasks = []
            for k in range(len(tiles)):
                t = _pack.write(window(k), tmp / f"t{k}")
                files = shared + [InputFile.of(p, f"t{k}/{p.name}") for p in _pack.files(t)]
                rs, cs = core(k)
                tasks.append(Task(f"{k}", "tile", {"tool": name, "args": "args", "tile": f"t{k}",
                                                   "core": [rs.start, rs.stop, cs.start, cs.stop]}, files, root=tmp))

            def collect(task, folder, stats):
                parts[int(task.name)] = _pack.read(Path(folder) / "result")

            report = run_job(on, Job(f"{name} tiles", tasks, collect))
        if report.failed:
            first, error = next(iter(report.failed.items()))
            raise RuntimeError(f"{len(report.failed)} tile(s) failed; tile {first}: {error}")

    def assemble(pieces):
        first = pieces[0]
        lead = first.shape[:-2]
        out = np.empty((*lead, ny, nx), dtype=first.dtype)
        for k, ((r0, r1, c0, c1), _) in enumerate(tiles):
            out[..., r0:r1, c0:c1] = np.asarray(pieces[k].values)
        coords = {d: raster.coords[d] for d in raster.dims[-2:] if d in raster.coords}
        for d in first.dims[:-2]:
            if d in first.coords:
                coords[d] = first.coords[d]
        for k, c in raster.coords.items():
            if c.ndim == 0:
                coords[k] = c
        da = xr.DataArray(out, dims=(*first.dims[:-2], *raster.dims[-2:]), coords=coords, name=first.name,
                          attrs=dict(first.attrs))
        return da

    first = parts[0]
    if isinstance(first, xr.DataArray):
        result = assemble(parts)
    elif isinstance(first, xr.Dataset):
        result = xr.Dataset({v: assemble([p[v] for p in parts]) for v in first.data_vars}, attrs=dict(first.attrs))
    else:
        maps = {k: assemble([p["maps"][k] for p in parts]) for k in first["maps"]}
        result = maps
        if first["complete"]:
            module, _, qualname = first["$class"].partition(":")
            cls = importlib.import_module(module)
            for part in qualname.split("."):
                cls = getattr(cls, part)
            names = {f.name for f in dataclasses.fields(cls) if f.init}
            if names <= set(maps) | {f.name for f in dataclasses.fields(cls) if f.default is None}:
                result = cls(**{k: maps.get(k) for k in names})
    if output is not None:
        return save(result, output)
    return result
