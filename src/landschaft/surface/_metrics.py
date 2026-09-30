# SPDX-License-Identifier: GPL-3.0-or-later
"""Every surface metric of a raster at once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .. import _core
from .._threads import threaded
from ._grid import Surface, read_surface

__all__ = ["metrics", "list_metrics"]


@dataclass(frozen=True)
class Metric:
    name: str
    group: str
    description: str
    # the computation shared by the metrics of a group: Surface -> {metric: value}
    compute: Callable[[Surface], dict]


def _amplitude(s: Surface) -> dict:
    a = _core.surf_amplitude(s.values)
    return {k: a[k] for k in ("sa", "sq", "ssk", "sku", "sv", "sph", "smean")}


METRICS: dict[str, Metric] = {m.name: m for m in [
    Metric("sa", "amplitude", "average roughness", _amplitude),
    Metric("sq", "amplitude", "root mean square roughness", _amplitude),
    Metric("ssk", "amplitude", "skewness (adjusted)", _amplitude),
    Metric("sku", "amplitude", "excess kurtosis", _amplitude),
    Metric("sv", "amplitude", "maximum valley depth", _amplitude),
    Metric("sph", "amplitude", "maximum peak height", _amplitude),
    Metric("smean", "amplitude", "mean peak height", _amplitude),
]}


def list_metrics() -> pd.DataFrame:
    """The surface metrics :func:`metrics` computes.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``group`` and ``description``.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.list_metrics().head(3)
      metric      group                 description
    0     sa  amplitude           average roughness
    1     sq  amplitude  root mean square roughness
    2    ssk  amplitude         skewness (adjusted)
    """
    return pd.DataFrame([(m.name, m.group, m.description) for m in METRICS.values()],
                        columns=["metric", "group", "description"])


@threaded
def metrics(x, metrics: str | Iterable[str] | None = None, *, extent=None, crs=None, band=None,
            nodata=None) -> pd.Series:
    """Surface metrics of a raster, as a Series indexed by metric.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (a single band; NaN or ``nodata`` = missing), usually with
        its trend removed (:func:`remove_plane`).
    metrics : str or list of str, optional
        Names from :func:`list_metrics` (default: all). Metrics that share a
        computation are computed once.
    extent : tuple of float, optional
        ``(xmin, xmax, ymin, ymax)`` of an array (default: unit cells).
    crs : optional
        CRS of an array.
    band : int, optional
        Band of a multi-band file or DataArray (1-based).
    nodata : float, optional
        Value that marks missing cells.

    Returns
    -------
    pandas.Series

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> ls.surface.metrics(z, ["sa", "sq", "sku"]).round(4)
    sa     0.0443
    sq     0.0648
    sku    6.4977
    dtype: float64
    """
    names = list(METRICS) if metrics is None else [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"unknown surface metric(s) {', '.join(map(repr, unknown))}; see ls.surface.list_metrics()")
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    done: dict[Callable, dict] = {}
    out = {}
    for n in names:
        f = METRICS[n].compute
        if f not in done:
            done[f] = f(s)
        out[n] = done[f][n]
    return pd.Series(out, dtype=float)
