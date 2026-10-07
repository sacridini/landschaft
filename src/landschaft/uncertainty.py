# SPDX-License-Identifier: GPL-3.0-or-later
"""Propagation of uncertainty: totals, products, the delta method and Monte Carlo.

General tools for the uncertainty of results computed from uncertain inputs, the ones plangea
(International Institute for Sustainability; GPL >= 3) uses in its restoration plans: the standard
deviation of a total of uncertain pixel values (``post_aggregate_table_row``), of a product of two
uncertain layers (``aux_calc_var_sd``), of a result with respect to uncertain parameters
(``harmonize_it``, ``multi_exrisk_aggregate``), and the confidence interval of a value.

References
----------
Goodman, L. A. (1960). On the exact variance of products. *Journal of the American Statistical
    Association* 55, 708-713. doi:10.1080/01621459.1960.10483369
JCGM 100:2008. Evaluation of measurement data - Guide to the expression of uncertainty in
    measurement (GUM), section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import xarray as xr

__all__ = ["DeltaResult", "MonteCarloResult", "delta_method", "interval", "monte_carlo", "product_sd", "sum_sd"]

_METHODS = ("quadratic", "linear")


def _values(a) -> np.ndarray:
    return np.asarray(getattr(a, "values", a), dtype=np.float64)


def _method(method: str) -> str:
    if method not in _METHODS:
        raise ValueError(f"method must be 'quadratic' or 'linear', got {method!r}")
    return method


def sum_sd(sd, weights=None, *, method: str = "quadratic", correlation: float | None = None) -> float:
    """Standard deviation of a total of uncertain values, sum of w_i v_i.

    Each value v_i (a pixel's carbon, say) has a standard deviation
    sigma_i, and the total weighs it by w_i (the pixel's area times the
    share restored). The total's standard deviation is

    - ``"quadratic"``: errors independent, sqrt(sum (sigma_i w_i)^2);
    - ``"linear"``: errors that add up, sum |sigma_i w_i| (an upper bound,
      reached when they all go the same way);
    - with ``correlation`` rho (one value for every pair):
      sqrt((1 - rho) sum s_i^2 + rho (sum s_i)^2), s_i = sigma_i w_i, which
      is the quadratic form for 0 and |sum s_i| for 1.

    Parameters
    ----------
    sd
        The standard deviations (an array or raster; NaN left out).
    weights
        The weights (same shape, or a number; default 1).
    method
        ``"quadratic"`` or ``"linear"`` (ignored with ``correlation``).
    correlation
        The correlation of every pair of errors, -1 to 1.

    Returns
    -------
    float

    Notes
    -----
    plangea's ``prop_quad`` and ``prop_lin`` are the two methods.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.uncertainty.sum_sd([3.0, 4.0])
    5.0
    >>> ls.uncertainty.sum_sd([3.0, 4.0], method="linear")
    7.0
    >>> ls.uncertainty.sum_sd([3.0, 4.0], correlation=1.0)
    7.0
    """
    s = _values(sd)
    if weights is not None:
        s = s * _values(weights)
    s = s[~np.isnan(s)] if s.ndim else np.atleast_1d(s)
    if correlation is not None:
        rho = float(correlation)
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"correlation must be between -1 and 1, got {correlation}")
        var = (1.0 - rho) * float(np.sum(s * s)) + rho * float(np.sum(s)) ** 2
        return float(np.sqrt(max(var, 0.0)))
    if _method(method) == "quadratic":
        return float(np.sqrt(np.sum(s * s)))
    return float(np.sum(np.abs(s)))


def product_sd(x, sd_x, v, sd_v, *, total: bool = False, method: str = "quadratic"):
    """Standard deviation of the product of two uncertain values, x v.

    With x and v independent, the exact variance of the product (Goodman
    1960) is x^2 sigma_v^2 + v^2 sigma_x^2 + sigma_x^2 sigma_v^2.

    Parameters
    ----------
    x, sd_x, v, sd_v
        The values and their standard deviations (arrays or rasters of one
        shape, or numbers).
    total
        Return the standard deviation of the sum of the products instead of
        each product's: quadratic, sqrt(sum of the variances); linear, the
        sum of |x| sigma_v + |v| sigma_x + sigma_x sigma_v.
    method
        For ``total``: ``"quadratic"`` (independent products) or
        ``"linear"``.

    Returns
    -------
    numpy.ndarray or xarray.DataArray or float

    Notes
    -----
    plangea's ``aux_calc_var_sd`` adds the cross term sigma_x sigma_v
    without squaring it in the quadratic form, and by default takes 10 % of
    the deviation of x as the deviation of v; here the deviations are always
    given and the variance is Goodman's.

    Examples
    --------
    >>> import landschaft as ls
    >>> round(float(ls.uncertainty.product_sd(2.0, 0.1, 3.0, 0.2)), 4)
    0.5004
    """
    xv, sx, vv, sv = (_values(a) for a in (x, sd_x, v, sd_v))
    var = xv * xv * sv * sv + vv * vv * sx * sx + sx * sx * sv * sv
    if total:
        keep = ~np.isnan(var)
        if _method(method) == "quadratic":
            return float(np.sqrt(np.sum(var[keep])))
        lin = np.abs(xv) * sv + np.abs(vv) * sx + sx * sv
        return float(np.sum(lin[~np.isnan(lin)]))
    out = np.sqrt(var)
    for a in (x, v, sd_x, sd_v):
        if isinstance(a, xr.DataArray) and a.shape == out.shape:
            r = a.copy(data=out)
            r.name = "sd"
            return r
    return out


@dataclass(frozen=True)
class DeltaResult:
    """Result of :func:`delta_method`.

    Attributes
    ----------
    value
        The function at the parameters.
    sd
        The standard deviation of every output.
    jacobian
        The derivatives of the outputs (rows, flattened) with respect to the
        parameters (columns).
    """

    value: object = field(repr=False)
    sd: object = field(repr=False)
    jacobian: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of :func:`monte_carlo`.

    Attributes
    ----------
    mean, sd
        The mean and the standard deviation of every output over the draws.
    quantiles
        The quantiles asked for, one row per quantile.
    samples
        The outputs of every draw (draws x outputs), if kept.
    """

    mean: object = field(repr=False)
    sd: object = field(repr=False)
    quantiles: np.ndarray = field(repr=False)
    samples: np.ndarray | None = field(default=None, repr=False)


def _shape_like(template, values: np.ndarray):
    if isinstance(template, xr.DataArray):
        return template.copy(data=values.reshape(template.shape))
    t = np.asarray(template)
    return values.reshape(t.shape) if t.ndim else float(values.reshape(-1)[0])


def delta_method(
    f: Callable,
    params,
    sd,
    *,
    covariance=None,
    method: str = "quadratic",
    step=None,
    scheme: str = "central",
) -> DeltaResult:
    """Uncertainty of a result from the uncertainty of its parameters (delta method).

    The result ``f(params)`` (a number, an array, a raster) changes with
    each parameter p_k by df/dp_k, found here by finite differences; its
    standard deviation is

    - ``"quadratic"`` (independent parameters): sqrt(sum_k (df/dp_k sigma_k)^2),
      or sqrt(J C J^T) with a ``covariance`` matrix C;
    - ``"linear"``: sum_k |df/dp_k sigma_k|.

    It is the first-order propagation of the GUM; :func:`monte_carlo`
    checks it when f is far from linear over the parameters' spread.

    Parameters
    ----------
    f
        The function: ``f(params)`` with the parameters as a 1-D array.
    params
        The parameters' values.
    sd
        Their standard deviations (ignored with ``covariance``).
    covariance
        The parameters' covariance matrix.
    method
        ``"quadratic"`` or ``"linear"``.
    step
        The finite-difference step of every parameter (default: the cube
        root (central) or square root (forward) of the machine epsilon,
        times max(|p|, 1)).
    scheme
        ``"central"`` (f at p + h and p - h) or ``"forward"`` (p + h and p),
        plangea's.

    Returns
    -------
    DeltaResult
        The value, the standard deviations and the Jacobian.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> r = ls.uncertainty.delta_method(lambda p: p[0] * p[1], [2.0, 3.0], [0.1, 0.2])
    >>> round(float(r.sd), 4)
    0.5
    """
    if scheme not in ("central", "forward"):
        raise ValueError(f"scheme must be 'central' or 'forward', got {scheme!r}")
    p = np.atleast_1d(np.asarray(params, dtype=np.float64))
    if step is None:
        eps = np.finfo(np.float64).eps
        h = (eps ** (1 / 3) if scheme == "central" else eps ** 0.5) * np.maximum(np.abs(p), 1.0)
    else:
        h = np.broadcast_to(np.asarray(step, dtype=np.float64), p.shape).copy()
    value = f(p.copy())
    base = _values(value).reshape(-1)
    jac = np.empty((base.size, p.size))
    for k in range(p.size):
        up = p.copy()
        up[k] += h[k]
        fu = _values(f(up)).reshape(-1)
        if scheme == "central":
            dn = p.copy()
            dn[k] -= h[k]
            jac[:, k] = (fu - _values(f(dn)).reshape(-1)) / (2.0 * h[k])
        else:
            jac[:, k] = (fu - base) / h[k]
    if covariance is not None:
        c = np.asarray(covariance, dtype=np.float64)
        if c.shape != (p.size, p.size):
            raise ValueError(f"covariance must be {p.size} x {p.size}")
        if _method(method) == "linear":
            raise ValueError("a covariance matrix needs the quadratic method")
        out = np.sqrt(np.maximum(np.einsum("ik,kl,il->i", jac, c, jac), 0.0))
    else:
        s = np.broadcast_to(np.asarray(sd, dtype=np.float64), p.shape)
        t = jac * s
        out = np.sqrt(np.sum(t * t, axis=1)) if _method(method) == "quadratic" else np.sum(np.abs(t), axis=1)
    return DeltaResult(value=value, sd=_shape_like(value, out), jacobian=jac)


def monte_carlo(
    f: Callable,
    params,
    sd,
    *,
    n: int = 1000,
    covariance=None,
    seed: int = 0,
    quantiles=(0.025, 0.975),
    keep: bool = False,
) -> MonteCarloResult:
    """Uncertainty of a result by drawing its parameters (Monte Carlo).

    The parameters are drawn ``n`` times from normal distributions (their
    ``sd``, or the ``covariance`` matrix) with a fixed ``seed``, and the
    result is computed for every draw: its mean, standard deviation and
    quantiles describe its uncertainty without assuming f linear.

    Parameters
    ----------
    f
        The function: ``f(params)`` with the parameters as a 1-D array.
    params, sd
        The parameters' means and standard deviations.
    n
        The number of draws.
    covariance
        The parameters' covariance matrix (instead of ``sd``).
    seed
        The seed of the draws (NumPy's PCG64).
    quantiles
        The quantiles to report.
    keep
        Keep every draw's outputs in ``samples``.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.uncertainty.monte_carlo(lambda p: p[0] * p[1], [2.0, 3.0], [0.1, 0.2], n=20000, seed=1)
    >>> round(float(r.sd), 2)
    0.5
    """
    if n < 2:
        raise ValueError("at least 2 draws are needed")
    p = np.atleast_1d(np.asarray(params, dtype=np.float64))
    rng = np.random.default_rng(seed)
    if covariance is not None:
        draws = rng.multivariate_normal(p, np.asarray(covariance, dtype=np.float64), size=n)
    else:
        draws = p + rng.standard_normal((n, p.size)) * np.broadcast_to(np.asarray(sd, dtype=np.float64), p.shape)
    first = f(draws[0].copy())
    out = np.empty((n, _values(first).size))
    out[0] = _values(first).reshape(-1)
    for i in range(1, n):
        out[i] = _values(f(draws[i].copy())).reshape(-1)
    q = np.quantile(out, np.asarray(quantiles, dtype=np.float64), axis=0)
    return MonteCarloResult(mean=_shape_like(first, out.mean(axis=0)), sd=_shape_like(first, out.std(axis=0, ddof=1)),
                            quantiles=q, samples=out if keep else None)


def interval(value, sd, level: float = 0.95):
    """The normal confidence interval of a value: value -/+ z sd, z the normal quantile of the level.

    plangea fixes z at 1.96 (the 95 % level).

    Parameters
    ----------
    value, sd
        The value and its standard deviation (numbers, arrays or rasters).
    level
        The confidence level, 0 to 1.

    Returns
    -------
    tuple
        The lower and the upper bound.

    Examples
    --------
    >>> import landschaft as ls
    >>> low, high = ls.uncertainty.interval(10.0, 1.0)
    >>> round(low, 3), round(high, 3)
    (8.04, 11.96)
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2.0))
    return value - z * sd, value + z * sd
