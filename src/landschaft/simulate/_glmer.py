# SPDX-License-Identifier: GPL-3.0-or-later
"""Binomial generalized linear mixed models with one grouping factor (lme4's ``glmer``, Laplace).

``y ~ X b + (1 | g)`` or ``y ~ X b + (z | g)``: the logit of the
probability is the fixed part plus a random intercept (and slope on ``z``)
per group, ``b_g = Lambda u_g`` with ``u_g ~ N(0, I)``. The deviance is
lme4's Laplace approximation, ``sum(dev.resid) + |u|^2 + log det(L L')``
at the conditional modes of ``u`` (penalized IRLS, independent per group
for one grouping factor), minimized over the covariance parameters
``theta`` (the lower triangle of ``Lambda``) and ``b``. lme4 stops its
Nelder-Mead at an absolute deviance tolerance of 1e-5; the optimum here is
found more tightly, so estimates agree with lme4's within its tolerance,
not bit for bit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.special import expit

__all__ = ["GlmerFit", "dredge", "glmer"]


@dataclass
class GlmerFit:
    """A fitted model: fixed effects, theta, conditional modes and fit statistics."""

    fixed: np.ndarray
    theta: np.ndarray
    ranef: np.ndarray            # (groups, q): b_g = Lambda u_g
    deviance: float
    names: list
    random_names: list
    levels: list
    nobs: int
    converged: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def npar(self) -> int:
        return len(self.fixed) + len(self.theta)

    @property
    def aic(self) -> float:
        return self.deviance + 2 * self.npar

    @property
    def loglik(self) -> float:
        return -self.deviance / 2

    def coef(self) -> dict:
        """lme4's ``coef()``: per level, the fixed effects plus the level's random effects."""
        out = {}
        for k, lev in enumerate(self.levels):
            row = dict(zip(self.names, self.fixed))
            for j, nm in enumerate(self.random_names):
                row[nm] = row.get(nm, 0.0) + self.ranef[k, j]
            out[lev] = row
        return out


def _lambda(theta: np.ndarray, q: int) -> np.ndarray:
    if q == 1:
        return np.array([[theta[0]]])
    return np.array([[theta[0], 0.0], [theta[1], theta[2]]])


class _Problem:
    def __init__(self, y, X, Z, g, ngroups):
        self.y, self.X, self.Z, self.g, self.m = y, X, Z, g, ngroups
        self.q = Z.shape[1]
        self.u = np.zeros((ngroups, self.q))

    def _sums(self, v):
        return np.bincount(self.g, weights=v, minlength=self.m)

    def conditional(self, theta, beta, u0=None):
        """PIRLS for u given theta and beta: (u, mu, logdet)."""
        lam = _lambda(theta, self.q)
        y, Z = self.y, self.Z
        off = self.X @ beta
        ZL = Z @ lam                    # n x q: Z_i' Lambda
        u = self.u.copy() if u0 is None else u0.copy()

        def pdev(u):
            eta = off + np.einsum("ij,ij->i", ZL, u[self.g])
            mu = expit(eta)
            dev = -2 * np.where(y > 0, np.log(mu), np.log1p(-mu))
            return self._sums(dev) + np.sum(u * u, axis=1), mu
        cur, mu = pdev(u)
        for _ in range(200):
            w = mu * (1 - mu)
            r = y - mu
            grad = np.stack([self._sums(ZL[:, a] * r) for a in range(self.q)], axis=1) - u
            H = np.empty((self.m, self.q, self.q))
            for a in range(self.q):
                for b in range(self.q):
                    H[:, a, b] = self._sums(w * ZL[:, a] * ZL[:, b]) + (1.0 if a == b else 0.0)
            step = np.linalg.solve(H, grad[..., None])[..., 0]
            new = u + step
            nd, nmu = pdev(new)
            worse = nd > cur + 1e-12
            k = 0
            while worse.any() and k < 30:  # step halving per group
                step[worse] *= 0.5
                new = u + step
                nd, nmu = pdev(new)
                worse = nd > cur + 1e-12
                k += 1
            delta = np.max(np.abs(new - u))
            u, cur, mu = new, nd, nmu
            if delta < 1e-12:
                break
        w = mu * (1 - mu)
        H = np.empty((self.m, self.q, self.q))
        for a in range(self.q):
            for b in range(self.q):
                H[:, a, b] = self._sums(w * ZL[:, a] * ZL[:, b]) + (1.0 if a == b else 0.0)
        logdet = float(np.sum(np.linalg.slogdet(H)[1]))
        self.u = u
        return u, mu, logdet

    def laplace(self, params):
        nt = 1 if self.q == 1 else 3
        theta, beta = params[:nt], params[nt:]
        u, mu, logdet = self.conditional(theta, beta)
        y = self.y
        dev = -2 * np.sum(np.where(y > 0, np.log(mu), np.log1p(-mu)))
        return float(dev + np.sum(u * u) + logdet)


def _glm_start(y, X):
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = expit(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        z = X @ beta + (y - mu) / w
        new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
        if np.max(np.abs(new - beta)) < 1e-12:
            beta = new
            break
        beta = new
    return beta


def glmer(y, X, groups, *, names, random=None, Zcol=None) -> GlmerFit:
    """Fit ``y ~ X + (1 | groups)`` or ``y ~ X + (z | groups)`` (``Zcol``: the slope's values).

    Parameters
    ----------
    y
        0/1 response.
    X
        Fixed-effects model matrix (with the intercept column).
    groups
        The level of each observation.
    names
        The names of X's columns.
    random
        Name of the random slope's variable (None: random intercept only).
    Zcol
        The random slope's values.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    levels, g = np.unique(np.asarray(groups), return_inverse=True)
    Z = np.ones((y.size, 1)) if random is None else np.column_stack([np.ones(y.size), np.asarray(Zcol, float)])
    prob = _Problem(y, X, Z, g, len(levels))
    q = Z.shape[1]
    theta0 = np.array([1.0]) if q == 1 else np.array([1.0, 0.0, 1.0])
    beta0 = _glm_start(y, X)
    x0 = np.concatenate([theta0, beta0])
    nt = theta0.size
    bounds = [(0, None)] + ([(None, None), (0, None)] if q == 2 else []) + [(None, None)] * beta0.size
    res = optimize.minimize(prob.laplace, x0, method="L-BFGS-B", bounds=bounds,
                            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000, "maxls": 50})
    best = res.x
    res2 = optimize.minimize(prob.laplace, best, method="Nelder-Mead", bounds=bounds,
                             options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "adaptive": True})
    if res2.fun < res.fun:
        best = res2.x
    theta, beta = best[:nt], best[nt:]
    u, _, _ = prob.conditional(theta, beta)
    lam = _lambda(theta, q)
    ranef = u @ lam.T
    rnames = ["(Intercept)"] + ([] if random is None else [random])
    return GlmerFit(beta, theta, ranef, prob.laplace(best), list(names), rnames, [lv.item() if hasattr(lv, "item")
                                                                                   else lv for lv in levels],
                    int(y.size), bool(res.success or res2.success))


def dredge(fit_fn, terms: list, *, fixed=(), m_min: int = 0, m_max: int | None = None) -> list:
    """MuMIn's dredge: every subset of ``terms`` (with the fixed ones) within the size limits, by AIC.

    ``fit_fn(subset) -> GlmerFit``. Returns [(subset, fit)] sorted by AIC.
    """
    fixed = [t for t in terms if t in set(fixed)]
    free = [t for t in terms if t not in set(fixed)]
    m_max = len(terms) if m_max is None else m_max
    out = []
    for k in range(len(free) + 1):
        for combo in itertools.combinations(free, k):
            subset = [t for t in terms if t in set(combo) | set(fixed)]
            if not m_min <= len(subset) <= m_max:
                continue
            out.append((subset, fit_fn(subset)))
    out.sort(key=lambda t: t[1].aic)
    return out
