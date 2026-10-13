# SPDX-License-Identifier: GPL-3.0-or-later
"""The C++ kernels of the simulation steps (rules, ordered allocation, demands, Markov chains, WoE ranges).

Each kernel replaced numpy code of landschaft's: that code is kept here as
the reference, and the kernels must give the same bits on random inputs
(the numbers are drawn so that ties, NaN and edge cases happen often).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from landschaft import _core

# ----------------------------------------------------------------------------- the numpy references


def ref_allow_matrix(x, hist, classes, cd, rules):
    index = {c: k for k, c in enumerate(classes)}
    lu = np.array([index.get(v, -1) for v in x])
    out = np.ones((x.size, len(classes)))
    r = rules[lu]
    h = hist[:, None]
    out[r == 0] = 0
    dec = cd[lu][:, None] < 0
    out = np.where(r == -1, np.where(dec, 1, 0), out)
    inc = (cd > 0)[None, :]
    out = np.where(r == -2, np.where(inc, 1, 0), out)
    out = np.where((r >= 100) & (r < 1000), np.where(h < (r - 100), 1, 0), out)
    out = np.where(r >= 1000, np.where(h < (r - 1000), 0, 1), out)
    return np.where(out == 0, np.nan, 1.0)


def ref_auto_convert(x, prob, classes, mask):
    index = {c: k for k, c in enumerate(classes)}
    lu = np.array([index[v] for v in x])
    own = prob[np.arange(x.size), lu]
    out = np.full(x.size, np.nan)
    gone = np.isnan(own)
    if gone.any():
        p = prob[gone]
        best = np.where(np.isnan(p), -np.inf, p)
        arg = np.argmax(best, axis=1)
        has = (best > -1).any(axis=1)
        values = np.asarray(classes, dtype=np.float64)[arg]
        out[gone] = np.where(has, values, x[gone])
    keep = ~gone & (mask == 0)
    out[keep] = x[keep]
    return out


def ref_neighbourhood(landscape, classes, rules):
    out = np.ones((landscape.size, len(classes)))
    for c, threshold, w in rules:
        if c not in classes:
            continue
        presence = np.where(np.isnan(landscape), np.nan, (landscape == c).astype(np.float64))
        frac = _core.sim_focal_mean(np.ascontiguousarray(presence), w).ravel()
        out[:, classes.index(c)] = np.where(frac >= threshold, 1.0, np.nan)
    return out


def ref_ordered(tprob, x, demand, classes, order, stochastic, rng):
    def draw(p, n, keep):
        if not stochastic:
            return np.arange(p.size)
        counter = 0
        while True:
            counter += 1
            sel = np.flatnonzero(keep(p, rng.runif(p.size)))
            if sel.size >= n or counter > 1000:
                return sel

    tprob = tprob.copy()
    area = np.array([np.sum(x == c) for c in classes], dtype=np.float64)
    if np.sum(np.abs(demand - area)) == 0:
        return x.copy(), 0
    x1 = x.copy()
    short = 0
    for cat in order:
        ix = classes.index(cat)
        n = int(demand[ix] - np.sum(x1 == cat))
        if n == 0:
            tprob[x == cat] = np.nan
        elif n > 0:
            cand = np.flatnonzero(x1 != cat)
            p = tprob[cand, ix]
            o = np.argsort(np.where(np.isnan(p), np.inf, -p), kind="stable")
            o = o[~np.isnan(p[o])]
            p, cand = p[o], cand[o]
            sel = draw(p, n, lambda p, u: p >= u)[:n]
            short += n - sel.size
            x1[cand[sel]] = cat
            tprob[x1 == cat] = np.nan
        else:
            cand = np.flatnonzero(x == cat)
            p = tprob[cand, ix]
            o = np.argsort(np.where(np.isnan(p), np.inf, p), kind="stable")
            o = o[~np.isnan(p[o])]
            p, cand = p[o], cand[o]
            sel = draw(p, -n, lambda p, u: p < u)[:-n]
            short += -n - sel.size
            x1[cand[sel]] = -1
            tprob[x1 == cat] = np.nan
    left = x1 == -1
    x1[left] = x[left]
    return x1, short


def ref_approx_extrap(x, y, xout):
    keep = ~pd.Series(x).duplicated().to_numpy()
    x, y = x[keep], y[keep]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    n = x.size
    out = np.empty(xout.size)
    for k, v in enumerate(xout):
        if v < x[0]:
            out[k] = (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0]) + y[0]
        elif v > x[-1]:
            out[k] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) * (v - x[n - 2]) + y[n - 2]
        else:
            i, j = 0, n - 1
            while i < j - 1:
                ij = (i + j) // 2
                if v < x[ij]:
                    j = ij
                else:
                    i = ij
            if v == x[j]:
                out[k] = y[j]
            elif v == x[i]:
                out[k] = y[i]
            else:
                out[k] = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    return out


def ref_round_sum(x, ncell):
    def rsum(v):
        return float(_core.nlm_r_sum(np.ascontiguousarray(v, dtype=np.float64)))

    out = np.empty_like(x)
    for r in range(x.shape[0]):
        y = x[r] / rsum(x[r]) * ncell
        xint = np.floor(y)
        frac = y - np.floor(y)
        ix = np.argsort(frac, kind="stable")
        tot = ncell - rsum(np.floor(y))
        if tot > 0:
            xint[ix[int(len(y) - tot):]] += 1
        out[r] = xint
    return out


def ref_from_counts(c):
    keep = c.sum(axis=1) > 0
    reached = keep | (c[keep].sum(axis=0) > 0)
    sub = c[np.ix_(reached, reached)]
    p = np.zeros_like(sub)
    r = sub.sum(axis=1)
    ok = r > 0
    p[ok] = sub[ok] / r[ok, None]
    p[~ok, np.flatnonzero(~ok)] = 1.0
    return p, np.flatnonzero(reached)


def ref_weights(nb, b, n, a, fix):
    abnormal_plus = nb == 0 or b - nb == 0 or n == 0 or a - n == 0
    if abnormal_plus:
        if fix and b > 0 and n > 0 and a - n > 0:
            wp = math.log(b * (a - n) + 1.0) if b - nb == 0 else -math.log(b * n + 1.0)
        else:
            wp = 0.0
    else:
        wp = math.log((nb / n) / ((b - nb) / (a - n)))
    abnormal_minus = n - nb == 0 or a - b - n + nb == 0 or n == 0 or a - n == 0
    if abnormal_plus or abnormal_minus:
        return wp, 0.0, False
    wm = math.log(((n - nb) / n) / ((a - b - n + nb) / (a - n)))
    contrast = wp - wm
    sd = math.sqrt(1.0 / nb + 1.0 / (b - nb) + 1.0 / (n - nb) + 1.0 / (a - b - n + nb))
    return wp, contrast, abs(contrast) > 1.96 * sd


def ref_generalise(possible, executed, min_delta, max_delta, tolerance, fix):
    def angle(dx, dy):
        return math.degrees(math.atan2(dy, dx))

    n = possible.size
    cum_b = np.cumsum(possible, dtype=np.float64)
    cum_nb = np.cumsum(executed, dtype=np.float64)
    a_total, n_total = cum_b[-1] if n else 0.0, cum_nb[-1] if n else 0.0
    xs, ys = [0.0], [0.0]
    for k in range(n):
        abnormal = cum_nb[k] == 0 or cum_b[k] - cum_nb[k] == 0
        ratio = 0.0 if abnormal and not fix else math.exp(ref_weights(cum_nb[k], cum_b[k], n_total, a_total, fix)[0])
        xs.append(cum_b[k])
        ys.append(cum_b[k] * ratio)
    bounds, anchor, prev = [], 0, None
    for i in range(1, n + 1):
        dx = xs[i] - xs[anchor]
        ref = 0.0 if prev is None else angle(xs[anchor] - xs[prev], ys[anchor] - ys[prev])
        phi = abs(angle(dx, ys[i] - ys[anchor]) - ref)
        if (dx >= min_delta and (dx > max_delta or phi > tolerance)) or i == n:
            bounds.append(i)
            prev, anchor = anchor, i
    return bounds


# ----------------------------------------------------------------------------- the kernels against them

def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("seed", range(5))
def test_allow_matrix(seed):
    rng = np.random.default_rng(seed)
    classes = [1.0, 2.0, 4.0]
    x = rng.choice(classes, 500)
    hist = rng.integers(0, 12, 500)
    cd = rng.integers(-3, 4, 3).astype(float)
    rules = rng.choice([0, 1, -1, -2, 105, 110, 1003, 1008], (3, 3))
    _same(_core.sim_allow_matrix(x, hist, np.array(classes), cd, rules), ref_allow_matrix(x, hist, classes, cd, rules))


def test_allow_matrix_unknown_class():
    with pytest.raises(ValueError, match="does not know"):
        _core.sim_allow_matrix(np.array([1.0, 3.0]), np.ones(2, np.int64), np.array([1.0, 2.0]), np.zeros(2),
                               np.ones((2, 2), np.int64))


@pytest.mark.parametrize("seed", range(5))
def test_auto_convert(seed):
    rng = np.random.default_rng(seed)
    classes = [1.0, 2.0, 3.0, 5.0]
    x = rng.choice(classes, 400)
    prob = rng.choice([np.nan, -2.0, 0.1, 0.5, 0.5, 0.9], (400, 4))  # ties, nothing above -1, NaN
    mask = rng.integers(0, 2, 400)
    _same(_core.sim_auto_convert(x, prob, np.array(classes), mask), ref_auto_convert(x, prob, classes, mask))


@pytest.mark.parametrize("seed", range(4))
def test_neighbourhood_allow(seed):
    rng = np.random.default_rng(seed)
    land = rng.choice([1.0, 2.0, 3.0, np.nan], (23, 17), p=[0.4, 0.3, 0.2, 0.1])
    classes = [1.0, 2.0, 3.0]
    w5 = rng.random((5, 3))
    rules = [(1.0, 0.3, np.ones((3, 3))), (3.0, 0.1, w5), (7.0, 0.5, np.ones((3, 3))), (1.0, 0.5, np.ones((3, 3)))]
    _same(_core.sim_neighbourhood_allow(land, np.array(classes), rules), ref_neighbourhood(land, classes, rules))


@pytest.mark.parametrize(("seed", "stochastic"), [(s, st) for s in range(6) for st in (True, False)])
def test_ordered(seed, stochastic):
    rng = np.random.default_rng(seed)
    classes = [1.0, 2.0, 3.0]
    n = 300
    x = rng.choice(classes, n)
    tprob = rng.choice([np.nan, 0.0, 0.2, 0.2, 0.7, 1.0], (n, 3))
    area = np.array([np.sum(x == c) for c in classes])
    shift = rng.integers(-40, 41, 2)
    demand = np.array([area[0] + shift[0], area[1] + shift[1], area[2] - shift.sum()], dtype=float)
    order = list(rng.permutation(classes))
    r = _core.sim_ordered(tprob, x, demand, np.array(classes), np.array(order), stochastic, _core.RRandom(seed + 1))
    ref, short = ref_ordered(tprob, x, demand, classes, order, stochastic, _core.RRandom(seed + 1))
    _same(r["allocation"], ref)
    assert r["short"] == short and r["converged"] == (short == 0)


def test_ordered_nothing_to_do():
    x = np.array([1.0, 2.0, 2.0])
    r = _core.sim_ordered(np.ones((3, 2)), x, np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([2.0, 1.0]), True,
                          _core.RRandom(1))
    _same(r["allocation"], x)
    assert r["iterations"] == 0 and r["converged"]


@pytest.mark.parametrize("seed", range(6))
def test_approx_extrap(seed):
    rng = np.random.default_rng(seed)
    x = rng.choice([0.0, 1.0, 3.0, 4.5, 7.0, 7.0, 10.0], 5)
    if np.unique(x).size < 2:
        x[0], x[1] = 0.0, 2.0
    y = rng.random(5) * 100
    xout = np.concatenate([np.linspace(-5, 15, 41), x])
    _same(_core.sim_approx_extrap(x, y, xout), ref_approx_extrap(x, y, xout))


@pytest.mark.parametrize("seed", range(6))
def test_round_sum(seed):
    rng = np.random.default_rng(seed)
    x = rng.random((8, 5)) * rng.choice([1, 1000, 1e6])
    x[rng.random((8, 5)) < 0.2] = 0.0
    x[:, 0] += 1e-3
    for ncell in (7, 1000, 123457):
        _same(_core.sim_round_sum(x, ncell), ref_round_sum(x, ncell))
        assert (_core.sim_round_sum(x, ncell).sum(axis=1) == ncell).all()


@pytest.mark.parametrize("seed", range(5))
def test_markov_from_counts(seed):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 50, (5, 5)).astype(float)
    c[rng.integers(0, 5)] = 0  # a class without cells at the start
    p, kept = _core.sim_markov_from_counts(c)
    rp, rk = ref_from_counts(c)
    _same(p, rp)
    _same(kept, rk)


def test_markov_root_and_project():
    c = np.array([[80, 15, 5], [10, 85, 5], [2, 3, 95]], float)
    p = c / c.sum(axis=1, keepdims=True)
    root = _core.sim_markov_root(p, 4)
    np.testing.assert_allclose(np.linalg.matrix_power(root, 4), p, atol=1e-14)
    np.testing.assert_allclose(root.sum(axis=1), 1, atol=1e-14)
    _same(_core.sim_markov_root(p, 1), p)
    proj = _core.sim_markov_project(np.array([100.0, 50.0, 25.0]), root, 4)
    assert proj.shape == (5, 3)
    np.testing.assert_allclose(proj[-1], np.array([100.0, 50.0, 25.0]) @ p, rtol=1e-13)


def test_markov_root_singular():
    p = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    root = _core.sim_markov_root(p, 2)
    np.testing.assert_allclose(root @ root, p, atol=1e-12)


def test_markov_root_negative_eigenvalue():
    with pytest.raises(ValueError, match="negative eigenvalue"):
        _core.sim_markov_root(np.array([[0.3, 0.7], [0.65, 0.35]]), 3)


@pytest.mark.parametrize("fix", [False, True])
def test_woe_weights(fix):
    cases = [(0, 10, 50, 1000), (10, 10, 50, 1000), (5, 10, 0, 1000), (5, 10, 50, 50), (50, 100, 50, 1000),
             (3, 200, 50, 1000), (40, 60, 50, 1000), (1, 2, 3, 7)]
    for nb, b, n, a in cases:
        assert _core.sim_woe_weights(nb, b, n, a, fix) == ref_weights(float(nb), float(b), float(n), float(a), fix)


@pytest.mark.parametrize(("seed", "fix"), [(s, f) for s in range(8) for f in (False, True)])
def test_woe_generalise(seed, fix):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 120))
    possible = rng.integers(0, 400, n)
    executed = np.minimum(possible, rng.integers(0, 60, n))
    executed[rng.random(n) < 0.2] = 0
    for min_d, max_d, tol in ((1.0, 100000.0, 5.0), (50.0, 2000.0, 1.0), (1.0, 300.0, 15.0)):
        assert _core.sim_woe_generalise(possible, executed, min_d, max_d, tol, fix) == \
            ref_generalise(possible, executed, min_d, max_d, tol, fix)


def test_woe_normalise():
    rng = np.random.default_rng(3)
    s = rng.normal(0, 3, (3, 1000))
    s[:, :10] = np.nan
    s[0, 10:20] = 800.0  # exp would overflow without the shift
    out = _core.sim_woe_normalise(s)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.exp(s[:, 20:]) / (1 + np.exp(s[:, 20:]).sum(axis=0))
    np.testing.assert_allclose(out[:, 20:], expected, rtol=1e-12)
    assert np.isnan(out[:, :10]).all() and np.isfinite(out[:, 10:20]).all()
    np.testing.assert_allclose(out[0, 10:20], 1.0, rtol=1e-12)


def ref_observed_patches(a, b, f, t):
    from scipy import ndimage

    eight = np.ones((3, 3), dtype=bool)
    changed = (a == f) & (b == t)
    lab, n = ndimage.label(changed, eight)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    near = ndimage.binary_dilation(a == t, eight)
    touching = np.zeros(n + 1, dtype=bool)
    touching[np.unique(lab[near & changed])] = True
    exp, new = sizes[touching[1:]], sizes[~touching[1:]]

    def stats(s):
        return (float(s.mean()), float(s.var())) if s.size else (1.0, 0.0)

    (me, ve), (mn, vn) = stats(exp), stats(new)
    return {"cells": int(changed.sum()), "expansion": float(exp.sum() / sizes.sum()) if sizes.sum() else 0.0,
            "expansion_patches": int(exp.size), "expansion_mean": me, "expansion_variance": ve,
            "new_patches": int(new.size), "new_mean": mn, "new_variance": vn}


@pytest.mark.parametrize("seed", range(6))
def test_observed_patches(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(5, 300)), int(rng.integers(5, 300)))
    a = rng.choice([1.0, 2.0, 3.0, np.nan], shape, p=[0.5, 0.25, 0.2, 0.05])
    b = np.where(rng.random(shape) < rng.choice([0.05, 0.3, 0.6]), rng.choice([1.0, 2.0, 3.0], shape), a)
    for f, t in ((1.0, 2.0), (1.0, 3.0), (2.0, 3.0), (3.0, 1.0), (5.0, 1.0)):
        assert _core.sim_observed_patches(a, b, f, t) == ref_observed_patches(a, b, f, t)


def test_observed_patches_many():
    # thousands of patches: numpy's pairwise sums in blocks
    rng = np.random.default_rng(11)
    a = np.ones((900, 900))
    a[rng.random(a.shape) < 0.02] = 2.0
    b = np.where(rng.random(a.shape) < 0.08, 2.0, a)
    assert _core.sim_observed_patches(a, b, 1.0, 2.0) == ref_observed_patches(a, b, 1.0, 2.0)


# ----------------------------------------------------------------------------- MOLUSCE (phase 13 D)

def _sigmoid(x):
    x = np.maximum(-100, np.minimum(x, 100))
    return 1 / (1 + np.exp(-x))


def ref_molusce_total(factors, weights):
    out = []
    for per in weights:
        total, nan = None, None
        for f, (cats, w) in zip(factors, per):
            layer = np.zeros(f.shape)
            for c, x in zip(cats, w):
                layer[f == c] = x
            total = layer if total is None else total + layer
            nan = np.isnan(f) if nan is None else nan | np.isnan(f)
        out.append(np.where(nan, np.nan, total))
    return np.stack(out)


def ref_molusce_prediction_woe(s, classes, total):
    m = len(classes)
    pred = np.full(s.shape, np.nan)
    conf = np.full(s.shape, np.nan)
    for i, c in enumerate(classes):
        cells = s == c
        if not cells.any():
            continue
        cand = [i * m + j for j in range(m) if i * m + j in total]
        best = np.full(int(cells.sum()), -1000.0)
        second = np.full(best.size, -1000.0)
        arg = np.full(best.size, -1)
        for code in cand:
            w = total[code][cells]
            valid = ~np.isnan(w)
            up = valid & (w > best)
            mid = valid & ~up & (w > second)
            second = np.where(up, best, np.where(mid, w, second))
            best = np.where(up, w, best)
            arg = np.where(up, code, arg)
        pred[cells] = np.where(arg >= 0, np.asarray(classes, dtype=np.float64)[arg % m], np.nan)
        conf[cells] = np.trunc(100 * (_sigmoid(best) - _sigmoid(second)))
    return pred, conf


def ref_molusce_prediction_classes(s, given, classes):
    layers = np.stack([np.nan_to_num(given[c], nan=-np.inf) if c in given else np.full(s.shape, -np.inf)
                       for c in classes])
    order = np.argsort(-layers, axis=0, kind="stable")
    top = np.take_along_axis(layers, order[:1], 0)[0]
    sec = np.take_along_axis(layers, order[1:2], 0)[0] if len(classes) > 1 else np.zeros(s.shape)
    pred = np.asarray(classes, dtype=np.float64)[order[0]]
    with np.errstate(invalid="ignore", over="ignore"):
        conf = np.trunc(100 * (top - np.where(np.isfinite(sec), sec, 0)))
    bad = np.isnan(s) | ~np.isfinite(top)
    return np.where(bad, np.nan, pred), np.where(np.isnan(s), np.nan, np.where(bad, 0, conf))


def ref_molusce_ca_step(state, pred, conf, counts, rng):
    """landschaft's numpy step, with the cells of equal value taken in a stable order (the later first)."""
    conf_filled = np.nan_to_num(conf, nan=0.0)
    new = state.copy()
    present = sorted(set(np.unique(state[~np.isnan(state)]).tolist()))
    changed = []
    for f in present:
        for t in present:
            if f == t:
                continue
            n = counts.get((f, t), 0)
            if n == 0:
                continue
            places = (state == f) & (pred == t)
            n = min(n, int(places.sum()))
            if n > 0:
                c = (conf_filled + rng.random_sample(size=state.shape) / 1000) * places
                n = min(n, int((c > 0).sum()))
                if n > 0:
                    ind = c.argsort(axis=None, kind="stable")[-n:]
                    new.flat[ind] = t
            changed.append((f, t, n))
    return new, changed


@pytest.mark.parametrize("seed", range(6))
def test_molusce_total_and_prediction(seed):
    rng = np.random.default_rng(seed)
    shape = (23, 31)
    classes = [1.0, 2.0, 3.0]
    factors = [rng.integers(1, 5, shape).astype(float) for _ in range(3)]
    for f in factors:
        f[rng.random(shape) < 0.05] = np.nan
    codes = sorted(rng.choice(9, size=int(rng.integers(3, 10)), replace=False).tolist())
    weights = [[(sorted(rng.choice([1.0, 2.0, 3.0, 4.0], size=int(rng.integers(1, 5)), replace=False).tolist()),
                 np.round(rng.normal(0, 3, 4), 1).tolist()) for _ in factors] for _ in codes]
    weights = [[(c, w[:len(c)]) for c, w in per] for per in weights]
    ref = ref_molusce_total(factors, weights)
    got = _core.sim_molusce_total(np.stack([f.ravel() for f in factors]), weights)
    np.testing.assert_array_equal(got, ref.reshape(len(codes), -1))
    s = rng.choice([1.0, 2.0, 3.0, 4.0, np.nan], size=shape)
    total = {c: ref[j] for j, c in enumerate(codes)}
    p_ref, c_ref = ref_molusce_prediction_woe(s, classes, total)
    p, c = _core.sim_molusce_prediction_woe(s.ravel(), classes, codes, ref.reshape(len(codes), -1))
    np.testing.assert_array_equal(p, p_ref.ravel())
    np.testing.assert_array_equal(c, c_ref.ravel())


@pytest.mark.parametrize("seed", range(6))
def test_molusce_prediction_classes(seed):
    rng = np.random.default_rng(seed)
    shape = (19, 27)
    classes = [1.0, 2.0, 3.0][: 1 + seed % 3]
    s = rng.choice([1.0, 2.0, np.nan], size=shape)
    given = {}
    for c in classes:
        if rng.random() < 0.2:
            continue  # a class without a layer
        v = rng.choice([0.0, 0.25, 0.5, 1.0, np.nan, np.inf, -np.inf], size=shape, p=[.3, .2, .2, .1, .1, .05, .05])
        given[c] = v
    p_ref, c_ref = ref_molusce_prediction_classes(s, given, classes)
    layers = np.stack([given[c].ravel() if c in given else np.full(s.size, np.nan) for c in classes])
    p, c = _core.sim_molusce_prediction_classes(s.ravel(), layers, classes)
    np.testing.assert_array_equal(p, p_ref.ravel())
    np.testing.assert_array_equal(c, c_ref.ravel())


@pytest.mark.parametrize("seed", range(8))
def test_molusce_ca_step(seed):
    rng = np.random.default_rng(seed)
    shape = (21, 33)
    state = rng.choice([1.0, 2.0, 3.0, np.nan], size=shape, p=[.4, .3, .2, .1])
    pred = rng.choice([1.0, 2.0, 3.0, np.nan], size=shape)
    conf = rng.choice([0.0, 5.0, 50.0, 100.0, np.nan], size=shape)
    counts = {(1.0, 2.0): int(rng.integers(0, 40)), (2.0, 3.0): int(rng.integers(-2, 400)),
              (3.0, 1.0): int(rng.integers(1, 10)), (1.0, 3.0): 0, (4.0, 1.0): 5}
    r = np.random.RandomState(seed)
    new_ref, ch_ref = ref_molusce_ca_step(state, pred, conf, counts, r)
    ours = _core.NumpyRandomState(seed)
    new, ch = _core.sim_molusce_ca_step(state.ravel(), pred.ravel(), conf.ravel(),
                                        [(f, t, n) for (f, t), n in counts.items()], ours)
    np.testing.assert_array_equal(new, new_ref.ravel())
    assert [tuple(x) for x in ch] == ch_ref
    # the two generators went through the same draws
    np.testing.assert_array_equal(ours.random_sample(5), r.random_sample(5))



# ----------------------------------------------------------------------------- CLUMondo's yearly suitability

def ref_clumondo_neighbourhood(suit, nb, oldco, region, shape):
    suit = suit.copy()
    o2, r2 = oldco.reshape(shape), region.reshape(shape)
    shares = {u: _core.sim_clumondo_fraction(o2, r2, u, p["kernel"]).ravel()
              for u, p in nb.items() if p["weight"] > 0}
    zero = np.zeros(oldco.size, dtype=np.float32)
    inside = region >= 0
    for i, p in nb.items():
        if not p["weight"] > 0:
            continue
        t = np.full(oldco.size, np.float32(p["constant"]), dtype=np.float32)
        for k, b in p["coefficients"]:
            t = t + b * shares.get(k, zero)
        cells = inside & p["from"][np.clip(oldco, 0, None)]
        suit[i, cells] = suit[i, cells] + p["weight"] * t[cells]
    return suit


@pytest.mark.parametrize("seed", range(5))
def test_clumondo_neighbourhood_and_preference(seed):
    rng = np.random.default_rng(seed)
    shape, nsys = (17, 23), 4
    region = rng.choice([0, 0, 0, -9998, -9999], size=shape).astype(np.int32).ravel()
    oldco = np.where(region == -9999, -9999, rng.integers(0, nsys, region.size)).astype(np.int32)
    suit = rng.normal(0, 1, (nsys, region.size)).astype(np.float32)
    nb = {}
    for i in rng.choice(nsys, size=3, replace=False):
        r = int(rng.integers(1, 3))
        nb[int(i)] = {"weight": np.float32(rng.choice([0.0, 0.7, 1.3])),
                      "kernel": rng.choice([0.0, 0.5, 1.0], size=(2 * r + 1, 2 * r + 1)).astype(np.float32),
                      "from": rng.random(nsys) < 0.7, "constant": float(rng.normal()),
                      "coefficients": [(int(k), np.float32(rng.normal())) for k in rng.choice(nsys, 2, replace=False)]}
    ref = ref_clumondo_neighbourhood(suit, nb, oldco, region, shape)
    parts = [(i, float(p["weight"]), p["kernel"], [bool(x) for x in p["from"]], float(np.float32(p["constant"])),
              [(k, float(b)) for k, b in p["coefficients"]]) for i, p in nb.items()]
    got = _core.sim_clumondo_add_neighbourhood(suit, oldco.reshape(shape), region.reshape(shape), parts)
    np.testing.assert_array_equal(got, ref)
    v = rng.normal(0, 2, region.size).astype(np.float32)
    w = np.float32(rng.normal())
    inside = region >= 0
    want = suit[1].copy()
    want[inside] = want[inside] + v[inside] * w
    np.testing.assert_array_equal(_core.sim_clumondo_add_preference(suit[1], region, v, float(w)), want)
