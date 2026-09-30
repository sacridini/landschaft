# SPDX-License-Identifier: GPL-3.0-or-later
"""A slow, direct port of Zonation 5's flexibilityMap (flexibilitymap.cpp) in NumPy.

Dense feature vectors and Zonation's BalancingMethod sums as written in its
source, to check the C++ flexibility map on rankings whose curves Zonation itself
does not report (it reports those of the order before its last sort).
"""

from __future__ import annotations

import numpy as np

LN2 = 0.6931471805599453


def fast_log(x0: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, np.float64)
    n = x0.view(np.uint64)
    e = ((n >> np.uint64(52)).astype(np.int64) - 1023).astype(np.float64)
    m = ((n & np.uint64(0x800FFFFFFFFFFFFF)) | np.uint64(0x3FE0000000000000)).view(np.float64)
    big = m > 0.7071067811865476
    fe = np.where(big, e + 1.0, e)
    mp = np.where(big, m + (-0.0), m + m)
    x = mp + (-1.0)
    px = x * 0.00010187566380458093 + 0.497494994976747
    for c in (4.705791198788817, 14.498922534161093, 17.936867850781983, 7.708387337558854):
        px = px * x + c
    x2 = x * x
    x3 = x2 * x
    qx = mp + 10.287358718916746
    for c in (45.227914583753225, 82.98752669127767, 71.15447506185639, 23.125162012676533):
        qx = qx * x + c
    r = ((fe * LN2 + x) + x2 * (-0.5)) + (x3 * px) / qx
    r = np.where(1e307 < x0, np.inf, r)
    return np.where(x0 < 0, np.nan, r)


def fast_exp(x0: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, np.float64)
    t = x0 * 1.4426950408889634 + 0.5
    with np.errstate(invalid="ignore"):
        ti = np.where(np.abs(t) < 2**31, np.trunc(t), -(2**31)).astype(np.int64)
    n = ti - np.signbit(t.astype(np.float32)).astype(np.int64)
    x = x0 + n.astype(np.float64) * -0.6931471805599453
    xx = x * x
    p = ((0.00012617719307481058 * xx + 0.030299440770744195) * xx + 1.0) * x
    q = 2.0 + ((3.0019850513866446e-06 * xx + 0.002524483403496841) * xx + 0.22726554820815503) * xx
    y = (p + p) / (q - p) + 1.0
    bits = ((n.astype(np.uint64) & np.uint64(0xFFFFFFFF)) << np.uint64(52)) + np.uint64(0x3FF0000000000000)
    y = y * bits.view(np.float64)
    y = np.where(708 < x0, np.inf, y)
    return np.where(~(-708 <= x0), 0.0, y)


def fast_pow(x, y):
    return fast_exp(fast_log(x) * y)


def balancing_sum(rule, param, w, z, init, remaining, features):
    """BalancingMethod::sum(remaining, features) over all layers: (pos, neg)."""
    nxt = remaining - features
    if rule == "ABF":
        def b(r):
            rr = np.where(w >= 0, r, init - r)
            return fast_pow(np.maximum(rr, 0.0), z)
        v = (b(remaining) - b(nxt)) * w
    else:
        o = remaining - nxt
        od = np.where(w >= 0, remaining, init - nxt)
        with np.errstate(divide="ignore", invalid="ignore"):
            od = np.where(od > 0, o / od, 0.0)
        if rule != "CAZPW":
            od = od * np.abs(w)
        if rule == "CAZ2":
            od = od * od
        elif rule in ("CAZP", "CAZPW"):
            od = fast_pow(od, param)
        if rule == "CAZPW":
            od = od * np.abs(w)
        v = od
    pos = neg = 0.0
    for i in range(v.size):
        if w[i] >= 0:
            pos = max(pos, v[i]) if rule == "CAZMAX" else pos + v[i]
        else:
            neg = max(neg, v[i]) if rule == "CAZMAX" else neg + v[i]
    if rule == "CAZ2":
        pos, neg = np.sqrt(pos), np.sqrt(neg)
    elif rule in ("CAZP", "CAZPW") and param > 0:
        pos, neg = pos ** (1 / param), neg ** (1 / param)  # std::pow: equal to Eigen's generic_pow to ~1 ulp
    return pos, neg


def flexibility(res, max_error: float, cells_subset=None):
    """Up and down flexibility (fractions) of the cells at the positions ``cells_subset``."""
    ctx = res._ctx
    data = ctx["data"]
    L = data.layers
    N = data.cells
    scale = np.asarray(data.layer_scale)
    init = np.asarray(data.initial_levels)
    feats = np.stack([np.where(np.asarray(data.layer(k, False, -1.0)) > 0,
                               np.asarray(data.layer(k, False, 0.0)) * scale[k], 0.0) for k in range(L)], axis=1)
    rule, param = ctx["rule"], float(ctx["param"])
    w = np.asarray(ctx["weights"], np.float32).astype(np.float64)
    z = np.where(np.isnan(np.asarray(ctx["abf_z"], np.float32)), param,
                 np.asarray(ctx["abf_z"], np.float32).astype(np.float64))
    order = ctx["order_packed"].astype(np.int64)
    crank = res.feature_curves["rank"].to_numpy(np.float32)
    levels = res.feature_curves.iloc[:, 1:].to_numpy(np.float32)
    S = crank.size
    scalar = ctx["curve_scalar"]
    cost, hmask = ctx["cost"], ctx["hmask"]
    cost_prio = cost is not None and ctx["cost_in_ranking"]
    cmr = N / float(np.add.reduce(cost.astype(np.float64), dtype=np.float64)) if cost_prio else 0.0
    if cost_prio:
        total = 0.0
        for c in cost:
            total += float(c)
        cmr = 0.0 if total <= 0 else N / total
    ndr = 1.0 / N
    maxerr = max_error / 100.0

    def err(index, off):
        ti = index + off
        if ti < 0 or ti >= N:
            return maxerr
        a, b = order[index], order[ti]
        rank = np.float32(ti * ndr)
        i1 = min(int(float(rank) * scalar), S - 1)
        i2 = min(S - 1, i1 + 1)
        r = np.float32(crank[i2] - crank[i1])
        t = np.float32((rank - crank[i1]) / r) if r > 0 else np.float32(0)
        rem = ((levels[i2] - levels[i1]) * t + levels[i1]).astype(np.float32).astype(np.float64)
        tp, tn = balancing_sum(rule, param, w, z, init, rem, feats[a])
        rp, rn = balancing_sum(rule, param, w, z, init, rem, feats[b])
        if hmask is not None and hmask[a] != hmask[b]:
            return maxerr
        if cost_prio:
            za, zb = not cost[a] > 0, not cost[b] > 0
            if za != zb:
                return maxerr
            if not za:
                ca, cb = float(cost[a]) * cmr, float(cost[b]) * cmr
                tp, tn, rp, rn = tp / ca, tn * ca, rp / cb, rn * cb
        diff = abs((tp - tn) - (rp - rn))
        div = tp + tn
        if diff == 0:
            return 0.0
        if div <= 0:
            return maxerr
        return diff / div

    def upper(index, first, last, d):
        count = last - first
        it = 9
        while count > 0 and it > 0:
            it -= 1
            step = count // 2
            i = first + step
            if not (err(index, d * i) + err(index, d * (i + 1))) / 2 >= maxerr:
                first = i + 1
                count -= step + 1
            else:
                count = step
        return first

    positions = range(N) if cells_subset is None else cells_subset
    up, down = {}, {}
    for index in positions:
        cell = order[index]
        up[cell] = np.float32(upper(index, 1, N - index, 1) * ndr)
        down[cell] = np.float32(upper(index, 1, index, -1) * ndr)
    return up, down
