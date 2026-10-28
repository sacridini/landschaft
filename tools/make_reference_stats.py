# SPDX-License-Identifier: GPL-3.0-or-later
"""What pymannkendall (1.4), PySAL's esda (2.10) and libpysal (4.15) and scipy give, for landschaft's tests (the same
files as Terraflux's), written to tests/data/reference/stats/stats.npz and .json:

    python tools/make_reference_stats.py

scipy's normal distribution function (ndtr) at many points; the Mann-Kendall test and Sen's slope of 300 series
of 15 values (trends, ties, missing values) as pymannkendall's original_test computes them; and on a field of
20 x 25 cells (a smooth surface with noise), esda's global Moran's I, local Moran's I (its quadrants and moments)
and Getis and Ord's Gi* with libpysal's rook and queen contiguity of a lattice.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pymannkendall as mk
from scipy import ndimage
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "stats"


def main() -> None:
    warnings.simplefilter("ignore")
    from esda.getisord import G_Local
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import lat2W

    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    g = np.random.default_rng(21)
    x = np.concatenate([g.normal(size=4000) * 3, g.normal(size=500) * 0.3, [0.0, 8.0, -8.0, 40.0, -40.0]])
    arrays["ndtr/x"], arrays["ndtr/y"] = x, ndtr(x)

    n, t = 300, 15
    series = g.normal(size=(n, t)) + np.linspace(0, 1, t) * g.normal(size=(n, 1)) * 2
    series[::7] = np.round(series[::7])  # ties
    series[3::11, 4] = np.nan  # missing values
    out = np.full((n, 9), np.nan)
    for i in range(n):
        r = mk.original_test(series[i])
        out[i] = [r.s, r.var_s, r.z, r.p, r.Tau, r.slope, r.intercept, float(r.h),
                  {"increasing": 1, "decreasing": -1, "no trend": 0}[r.trend]]
    arrays["mk/series"], arrays["mk/out"] = series, out

    ny, nx = 20, 25
    field = ndimage.gaussian_filter(g.normal(size=(ny, nx)), 2) * 5 + g.normal(size=(ny, nx)) * 0.5 + 10
    arrays["field"] = field
    y = field.ravel()
    for kind, rook in [("rook", True), ("queen", False)]:
        w = lat2W(ny, nx, rook=rook)
        m = Moran(y, w, permutations=0)
        arrays[f"moran/{kind}"] = np.array([m.I, m.EI, m.VI_norm, m.z_norm, m.p_norm, m.VI_rand, m.z_rand, m.p_rand])
        lm = Moran_Local(y, w, permutations=0)
        arrays[f"lisa/{kind}/I"], arrays[f"lisa/{kind}/q"] = lm.Is, lm.q.astype(float)
        arrays[f"lisa/{kind}/EI"], arrays[f"lisa/{kind}/VI"] = lm.EI, lm.VI
        gs = G_Local(y, lat2W(ny, nx, rook=rook), transform="R", star=True, permutations=0)
        arrays[f"gi/{kind}/G"], arrays[f"gi/{kind}/Z"], arrays[f"gi/{kind}/p"] = gs.Gs, gs.Zs, gs.p_norm
    meta["shape"] = [ny, nx]

    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "stats.npz", **arrays)
    (REF / "stats.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(REF / "stats.npz", len(arrays))


if __name__ == "__main__":
    main()
