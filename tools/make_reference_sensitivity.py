# SPDX-License-Identifier: GPL-3.0-or-later
"""What SALib (1.6) and scipy.stats.qmc give, for landschaft's tests (the same files as Terraflux's), written to
tests/data/reference/sensitivity/sensitivity.npz and .json:

    python tools/make_reference_sensitivity.py

scipy's Sobol' points (6 dimensions, scrambled with seed 7, and not scrambled); Morris's trajectories of five
factors (10 trajectories, 4 and 6 levels) and the analysis of a function of them with interactions; Saltelli's runs
of the Ishigami function's three factors (64 base points, with and without the second order) and Sobol's indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from scipy.stats import norm, qmc

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "sensitivity"


def ishigami(x):
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


def morris_function(x):
    return 2.0 * x[:, 0] + x[:, 1] ** 2 + 3.0 * x[:, 2] * x[:, 3] - 0.5 * x[:, 4]


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    arrays["qmc/scrambled"] = qmc.Sobol(d=6, scramble=True, seed=7).random(16)
    arrays["qmc/plain"] = qmc.Sobol(d=6, scramble=False).random(8)

    p = np.concatenate([np.random.default_rng(1).random(3000), [0.975, 0.995, 0.5, 1e-10, 1 - 1e-12, 0.13, 0.87]])
    arrays["ndtri/p"], arrays["ndtri/x"] = p, norm.ppf(p)

    problem = {"num_vars": 5, "names": ["a", "b", "c", "d", "e"],
               "bounds": [[0, 1], [-1, 2], [0, 10], [1, 3], [-5, 5]]}
    meta["morris/problem"] = problem
    for levels in (4, 6):
        x = morris_sample.sample(problem, 10, num_levels=levels, seed=3)
        y = morris_function(x)
        si = morris_analyze.analyze(problem, x, y, num_resamples=50, num_levels=levels, seed=5)
        arrays[f"morris/{levels}/x"] = x
        arrays[f"morris/{levels}/y"] = y
        for k in ("mu", "mu_star", "sigma", "mu_star_conf"):
            arrays[f"morris/{levels}/{k}"] = np.asarray(si[k], dtype=float)

    problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}
    meta["sobol/problem"] = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}
    for second in (True, False):
        tag = "second" if second else "first"
        x = sobol_sample.sample(problem, 64, calc_second_order=second, seed=11)
        y = ishigami(x)
        si = sobol_analyze.analyze(problem, y, calc_second_order=second, num_resamples=40, seed=13)
        arrays[f"sobol/{tag}/x"] = x
        arrays[f"sobol/{tag}/y"] = y
        for k in ("S1", "S1_conf", "ST", "ST_conf") + (("S2", "S2_conf") if second else ()):
            arrays[f"sobol/{tag}/{k}"] = np.asarray(si[k], dtype=float)

    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "sensitivity.npz", **arrays)
    (REF / "sensitivity.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(REF / "sensitivity.npz", len(arrays))


if __name__ == "__main__":
    main()
