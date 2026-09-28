# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_nlm.py and the NLM tests."""

from __future__ import annotations

import numpy as np

# -- generators ------------------------------------------------------------------

R_SEEDS = [1, 42, 123456, -5, 0, 2147483647]

# Operations of landschaft._core.nlm_r_sequence (and their R equivalents).
_PROB5 = [0.1, 0.2, 0.3, 0.2, 0.2]
_PROB_TIES = [1.0, 2.0, 1.0, 2.0, 0.0, 1.0]
_PROB_WALKER = list(np.linspace(0.5, 3.0, 300))  # > 200 values: Walker's alias
R_SEQUENCES = {
    "unif_norm": [
        ("runif", 1000, 0.0, 1.0),
        ("rnorm", 10000, 0.0, 1.0),
        ("runif", 10, 3.0, 7.0),
        ("rnorm", 1000, 2.0, 0.5),
        ("runif", 5, 360.0, 360.0),  # a == b: no draw
    ],
    "sample": [
        ("sample", 10, 10, False, None),
        ("sample", 1000, 100, False, None),
        ("sample", 100, 1000, True, None),
        ("sample", 1, 1, False, None),
        ("sample", 65537, 3, False, None),
        ("sample", 5, 50, True, _PROB5),
        ("sample", 5, 3, False, _PROB5),
        ("sample", 6, 20, True, _PROB_TIES),
        ("sample", 6, 4, False, _PROB_TIES),
        ("sample", 300, 500, True, _PROB_WALKER),
        ("sample", 20_000_000, 5, False, None),  # sample2 (hash)
        ("sample", 3, 1, False, None),
    ],
}

STD_SEEDS = [1, 42, 2147483647, 4000000000]

# Operations of landschaft._core.nlm_std_sequence and of the probe.
STD_SEQUENCES = {
    "draws": [
        ("canonical", 2000),
        ("uniform_int", 500, 0, 9),
        ("uniform_int", 500, 3, 17),
        ("uniform_int", 500, 0, 999_999),
        ("uniform_int", 200, -5, 5),
        ("normal", 1001, 0.0, 1.0),
        ("normal_fresh", 1000, 0.0, 0.7),
        ("normal", 11, 2.0, 3.0),
    ],
    "shuffle": [
        ("shuffle", 1),
        ("shuffle", 2),
        ("shuffle", 101),
        ("shuffle", 2500),
    ],
    "shuffle_big": [
        ("shuffle", 65535),  # last size with two positions per draw
        ("shuffle", 65536),
    ],
}
# Sequences drawn with fewer seeds (large outputs).
STD_SEQUENCE_SEEDS = {"shuffle_big": [42]}

# -- models ----------------------------------------------------------------------

# key -> (R expression returning a SpatRaster, the same call in landschaft).
MODELS: dict[str, tuple[str, object]] = {
    # random / percolation
    "random_50x40": ("nlm_random(50, 40, user_seed = 1)", lambda ls: ls.nlm.random(50, 40, seed=1)),
    "random_7x13_raw": ("nlm_random(7, 13, user_seed = 42, rescale = FALSE)",
                        lambda ls: ls.nlm.random(7, 13, seed=42, rescale=False)),
    "random_300x200": ("nlm_random(300, 200, resolution = 5, user_seed = 2147483647)",
                       lambda ls: ls.nlm.random(300, 200, resolution=5, seed=2147483647)),
    "percolation_30x20": ("nlm_percolation(30, 20, prob = 0.4, user_seed = 2)",
                          lambda ls: ls.nlm.percolation(30, 20, prob=0.4, seed=2)),
    "percolation_100x100": ("nlm_percolation(100, 100, prob = 0.59, user_seed = 10)",
                            lambda ls: ls.nlm.percolation(100, 100, prob=0.59, seed=10)),
    # gradients
    "planar_45": ("nlm_planargradient(30, 20, direction = 45)",
                  lambda ls: ls.nlm.planar_gradient(30, 20, direction=45)),
    "planar_seed": ("nlm_planargradient(30, 20, user_seed = 7)", lambda ls: ls.nlm.planar_gradient(30, 20, seed=7)),
    "planar_300_raw": ("nlm_planargradient(25, 40, direction = 300, rescale = FALSE)",
                       lambda ls: ls.nlm.planar_gradient(25, 40, direction=300, rescale=False)),
    "planar_90": ("nlm_planargradient(64, 3, direction = 90)", lambda ls: ls.nlm.planar_gradient(64, 3, direction=90)),
    "edge_120": ("nlm_edgegradient(30, 20, direction = 120)", lambda ls: ls.nlm.edge_gradient(30, 20, direction=120)),
    "edge_seed": ("nlm_edgegradient(40, 30, user_seed = 3)", lambda ls: ls.nlm.edge_gradient(40, 30, seed=3)),
    "edge_raw": ("nlm_edgegradient(33, 17, direction = 10, rescale = FALSE)",
                 lambda ls: ls.nlm.edge_gradient(33, 17, direction=10, rescale=False)),
    # distance gradients: lon/lat (<= 360 x 90 cells) and planar rasters
    "distance_lonlat": ("nlm_distancegradient(50, 40, origin = c(10, 15, 5, 8))",
                        lambda ls: ls.nlm.distance_gradient(50, 40, origin=(10, 15, 5, 8))),
    "distance_lonlat_raw": ("nlm_distancegradient(60, 45, origin = c(3, 3, 40, 42), rescale = FALSE)",
                            lambda ls: ls.nlm.distance_gradient(60, 45, origin=(3, 3, 40, 42), rescale=False)),
    "distance_lonlat_max": ("nlm_distancegradient(360, 90, origin = c(1, 2, 1, 1), rescale = FALSE)",
                            lambda ls: ls.nlm.distance_gradient(360, 90, origin=(1, 2, 1, 1), rescale=False)),
    "distance_planar": ("nlm_distancegradient(361, 50, origin = c(10, 20, 30, 40))",
                        lambda ls: ls.nlm.distance_gradient(361, 50, origin=(10, 20, 30, 40))),
    "distance_planar_raw": ("nlm_distancegradient(120, 91, origin = c(40, 41, 7, 7), rescale = FALSE)",
                            lambda ls: ls.nlm.distance_gradient(120, 91, origin=(40, 41, 7, 7), rescale=False)),
    # classification
    "classify_weighting": ("landscape_classify(nlm_random(40, 30, user_seed = 4), weighting = c(0.5, 0.3, 0.2))",
                           lambda ls: ls.nlm.classify(ls.nlm.random(40, 30, seed=4), weighting=[0.5, 0.3, 0.2])),
    "classify_weighting_na": (
        "landscape_classify(nlm_random(40, 30, user_seed = 5), weighting = c(0.2, NA, 0.3, NA))",
        lambda ls: ls.nlm.classify(ls.nlm.random(40, 30, seed=5), weighting=[0.2, None, 0.3, None])),
    "classify_thirds": ("landscape_classify(nlm_planargradient(30, 30, direction = 33), weighting = c(1, 1, 1))",
                        lambda ls: ls.nlm.classify(ls.nlm.planar_gradient(30, 30, direction=33), weighting=[1, 1, 1])),
    "classify_jenks": ("landscape_classify(nlm_random(25, 20, user_seed = 6), n = 4)",
                       lambda ls: ls.nlm.classify(ls.nlm.random(25, 20, seed=6), n=4)),
    "classify_jenks_gradient": ("landscape_classify(nlm_edgegradient(30, 30, direction = 70), n = 5)",
                                lambda ls: ls.nlm.classify(ls.nlm.edge_gradient(30, 30, direction=70), n=5)),
    "classify_real_land": (
        "landscape_classify(nlm_random(40, 30, user_seed = 8), "
        "real_land = landscape_classify(nlm_random(40, 30, user_seed = 9), weighting = c(0.2, 0.3, 0.5)), mask_val = 1)",
        lambda ls: ls.nlm.classify(
            ls.nlm.random(40, 30, seed=8),
            real_land=ls.nlm.classify(ls.nlm.random(40, 30, seed=9), weighting=[0.2, 0.3, 0.5]), mask_val=1)),
}

MODEL_CASES: dict[str, str] = {key: r for key, (r, _) in MODELS.items()}
