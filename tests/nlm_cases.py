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
    "pois": [
        ("rpois", 300, 4 * 2 * np.pi * (np.sqrt(2.0) / 2)),  # rpoislinetess(4) on the unit square
        ("rpois", 200, 3.5),
        ("rpois", 100, 0.3),
        ("rpois", 300, 10.0),
        ("rpois", 100, 150.5),
        ("rpois", 50, 9.99),
        ("rexp", 500),
        ("rpois", 100, 17.5),
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

# -- R's arithmetic ---------------------------------------------------------------


def rmath_cases() -> dict[str, tuple[str, np.ndarray, float]]:
    """key -> (function, inputs, exponent): R's exp(), x^y (powl on Windows) and
    vector sqrt() on the kinds of values the models feed them."""
    rs = np.random.RandomState(20260928)
    neg = -np.concatenate([rs.uniform(0, 1, 3000), rs.uniform(0, 60, 3000), rs.uniform(0, 745, 500),
                           np.arange(0, 200) / 10.0])
    grid = np.concatenate([np.arange(1, 5001, dtype=float), rs.uniform(1, 1e6, 2000), rs.uniform(1, 1.3, 1000)])
    return {
        "exp_neg": ("exp", neg, 0.0),
        "exp_pos": ("exp", rs.uniform(-5, 700, 2000), 0.0),
        "pow_m1": ("pow", grid, -1.0),
        "pow_m075": ("pow", grid, -0.75),
        "pow_m125": ("pow", grid, -1.25),
        "pow_m15": ("pow", grid, -1.5),
        "pow_m0999": ("pow", grid, -0.9995),
        "pow_half": ("pow", grid, 0.5),
        "pow_3": ("pow", rs.uniform(-10, 10, 2000), 3.0),
        "sqrt": ("sqrt", np.concatenate([grid, rs.uniform(0, 1, 2000)]), 0.0),
    }


def fft_cases() -> dict[str, tuple[np.ndarray, np.ndarray, int, int, bool]]:
    """key -> (real, imaginary, nrow, ncol, inverse): R's fft() on sizes with
    factors 2, 3, 4, 5, odd primes and square factors."""
    rs = np.random.RandomState(7)
    sizes = [(1, 7), (8, 8), (12, 10), (7, 13), (30, 45), (64, 100), (17, 1), (243, 5), (49, 18), (22, 121),
             (96, 60), (11, 1024)]
    out = {}
    for nrow, ncol in sizes:
        for inverse in (False, True):
            re = rs.standard_normal(nrow * ncol)
            im = rs.standard_normal(nrow * ncol) if (nrow + ncol) % 2 else np.zeros(nrow * ncol)
            out[f"fft_{nrow}x{ncol}_{'inv' if inverse else 'fwd'}"] = (re, im, nrow, ncol, inverse)
    return out


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
    # NLMR's C++ models (std::mt19937)
    "mpd_33": ("nlm_mpd(33, 33, user_seed = 1)", lambda ls: ls.nlm.mpd(33, 33, seed=1)),
    "mpd_100x60": ("nlm_mpd(100, 60, roughness = 0.7, user_seed = 5, verbose = FALSE)",
                   lambda ls: ls.nlm.mpd(100, 60, roughness=0.7, seed=5)),
    "mpd_torus": ("nlm_mpd(64, 64, torus = TRUE, user_seed = 3, verbose = FALSE)",
                  lambda ls: ls.nlm.mpd(64, 64, torus=True, seed=3)),
    "mpd_raw": ("nlm_mpd(17, 9, rand_dev = 2, rescale = FALSE, user_seed = 11, verbose = FALSE)",
                lambda ls: ls.nlm.mpd(17, 9, rand_dev=2, rescale=False, seed=11)),
    "mpd_torus_big": ("nlm_mpd(257, 200, roughness = 0.3, torus = TRUE, user_seed = 42)",
                      lambda ls: ls.nlm.mpd(257, 200, roughness=0.3, torus=True, seed=42)),
    "mpd_big": ("nlm_mpd(300, 250, user_seed = 2147483647, verbose = FALSE)",
                lambda ls: ls.nlm.mpd(300, 250, seed=2147483647)),
    "neigh_50x40": ("nlm_neigh(50, 40, p_neigh = 0.7, p_empty = 0.1, user_seed = 2)",
                    lambda ls: ls.nlm.neigh(50, 40, p_neigh=0.7, p_empty=0.1, seed=2)),
    "neigh_8": ("nlm_neigh(60, 30, p_neigh = 0.5, p_empty = 0.05, categories = 5, neighbourhood = 8, user_seed = 4)",
                lambda ls: ls.nlm.neigh(60, 30, p_neigh=0.5, p_empty=0.05, categories=5, neighbourhood=8, seed=4)),
    "neigh_props": ("nlm_neigh(40, 40, p_neigh = 0.6, p_empty = 0.2, proportions = c(0.5, 0.3, 0.2), "
                    "user_seed = 6, rescale = FALSE)",
                    lambda ls: ls.nlm.neigh(40, 40, p_neigh=0.6, p_empty=0.2, proportions=[0.5, 0.3, 0.2], seed=6,
                                            rescale=False)),
    "neigh_big": ("nlm_neigh(300, 250, p_neigh = 0.8, p_empty = 0.01, categories = 4, user_seed = 9)",
                  lambda ls: ls.nlm.neigh(300, 250, p_neigh=0.8, p_empty=0.01, categories=4, seed=9)),
    "rrc_50x40": ("nlm_randomrectangularcluster(50, 40, minl = 3, maxl = 7, user_seed = 1)",
                  lambda ls: ls.nlm.random_rectangular_cluster(50, 40, minl=3, maxl=7, seed=1)),
    "rrc_raw": ("nlm_randomrectangularcluster(120, 80, minl = 1, maxl = 10, user_seed = 9, rescale = FALSE)",
                lambda ls: ls.nlm.random_rectangular_cluster(120, 80, minl=1, maxl=10, seed=9, rescale=False)),
    "rrc_wide": ("nlm_randomrectangularcluster(30, 30, minl = 5, maxl = 30, user_seed = 2)",
                 lambda ls: ls.nlm.random_rectangular_cluster(30, 30, minl=5, maxl=30, seed=2)),
    # R-level cluster models
    "randomcluster_50x40": ("nlm_randomcluster(50, 40, p = 0.4, user_seed = 1)",
                            lambda ls: ls.nlm.random_cluster(50, 40, p=0.4, seed=1)),
    "randomcluster_ai": ("nlm_randomcluster(60, 45, p = 0.5, ai = c(0.3, 0.6, 0.1), user_seed = 7, rescale = FALSE)",
                         lambda ls: ls.nlm.random_cluster(60, 45, p=0.5, ai=[0.3, 0.6, 0.1], seed=7, rescale=False)),
    "randomcluster_8": ("nlm_randomcluster(70, 30, p = 0.3, ai = c(1, 1, 1, 1), neighbourhood = 8, user_seed = 3)",
                        lambda ls: ls.nlm.random_cluster(70, 30, p=0.3, ai=[1, 1, 1, 1], neighbourhood=8, seed=3)),
    "randomcluster_sparse": ("nlm_randomcluster(40, 40, p = 0.05, ai = c(0.2, 0.8), user_seed = 11)",
                             lambda ls: ls.nlm.random_cluster(40, 40, p=0.05, ai=[0.2, 0.8], seed=11)),
    "randomcluster_dense": ("nlm_randomcluster(80, 60, p = 0.62, ai = c(0.5, 0.25, 0.25), user_seed = 12)",
                            lambda ls: ls.nlm.random_cluster(80, 60, p=0.62, ai=[0.5, 0.25, 0.25], seed=12)),
    "curds": ("nlm_curds(c(0.5, 0.3, 0.6), c(32, 6, 2), user_seed = 1)",
              lambda ls: ls.nlm.curds([0.5, 0.3, 0.6], [32, 6, 2], seed=1)),
    "curds_wheyes": ("nlm_curds(c(0.5, 0.3, 0.6), c(32, 6, 2), wheyes = c(0.1, 0.05, 0.2), user_seed = 2)",
                     lambda ls: ls.nlm.curds([0.5, 0.3, 0.6], [32, 6, 2], wheyes=[0.1, 0.05, 0.2], seed=2)),
    "curds_small": ("nlm_curds(c(0.8, 0.5), c(3, 5), user_seed = 3)",
                    lambda ls: ls.nlm.curds([0.8, 0.5], [3, 5], seed=3)),
    # spectral models (NLMR 1.4's internal simulate_* functions; nlm_gaussianfield
    # and nlm_fbm only print a message), rescaled as the previous implementation
    "gaussianfield": ("NLMR:::util_rescale(terra::rast(NLMR:::simulate_gaussian_field(40, 50, 10, 5, 0.2, 0.5, seed = 1)))",
                      lambda ls: ls.nlm.gaussian_field(50, 40, seed=1)),
    "gaussianfield_raw": ("terra::rast(NLMR:::simulate_gaussian_field(33, 27, 4, 1, 0, 2, seed = 5))",
                          lambda ls: ls.nlm.gaussian_field(27, 33, autocorr_range=4, mag_var=1, nug=0, mean=2, seed=5,
                                                           rescale=False)),
    "gaussianfield_big": ("NLMR:::util_rescale(terra::rast(NLMR:::simulate_gaussian_field(120, 90, 25, 3, 0.5, 0, seed = 9)))",
                          lambda ls: ls.nlm.gaussian_field(90, 120, autocorr_range=25, mag_var=3, nug=0.5, mean=0,
                                                           seed=9)),
    "fbm": ("NLMR:::util_rescale(terra::rast(NLMR:::simulate_fbm(40, 50, 1, seed = 1)))",
            lambda ls: ls.nlm.fbm(50, 40, seed=1)),
    "fbm_rough": ("terra::rast(NLMR:::simulate_fbm(45, 33, 0.4, seed = 2))",
                  lambda ls: ls.nlm.fbm(33, 45, fract_dim=0.4, seed=2, rescale=False)),
    "fbm_smooth": ("NLMR:::util_rescale(terra::rast(NLMR:::simulate_fbm(64, 64, 1.7, seed = 3)))",
                   lambda ls: ls.nlm.fbm(64, 64, fract_dim=1.7, seed=3)),
    "fbm_linear": ("terra::rast(NLMR:::simulate_fbm(20, 30, 2, seed = 4))",
                   lambda ls: ls.nlm.fbm(30, 20, fract_dim=2, seed=4, rescale=False)),
    "fbm_big": ("NLMR:::util_rescale(terra::rast(NLMR:::simulate_fbm(150, 121, 1.25, seed = 5)))",
                lambda ls: ls.nlm.fbm(121, 150, fract_dim=1.25, seed=5)),
    # mosaic fields (spatstat)
    "mosaicfield": ("nlm_mosaicfield(50, 40, user_seed = 1)", lambda ls: ls.nlm.mosaic_field(50, 40, seed=1)),
    "mosaicfield_n3_raw": ("nlm_mosaicfield(30, 25, n = 3, mosaic_mean = 2, mosaic_sd = 1, user_seed = 4, rescale = FALSE)",
                           lambda ls: ls.nlm.mosaic_field(30, 25, n=3, mosaic_mean=2, mosaic_sd=1, seed=4,
                                                          rescale=False)),
    "mosaicfield_n1": ("nlm_mosaicfield(20, 20, n = 1, user_seed = 5)",
                       lambda ls: ls.nlm.mosaic_field(20, 20, n=1, seed=5)),
    "mosaicfield_big": ("nlm_mosaicfield(200, 150, n = 10, user_seed = 6)",
                        lambda ls: ls.nlm.mosaic_field(200, 150, n=10, seed=6)),
    "mosaicfield_collect_sum": ("nlm_mosaicfield(30, 20, n = 4, collect = TRUE, user_seed = 7)$mosaicfield_raster",
                                lambda ls: ls.nlm.mosaic_field(30, 20, n=4, collect=True, seed=7)["mosaic_field"]),
    "mosaicfield_collect_steps": (
        "nlm_mosaicfield(30, 20, n = 4, collect = TRUE, user_seed = 7)$steps",
        lambda ls: _hstack_steps(ls.nlm.mosaic_field(30, 20, n=4, collect=True, seed=7)["steps"])),
    "mosaicfield_infinit": ("nlm_mosaicfield(40, 30, n = 2, infinit = TRUE, user_seed = 8)$mosaicfield_inf",
                            lambda ls: ls.nlm.mosaic_field(40, 30, n=2, infinit=True, seed=8)["infinit"]),
    "mosaicfield_infinit_only": ("nlm_mosaicfield(35, 25, n = NA, infinit = TRUE, user_seed = 9, rescale = FALSE)",
                                 lambda ls: ls.nlm.mosaic_field(35, 25, n=None, infinit=True, seed=9, rescale=False)),
    # Voronoi mosaics (NLMR's value order comes from GEOS, through shapely)
    "mosaictess": ("nlm_mosaictess(50, 40, germs = 30, user_seed = 1)",
                   lambda ls: ls.nlm.mosaic_tess(50, 40, germs=30, seed=1)),
    "mosaictess_raw": ("nlm_mosaictess(100, 80, germs = 200, user_seed = 2, rescale = FALSE)",
                       lambda ls: ls.nlm.mosaic_tess(100, 80, germs=200, seed=2, rescale=False)),
    "mosaictess_planar": ("nlm_mosaictess(400, 300, germs = 500, user_seed = 4)",
                          lambda ls: ls.nlm.mosaic_tess(400, 300, germs=500, seed=4)),
    "mosaictess_few": ("nlm_mosaictess(30, 30, germs = 3, user_seed = 3)",
                       lambda ls: ls.nlm.mosaic_tess(30, 30, germs=3, seed=3)),
    "mosaicgibbs": ("nlm_mosaicgibbs(50, 40, germs = 20, R = 4, patch_classes = 3, user_seed = 1)",
                    lambda ls: ls.nlm.mosaic_gibbs(50, 40, germs=20, R=4, patch_classes=3, seed=1)),
    "mosaicgibbs_raw": ("nlm_mosaicgibbs(120, 90, germs = 150, R = 3, patch_classes = 5, user_seed = 2, rescale = FALSE)",
                        lambda ls: ls.nlm.mosaic_gibbs(120, 90, germs=150, R=3, patch_classes=5, seed=2,
                                                       rescale=False)),
    "mosaicgibbs_dense": ("nlm_mosaicgibbs(60, 60, germs = 100, R = 5, patch_classes = 4, user_seed = 3)",
                          lambda ls: ls.nlm.mosaic_gibbs(60, 60, germs=100, R=5, patch_classes=4, seed=3)),
}


def _hstack_steps(steps):
    """terra's as.matrix(wide = TRUE) of a multi-layer raster: layers side by side."""
    return np.hstack([steps.values[k] for k in range(steps.shape[0])])


MODEL_CASES: dict[str, str] = {key: r for key, (r, _) in MODELS.items()}
