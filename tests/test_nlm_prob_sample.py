# SPDX-License-Identifier: GPL-3.0-or-later
"""R's weighted sampling (sample.int with prob) by exact faster searches.

``RRng::sample_int`` replaces the linear scans of R's ProbSampleNoReplace and
ProbSampleReplace (kept as the reference) by run-length closed forms and a
bisection. They must return the same indices and leave the generator in the
same state for every input, so every case is compared with the reference
draw for draw, plus three tie-heavy cases with R 4.4.2 itself.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from landschaft import _core

SEEDS = [1, 42, 123456, -5]


def draws(p, size, replace, seed, reference=False):
    idx, nxt = _core.nlm_prob_sample_test(np.asarray(p, dtype=float), size, replace, seed, reference)
    return np.asarray(idx), nxt


def assert_same_as_reference(p, size, replace=False, seeds=SEEDS):
    for seed in seeds:
        fast, fast_next = draws(p, size, replace, seed)
        ref, ref_next = draws(p, size, replace, seed, reference=True)
        assert fast.shape == (size,)
        np.testing.assert_array_equal(fast, ref, err_msg=f"seed {seed}")
        # the same number of unif_rand() calls
        assert fast_next == ref_next, f"seed {seed}"


def exact_dyadic(parts):
    """Small dyadic runs plus one large weight making the sum exactly 1.

    The small weights come first in the input, so R's sequential sum (and
    so the normalisation) is exact and the sampler sees the values as given;
    sorted, the large one comes first and the runs are added to a mass just
    below 1.
    """
    small = np.concatenate([np.full(c, v) for v, c in parts])
    total = sum(Fraction(v) * c for v, c in parts)
    big = 1.0 - float(total)
    assert Fraction(big) + total == 1
    p = np.concatenate([small, [big]])
    s = 0.0
    for x in p:
        s += x
    assert s == 1.0
    return p


def cos_rows(nrow, ncol, lat0, lat1):
    lat = np.linspace(lat0, lat1, nrow)
    return np.repeat(np.abs(np.cos(np.deg2rad(lat))), ncol)


# -- without replacement -----------------------------------------------------------


def test_random_weights_without_ties():
    rng = np.random.default_rng(1)
    assert_same_as_reference(rng.random(3000), 3000)
    assert_same_as_reference(rng.random(5000) ** 4, 700)


def test_long_runs_of_equal_weights():
    rng = np.random.default_rng(2)
    p = np.repeat(rng.random(1000), 1000)
    rng.shuffle(p)
    assert_same_as_reference(p, 120, seeds=SEEDS[:2])
    assert_same_as_reference(np.repeat(rng.random(100), 100), 10_000)
    # one value: the scan adds the same weight from 0 through every binade
    assert_same_as_reference(np.full(20_000, 0.37), 2_000)


def test_weights_per_raster_row():
    # |cos(latitude)| per row: runs of ncol cells; rows at opposite latitudes
    # share a value
    assert_same_as_reference(cos_rows(400, 250, -60.0, 75.0), 2_000)
    assert_same_as_reference(cos_rows(301, 40, -90.0, 90.0), 12_040)
    # a geodesic-like row width with few distinct values
    lat = np.round(np.linspace(-80, 80, 120), 0)
    assert_same_as_reference(np.repeat(np.cos(np.deg2rad(lat)) * 1000.0, 90), 5_000, seeds=SEEDS[:2])


@pytest.mark.parametrize(
    "parts",
    [
        [(2.0**-54, 4000)],  # half an ulp of the mass: ties, a even
        [(2.0**-54, 4002)],  # ties from an odd a
        [(3 * 2.0**-55, 4000)],  # rounds up, then into [1, 2) where it rounds down
        [(3 * 2.0**-54, 4000)],  # 1.5 ulps: ties with an odd quotient
        [(2.0**-54, 2000), (3 * 2.0**-56, 4000), (2.0**-60, 64_000), (5 * 2.0**-55, 800)],
    ],
)
def test_ties_and_binade_ends(parts):
    p = exact_dyadic(parts)
    assert_same_as_reference(p, min(p.size, 3000))


def test_dyadic_weights_in_any_order():
    rng = np.random.default_rng(3)
    for k in range(4):
        values = np.where(rng.random(40) < 0.5, 1.0, 3.0) * 2.0 ** -rng.integers(0, 40 + 10 * k, 40)
        p = np.repeat(values, rng.integers(1, 400, 40))
        rng.shuffle(p)
        assert_same_as_reference(p, p.size // 2)


def test_small_populations():
    for p in ([1.0], [1.0, 1.0], [0.3, 0.7], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 0.0, 1.0]):
        npos = sum(v > 0 for v in p)
        for size in range(npos + 1):
            assert_same_as_reference(p, size)


def test_zero_probabilities_and_all_drawn():
    rng = np.random.default_rng(4)
    p = np.repeat(rng.random(50), 40)
    p[rng.random(p.size) < 0.3] = 0.0
    assert_same_as_reference(p, int((p > 0).sum()))


def test_subnormal_weights():
    # once the large weights are drawn, the scan adds subnormals from 0
    p = np.concatenate([[1.0, 1.0], 1e-320 * (np.arange(60) % 3 + 1)])
    assert_same_as_reference(p, p.size)


def test_many_small_cases():
    rng = np.random.default_rng(7)
    for trial in range(1500):
        n = int(rng.integers(1, 300))
        kind = trial % 5
        if kind == 0:
            p = rng.random(n)
        elif kind == 1:
            p = np.repeat(rng.random(int(rng.integers(1, 12))), int(rng.integers(1, 60)))
        elif kind == 2:
            p = 2.0 ** -rng.integers(0, 60, n)
        elif kind == 3:
            p = np.repeat(np.where(rng.random(8) < 0.5, 1.0, 3.0) * 2.0 ** -rng.integers(0, 55, 8),
                          int(rng.integers(1, 80)))
            p[rng.random(p.size) < 0.2] = 0.0
        else:
            p = np.concatenate([rng.random(3), np.full(int(rng.integers(1, 300)), 1e-17),
                                1e-320 * rng.integers(1, 4, 20)])
        rng.shuffle(p)
        npos = int((p > 0).sum())
        if npos == 0:
            continue
        size = npos if rng.random() < 0.5 else int(rng.integers(0, npos + 1))
        seed = int(rng.integers(-2**31, 2**31))
        for replace in (False, True):
            fast = draws(p, size, replace, seed)
            ref = draws(p, size, replace, seed, reference=True)
            assert np.array_equal(fast[0], ref[0]) and fast[1] == ref[1], (trial, replace, seed)


# -- with replacement (bisection of the cumulative weights) ----------------------------


def test_with_replacement():
    rng = np.random.default_rng(5)
    # few large weights: R's linear search rather than Walker's alias
    p = np.concatenate([np.repeat(rng.random(5), 10), np.full(10_000, 1e-5), np.zeros(10)])
    assert_same_as_reference(p, 5000, replace=True)
    assert_same_as_reference([0.3, 0.7], 100, replace=True)
    assert_same_as_reference([1.0], 10, replace=True)
    assert_same_as_reference(exact_dyadic([(2.0**-54, 4000)]), 1000, replace=True)


# -- the closed form of the repeated addition --------------------------------------------


def test_repeated_add_matches_the_loop():
    rng = np.random.default_rng(6)

    def check(mass, w, c, rT):
        got = _core.nlm_repeated_add_test(mass, w, c, rT, False)
        ref = _core.nlm_repeated_add_test(mass, w, c, rT, True)
        assert got == ref, (mass.hex(), w.hex(), c, rT)

    for _ in range(3000):
        mass = math.ldexp(1 + rng.random(), int(rng.integers(-60, 3))) if rng.random() < 0.9 else 0.0
        u = math.ulp(mass) if mass > 0 else 2.0**-60
        kind = rng.integers(0, 6)
        if kind == 0:
            w = u / 2
        elif kind == 1:
            w = u * (int(rng.integers(0, 8)) + 0.5)
        elif kind == 2:
            w = u * rng.random() * 4
        elif kind == 3:
            w = math.ldexp(1.0, int(rng.integers(-70, 0)))
        elif kind == 4:
            w = 3 * math.ldexp(1.0, int(rng.integers(-70, 0)))
        else:
            w = rng.random() * 2.0 ** int(rng.integers(-30, 0))
        c = int(rng.integers(1, 50_000))
        end = _core.nlm_repeated_add_test(mass, w, c, math.inf, True)[1]
        r = rng.random()
        if r < 0.3:
            rT = mass + (end - mass) * rng.random()
        elif r < 0.6:
            # a mass of the loop itself, or a neighbour
            rT = _core.nlm_repeated_add_test(mass, w, int(rng.integers(1, c + 1)), math.inf, True)[1]
            rT = [rT, math.nextafter(rT, 0), math.nextafter(rT, 2)][int(rng.integers(0, 3))]
        elif r < 0.7:
            rT = 2 * end
        elif r < 0.8:
            rT = [mass, -1.0][int(rng.integers(0, 2))]
        else:
            # around the power of 2 the loop ends below
            p2 = 2.0 ** math.floor(math.log2(max(end, 1e-300)))
            rT = [p2, math.nextafter(p2, 0), math.nextafter(p2, 2)][int(rng.integers(0, 3))]
        check(mass, w, c, rT)
    # subnormal masses, into the normal range
    for _ in range(500):
        mass = float(rng.integers(0, 2**52)) * 5e-324 if rng.random() < 0.8 else 0.0
        w = float(rng.integers(0, 2**30)) * 5e-324
        c = int(rng.integers(1, 50_000))
        end = _core.nlm_repeated_add_test(mass, w, c, math.inf, True)[1]
        check(mass, w, c, mass + (end - mass) * rng.random())


# -- R ------------------------------------------------------------------------------------

# From R 4.4.2 (all positions drawn, then runif(1)):
#
#   cases <- list(
#     rows = rep(((0:199) %% 37 + 1) / 37, each = 100),
#     dyadic = rep(c(3, 1, 0.5, 3 * 2^-20), times = c(1000, 3000, 5000, 4000)),
#     interleaved = 1 / (1 + (0:9999) %% 7))
#   sizes <- c(rows = 20000, dyadic = 13000, interleaved = 5000)
#   set.seed(seed); x <- sample.int(length(p), sizes[[name]], prob = p); u <- runif(1)
#
# stored as the first and last 12 draws, sum(x * seq_along(x)) and u.
R_CASES = {
    "rows": (np.repeat((np.arange(200) % 37 + 1) / 37, 100), 20000),
    "dyadic": (np.repeat([3.0, 1.0, 0.5, 3 * 2.0**-20], [1000, 3000, 5000, 4000]), 13000),
    "interleaved": (1 / (1 + np.arange(10000) % 7), 5000),
}
R_EXPECTED = {
    ("rows", 1): ([3135, 13969, 6084, 8406, 10698, 12140, 787, 9419, 5893, 3515, 10685, 3371],
                  [14801, 93, 18554, 11161, 3708, 18580, 14870, 14864, 18518, 3891, 7481, 3784],
                  1988221129701, 0.2106027458794415),
    ("rows", 42): ([12081, 4540, 6794, 12437, 2160, 17361, 1781, 18246, 13165, 16722, 13766, 16693],
                   [18523, 3748, 7430, 18576, 11240, 18575, 3710, 14898, 87, 85, 11139, 14845],
                   1985053148653, 0.8776073509361595),
    ("dyadic", 1): ([798, 3213, 3421, 6374, 290, 7275, 8062, 1960, 3141, 924, 377, 503],
                    [12363, 9997, 10525, 12664, 12883, 10957, 10001, 11591, 9737, 10954, 9856, 10622],
                    688086802880, 0.8125113602727652),
    ("dyadic", 42): ([4529, 4354, 59, 5470, 1899, 1137, 7574, 326, 1914, 2197, 1889, 7936],
                     [9759, 11670, 11523, 12474, 9641, 11289, 11642, 12969, 11756, 11727, 12456, 10576],
                     688217973837, 0.8937958730384707),
    ("interleaved", 1): ([2374, 5594, 9907, 6054, 3165, 1700, 8210, 9446, 9194, 9710, 9521, 358],
                         [8231, 4877, 8949, 3870, 5804, 2871, 1306, 8291, 3669, 4847, 7225, 3755],
                         63139466881, 0.03549952548928559),
    ("interleaved", 42): ([5557, 8784, 7792, 1615, 542, 3369, 6612, 7190, 1998, 7122, 9102, 466],
                          [8192, 1613, 6429, 9481, 6092, 961, 4240, 8698, 9313, 2149, 187, 7442],
                          62703275088, 0.7314936732873321),
}


@pytest.mark.parametrize("name,seed", sorted(R_EXPECTED))
def test_matches_r(name, seed):
    p, size = R_CASES[name]
    first, last, checksum, u = R_EXPECTED[(name, seed)]
    x, nxt = draws(p, size, False, seed)
    assert x[:12].tolist() == first
    assert x[-12:].tolist() == last
    assert int((x * np.arange(1, size + 1)).sum()) == checksum
    assert nxt == u
