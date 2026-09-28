# SPDX-License-Identifier: GPL-3.0-or-later
"""The random number generators behind the neutral landscape models.

R's generator (set.seed, runif, rnorm, sample.int) and std::mt19937 with
libstdc++'s distributions must give the same bits as R 4.4 and as the Rtools
g++ that builds NLMR (``tools/make_reference_nlm.py rng``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nlm_cases
from conftest import assert_same_bits
from landschaft import _core

REF = Path(__file__).parent / "data" / "reference" / "nlm" / "rng.npz"


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("no NLM generator reference (tools/make_reference_nlm.py rng)")
    return np.load(REF)


@pytest.mark.parametrize("name", sorted(nlm_cases.R_SEQUENCES))
@pytest.mark.parametrize("seed", nlm_cases.R_SEEDS)
def test_r_generator_matches_r(ref, name, seed):
    ops = nlm_cases.R_SEQUENCES[name]
    got = _core.nlm_r_sequence(seed, ops)
    assert len(got) == len(ops)
    for k, values in enumerate(got):
        expected = ref[f"r__{name}__{seed}__{k}"]
        np.testing.assert_array_equal(np.asarray(values, dtype=float), expected.astype(float), err_msg=str(ops[k][:4]))


@pytest.mark.parametrize("name", sorted(nlm_cases.STD_SEQUENCES))
def test_libstdcxx_distributions_match_gcc(ref, name):
    ops = nlm_cases.STD_SEQUENCES[name]
    for seed in nlm_cases.STD_SEQUENCE_SEEDS.get(name, nlm_cases.STD_SEEDS):
        got = _core.nlm_std_sequence(seed, ops)
        for k, values in enumerate(got):
            expected = ref[f"std__{name}__{seed}__{k}"]
            # the normal distribution takes log and sqrt of the C library
            assert_same_bits(values, expected, str(ops[k]))


def test_rnorm_does_not_depend_on_threads():
    ops = [("rnorm", 100_000, 0.0, 1.0), ("runif", 10, 0.0, 1.0)]
    one = _core.nlm_r_sequence(7, ops, n_threads=1)
    many = _core.nlm_r_sequence(7, ops, n_threads=4)
    for a, b in zip(one, many):
        np.testing.assert_array_equal(a, b)


def test_r_generator_properties():
    u, x, s = _core.nlm_r_sequence(3, [("runif", 10_000, 0.0, 1.0), ("rnorm", 10_000, 0.0, 1.0),
                                      ("sample", 50, 50, False, None)])
    assert np.all((u > 0) & (u < 1))
    assert abs(x.mean()) < 0.05 and abs(x.std() - 1) < 0.05
    assert sorted(s) == list(range(1, 51))
    # a == b and sd == 0 do not draw: the next value is the same
    ones, fives, after = _core.nlm_r_sequence(9, [("runif", 2, 1.0, 1.0), ("rnorm", 3, 5.0, 0.0),
                                                  ("runif", 1, 0.0, 1.0)])
    (first,) = _core.nlm_r_sequence(9, [("runif", 1, 0.0, 1.0)])
    assert list(ones) == [1.0, 1.0] and list(fives) == [5.0, 5.0, 5.0]
    assert after[0] == first[0]


def test_sample_errors():
    with pytest.raises(ValueError):
        _core.nlm_r_sequence(1, [("sample", 3, 4, False, None)])
    with pytest.raises(ValueError):
        _core.nlm_r_sequence(1, [("sample", 3, 2, False, [1.0, 0.0, 0.0])])
    with pytest.raises(ValueError):
        _core.nlm_r_sequence(1, [("sample", 3, 2, True, [1.0, 0.0])])


def test_std_shuffle_is_a_permutation():
    (p,) = _core.nlm_std_sequence(11, [("shuffle", 100_001)])
    assert np.array_equal(np.sort(p), np.arange(100_001))


RMATH = Path(__file__).parent / "data" / "reference" / "nlm" / "rmath.npz"


@pytest.fixture(scope="module")
def rmath():
    if not RMATH.exists():
        pytest.skip("no R arithmetic reference (tools/make_reference_nlm.py rmath)")
    return np.load(RMATH)


@pytest.mark.parametrize("key", sorted(nlm_cases.rmath_cases()))
def test_r_arithmetic_matches_r(rmath, key):
    # exp() and x^y (powl) of mingw-w64 and R's vector sqrt(), as on R for Windows
    fn, x, y = nlm_cases.rmath_cases()[key]
    assert_same_bits(_core.nlm_r_math(fn, x, y), rmath[key], key)


@pytest.mark.parametrize("key", sorted(nlm_cases.fft_cases()))
@pytest.mark.parametrize("parallel", [False, True])
def test_r_fft_matches_r(rmath, key, parallel):
    re, im, nrow, ncol, inverse = nlm_cases.fft_cases()[key]
    expected = rmath[key]
    # the terms of each value carry R's sin and cos: ulps of the largest value
    assert_same_bits(_core.nlm_r_fft(re, im, nrow, ncol, inverse, parallel), expected, key,
                     scale=np.abs(expected).max())


def test_r_fft_is_a_dft():
    rs = np.random.RandomState(1)
    z = rs.standard_normal((30, 21)) + 1j * rs.standard_normal((30, 21))
    col_major = z.T.ravel()
    out = _core.nlm_r_fft(col_major.real.copy(), col_major.imag.copy(), 30, 21)
    got = (out[0::2] + 1j * out[1::2]).reshape(21, 30).T
    np.testing.assert_allclose(got, np.fft.fft2(z), rtol=1e-12, atol=1e-12)
