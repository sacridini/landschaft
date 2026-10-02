# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.MaxNet against R's maxnet 0.1.4 / glmnet 4.1-10 (tools/make_reference_maxnet.py).

With R's x87 maths (conftest.BITWISE) everything is compared bit for bit:
the penalties, the lambdas, every coefficient at the last lambda, the number
of coordinate-descent passes of the whole path, alpha, entropy, the feature
ranges and the four outputs with and without clamping. Elsewhere glmnet's
thresholds can branch differently after a last-bit difference of exp() or
log(), so the comparison allows a tolerance (and skips the ill-conditioned
case).
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from conftest import BITWISE

REF = Path(__file__).parent / "data" / "reference" / "sdm"
CASES = ["brad_default", "brad_lq", "brad_h_rm2", "brad_t", "brad_lqpht_rm05", "brad_num_lqp15", "brad_num_dup",
         "syn_np40", "syn_np12", "syn_np8", "syn_np90_noadd", "syn_qp", "big"]
TYPES = ("link", "exponential", "cloglog", "logistic")


@lru_cache(maxsize=None)
def _reference() -> tuple[dict, dict]:
    if not (REF / "maxnet.npz").exists():
        pytest.skip("no maxnet reference (run tools/make_reference_maxnet.py)")
    with np.load(REF / "maxnet.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads((REF / "maxnet.json").read_text(encoding="utf-8"))["arrays"]


def reference(case: str) -> dict:
    arrays, meta = _reference()
    out = {}
    for key, info in meta.items():
        if not key.startswith(case + "/"):
            continue
        name = key[len(case) + 1:]
        if info["kind"] == "text":
            out[name] = list(info["values"])
        else:
            out[name] = arrays[info.get("same_as", key)]
    return out


def big_centi() -> np.ndarray:
    """The data of the "big" case (tools/maxnet_reference.R) from R's generator: 100 x values."""
    rng = ls._core.RRandom(11)
    nb, npres, nv = 10000, 200, 20
    n = nb + npres
    z = rng.rnorm(n * 5).reshape(5, n).T
    a = rng.rnorm(5 * nv).reshape(nv, 5).T
    x = np.zeros((n, nv))
    for k in range(5):  # Z %*% A: reference BLAS dgemm, one product at a time
        x = x + z[:, k:k + 1] * a[k][None, :]
    x = x + rng.rnorm(n * nv, 0.0, 0.5).reshape(nv, n).T
    centi = np.rint(x * 100)
    xr = centi / 100 + 0.0
    score = xr[:, 0] - 0.5 * xr[:, 1] * xr[:, 1] / 5 + xr[:, 2] * 0.3
    idx = np.argsort(-(score + rng.rnorm(n)), kind="stable")[:npres]
    sel = np.concatenate([idx, rng.sample_int(n, nb) - 1])
    return centi[sel]


@lru_cache(maxsize=None)
def case_data(case: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    ref = reference(case)
    if "centi_colsums" in ref:
        centi = big_centi()
        np.testing.assert_array_equal(centi.sum(axis=0), ref["centi_colsums"])
        cont = centi / 100 + 0.0
    else:
        cont = ref["cont"].T
    X = pd.DataFrame(cont, columns=ref["cont_names"])
    for k, name in enumerate(ref["cat_names"]):
        levels = ref[f"levels/{name}"]
        X[name] = pd.Categorical.from_codes(ref["cat"][k], categories=levels)
    return X, ref["p"], ref


@lru_cache(maxsize=None)
def fitted(case: str) -> ls.sdm.MaxNet:
    X, p, ref = case_data(case)
    return ls.sdm.MaxNet(classes=ref["classes"][0], regmult=float(ref["regmult"][0]),
                         add_samples_to_background=bool(ref["add"][0])).fit(X, p)


def newdata(case: str) -> pd.DataFrame:
    X, _, ref = case_data(case)
    ex = pd.DataFrame(ref["new_cont"].T, columns=ref["cont_names"])
    for k, name in enumerate(ref["cat_names"]):
        ex[name] = pd.Categorical.from_codes(ref["new_cat"][k], categories=ref[f"levels/{name}"])
    return pd.concat([X.iloc[ref["pred_rows"]], ex], ignore_index=True)


def check(ours, theirs, what: str, rtol: float = 0.0, atol: float = 0.0) -> None:
    a = np.asarray(ours, dtype=float).ravel()
    b = np.asarray(theirs, dtype=float).ravel()
    assert a.shape == b.shape, f"{what}: {a.shape} vs {b.shape}"
    if BITWISE:
        same = (a == b) | (np.isnan(a) & np.isnan(b))
    else:
        same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not same.all():
        bad = np.flatnonzero(~same)
        raise AssertionError(f"{what}: {bad.size} of {a.size} differ; first at {bad[:5]}: "
                             f"ours {a[bad[:5]]} vs R {b[bad[:5]]}")


def tolerant(case: str) -> None:
    if not BITWISE and case == "brad_lqpht_rm05":
        pytest.skip("ill-conditioned: without R's x87 exp/log glmnet's path branches differently")


@pytest.mark.parametrize("case", CASES)
def test_model_matches_r(case):
    tolerant(case)
    m = fitted(case)
    ref = reference(case)
    check(m.penalty_factor_, ref["penalty"], "penalty.factor", rtol=1e-13)
    check(m.lambdas_, ref["lambda"], "lambda", rtol=1e-13)
    beta = np.zeros(len(m.feature_names_))
    index = {name: j for j, name in enumerate(m.feature_names_)}
    for name, b in m.betas_.items():
        beta[index[name]] = b
    scale = max(1.0, float(np.abs(ref["beta"]).max()))
    check(beta, ref["beta"], "beta[, 200]", atol=1e-9 * scale)
    check([m.alpha_], ref["alpha"], "alpha", rtol=1e-9)
    check([m.entropy_], ref["entropy"], "entropy", rtol=1e-9)
    if BITWISE:
        assert list(m.betas_) == ref["betas_names"]
        check(list(m.betas_.values()), ref["betas"], "betas")
        assert m.n_passes_ == int(ref["npasses"][0])
    check(m.feature_mins_, ref["featuremins"], "featuremins")
    check(m.feature_maxs_, ref["featuremaxs"], "featuremaxs")
    check(list(m.var_min_.values()), ref["varmin"], "varmin")
    check(list(m.var_max_.values()), ref["varmax"], "varmax")
    means = [m.sample_means_[v] for v in ref["cont_names"]]
    majority = [ref[f"levels/{v}"].index(m.sample_means_[v]) + 1 for v in ref["cat_names"]]
    check(means + majority, ref["samplemeans"], "samplemeans", rtol=1e-15)
    if "feature_names" in ref:
        assert m.feature_names_ == ref["feature_names"]
    assert m.lambda_ == m.lambdas_[-1]


@pytest.mark.parametrize("case", CASES)
def test_predictions_match_r(case):
    tolerant(case)
    m = fitted(case)
    ref = reference(case)
    nd = newdata(case)
    for clamp in (True, False):
        for t in TYPES:
            ours = m.predict_output(nd, type=t, clamp=clamp)
            theirs = ref[f"pred/{t}/{'clamp' if clamp else 'noclamp'}"]
            if t in ("cloglog", "logistic"):
                check(ours, theirs, f"{t} clamp={clamp}", atol=1e-9)
            else:
                check(ours, theirs, f"{t} clamp={clamp}", rtol=1e-9, atol=1e-12)


def test_estimator_api():
    X, p, _ = case_data("brad_lq")
    m = ls.sdm.MaxNet(classes="lq", type="logistic", threshold=0.3)
    assert m.get_params()["classes"] == "lq"
    m.set_params(regmult=1.0)
    m.fit(X, p, sample_weight=np.ones(len(p)))
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(m.predict(X), (proba[:, 1] >= 0.3).astype(int))
    np.testing.assert_array_equal(m.decision_function(X), m.predict_output(X, type="link"))
    np.testing.assert_array_equal(m.classes_, [0, 1])
    assert list(m.feature_names_in_) == list(X.columns)
    # columns by name, in any order; pickling keeps the model
    shuffled = X[X.columns[::-1]]
    np.testing.assert_array_equal(m.predict_proba(shuffled), proba)
    again = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(again.predict_proba(X), proba)
    # the same model with any number of threads
    one = ls.sdm.MaxNet(classes="lq", type="logistic", n_threads=1).fit(X, p)
    assert one.betas_ == m.betas_


def test_without_scikit_learn(monkeypatch):
    import importlib
    import sys

    import landschaft.sdm._maxnet as module

    original = module.MaxNet
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.base", None)
    try:
        bare = importlib.reload(module)
        m = bare.MaxNet(classes="lq")
        assert bare.MaxNet.__mro__[-1] is object and len(bare.MaxNet.__mro__) == 3
        assert m.get_params()["classes"] == "lq"
        assert repr(m.set_params(regmult=2.0)) == "MaxNet(classes='lq', regmult=2.0)"
        X, p, _ = case_data("brad_lq")
        np.testing.assert_array_equal(m.set_params(regmult=1.0).fit(X, p).predict_proba(X),
                                      fitted("brad_lq").predict_proba(X))
        with pytest.raises(ValueError, match="invalid parameter"):
            m.set_params(alpha=1)
    finally:
        monkeypatch.undo()
        importlib.reload(module)
        module.MaxNet = original  # the class ls.sdm.MaxNet is (pickling)


def test_missing_and_unseen_values():
    X, p, _ = case_data("syn_np40")
    m = ls.sdm.MaxNet().fit(X, p)
    nd = X.iloc[:4].copy()
    nd["k"] = nd["k"].cat.add_categories(["new"])
    nd.loc[nd.index[0], "a"] = np.nan  # a missing value: NaN (R drops the row)
    nd.loc[nd.index[1], "k"] = np.nan
    nd.loc[nd.index[2], "k"] = "new"  # an unseen level: all its indicators 0
    out = m.predict_output(nd)
    assert np.isnan(out[0])
    uses_k = any(name.startswith("categorical(k)") for name in m.betas_)
    assert np.isnan(out[1]) == uses_k
    assert np.isfinite(out[2]) and np.isfinite(out[3])
    with pytest.raises(ValueError, match="missing"):
        ls.sdm.MaxNet().fit(nd, np.array([1, 0, 1, 0]))


def test_numpy_input_and_sdmdata():
    X, p, _ = case_data("brad_num_lqp15")
    a = ls.sdm.MaxNet(classes="lqp", regmult=1.5).fit(X.to_numpy(), p)
    b = fitted("brad_num_lqp15")
    np.testing.assert_array_equal(list(a.betas_.values()), list(b.betas_.values()))
    assert not hasattr(a, "feature_names_in_")
    assert next(iter(a.betas_)).startswith("x")
    data = ls.sdm.SDMData(X, p.astype(np.int8), None)
    c = ls.sdm.MaxNet(classes="lqp", regmult=1.5).fit(data)
    np.testing.assert_array_equal(c.predict_proba(data), b.predict_proba(X))


def test_errors():
    X, p, _ = case_data("syn_np12")
    with pytest.raises(ValueError, match="presence"):
        ls.sdm.MaxNet().fit(X, np.where(p == 1, 2, 0))
    with pytest.raises(ValueError, match="classes"):
        ls.sdm.MaxNet(classes="lqz").fit(X, p)
    with pytest.raises(ValueError, match="hinge"):
        Xc = X.assign(a=1.0)
        ls.sdm.MaxNet(classes="h").fit(Xc, p)
    with pytest.raises(AttributeError, match="fit"):
        ls.sdm.MaxNet().predict(X)


def test_r_as_character():
    f = ls._core.sdm_r_as_character
    values = [1234567890123456, 1e15, 123456789012345, 100000, 123456, 0.0001, 0.00012, 1 / 3, 2 / 3 * 1e-10,
              1e5 + 0.1, -0.5, 1e-300, 5e22, 0.1 + 0.2, 100, 3e5, 1234567.1, 12345678901234567890.0, -1e5, 1e100,
              1.5e-5, 0.1234567890123456, 0.0]
    expected = ["1234567890123456", "1e+15", "123456789012345", "1e+05", "123456", "1e-04", "0.00012",
                "0.333333333333333", "6.66666666666667e-11", "100000.1", "-0.5", "1e-300", "5e+22", "0.3", "100",
                "3e+05", "1234567.1", "12345678901234567168", "-1e+05", "1e+100", "1.5e-05", "0.123456789012346",
                "0"]
    assert [f(float(v)) for v in values] == expected
