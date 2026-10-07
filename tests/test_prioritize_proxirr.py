# SPDX-License-Identifier: GPL-3.0-or-later
"""ProxIRR against the CRAN package proxirr 0.5 (tests/optimize_cases.py, group "proxirr"), bit for bit;
its Marxan functions with their bugs fixed (features looked up by id, NA alphas left out)."""

from __future__ import annotations

import numpy as np
import optimize_cases as oc
import pandas as pd
import pytest
from optimize_reference import result

import landschaft as ls
from landschaft import _core

P = ls.prioritize


def _bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    np.testing.assert_array_equal(a[ok].view(np.uint64), b[ok].view(np.uint64))


@pytest.mark.parametrize("triage", [False, True])
def test_kernel_is_proxirr(triage):
    e = oc.proxirr_entries()
    ref = result("proxirr", "proxirr_entries")
    suffix = "_triage" if triage else ""
    a = _core.opt_proxirr_alpha(e["local"].to_numpy(), e["global"].to_numpy(), e["target"].to_numpy(), triage, 2)
    _bits(a, ref["alpha" + suffix])
    start = np.arange(0, len(e) + 1, 5, dtype=np.int64)  # 5 features per unit, in order
    b, g = _core.opt_proxirr_beta_gamma(start, np.asarray(a), 3)
    _bits(b, ref["beta" + suffix])
    _bits(g, ref["gamma" + suffix])


def test_long_products_are_r():
    """25 alphas per unit: R's prod() runs in 80 bits; a product in double differs from it."""
    e = oc.proxirr_many()
    ref = result("proxirr", "proxirr_many")
    a = np.asarray(_core.opt_proxirr_alpha(e["local"].to_numpy(), e["global"].to_numpy(), e["target"].to_numpy(),
                                           False, 2))
    _bits(a, ref["alpha"])
    b, g = _core.opt_proxirr_beta_gamma(np.arange(0, len(e) + 1, 25, dtype=np.int64), a, 2)
    _bits(b, ref["beta"])
    _bits(g, ref["gamma"])
    dbl = np.array([1.0 - np.prod(1.0 - row) for row in a.reshape(40, 25)])
    assert (dbl != ref["beta"]).any()


def _problem():
    glob = oc.proxirr_entries().groupby("feature")["global"].first().to_numpy()
    feats = oc.grid(oc._dyadic_features())
    return P.problem(oc.pu_cost(), feats).add_absolute_targets(np.array(oc.PROXIRR_TARGETS) * glob)


@pytest.mark.parametrize("triage", [False, True])
def test_problem_is_proxirr(triage):
    p = _problem()
    ref = result("proxirr", "proxirr_entries")
    suffix = "_triage" if triage else ""
    irr = P.eval_proxirr(p, triage=triage, n_threads=2)
    u = p.units
    _bits(irr.beta.values.ravel()[u.index], ref["beta" + suffix])
    _bits(irr.gamma.values.ravel()[u.index], ref["gamma" + suffix])
    _bits(irr.alpha.toarray().T.ravel(), ref["alpha" + suffix])
    e = oc.proxirr_entries()
    np.testing.assert_array_equal(irr.targets["global"], e.groupby("feature")["global"].first())
    assert np.isnan(irr.beta.values.ravel()[np.setdiff1d(np.arange(u.template.size), u.index)]).all()


def _marxan_check(irr, fixed):
    _bits(irr.alpha["global"], fixed["global"])
    _bits(irr.alpha["target"], fixed["target"])
    _bits(irr.alpha["alpha"], fixed["alpha"])
    _bits(irr.beta["beta"], fixed["beta"])
    _bits(irr.gamma["gamma"], fixed["gamma"])


def test_marxan_is_proxirr():
    ref = result("proxirr", "proxirr_marxan")
    irr = P.eval_proxirr_marxan(oc.marxan_pu(), oc.marxan_spec(), oc.marxan_puvspr(), n_threads=2)
    _bits(irr.alpha["alpha"], ref["alpha"])  # proxirr itself: ids 1..n in order
    _bits(irr.beta["beta"], ref["beta"])
    _bits(irr.gamma["gamma"], ref["gamma"])
    _marxan_check(irr, {k[len("fixed/"):]: v for k, v in ref.items() if k.startswith("fixed/")})
    tri = P.eval_proxirr_marxan(oc.marxan_pu(), oc.marxan_spec(), oc.marxan_puvspr(), triage=True)
    _marxan_check(tri, {k[len("fixed_triage/"):]: v for k, v in ref.items() if k.startswith("fixed_triage/")})


def test_marxan_ids_fixed():
    """Ids out of order and a feature missing from spec: proxirr gives every alpha NA (its lookup by
    position); the fixed reference and ours look the features up by id."""
    ref = result("proxirr", "proxirr_marxan_ids")
    irr = P.eval_proxirr_marxan(oc.marxan_pu_ids(), oc.marxan_spec_ids(), oc.marxan_puvspr_ids())
    _marxan_check(irr, {k[len("fixed/"):]: v for k, v in ref.items() if k.startswith("fixed/")})
    assert np.isnan(ref["alpha_as_released"]).all()
    assert irr.gamma["gamma"].iloc[-1] == 0.0 and irr.beta["beta"].iloc[-1] == 0.0  # a unit without features


def test_targets_and_errors():
    p = P.problem(oc.pu_cost(), oc.features()).add_relative_targets(0.3)
    irr = P.eval_proxirr(p.add_min_set_objective().add_boundary_penalties(1.0))
    u = p.units
    np.testing.assert_array_equal(irr.targets["global"], u.totals)
    np.testing.assert_array_equal(irr.targets["target"], 0.3 * u.totals)
    b = irr.beta.values.ravel()[u.index]
    g = irr.gamma.values.ravel()[u.index]
    assert ((b >= 0) & (b <= 1)).all() and (g >= b).all()
    assert "ProxIRR(" in repr(irr)
    assert "Plot" in repr(ls.plot(irr))
    assert "Plot" in repr(ls.plot(P.eval_proxirr_marxan(oc.marxan_pu(), oc.marxan_spec(), oc.marxan_puvspr())))
    with pytest.raises(ValueError, match="needs targets"):
        P.eval_proxirr(P.problem(oc.pu_cost(), oc.features()))
    with pytest.raises(ValueError, match="no targets"):
        P.eval_proxirr(p.add_max_utility_objective(100))
    with pytest.raises(ValueError, match="'prop' or a 'target'"):
        P.eval_proxirr_marxan(oc.marxan_pu(), pd.DataFrame({"id": [1.0]}), oc.marxan_puvspr())
    with pytest.raises(ValueError, match="non-negative"):
        _core.opt_proxirr_alpha([-1.0], [2.0], [1.0], False, 1)
    with pytest.raises(ValueError, match="greater than"):
        _core.opt_proxirr_alpha([3.0], [2.0], [1.0], False, 1)
