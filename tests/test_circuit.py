# SPDX-License-Identifier: GPL-3.0-or-later
"""Raster circuits (Circuitscape's advanced mode, ls.circuit.advanced)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "circuit"


def _dense(g, s, gr, four=False):
    """Direct dense solve of the same circuit (one component with one direct ground)."""
    ny, nx = g.shape
    nodes = [(y, x) for x in range(nx) for y in range(ny) if g[y, x] > 0]  # column-major, as Julia
    index = {c: k for k, c in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n))
    steps = [(0, 1), (1, 0), (-1, 0), (0, -1)] + ([] if four else [(1, 1), (-1, -1), (1, -1), (-1, 1)])
    for (y, x), k in index.items():
        for dy, dx in steps:
            j = index.get((y + dy, x + dx))
            if j is None:
                continue
            w = (g[y, x] + g[y + dy, x + dx]) / (2 * np.sqrt(2) if dy and dx else 2)
            L[k, j] -= w
            L[k, k] += w
    b = np.array([s[c] for c in nodes])
    grounded = [index[c] for c in nodes if gr[c] == np.inf]
    keep = [k for k in range(n) if k not in grounded]
    b[grounded] = 0
    v = np.zeros(n)
    v[keep] = np.linalg.solve(L[np.ix_(keep, keep)], b[keep])
    return nodes, L, v


def test_two_cells():
    r = ls.circuit.advanced(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, np.inf]]))
    # The 10 eps diagonal shift of Circuitscape's CHOLMOD path moves the last bits.
    np.testing.assert_allclose(r.current, [[1.0, 1.0]], rtol=1e-14)
    np.testing.assert_allclose(r.voltage, [[1.0, 0.0]], rtol=1e-14)


def test_series_and_diagonal_weights():
    # A chain: 2 A through edges of (1+3)/2 = 2 and (3+2)/2 = 2.5.
    r = ls.circuit.advanced(np.array([[1.0, 3.0, 2.0]]), np.array([[2.0, 0, 0]]), np.array([[0, 0, np.inf]]))
    np.testing.assert_allclose(r.voltage, [[2 / 2.5 + 2 / 2, 2 / 2.5, 0]], rtol=1e-13)
    # Only a diagonal joins the two cells: (x + y) / (2 sqrt 2).
    g = np.array([[1.0, np.nan], [np.nan, 1.0]])
    r = ls.circuit.advanced(g, np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, np.inf]]))
    np.testing.assert_allclose(r.voltage[0, 0], np.sqrt(2), rtol=1e-13)
    assert np.isnan(r.voltage[0, 1]) and r.current[0, 1] == 0
    # With four neighbours they are two components and nothing flows.
    r = ls.circuit.advanced(g, np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, np.inf]]), four_neighbors=True)
    np.testing.assert_array_equal(r.current, 0)


@pytest.mark.parametrize("four", [False, True])
def test_matches_dense_solve(four):
    rng = np.random.default_rng(3)
    g = rng.uniform(0.1, 2, (9, 11))
    s = np.where(rng.random(g.shape) < 0.3, rng.uniform(0, 1, g.shape), 0.0)
    gr = np.zeros(g.shape)
    gr[4, 5] = np.inf
    r = ls.circuit.advanced(g, s, gr, four_neighbors=four)
    nodes, L, v = _dense(g, s, gr, four)
    ours = np.array([r.voltage[c] for c in nodes])
    np.testing.assert_allclose(ours, v, rtol=1e-12, atol=1e-14)
    # Node currents: the larger of inflow and outflow over the branches.
    B = -L * (v[:, None] - v[None, :])  # current from node i to node j
    np.fill_diagonal(B, 0)
    node_current = np.maximum(np.clip(B, 0, None).sum(axis=0), np.clip(B, 0, None).sum(axis=1))
    np.testing.assert_allclose([r.current[c] for c in nodes], node_current, rtol=1e-9, atol=1e-12)
    # All the current reaches the ground (up to the 1e-8 branch threshold).
    np.testing.assert_allclose(r.current[4, 5], s.sum() - s[4, 5], rtol=1e-7)


def test_components_and_conflicts():
    g = np.ones((3, 7))
    g[:, 3] = np.nan
    s = np.zeros(g.shape)
    gr = np.zeros(g.shape)
    s[0, 0] = 1.0
    gr[2, 2] = np.inf
    s[2, 2] = 5.0  # a source on a ground is dropped
    s[1, 5] = 1.0  # no ground on the right
    r = ls.circuit.advanced(g, s, gr)
    np.testing.assert_allclose(r.current[2, 2], 1.0, rtol=1e-12)
    np.testing.assert_array_equal(r.current[:, 4:], 0)
    np.testing.assert_array_equal(r.voltage[:, 4:], 0)
    assert np.isnan(r.voltage[:, 3]).all()
    np.testing.assert_array_equal(r.current[:, 3], 0)


def test_finite_grounds_split_the_current():
    g = np.ones((1, 3))
    s = np.array([[0.0, 1.0, 0.0]])
    gr = np.array([[1.0, 0.0, 1.0]])
    r = ls.circuit.advanced(g, s, gr)
    np.testing.assert_allclose(r.current, [[0.5, 1.0, 0.5]], rtol=1e-13)


def test_dataarray_and_threads():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(5)
    g = rng.uniform(0.1, 1, (30, 40))
    g[:, 20] = np.nan
    s = rng.uniform(0, 1, g.shape)
    gr = np.zeros(g.shape)
    gr[5, 5] = gr[25, 30] = np.inf
    da = xr.DataArray(g, dims=("y", "x"), coords={"y": np.arange(30)[::-1] + 0.5, "x": np.arange(40) + 0.5})
    r1 = ls.circuit.advanced(da, s, gr, n_threads=1)
    r2 = ls.circuit.advanced(g, s, gr)
    assert isinstance(r1.current, xr.DataArray) and r1.current.name == "current"
    np.testing.assert_array_equal(r1.current.values, r2.current)
    np.testing.assert_array_equal(r1.voltage.values, r2.voltage)


def test_bad_inputs():
    with pytest.raises(ValueError, match="same shape"):
        ls.circuit.advanced(np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="negative"):
        ls.circuit.advanced(np.ones((2, 2)), np.ones((2, 2)), -np.ones((2, 2)))


def _advanced_cases():
    inputs = np.load(REF / "advanced_inputs.npz")
    return sorted({k.split("/")[0] for k in inputs.files})


@pytest.mark.reference
@pytest.mark.parametrize("name", _advanced_cases())
def test_advanced_matches_circuitscape(name):
    inputs = np.load(REF / "advanced_inputs.npz")
    ref = np.load(REF / "advanced.npz")
    r = ls.circuit.advanced(inputs[f"{name}/conductance"], inputs[f"{name}/source"], inputs[f"{name}/ground"],
                            four_neighbors=bool(inputs[f"{name}/four"]))
    chol = ref[f"{name}/current_cholmod"]
    top = chol.max()
    # Same cells carry current; values as CHOLMOD's to rounding.
    np.testing.assert_array_equal(r.current != 0, chol != 0)
    np.testing.assert_allclose(r.current, chol, rtol=1e-9, atol=1e-12 * top)
    # Circuitscape's default cg+amg stops at a relative residual of 1e-6.
    np.testing.assert_allclose(r.current, ref[f"{name}/current_cgamg"], rtol=0, atol=1e-5 * top)
