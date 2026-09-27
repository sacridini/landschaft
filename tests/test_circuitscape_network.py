# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuitscape's network (graph) mode (ls.circuit.circuitscape_network_*).

``tests/data/reference/circuit/circuitscape_network.npz`` holds Circuitscape
6.1's outputs (CHOLMOD) for its own network test cases and synthetic ones
(``make_reference_circuitscape.py network``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft.circuit import circuitscape_ini, circuitscape_network_advanced, circuitscape_network_pairwise

REF = Path(__file__).parent / "data" / "reference" / "circuit"
CS = REF / "circuitscape"
INIS = sorted(p.relative_to(CS).as_posix() for d in ("network", "network_extra") for p in (CS / "input" / d).glob("*.ini"))


@pytest.fixture
def in_cs(monkeypatch):
    monkeypatch.chdir(CS)


def _outputs(r) -> dict:
    if r.resistances is not None:
        out = {"result.txt": r.resistances, "resistances.out": r.resistances}
        out.update({f"{k}_{p}_{q}.txt": t for (k, p, q), t in r.maps.items()})
        if r.cum_node_currents is not None:
            out["node_currents_cum.txt"] = r.cum_node_currents
            out["branch_currents_cum.txt"] = r.cum_branch_currents
        return out
    return {"result.txt": r.voltages, "voltages.txt": r.voltages, "node_currents.txt": r.node_currents,
            "branch_currents.txt": r.branch_currents}


@pytest.mark.reference
@pytest.mark.parametrize("ini", INIS)
def test_matches_circuitscape(ini, in_cs):
    z = np.load(REF / "circuitscape_network.npz")
    name = Path(ini).stem
    refs = {k.split("/", 1)[1]: z[k] for k in z.files if k.split("/", 1)[0] == name}
    assert refs, name
    ours = _outputs(circuitscape_ini(ini, write_outputs=False))
    # every output Circuitscape wrote, and no other
    assert set(refs) - {"resistances_3columns.out"} == set(ours), name
    for key, want in refs.items():
        if key not in ours:
            continue
        got = np.asarray(ours[key], dtype=float)
        want = np.asarray(want, dtype=float)
        if want.size == 0:
            assert got.size == 0, key
            continue
        got = got.reshape(want.shape) if got.size == want.size else got
        assert got.shape == want.shape, key
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12 * np.abs(want).max(), err_msg=f"{name} {key}")


def test_series_and_parallel():
    # 1 -2- 2 -1- 3, and the edge 1-2 twice (conductances add: 0.5 + 0.5)
    edges = [[1, 2, 2.0], [2, 3, 1.0], [2, 1, 2.0]]
    r = circuitscape_network_pairwise(edges, [1, 3, 2])
    np.testing.assert_allclose(r.resistances[1:, 1:], [[0, 2, 1], [2, 0, 1], [1, 1, 0]], rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(r.resistances[0, 1:], [1, 3, 2])
    # conductances instead of resistances
    r = circuitscape_network_pairwise([[1, 2, 4.0]], [1, 2], graph_is_resistance=False)
    assert r.resistances[1, 2] == pytest.approx(0.25, rel=1e-13)


def test_numbering_from_zero():
    a = circuitscape_network_pairwise([[0, 1, 1.0], [1, 2, 1.0]], [0, 2])
    b = circuitscape_network_pairwise([[1, 2, 1.0], [2, 3, 1.0]], [1, 3])
    np.testing.assert_array_equal(a.resistances, b.resistances)  # ids written from 1
    with pytest.raises(ValueError, match="from 2"):
        circuitscape_network_pairwise([[2, 3, 1.0]], [2, 3])


def test_disconnected_and_missing_points():
    # a zero-sum pair of lines drops the edge 3-4; node 5 has no edges
    edges = [[1, 2, 1.0], [3, 4, 1.0], [4, 3, -1.0], [2, 5, 0.0], [1, 1, 3.0]]
    r = circuitscape_network_pairwise(edges, [1, 2, 3, 4, 9, 2], graph_is_resistance=False)
    res = r.resistances[1:, 1:]
    assert res[0, 1] == pytest.approx(1.0) and res[1, 5] == 0 and res[0, 5] == pytest.approx(1.0)
    assert res[2, 3] == -1  # not connected any more
    assert np.all(res[4, [0, 1, 2, 3, 5]] == -1)  # node 9 is not in the graph
    assert np.all(np.diag(res) == 0)
    assert r.n_components == 4  # {1, 2}, {3}, {4}, {5}


def test_advanced_by_hand():
    # 1 A into node 1 of 1 -1- 2 -2- 3, node 3 grounded directly: V = 3, 2, 0
    r = circuitscape_network_advanced([[1, 2, 1.0], [2, 3, 2.0]], [[1, 1.0]], [[3, 0.0]])
    np.testing.assert_allclose(r.voltages, [[1, 3], [2, 2], [3, 0]], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(r.node_currents[:, 1], [1, 1, 1], rtol=1e-13)
    np.testing.assert_allclose(r.branch_currents, [[1, 2, 1], [2, 3, 1]], rtol=1e-13)
    # a ground of 1 ohm at node 3 instead: 1 V more everywhere
    r = circuitscape_network_advanced([[1, 2, 1.0], [2, 3, 2.0]], [[1, 1.0]], [[3, 1.0]])
    np.testing.assert_allclose(r.voltages[:, 1], [4, 3, 1], rtol=1e-13)
    with pytest.raises(ValueError, match="not in the graph"):
        circuitscape_network_advanced([[1, 2, 1.0]], [[5, 1.0]], [[2, 0.0]])


def _random_graph(n, chords, seed):
    rng = np.random.default_rng(seed)
    e = [[v, v + 1, rng.uniform(0.5, 3)] for v in range(1, n)]
    for _ in range(chords):
        a, b = rng.integers(1, n + 1, 2)
        e.append([a, b, rng.uniform(0.5, 3)])
    return np.array(e, dtype=float)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_effective_resistances_match_the_pseudo_inverse(seed):
    # the graph's own nested dissection against a dense solve
    edges = _random_graph(600, 900, seed)
    pts = np.random.default_rng(seed).choice(np.arange(1, 601), 7, replace=False)
    r = circuitscape_network_pairwise(edges, pts)
    n = 600
    L = np.zeros((n, n))
    for a, b, w in edges:
        a, b = int(a) - 1, int(b) - 1
        if a == b:
            continue
        L[a, b] -= 1 / w
        L[b, a] -= 1 / w
        L[a, a] += 1 / w
        L[b, b] += 1 / w
    # pinv(L) plus a constant, which cancels below; pinv itself may keep the null
    # space's eigenvalue (OpenBLAS) and lose the digits of the differences
    P = np.linalg.inv(L + 1.0 / n)
    idx = pts - 1
    want = P[idx][:, idx].diagonal()[:, None] + P[idx][:, idx].diagonal()[None, :] - 2 * P[idx][:, idx]
    np.testing.assert_allclose(r.resistances[1:, 1:], want, rtol=1e-9, atol=1e-12)


def test_threads_give_the_same_bits():
    edges = _random_graph(3000, 2000, 7)
    pts = [5, 900, 1777, 2999]
    a = circuitscape_network_pairwise(edges, pts, current_maps=True, voltage_maps=True, n_threads=1)
    b = circuitscape_network_pairwise(edges, pts, current_maps=True, voltage_maps=True)
    np.testing.assert_array_equal(a.resistances, b.resistances)
    np.testing.assert_array_equal(a.cum_node_currents, b.cum_node_currents)
    for k in a.maps:
        np.testing.assert_array_equal(a.maps[k], b.maps[k])
    src, gnd = [[5, 1.0], [900, 2.0]], [[2999, 0.0], [1777, 3.0]]
    x = circuitscape_network_advanced(edges, src, gnd, n_threads=1)
    y = circuitscape_network_advanced(edges, src, gnd)
    np.testing.assert_array_equal(x.voltages, y.voltages)
    np.testing.assert_array_equal(x.branch_currents, y.branch_currents)


def test_included_pairs(tmp_path):
    edges = _random_graph(50, 20, 3)
    f = tmp_path / "pairs.txt"
    f.write_text("mode include\n1 20\n20 40\n")
    r = circuitscape_network_pairwise(edges, [1, 20, 40, 7], included_pairs=f)
    # include mode drops the focal nodes it does not name (7) and solves only the pairs it names
    np.testing.assert_array_equal(r.resistances[0, 1:], [1, 20, 40])
    res = r.resistances[1:, 1:]
    assert res[0, 1] > 0 and res[1, 2] > 0 and res[0, 2] == -1
    f.write_text("mode exclude\n1 20\n")
    r = circuitscape_network_pairwise(edges, [1, 20, 40], included_pairs=f)
    res = r.resistances[1:, 1:]
    assert res[0, 1] == -1 and res[0, 2] > 0 and res[1, 2] > 0


def test_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(CS)
    ini = (CS / "input" / "network_extra" / "net1.ini").read_text()
    out = tmp_path / "out" / "net.out"
    ini = ini.replace("output_file = output/net1.out", f"output_file = {out.as_posix()}")
    (tmp_path / "net.ini").write_text(ini)
    r = circuitscape_ini(tmp_path / "net.ini")
    names = sorted(p.name for p in out.parent.iterdir())
    assert "net_resistances.out" in names and "net_resistances_3columns.out" in names
    assert "net_node_currents_cum.txt" in names and "net_branch_currents_cum.txt" in names
    p, q = next((p, q) for (k, p, q) in r.maps if k == "voltages")
    assert f"net_voltages_{p}_{q}.txt" in names and f"net_branch_currents_{p}_{q}.txt" in names
    np.testing.assert_allclose(np.loadtxt(out.parent / "net_resistances.out"), r.resistances, rtol=1e-15)
    np.testing.assert_allclose(np.loadtxt(out.parent / f"net_voltages_{p}_{q}.txt"), r.maps[("voltages", p, q)],
                               rtol=1e-15)
    assert "data_type = network" in out.read_text()


def test_top_level_names():
    assert ls.circuit.NetworkResult is not None
    assert callable(ls.circuit.circuitscape_network_advanced)
