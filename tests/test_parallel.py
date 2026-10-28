# SPDX-License-Identifier: GPL-3.0-or-later
"""Work shared with helpers (ls.parallel): the same results as on this computer alone."""

from __future__ import annotations

import io
import json
import tarfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import simulate_cases

import landschaft as ls
from landschaft._dist import _group, _net, _pairing, _tasks, _worker

S = ls.simulate


@pytest.fixture
def group(tmp_path, monkeypatch):
    monkeypatch.setenv("LANDSCHAFT_CONFIG_DIR", str(tmp_path / "config"))
    return _group.ensure_secret()


class _Helper:
    """A `landschaft worker` in a thread of this process, connected to ``pool``."""

    def __init__(self, pool, cache: Path, cores: int = 2):
        self.stop = threading.Event()
        self.lines: list = []
        self.thread = threading.Thread(target=_worker.run_worker, daemon=True, kwargs=dict(
            cores=cores, connect=f"127.0.0.1:{pool.port}", cache_dir=str(cache), stop=self.stop,
            out=self.lines.append))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=30)


def _pool(local: bool) -> ls.Helpers:
    return ls.helpers(local=local, local_threads=2, port=0, discovery=False, verbose=False)


def _tiles(folder: Path, n: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    forest = ls.load_raster(ls.example_data("input"))
    for k in range(n):
        ls.write_raster(forest[k * 200:k * 200 + 300, :300], folder / f"tile_{k}.tif")
    return folder


def _same_outputs(a: list, b: list) -> None:
    assert [r.ok for r in a] == [r.ok for r in b]
    for ra, rb in zip(a, b):
        assert [p.name for p in ra.outputs] == [p.name for p in rb.outputs]
        for pa, pb in zip(ra.outputs, rb.outputs):
            if pa.suffix == ".tif":
                np.testing.assert_array_equal(ls.load_raster(pa).values, ls.load_raster(pb).values)
            elif pa.suffix == ".csv":
                pd.testing.assert_frame_equal(pd.read_csv(pa), pd.read_csv(pb))


def test_batch_on_a_helper_gives_the_local_outputs(group, tmp_path):
    tiles = _tiles(tmp_path / "tiles")
    here = ls.batch("accounting", tiles, tmp_path / "here")
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.batch("accounting", tiles, tmp_path / "there", on=pool)
        assert any(not m["local"] and m["tasks"] == 3 for m in pool.machines)
    _same_outputs(here, there)
    meta = json.loads((tmp_path / "there" / "tile_0_accounting.json").read_text())
    assert all(Path(p).parent == tmp_path / "there" for p in meta["outputs"])
    assert not (tmp_path / "there" / _tasks.PARTIAL).exists()
    # done before: skipped, nothing sent
    with _pool(local=True) as pool:
        again = ls.batch("accounting", tiles, tmp_path / "there", on=pool)
    assert [r.seconds for r in again] == [0.0, 0.0, 0.0]


def test_mspa_batch_here_and_on_a_helper(group, tmp_path):
    tiles = _tiles(tmp_path / "tiles", 4)
    here = ls.mspa_batch(tiles, tmp_path / "here", edge_width=2)
    with _pool(local=True) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.mspa_batch(tiles, tmp_path / "there", edge_width=2, on=pool)
    for a, b in zip(here, there):
        assert a.output.name == b.output.name
        np.testing.assert_array_equal(ls.load_raster(a.output).values, ls.load_raster(b.output).values)


def test_a_broken_raster_fails_alone(group, tmp_path):
    tiles = _tiles(tmp_path / "tiles", 2)
    (tiles / "broken.tif").write_bytes(b"not a raster")
    with _pool(local=True) as pool:
        runs = ls.batch("accounting", tiles, tmp_path / "out", on=pool)
    by_name = {r.input.name: r for r in runs}
    assert by_name["broken.tif"].error and not by_name["tile_0.tif"].error
    with _pool(local=True) as pool, pytest.raises(RuntimeError, match="broken.tif"):
        ls.batch("accounting", tiles, tmp_path / "out2", on=pool, raise_errors=True)


def test_batch_on_needs_named_tools_and_json_params(group, tmp_path):
    tiles = _tiles(tmp_path / "tiles", 1)
    with _pool(local=True) as pool:
        with pytest.raises(TypeError, match="BATCH_TOOLS"):
            ls.batch(lambda x: x, tiles, tmp_path / "out", on=pool)
        with pytest.raises(TypeError, match="JSON"):
            ls.batch("accounting", tiles, tmp_path / "out", on=pool, classes=np.array([1, 2]))


def _model(n=40):
    land = np.ones((n, n))
    land[10:20, 10:20] = 2
    land[30:, :5] = 3
    pot = S.RasterPotential({1: np.full((n, n), 0.2), 2: np.random.default_rng(0).random((n, n)),
                             3: np.full((n, n), 0.1)})
    start = np.array([np.sum(land == c) for c in (1, 2, 3)], float)
    demand = S.AreaDemand(np.array([start + k * np.array([-50, 50, 0]) for k in range(4)]), classes=[1, 2, 3])
    fast = S.AreaDemand(np.array([start + k * np.array([-100, 100, 0]) for k in range(4)]), classes=[1, 2, 3])
    return S.Model(pot, demand, S.CARS(step_size=10)), simulate_cases.grid(land), fast


def test_scenarios_on_helpers_are_bit_identical(group, tmp_path):
    model, land, fast = _model()
    plan = {"trend": {}, "fast": {"demand": fast}}
    here = S.scenarios(model, land, replicates=3, seed=7, scenarios=plan, years=2020)
    with _pool(local=True) as pool, _Helper(pool, tmp_path / "cache"):
        there = S.scenarios(model, land, replicates=3, seed=7, scenarios=plan, years=2020, on=pool)
    np.testing.assert_array_equal(here.codes, there.codes)
    assert here.classes == there.classes and here.years == there.years
    pd.testing.assert_frame_equal(here.steps, there.steps)


def test_scenarios_with_dinamica_pieces(group, tmp_path):
    lu = ls.load_raster(S.example_data("sibuyan_landuse"))
    dr = ls.load_raster(S.example_data("sibuyan_drivers"))
    model = S.presets.dinamica(lu[0], lu[1], dr, steps=2, woe=250)
    here = S.scenarios(model, lu[0], dr, replicates=2, seed=3)
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = S.scenarios(model, lu[0], dr, replicates=2, seed=3, on=pool)
    np.testing.assert_array_equal(here.codes, there.codes)


def test_calibrate_grid_on_helpers(group, tmp_path):
    rng = np.random.default_rng(0)
    d1, d2 = rng.random((30, 30)), rng.random((30, 30))
    land = np.ones((30, 30))
    demand = S.AreaDemand(np.array([[900, 0], [800, 100]]), classes=[1, 2])

    def build(w):
        pot = S.RasterPotential({1: np.full((30, 30), 0.5), 2: w * d1 + (1 - w) * d2})
        return S.Model(pot, demand, S.Ordered(order=[2, 1], stochastic=False))

    observed = build(0.3).run(land, seed=1).maps[-1]
    grid = {"w": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    here = S.calibrate(build, land, observed, params=grid, method="grid", seed=1)
    with _pool(local=True) as pool, _Helper(pool, tmp_path / "cache"):
        there = S.calibrate(build, land, observed, params=grid, method="grid", seed=1, on=pool)
    assert there.best == {"w": 0.3}
    pd.testing.assert_frame_equal(here.history, there.history)
    np.testing.assert_array_equal(here.simulation.maps.values, there.simulation.maps.values)
    with _pool(local=True) as pool:
        with pytest.raises(ValueError, match="grid and random"):
            S.calibrate(build, land, observed, params={"w": (0.0, 1.0)}, on=pool)
        with pytest.raises(ValueError, match="built-in"):
            S.calibrate(build, land, observed, params=grid, method="grid", on=pool,
                        targets={"mine": lambda o, s: 0.0})


def test_the_cache_keeps_what_was_sent(group, tmp_path):
    tiles = _tiles(tmp_path / "tiles", 2)
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache") as helper:
        ls.batch("accounting", tiles, tmp_path / "a", on=pool)
        ls.batch("fos", tiles, tmp_path / "b", on=pool)
    received = [line for line in helper.lines if line.startswith(">")]
    assert len(received) == 2          # the second job found the tiles in the cache
    assert len(list((tmp_path / "cache").iterdir())) == 2


def test_another_group_is_refused(group, tmp_path):
    with _pool(local=True) as pool:
        session = _net.Session("127.0.0.1", pool.port)
        with pytest.raises(_net.ProtocolError, match="not in this group"):
            session.handshake(lambda salt: b"x" * 32, _worker.info(1))
        session.close()


def test_other_code_is_refused(group, tmp_path, monkeypatch):
    with _pool(local=True) as pool:
        session = _net.Session("127.0.0.1", pool.port)
        info = {**_worker.info(1), "code": "0" * 16}
        with pytest.raises(_net.ProtocolError, match="other code"):
            session.handshake(lambda salt: group, info)
        session.close()


def test_pairing_hands_over_the_secret(group, tmp_path):
    inv = _pairing.Invitation(minutes=1, port=0, discovery=False, code="K7QM-2XDP", out=lambda t: None)
    try:
        with pytest.raises(_net.ProtocolError, match="wrong invitation code"):
            _pairing.join("K7QM-2XDQ", connect=f"127.0.0.1:{inv.port}", out=lambda t: None)
        assert _pairing.join("k7qm 2xdp", connect=f"127.0.0.1:{inv.port}", out=lambda t: None) == group
    finally:
        inv.close()
    with pytest.raises(ValueError, match="invitation code"):
        _group.normalize_code("ABC-12")


def test_names_and_archives_stay_inside(tmp_path):
    for bad in ("../x", "/etc/passwd", "C:/x", "a\\b", "", "a/../../b"):
        with pytest.raises(ValueError):
            _tasks.check_name(bad)
    assert _tasks.check_name("scenario_0_data/start.tif") == "scenario_0_data/start.tif"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    buf.seek(0)
    with pytest.raises(ValueError, match="refusing"):
        _tasks.unpack(buf, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_only_registered_operations_run(tmp_path):
    from landschaft._dist import _ops

    with pytest.raises(ValueError, match="unknown operation"):
        _ops.run("os.system", {}, tmp_path, tmp_path / "out", 1)
    with pytest.raises(ValueError, match="not a tool"):
        _ops.run("batch", {"tool": "write_raster"}, tmp_path, tmp_path / "out", 1)
    with pytest.raises(ValueError):
        _ops.run("mspa", {"input": "../secret.tif"}, tmp_path, tmp_path / "out", 1)


def test_job_threads_split_the_cores():
    from landschaft._dist import Job

    assert Job("x", [None] * 100).threads_on(8) == 1
    assert Job("x", [None] * 2).threads_on(8) == 4
    assert Job("x", [None] * 2, threads=3).threads_on(8) == 3
    assert Job("x", [None] * 2, threads=16).threads_on(8) == 8


def test_load_result_reads_a_saved_result(tmp_path):
    forest = ls.load_raster(ls.example_data("input"))
    result = ls.fos(forest)
    parts = ls.load_result(result.save(tmp_path / "fos"))
    assert parts["result"] == "FOSResult"
    np.testing.assert_array_equal(parts["map"].values, np.asarray(result.map))
    assert isinstance(parts["stats"], pd.DataFrame)


def test_cli_commands():
    from landschaft._cli import build_parser

    p = build_parser()
    a = p.parse_args(["worker", "--join", "K7QM-2XDP", "--cores", "3", "--map", "Z:/=/mnt/nas"])
    assert (a.command, a.join, a.cores, a.map) == ("worker", "K7QM-2XDP", 3, ["Z:/=/mnt/nas"])
    assert p.parse_args(["helpers", "list"]).helpers_command == "list"
    assert _worker.map_path("Z:/data/a.tif", [("Z:/", "/mnt/nas")])[0].replace("\\", "/") == "/mnt/nas/data/a.tif"


def test_discovery_answers_only_its_group(group):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    gid = _group.group_id(group)
    responder = _net.Responder(lambda m, addr: {"kind": "coordinator", "group": gid, "port": 1234}
                               if m.get("group") == gid else None, port=port, host="127.0.0.1").start()
    try:
        found = _net.find({"kind": "helper", "group": gid}, lambda r: r.get("group") == gid, timeout=5,
                          port=port, targets=["127.0.0.1"])
        assert found is not None and found[1]["port"] == 1234
        assert _net.find({"kind": "helper", "group": "other"}, lambda r: True, timeout=1, port=port,
                         targets=["127.0.0.1"]) is None
    finally:
        responder.close()


# --------------------------------------------------------------------------- phase C: tools by name

def _equal(a, b):
    import dataclasses

    import xarray as xr

    if isinstance(a, xr.DataArray):
        np.testing.assert_array_equal(a.values, b.values)
        assert a.dims == b.dims
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif dataclasses.is_dataclass(a):
        assert type(a) is type(b)
        for f in dataclasses.fields(a):
            _equal(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _equal(a[k], b[k])
    else:
        assert a == b or (a != a and b != b)


def test_values_travel_without_pickle(tmp_path):
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import Point

    from landschaft._dist import _pack

    forest = ls.load_raster(ls.example_data("input"))[:60, :80]
    frame = pd.DataFrame({1: [1.5, np.nan], 2: [3, 4]}, index=pd.Index(["a", "b"], name="k"))
    multi = pd.DataFrame([[1.0, 2.0]], columns=pd.MultiIndex.from_tuples([("auc", "mean"), ("auc", "std")]))
    geo = gpd.GeoDataFrame({"v": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    value = {"map": forest, "frame": frame, "multi": multi, "geo": geo, "series": frame[1],
             "fad": ls.fad(forest, 7), (1, 2): [np.float32(2.5), float("inf"), None, ("t", 3)],
             "ds": xr.Dataset({"a": forest.astype(float)})}
    back = _pack.read(_pack.write(value, tmp_path / "v"))
    _equal(back["map"], value["map"])
    assert back["map"].rio.crs == forest.rio.crs and back["map"].rio.transform() == forest.rio.transform()
    pd.testing.assert_frame_equal(back["frame"], frame)
    pd.testing.assert_frame_equal(back["multi"], multi)
    assert back["geo"].equals(geo)
    pd.testing.assert_series_equal(back["series"], frame[1])
    _equal(back["fad"], value["fad"])
    assert back[(1, 2)][1] == float("inf") and back[(1, 2)][3] == ("t", 3)
    np.testing.assert_array_equal(back["ds"]["a"].values, value["ds"]["a"].values)
    with pytest.raises(_pack.Unpackable):
        _pack.write(lambda x: x, tmp_path / "f")


def test_map_on_a_helper_gives_the_local_results(group, tmp_path):
    forest = ls.load_raster(ls.example_data("input"))
    rasters = [forest[k * 150:k * 150 + 200, :250] for k in range(3)]
    here = ls.parallel.map("fad", rasters, window_size=13)
    seeds = [dict(nrow=40, ncol=40, seed=s) for s in range(3)]
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.parallel.map(ls.fad, rasters, window_size=13, on=pool)
        nlm_there = ls.parallel.map("nlm.random", seeds, on=pool)
    for a, b in zip(here, there):
        _equal(a, b)
    for a, b in zip(ls.parallel.map("nlm.random", seeds), nlm_there):
        _equal(a, b)
    saved = ls.parallel.map("nlm.random", seeds, output_dir=tmp_path / "out")
    assert [p.name for p in saved] == ["0.tif", "1.tif", "2.tif"]


def test_map_takes_registered_tools_and_data_only(group):
    with _pool(local=True) as pool:
        with pytest.raises(ValueError, match="not a tool"):
            ls.parallel.map(ls.write_raster, [1], on=pool)
        with pytest.raises(ValueError, match="file path"):
            ls.parallel.map("fad", ["C:/data/forest.tif"], on=pool)


def test_monte_carlo_on_a_helper(group, tmp_path):
    src = np.zeros((25, 25))
    src[5, 7] = 1
    src[18, 12] = 2
    f = ls.parallel.call("surface.gravity", src, exponent=ls.parallel.param(0))
    here = ls.uncertainty.monte_carlo(f, [2.0], [0.2], n=40, seed=3, keep=True)
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.uncertainty.monte_carlo(f, [2.0], [0.2], n=40, seed=3, keep=True, on=pool, chunk=7)
    np.testing.assert_array_equal(np.asarray(here.mean), np.asarray(there.mean))
    np.testing.assert_array_equal(np.asarray(here.sd), np.asarray(there.sd))
    with _pool(local=True) as pool, pytest.raises(TypeError, match="parallel.call"):
        ls.uncertainty.monte_carlo(lambda p: p[0], [1.0], [0.1], n=5, on=pool)


def test_sensitivity_runs_on_a_helper(group, tmp_path):
    src = np.zeros((25, 25))
    src[5, 7] = 1
    src[18, 12] = 2
    f = ls.parallel.call("surface.gravity", src, exponent=ls.parallel.param(0))
    d = ls.sensitivity.design("morris", {"exponent": (1.0, 3.0), "unused": (5.0, 15.0)}, n=6, seed=2)
    here = ls.sensitivity.evaluate(f, d, summary=np.mean)
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.sensitivity.evaluate(f, d, on=pool, chunk=5, summary=np.mean)
    np.testing.assert_array_equal(here, there)


def test_sdm_folds_on_a_helper(group, tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    data = ls.sdm.prepare(occ, env, clean=False, background=600)
    models = {"glm": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
              "glm_c": LogisticRegression(C=0.5, max_iter=500)}
    here = ls.sdm.fit(data, models)
    with _pool(local=True) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.sdm.fit(data, models, on=pool)
    pd.testing.assert_frame_equal(here.metrics, there.metrics)
    pd.testing.assert_frame_equal(here.summary, there.summary)
    pd.testing.assert_frame_equal(here.predictions, there.predictions)
    with _pool(local=True) as pool, pytest.raises(TypeError, match="scikit-learn"):
        ls.sdm.fit(data, {"x": object()}, on=pool)


def _problem():
    P = ls.prioritize
    cost = ls.load_raster(P.example_data("cost"))
    features = ls.load_raster(P.example_data("features"))
    return P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()


def test_prioritize_resolves_on_a_helper(group, tmp_path):
    P = ls.prioritize
    p = _problem()
    sol = p.solve()

    def make(t):
        return p.add_relative_targets(t)

    def criterion(q, s):
        return float(s.objective)

    here = P.eval_replacement_importance(p, sol)
    cal_here = P.calibrate(make, [0.1, 0.2, 0.3], criterion)
    with _pool(local=False) as pool, _Helper(pool, tmp_path / "cache"):
        there = P.eval_replacement_importance(p, sol, on=pool)
        cal_there = P.calibrate(make, [0.1, 0.2, 0.3], criterion, on=pool)
    np.testing.assert_array_equal(np.asarray(here), np.asarray(there))
    pd.testing.assert_frame_equal(cal_here.table, cal_there.table)
    for a, b in zip(cal_here.solutions, cal_there.solutions):
        np.testing.assert_array_equal(np.asarray(a.selection), np.asarray(b.selection))


# --------------------------------------------------------------------------- phase D: tiles

def _dem():
    return ls.nlm.fbm(130, 170, fract_dim=1.4, seed=11) * 300.0


def test_tiles_give_the_whole_map():
    dem = _dem()
    for tool, kw in (("surface.tri", {}), ("surface.terrain", dict(v="slope")), ("surface.curvature", {}),
                     ("surface.dissection", {})):
        whole = ls.parallel._registry.resolve(tool)(dem, **kw)
        tiles = ls.parallel.tiled(tool, dem, tile=45, **kw)
        np.testing.assert_array_equal(tiles.values, whole.values)
    # running sums along the rows start elsewhere in a tile: to about 1e-12
    tpi = ls.parallel.tiled("surface.tpi", dem, scale=7, halo=4, tile=45)
    np.testing.assert_allclose(tpi.values, ls.surface.tpi(dem, scale=7).values, rtol=1e-11, atol=1e-11)
    forest = ls.load_raster(ls.example_data("input"))[:300, :340]
    fad = ls.parallel.tiled("fad", forest, window_size=13, tile=90)            # a dict of its maps (not stats)
    whole = ls.fad(forest, 13)
    for k, v in fad.items():
        np.testing.assert_array_equal(v.values, getattr(whole, k).values)
    wm = ls.parallel.tiled("window_metrics", forest, 7, metric=["np"], level="landscape", tile=100)
    np.testing.assert_array_equal(wm["np"].values, ls.window_metrics(forest, 7, metric=["np"])["np"].values)
    g = ls.surface.geomorphons(dem, search=8)
    gt = ls.parallel.tiled("surface.geomorphons", dem, search=8, tile=40, n_workers=2)
    assert type(gt) is type(g)
    np.testing.assert_array_equal(gt.forms.values, g.forms.values)
    np.testing.assert_array_equal(gt.ternary.values, g.ternary.values)
    with pytest.raises(ValueError, match="halo"):
        ls.parallel.tiled("hydro.fill", dem)


def test_tiles_on_a_helper(group, tmp_path):
    dem = _dem()
    whole = ls.surface.geomorphons(dem, search=6)
    with _pool(local=True) as pool, _Helper(pool, tmp_path / "cache"):
        there = ls.parallel.tiled("surface.geomorphons", dem, search=6, tile=64, on=pool)
    np.testing.assert_array_equal(there.forms.values, whole.forms.values)
