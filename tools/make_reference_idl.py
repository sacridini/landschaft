# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of GTB tools that exist only as IDL code.

Usage::

    python tools/make_reference_idl.py [path/to/gdl.exe [path/to/spatcon64.exe [path/to/ggeo64.exe]]]

Runs the verbatim GTB code in ``tests/gtb_idl`` with GDL (or IDL) and writes
``tests/data/reference/idl_reference.json``: for every case, the SHA-256 of
each output map and the numbers of the GTB report. The test-suite compares
landschaft against this file, so it runs anywhere without GDL.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import scenes  # noqa: E402
from gtb_reference import (  # noqa: E402
    FAD_SCALES,
    find_gdl,
    find_ggeo,
    find_grayspatcon,
    find_gtb_fwtools,
    find_gtb_mspa,
    find_spatcon,
    run_gtb_accounting,
    run_gtb_accounting_scaled_classes,
    run_gtb_fad_change,
    run_gtb_fad_mscale,
    run_gtb_fos,
    run_gtb_isochrone,
    run_gtb_lm_delta,
    run_gtb_lm_heatmap,
    run_gtb_mspa_tile,
    run_gtb_spa,
)

OUT = ROOT / "tests" / "data" / "reference" / "idl_reference.json"
GTB_INPUT = ROOT / "src" / "landschaft" / "data" / "input.tif"

# (connectivity, thresholds, highlight_largest)
ACCOUNTING_PARAMS = [
    (8, (50, 250, 1000, 5000, 20000), False),
    (4, (50, 250, 1000, 5000, 20000), True),
    (8, (1, 10, 100), True),
    (4, (5,), False),
    (8, (3, 3, 40, 40, 900), True),
]


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def accounting_inputs() -> dict[str, np.ndarray]:
    with rasterio.open(GTB_INPUT) as src:
        inputs = {"gtb_input": src.read(1)}
    for kind in scenes.SCENES:
        inputs[f"{kind}_101"] = scenes.make(kind, 101)
    special = scenes.make("blobs", 202).copy()
    rng = np.random.default_rng(7)
    bg = special == 1
    special[bg & (rng.random(special.shape) < 0.2)] = 3
    special[bg & (rng.random(special.shape) < 0.1)] = 4
    inputs["special_bg_202"] = special
    return inputs


def gtb_input() -> np.ndarray:
    with rasterio.open(GTB_INPUT) as src:
        return src.read(1)


def with_special_background(data: np.ndarray, seed: int) -> np.ndarray:
    """Some background pixels recoded to 3 and 4 (special backgrounds)."""
    out = data.copy()
    rng = np.random.default_rng(seed)
    bg = out == 1
    out[bg & (rng.random(out.shape) < 0.2)] = 3
    out[bg & (rng.random(out.shape) < 0.1)] = 4
    return out


def fad_inputs() -> dict[str, np.ndarray]:
    """FAD-MS inputs (sides >= 250 pixels, as GTB requires for the 243 window)."""
    base = gtb_input()
    changed = base.copy()
    changed[300:500, 200:700] = 1
    changed[600:650, :] = 2
    changed[:30, :30] = 0
    return {
        "gtb_input": base,
        "gtb_input_changed": changed,
        "gtb_input_special": with_special_background(base, 1),
        "gtb_input_changed_special": with_special_background(changed, 2),
        "blobs_a": scenes.blobs(11, shape=(300, 420), scale=15, missing=0.03),
        "blobs_b": scenes.blobs(12, shape=(300, 420), scale=15, missing=0.03),
    }


FAD_CHANGE_PAIRS = [
    ("gtb_input", "gtb_input_changed"),
    ("gtb_input_special", "gtb_input_changed_special"),
    ("blobs_a", "blobs_b"),
]


def lm_inputs() -> dict[str, np.ndarray]:
    """Land-cover maps coded 0 missing, 1 agriculture, 2 forest, 3 developed."""
    rng = np.random.default_rng(3)
    inputs = {
        "random_a": rng.choice([0, 1, 2, 3], size=(300, 300), p=[0.05, 0.35, 0.35, 0.25]).astype(np.uint8),
        "random_b": rng.choice([0, 1, 2, 3], size=(300, 300), p=[0.05, 0.15, 0.55, 0.25]).astype(np.uint8),
    }
    for name, seed in (("smooth_a", 21), ("smooth_b", 22)):
        f = scenes._smooth_noise(np.random.default_rng(seed), (260, 310), 9)
        inputs[name] = np.digitize(f, np.quantile(f, [0.02, 0.4, 0.75])).astype(np.uint8)
    return inputs


LM_WINDOWS = {"random_a": (7, 13), "random_b": (13,), "smooth_a": (7, 27), "smooth_b": (27,)}
LM_DELTA_PAIRS = [(("random_a", 13), ("random_b", 13)), (("smooth_a", 7), ("smooth_b", 27))]


def isochrone_inputs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(resistance, marker) pairs: resistance 2 = foreground, 3-100 = background;
    marker 1 = object A, 2 = object B, 3 = missing."""
    rng = np.random.default_rng(31)
    cases = {}
    r = rng.integers(3, 101, (90, 120)).astype(np.uint8)
    r[rng.random(r.shape) < 0.1] = 2
    m = np.zeros(r.shape, np.uint8)
    m[5:8, 6:9] = 1
    m[80, 110] = 2
    cases["random"] = (r, m)
    m2 = m.copy()
    m2[45, 10:115] = 3  # a wall of missing pixels with a gap on the left
    m2[20:70, 60] = 3
    cases["random_walls"] = (r, m2)
    u = np.full((70, 90), 20, np.uint8)  # uniform: many equal-cost routes
    mu = np.zeros(u.shape, np.uint8)
    mu[10, 10] = 1
    mu[60, 75:78] = 2
    cases["uniform"] = (u, mu)
    base = gtb_input()[300:600, 200:600]
    rr = np.where(base == 2, 2, np.clip(3 + 97 * (base == 1) * rng.random(base.shape), 3, 100)).astype(np.uint8)
    mm = np.where(base == 0, 3, 0).astype(np.uint8)
    mm[20:24, 30:34] = 1
    mm[280, 380] = 2
    cases["gtb_input_crop"] = (rr, mm)
    return cases


def spa_inputs() -> dict[str, np.ndarray]:
    inputs = {"gtb_input": gtb_input()}
    for kind in scenes.SCENES:
        inputs[f"{kind}_101"] = scenes.make(kind, 101)
    return inputs


def gray_scene(seed: int, shape=(180, 240)) -> np.ndarray:
    """Grayscale cover map (0-100) with missing (255) and special backgrounds (103/104)."""
    rng = np.random.default_rng(seed)
    import scipy.ndimage as ndi

    cover = ndi.gaussian_filter(rng.random(shape), 6)
    cover = np.round((cover - cover.min()) / np.ptp(cover) * 100).astype(np.uint8)
    cover[rng.random(shape) < 0.03] = 255
    low = cover < 20
    cover[low & (rng.random(shape) < 0.1)] = 103
    cover[low & (rng.random(shape) < 0.05)] = 104
    return cover


def fos_inputs() -> dict[str, np.ndarray]:
    with rasterio.open(ROOT / "src" / "landschaft" / "data" / "gscinput.tif") as src:
        gsc = src.read(1)
    return {"gtb_input_special": with_special_background(gtb_input(), 1),
            "blobs_101": scenes.make("blobs", 101), "gscinput": gsc, "gray_7": gray_scene(7)}


# (input, fosclass, window, connectivity, gray threshold)
FOS_CASES = [
    ("gtb_input_special", "FAD_6class", 13, 8, 30),
    ("gtb_input_special", "FAD_5class", 27, 4, 30),
    ("gtb_input_special", "FED_6class", 13, 8, 30),
    ("gtb_input_special", "FED_5class", 7, 4, 30),
    ("gtb_input_special", "FAC_6class", 7, 8, 30),
    ("gtb_input_special", "FAC_5class", 13, 4, 30),
    ("blobs_101", "FAD_6class", 5, 8, 30),
    ("blobs_101", "FED_6class", 9, 8, 30),
    ("blobs_101", "FAC_5class", 11, 8, 30),
    ("gscinput", "FAD_6class", 13, 8, 30),
    ("gscinput", "FED_5class", 13, 8, 30),
    ("gray_7", "FAD_5class", 7, 4, 50),
    ("gray_7", "FED_6class", 11, 8, 40),
]


def mspa_tile_inputs() -> dict[str, np.ndarray]:
    shortcuts = scenes.make("blobs", 9).copy()  # tiles without background or without foreground
    shortcuts[:80, :100] = 0
    shortcuts[:80, 100:200] = 2
    shortcuts[80:160, :100] = np.where(np.arange(100) % 7 < 3, 0, 2)
    shortcuts[160:, :100] = np.where(np.arange(100) % 5 < 2, 0, 1)
    shortcuts[160:, 100:160] = 1
    strip = scenes.make("blobs", 21)
    return {"blobs_3": scenes.make("blobs", 3), "blobs_4": scenes.make("blobs", 4), "salt_5": scenes.make("salt", 5),
            "network_6": scenes.make("network", 6), "shortcuts": shortcuts, "strip_row": strip[:50, :],
            "strip_col": strip[:, :55], "gtb_input": gtb_input()}


# (input, tile size, overlap, MSPA parameters)
MSPA_TILE_CASES = [
    ("blobs_3", 60, 10, {}),
    ("blobs_4", 100, 20, {"connectivity": 4, "internal": False}),
    ("salt_5", 70, 30, {"transition": False}),
    ("network_6", 90, 11, {"edge_width": 2}),
    ("shortcuts", 60, 20, {}),
    ("shortcuts", 80, 10, {"connectivity": 4}),
    ("shortcuts", 45, 4, {"internal": False}),
    ("strip_row", 60, 16, {}),
    ("strip_col", 60, 16, {}),
    ("gtb_input", 300, 60, {}),
]


def float32_hex(a: np.ndarray) -> str:
    return np.asarray(a, dtype="<f4").tobytes().hex()


def main(exe: Path, spatcon: Path, ggeo: Path | None = None) -> None:
    cases = []
    for name, data in accounting_inputs().items():
        for conn, thresholds, highlight in ACCOUNTING_PARAMS:
            ref = run_gtb_accounting(exe, data, thresholds, connectivity=conn, highlight_largest=highlight)
            cases.append({
                "input": name,
                "input_sha256": digest(data),
                "connectivity": conn,
                "thresholds": list(thresholds),
                "highlight_largest": highlight,
                "sha256": {k: digest(ref[k]) for k in ("classes", "ids", "area")},
                "stats": ref["stats"],
            })
            print("accounting", name, conn, thresholds, highlight)
        cases.append({
            "input": name,
            "input_sha256": digest(data),
            "scaled_classes": run_gtb_accounting_scaled_classes(exe, data),
        })

    fad_cases, fad_change_cases = [], []
    inputs = fad_inputs()
    for name, data in inputs.items():
        ref = run_gtb_fad_mscale(exe, spatcon, data)
        fad_cases.append({
            "input": name,
            "input_sha256": digest(data),
            "sha256": {str(k): digest(m) for k, m in ref["maps"].items()},
            "fad_av": ref["fad_av"].tolist(),
            "fadru_av": ref["fadru_av"].tolist(),
        })
        print("fad_mscale", name)
    for name_a, name_b in FAD_CHANGE_PAIRS:
        ref = run_gtb_fad_change(exe, spatcon, inputs[name_a], inputs[name_b])
        fad_change_cases.append({
            "a": name_a,
            "b": name_b,
            "change": ref["change"].tolist(),
            "perc": np.nan_to_num(ref["perc"], nan=-1.0).tolist(),
            "sums": np.nan_to_num(ref["sums"], nan=-1.0).tolist(),
            "delta_fad_av": ref["delta_fad_av"].tolist(),
            "delta_avcon": ref["delta_avcon"].tolist(),
        })
        print("fad_change", name_a, name_b)

    lm_cases, lm_delta_cases = [], []
    heatmaps = {}
    for name, data in lm_inputs().items():
        for window in LM_WINDOWS[name]:
            ref = run_gtb_lm_heatmap(exe, spatcon, data, window)
            heatmaps[name, window] = ref["hn"]
            lm_cases.append({
                "input": name,
                "input_sha256": digest(data),
                "window": window,
                "im103_sha256": digest(ref["im103"]),
                "hn": float32_hex(ref["hn"]),
                "csv": ref["csv"],
            })
            print("lm_heatmap", name, window)
    for (name_a, win_a), (name_b, win_b) in LM_DELTA_PAIRS:
        ref = run_gtb_lm_delta(exe, heatmaps[name_a, win_a], heatmaps[name_b, win_b], str(win_a), str(win_b))
        lm_delta_cases.append({"a": [name_a, win_a], "b": [name_b, win_b], "hn": float32_hex(ref["hn"]),
                               "csv": ref["csv"]})
        print("lm_delta", name_a, name_b)

    spa_cases = []
    for name, data in spa_inputs().items():
        for variant in (2, 3, 5, 6):
            ref = run_gtb_spa(exe, data, f"spa{variant}")
            spa_cases.append({"input": name, "variant": variant, "input_sha256": digest(data),
                              "sha256": digest(ref["map"]), "stats": ref["stats"]})
            print("spa", name, variant)

    fos_cases = []
    grayspatcon = find_grayspatcon()
    if grayspatcon is None:
        print("grayspatcon not found (LANDSCHAFT_GRAYSPATCON_EXE): FOS references are not generated")
    else:
        inputs = fos_inputs()
        for name, fosclass, window, conn, grayt in FOS_CASES:
            data = inputs[name]
            ref = run_gtb_fos(exe, spatcon, grayspatcon, data, fosclass, window, conn, grayt)
            fos_cases.append({"input": name, "fosclass": fosclass, "window": window, "connectivity": conn,
                              "gray_threshold": grayt, "input_sha256": digest(data), "sha256": digest(ref["map"]),
                              "stats": ref["stats"]})
            print("fos", name, fosclass, window)

    tile_cases = []
    mspa_exe, fwtools = find_gtb_mspa(), find_gtb_fwtools()
    if mspa_exe is None or fwtools is None:
        print("GTB mspa.exe or gdal tools (setfw.bat) not found: MSPA Tiling references are not generated")
    else:
        inputs = mspa_tile_inputs()
        for name, tile, overlap, params in MSPA_TILE_CASES:
            ref = run_gtb_mspa_tile(exe, mspa_exe, fwtools, inputs[name], tile_size=tile, overlap=overlap, **params)
            tile_cases.append({"input": name, "tile_size": tile, "overlap": overlap, "params": params,
                               "input_sha256": digest(inputs[name]), "sha256": digest(ref)})
            print("mspa_tile", name, tile, overlap, params)

    iso_cases = []
    if ggeo is not None:
        for name, (resist, marker) in isochrone_inputs().items():
            for mode in ("cost_map_aa", "cost_map_ab"):
                ref = run_gtb_isochrone(exe, ggeo, resist, marker, mode)
                iso_cases.append({"input": name, "mode": mode, "resist_sha256": digest(resist),
                                  "marker_sha256": digest(marker), "sha256": digest(ref)})
                print("isochrone", name, mode)

    OUT.write_text(json.dumps({
        "generated": dt.date.today().isoformat(),
        "interpreter": exe.name,
        "spatcon_sha256": hashlib.sha256(spatcon.read_bytes()).hexdigest(),
        "sources": sorted(f"tests/gtb_idl/{p.name}" for p in (ROOT / "tests" / "gtb_idl").glob("*.pro")),
        "accounting": cases,
        "fad_mscale": fad_cases,
        "fad_change": fad_change_cases,
        "lm_heatmap": lm_cases,
        "lm_delta": lm_delta_cases,
        "isochrone": iso_cases,
        "spa": spa_cases,
        "fos": fos_cases,
        "mspa_tile": tile_cases,
        "grayspatcon_sha256": hashlib.sha256(grayspatcon.read_bytes()).hexdigest() if grayspatcon else None,
        "ggeo_sha256": hashlib.sha256(ggeo.read_bytes()).hexdigest() if ggeo else None,
    }, indent=1))


if __name__ == "__main__":
    gdl = Path(sys.argv[1]) if len(sys.argv) > 1 else find_gdl()
    if gdl is None:
        sys.exit("GDL not found: pass its path or set LANDSCHAFT_GDL_EXE")
    spatcon_exe = Path(sys.argv[2]) if len(sys.argv) > 2 else find_spatcon()
    if spatcon_exe is None:
        sys.exit("spatcon not found: pass its path or set LANDSCHAFT_SPATCON_EXE")
    ggeo_exe = Path(sys.argv[3]) if len(sys.argv) > 3 else find_ggeo()
    if ggeo_exe is None:
        print("ggeo not found (LANDSCHAFT_GGEO_EXE): isochrone references are not generated")
    main(gdl, spatcon_exe, ggeo_exe)
