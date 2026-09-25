# SPDX-License-Identifier: GPL-3.0-or-later
"""Bit-exact agreement with the original GuidosToolbox ``mspa`` executable.

* ``reference``: outputs stored in tests/data/reference, produced by
  tools/make_reference.py with the original executable. Runs everywhere.
* ``live_reference``: runs the original executable now (set
  ``LANDSCHAFT_MSPA_EXE``) on fresh random inputs and parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
import scenes
from gtb_reference import run_original_mspa

pytestmark = pytest.mark.reference

REFERENCE = Path(__file__).parent / "data" / "reference"


def _cases(npz_path, prefix=""):
    with np.load(npz_path) as z:
        keys = list(z.keys())
    return [k for k in keys if k.startswith(prefix) and not k.endswith("input")]


def _params(key: str) -> dict:
    conn, eew, internal = key.split("/")[-1].split("_")
    return dict(connectivity=int(conn[1:]), edge_width=int(eew[1:]), internal=bool(int(internal[1:])))


def assert_same(got: np.ndarray, expected: np.ndarray) -> None:
    diff = got != expected
    if diff.any():
        ys, xs = np.nonzero(diff)
        sample = [(int(y), int(x), int(got[y, x]), int(expected[y, x])) for y, x in zip(ys[:10], xs[:10])]
        pytest.fail(f"{diff.sum()} pixels differ from the original; (y, x, landschaft, original): {sample}")


@pytest.fixture(scope="module")
def gtb_input(reference_dir):
    return np.load(reference_dir / "gtb_input.npz")


@pytest.fixture(scope="module")
def synthetic(reference_dir):
    return np.load(reference_dir / "synthetic.npz")


@pytest.mark.parametrize("key", _cases(REFERENCE / "gtb_input.npz"))
def test_gtb_sample_image(gtb_input, key):
    assert_same(ls.mspa(gtb_input["input"], **_params(key)), gtb_input[key])


@pytest.mark.parametrize("key", _cases(REFERENCE / "synthetic.npz"))
def test_synthetic_scenes(synthetic, key):
    case = key.split("/")[0]
    assert_same(ls.mspa(synthetic[f"{case}/input"], **_params(key)), synthetic[key])


# --- live comparison with the executable ---------------------------------------

LIVE_SEEDS = range(6)


@pytest.mark.live_reference
@pytest.mark.parametrize("kind", sorted(scenes.SCENES))
@pytest.mark.parametrize("seed", LIVE_SEEDS)
def test_live_random_scenes(mspa_exe, kind, seed):
    rng = np.random.default_rng(1000 + seed)
    params = dict(
        connectivity=int(rng.choice([4, 8])),
        edge_width=int(rng.choice([1, 2, 3, 4, 6, 9, 15, 30])),
        internal=bool(rng.integers(0, 2)),
    )
    data = scenes.make(kind, 5000 + seed)
    expected = run_original_mspa(mspa_exe, data, **params)
    assert_same(ls.mspa(data, **params), expected)


GTB_GSC_EXE = Path(r"C:\Users\eduar\Documents\dev\GTB\external_sources\grayspatcon64.exe")


@pytest.mark.skipif(not GTB_GSC_EXE.is_file(), reason="grayspatcon64.exe not found")
@pytest.mark.parametrize(
    "mname, mcode",
    [
        ("mean", 1),
        ("median", 9),
        ("majority", 16),
        ("range", 27),
        ("stddev", 25),
        ("shannon_diversity", 7),
        ("shannon_evenness", 8),
    ],
)
def test_live_grayspatcon_against_gtb(tmp_path, mname, mcode):
    import subprocess

    rng = np.random.default_rng(42)
    img = rng.integers(0, 101, size=(40, 40), dtype=np.uint8)
    img[0, 0] = 255
    img[10, 10] = 255

    img.tofile(tmp_path / "gscinput")
    w = 7
    with open(tmp_path / "gscpars.txt", "w") as f:
        f.write(f"R 40\nC 40\nM {mcode}\nP 0\nG 0\nW {w}\nF 2\nB 6\nA 1\nX 5\nY 10\nK 5\n")

    subprocess.run([str(GTB_GSC_EXE)], cwd=str(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    gtb_out = np.fromfile(tmp_path / "gscoutput", dtype=np.float32).reshape((40, 40))

    gpy_out = ls.gray_spatial_convolution(img, metric=mname, window_size=w, mask_missing=True)

    diff = np.abs(gtb_out - gpy_out)
    assert np.all(diff <= 1e-4)
