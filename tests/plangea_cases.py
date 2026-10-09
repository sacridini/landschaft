# SPDX-License-Identifier: GPL-3.0-or-later
"""Inputs of the plangea reference cases (tools/make_reference_plangea.py, tests/test_plangea_reference.py).

Small rasters drawn with uniform numbers only (no libm), so every platform builds the same inputs:
land cover as shares of six classes (three natural, two anthropic, water), past cover, ecoregion
shares, species ranges with their habitat table and elevation limits, a human footprint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NATURAL = ["forest", "savanna", "wetland"]
ANTHROPIC = ["crops", "pasture"]
CLASSES = NATURAL + ANTHROPIC + ["water"]
TYPES = ["N", "N", "N", "A", "A", "I"]
SHAPE = (9, 11)
RES = 1000.0
CRS = "EPSG:32723"
SPECIES = {"birds": ["sp_a", "sp_b", "sp_c"], "mammals": ["sp_d", "sp_e", "sp_f"]}
# the habitat table: sp_c lives only in crops (no natural class), sp_e has no row
HABITATS = [("sp_a", "forest"), ("sp_a", "crops"), ("sp_b", "savanna"), ("sp_b", "wetland"), ("sp_c", "crops"),
            ("sp_d", "forest"), ("sp_f", "forest")]
# elevation limits (NaN: none); species without a row have none
LIMITS = [("sp_a", 500.0, np.nan), ("sp_b", np.nan, 1200.0), ("sp_d", 300.0, 1500.0)]


def grid(values: np.ndarray, res: float = RES, names=None):
    """A georeferenced DataArray (2-D, or bands first) on the cases' grid."""
    import rioxarray  # noqa: F401
    import xarray as xr

    v = np.asarray(values, dtype=np.float64)
    ny, nx = v.shape[-2:]
    coords = {"y": (np.arange(ny)[::-1] + 0.5) * res + 7_000_000.0, "x": (np.arange(nx) + 0.5) * res + 500_000.0}
    if v.ndim == 3:
        coords["band"] = list(names) if names is not None else [f"b{k + 1}" for k in range(v.shape[0])]
        da = xr.DataArray(v, dims=("band", "y", "x"), coords=coords)
    else:
        da = xr.DataArray(v, dims=("y", "x"), coords=coords)
    return da.rio.write_crs(CRS)


def land_cover() -> np.ndarray:
    """Shares of the six classes (6, 9, 11), with the special pixels the reference cases need."""
    rng = np.random.default_rng(11)
    u = rng.random((6,) + SHAPE)
    u[5] *= rng.random(SHAPE) < 0.2           # water in some pixels
    u[3:5] *= (rng.random(SHAPE) < 0.8)       # some pixels without anthropic land
    u[0:3] *= (rng.random(SHAPE) < 0.85)      # some without natural land
    u[5] = np.where(u.sum(axis=0) == 0, 1.0, u[5])  # nothing drawn: water
    s = u.sum(axis=0)
    v = u / s
    v[:, 0, 0] = [0, 0, 0, 0, 0, 1.0]         # water only: not land
    v[:, 1, 1] = [0.03, 0, 0, 0.97, 0, 0]     # nearly all converted
    v[:, 2, 2] = [0, 0, 0, 0.6, 0.4, 0]       # all converted
    v[:, 8, 10] = [0, 0, 0, 0.5, 0.5, 0]      # all converted, outside the ecoregions, no past
    v[0, 0, 1] = np.nan                       # a missing share
    return v


def past() -> np.ndarray:
    """Past shares of the natural classes (3, 9, 11): some pixels without past cover, one NaN."""
    rng = np.random.default_rng(12)
    u = rng.random((3,) + SHAPE)
    u *= rng.random(SHAPE) < 0.85
    p = u / np.where(u.sum(axis=0) > 0, u.sum(axis=0), 1.0) * 0.9
    p[:, 2, 2] = 0.0
    p[:, 8, 10] = 0.0
    p[1, 4, 4] = np.nan
    return p


def ecoregions() -> np.ndarray:
    """Every natural class's share in the pixel's ecoregion (3, 9, 11): three zones, none in the last column."""
    zones = [np.array([0.7, 0.2, 0.1]), np.array([0.2, 0.5, 0.3]), np.array([0.1, 0.3, 0.6])]
    out = np.zeros((3,) + SHAPE)
    for j in range(SHAPE[1] - 1):
        out[:, :, j] = zones[min(j * 3 // (SHAPE[1] - 1), 2)][:, None]
    return out


def ranges() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(13)
    names = [s for group in SPECIES.values() for s in group]
    r = (rng.random((len(names),) + SHAPE) < 0.55).astype(np.float64)
    r[:, 0:3, 0:4] = 1.0
    r[names.index("sp_f")] = 0.0
    r[names.index("sp_f"), 0, 0] = 1.0       # only on water: no habitat
    r[names.index("sp_b"), 5, 5] = np.nan
    return names, r


def elevation() -> np.ndarray:
    rng = np.random.default_rng(14)
    return np.floor(rng.random(SHAPE) * 2000.0)


def habitat_table() -> pd.DataFrame:
    return pd.DataFrame(HABITATS, columns=["taxonid", "habitat_class"])


def limits_table() -> pd.DataFrame:
    return pd.DataFrame(LIMITS, columns=["species", "lower", "upper"])


def footprint(shape=(10, 12), seed: int = 21, low: float = 0.0, nan: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = low + rng.random(shape) * (30.0 - low)
    if nan:
        h[0, 5] = np.nan
        h[4:6, 2:4] = np.nan
    return h


def conversion(shape=(10, 12), seed: int = 22) -> np.ndarray:
    """Two anthropic classes' shares (2, ny, nx): plangea's prop_convert."""
    rng = np.random.default_rng(seed)
    return rng.random((2,) + shape) * 0.5


def restored(shape=(10, 12), seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape)


HFI_WEIGHTS = (4.0, 7.0)


def reduction(pc: np.ndarray, weights=HFI_WEIGHTS) -> np.ndarray:
    """plangea's prop_convert_weight: Reduce('+', Map('*', hfi_weights, prop_convert))."""
    out = weights[0] * pc[0]
    for w, layer in zip(weights[1:], pc[1:]):
        out = out + w * layer
    return out


# the intactness cases: footprint, radius, parameters, cell size (m), restoration, gradient
IT_CASES = {
    "it_default": dict(shape=(10, 12), seed=21, radius=2, gamma=-np.log(0.2) / 4, beta=-0.2, z=0.5, res=1000.0),
    "it_params": dict(shape=(10, 12), seed=24, radius=3, gamma=0.3, beta=-0.1, z=0.7, res=2000.0),
    "it_restored": dict(shape=(10, 12), seed=25, radius=1, gamma=-np.log(0.2) / 4, beta=-0.2, z=0.5, res=1000.0,
                        restore=True),
    "it_gradient": dict(shape=(7, 8), seed=26, radius=2, gamma=-np.log(0.2) / 4, beta=-0.2, z=0.5, res=1000.0,
                        restore=True, gradient=True, low=2.0),
}

# the original-vegetation cases: past cover, ecoregions, failsafe classes, problem type, fixed reference
OA_CASES = {
    "oa_past_er": dict(past=True, er=True, failsafe=["forest", "savanna"], problem=["R"]),
    "oa_er": dict(past=False, er=True, failsafe=NATURAL, problem=["R"]),
    "oa_flat": dict(past=True, er=False, failsafe=["forest"], problem=["R"], fixed=True),
    "oa_convert": dict(past=True, er=True, failsafe=NATURAL, problem=["R", "D"]),
}

# the area-of-habitat cases (on an original-vegetation case)
BD_CASES = {
    "bd_absent": dict(oa="oa_past_er", include_absent=True, elevation=False, exrisk=True),
    "bd_elevation": dict(oa="oa_er", include_absent=False, elevation=True),
}


def uncertainty_inputs() -> dict:
    rng = np.random.default_rng(31)
    n = 40
    return {"v": rng.random(n) * 3.0, "px_area": 0.5 + rng.random(n), "x_vals": rng.random(n) * 2.0 - 0.5,
            "var_x": 1.0 + rng.random(n), "var_v": 2.0 + rng.random(n) * 5.0, "sd_x": rng.random(n) * 0.2,
            "sd_v": rng.random(n) * 0.5}


def exrisk_x(n: int) -> np.ndarray:
    """The restored shares of bd_absent's master pixels."""
    rng = np.random.default_rng(32)
    return rng.random(n) * 0.5


def ecoregion_codes() -> np.ndarray:
    """Ecoregion codes (9, 11): three zones by columns, every cell coded (multi_calc_ec stops on NA)."""
    out = np.empty(SHAPE)
    for j in range(SHAPE[1]):
        out[:, j] = 1 + min(j * 3 // SHAPE[1], 2)
    return out


def ec_x(n: int) -> np.ndarray:
    """Restored shares of the master pixels for the ecosystem cases."""
    rng = np.random.default_rng(33)
    return rng.random(n) * 0.4


# plangea's restoration procedure (scen's loop: benefit per pixel, the area target in steps, bd refreshed
# between them) on bd_absent: steps and the area target as a share of the restorable area (sense "<=")
SCEN_CASES = {
    "scen_1": dict(steps=1, share=0.3),
    "scen_3": dict(steps=3, share=0.3),
    "scen_4_60": dict(steps=4, share=0.6),
}

# the same for conversion ("D") and conservation ("C", corrected) on bd_absent's habitats: the target a
# share of the natural area (converted, at least; conserved, at most); conservation on the natural land of
# the first `cols` columns (with all of it, its baseline loses every species' habitat: slopes at A ~ 0)
SCEN_CD_CASES = {
    "scen_d_1": dict(type="D", steps=1, share=0.2),
    "scen_d_3": dict(type="D", steps=3, share=0.3),
    "scen_c_1": dict(type="C", steps=1, share=0.3, cols=7),
    "scen_c_4": dict(type="C", steps=4, share=0.5, cols=7),
}

# the risk of ecosystem collapse (multi_calc_ec): the ecoregions' natural area over plangea's master pixels
# (its form) or over all the land (the corrected one)
EC_CASES = {
    "ec_master": dict(domain="master"),
    "ec_land": dict(domain="land"),
}


# the ecosystemic targets (harmonize_sr_ecosys) over the ecoregion codes: the landscape target, the least
# share of every ecosystem, the damping exponent, the steps; with projected changes for
# harmonize_sr_sparable
ESYS_CASES = {
    "esys_10": {"target": 8.0, "minimum": 0.5, "hl": 0.0, "steps": 10, "change": [4.0, -1.5, 2.0]},
    "esys_damped": {"target": 15.0, "minimum": 0.2, "hl": -1.0, "steps": 5, "change": [3.0, 1.0, -5.0]},
}
