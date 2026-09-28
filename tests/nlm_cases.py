# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_nlm.py and the NLM tests."""

from __future__ import annotations

import numpy as np

# -- generators ------------------------------------------------------------------

R_SEEDS = [1, 42, 123456, -5, 0, 2147483647]

# Operations of landschaft._core.nlm_r_sequence (and their R equivalents).
_PROB5 = [0.1, 0.2, 0.3, 0.2, 0.2]
_PROB_TIES = [1.0, 2.0, 1.0, 2.0, 0.0, 1.0]
_PROB_WALKER = list(np.linspace(0.5, 3.0, 300))  # > 200 values: Walker's alias
R_SEQUENCES = {
    "unif_norm": [
        ("runif", 1000, 0.0, 1.0),
        ("rnorm", 10000, 0.0, 1.0),
        ("runif", 10, 3.0, 7.0),
        ("rnorm", 1000, 2.0, 0.5),
        ("runif", 5, 360.0, 360.0),  # a == b: no draw
    ],
    "sample": [
        ("sample", 10, 10, False, None),
        ("sample", 1000, 100, False, None),
        ("sample", 100, 1000, True, None),
        ("sample", 1, 1, False, None),
        ("sample", 65537, 3, False, None),
        ("sample", 5, 50, True, _PROB5),
        ("sample", 5, 3, False, _PROB5),
        ("sample", 6, 20, True, _PROB_TIES),
        ("sample", 6, 4, False, _PROB_TIES),
        ("sample", 300, 500, True, _PROB_WALKER),
        ("sample", 20_000_000, 5, False, None),  # sample2 (hash)
        ("sample", 3, 1, False, None),
    ],
}

STD_SEEDS = [1, 42, 2147483647, 4000000000]

# Operations of landschaft._core.nlm_std_sequence and of the probe.
STD_SEQUENCES = {
    "draws": [
        ("canonical", 2000),
        ("uniform_int", 500, 0, 9),
        ("uniform_int", 500, 3, 17),
        ("uniform_int", 500, 0, 999_999),
        ("uniform_int", 200, -5, 5),
        ("normal", 1001, 0.0, 1.0),
        ("normal_fresh", 1000, 0.0, 0.7),
        ("normal", 11, 2.0, 3.0),
    ],
    "shuffle": [
        ("shuffle", 1),
        ("shuffle", 2),
        ("shuffle", 101),
        ("shuffle", 2500),
    ],
    "shuffle_big": [
        ("shuffle", 65535),  # last size with two positions per draw
        ("shuffle", 65536),
    ],
}
# Sequences drawn with fewer seeds (large outputs).
STD_SEQUENCE_SEEDS = {"shuffle_big": [42]}

# -- models ----------------------------------------------------------------------

# key -> R expression returning a SpatRaster (filled in by the later phases).
MODEL_CASES: dict[str, str] = {}
