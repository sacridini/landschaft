# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_marxan.py and tests/test_prioritize_marxan.py.

The inputs are Marxan's tables (pu, spec, puvspr, bound) built from the
optimisation test data (tests/optimize_cases.py): the 166 planning units of
``pu_table``, its five features, and the boundaries of their raster cells (100
per shared edge, every unit's edges with no planning unit beyond as its fixed
boundary). Each case gives Marxan's options and the executable it is compared
with: ``original`` (Marxan 4.0.6 as released) or ``fixed`` (with the
correction of opt/marxan.hpp: the Connolly and adaptive schedules score their
random reserves with these reserves' amounts; and the summary and penalties
written with 17 digits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optimize_cases as oc
import pandas as pd


@dataclass
class MarxanCase:
    options: dict
    exe: str = "fixed"
    status: bool = False
    spec: str = "target"  # "target", "prop" or "mixed"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def tables(status: bool = False, spec_kind: str = "target"):
    """pu, spec, puvspr and bound as DataFrames."""
    t = oc.pu_table()
    ids = (np.arange(len(t)) * 3 + 5).astype(np.int64)  # ids that are not row numbers
    pu = pd.DataFrame({"id": ids, "cost": t["cost"].to_numpy()})
    if status:
        s = np.zeros(len(t), dtype=np.int64)
        s[t["locked_in"].to_numpy()] = 2
        s[t["locked_out"].to_numpy()] = 3
        s[[7, 40, 90]] = 1
        pu["status"] = s
    amounts = t[oc.FEATURES].to_numpy()
    totals = amounts.sum(axis=0)
    sp_ids = np.array([3, 1, 4, 10, 7])
    spec = pd.DataFrame({"id": sp_ids})
    if spec_kind == "target":
        spec["target"] = np.round(0.3 * totals, 6)
        spec["spf"] = [2.0, 5.0, 1.0, 3.0, 2.5]
    elif spec_kind == "prop":
        spec["prop"] = [0.3, 0.25, 0.4, 0.2, 0.35]
        spec["spf"] = [2.0, 5.0, 1.0, 3.0, 2.5]
    else:
        spec["target"] = np.round(0.3 * totals, 6)
        spec.loc[2, "target"] = 0.0
        spec["prop"] = [0.0, 0.0, 0.4, 0.0, 0.0]
        spec["targetocc"] = [0, 12, 0, 0, 5]
        spec["spf"] = [2.0, 5.0, 1.0, 3.0, 2.5]
    rows = [(sp_ids[k], ids[i], amounts[i, k]) for i in range(len(t)) for k in range(5) if amounts[i, k] != 0]
    puvspr = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    cells = oc._units()
    pos = {c: k for k, c in enumerate(cells)}
    b = []
    for k, c in enumerate(cells):
        r, q = divmod(int(c), oc.NX)
        shared = 0
        for rr, qq in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)):
            if 0 <= rr < oc.NY and 0 <= qq < oc.NX and rr * oc.NX + qq in pos:
                shared += 1
                m = pos[rr * oc.NX + qq]
                if m > k:
                    b.append((ids[k], ids[m], 100.0))
        if shared < 4:
            b.append((ids[k], ids[k], 100.0 * (4 - shared)))
    bound = pd.DataFrame(b, columns=["id1", "id2", "boundary"])
    return pu, spec, puvspr, bound


BASE = {"num_reps": 4, "num_iters": 100_000, "num_temp": 1000, "blm": 0.1}

CASES: dict[str, MarxanCase] = {
    "fixed_schedule": MarxanCase({**BASE, "start_temp": 100.0, "cool_fac": 0.99, "seed": 7}, exe="original",
                                 notes="a fixed schedule: Marxan as released"),
    "fixed_schedule_patched": MarxanCase({**BASE, "start_temp": 100.0, "cool_fac": 0.99, "seed": 7},
                                         notes="the same on the corrected executable: identical runs"),
    "connolly": MarxanCase({**BASE, "start_temp": -1.0, "seed": 11}),
    "adaptive_annealing_only": MarxanCase({**BASE, "start_temp": -2.0, "run_mode": 6, "seed": 13}),
    "heuristic_richness": MarxanCase({**BASE, "start_temp": -1.0, "run_mode": 0, "heur_type": 0, "prop": 0.3,
                                      "seed": 17}),
    "heuristic_greedy_threshold": MarxanCase({**BASE, "start_temp": -1.0, "run_mode": 2, "heur_type": 1,
                                              "itimp_type": 3, "cost_thresh": 2000.0, "thresh_pen1": 2.0,
                                              "thresh_pen2": 1.0, "seed": 19}),
    "status_mixed_targets": MarxanCase({**BASE, "start_temp": 50.0, "cool_fac": 0.98, "misslevel": 0.9,
                                        "seed": 23}, exe="original", status=True, spec="mixed",
                                       notes="locked units, occurrence and proportion targets"),
    "prop_targets_connectivity_in": MarxanCase({**BASE, "start_temp": -1.0, "connectivity_in": True, "seed": 29},
                                               spec="prop"),
}
