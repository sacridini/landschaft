# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.download_gbif with the GBIF API replaced by a fake one (no network)."""

from __future__ import annotations

import pytest

import landschaft as ls
from landschaft.sdm import _gbif


def fake_api(n_records: int, calls: list):
    def get(path, params, timeout=60.0):
        calls.append((path, dict(params)))
        if path == "species/match":
            if params["name"] == "Nonexistent":
                return {"matchType": "NONE"}
            return {"usageKey": 10, "acceptedUsageKey": 7, "matchType": "EXACT", "status": "SYNONYM"}
        offset, limit = params["offset"], params["limit"]
        results = [{"key": k, "species": "A b", "decimalLongitude": -50 + k / 1000, "decimalLatitude": -10.0,
                    "coordinateUncertaintyInMeters": 10 * k if k % 2 else None, "issues": ["A", "B"] if k == 0 else []}
                   for k in range(offset, min(offset + limit, n_records))]
        return {"count": n_records, "results": results}
    return get


def test_pages_synonym_and_filters(monkeypatch):
    calls = []
    monkeypatch.setattr(_gbif, "_get", fake_api(1000, calls))
    occ = ls.sdm.download_gbif("A b", country="BR", year=(1990, 2020))
    assert len(occ) == 1000 and occ["gbifID"].tolist() == list(range(1000))
    assert occ.loc[0, "issue"] == "A;B" and occ.loc[1, "issue"] == ""
    searches = [c[1] for c in calls if c[0] == "occurrence/search"]
    assert sorted(s["offset"] for s in searches) == [0, 300, 600, 900]
    assert all(s["taxonKey"] == 7 and s["country"] == "BR" and s["year"] == "1990,2020" for s in searches)
    assert occ.attrs["gbif"]["taxonKey"] == 7 and not occ.attrs["gbif"]["truncated"]


def test_limit_and_uncertainty(monkeypatch):
    monkeypatch.setattr(_gbif, "_get", fake_api(5000, []))
    occ = ls.sdm.download_gbif(7, limit=400, max_uncertainty=1000)
    # 400 records, the odd ones above 1000 m dropped (k >= 101), the even ones have none
    assert len(occ) == 200 + 50 and occ.attrs["gbif"]["truncated"]


def test_unknown_species(monkeypatch):
    monkeypatch.setattr(_gbif, "_get", fake_api(0, []))
    with pytest.raises(ValueError, match="no taxon"):
        ls.sdm.download_gbif("Nonexistent")
