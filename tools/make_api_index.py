# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the overview of the API reference (docs/api/index.md).

The table of every function and class is read from the ``:::`` entries of the
API pages and the first sentence of each docstring; run it again after adding
an entry to a page.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api"

PAGES = [("data.md", "Data & configuration", ":material-database-outline:",
          "Read, write and reclassify rasters, collections, colour tables, threads."),
         ("patterns.md", "Habitat patterns", ":material-shape-outline:",
          "MSPA, SPA, objects by size, parcels and contortion."),
         ("fragmentation.md", "Fragmentation", ":material-view-grid-outline:",
          "Density, fragmentation classes, landscape mosaics, contagion, entropy."),
         ("landscape-metrics.md", "Landscape metrics", ":material-chart-box-outline:",
          "Patch, class and landscape metrics; windows, plots, zones, time series."),
         ("networks.md", "Habitat networks", ":material-graph-outline:",
          "Habitat graphs, IIC/PC/ECA, importance, ProtConn, grids, MSPA networks."),
         ("circuits.md", "Circuit connectivity", ":material-lightning-bolt-outline:",
          "Omniscape and Circuitscape: current, resistance distance, networks."),
         ("surface.md", "Surface metrics", ":material-terrain:",
          "Texture of continuous rasters (ISO 25178): roughness, shape, trend surfaces."),
         ("cost.md", "Grid distances", ":material-routes:",
          "Transitions, cost surfaces, least-cost and random-walk distances and paths."),
         ("distance.md", "Distance & restoration", ":material-map-marker-distance:",
          "Distances, resistance, least-cost paths, corridors, zones of influence."),
         ("change.md", "Change detection", ":material-compare-horizontal:",
          "Habitat, fragmentation and landscape-mosaic change between dates."),
         ("batch.md", "Large data & batch", ":material-layers-triple-outline:",
          "Folders of rasters, tiled MSPA, maps larger than memory."),
         ("nlm.md", "Neutral landscape models", ":material-dice-multiple-outline:",
          "Simulated landscapes as null models: random, gradients, clusters, fractals."),
         ("biodiversity.md", "Biodiversity analysis", ":material-paw:",
          "Species counts per cell, phylogenies and dissimilarity matrices (Biodiverse)."),
         ("prioritization.md", "Spatial prioritization", ":material-star-circle-outline:",
          "Zonation's ranking of the cells for conservation: rules, weights, costs, curves."),
         ("sdm.md", "Species distributions", ":material-map-search-outline:",
          "Clean records, choose predictors and background, spatial folds, fit, evaluate, project.")]

INTRO = '''# API reference

Every public function and class, with its parameters, what it returns and an
example you can run. Each page covers one topic; the
[tutorials](../tutorials/index.md) show the same tools at work on real maps.

<!-- no-run -->
```python
import landschaft as ls             # most tools: ls.mspa, ls.fad, ls.Landscape...
ls.conn.protconn(...)               # habitat networks and protected areas
from landschaft import circuit      # Circuitscape
ls.cost.transition(r, "mean", 8)    # distances and routes on grids
ls.surface.metrics(z)               # surface texture of continuous rasters
ls.nlm.random(100, 100, seed=1)     # neutral landscape models
ls.bio.BaseData.from_table(...)     # biodiversity analysis
ls.prioritize.zonation(features)    # spatial conservation prioritization
ls.sdm.prepare(occ, env)            # species distribution models
```

!!! example "Every example runs"
    The examples use the maps that ship with landschaft (`ls.example_data`) or
    small arrays built in place, and they are run as tests, outputs included.
    Copy one into a Python session to try a function.

<div class="grid cards" markdown>
'''


def resolve(path: str):
    """The object of a dotted path (the longest importable module, then attributes)."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def summary(obj) -> str:
    """First sentence of the docstring, Sphinx roles as code."""
    doc = " ".join((obj.__doc__ or "").strip().split("\n\n")[0].split())
    doc = re.sub(r":(?:func|class|meth|data):`~?([^`]+)`", lambda m: "`" + m.group(1).split(".")[-1] + "`", doc)
    doc = re.split(r"\. (?=[A-Z])", doc)[0]
    return doc.rstrip(".").replace("|", r"\|")


def prefix(ident: str) -> str:
    name = ident.split(".")[-1]
    if ".conn." in ident:
        return "ls.conn."
    if ".lsm." in ident:
        return "ls.lsm."
    if ".nlm." in ident:
        return "ls.nlm."
    if ".bio." in ident:
        return "ls.bio."
    if ".cost." in ident:
        return "ls.cost."
    if ".surface." in ident:
        return "ls.surface."
    if ".circuit." in ident and name != "omniscape":
        return "circuit."
    return "ls."


def main() -> None:
    out = [INTRO]
    for page, title, icon, text in PAGES:
        out.append(f"-   {icon}{{ .lg .middle }} **[{title}]({page})**\n\n    ---\n\n    {text}\n")
    out.append("</div>\n\n## All functions and classes\n")
    for page, title, _, _ in PAGES:
        ids = re.findall(r"^::: (\S+)", (API / page).read_text(encoding="utf-8"), flags=re.MULTILINE)
        out.append(f"### {title}\n\n| Name | Summary |\n|---|---|")
        for ident in ids:
            name = ident.split(".")[-1]
            out.append(f"| [`{prefix(ident)}{name}`]({page}#{ident}) | {summary(resolve(ident))} |")
        out.append("")
    (API / "index.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
