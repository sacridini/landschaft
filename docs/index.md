---
title: Landschaft
description: Fast, exact landscape pattern, fragmentation, landscape-metric and connectivity analysis for Python.
hide:
  - navigation
  - toc
---

# Landschaft

<div class="ls-hero" markdown>

![Landschaft](assets/images/logo-stacked.svg#only-light){ .ls-hero__logo }
![Landschaft](assets/images/logo-stacked-dark.svg#only-dark){ .ls-hero__logo }

<p class="ls-hero__tagline">
Fast, exact landscape analysis for Python. Map habitat structure, measure
fragmentation, compute landscape metrics and model ecological connectivity,
from a single raster on a laptop to continental maps larger than memory.
</p>

<div class="ls-hero__actions" markdown>
[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Tutorials](tutorials/index.md){ .md-button }
[Learn](learn/index.md){ .md-button }
[GitHub :fontawesome-brands-github:](https://github.com/sacridini/landschaft){ .md-button }
</div>

<p class="ls-hero__meta">
Python 3.10+ · C++17 / OpenMP core · numpy · xarray · dask · GPL-3.0-or-later
</p>

</div>

## What you can do

<div class="ls-areas" markdown>

-   [**Habitat patterns**](tutorials/patterns.md)

    Split habitat into cores, edges, bridges, loops and islets; classify
    objects by size; describe the shape and texture of every patch.

-   [**Fragmentation**](tutorials/fragmentation.md)

    Habitat density at many observation scales, fragmentation classes,
    landscape mosaics, contagion and spatial entropy, as maps and statistics.

-   [**Landscape metrics**](tutorials/landscape-metrics.md)

    130+ patch, class and landscape metrics, sample plots, moving windows,
    time series, spatial signatures and rasters larger than memory.

-   [**Habitat networks**](tutorials/habitat-networks.md)

    Patches and distances as a graph: equivalent connected area, probability
    of connectivity, node and link importance, protected-area connectivity.

-   [**Circuit connectivity**](tutorials/circuit-connectivity.md)

    Current flow between habitat areas and omnidirectional connectivity
    across whole landscapes, on rasters and on networks.

-   [**Distance & restoration**](tutorials/distance-restoration.md)

    Distance maps, cost surfaces, least-cost paths and corridors,
    restoration priorities and travel-time (isochrone) maps.

-   [**Routes on grids**](tutorials/grid-routes.md)

    Travel-time surfaces, least-cost and random-walk distances, randomised
    shortest paths and the overlap of dispersal routes.

-   [**Change detection**](tutorials/change-detection.md)

    Compare two dates: simple and morphological change, shifts between
    fragmentation classes and heat maps of landscape change.

-   [**Surface texture**](tutorials/surface-texture.md)

    Roughness, height distribution and texture of continuous rasters
    (ISO 25178) such as NDVI or elevation, terrain indices of DEMs and
    protected hilltops.

-   [**Neutral landscapes**](tutorials/neutral-landscapes.md)

    Simulated landscapes as null models: random maps, gradients, fractal
    surfaces, Gaussian fields, clusters and mosaics.

-   [**Biodiversity analysis**](tutorials/biodiversity.md)

    Species records in grid cells: richness, endemism, phylogenetic
    diversity and endemism, randomisations and regions.

-   [**Conservation priorities**](tutorials/prioritization.md)

    Rank whole landscapes for many species: performance curves, costs,
    reserves, connectivity and impact assessment.

-   [**Species distributions**](tutorials/species-distributions.md)

    Clean occurrence records, choose predictors and background points, fit
    and evaluate models with spatial folds, and project them to new climates.

-   [**Land change simulation**](tutorials/land-change-simulation.md)

    Project land use from two maps: demand, transition potential and
    allocation, validation against the observed change, calibration,
    scenarios with replicates, and roads and stocks that change with the
    model.

-   [**Large data & batch**](tutorials/large-data.md)

    Folders of rasters in parallel, tiles for maps larger than memory,
    streaming tables with millions of rows.

</div>

## A taste

=== "Habitat patterns"

    ```python
    import landschaft as ls

    forest = ls.load_raster(ls.example_data("input"))  # 0 missing, 1 background, 2 forest
    morph = ls.mspa(forest, edge_width=2)              # cores, edges, bridges, loops...
    ls.mspa_stats(morph)                                # area of every class
    ```

=== "Landscape metrics"

    ```python
    import landschaft as ls

    landcover = ls.load_raster(ls.example_data("clc"))
    ls.landscape_metrics(landcover, level="landscape", metric=["shdi", "contag", "ed"])
    ```

=== "Habitat networks"

    ```python
    import landschaft as ls

    landcover = ls.load_raster(ls.example_data("clc"))
    graph = ls.HabitatGraph.from_raster(
        landcover, habitat=[23, 24, 25],          # forest classes
        distance="edge", max_distance=2000,       # edge-to-edge distances, metres
        landscape_area=landcover.size * 25 * 25,  # 25 m cells
    )
    graph.overall(["PC", "IIC"], distance=1000, probability=0.5)
    ```

=== "Circuit connectivity"

    ```python
    import landschaft as ls

    landcover = ls.load_raster(ls.example_data("clc"))[:600, :600]
    resistance = ls.reclassify(landcover, {23: 1.0, 24: 1.0, 25: 1.0},   # forest is easy to cross
                               others=25.0, dtype="float64")
    flow = ls.omniscape(resistance, radius=30, block_size=5)
    ls.plot(flow)                     # on an interactive map over satellite imagery
    ```

## Why Landschaft

<div class="ls-principles" markdown>

<div markdown>

### Fast

A C++17 core with OpenMP, cache-friendly algorithms and exact sparse
solvers. Typical runs take seconds where the original tools take minutes
or hours. [See the benchmarks](benchmarks.md)

</div>

<div markdown>

### Exact

The tools are checked against the reference implementations of their
methods, to the last bit where the method allows it and to the solver's
precision where it does not. [How each tool is checked](benchmarks.md#accuracy)

</div>

<div markdown>

### Pythonic

numpy arrays, georeferenced `xarray.DataArray`s, dask, GeoTIFFs,
GeoDataFrames and pandas tables in and out. No GUI, no temporary files,
no other language to install.

</div>

<div markdown>

### Scales

Rasters larger than memory by tiles or row blocks, collections of maps in
parallel within a memory budget, and all cores but one by default.

</div>

</div>

## Validated against established tools

<div class="ls-validation" markdown>

Landschaft reproduces the methods of GuidosToolbox (MSPA, fragmentation,
accounting, restoration planning), the R package landscapemetrics and
FRAGSTATS (landscape metrics), Conefor Sensinode and Makurhini (habitat
networks), Circuitscape and Omniscape (circuit connectivity), gdistance (routes on
grids), geodiv and spatialEco (surface metrics), NLMR (neutral landscape models),
Zonation 5 (spatial prioritization), the R packages for species distributions
(CoordinateCleaner, usdm, dismo, blockCV, ENMeval, predicts, ecospat, maxnet) and
the land change models (lulcc, CLUMondo, FUTURES, MOLUSCE, Dinamica EGO), and is
[tested against their outputs](benchmarks.md#accuracy).

You do not need to know any of them to use it: the [tutorials](tutorials/index.md)
start from scratch, and the [Learn](learn/index.md) course explains the landscape
ecology behind every tool.

</div>
