# Tutorials

Step-by-step guides, one per topic. Each starts with what the analysis means
and why you would use it, walks through a complete example on the maps that
ship with Landschaft, shows the results and how to read them, and ends with
the advanced options. Every code block runs as it is: copy it into a notebook
and follow along.

!!! tip "Before you start"
    [Install Landschaft](../getting-started/installation.md) and read [core concepts](../getting-started/concepts.md): it explains how
    habitat maps are coded, which all tutorials rely on.

!!! info "New to landscape ecology?"
    The [Learn](../learn/index.md) course explains what each analysis
    measures, the theory behind it and when to use it. The tutorials focus on
    how to run the tools.

<div class="grid cards" markdown>

-   :material-shape-outline:{ .lg .middle } **[Habitat patterns](patterns.md)**

    ---

    Cores, edges, bridges and islets; objects by size; parcels and the shape
    of habitat boundaries.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[Fragmentation](fragmentation.md)**

    ---

    Habitat density at one or many scales, fragmentation classes, landscape
    mosaics, contagion and entropy.

-   :material-table-large:{ .lg .middle } **[Landscape metrics](landscape-metrics.md)**

    ---

    Patch, class and landscape metrics, sample plots, moving windows, time
    series and spatial signatures.

-   :material-graph-outline:{ .lg .middle } **[Habitat networks](habitat-networks.md)**

    ---

    Patches and distances as a graph: connectivity indices, the importance
    of every patch and link, protected-area connectivity.

-   :material-flash-outline:{ .lg .middle } **[Circuit connectivity](circuit-connectivity.md)**

    ---

    Current flow between habitat areas and omnidirectional connectivity
    maps, on rasters and on networks.

-   :material-map-marker-path:{ .lg .middle } **[Distance & restoration](distance-restoration.md)**

    ---

    Distance and cost maps, least-cost paths and corridors, restoration
    status and travel-time maps.

-   :material-compare-horizontal:{ .lg .middle } **[Change detection](change-detection.md)**

    ---

    What changed between two dates: habitat gains and losses, structural
    change, shifts between fragmentation classes.

-   :material-database-arrow-right-outline:{ .lg .middle } **[Large data & batch](large-data.md)**

    ---

    Folders of rasters, maps larger than memory, time stacks with dask and
    very large tables.

-   :material-server-network:{ .lg .middle } **[On a cluster (HPC, Slurm)](hpc-slurm.md)**

    ---

    Scripts in batch jobs with every processor of the job, folders of
    rasters over job arrays, time limits and nodes without internet access.

-   :material-routes:{ .lg .middle } **[Routes on grids](grid-routes.md)**

    ---

    Travel-time surfaces, least-cost routes, random walks and randomised
    shortest paths; how much routes from a common origin overlap.

-   :material-dice-multiple-outline:{ .lg .middle } **[Neutral landscapes](neutral-landscapes.md)**

    ---

    Simulated landscapes as null models: random, gradients, fractals,
    clusters and mosaics; habitat maps and null-model tests.

-   :material-terrain:{ .lg .middle } **[Surface texture](surface-texture.md)**

    ---

    Continuous rasters without classes: roughness, height distribution,
    texture images, spatial structure, terrain indices of DEMs and the
    hilltops protected by Brazil's Forest Code.


-   :material-water-outline:{ .lg .middle } **[Terrain hydrology](terrain-hydrology.md)**

    ---

    Filling and breaching depressions, flow directions and accumulation,
    streams, height above the nearest drainage and wetness.

-   :material-waves:{ .lg .middle } **[Watershed simulation](watershed-model.md)**

    ---

    The SWAT+ model from a DEM, land use and soils: daily water balance,
    streamflow, soil loss on the map and a land use scenario.

-   :material-leaf:{ .lg .middle } **[Biodiversity analysis](biodiversity.md)**

    ---

    Species records in grid cells: richness, endemism, phylogenetic
    diversity and endemism, randomisations, CANAPE and regions.

-   :material-map-marker-star-outline:{ .lg .middle } **[Conservation priorities](prioritization.md)**

    ---

    Rank a landscape for many species with Zonation: performance curves,
    weights, costs, existing reserves and connectivity.

-   :material-map-check-outline:{ .lg .middle } **[Optimized conservation plans](conservation-plans.md)**

    ---

    The cheapest plan that meets every target: reserves, compactness,
    budgets, importance, actions against threats, and a report.

-   :material-sprout-outline:{ .lg .middle } **[Restoration priorities](restoration-plans.md)**

    ---

    How much of every converted cell to restore: original vegetation,
    habitat gains, extinction risk, carbon, intactness, budgets, regions
    and uncertainty.

-   :material-map-marker-radius-outline:{ .lg .middle } **[Species distributions](species-distributions.md)**

    ---

    Clean and thin occurrence records, prepare predictors and background,
    fit and cross-validate models on spatial folds, project and check
    extrapolation.

-   :material-map-clock-outline:{ .lg .middle } **[Land change simulation](land-change-simulation.md)**

    ---

    Project land use from two maps: demand, transition potential and four
    allocations, validation at several resolutions, calibration, scenarios
    with replicates, roads and stocks that change with the model.

</div>
