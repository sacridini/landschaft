# Tutorials

Step-by-step guides, one per topic. Each starts with what the analysis means
and why you would use it, walks through a complete example on the maps that
ship with Landschaft, shows the results and how to read them, and ends with
the advanced options. Every code block runs as it is: copy it into a notebook
and follow along.

!!! tip "Before you start"
    Install Landschaft with the `plot` extra ([installation](../getting-started/installation.md))
    and read [core concepts](../getting-started/concepts.md): it explains how
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

-   :material-dice-multiple-outline:{ .lg .middle } **[Neutral landscapes](neutral-landscapes.md)**

    ---

    Simulated landscapes as null models: random, gradients, fractals,
    clusters and mosaics; habitat maps and null-model tests.

</div>
