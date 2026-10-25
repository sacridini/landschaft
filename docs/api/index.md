# API reference

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

-   :material-database-outline:{ .lg .middle } **[Data & configuration](data.md)**

    ---

    Read, write and reclassify rasters, collections, colour tables, threads.

-   :material-shape-outline:{ .lg .middle } **[Habitat patterns](patterns.md)**

    ---

    MSPA, SPA, objects by size, parcels and contortion.

-   :material-view-grid-outline:{ .lg .middle } **[Fragmentation](fragmentation.md)**

    ---

    Density, fragmentation classes, landscape mosaics, contagion, entropy.

-   :material-chart-box-outline:{ .lg .middle } **[Landscape metrics](landscape-metrics.md)**

    ---

    Patch, class and landscape metrics; windows, plots, zones, time series.

-   :material-graph-outline:{ .lg .middle } **[Habitat networks](networks.md)**

    ---

    Habitat graphs, IIC/PC/ECA, importance, ProtConn, grids, MSPA networks.

-   :material-lightning-bolt-outline:{ .lg .middle } **[Circuit connectivity](circuits.md)**

    ---

    Omniscape and Circuitscape: current, resistance distance, networks.

-   :material-terrain:{ .lg .middle } **[Surface metrics](surface.md)**

    ---

    Texture of continuous rasters (ISO 25178): roughness, shape, trend surfaces.

-   :material-waves:{ .lg .middle } **[Terrain hydrology](hydro.md)**

    ---

    Filling, flow directions and accumulation, streams, watersheds, HAND, wetness indices.

-   :material-water-outline:{ .lg .middle } **[Watershed model (SWAT+)](swat.md)**

    ---

    SWAT+ projects from TxtInOut folders or rasters, daily water, sediment and nutrients, results on maps.

-   :material-routes:{ .lg .middle } **[Grid distances](cost.md)**

    ---

    Transitions, cost surfaces, least-cost and random-walk distances and paths.

-   :material-map-marker-distance:{ .lg .middle } **[Distance & restoration](distance.md)**

    ---

    Distances, resistance, least-cost paths, corridors, zones of influence, original vegetation, intactness.

-   :material-compare-horizontal:{ .lg .middle } **[Change detection](change.md)**

    ---

    Habitat, fragmentation and landscape-mosaic change between dates.

-   :material-layers-triple-outline:{ .lg .middle } **[Large data & batch](batch.md)**

    ---

    Folders of rasters, tiled MSPA, maps larger than memory.

-   :material-dice-multiple-outline:{ .lg .middle } **[Neutral landscape models](nlm.md)**

    ---

    Simulated landscapes as null models: random, gradients, clusters, fractals.

-   :material-paw:{ .lg .middle } **[Biodiversity analysis](biodiversity.md)**

    ---

    Species counts per cell, phylogenies and dissimilarity matrices (Biodiverse).

-   :material-star-circle-outline:{ .lg .middle } **[Spatial prioritization](prioritization.md)**

    ---

    Zonation's ranking of the cells for conservation: rules, weights, costs, curves.

-   :material-map-search-outline:{ .lg .middle } **[Species distributions](sdm.md)**

    ---

    Clean records, choose predictors and background, spatial folds, fit, evaluate, project.

-   :material-timeline-clock-outline:{ .lg .middle } **[Land change simulation](simulate.md)**

    ---

    Land use change: transitions, potentials, CLUE / Dinamica / PLUS allocation, validation.

-   :material-leaf-circle-outline:{ .lg .middle } **[Ecosystem services](services.md)**

    ---

    InVEST's models: carbon, habitat quality, water yield and hydropower, sediment delivery.

-   :material-plus-minus-variant:{ .lg .middle } **[Uncertainty](uncertainty.md)**

    ---

    Totals, products, the delta method, Monte Carlo and confidence intervals.

</div>

## All functions and classes

### Data & configuration

| Name | Summary |
|---|---|
| [`ls.load_raster`](data.md#landschaft.load_raster) | Read one raster, or collect many |
| [`ls.write_raster`](data.md#landschaft.write_raster) | Write a raster: one band or several, any format GDAL writes |
| [`ls.match_grid`](data.md#landschaft.match_grid) | Put a raster on the grid of another: same CRS, cells and coordinates |
| [`ls.reproject`](data.md#landschaft.reproject) | Reproject a raster or vector to another CRS, or onto the grid of another raster |
| [`ls.clip`](data.md#landschaft.clip) | Cut a raster to a region: the study area, a biome, a municipality |
| [`ls.reclassify`](data.md#landschaft.reclassify) | Give the values of a raster new values: from a table, or into a habitat map |
| [`ls.load_vector`](data.md#landschaft.load_vector) | Read a vector layer (polygons, lines, points) as a GeoDataFrame |
| [`ls.write_vector`](data.md#landschaft.write_vector) | Write a vector layer, the format given by the file extension |
| [`ls.rasterize`](data.md#landschaft.rasterize) | Burn vector features into a raster on the grid of another raster |
| [`ls.zonal_stats`](data.md#landschaft.zonal_stats) | Statistics of a raster in every zone: polygons, or the labels of a raster |
| [`ls.extract`](data.md#landschaft.extract) | The values of a raster at points |
| [`ls.load_table`](data.md#landschaft.load_table) | Read a table, the format given by the file extension |
| [`ls.write_table`](data.md#landschaft.write_table) | Write a table, the format given by the file extension |
| [`ls.save`](data.md#landschaft.save) | Save data or a result, each part in the format that suits it |
| [`ls.load_result`](data.md#landschaft.load_result) | Read back a result folder written by `save` (or ``result.save``) |
| [`ls.plot`](data.md#landschaft.plot) | Plot anything: a raster, a vector layer, a table or the result of a tool, as it calls for |
| [`ls.Plot`](data.md#landschaft.Plot) | A plot made by `plot`: interactive maps and charts (or a static figure) |
| [`ls.example_data`](data.md#landschaft.example_data) | Path of an example data set shipped with the package |
| [`ls.RasterCollection`](data.md#landschaft.RasterCollection) | Many rasters handled as one object; pixels are read only when processed |
| [`ls.RasterInfo`](data.md#landschaft.RasterInfo) | Header of one raster of a collection (the pixels are not read) |
| [`ls.validate_mspa_input`](data.md#landschaft.validate_mspa_input) | Check that ``data`` is a habitat map MSPA can process |
| [`ls.validate_fad_input`](data.md#landschaft.validate_fad_input) | Check that ``data`` is a habitat map the density and object tools accept |
| [`ls.InvalidInputError`](data.md#landschaft.InvalidInputError) | The raster cannot be processed by the requested tool |
| [`ls.mspa_colormap`](data.md#landschaft.mspa_colormap) | Colour table of MSPA class maps (`mspa`) |
| [`ls.fad_colormap`](data.md#landschaft.fad_colormap) | Colour table of density maps of `fad`, `fad_multiscale` and `fos` (6 classes) |
| [`ls.fad5_colormap`](data.md#landschaft.fad5_colormap) | Colour table of 5-class fragmentation maps (`fos` with ``classes=5``) |
| [`ls.lm_colormap`](data.md#landschaft.lm_colormap) | Colour table of landscape mosaic maps (`landscape_mosaic`) |
| [`ls.accounting_colormap`](data.md#landschaft.accounting_colormap) | Colour table of object size-class maps (`accounting`) |
| [`ls.change_colormap`](data.md#landschaft.change_colormap) | Colour table of change maps (`simple_change`, `morphological_change`) |
| [`ls.foschange_colormap`](data.md#landschaft.foschange_colormap) | Colour table of fragmentation change maps (`fos_change`) |
| [`ls.dist_colormap`](data.md#landschaft.dist_colormap) | Colour table of distance maps (`euclidean_distance`) |
| [`ls.entropy_colormap`](data.md#landschaft.entropy_colormap) | Colour table of spatial entropy maps (`spatial_entropy_map`) |
| [`ls.contortion_colormap`](data.md#landschaft.contortion_colormap) | Colour table of contortion maps (`contortion`) |
| [`ls.resistance_colormap`](data.md#landschaft.resistance_colormap) | Colour table of resistance and isochrone maps (`distance_to_resistance`, `isochrone_map`) |
| [`ls.gtn_components_colormap`](data.md#landschaft.gtn_components_colormap) | Colour table of network component maps (`gtn_components`) |
| [`ls.gtn_importance_colormap`](data.md#landschaft.gtn_importance_colormap) | Colour table of network importance maps (`gtn_importance`) |
| [`ls.default_threads`](data.md#landschaft.default_threads) | The number of threads a parallel function uses when not given one |
| [`ls.set_default_threads`](data.md#landschaft.set_default_threads) | Set the default thread count of every parallel function for this process |
| [`ls.max_threads`](data.md#landschaft.max_threads) | Number of processors the compiled core can use |
| [`ls.doctor`](data.md#landschaft.doctor) | Print what landschaft finds on this computer |

### Habitat patterns

| Name | Summary |
|---|---|
| [`ls.mspa`](patterns.md#landschaft.mspa) | Morphological Spatial Pattern Analysis (MSPA) |
| [`ls.mspa_stats`](patterns.md#landschaft.mspa_stats) | Area of every MSPA class |
| [`ls.spa`](patterns.md#landschaft.spa) | Simplified Pattern Analysis (GuidosToolbox *Pattern -> Morphological -> SPA2/3/5/6*) |
| [`ls.SPAResult`](patterns.md#landschaft.SPAResult) | Result of `spa` |
| [`ls.accounting`](patterns.md#landschaft.accounting) | Accounting: foreground objects grouped into size classes |
| [`ls.AccountingResult`](patterns.md#landschaft.AccountingResult) | Result of `accounting` |
| [`ls.parcellation`](patterns.md#landschaft.parcellation) | How finely the landscape is divided into parcels |
| [`ls.ParcellationResult`](patterns.md#landschaft.ParcellationResult) | Results from landscape parcellation analysis |
| [`ls.contortion`](patterns.md#landschaft.contortion) | How convoluted the boundary of every habitat object is |

### Fragmentation

| Name | Summary |
|---|---|
| [`ls.fad`](fragmentation.md#landschaft.fad) | Forest Area Density (FAD) & Multiscale Fragmentation Analysis |
| [`ls.FADResult`](fragmentation.md#landschaft.FADResult) | Result of single-scale FAD analysis (`fad`) |
| [`ls.fad_multiscale`](fragmentation.md#landschaft.fad_multiscale) | Compute Multiscale Forest Area Density (FAD-MS) matching GuidosToolbox |
| [`ls.FADMultiscaleResult`](fragmentation.md#landschaft.FADMultiscaleResult) | Result of multiscale FAD analysis (`fad_multiscale`) |
| [`ls.fos`](fragmentation.md#landschaft.fos) | Fragmentation at a Fixed Observation Scale (GuidosToolbox *Fragmentation -> FOS*) |
| [`ls.FOSResult`](fragmentation.md#landschaft.FOSResult) | Result of `fos` |
| [`ls.landscape_mosaic`](fragmentation.md#landschaft.landscape_mosaic) | Compute Landscape Mosaic (LM) moving-window metric matching GuidosToolbox batch_lm |
| [`ls.landscape_mosaic_stats`](fragmentation.md#landschaft.landscape_mosaic_stats) | Compute Landscape Mosaic 103-class frequency distribution matching GTB heatmap.csv |
| [`ls.landscape_mosaic_heatmap`](fragmentation.md#landschaft.landscape_mosaic_heatmap) | Heatmap of a Landscape Mosaic, as GTB computes it |
| [`ls.LMHeatmap`](fragmentation.md#landschaft.LMHeatmap) | Landscape Mosaic heatmap of one map (GTB ``*_heatmap.sav`` / ``.csv``) |
| [`ls.contagion_pff`](fragmentation.md#landschaft.contagion_pff) | Compute moving-window Contagion (Pff) matching GuidosToolbox batch_pff |
| [`ls.clustering_fac`](fragmentation.md#landschaft.clustering_fac) | Compute moving-window Foreground Area Clustering (FAC) matching GuidosToolbox batch_fac |
| [`ls.spatial_entropy`](fragmentation.md#landschaft.spatial_entropy) | Compute global normalized spatial joint directional entropy index |
| [`ls.SpatialEntropyResult`](fragmentation.md#landschaft.SpatialEntropyResult) | Result of global spatial entropy analysis (`spatial_entropy`) |
| [`ls.spatial_entropy_map`](fragmentation.md#landschaft.spatial_entropy_map) | Compute moving-window spatial entropy map matching GuidosToolbox batch_ent |
| [`ls.gray_spatial_convolution`](fragmentation.md#landschaft.gray_spatial_convolution) | Compute moving-window spatial convolution metric over a grayscale raster |

### Landscape metrics

| Name | Summary |
|---|---|
| [`ls.Landscape`](landscape-metrics.md#landschaft.Landscape) | Categorical raster prepared for landscape metrics |
| [`ls.landscape_metrics`](landscape-metrics.md#landschaft.landscape_metrics) | Landscape metrics of one or several rasters, in long format |
| [`ls.list_metrics`](landscape-metrics.md#landschaft.list_metrics) | Available landscape metrics |
| [`ls.sample_metrics`](landscape-metrics.md#landschaft.sample_metrics) | Landscape metrics of sample plots, like landscapemetrics' ``sample_lsm`` |
| [`ls.scale_sample`](landscape-metrics.md#landschaft.scale_sample) | Landscape metrics of buffers of several sizes around points, like landscapemetrics' ``scale_sample`` |
| [`ls.extract_metrics`](landscape-metrics.md#landschaft.extract_metrics) | Patch metrics of the patches under points, like landscapemetrics' ``extract_lsm`` |
| [`ls.window_metrics`](landscape-metrics.md#landschaft.window_metrics) | Landscape-level metrics in a moving window, like landscapemetrics' ``window_lsm`` |
| [`ls.zonal_metrics`](landscape-metrics.md#landschaft.zonal_metrics) | Landscape and class metrics of every zone, as a GeoDataFrame |
| [`ls.grid_zones`](landscape-metrics.md#landschaft.grid_zones) | Square zones tiling a raster, for zonal analyses |
| [`ls.SpatioTemporalAnalysis`](landscape-metrics.md#landschaft.SpatioTemporalAnalysis) | Landscape metrics along a time series of rasters (pylandstats' ``SpatioTemporalAnalysis`` and its zonal, buffer and grid variants) |
| [`ls.SpatialSignatureAnalysis`](landscape-metrics.md#landschaft.SpatialSignatureAnalysis) | Landscapes (or zones) described by their metrics: the "spatial signature" matrix, its principal components and clusters (pylandstats' ``SpatialSignatureAnalysis``) |
| [`ls.lsm.PCAResult`](landscape-metrics.md#landschaft.lsm.PCAResult) | Principal components of the spatial signatures |
| [`ls.lsm.NMFResult`](landscape-metrics.md#landschaft.lsm.NMFResult) | Non-negative factors of the spatial signatures (min-max scaled metrics ~ scores @ components.T) |
| [`ls.lsm.ClustergramResult`](landscape-metrics.md#landschaft.lsm.ClustergramResult) | Clusters for a range of k (Schonlau's clustergram, as the clustergram package that pylandstats' ``get_cgram`` uses) |
| [`ls.show_patches`](landscape-metrics.md#landschaft.show_patches) | Map of the patches, each in its own colour (landscapemetrics' ``show_patches``) |
| [`ls.show_cores`](landscape-metrics.md#landschaft.show_cores) | Core and edge cells of the patches (landscapemetrics' ``show_cores``): core cells dark, edge cells light, other classes blank |
| [`ls.show_lsm`](landscape-metrics.md#landschaft.show_lsm) | A patch-level metric painted on the patches (landscapemetrics' ``show_lsm``), with a colour bar; ``labels`` writes the values |
| [`ls.show_correlation`](landscape-metrics.md#landschaft.show_correlation) | Heatmap of the correlations between metrics (landscapemetrics' ``show_correlation``), one panel per level |
| [`ls.check_landscape`](landscape-metrics.md#landschaft.check_landscape) | Is a raster suitable for landscape metrics? Like landscapemetrics' ``check_landscape`` |
| [`ls.calculate_correlation`](landscape-metrics.md#landschaft.calculate_correlation) | Correlation between metrics, like landscapemetrics' ``calculate_correlation`` |
| [`ls.raster_boundaries`](landscape-metrics.md#landschaft.raster_boundaries) | landscapemetrics' ``get_boundaries`` on the values of a raster |
| [`ls.construct_buffer`](landscape-metrics.md#landschaft.construct_buffer) | Square or circular sample plots around points, like landscapemetrics' ``construct_buffer`` |
| [`ls.pad_raster`](landscape-metrics.md#landschaft.pad_raster) | Add rows and columns of ``pad_raster_value`` around a raster, like landscapemetrics' ``pad_raster`` |
| [`ls.unpad_raster`](landscape-metrics.md#landschaft.unpad_raster) | Remove ``unpad_raster_cells`` rows and columns on each side of a raster, like landscapemetrics' ``unpad_raster`` (the inverse of `pad_raster`) |
| [`ls.raster_to_points`](landscape-metrics.md#landschaft.raster_to_points) | Cell centres and values, like landscapemetrics' ``raster_to_points`` |
| [`ls.get_unique_values`](landscape-metrics.md#landschaft.get_unique_values) | The distinct values of rasters (NA left out, doubles truncated to integers), sorted, like landscapemetrics' ``get_unique_values`` |

### Habitat networks

| Name | Summary |
|---|---|
| [`ls.HabitatGraph`](networks.md#landschaft.HabitatGraph) | Habitat patches (nodes with an attribute) and the connections between them |
| [`ls.conn.node_table`](networks.md#landschaft.conn.node_table) | Node ids and attributes, as Makurhini's ``nodesfile`` |
| [`ls.conn.distance_table`](networks.md#landschaft.conn.distance_table) | Euclidean distances between nodes, as Makurhini's ``distancefile`` |
| [`ls.conn.settlement_probabilities`](networks.md#landschaft.conn.settlement_probabilities) | Probabilities of settling in one patch from another under a spatial absorbing Markov chain |
| [`ls.conn.boundary_vertices`](networks.md#landschaft.conn.boundary_vertices) | Vertices of a polygon picked as Makurhini's ``SR_vertices`` does (systematic) |
| [`ls.conn.protconn`](networks.md#landschaft.conn.protconn) | Protected Connected land (ProtConn) and its fractions, as Makurhini's MK_ProtConn (Saura et al. 2017, 2018) |
| [`ls.conn.protconn_regions`](networks.md#landschaft.conn.protconn_regions) | ProtConn of several regions, one row per region and distance (Makurhini's ``MK_ProtConnMult``) |
| [`ls.conn.protconn_raster`](networks.md#landschaft.conn.protconn_raster) | ProtConn of a region from a raster of protected areas, as Makurhini's ``MK_ProtConn_raster`` |
| [`ls.conn.protconn_nodes`](networks.md#landschaft.conn.protconn_nodes) | Nodes of ProtConn in a region, as Makurhini's ``Protconn_nodes`` builds them |
| [`ls.conn.ProtConnNodes`](networks.md#landschaft.conn.ProtConnNodes) | The nodes of a ProtConn analysis of one region (`protconn_nodes`) |
| [`ls.conn.protconn_indicators`](networks.md#landschaft.conn.protconn_indicators) | ProtConn and its fractions from prepared nodes and distances (Makurhini's ``get_protconn_grid``) |
| [`ls.conn.make_grid`](networks.md#landschaft.conn.make_grid) | Grid of cells of `cell_area` over a region, as Makurhini's ``make_grid`` |
| [`ls.conn.connectivity_grid`](networks.md#landschaft.conn.connectivity_grid) | ProtConn (or PC and ECA) in every cell of a grid, as Makurhini's ``MK_Connect_grid`` |
| [`ls.conn.eca_change_grid`](networks.md#landschaft.conn.eca_change_grid) | ECA change between times in every cell of a grid, as Makurhini's ``MK_dECA_grid`` |
| [`ls.conn.eca_change`](networks.md#landschaft.conn.eca_change) | ECA of habitat graphs over time and its change, as Makurhini's ``MK_dECA`` |
| [`ls.conn.focal_nodes`](networks.md#landschaft.conn.focal_nodes) | IIC or PC of every node's neighbourhood and the node's importance in it (``MK_Focal_nodes``) |
| [`ls.gtn_components`](networks.md#landschaft.gtn_components) | Components of the habitat network formed by MSPA Cores and Bridges |
| [`ls.GTNComponentsResult`](networks.md#landschaft.GTNComponentsResult) | Result of `gtn_components` |
| [`ls.gtn_importance`](networks.md#landschaft.gtn_importance) | Importance of every link (Bridge) and stepping-stone node (Core) of the MSPA network |
| [`ls.GTNImportanceResult`](networks.md#landschaft.GTNImportanceResult) | Result of `gtn_importance` |
| [`ls.conefor_inputs`](networks.md#landschaft.conefor_inputs) | Nodes and links of the MSPA network as Conefor Sensinode input files |

### Circuit connectivity

| Name | Summary |
|---|---|
| [`ls.omniscape`](circuits.md#landschaft.circuit.omniscape.omniscape) | Omnidirectional connectivity with Omniscape's moving-window circuits |
| [`circuit.omniscape_ini`](circuits.md#landschaft.circuit.omniscape_ini) | Run an Omniscape INI file, as ``run_omniscape(path)`` does |
| [`circuit.source_from_resistance`](circuits.md#landschaft.circuit.source_from_resistance) | Source strengths from the resistance, as Omniscape's ``source_from_resistance`` |
| [`circuit.read_reclass_table`](circuits.md#landschaft.circuit.read_reclass_table) | Read an Omniscape reclass table |
| [`circuit.reclassify`](circuits.md#landschaft.circuit.reclassify) | Reclassify a resistance array, as Omniscape's ``reclassify_resistance!`` |
| [`circuit.parse_ini`](circuits.md#landschaft.circuit.parse_ini) | Read an INI file as Omniscape's ``parse_cfg`` does |
| [`circuit.circuitscape_pairwise`](circuits.md#landschaft.circuit.circuitscape_pairwise) | Circuitscape's pairwise mode on rasters (``scenario = pairwise``) |
| [`circuit.circuitscape_one_to_all`](circuits.md#landschaft.circuit.circuitscape_one_to_all) | Circuitscape's one-to-all (or, with ``all_to_one``, all-to-one) mode on rasters |
| [`circuit.circuitscape_ini`](circuits.md#landschaft.circuit.circuitscape_ini) | Run a Circuitscape INI file, as ``Circuitscape.compute(path)`` does |
| [`circuit.CircuitscapeResult`](circuits.md#landschaft.circuit.CircuitscapeResult) | Circuitscape's outputs of a raster run |
| [`circuit.FocalPoints`](circuits.md#landschaft.circuit.FocalPoints) | Focal points as Circuitscape reads them: 1-based rows and columns and ids, sorted by id |
| [`circuit.IncludedPairs`](circuits.md#landschaft.circuit.IncludedPairs) | Circuitscape's include / exclude pairs |
| [`circuit.circuitscape_advanced`](circuits.md#landschaft.circuit.circuitscape_advanced) | Circuitscape's advanced mode on rasters (``scenario = advanced``) |
| [`circuit.advanced`](circuits.md#landschaft.circuit.advanced.advanced) | Currents and voltages of a raster circuit with sources and grounds |
| [`circuit.AdvancedResult`](circuits.md#landschaft.circuit.AdvancedResult) | Node currents and voltages of `advanced` |
| [`circuit.circuitscape_network_pairwise`](circuits.md#landschaft.circuit.circuitscape_network_pairwise) | Circuitscape's pairwise mode on a network (``data_type = network``, ``scenario = pairwise``) |
| [`circuit.circuitscape_network_advanced`](circuits.md#landschaft.circuit.circuitscape_network_advanced) | Circuitscape's advanced mode on a network (``data_type = network``, ``scenario = advanced``) |
| [`circuit.NetworkResult`](circuits.md#landschaft.circuit.NetworkResult) | Circuitscape's network outputs, with the node numbers it writes (from 1) |
| [`circuit.read_graph`](circuits.md#landschaft.circuit.read_graph) | Read a Circuitscape edge list, as its ``load_graph`` |

### Surface metrics

| Name | Summary |
|---|---|
| [`ls.surface.metrics`](surface.md#landschaft.surface.metrics) | Surface metrics of a raster, as a Series indexed by metric |
| [`ls.surface.list_metrics`](surface.md#landschaft.surface.list_metrics) | The surface metrics `metrics` computes |
| [`ls.surface.sa`](surface.md#landschaft.surface.sa) | Average roughness (Sa): the mean absolute deviation from the mean height |
| [`ls.surface.sq`](surface.md#landschaft.surface.sq) | Root mean square roughness (Sq): the root mean square of the heights about their mean, sqrt(sum((z - mean(z))^2) / N) (ISO 25178; geodiv uses the sample standard deviation, over N - 1) |
| [`ls.surface.ssk`](surface.md#landschaft.surface.ssk) | Skewness of the height distribution (Ssk) |
| [`ls.surface.sku`](surface.md#landschaft.surface.sku) | Kurtosis of the height distribution (Sku) |
| [`ls.surface.sv`](surface.md#landschaft.surface.sv) | Maximum valley depth (Sv): the absolute value of the lowest height |
| [`ls.surface.sph`](surface.md#landschaft.surface.sph) | Maximum peak height (Sp): the absolute value of the highest height |
| [`ls.surface.smean`](surface.md#landschaft.surface.smean) | Mean peak height: the mean of the positive heights |
| [`ls.surface.sdq`](surface.md#landschaft.surface.sdq) | Root mean square slope (Sdq) |
| [`ls.surface.sdq6`](surface.md#landschaft.surface.sdq6) | Area root mean square slope (Sdq6), from seven-point derivatives |
| [`ls.surface.sdr`](surface.md#landschaft.surface.sdr) | Developed interfacial area ratio (Sdr), in percent |
| [`ls.surface.surface_area`](surface.md#landschaft.surface.surface_area) | Area of the surface, triangulated, in geodiv's units |
| [`ls.surface.flat_area`](surface.md#landschaft.surface.flat_area) | Area of the surface's projection on a plane (geodiv's ``flatsa``) |
| [`ls.surface.find_peaks`](surface.md#landschaft.surface.find_peaks) | The peaks of a surface: cells higher than their 8 neighbours and than 0 |
| [`ls.surface.find_valleys`](surface.md#landschaft.surface.find_valleys) | The valleys of a surface: cells lower than their 8 neighbours and than 0 |
| [`ls.surface.sds`](surface.md#landschaft.surface.sds) | Density of summits (Sds): peaks per cell |
| [`ls.surface.ssc`](surface.md#landschaft.surface.ssc) | Mean summit curvature (Ssc) |
| [`ls.surface.s10z`](surface.md#landschaft.surface.s10z) | Ten-point height (S10z) |
| [`ls.surface.bearing_area`](surface.md#landschaft.surface.bearing_area) | The bearing area (Abbott-Firestone) curve: the share of the surface at or above each height |
| [`ls.surface.height_ba`](surface.md#landschaft.surface.height_ba) | Height of the bearing area curve at a bearing fraction |
| [`ls.surface.sdc`](surface.md#landschaft.surface.sdc) | Areal material ratio height difference (Sdc) |
| [`ls.surface.sbi`](surface.md#landschaft.surface.sbi) | Surface bearing index (Sbi): Sq divided by the height at 5 % bearing |
| [`ls.surface.svi`](surface.md#landschaft.surface.svi) | Valley fluid retention index (Svi): the void volume of the valley zone |
| [`ls.surface.sci`](surface.md#landschaft.surface.sci) | Core fluid retention index (Sci): the void volume of the core zone |
| [`ls.surface.sk`](surface.md#landschaft.surface.sk) | Core roughness depth (Sk): the height range of the core line |
| [`ls.surface.spk`](surface.md#landschaft.surface.spk) | Reduced peak height (Spk): the peaks above the core |
| [`ls.surface.svk`](surface.md#landschaft.surface.svk) | Reduced valley depth (Svk): the valleys below the core |
| [`ls.surface.aacf`](surface.md#landschaft.surface.aacf) | Areal autocorrelation function of a surface |
| [`ls.surface.scl`](surface.md#landschaft.surface.scl) | Correlation lengths: how far the autocorrelation takes to decay |
| [`ls.surface.stxr`](surface.md#landschaft.surface.stxr) | Texture aspect ratio (Str): fastest over slowest decay length |
| [`ls.surface.std`](surface.md#landschaft.surface.std) | Texture direction (Std) and texture direction index (Stdi) |
| [`ls.surface.srw`](surface.md#landschaft.surface.srw) | Dominant radial wavelength (Srw), its index (Srwi) and the half-power wavelength (Shw) |
| [`ls.surface.sfd`](surface.md#landschaft.surface.sfd) | Fractal dimension of the surface (Sfd), by triangular prisms |
| [`ls.surface.window_metrics`](surface.md#landschaft.surface.window_metrics) | Surface metrics in a moving window: one texture image per metric |
| [`ls.surface.terrain`](surface.md#landschaft.surface.terrain) | Slope or aspect of a DEM, as terra's ``terrain`` |
| [`ls.surface.tpi`](surface.md#landschaft.surface.tpi) | Topographic position index: the elevation minus the mean of the window |
| [`ls.surface.tri`](surface.md#landschaft.surface.tri) | Terrain ruggedness index (Riley et al. 1999) |
| [`ls.surface.vrm`](surface.md#landschaft.surface.vrm) | Vector ruggedness measure (Sappington et al. 2007) |
| [`ls.surface.srr`](surface.md#landschaft.surface.srr) | Surface relief ratio (Pike & Wilson 1971): (mean - min) / (max - min) of the window |
| [`ls.surface.dissection`](surface.md#landschaft.surface.dissection) | Martonne's modified dissection: (z - min) / (max - min) of the window |
| [`ls.surface.curvature`](surface.md#landschaft.surface.curvature) | Surface curvature of a DEM |
| [`ls.surface.spherical_sd`](surface.md#landschaft.surface.spherical_sd) | Spherical standard deviation of the slope and aspect (terrain heterogeneity) |
| [`ls.surface.hsp`](surface.md#landschaft.surface.hsp) | Hierarchical slope position (Murphy et al. 2010) |
| [`ls.surface.trasp`](surface.md#landschaft.surface.trasp) | Transformed aspect (Roberts & Cooper 1989): (1 - cos(aspect - 30 deg)) / 2 |
| [`ls.surface.hli`](surface.md#landschaft.surface.hli) | Heat load index (McCune & Keon 2002) |
| [`ls.surface.geomorphons`](surface.md#landschaft.surface.geomorphons) | Landforms of a DEM by geomorphons (Jasiewicz & Stepinski 2013) |
| [`ls.surface.Geomorphons`](surface.md#landschaft.surface.Geomorphons) | The geomorphons of a DEM (`geomorphons`) |
| [`ls.surface.hilltop`](surface.md#landschaft.surface.hilltop) | Hilltops as permanent protection areas (APP de topo de morro) |
| [`ls.surface.Hilltop`](surface.md#landschaft.surface.Hilltop) | The hilltops of a DEM (`hilltop`) |
| [`ls.surface.gravity`](surface.md#landschaft.surface.gravity) | Gravitational interaction potential of source cells |
| [`ls.surface.fit_plane`](surface.md#landschaft.surface.fit_plane) | The least-squares polynomial surface of a given order |
| [`ls.surface.best_fit_plane`](surface.md#landschaft.surface.best_fit_plane) | The polynomial surface of order 0 to 3 that fits a surface best |
| [`ls.surface.remove_plane`](surface.md#landschaft.surface.remove_plane) | A surface minus its best-fitting polynomial surface (geodiv's ``remove_plane``) |
| [`ls.surface.example_data`](surface.md#landschaft.surface.example_data) | Path of an example surface |

### Terrain hydrology

| Name | Summary |
|---|---|
| [`ls.hydro.fill`](hydro.md#landschaft.hydro.fill) | Fill the depressions of a DEM, so that water can flow from every cell to the edge |
| [`ls.hydro.breach`](hydro.md#landschaft.hydro.breach) | Remove the depressions of a DEM by cutting channels through their rims |
| [`ls.hydro.flow_direction`](hydro.md#landschaft.hydro.flow_direction) | The direction each cell drains to |
| [`ls.hydro.flow_accumulation`](hydro.md#landschaft.hydro.flow_accumulation) | The area that drains through each cell (the cell itself included) |
| [`ls.hydro.streams`](hydro.md#landschaft.hydro.streams) | The stream cells: those whose flow accumulation is at least ``threshold`` |
| [`ls.hydro.distance_to_channel`](hydro.md#landschaft.hydro.distance_to_channel) | The mean distance water travels from each cell to a stream, along pygeoprocessing's multiple flow directions (as InVEST's sediment delivery model measures its downslope component) |
| [`ls.hydro.stream_links`](hydro.md#landschaft.hydro.stream_links) | The links of a stream network: the stretches between sources, junctions and outlets |
| [`ls.hydro.watersheds`](hydro.md#landschaft.hydro.watersheds) | The area draining to each seed: every cell labelled with the first seed below it |
| [`ls.hydro.snap_outlets`](hydro.md#landschaft.hydro.snap_outlets) | The stream cell each outlet point stands for |
| [`ls.hydro.hand`](hydro.md#landschaft.hydro.hand) | Height Above the Nearest Drainage (HAND; Rennó et al. 2008, Nobre et al. 2011) |
| [`ls.hydro.twi`](hydro.md#landschaft.hydro.twi) | Topographic wetness index: ln(a / tan(slope)) (Beven & Kirkby 1979) |
| [`ls.hydro.spi`](hydro.md#landschaft.hydro.spi) | Stream power index: a · tan(slope) (Moore, Grayson & Ladson 1991), the erosive power of overland flow |
| [`ls.hydro.sti`](hydro.md#landschaft.hydro.sti) | Sediment transport index, the LS factor of the USLE from the contributing area (Moore & Burch 1986): (m + 1) (a / 22.13)^m (sin(slope) / 0.0896)^n |
| [`ls.hydro.connectivity_index`](hydro.md#landschaft.hydro.connectivity_index) | The index of connectivity of Borselli et al. (2008), as InVEST's SDR model computes it |

### Watershed model (SWAT+)

| Name | Summary |
|---|---|
| [`ls.hydro.swat.load_txtinout`](swat.md#landschaft.hydro.swat.load_txtinout) | Read a SWAT+ TxtInOut folder, or a zip file of one (the outputs of earlier runs in it are left out) |
| [`ls.hydro.swat.SwatProject`](swat.md#landschaft.hydro.swat.SwatProject) | The input files of a SWAT+ model (the TxtInOut folder), by name |
| [`ls.hydro.swat.run`](swat.md#landschaft.hydro.swat.run) | Simulate a SWAT+ project, as SWAT+ 62.0.1 does |
| [`ls.hydro.swat.SwatResult`](swat.md#landschaft.hydro.swat.SwatResult) | What `run` simulated |
| [`ls.hydro.swat.watershed`](swat.md#landschaft.hydro.swat.watershed) | Divide a watershed into subbasins, channels and HRUs, as QSWAT+ does |
| [`ls.hydro.swat.SwatWatershed`](swat.md#landschaft.hydro.swat.SwatWatershed) | A watershed divided as QSWAT+ divides it, ready for the SWAT+ Editor's import |
| [`ls.hydro.swat.from_tables`](swat.md#landschaft.hydro.swat.from_tables) | A SWAT+ project from QSWAT+'s tables and the weather, as the SWAT+ Editor makes it |
| [`ls.hydro.swat.from_rasters`](swat.md#landschaft.hydro.swat.from_rasters) | A SWAT+ project from a DEM, land use and soil maps, tables and the weather |
| [`ls.hydro.swat.load_qswat_project`](swat.md#landschaft.hydro.swat.load_qswat_project) | The tables of a QSWAT+ project database, for `from_tables` |
| [`ls.hydro.swat.example_landscape`](swat.md#landschaft.hydro.swat.example_landscape) | The inputs of a small watershed model, from the package's example rasters |
| [`ls.hydro.swat.climate`](swat.md#landschaft.hydro.swat.climate) | The weather of a SWAT+ project from tables |
| [`ls.hydro.swat.SwatClimate`](swat.md#landschaft.hydro.swat.SwatClimate) | The weather of a SWAT+ project (made by `climate`) |
| [`ls.hydro.swat.wgn_from_series`](swat.md#landschaft.hydro.swat.wgn_from_series) | SWAT's weather generator statistics (WXGEN) of each station, from daily series |
| [`ls.hydro.swat.hru_map`](swat.md#landschaft.hydro.swat.hru_map) | An HRU result on the map: each cell takes the value of its HRU |
| [`ls.hydro.swat.channel_map`](swat.md#landschaft.hydro.swat.channel_map) | A channel result on the stream cells (by default the mean discharge, m3/s) |
| [`ls.hydro.swat.reach_discharge`](swat.md#landschaft.hydro.swat.reach_discharge) | The daily discharge leaving each channel (m3/s, ``channel_sd``'s ``flo_out``): dates x channel number. ``result``: a run that printed ``channel_sd`` by day |

### Grid distances

| Name | Summary |
|---|---|
| [`ls.cost.transition`](cost.md#landschaft.cost.transition) | Transition matrix from a raster: gdistance's ``transition()`` |
| [`ls.cost.Transition`](cost.md#landschaft.cost.Transition) | A transition matrix on a raster grid: gdistance's ``TransitionLayer`` |
| [`ls.cost.geo_correction`](cost.md#landschaft.cost.geo_correction) | Correct a transition for the distances between cell centres: gdistance's ``geoCorrection()`` |
| [`ls.cost.normalize`](cost.md#landschaft.cost.normalize) | Normalise a transition matrix: gdistance's ``normalize()`` |
| [`ls.cost.sum_reciprocal`](cost.md#landschaft.cost.sum_reciprocal) | Combine two transitions as resistances in series: gdistance's ``sumReciprocal()`` |
| [`ls.cost.acc_cost`](cost.md#landschaft.cost.acc_cost) | Accumulated cost surface from the nearest origin: gdistance's ``accCost()`` |
| [`ls.cost.cost_distance`](cost.md#landschaft.cost.cost_distance) | Least-cost distances between points: gdistance's ``costDistance()`` |
| [`ls.cost.shortest_path`](cost.md#landschaft.cost.shortest_path) | Least-cost path from an origin to one or more goals: gdistance's ``shortestPath()`` |
| [`ls.cost.commute_distance`](cost.md#landschaft.cost.commute_distance) | Expected commute distances of a random walk between points: gdistance's ``commuteDistance()`` |
| [`ls.cost.passage`](cost.md#landschaft.cost.passage) | Passages of a random walk from origins to goals: gdistance's ``passage()`` |
| [`ls.cost.rsp_distance`](cost.md#landschaft.cost.rsp_distance) | Randomised-shortest-path distances between points: gdistance's ``rSPDistance()`` |
| [`ls.cost.path_inc`](cost.md#landschaft.cost.path_inc) | Overlap and non-overlap of the routes from a common origin to each location: gdistance's ``pathInc()`` |
| [`ls.cost.overlap`](cost.md#landschaft.cost.overlap) | gdistance's ``overlap``: the flow two routes share on each connection, ``min(a, b)`` |
| [`ls.cost.nonoverlap`](cost.md#landschaft.cost.nonoverlap) | gdistance's ``nonoverlap``: ``max(max(a, b) * (1 - min(a, b)) - min(a, b), 0)`` |
| [`ls.cost.barriers`](cost.md#landschaft.cost.barriers) | Barrier transitions: gdistance's ``transition(x, "barriers", directions)`` |
| [`ls.cost.areas`](cost.md#landschaft.cost.areas) | Area transitions: gdistance's ``transition(x, "areas", directions)`` |
| [`ls.cost.example_data`](cost.md#landschaft.cost.example_data) | Path of an example data set for the grid distances |

### Distance & restoration

| Name | Summary |
|---|---|
| [`ls.euclidean_distance`](distance.md#landschaft.euclidean_distance) | Euclidean distance to the habitat edge, inside and outside the habitat (GTB's ``batch_eucldist``) |
| [`ls.EuclideanDistanceResult`](distance.md#landschaft.EuclideanDistanceResult) | Result of `euclidean_distance` |
| [`ls.distance_to_resistance`](distance.md#landschaft.distance_to_resistance) | Resistance growing with the distance to the foreground (GTB Setup Tools -> Distance -> Resistance) |
| [`ls.restoration_status`](distance.md#landschaft.restoration_status) | How connected the habitat is and how much reconnecting it could add (GTB's Restoration Status, ``batch_rss``) |
| [`ls.RestorationStatusResult`](distance.md#landschaft.RestorationStatusResult) | Result of `restoration_status` |
| [`ls.cost_distance`](distance.md#landschaft.cost_distance) | Accumulated cost of travelling from the source cells across a resistance surface (8 neighbours) |
| [`ls.least_cost_path`](distance.md#landschaft.least_cost_path) | The least-cost path between two areas A and B |
| [`ls.LeastCostPathResult`](distance.md#landschaft.LeastCostPathResult) | Result of `least_cost_path` |
| [`ls.restoration_corridor`](distance.md#landschaft.restoration_corridor) | Connectivity gain of restoring the least-cost corridor between two areas (GTB's Restoration Planner) |
| [`ls.RestorationCorridorResult`](distance.md#landschaft.RestorationCorridorResult) | Result of `restoration_corridor` |
| [`ls.isochrone_map`](distance.md#landschaft.isochrone_map) | Accumulated cost from object A (GTB Restoration Planner -> Isochrone Map A) |
| [`ls.isochrone_map_ab`](distance.md#landschaft.isochrone_map_ab) | Cheapest route between objects A and B (GTB Restoration Planner -> Isochrone Map AB) |
| [`ls.IsochroneResult`](distance.md#landschaft.IsochroneResult) | Result of `isochrone_map` / `isochrone_map_ab` |
| [`ls.influence_zones`](distance.md#landschaft.influence_zones) | Zones of influence of the foreground objects (GTB Distance -> Influence Zones) |
| [`ls.InfluenceZonesResult`](distance.md#landschaft.InfluenceZonesResult) | Result of `influence_zones` |
| [`ls.proximity`](distance.md#landschaft.proximity) | Proximity between neighbouring objects (GTB Distance -> Proximity) |
| [`ls.ProximityResult`](distance.md#landschaft.ProximityResult) | Result of `proximity` |
| [`ls.original_vegetation`](distance.md#landschaft.original_vegetation) | The original vegetation of converted land, and what restoration brings back |
| [`ls.OriginalVegetation`](distance.md#landschaft.OriginalVegetation) | Result of `original_vegetation` |
| [`ls.intactness`](distance.md#landschaft.intactness) | Ecological intactness from the human footprint (Beyer et al. 2020) |
| [`ls.intactness_gradient`](distance.md#landschaft.intactness_gradient) | What restoring each cell adds to the total ecological intactness |

### Change detection

| Name | Summary |
|---|---|
| [`ls.simple_change`](change.md#landschaft.simple_change) | Pixel-by-pixel change of a habitat map between two dates |
| [`ls.SimpleChangeResult`](change.md#landschaft.SimpleChangeResult) | Result of `simple_change` |
| [`ls.morphological_change`](change.md#landschaft.morphological_change) | Change of a habitat map that separates real structural change from edge noise (GTB's Morphological Change Detection) |
| [`ls.MorphologicalChangeResult`](change.md#landschaft.MorphologicalChangeResult) | Result of `morphological_change` |
| [`ls.fos_change`](change.md#landschaft.fos_change) | Change of fragmentation between two dates (GTB's FOS change) |
| [`ls.FOSChangeResult`](change.md#landschaft.FOSChangeResult) | Result of `fos_change` |
| [`ls.fad_multiscale_change`](change.md#landschaft.fad_multiscale_change) | Fragmentation class change between two FAD-MS analyses (GTB Change -> FAD Multiscale) |
| [`ls.FADMultiscaleChangeResult`](change.md#landschaft.FADMultiscaleChangeResult) | Result of `fad_multiscale_change` |
| [`ls.FADChangeScale`](change.md#landschaft.FADChangeScale) | FAD change A -> B at one observation scale (one GTB ``FAD_change_<scale>`` table) |
| [`ls.landscape_mosaic_change`](change.md#landschaft.landscape_mosaic_change) | Change of the Landscape Mosaic heatmap from A to B (GTB Change -> Landscape Mosaic Heatmap) |
| [`ls.LMHeatmapChange`](change.md#landschaft.LMHeatmapChange) | Result of `landscape_mosaic_change`: the heatmap difference B - A (GTB's ``delta_heatmap.csv``) |

### Large data & batch

| Name | Summary |
|---|---|
| [`ls.batch`](batch.md#landschaft.batch) | Parallel processing of many rasters, one thread per raster |
| [`ls.mspa_batch`](batch.md#landschaft.mspa_batch) | Run MSPA on many rasters in parallel and write the results |
| [`ls.BatchResult`](batch.md#landschaft.BatchResult) | Outcome of one input of `batch` / `mspa_batch` |
| [`ls.batch_status`](batch.md#landschaft.batch_status) | What the runs of `batch` / `mspa_batch` did in an output folder: the inputs done, failed (with their error) and not done, and the job array tasks to submit again |
| [`ls.Stopped`](batch.md#landschaft.batch.Stopped) | A batch stopped by SIGTERM (a scheduler's warning before the job's time limit, or its cancellation): no input started after it; ``results`` holds those that ended, in input order (``None`` for the inputs not processed) |
| [`ls.helpers`](batch.md#landschaft.parallel.helpers) | This computer and the helpers of its group, for ``on=`` (see `Helpers`) |
| [`ls.Helpers`](batch.md#landschaft.parallel.Helpers) | This computer and the helpers of its group, for the ``on=`` argument of landschaft's functions that run many independent tasks |
| [`ls.map`](batch.md#landschaft.parallel.map) | Run a tool on many inputs, on this computer or on helpers, in parallel |
| [`ls.tiled`](batch.md#landschaft.parallel.tiled) | Run a neighbourhood tool on a large raster tile by tile, on this computer or on helpers |
| [`ls.call`](batch.md#landschaft.parallel.call) | A registered tool with its arguments, the drawn ones marked by `param` |
| [`ls.param`](batch.md#landschaft.parallel.param) | The ``index``-th drawn parameter, in the arguments of a `call` |
| [`ls.run_worker`](batch.md#landschaft.parallel.run_worker) | Lend this computer to the group: wait for tasks and run them, until Ctrl+C |
| [`ls.invite_helpers`](batch.md#landschaft.parallel.invite_helpers) | Show an invitation code and let other computers join this one's group with ``landschaft worker --join <code>``, for ``minutes`` |
| [`ls.list_helpers`](batch.md#landschaft.parallel.list_helpers) | The helpers of this computer's group waiting for work now (busy ones are not looking): ``[{"name", "address", "cores", "version"}]`` |
| [`ls.leave_group`](batch.md#landschaft.parallel.leave_group) | Forget this computer's group (it no longer helps nor uses helpers) |
| [`ls.open_firewall`](batch.md#landschaft.parallel.open_firewall) | Let helpers reach this computer: the Windows firewall rules for the local network |
| [`ls.mspa_tiled`](batch.md#landschaft.mspa_tiled) | MSPA of a large map by overlapping tiles, identical to GTB's MSPA Tiling |
| [`ls.tile_layout`](batch.md#landschaft.tile_layout) | GTB's tiles along one axis of ``size`` pixels |

### Neutral landscape models

| Name | Summary |
|---|---|
| [`ls.nlm.random`](nlm.md#landschaft.nlm.random) | Spatially random neutral landscape model (NLMR's ``nlm_random``) |
| [`ls.nlm.percolation`](nlm.md#landschaft.nlm.percolation) | Percolation neutral landscape model (NLMR's ``nlm_percolation``) |
| [`ls.nlm.planar_gradient`](nlm.md#landschaft.nlm.planar_gradient) | Planar gradient neutral landscape model (NLMR's ``nlm_planargradient``) |
| [`ls.nlm.edge_gradient`](nlm.md#landschaft.nlm.edge_gradient) | Edge gradient neutral landscape model (NLMR's ``nlm_edgegradient``) |
| [`ls.nlm.distance_gradient`](nlm.md#landschaft.nlm.distance_gradient) | Distance gradient neutral landscape model (NLMR's ``nlm_distancegradient``) |
| [`ls.nlm.mpd`](nlm.md#landschaft.nlm.mpd) | Midpoint displacement neutral landscape model (NLMR's ``nlm_mpd``) |
| [`ls.nlm.fbm`](nlm.md#landschaft.nlm.fbm) | Fractional Brownian motion neutral landscape model (NLMR's ``nlm_fbm``) |
| [`ls.nlm.gaussian_field`](nlm.md#landschaft.nlm.gaussian_field) | Gaussian random field neutral landscape model (NLMR's ``nlm_gaussianfield``) |
| [`ls.nlm.mosaic_field`](nlm.md#landschaft.nlm.mosaic_field) | Mosaic random field neutral landscape model (NLMR's ``nlm_mosaicfield``, Schlather 1999) |
| [`ls.nlm.mosaic_tess`](nlm.md#landschaft.nlm.mosaic_tess) | Polygonal (Voronoi) mosaic neutral landscape model (NLMR's ``nlm_mosaictess``, Gaucherel 2008) |
| [`ls.nlm.mosaic_gibbs`](nlm.md#landschaft.nlm.mosaic_gibbs) | Gibbs-process Voronoi mosaic (NLMR's ``nlm_mosaicgibbs``, Gauchel et al. 2008) |
| [`ls.nlm.random_cluster`](nlm.md#landschaft.nlm.random_cluster) | Random cluster neutral landscape model (NLMR's ``nlm_randomcluster``, Saura & Martínez-Millán 2000) |
| [`ls.nlm.neigh`](nlm.md#landschaft.nlm.neigh) | Neighbourhood-based clusters (NLMR's ``nlm_neigh``, Scherer et al. 2016) |
| [`ls.nlm.random_rectangular_cluster`](nlm.md#landschaft.nlm.random_rectangular_cluster) | Random rectangular clusters (NLMR's ``nlm_randomrectangularcluster``, Gaucherel 2008) |
| [`ls.nlm.curds`](nlm.md#landschaft.nlm.curds) | Hierarchical curdling (NLMR's ``nlm_curds``, Keitt 2000) |
| [`ls.nlm.classify`](nlm.md#landschaft.nlm.classify) | Classify a continuous landscape into categories (NLMR's ``landscape_classify``) |
| [`ls.nlm.binarize`](nlm.md#landschaft.nlm.binarize) | Habitat / matrix maps at given habitat shares (landscapetools' ``util_binarize``) |
| [`ls.nlm.rescale`](nlm.md#landschaft.nlm.rescale) | Linearly rescale values to 0-1 (NLMR's ``util_rescale``) |
| [`ls.nlm.merge`](nlm.md#landschaft.nlm.merge) | Merge neutral landscapes (landscapetools' ``util_merge``) |

### Biodiversity analysis

| Name | Summary |
|---|---|
| [`ls.bio.BaseData`](biodiversity.md#landschaft.bio.BaseData) | Groups × labels sample counts with the group coordinates |
| [`ls.bio.snap_coordinates`](biodiversity.md#landschaft.bio.snap_coordinates) | Cell centres as Biodiverse's text import computes them (before naming) |
| [`ls.bio.example_data`](biodiversity.md#landschaft.bio.example_data) | Path of Biodiverse's example data (GPL-3, Laffan et al.) |
| [`ls.bio.read_occurrences`](biodiversity.md#landschaft.bio.read_occurrences) | Occurrence records of a GBIF download or a Darwin Core Archive, cleaned |
| [`ls.bio.neighbours`](biodiversity.md#landschaft.bio.neighbours) | Neighbour sets of every group, as Biodiverse's spatial analysis finds them |
| [`ls.bio.definition_query`](biodiversity.md#landschaft.bio.definition_query) | Boolean mask of the groups passing a definition query |
| [`ls.bio.SpatialCondition`](biodiversity.md#landschaft.bio.SpatialCondition) | A Biodiverse spatial condition (Perl syntax) compiled for a BaseData |
| [`ls.bio.Neighbourhoods`](biodiversity.md#landschaft.bio.Neighbourhoods) | Neighbour sets of the processing groups |
| [`ls.bio.spatial_analysis`](biodiversity.md#landschaft.bio.spatial_analysis) | Biodiverse's spatial analysis: indices over the neighbourhood of every group |
| [`ls.bio.list_calculations`](biodiversity.md#landschaft.bio.list_calculations) | The calculations available to `spatial_analysis` |
| [`ls.bio.SpatialResult`](biodiversity.md#landschaft.bio.SpatialResult) | Results of `spatial_analysis` |
| [`ls.bio.randomise`](biodiversity.md#landschaft.bio.randomise) | Randomise the basedata of a spatial analysis and compare the results |
| [`ls.bio.Randomisation`](biodiversity.md#landschaft.bio.Randomisation) | Randomisations of a `SpatialResult` (see `randomise`) |
| [`ls.bio.cluster`](biodiversity.md#landschaft.bio.cluster) | Cluster the groups by the dissimilarity of their labels (Biodiverse's cluster analysis) |
| [`ls.bio.region_grower`](biodiversity.md#landschaft.bio.region_grower) | Grow regions by joining the pairs with the highest value of an index (Biodiverse's RegionGrower) |
| [`ls.bio.ClusterResult`](biodiversity.md#landschaft.bio.ClusterResult) | Results of `cluster` |
| [`ls.bio.Tree`](biodiversity.md#landschaft.bio.Tree) | A rooted phylogenetic tree |
| [`ls.bio.read_tree`](biodiversity.md#landschaft.bio.read_tree) | Read one tree (the first, or the one with index/name ``index``) |
| [`ls.bio.read_trees`](biodiversity.md#landschaft.bio.read_trees) | Read every tree in a Newick, Phylip (several Newicks) or Nexus file |
| [`ls.bio.Matrix`](biodiversity.md#landschaft.bio.Matrix) | A symmetric matrix of values between elements (usually labels) |
| [`ls.bio.read_matrix`](biodiversity.md#landschaft.bio.read_matrix) | Read a Biodiverse matrix file (full or lower/upper triangle, or sparse pairs) |
| [`ls.bio.hex_centres`](biodiversity.md#landschaft.bio.hex_centres) | Centres of the hexagons (circumradius ``size``) containing each point |
| [`ls.bio.h3_cells`](biodiversity.md#landschaft.bio.h3_cells) | H3 cell of each point (``h3`` package) |
| [`ls.bio.save_session`](biodiversity.md#landschaft.bio.save_session) | Save basedatas, trees, matrices and analysis results in one file |
| [`ls.bio.load_session`](biodiversity.md#landschaft.bio.load_session) | Load the objects saved with `save_session` (a dict by name) |
| [`ls.bio.read_properties`](biodiversity.md#landschaft.bio.read_properties) | Read a Biodiverse element properties file (label or group properties) |

### Spatial prioritization

| Name | Summary |
|---|---|
| [`ls.zonation`](prioritization.md#landschaft.prioritize.zonation) | Rank the cells of a landscape for conservation, as Zonation 5 does |
| [`ls.ZonationResult`](prioritization.md#landschaft.prioritize.ZonationResult) | The result of `zonation` |
| [`ls.run_project`](prioritization.md#landschaft.prioritize.run_project) | Run a Zonation 5 project (see `read_project`) |
| [`ls.read_project`](prioritization.md#landschaft.prioritize.read_project) | The `zonation` arguments of a Zonation 5 project |
| [`ls.Kernel`](prioritization.md#landschaft.prioritize.Kernel) | A dispersal kernel of Zonation 5's connectivity transforms |
| [`ls.Interaction`](prioritization.md#landschaft.prioritize.Interaction) | An interaction connectivity transform (Zonation's ``-i``/``-I``): a feature weighted by the connectivity of a resource (``positive``) or reduced by it |
| [`ls.convolve`](prioritization.md#landschaft.prioritize.convolve) | Convolve a raster with a kernel normalized to sum 1, as Zonation 5 does |
| [`ls.AreaStatistics`](prioritization.md#landschaft.prioritize.AreaStatistics) | Statistics of areas of a ranking (`area_statistics`) |
| [`ls.ImpactScenario`](prioritization.md#landschaft.prioritize.ImpactScenario) | An impact assessment scenario (Zonation's ``-s``): land-use change at sources |
| [`ls.ImpactSource`](prioritization.md#landschaft.prioritize.ImpactSource) | An impact source of an impact group (a line of Zonation's impact buffers file) |
| [`ls.CumulativeScenario`](prioritization.md#landschaft.prioritize.CumulativeScenario) | A cumulative impact assessment scenario (Zonation's ``-e``) |
| [`ls.Pressure`](prioritization.md#landschaft.prioritize.Pressure) | A pressure raster of a cumulative impact scenario |
| [`ls.problem`](prioritization.md#landschaft.prioritize.problem) | A conservation planning problem (prioritizr's ``problem()``) |
| [`ls.Problem`](prioritization.md#landschaft.prioritize.Problem) | A conservation planning problem (prioritizr's ``problem()``): see `problem` |
| [`ls.Solution`](prioritization.md#landschaft.prioritize.Solution) | A solution of a `Problem` |
| [`ls.Portfolio`](prioritization.md#landschaft.prioritize.Portfolio) | Several solutions of a problem (`add_cuts_portfolio` and the others) |
| [`ls.OptimizationProblem`](prioritization.md#landschaft.prioritize.OptimizationProblem) | A compiled model: ``modelsense`` of ``obj @ x`` subject to ``A x (sense) rhs``, ``lb <= x <= ub`` and ``x`` of type ``vtype`` (``"B"`` binary, ``"C"`` continuous, ``"I"`` integer); the layout of prioritizr's ``compile()`` |
| [`ls.h3_units`](prioritization.md#landschaft.prioritize.h3_units) | H3 hexagons as planning units: the hexagons of an H3 ``resolution`` that cover a cost raster, with their cost |
| [`ls.example_data`](prioritization.md#landschaft.prioritize.example_data) | Path of an example data set for prioritization: **Serra Verde**, an imaginary coastal region of 60 x 90 cells of 1 km (SIRGAS 2000 / UTM 23S) |
| [`ls.category_layer`](prioritization.md#landschaft.prioritize.category_layer) | The zone of every cell of a stack of zone layers (prioritizr's ``category_layer()``): the number (1, 2, ...) of the layer with the largest value above 0, 0 where none is, NaN where every layer is NaN. ``stack``: a raster with a layer per zone (a solution's ``selection``) or a mapping zone -> raster |
| [`ls.binary_stack`](prioritization.md#landschaft.prioritize.binary_stack) | One 0 / 1 raster per zone from a raster of zone numbers (prioritizr's ``binary_stack()``): a mapping zone -> raster, ready for `add_locked_in_constraints` (for example an existing zoning plan). ``zones``: the zones' names in the order of their numbers (default: those of a solution's ``zones`` map, else ``"1"``, ``"2"``, ...); cells with 0 or NaN are in no zone |
| [`ls.boundary_matrix`](prioritization.md#landschaft.prioritize.boundary_matrix) | The length of the boundary every two units share, and every unit's perimeter on the diagonal (prioritizr's ``boundary_matrix()``): raster cells with data share ``xres`` with the cells above and below and ``yres`` with those left and right, perimeter ``2 (xres + yres)``; polygons share the length of their common paths, perimeters their own, rounded to 8 decimals |
| [`ls.adjacency_matrix`](prioritization.md#landschaft.prioritize.adjacency_matrix) | 1 for every two adjacent units (prioritizr's ``adjacency_matrix()``): raster cells with data that share an edge (``directions`` 4) or also a corner (8); polygons that intersect, touching at a corner included |
| [`ls.connectivity_matrix`](prioritization.md#landschaft.prioritize.connectivity_matrix) | Connectivity between adjacent units (prioritizr's ``connectivity_matrix()``): the length of their shared boundary times the mean of ``y`` in the two (``y``: a raster on the units' grid; with polygons a raster summed over each, or a column name) |
| [`ls.proximity_matrix`](prioritization.md#landschaft.prioritize.proximity_matrix) | 1 for every two units within ``distance`` of each other (prioritizr's ``proximity_matrix()``): raster cell centres, or polygons by their nearest points |
| [`ls.decay_connectivity`](prioritization.md#landschaft.prioritize.decay_connectivity) | Connectivity from distances between units as a dispersal probability (Conefor's negative exponential): ``exp(-ln(2) d / median)``, 0.5 at the ``median`` dispersal distance, dropped below ``min_probability`` (sparse), 0 on the diagonal and for unreachable pairs (``inf``, NaN) |
| [`ls.unit_points`](prioritization.md#landschaft.prioritize.unit_points) | A point per unit, (n, 2) coordinates: raster cell centres (every cell), or a point inside every polygon (shapely's point on surface), to measure distances between units with `cost_distance` or `commute_distance` |
| [`ls.eval_n_summary`](prioritization.md#landschaft.prioritize.eval_n_summary) | The number of selected planning units (the sum of the solution's values) |
| [`ls.eval_cost_summary`](prioritization.md#landschaft.prioritize.eval_cost_summary) | The solution's total cost |
| [`ls.eval_boundary_summary`](prioritization.md#landschaft.prioritize.eval_boundary_summary) | The solution's boundary length: the edges between selected and other units, plus ``edge_factor`` x the selected units' edges on the outside of the planning units |
| [`ls.eval_connectivity_summary`](prioritization.md#landschaft.prioritize.eval_connectivity_summary) | The connectivity among the selected units: the sum of ``data[i, j]`` x the solution's values of units i and j (a symmetric matrix over the units) |
| [`ls.eval_asym_connectivity_summary`](prioritization.md#landschaft.prioritize.eval_asym_connectivity_summary) | As `eval_connectivity_summary` for an asymmetric matrix (its diagonal counted once) |
| [`ls.eval_feature_representation_summary`](prioritization.md#landschaft.prioritize.eval_feature_representation_summary) | Every feature's total amount and the amount the solution holds, absolute and relative |
| [`ls.eval_target_coverage_summary`](prioritization.md#landschaft.prioritize.eval_target_coverage_summary) | Every target's feature, whether it is met, the feature's total, and the target, the amount held and the shortfall, absolute and relative (shares of the total; 0 when the total is 0) |
| [`ls.eval_replacement_importance`](prioritization.md#landschaft.prioritize.eval_replacement_importance) | The replacement cost of every selected unit (Cabeza & Moilanen 2006): how much the objective worsens when the unit is left out and the problem solved again with the other selected units kept (infinite when nothing can replace it), rescaled to 0.01-1 |
| [`ls.eval_ferrier_importance`](prioritization.md#landschaft.prioritize.eval_ferrier_importance) | Ferrier et al. (2000)'s irreplaceability of the selected units for every feature and in total (prioritizr's version: the problem needs one ">=" target per feature, amounts >= 0) |
| [`ls.eval_rare_richness_importance`](prioritization.md#landschaft.prioritize.eval_rare_richness_importance) | The rarity-weighted richness of the selected units (Albuquerque & Beier 2015): the sum over the features of the unit's amount / the feature's largest amount / its total, rescaled to 0.01-1 |
| [`ls.eval_rank_importance`](prioritization.md#landschaft.prioritize.eval_rank_importance) | Rank importance (Jung et al. 2021, prioritizr's eval_rank_importance): the problem is solved with increasing budgets (the min set objective becomes min shortfall), only the solution's units allowed and each step keeping the units of the step before; a unit's rank is how early it enters (n = the first budget, ..., 1 = the last), rescaled to 0.01-1. ``n`` budgets evenly spaced between the locked-in units' cost and the solution's, or the ``budgets`` themselves |
| [`ls.eval_extinction_risk`](prioritization.md#landschaft.prioritize.eval_extinction_risk) | Every feature's extinction risk by the species-area relationship, 1 - (held / total)^z (z = 0.25 by default, one value or one per feature), and their sum (the expected number of features lost) |
| [`ls.eval_landscape_metrics`](prioritization.md#landschaft.prioritize.eval_landscape_metrics) | Landscape metrics of a raster solution (`landscape_metrics` of the map of the selected cells: class 1 selected, 0 not, outside the planning units no data): e.g. the number, sizes and shapes of the selected patches |
| [`ls.eval_mspa`](prioritization.md#landschaft.prioritize.eval_mspa) | MSPA (`mspa`) of a raster solution: the selected cells as foreground, the other planning units as background, the cells outside them missing |
| [`ls.eval_landscape_identification`](prioritization.md#landschaft.prioritize.eval_landscape_identification) | Landscape identification: the separate areas of the top fraction of a priority surface, and what each holds (Zonation's area statistics of separate areas, over any surface) |
| [`ls.eval_landscape_comparison`](prioritization.md#landschaft.prioritize.eval_landscape_comparison) | Landscape comparison: how much the top fractions of two priority surfaces overlap (Zonation's solution comparison at several fractions, prioripy's ``eval_landscape_comparison``) |
| [`ls.position_rank`](prioritization.md#landschaft.prioritize.position_rank) | Zonation's ranks of a surface by position: the i-th lowest of the N finite cells gets float32(i * (1 / N)); ties go to the earlier cell in row order first; NaN stays NaN |
| [`ls.LandscapeAreas`](prioritization.md#landschaft.prioritize.LandscapeAreas) | The areas of the top fraction of a priority surface (`eval_landscape_identification`) |
| [`ls.LandscapeComparison`](prioritization.md#landschaft.prioritize.LandscapeComparison) | Two priority surfaces compared at several top fractions (`eval_landscape_comparison`) |
| [`ls.eval_proxirr`](prioritization.md#landschaft.prioritize.eval_proxirr) | ProxIRR (Baisero et al. 2022) of every planning unit for the problem's targets, as the CRAN package proxirr computes it |
| [`ls.eval_proxirr_marxan`](prioritization.md#landschaft.prioritize.eval_proxirr_marxan) | ProxIRR of a Marxan problem's planning units (proxirr's ``marxan_alphas``, ``marxan_betas`` and ``marxan_gammas``), from the tables `marxan` takes |
| [`ls.ProxIRR`](prioritization.md#landschaft.prioritize.ProxIRR) | ProxIRR of every planning unit (`eval_proxirr`, `eval_proxirr_marxan`) |
| [`ls.calibrate_penalty`](prioritization.md#landschaft.prioritize.calibrate_penalty) | The blended approach: the problem solved with a range of penalty weights (``values``; default ``n`` values evenly spaced in log scale over 1/100 to 100 times the Cohon et al. penalty), every candidate's cost and criterion (boundary length, connectivity, ...), and the candidates TOPSIS and the knee of the curve choose. ``p``: the problem without the penalty |
| [`ls.calibrate_budget`](prioritization.md#landschaft.prioritize.calibrate_budget) | The hierarchical approach: the least cost solution first (``p``, a min set problem without the penalty, solved to optimality), then the least penalty within cost thresholds above its cost (``thresholds``, or the cost x (1 + ``inflation``); default inflation 1e-5 to 4 in 9 steps, the thresholds rounded up as in prioritizr's tutorial) |
| [`ls.cohon_penalty`](prioritization.md#landschaft.prioritize.cohon_penalty) | The penalty of Cohon et al. (1979): the trade-off rate between the solution with the least cost (``p`` as given, a min set problem without the penalty) and the one with the least penalty (min penalties within a budget above every unit's cost): \|cost difference / criterion difference\|, rounded to 5 decimals |
| [`ls.pareto_front`](prioritization.md#landschaft.prioritize.pareto_front) | The trade-off between cost and representation: the least total shortfall of the targets (min shortfall objective: the sum of the shortfalls as shares of their targets) within ``n`` budgets evenly spaced up to the least-cost solution's cost (above the cost of the locked-in units, which every budget includes), or the ``budgets`` given. ``p``: a problem with targets (its objective is replaced) |
| [`ls.calibrate`](prioritization.md#landschaft.prioritize.calibrate) | Any trade-off parameter: ``make(value)`` gives the problem for every value (a feature's weight, a target, a penalty of any kind...), ``criterion(problem, solution)`` the quantity traded off against the cost (smaller is better) |
| [`ls.TradeOffs`](prioritization.md#landschaft.prioritize.TradeOffs) | Candidate solutions over a trade-off parameter |
| [`ls.topsis`](prioritization.md#landschaft.prioritize.topsis) | TOPSIS (as R's ``topsis`` package): the alternatives (rows of ``decision``) scored by their closeness to the ideal (``impacts``: "+" larger is better, "-" smaller is better, per column; equal ``weights`` by default) |
| [`ls.knee_point`](prioritization.md#landschaft.prioritize.knee_point) | The knee (elbow) of a trade-off curve: the point farthest from the straight line between its ends, both axes scaled to 0-1 (the points in the order of x) |
| [`ls.mean_sd`](prioritization.md#landschaft.prioritize.mean_sd) | plangea's choice of weights (``post_choose_weights``): every criterion (a column of ``decision``) rescaled to 0-1 over the alternatives (a constant one to 1), the "-" ones (smaller is better) inverted, and every alternative scored by the mean of its criteria divided by their standard deviation (good on all of them at once) |
| [`ls.iterative_solve`](prioritization.md#landschaft.prioritize.iterative_solve) | Solve, change the problem, solve again: ``update(step, solution, problem)`` gives the next problem (for instance from the landscape a ``ls.simulate`` step leaves, with the solution's cells protected by `protected_cells` in the simulation's rules), or ``None`` to stop |
| [`ls.protected_cells`](prioritization.md#landschaft.prioritize.protected_cells) | The selected cells of a raster solution as a boolean raster (for ``ls.simulate``'s restrictions) |
| [`ls.zonation_features`](prioritization.md#landschaft.prioritize.zonation_features) | The features as Zonation 5 transforms them before ranking (condition, connectivity kernels, interactions, matrix connectivity, retention, impacts: see `zonation` for each), back in the features' own units, for an optimization problem (``problem(cost, zonation_features(...))``) |
| [`ls.compare_zonation`](prioritization.md#landschaft.prioritize.compare_zonation) | Compare an optimization solution of ``p`` (solved if not given) with a Zonation ranking of the same features and costs (run with ``rule`` and ``zonation_kw`` if not given: a `ZonationResult` or a rank raster) cut at the solution's cost |
| [`ls.ZonationComparison`](prioritization.md#landschaft.prioritize.ZonationComparison) | A Zonation ranking and an optimization solution with the same budget |
| [`ls.sdm_features`](prioritization.md#landschaft.prioritize.sdm_features) | Features from species distribution models (``ls.sdm``): every result's prediction on ``env`` (its ensemble when it has several models, else its model), as a stack for `problem` |
| [`ls.report`](prioritization.md#landschaft.prioritize.report) | Solve ``p`` (unless a solution is given) and evaluate the solution: the number of units, cost, boundary (raster and polygon units), the features' representation and the targets' coverage; with ``importance`` the Ferrier irreplaceability of the selected units (problems with one target per feature) or else their rarity-weighted richness; with ``extinction_z`` the species-area extinction risk; with ``zonation`` the comparison with a Zonation ranking at the same cost (raster units) |
| [`ls.PrioritizationReport`](prioritization.md#landschaft.prioritize.PrioritizationReport) | A solution and its evaluation in one object: ``ls.plot(report)`` draws it, ``ls.plot(report).save( "report.html")`` writes it as a page that works offline, ``ls.save(report, folder)`` its tables and maps |
| [`ls.coarsen`](prioritization.md#landschaft.prioritize.coarsen) | A problem's rasters on blocks of ``factor`` x ``factor`` cells (for continental problems: solve the coarse problem, then `refine` the selection): every block's cost and amounts are the sums of its cells' with a cost (the fine planning units; a block without any has none), so the refined selection holds what the coarse one does |
| [`ls.refine`](prioritization.md#landschaft.prioritize.refine) | A selection on coarse blocks (a `Solution` of a `coarsen`-ed problem, or its selection raster) back on the fine grid of ``like`` (the fine cost raster): every cell takes its block's value, NaN where ``like`` has none |
| [`ls.actions_data`](prioritization.md#landschaft.prioritize.actions_data) | The inputs of a multi-action problem, checked and completed as prioriactions's ``inputData()`` |
| [`ls.actions_raster_data`](prioritization.md#landschaft.prioritize.actions_raster_data) | The inputs of a multi-action problem from rasters (see `actions_data` for the tables) |
| [`ls.ActionsData`](prioritization.md#landschaft.prioritize.ActionsData) | Checked inputs of a multi-action problem (prioriactions's ``inputData()``): see `actions_data` |
| [`ls.ActionsProblem`](prioritization.md#landschaft.prioritize.ActionsProblem) | A multi-action problem (prioriactions's ``problem()``): see `problem` |
| [`ls.ActionsSolution`](prioritization.md#landschaft.prioritize.ActionsSolution) | A solution of an `ActionsProblem` |
| [`ls.ActionsPortfolio`](prioritization.md#landschaft.prioritize.ActionsPortfolio) | Several solutions of an `ActionsProblem` (`add_cuts_portfolio` and the others) |
| [`ls.restoration_problem`](prioritization.md#landschaft.prioritize.restoration_problem) | A restoration problem: how much of every converted cell to restore (and of natural land to conserve or convert) |
| [`ls.RestorationProblem`](prioritization.md#landschaft.prioritize.RestorationProblem) | A restoration problem (see `restoration_problem`), built up by its ``add_*`` methods; every method returns a new problem |
| [`ls.RestorationSolution`](prioritization.md#landschaft.prioritize.RestorationSolution) | A restoration plan |
| [`ls.RestorationTradeOff`](prioritization.md#landschaft.prioritize.RestorationTradeOff) | Plans over a trade-off (`tradeoff`) or over sets of weights (`compare_weights`) |
| [`ls.ecosystem_areas`](prioritization.md#landschaft.prioritize.ecosystem_areas) | The natural area of every ecosystem (ecoregion) now and before conversion, and what restoring a cell gives it: the input of `add_extinction_risk_benefit` for the risk of ecosystem collapse (plangea's ``ec``) |
| [`ls.ecosystem_targets`](prioritization.md#landschaft.prioritize.ecosystem_targets) | Share a restoration target among ecosystems for the least risk of collapse (plangea's ecosystemic sub-region targets, ``harmonize_sr_ecosys``) |
| [`ls.eval_risk_reduction`](prioritization.md#landschaft.prioritize.eval_risk_reduction) | The reduction of the extinction risk a plan brings (plangea's ``multi_exrisk_aggregate``), with the uncertainty of the species-area exponent z |
| [`ls.RiskReduction`](prioritization.md#landschaft.prioritize.RiskReduction) | Result of `eval_risk_reduction`: per species, its habitat now (``current``), without the plan (``baseline``: less than now when conservation's baseline loses land) and gained by it, its risk before and after; the total reduction (negative when conversion raises the risk) and its standard deviation from the uncertainty of z |
| [`ls.marxan`](prioritization.md#landschaft.prioritize.marxan) | Run Marxan: ``num_reps`` independent runs of annealing and iterative improvement (or the greedy heuristics) minimising cost + ``blm`` x boundary + penalties for the features' shortfalls |
| [`ls.MarxanResult`](prioritization.md#landschaft.prioritize.MarxanResult) | The runs of `marxan` |

### Species distributions

| Name | Summary |
|---|---|
| [`ls.prepare`](sdm.md#landschaft.sdm.prepare) | Prepare occurrence records and predictors for a species distribution model, in one call |
| [`ls.download_gbif`](sdm.md#landschaft.sdm.download_gbif) | Occurrence records of a species from GBIF, with coordinates |
| [`ls.clean_occurrences`](sdm.md#landschaft.sdm.clean_occurrences) | Flag or drop occurrence records with suspicious coordinates |
| [`ls.reference_data`](sdm.md#landschaft.sdm.reference_data) | A geographic reference data set for cleaning occurrence records |
| [`ls.one_per_cell`](sdm.md#landschaft.sdm.one_per_cell) | Keep one occurrence record per cell of the predictor grid |
| [`ls.thin`](sdm.md#landschaft.sdm.thin) | Thin occurrence records so that no two kept records are closer than ``distance`` |
| [`ls.accessible_area`](sdm.md#landschaft.sdm.accessible_area) | The area the species could have reached: where its background is drawn |
| [`ls.select_predictors`](sdm.md#landschaft.sdm.select_predictors) | Drop predictors that the others already explain (collinearity) |
| [`ls.vif`](sdm.md#landschaft.sdm.vif) | Variance inflation factor of each predictor: how well the others predict it |
| [`ls.background`](sdm.md#landschaft.sdm.background) | Background points: a sample of the conditions available to the species |
| [`ls.dataset`](sdm.md#landschaft.sdm.dataset) | Put presences, background points and predictor values together |
| [`ls.SDMData`](sdm.md#landschaft.sdm.SDMData) | Presences and background (or absences) with their predictor values, ready for a model |
| [`ls.spatial_blocks`](sdm.md#landschaft.sdm.spatial_blocks) | Folds made of spatial blocks: no fold is a neighbour of the records it tests |
| [`ls.autocorrelation_range`](sdm.md#landschaft.sdm.autocorrelation_range) | The range of spatial autocorrelation of each predictor (and their median), in metres |
| [`ls.spatial_clusters`](sdm.md#landschaft.sdm.spatial_clusters) | Folds from k-means clusters of the records' coordinates (or of their environment) |
| [`ls.buffer_folds`](sdm.md#landschaft.sdm.buffer_folds) | Leave-one-out folds with a buffer: each presence is a test, nothing within ``size`` trains |
| [`ls.quadrant_folds`](sdm.md#landschaft.sdm.quadrant_folds) | Four folds from the occurrences split at their median latitude and longitude (ENMeval "block") |
| [`ls.checkerboard_folds`](sdm.md#landschaft.sdm.checkerboard_folds) | Folds from a checkerboard of aggregated cells (ENMeval "checkerboard") |
| [`ls.random_folds`](sdm.md#landschaft.sdm.random_folds) | Random k-fold of the presences, background always training (ENMeval "randomkfold") |
| [`ls.jackknife_folds`](sdm.md#landschaft.sdm.jackknife_folds) | Leave-one-out: each presence its own fold, background always training (ENMeval "jackknife") |
| [`ls.MaxNet`](sdm.md#landschaft.sdm.MaxNet) | Maxent fitted as R's maxnet does it, natively: the same model, faster |
| [`ls.evaluate`](sdm.md#landschaft.sdm.evaluate) | Evaluate predictions at presences against background points (or absences) |
| [`ls.Evaluation`](sdm.md#landschaft.sdm.Evaluation) | How well predictions separate presences from background (or absences) |
| [`ls.boyce`](sdm.md#landschaft.sdm.boyce) | Continuous Boyce index: does predicted suitability rise with the frequency of presences? |
| [`ls.threshold`](sdm.md#landschaft.sdm.threshold) | A threshold that turns continuous predictions into presence / absence |
| [`ls.omission_rate`](sdm.md#landschaft.sdm.omission_rate) | Share of test presences predicted below a threshold set on the training presences |
| [`ls.mess`](sdm.md#landschaft.sdm.mess) | Multivariate environmental similarity surface: how far conditions lie outside the reference |
| [`ls.mop`](sdm.md#landschaft.sdm.mop) | Mobility-oriented parity: where new conditions are outside or far from the reference ones |
| [`ls.exdet`](sdm.md#landschaft.sdm.exdet) | Extrapolation detection: univariate (NT1) and combinatorial (NT2) novelty |
| [`ls.fit`](sdm.md#landschaft.sdm.fit) | Fit models to SDMData, cross-validate them on its folds, and fit them to all the data |
| [`ls.SDMResult`](sdm.md#landschaft.sdm.SDMResult) | Fitted models, their cross-validated scores and what is needed to project them |
| [`ls.predict`](sdm.md#landschaft.sdm.predict) | Project fitted models onto rasters of predictors (present or future conditions) |
| [`ls.response_curves`](sdm.md#landschaft.sdm.response_curves) | How each model's suitability changes along one predictor, the others held at their mean (or median) |
| [`ls.importance`](sdm.md#landschaft.sdm.importance) | Permutation importance: how much AUC drops when a predictor's values are shuffled |
| [`ls.area_of_habitat`](sdm.md#landschaft.sdm.area_of_habitat) | Area of habitat of species: their ranges refined by habitats and elevations |
| [`ls.HabitatAreas`](sdm.md#landschaft.sdm.HabitatAreas) | Result of `area_of_habitat` |
| [`ls.cell_area_km2`](sdm.md#landschaft.sdm.cell_area_km2) | The area of every cell of a raster's grid in km²: the product of the resolutions in a projected coordinate system (in its linear unit), the cell on the ellipsoid in a geographic one |
| [`ls.example_data`](sdm.md#landschaft.sdm.example_data) | Path of an example data set for species distribution modelling |

### Land change simulation

| Name | Summary |
|---|---|
| [`ls.changes`](simulate.md#landschaft.simulate.changes) | Land change along a series of categorical maps |
| [`ls.Changes`](simulate.md#landschaft.simulate.Changes) | Land change along a series of categorical maps (see `changes`) |
| [`ls.IntensityAnalysis`](simulate.md#landschaft.simulate.IntensityAnalysis) | Intensity analysis of land change (Aldwaik & Pontius 2012), as OpenLand computes it |
| [`ls.crosstab`](simulate.md#landschaft.simulate.crosstab) | Cells of each class of one map that hold each class of another (a transition matrix of counts) |
| [`ls.difference_components`](simulate.md#landschaft.simulate.difference_components) | Agreement and the components of difference of each category and overall (diffeR's ``diffTablej``) |
| [`ls.overall_difference`](simulate.md#landschaft.simulate.overall_difference) | The overall difference and its components (diffeR's ``overallDiff``, ``overallQtyD``...) |
| [`ls.difference_resolutions`](simulate.md#landschaft.simulate.difference_resolutions) | Quantity, exchange and shift differences at increasingly coarse resolutions (diffeR's ``differenceMR``) |
| [`ls.composite_matrix`](simulate.md#landschaft.simulate.composite_matrix) | Cross-tabulation at a coarser resolution by the composite method (diffeR's ``composite``) |
| [`ls.memberships`](simulate.md#landschaft.simulate.memberships) | The share of each category in each block of factor x factor cells (diffeR's ``memberships``) |
| [`ls.mad`](simulate.md#landschaft.simulate.mad) | Mean absolute deviation between two continuous maps and its components (diffeR's ``MAD``) |
| [`ls.sample_to_population`](simulate.md#landschaft.simulate.sample_to_population) | A sample's cross-tabulation scaled to the population of each row category (diffeR's ``sample2pop``) |
| [`ls.Markov`](simulate.md#landschaft.simulate.Markov) | A Markov chain of land change: the transition probabilities of a period and of each time step |
| [`ls.WeightsOfEvidence`](simulate.md#landschaft.simulate.WeightsOfEvidence) | Weights of evidence of land use transitions, calibrated on two maps (Dinamica EGO's method) |
| [`ls.weights_of_evidence`](simulate.md#landschaft.simulate.weights_of_evidence) | Calibrate weights of evidence in one call: ``WeightsOfEvidence(ranges, **kwargs).fit(t0, t1, drivers)`` |
| [`ls.LogisticRegression`](simulate.md#landschaft.simulate.LogisticRegression) | Logistic regression of the presence of each land use class on the drivers (lulcc's ``glmModels``) |
| [`ls.Estimator`](simulate.md#landschaft.simulate.Estimator) | Potential of each class from a scikit-learn classifier, one model per class |
| [`ls.MultiCriteria`](simulate.md#landschaft.simulate.MultiCriteria) | Potential from weighted factors, the weights from pairwise comparisons (AHP; MOLUSCE's MCE) |
| [`ls.RasterPotential`](simulate.md#landschaft.simulate.RasterPotential) | A potential given as maps: one layer per class (or transition), as computed elsewhere |
| [`ls.Model`](simulate.md#landschaft.simulate.Model) | A land change model: potential, demand, allocation and rules |
| [`ls.Simulation`](simulate.md#landschaft.simulate.Simulation) | The maps of a simulation and what happened at each step |
| [`ls.AreaDemand`](simulate.md#landschaft.simulate.AreaDemand) | Cells each class must cover at each time step (CLUE-S's demand) |
| [`ls.Rules`](simulate.md#landschaft.simulate.Rules) | Decision rules of the allocation (lulcc's ``rules``, ``elas``, ``mask``, ``hist``, ``neighb``) |
| [`ls.Neighbourhood`](simulate.md#landschaft.simulate.Neighbourhood) | Neighbourhood thresholds (lulcc's ``NeighbRasterStack`` and ``nb.rules``) |
| [`ls.CLUE`](simulate.md#landschaft.simulate.CLUE) | CLUE-S allocation (Verburg et al. 2002), as lulcc's ``CluesModel`` |
| [`ls.Ordered`](simulate.md#landschaft.simulate.Ordered) | The ordered allocation of Fuchs et al. (2013) with stochastic choice, as lulcc's ``OrderedModel`` |
| [`ls.distance_to`](simulate.md#landschaft.simulate.distance_to) | Distance from each cell to the nearest cell of some classes |
| [`ls.share_of`](simulate.md#landschaft.simulate.share_of) | Share of the cells around each cell that hold some classes |
| [`ls.gravity_of`](simulate.md#landschaft.simulate.gravity_of) | Gravitational interaction potential of the cells of some classes |
| [`ls.Regional`](simulate.md#landschaft.simulate.Regional) | A demand per region (municipalities, biomes, basins...) |
| [`ls.CLUMondo`](simulate.md#landschaft.simulate.CLUMondo) | CLUMondo's demand-driven allocation of land systems |
| [`ls.ServiceDemand`](simulate.md#landschaft.simulate.ServiceDemand) | The demand for goods and services at each time step (CLUMondo's ``demand.in``) |
| [`ls.Supply`](simulate.md#landschaft.simulate.Supply) | What one cell of each land system supplies of each good or service (CLUMondo's ``lusmatrix``) |
| [`ls.NeighbourhoodEffect`](simulate.md#landschaft.simulate.NeighbourhoodEffect) | CLUMondo's neighbourhood function (``neighmat.txt`` and ``alloc2.reg``) |
| [`ls.LandSystemStack`](simulate.md#landschaft.simulate.LandSystemStack) | Binary layers, possibly overlapping, as land systems (bit masks) |
| [`ls.Roles`](simulate.md#landschaft.simulate.Roles) | What each transition between land systems does to their layers |
| [`ls.Cap`](simulate.md#landschaft.simulate.Cap) | At most so many cells (or so much area) a year may make a role's transitions (or given pairs) |
| [`ls.Succession`](simulate.md#landschaft.simulate.Succession) | Autonomous change after some years, whatever the demand (CLUMondo's code 1000 + years) |
| [`ls.ExpanderPatcher`](simulate.md#landschaft.simulate.ExpanderPatcher) | Dinamica's Expander and Patcher allocation of transitions (``AllocateTransitions``) |
| [`ls.ContinuousPatcher`](simulate.md#landschaft.simulate.ContinuousPatcher) | New patches that take a stock (biomass, carbon, timber) up to a quota |
| [`ls.StockAllocation`](simulate.md#landschaft.simulate.StockAllocation) | What `allocate` took |
| [`ls.StockSimulation`](simulate.md#landschaft.simulate.StockSimulation) | What `run` did, step by step |
| [`ls.RoadBuilder`](simulate.md#landschaft.simulate.RoadBuilder) | New roads each step, from the network towards attractive places |
| [`ls.RoadDistance`](simulate.md#landschaft.simulate.RoadDistance) | Distance to a growing road network: a dynamic variable (see `dynamic`) |
| [`ls.friction_cost`](simulate.md#landschaft.simulate.friction_cost) | Cost of reaching every cell from the nearest source over a friction surface |
| [`ls.destinations`](simulate.md#landschaft.simulate.destinations) | Cells drawn by attraction in quadrants of a map |
| [`ls.pathways`](simulate.md#landschaft.simulate.pathways) | Paths from source cells to a network down a cost surface |
| [`ls.CARS`](simulate.md#landschaft.simulate.CARS) | PLUS's CA based on multi-type random patch seeds (Liang et al. 2021) |
| [`ls.MolusceWoE`](simulate.md#landschaft.simulate.MolusceWoE) | MOLUSCE's weights of evidence (``WoeManager``): one model per transition, persistence included |
| [`ls.MolusceCA`](simulate.md#landschaft.simulate.MolusceCA) | MOLUSCE's simulator: each iteration makes the observed transitions where the model is most confident |
| [`ls.PatchGrowing`](simulate.md#landschaft.simulate.PatchGrowing) | FUTURES' patch growing allocation (r.futures.simulation) |
| [`ls.DevelopmentDemand`](simulate.md#landschaft.simulate.DevelopmentDemand) | Cells each subregion develops at each step (r.futures.simulation's ``demand`` file) |
| [`ls.DevelopmentPotential`](simulate.md#landschaft.simulate.DevelopmentPotential) | The development potential's coefficients per subregion (r.futures.potential's table) |
| [`ls.development_pressure`](simulate.md#landschaft.simulate.development_pressure) | The development pressure of each cell (r.futures.devpressure) |
| [`ls.Flooding`](simulate.md#landschaft.simulate.Flooding) | FUTURES' flood submodel (Sanchez et al. 2023): floods, damage, response, abandonment, migration |
| [`ls.patch_library`](simulate.md#landschaft.simulate.patch_library) | The sizes of the patches developed between two maps (r.futures.calib's patch library) |
| [`ls.calibrate_patches`](simulate.md#landschaft.simulate.calibrate_patches) | Calibrate FUTURES' patch compactness and discount factor (r.futures.calib) |
| [`ls.three_map_comparison`](simulate.md#landschaft.simulate.three_map_comparison) | Compare the reference maps at the start and the end with the simulated end at several resolutions |
| [`ls.ThreeMapComparison`](simulate.md#landschaft.simulate.ThreeMapComparison) | Three maps compared at several resolutions (lulcc's ``ThreeMapComparison``) |
| [`ls.compare_maps`](simulate.md#landschaft.simulate.compare_maps) | MOLUSCE's comparison of two maps (``DependenceCoef``): kappa, agreement and dependence |
| [`ls.error_budget`](simulate.md#landschaft.simulate.error_budget) | MOLUSCE's error budget (Pontius 2002): agreement for levels of information, at coarser resolutions |
| [`ls.futures_validation`](simulate.md#landschaft.simulate.futures_validation) | FUTURES' validation (``r.futures.validation``): disagreement, kappa, kappa simulation, change metrics |
| [`ls.futures_validation_grid`](simulate.md#landschaft.simulate.futures_validation_grid) | FUTURES' validation per tile or per region (``r.futures.gridvalidation``) |
| [`ls.fuzzy_similarity`](simulate.md#landschaft.simulate.fuzzy_similarity) | Fuzzy similarity of two categorical maps (Dinamica EGO's ``CalcReciprocalSimilarityMap``) |
| [`ls.FuzzySimilarity`](simulate.md#landschaft.simulate.FuzzySimilarity) | The fuzzy similarity of two maps (`fuzzy_similarity`) |
| [`ls.differences`](simulate.md#landschaft.simulate.differences) | The map of differences: each cell that changed between the maps, in its new class; NaN elsewhere |
| [`ls.similarity_of_differences`](simulate.md#landschaft.simulate.similarity_of_differences) | Dinamica EGO's validation: fuzzy similarity of the observed and the simulated change, per window |
| [`ls.validate`](simulate.md#landschaft.simulate.validate) | Compare a simulated map with the observed one in every way the land change tools do |
| [`ls.Validation`](simulate.md#landschaft.simulate.Validation) | Every comparison of a simulated map with the observed one (`validate`) |
| [`ls.calibrate`](simulate.md#landschaft.simulate.calibrate) | Search the settings of a model that best reproduce an observed map |
| [`ls.Calibration`](simulate.md#landschaft.simulate.Calibration) | The result of `calibrate` |
| [`ls.run`](simulate.md#landschaft.simulate.run) | Fit a land change model on two maps, check it on the observed period and project it |
| [`ls.Projection`](simulate.md#landschaft.simulate.Projection) | The result of `run` |
| [`ls.agreement`](simulate.md#landschaft.simulate.agreement) | How a simulated map matches the observed one, cell by cell and in quantity |
| [`ls.presets`](simulate.md#landschaft.simulate.presets) | The classic land change models, assembled from ls.simulate's pieces |
| [`ls.terraflux`](simulate.md#landschaft.simulate.terraflux) | Run a Terraflux model with the ``terraflux`` command; the simulated maps ``(step, y, x)`` |
| [`ls.scenarios`](simulate.md#landschaft.simulate.scenarios) | Run a model under several scenarios, several times each, in parallel |
| [`ls.Ensemble`](simulate.md#landschaft.simulate.Ensemble) | The simulations of `scenarios`: every scenario, replicate and year |
| [`ls.example_data`](simulate.md#landschaft.simulate.example_data) | Path of an example data set for land change modelling |

### Ecosystem services

| Name | Summary |
|---|---|
| [`ls.carbon`](services.md#landschaft.services.carbon) | Carbon storage of a landscape and its change between two maps (InVEST's carbon model) |
| [`ls.Carbon`](services.md#landschaft.services.Carbon) | Carbon stored in a landscape, and its change (`carbon`) |
| [`ls.valuation_constant`](services.md#landschaft.services.valuation_constant) | The factor turning a change in carbon into its net present value (InVEST's) |
| [`ls.water_yield`](services.md#landschaft.services.water_yield) | Annual water yield, consumption and hydropower value (InVEST's annual water yield model) |
| [`ls.WaterYield`](services.md#landschaft.services.WaterYield) | Annual water yield of a landscape (`water_yield`) |
| [`ls.sdr`](services.md#landschaft.services.sdr) | Sediment export, deposition and retention (InVEST's sediment delivery ratio model) |
| [`ls.SDR`](services.md#landschaft.services.SDR) | Sediment export and retention of a landscape (`sdr`) |
| [`ls.habitat_quality`](services.md#landschaft.services.habitat_quality) | Habitat degradation, quality and rarity (InVEST's habitat quality model) |
| [`ls.HabitatQuality`](services.md#landschaft.services.HabitatQuality) | Habitat degradation, quality and rarity (`habitat_quality`) |

### Uncertainty

| Name | Summary |
|---|---|
| [`ls.uncertainty.sum_sd`](uncertainty.md#landschaft.uncertainty.sum_sd) | Standard deviation of a total of uncertain values, sum of w_i v_i |
| [`ls.uncertainty.product_sd`](uncertainty.md#landschaft.uncertainty.product_sd) | Standard deviation of the product of two uncertain values, x v |
| [`ls.uncertainty.delta_method`](uncertainty.md#landschaft.uncertainty.delta_method) | Uncertainty of a result from the uncertainty of its parameters (delta method) |
| [`ls.uncertainty.DeltaResult`](uncertainty.md#landschaft.uncertainty.DeltaResult) | Result of `delta_method` |
| [`ls.uncertainty.monte_carlo`](uncertainty.md#landschaft.uncertainty.monte_carlo) | Uncertainty of a result by drawing its parameters (Monte Carlo) |
| [`ls.uncertainty.MonteCarloResult`](uncertainty.md#landschaft.uncertainty.MonteCarloResult) | Result of `monte_carlo` |
| [`ls.uncertainty.interval`](uncertainty.md#landschaft.uncertainty.interval) | The normal confidence interval of a value: value -/+ z sd, z the normal quantile of the level |
