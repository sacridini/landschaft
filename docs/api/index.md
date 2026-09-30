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

-   :material-routes:{ .lg .middle } **[Grid distances](cost.md)**

    ---

    Transitions, cost surfaces, least-cost and random-walk distances and paths.

-   :material-map-marker-distance:{ .lg .middle } **[Distance & restoration](distance.md)**

    ---

    Distances, resistance, least-cost paths, corridors, zones of influence.

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

</div>

## All functions and classes

### Data & configuration

| Name | Summary |
|---|---|
| [`ls.load_raster`](data.md#landschaft.load_raster) | Read one raster, or collect many, checking them for MSPA |
| [`ls.write_raster`](data.md#landschaft.write_raster) | Write a 2-D raster to a GeoTIFF |
| [`ls.reclassify`](data.md#landschaft.reclassify) | Build a habitat map (0 = missing, 1 = background, 2 = habitat) from any raster |
| [`ls.example_data`](data.md#landschaft.example_data) | Path of an example raster shipped with the package |
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
| [`ls.write_table`](landscape-metrics.md#landschaft.write_table) | Write a metric table, the format given by the file extension |
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
| [`ls.surface.fit_plane`](surface.md#landschaft.surface.fit_plane) | The least-squares polynomial surface of a given order |
| [`ls.surface.best_fit_plane`](surface.md#landschaft.surface.best_fit_plane) | The polynomial surface of order 0 to 3 that fits a surface best |
| [`ls.surface.remove_plane`](surface.md#landschaft.surface.remove_plane) | A surface minus its best-fitting polynomial surface (geodiv's ``remove_plane``) |
| [`ls.surface.example_data`](surface.md#landschaft.surface.example_data) | Path of an example surface |

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
