# Changelog

All notable changes to landschaft are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0 the API may still change
between minor versions.

## [Unreleased]

## [0.44.0] - 2026-09-26

### Added
- Circuit connectivity, part 4: the rest of Omniscape's options. Conditional connectivity
  in `ls.omniscape` (`condition1`/`condition2`, their futures, `compare_to_future`,
  `"within"` with the block median and `"equal"` with StatsBase's mode, lower and upper
  limits), `reclass_table` (`ls.circuit.reclassify`, `read_reclass_table`),
  `precision="single"`, and `ls.circuit.omniscape_ini`, which runs an Omniscape INI file
  as `run_omniscape(path)` does (Omniscape's parser `ls.circuit.parse_ini`, defaults,
  alignment checks, outputs in `project_name`). Matched with Omniscape's own test INI
  files (conditional, reclass, 32 bit) run with CHOLMOD to 5e-13.

## [0.43.0] - 2026-09-26

### Changed
- Circuit connectivity, part 3: performance. Windows are factorised by a multifrontal
  Cholesky over the grid's nested dissection (dense fronts on the separator lines, Eigen's
  dense kernels) instead of Eigen's sparse LDLT: 2x faster at radius 100, 2.7x at 200.
  Window shapes share one symbolic structure between threads; the forward substitution
  runs inside the factorisation; flow potentials of up to 8 targets are solved together;
  buffers keep their capacity. On x86-64 the circuit code is also built for AVX2 and FMA
  (`landschaft._circuit_avx2`, loaded only on CPUs that have them;
  `LANDSCHAFT_CIRCUIT_AVX2=0` to keep the baseline), 1.2-1.5x faster dense kernels.
  Omniscape benchmarks with 20 threads: 800x800 radius 10 in 0.64 s (Omniscape 29.4 s),
  700x700 radius 25 in 1.36 s (46.7 s), 1000x1000 radius 100 in 12.4 s (275 s).

## [0.42.0] - 2026-09-26

### Added
- Circuit connectivity, part 2: Omniscape. `ls.omniscape` is Omniscape 0.7's
  `run_omniscape`: targets every `block_size` cells with a positive block of sources,
  circular moving windows of `radius` + `buffer`, the block's strength shared among the
  sources and grounded at the target, cumulative current, flow potential and normalised
  current, the block-artifact correction, `source_threshold`, `r_cutoff`
  (`ls.circuit.source_from_resistance`), conductance input, four neighbours and NoData
  masking, with Omniscape's edge quirks. Windows of one shape share their ordering and
  symbolic analysis, the flow potential its whole factorisation; windows are solved in
  parallel and summed in a fixed order (the result does not depend on the threads).
  Matched with Omniscape run with CHOLMOD to 1.7e-11 on its own test configurations and
  synthetic maps; about 20x faster than Omniscape with the same 20 threads.

## [0.41.0] - 2026-09-26

### Added
- Circuit connectivity, part 1 (port of Omniscape 0.7 / Circuitscape 6.1): the circuit
  core. `ls.circuit.advanced` is Circuitscape's advanced mode on in-memory rasters
  (`compute_omniscape_current`, the solve behind every Omniscape window): cells with a
  conductance > 0 joined to 8 or 4 neighbours, sources, finite and direct grounds, every
  component with a source and a ground solved exactly (Eigen's sparse LDLT in a
  nested-dissection order of the cells, components in parallel), node currents with
  Circuitscape's 1e-8 branch threshold. Matched with Circuitscape's CHOLMOD solve to
  1.2e-13 of the largest current (its default cg+amg is 2e-7 away).
- Eigen 3.4.0 (headers only) is fetched by CMake at build time
  (`FETCHCONTENT_SOURCE_DIR_EIGEN` for offline builds).

## [0.40.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 11: spatial absorbing Markov chains.
  `ls.conn.settlement_probabilities` and `HabitatGraph.from_samc` are Makurhini's
  `MK_dPC_SAMC` (samc's random walk over a resistance raster, absorbed with an absorption
  raster or Fletcher et al.'s rate for a dispersal distance; settlement summed or averaged
  over the patches' boundary vertices, `ls.conn.boundary_vertices`), giving a probability
  graph for PC and its node importances. Matched with samc to 1e-14 and with Makurhini's
  importances after its rounding.

## [0.39.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 10: commute-time distances.
  `distance="commute-time"` in `ls.conn.distance_table`, `HabitatGraph.from_raster` and
  `from_vector` is gdistance's `commuteDistance` as Makurhini's `distancefile` uses it
  (effective resistance times the total conductance over the least-cost conductances),
  one sparse factorisation per connected part. Matched with Makurhini to 1e-11, 13×
  faster.

## [0.38.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 9: focal nodes. `ls.conn.focal_nodes` is Makurhini's
  `MK_Focal_nodes`: for every node the graph of the polygons within its search buffer
  (points on surface, the bounding box as landscape area), its IIC or PC, the node's dIIC
  or dPC with its fractions and the composite connectivity index IComp. Matched with
  Makurhini to 1e-12 (importances after its rounding).

## [0.37.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 8: centrality. `HabitatGraph.node_importance` gives
  Conefor's betweenness centralities `BC`, `BCIIC` (`dBC_IIC`, `varBC_IIC`) and `BCPC`
  (`dBC_PC`, `varBC_PC`) when asked for, from the same searches as the other indices;
  `HabitatGraph.centrality(distance, probability=None, binary=True)` is Makurhini's
  `MK_RMCentrality` (degree or strength, eigenvector, closeness, igraph's betweenness
  with its 1e-10 ties, components; walktrap communities with python-igraph).
  Matched with Makurhini to 1e-12 and with Conefor 2.7.1's command line to its 6 digits.

## [0.36.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 7: raster ProtConn and grids.
  `ls.conn.protconn_raster` (Makurhini's `MK_ProtConn_raster`), `ls.conn.make_grid`
  (hexagons of sf's `st_make_grid` and raster squares, the same coordinates),
  `ls.conn.connectivity_grid` (`MK_Connect_grid`: ProtConn, or PC and ECA, of every cell;
  `pij_min=np.inf` reproduces Makurhini's default, which cuts every link) and
  `ls.conn.eca_change_grid` (`MK_dECA_grid`). `protconn` and `protconn_nodes` gain
  `pij_min` and `preselect`; one protected area touching the region is a single node, as
  in Makurhini. Matched with Makurhini after its rounding.

## [0.35.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 6: ProtConn. `ls.conn.protconn(protected_areas,
  region, distance=...)` (Makurhini's `MK_ProtConn`: ECA, Prot, ProtConn, ProtUnconn,
  RelConn, the Prot/Trans/Unprot/Within/Contig fractions and their land shares,
  ProtUnconn[Design] and ProtConn[Bound], with transboundary zones around the nodes or
  the region, edge or least-cost distances, and `delta` for the importance of each
  protected area), `protconn_regions` (`MK_ProtConnMult`), `protconn_nodes` (the nodes,
  built with shapely/GEOS) and `protconn_indicators` (`get_protconn_grid`'s formulas on
  any nodes and distances). The formulas give Makurhini's bits on its own nodes, and the
  whole pipeline reproduces its rounded table on synthetic protected areas.

## [0.34.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 5: `ls.conn.eca_change(graphs, metric, distance=,
  landscape_area=)`, Makurhini's `MK_dECA`: habitat area, ECA (capped at LA), normalised
  ECA, `dA`, `dECA`, `rECA` and Makurhini's change classes for one graph per time. Graphs
  from `from_raster` / `from_vector` keep their habitat area (`habitat_area`, summed as R
  does); a graph may now hold a single node. Matched with `MK_dECA` on polygon and raster
  time series (`dA` to the bit, the rest to 1e-13).

## [0.33.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 4: links and nodes to add.
  `HabitatGraph.link_importance(metric)` (Makurhini's `MK_dPCIIC_links` removal: the
  loss of IIC or PC without each link) and `link_change(changes, metric)` (its `change`:
  pairs with new distances); `HabitatGraph(..., restoration=)` rates the nodes to add
  (Makurhini's `dPCres` / `dIICres`, or with `convention="conefor"` Conefor's "adding
  nodes" mode with a `d...add` gain for every index). Removing a link repairs the
  subtree below it in each shortest-path tree; shorter pairs and added nodes use the
  lengths between all nodes. Matched with Makurhini (the same rounded values, 1e-12
  for the restoration gains) and with Conefor's "adding nodes" (its precision). All
  4.3 M PC links of 5,000 patches rated in 22 s.

## [0.32.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 3: least-cost distances.
  `distance_table(..., distance="least-cost", resistance=...)` and the same option in
  `HabitatGraph.from_raster` / `from_vector`: gdistance's `costDistance` as Makurhini
  calls it (conductance 1 / the resistance of the cell of larger index, or 1 / the mean
  with `cost_function="mean"`, over the distance between cell centres; 8 or 4
  directions; NA cells cannot be crossed), from the cells of the node centroids, with
  Makurhini's fallback to the centroid distance for pairs without any path
  (`unreachable="drop"` to leave them out). A parallel Dijkstra per node over the
  raster with the same step costs (raster centre coordinates, R's x87 `sqrt`,
  `r_mean`) and pair direction (summed from the later node, as `as.dist` keeps the
  lower triangle) gives gdistance's distances to the last bit. Makurhini's example
  (398 patches, 222 k cells) in 0.5 s instead of 10.2 s.

## [0.31.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 2: nodes from maps. `HabitatGraph.from_raster`
  (a raster of patch ids, or the patches of `habitat` classes) and
  `HabitatGraph.from_vector` (polygons or points), with centroid or edge distances up
  to `max_distance` in `distance_unit`; `to_raster` / `to_geodataframe` put node values
  back on the map. `ls.conn.node_table` and `ls.conn.distance_table` are Makurhini's
  `nodesfile` and `distancefile`, with the same bits: raster areas and centroids
  (`raster::xyFromCell` cell centres averaged in 80-bit precision as R's `colMeans`),
  polygon areas (GDAL's shoelace sums, as `sf::st_area`), centroids of the largest part
  with Makurhini's point-on-surface fallback, GEOS edge distances; raster edge
  distances (gap between the closest cells, searched row by row over the boundary
  cells) to the last bit. `MK_dPCIIC` run end to end on Makurhini's raster and on
  synthetic polygons is reproduced exactly. 8,395 patches of a 4000×4000 raster: all
  35 M centroid distances in 0.9 s, edge distances up to 5 km in 0.24 s.

## [0.30.0] - 2026-09-26

### Added
- Habitat connectivity graphs, part 1 (Conefor Sensinode / Makurhini): `HabitatGraph`
  (nodes with an attribute, and distances, dispersal probabilities or links between
  them, as a matrix or a pair table) with Conefor's indices NC, NL, H, CCP, LCP, IIC, F,
  AWF and PC (`overall`, with EC and Makurhini's intra/direct/step split of IIC and PC)
  and the importance of every node (`node_importance`: dA, dNC, dNL, dH, dCCP, dLCP, dF,
  dAWF, dIIC and dPC with their intra, flux and connector fractions, varPC), for one or
  several distances; `components`; Conefor's node and connection files
  (`from_conefor`, `to_conefor`). `convention="makurhini"` (default) matches
  Makurhini's `MK_dPCIIC` (its unrounded values to 1e-12, its rounded output exactly),
  `convention="conefor"` Conefor Sensinode 2.2 (to its single precision). The node
  importances do not recompute the graph once per node: only the shortest paths
  through the removed node are repaired, in parallel over the sources: Makurhini's
  example data (404 patches) in 5 ms (IIC) / 17 ms (PC) instead of 1.4 s / 10.6 s.

## [0.29.0] - 2026-09-26

### Changed
- Landscape metrics part 18: R's 80-bit sums run on the x87 unit under MSVC x64 (MASM
  routines for `sum()`, `mean()`, the squared deviations of `sd()` and blocked
  accumulation for other sums), the same bits as the emulation: summary statistics over
  27 M patches 5–8 times faster, 21×21 moving windows 97 → 73 µs each on one thread.
- Portability: a MinGW gcc build now passes every test, bitwise ones included (Windows
  runs the x87 unit at 53-bit precision, so the 80-bit type is emulated there, and the
  gyrate sums use `double` on every compiler instead of `long double`); the MASM file and
  the MSVC 128-bit intrinsics are limited to x86-64. DOCUMENTATION lists how each platform
  (Windows, Linux, macOS, ARM64) gets R's bits.

## [0.28.0] - 2026-09-26

### Changed
- Landscape metrics part 17, performance: a `Landscape` is prepared in parallel (value
  range, per-thread class histogram, no zero-filling of the class raster; the int32 range
  check moved to C++): 100 M cells in 0.07 s instead of 0.43 s. Moving windows: the ENN
  of small maps takes the minimum per pair of patches with SSE2, summary statistics are
  returned without copies, metric names are compared as `string_view`s and the remaining
  `omp if` loops use the `parallel_for` helpers (no OpenMP runtime on small maps): 130 →
  97 µs per 21×21 window on one thread, 1000×1000 with all 66 metrics in 8.7 s instead
  of 12.0 s, with the same values. Circles are found on the convex hull of their corner
  points, in memory and out of core alike.

## [0.27.0] - 2026-09-26

### Added
- Landscape metrics part 16, spatial signatures: `SpatialSignatureAnalysis.nmf(k)`
  (non-negative factors of the min-max scaled metrics, NNDSVDa start and HALS sweeps: the
  loss of scikit-learn's `NMF(init="nndsvda", solver="cd")`), hierarchical clusters
  (`cluster(k, method="ward" | "average" | "complete" | "single" | ...)`, scipy's linkages)
  and the clustergram (`clustergram(k_range)` with every row's cluster and cluster mean per
  k, `plot_clustergram`), as pylandstats' `decompose` / `get_cgram` and the clustergram
  package compute them; `NMFResult` and `ClustergramResult`.

## [0.26.0] - 2026-09-26

### Added
- Landscape metrics part 15, large tables: `Landscape.write_metrics(path, ...)` writes the
  long table of `metrics(...)` one metric at a time and the patch metrics in chunks of
  `chunk_rows`, and `Landscape.write_patch_metrics(path, metrics)` the wide patch table, to
  Parquet (one row group per chunk), CSV or CSV.gz without building the table in memory
  (9 patch metrics of 9 M patches: 8.0 s and 1.0 GB instead of 10.3 s and 6.1 GB).
  `batch("landscape_metrics")` writes its tables this way.

## [0.25.0] - 2026-09-26

### Changed
- Landscape metrics part 14, `convention="fragstats"` completed: `shape` is FRAGSTATS'
  raster SHAPE (perimeter over the smallest perimeter of as many cells), `_sd`/`_cv` use
  the standard deviation over n, cells that are not square get pylandstats' .25 p / sqrt(a)
  for `shape` and `lsi`, and the `circle` encloses the whole patch (landscapemetrics' only
  encloses the corners of the first and last rows and columns): the convex hull of the
  first and last cell of each row, exact on the grid, in parallel and out of core with
  the same values. Checked against pylandstats (shape, lsi, sd, cv, dcore) and a brute-force
  smallest circle.

## [0.24.0] - 2026-09-26

### Changed
- Landscape metrics part 13: `pafrac` is now the same bits as landscapemetrics. Its
  regression runs what R's `lm()` runs (LINPACK's `dqrdc2`/`dqrsl` over R's reference
  BLAS, LAPACK 3.12's `dnrm2`; `src/cpp/lsm/rlm.cpp`), checked on 400 stored `lm()`
  slopes (166 collinear, NA in R) and random maps; landscape-level `pafrac` regresses
  the area in hectares as landscapemetrics does. `window_metrics` transposes each
  window as `window_lsm` does (with the cell sides swapped, keeping the geometry), so
  the patches come in R's order and window `pafrac` values match too.

## [0.23.0] - 2026-09-26

### Changed
- Landscape metrics part 12, performance: patch labelling sums the pieces, sorts
  (parallel radix sort) and permutes the patch tables in parallel (27 M patches on a
  10,000×10,000 map: 5.0 s → 3.3 s); small loops no longer enter the OpenMP runtime and
  single-strip maps skip the seam joining (moving windows ~7% faster); GDAL's
  all-touched polygon rasterization is ported to C++ for `sample_metrics` (and
  `zonal_metrics`, `scale_sample`): the same cells as GDAL/terra (checked on 3,000 random
  polygons and the stored references), 10,000 circular plots in 2.2 s instead of 5.4 s.

## [0.22.0] - 2026-09-26

### Added
- Landscape metrics part 11, pylandstats-style analyses: `ls.SpatioTemporalAnalysis`
  (metrics along a time series of rasters, for whole maps, polygons, buffers around
  points or a grid of zones; wide landscape/class tables with 0 for absent classes in
  area/count metrics, and `plot_metric`), `ls.SpatialSignatureAnalysis` (signature matrix
  of landscapes or zones, principal components on standardised metrics and k-means
  clusters in numpy, the same as scikit-learn's; `plot_pca`, `plot_cluster_metrics`) and
  `ls.grid_zones()`.

## [0.21.0] - 2026-09-26

### Added
- Landscape metrics part 10, plots (matplotlib, optional: `pip install
  landschaft[plot]`): `ls.show_patches()` / `Landscape.show_patches()` (patches in their
  own colours, one map or one per class, with patch ids), `show_cores()` (core and edge
  cells, with core areas), `show_lsm()` (a patch metric on the map, with values) and
  `show_correlation()` (correlation heatmaps per level), like landscapemetrics'
  `show_*`; they return matplotlib Figures.
- Optional dependency groups `plot` (matplotlib) and `vector` (geopandas, pyarrow).

## [0.20.0] - 2026-09-26

### Added
- Landscape metrics part 9, export and integration: `ls.write_table()` (CSV, `.csv.gz`,
  Parquet, and GeoPackage/GeoJSON/Shapefile for GeoDataFrames),
  `ls.zonal_metrics()` (class and landscape metrics of polygons as a GeoDataFrame with
  the zones' geometry and attributes, one column per metric and class),
  `ls.scale_sample()` (landscapemetrics' `scale_sample`: several buffer sizes around
  points; the same bits, stored references), `Landscape.to_raster()` for class-level
  metrics, several metrics and writing GeoTIFFs (`path=`), and `batch()` with
  `"landscape_metrics"` and `"window_metrics"` (tables as CSV or `table_format="parquet"`;
  `block_rows` reads the files by blocks).

## [0.19.0] - 2026-09-26

### Added
- Landscape metrics part 8, FRAGSTATS metrics that landscapemetrics lacks (extras):
  `prox` (proximity index, `search_radius`) and `simi` (similarity index, with a
  `similarity` table), `connect` (connectance, `connect_threshold`), `econ` (edge
  contrast index), `cwed` (contrast-weighted edge density) and `teci` (total edge
  contrast index) with a `contrast` table and `boundary_contrast`, plus the
  `_mn/_sd/_cv/_am/_md/_ra` of `prox`, `simi` and `econ`. Distances between patches are
  the shortest cell-centre distances (as for `enn`), found by a range search over the
  boundary cells, one focal patch at a time. Checked against direct implementations of
  FRAGSTATS' definitions.
- `convention="fragstats"`: the physical perimeter with non-square cells (left/right
  sides as long as the cell height) and `dcore_*` over the sizes of the disjunct core
  areas; both checked against pylandstats.

## [0.18.0] - 2026-09-26

### Added
- Landscape metrics part 7, landscapemetrics' utilities, the same bits as in R:
  `Landscape.centroids()` (`get_centroids`, also `cell_center` with R's order of
  ties), `.nearest_neighbours()` (`get_nearestneighbour` per class, with the ids of
  the neighbours and landscapemetrics' tie rule, found by a range search instead of
  its sweep: 0.05 s instead of 1.8 s on 1000x1000), `.circumscribing_circles()`
  (patch and class level, with centres), `.adjacencies()` (`get_adjacencies`: 4, 8 or
  any neighbourhood matrix; full, like, unlike or triangle), `.boundaries()`
  (`get_boundaries` of the patches of each class), `.patches()` (`get_patches`),
  `.extract_metrics()` / `ls.extract_metrics()` (`extract_lsm`, several rasters too),
  `ls.raster_boundaries()` (`get_boundaries` on raster values, with its one-sided
  rule), `ls.check_landscape()` and `ls.calculate_correlation()` (R's `cor` for
  Pearson, Spearman and Kendall, and `xtabs`' handling of NA). Stored references on 8
  maps (`make_reference_lsm.py utils`) and live R runs.
- `Landscape.extent` (terra's extent of the raster).

## [0.17.0] - 2026-09-26

### Added
- Landscape metrics part 6, FRAGSTATS extras: the area-weighted mean (`_am`), median
  (`_md`) and range (`_ra`) of every patch metric at the class and landscape level
  (`gyrate_am` is FRAGSTATS' correlation length), and the `perim_*` statistics of
  pylandstats; 78 more metrics, selected with `extras=True` in `list_metrics`,
  `metrics`, `landscape_metrics`, `sample_metrics` and `window_metrics` (or by name).
  Checked against numpy and against pylandstats' `_mn`/`_am`/`_md`/`_ra`.
- Metric names as FRAGSTATS abbreviations in capitals (`"SHDI"`, `"AREA_MN"`) and as
  the full names of pylandstats / FRAGSTATS (`"shannon_diversity_index"`,
  `"total_area"`, `"shape_index_am"`, ...); wide tables keep the names as asked.
- `n_threads` for `Landscape`, `landscape_metrics`, `sample_metrics` and
  `window_metrics` (default: all cores).

### Changed
- Landscape metrics are now the same bits as landscapemetrics' for everything but
  `circle`, `gyrate`, `pafrac` and rare `enn` ties (still within 1e-10): sums, means
  and standard deviations reproduce R's 80-bit `long double` accumulation (emulated
  in software, MSVC has no such type; validated against gcc's native type on 40
  million random operations), and `log`/`sqrt` are computed on the x87 unit as
  R's mingw-w64 math library does (a MASM routine under MSVC). The reference tests
  now require identical bits.
- Summary statistics of all requested metrics are computed side by side, and
  class-level statistics per class in parallel; per-patch formulas run in parallel.

### Fixed
- Patch perimeters (and `para`, `shape`, `frac`, ...) of rasters whose cell width
  and height differ by rounding only (e.g. geographic rasters) use landscapemetrics'
  `all.equal` rule.

## [0.16.0] - 2026-09-26

### Added
- Landscape metrics part 5, rasters larger than memory: `Landscape(path,
  block_rows=...)` (and `landscape_metrics(..., block_rows=...)`) reads the file by
  blocks of rows. Each block is labelled by the in-memory engine; patch pieces cut by
  the seams are joined with a union-find and the seam statistics corrected, so all 133
  metrics are identical to the in-memory ones (and to landscapemetrics). Core areas,
  gyrate, circle and enn come from a detail pass run only when asked for. A 400 M-cell
  map needs 0.8 GB instead of 3.7 GB for the class and landscape metrics.

### Changed
- `label_patches` works on a pointer (no copy of the raster), and the core-area
  distance is a separate function shared by both engines.

## [0.15.0] - 2026-09-26

### Added
- Landscape metrics part 4: `sample_metrics()` (landscapemetrics' `sample_lsm`: square
  or circular plots around points, or polygons, cut exactly as `terra::crop(mask =
  TRUE)` does, `plot_id`, `all_classes`, `percentage_inside` with geodesic plot areas
  for data with a CRS) and `window_metrics()` (`window_lsm`: landscape-level metrics in
  any odd-sized window shape). Plots, windows and lists of rasters
  (`landscape_metrics(list)`) are computed in parallel by the C++ core, which also
  builds the long table directly. Identical to landscapemetrics on stored and live
  cases; thousands of times faster (a `sample_lsm` call on 6 plots of `augusta_nlcd`:
  3.7 s in R, 0.02 s here).
- `Landscape.metrics()` returns one NA row per metric for an all-NA map, as
  `calculate_lsm` does.

### Changed
- Small maps (windows, plots) compute ENN from all pairs of edge cells and sort their
  patches with a comparison sort: 2-3× faster there, same values.

## [0.14.0] - 2026-09-26

### Added
- Landscape metrics part 3: `enn` (Euclidean nearest-neighbour distance) and
  `circle` (related circumscribing circle) at patch level, and their
  `_mn`/`_sd`/`_cv` at class and landscape level (133 metrics). ENN searches the
  patch edge cells of each class in per-class Morton-order trees whose nodes know
  when they hold a single patch, leaf by leaf and in parallel, after a 5×5 raster
  look-up; the distances are identical to landscapemetrics (0.014 s instead of
  2.4 s on `augusta_nlcd`). The circle uses Welzl's algorithm on the same points as
  landscapemetrics, with a fixed shuffle per patch (repeatable), and also works
  with non-square cells.
- Reference maps with few distant patches and a non-integer cell size, and live
  comparisons on such maps.

### Fixed
- The landscapemetrics references were read back with pandas' fast float parser
  before being stored, which changed some values in the last bit; they are now
  read with round-trip precision and were regenerated. `lsm_reference.R` decides
  whether cells are square as landscapemetrics does (`all.equal`), so `podlasie`
  now has circle references.

## [0.13.0] - 2026-09-26

### Added
- Landscape metrics (FRAGSTATS) matching the R package landscapemetrics, part 1:
  `Landscape` (patch/class/landscape tables, long format, patch labels,
  `to_raster`), `landscape_metrics()` (`calculate_lsm`) and `list_metrics()`
  (`list_lsm`, same selection rules). 91 metrics: area and edge, shape (except
  circle), aggregation (except ENN), diversity and complexity, with the options
  `directions`, `count_boundary`, `cell_center`, `classes_max`, `neighbourhood`,
  `ordered` and `base`. A C++ core labels the patches of all classes in one pass
  (parallel strips joined at their seams) and accumulates every patch statistic
  and the class adjacencies once. Tested against landscapemetrics 2.2.2 on 12
  stored maps × 4 option sets and live random maps (relative agreement 1e-10).
- Landscape metrics part 2, core areas: `core`, `ncore`, `cai` (patch), `tca`,
  `cpland`, `ndca`, `dcad` and `_mn`/`_sd`/`_cv` of `core`, `cai` and `dcore`, with
  `edge_depth` and `consider_boundary`. The core cells come from a city-block distance
  computed in two parallel separable passes (equivalent to landscapemetrics' repeated
  `get_boundaries`); core areas are labelled with the patch engine. 28 more metrics
  (119), tested on a new stored option set (`edge_depth = 3`, `directions = 4`) and
  live maps with `edge_depth` 0 to 4.
- `IDEAS.md`: ideas for later (extra statistics and FRAGSTATS metrics, aliases,
  exports, pylandstats-style analyses).

## [0.12.0] - 2026-09-26

### Added
- `mspa_tiled()`: GTB's MSPA Tiling for maps too large for memory. The map is cut
  into overlapping tiles (GTB: 6500 pixels with a 2000-pixel overlap on MS-Windows),
  MSPA runs on every tile, and the tiles are put back together as GTB does, including
  its shortcuts for tiles without background or without foreground. GeoTIFF paths are
  processed out of core, window by window, with tiles in parallel (`n_workers`);
  arrays and DataArrays in memory. `tile_layout()` gives the tiles of an axis. Outputs
  are identical to GTB's MSPA Tiling, i.e. its IDL code with its gdal tools and `mspa`,
  run in GDL. Like GTB's, the result is an approximation of MSPA near the seams when
  objects are larger than the overlap. Unlike GTB, a map that fits in one tile gets
  MSPA of the whole map (GTB drops the first `overlap / 2` columns).

## [0.11.0] - 2026-09-26

### Added
- `fos()`: GTB Fragmentation at a Fixed Observation Scale, complete. All of GTB's
  methods (FAD, FED, FAC) in 5 or 6 classes, on binary maps (with special
  backgrounds 3/4) or grayscale cover maps (0-100 %, foreground at or above a
  threshold, 103/104 special backgrounds). Returns the map with GTB's codes, the class
  proportions, the foreground histogram, the patch statistics (APS, ECA, COH, COH_ru),
  the average connectivity and AVCON, and a `fos.txt`-style `summary()`. Maps are
  identical to GTB's batch FOS (its IDL code run in GDL with GTB's spatcon and
  grayspatcon), and so are the statistics.
- GraySpatCon metrics 1 (mean) and 52 (clustering, FED) in C++ with sliding-window
  frequency/adjacency counts and OpenMP, identical to `grayspatcon64.exe`.
- `fad5_colormap()` (GTB's `fadcolors5` table), used automatically for 5-class FOS
  maps. `"fos"` is available in `batch()`.

## [0.10.0] - 2026-09-26

### Added
- `batch(tool, inputs, output_dir, **params)`: GTB's Batch Process for every tool.
  It runs any landschaft tool, or any function of one raster, on many rasters in
  parallel, with the memory budget of `mspa_batch`, and writes everything the tool
  returns: maps as georeferenced GeoTIFFs (with GTB's colour tables), tables as CSV,
  `summary()` reports as TXT, GTB's CSV reports, and a JSON record (parameters,
  outputs, scalar results, run time) that lets reruns skip finished inputs.
  `BATCH_TOOLS` lists the tools known by name.

### Changed
- The C++ core now releases the GIL in every tool (parcellation, contortion,
  restoration status, cost distance, LM statistics, change analyses), so batch
  workers run in parallel.

## [0.9.0] - 2026-09-26

### Added
- `spa()`: GTB Simplified Pattern Analysis SPA2 (contiguous / small linear
  features), SPA3 (core, core-opening, margin), SPA5 (core, edge, perforation,
  margin) and SPA6 (+ islets), with GTB's areas, proportions, object counts and
  porosity. Maps and statistics are identical to GTB's code (run in GDL, with the
  documented MORPH_OPEN). C++ core with separable 3×3 morphology and OpenMP: about
  0.4–1 s for 25 Mpx.

## [0.8.0] - 2026-09-26

### Added
- `isochrone_map()` / `isochrone_map_ab()`: GTB Restoration Planner → Isochrone Map
  A / AB, identical to GTB. They give the accumulated cost from object A (or the
  cheapest A–B route through every pixel) across a resistance map, scaled to GTB's
  0–100 display codes, with the least-cost path for map AB. They port GTB's `ggeo`
  program: miallib's symmetric geodesic time (`sqtgsym`, in C++) and its
  least-cost path (the pixels of minimal A–B cost, thinned with miallib's
  `binOIthin_FIFO`), plus GTB's IDL post-processing. This includes a GTB quirk: a
  resistance map without foreground makes its top-right pixel cost 1.
- `distance_to_resistance()`: GTB Setup Tools → Distance → Resistance
  (`round(distance + 2)` capped at 100). It relies on IDL's MORPH_DISTANCE and
  shares the caveat of `influence_zones()`.
- Parity tests for the isochrone maps against GTB's IDL code and `ggeo`
  (`LANDSCHAFT_GGEO_EXE`, or a GTB desktop installation).

## [0.7.0] - 2026-09-26

### Added
- `influence_zones()` and `proximity()` (**experimental**): GTB Distance →
  Influence Zones / Proximity. They return the GTB viewport codes, object ids,
  zone labels, the background buffer, the proximity distance map, the CAG on the
  watershed lines and GTB's proximity CSV. C++ ports of the IDL routines they use:
  MORPH_DISTANCE (quasi-Euclidean chamfer), HIST_EQUAL (identical to GDL's) and
  WATERSHED (Vincent–Soille).
  **Not yet identical to GTB.** IDL's MORPH_DISTANCE does not always pick the
  nearest source next to the boundaries between the zones of different objects
  (about 1% of the pixels of the test probes), and GTB's watershed then produces
  extra basins and lines. The GTB desktop outputs in
  `tools/make_reference_gtb_desktop.md` document the differences.

### Known issues
- `euclidean_distance()` does not match GTB's Euclidean Distance, which uses the
  same IDL MORPH_DISTANCE; it will be aligned together with the tools above.

## [0.6.0] - 2026-09-25

### Added
- `fad_multiscale_change()`: GTB Change → FAD Multiscale. It gives the 7×7
  fragmentation-class transition tables A → B for every FAD-MS observation scale and
  for the multiscale map, the % of changed pixels above and below the diagonal, and
  the FAD_AV/AVCON changes. Inputs can be FAD-MS results, input maps or GTB
  `*_fad_mscale.tif` files. Counting runs in C++ with OpenMP.
- `landscape_mosaic_heatmap()` / `LMHeatmap` and `landscape_mosaic_change()` /
  `LMHeatmapChange`: GTB's Landscape Mosaic heatmap and its change, with the
  triangle grid and `to_csv()` writing GTB's `heatmap.csv` / `delta_heatmap.csv`.
- `fad_multiscale()` accepts GTB's special backgrounds 3 and 4 (codes 105/106; 4 is
  missing for the density) and reports GTB's AVCON.
- Parity tests for FAD-MS, FAD change and the LM heatmap/change against GTB's IDL
  code and `spatcon` (`LANDSCHAFT_SPATCON_EXE`).

### Fixed
- `fad()` / `fad_multiscale()`: the spatcon proportion is now computed in single
  precision, as in `spatcon.c`. Before, a few pixels could differ by 1% from GTB
  (4 pixels of GTB's `input.tif` at the 243 window).
- `landscape_mosaic()`: class boundaries now compare the single-precision
  proportions with double constants, and the pure-class test uses `FLT_EPSILON`, as
  in `spatcon.c`. Windows exactly on a boundary (e.g. 70% forest) were assigned to the
  neighbouring class before.

## [0.5.0] - 2026-09-25

### Added
- `accounting()`: GTB Accounting. It groups foreground objects into up to six size
  classes and returns the GTB colour-coded class map, object ID and object area
  maps, a per-class table and a `summary()` in the layout of GTB's `acc.txt`. It
  supports GTB's batch default classes, the dialog's `"auto"` classes, 4/8
  connectivity and "big3pink" highlighting. The C++ core uses run-length labelling,
  with OpenMP for the per-pixel maps.
- `validate_fad_input()`, `accounting_colormap()` / `ACCOUNTING_COLORMAP` (GTB's
  `mspacolorston` table), `ACCOUNTING_CLASSES`.
- Verification of IDL-only GTB tools with GDL: `tests/gtb_idl` (GTB statements
  copied verbatim), `tools/make_reference_idl.py` and
  `tests/data/reference/idl_reference.json`. Live runs use `LANDSCHAFT_GDL_EXE`.

### Changed
- **Project renamed from `guidospy` to `landschaft`** (repository:
  https://github.com/sacridini/landschaft). Use `import landschaft as ls`.
  - Output attribute `guidospy_product` is now `landschaft_product` (also in GeoTIFF tags).
  - Environment variables `GUIDOSPY_MSPA_EXE`, `GUIDOSPY_PROFILE` and the CMake option
    `GUIDOSPY_OPENMP` are now `LANDSCHAFT_MSPA_EXE`, `LANDSCHAFT_PROFILE` and `LANDSCHAFT_OPENMP`.

## [0.4.0] - 2026-09-25

### Added
- **Change Analysis**:
  - `simple_change()`: 4-class binary transition map (Stable BG, Gain, Loss, Stable FG).
  - `morphological_change()`: Morphological Change Detection (MCD) with geodesic core reconstruction ($d > 2$), $3 \times 3$ morphological closing, boundary chatter segregation (`176`), and landscape elasticity index.
  - `fos_change()`: Fragmentation (FOS) change analysis, delta FOS continuous map (+100 offset), $6 \times 6$ transition matrix across density classes, and fragmentation shift percentages.
- **Continuous Grayscale Spatial Convolution (GraySpatCon)**:
  - `gray_spatial_convolution()`: Moving-window spatial convolution over continuous grayscale/percentage rasters using an $O(1)$ 101-bin sliding histogram and OpenMP parallelization.
  - Supported metrics: Mean, Median, Majority, Min, Max, Range, StdDev, Shannon Diversity, Shannon Evenness, and Diagonal Contagion.
  - Bit-for-bit / single-precision parity verified against original GTB binary `grayspatcon64.exe`.
- **Colormaps**: Added `FOSCHANGE_PALETTE`, `CHANGE_PALETTE`, `foschange_colormap()`, and `change_colormap()`.
- **Documentation**: Complete user manual and API reference added in `DOCUMENTATION.md`.

## [0.3.0] - 2026-09-25

### Added
- **Restoration & Corridors**:
  - `restoration_status()`: Evaluates landscape connectivity before/after hypothetical restoration (ECA, DOC, CNOA, RAC).
  - `cost_distance()`: Anisotropic 8-neighbor Dijkstra shortest-cost propagation across resistance surfaces.
  - `least_cost_path()` & `restoration_corridor()`: Two-ended corridor delineation with gradient descent path tracing.
- **Landscape Morphology & Shape**:
  - `parcellation()`: Multi-class landscape division, average patch size (APS), effective mesh size (AWAPS), division index, and logarithmic PARC.
  - `contortion()`: Moore-neighborhood boundary tracing and inflection point counting (tortuosity).
- **Colormaps**: Added `CONTORTION_PALETTE`, `RESISTANCE_PALETTE`, `contortion_colormap()`, and `resistance_colormap()`.

## [0.2.0] - 2026-09-25

### Added
- **Topological Network Analysis (GTN & Conefor)**:
  - `gtn_components()`: Connected components, Equivalent Connected Area (ECA), Degree of Coherence (DOC).
  - `gtn_importance()`: Node & link importance ranking using Probability of Connectivity (dPC, dPCflux, dPCintra, dPCconnector).
  - `conefor_inputs()`: Generates `nodes.txt` and `distances.txt` ready for Conefor Sensinode.
- **Fragmentation & Distance Transforms**:
  - `fad()` & `fad_multiscale()`: Single and multiscale Forest Area Density using 2D Integral Images ($O(1)$ per pixel).
  - `euclidean_distance()`: Meijster $O(N)$ Signed Euclidean Distance transform (float and 256-level byte maps).
  - `spatial_entropy()` & `spatial_entropy_map()`: Joint directional co-occurrence spatial entropy (global index & moving-window map).
- **Landscape Mosaic & Contagion**:
  - `contagion_pff()`: Foreground pixel-adjacency co-occurrence ($P_{ff}$).
  - `clustering_fac()`: Foreground Area Clustering normalized by density ($P_{ff} / P_f$).
  - `landscape_mosaic()` & `landscape_mosaic_stats()`: 19-class and 103-class tri-polar Landscape Mosaic with barycentric statistics.
- **Colormaps**: Added `FAD_PALETTE`, `DIST_PALETTE`, `ENTROPY_PALETTE`, `LM_PALETTE` and corresponding colormap factories.

## [0.1.0] - 2026-09-25

First release: Morphological Spatial Pattern Analysis (MSPA).

### Added
- `mspa()`: MSPA bit-identical to GuidosToolbox's `mspa` v2.3 (run with `-disk`
  as GTB does), for numpy, dask and xarray inputs; parameters `connectivity`,
  `edge_width`, `transition`, `internal`, `n_threads`.
- `load_raster()` with GTB's MSPA input checks, `write_raster()` (GeoTIFF with the
  GTB MSPA colour table and processing parameters), `reclassify()`.
- `mspa_batch()` for parallel processing of many files, `mspa_stats()`.
- C++17 core with OpenMP: exact thresholded distances via disk dilation,
  run-length connected components, task-parallel pipeline stages.
- Test-suite: unit tests of the primitives, MSPA semantics, I/O and API, and
  pixel-by-pixel comparison with outputs of the original executable
  (`tests/data/reference`, optional live comparison via `LANDSCHAFT_MSPA_EXE`).
