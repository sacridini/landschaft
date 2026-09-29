# Changelog

All notable changes to Landschaft are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0 the API may still change
between minor versions.

## [Unreleased]

## [0.79.0] - 2026-09-29

### Added
- Grid distances, part 5: route overlap. `ls.cost.path_inc` (gdistance's `pathInc`): the
  flows of random walks (or, with `theta`, randomised shortest paths) from a common origin
  to every location, and for every pair of locations the weighted sum of `overlap` /
  `nonoverlap` of their flows (or of any vectorised function), with the cost of each
  connection, another transition or a list of transitions as weights. `ls.cost.barriers`
  (continuous breaks or categories, symmetric or per direction) and `ls.cost.areas`, also
  as `transition(x, "barriers" / "areas")`. Flows come from one block of solves (random
  walk) or one factorisation per location (randomised shortest paths) and the pairwise
  sums run in C++ in parallel. Agrees with gdistance to 1e-14 on 4 reference setups;
  barrier and area transitions identical. Where gdistance's `pathInc` is wrong, landschaft
  gives what its code intends: with `to` (gdistance returns only missing values) and on
  grids with cells without connections (gdistance reads its default weights from the
  wrong connections).
- Tutorial *Routes on grids*: Tobler's hiking function on the Maungawhau volcano (travel
  times, routes there and back), from least-cost paths to random walks with randomised
  shortest paths, and the routes of the R1b1b2 haplogroup across Europe against genetic
  distances. Benchmarks against gdistance on the benchmarks page.

### Changed
- `transition(x, "diff")` no longer warns about negative values (signed differences are
  its purpose).

## [0.78.0] - 2026-09-29

### Added
- Grid distances, part 4: randomised shortest paths (Saerens et al. 2009). `ls.cost.passage`
  with `theta` (net or total passages per cell or per connection, several goals as
  gdistance's `1 - 1/n` absorption) and `ls.cost.rsp_distance` (expected costs between
  points, net or total, methods 1 and 2), gdistance's `passage(theta = ...)` and
  `rSPDistance`. For symmetric transitions `I - W` scaled row by row is symmetric positive
  definite and is factorised with the circuit module's multifrontal Cholesky; asymmetric
  ones (and method 2, whose `P^theta` rows can add up to more than 1) use a sparse LU in
  nested-dissection order. gdistance factorises once per pair of points, landschaft once per
  goal (goals in parallel). Results agree with gdistance to 1e-15 typically and 1e-10 at
  worst on 6 reference setups (among them an asymmetric transition and longitude/latitude
  and global grids); method 2 is checked against a dense computation, as gdistance's own
  `rSPDistance(method = 2)` fails (it calls `adjacencyFromTransition` on a matrix). A
  1000 x 1000 grid takes 1.7 s per passage map.

## [0.77.0] - 2026-09-29

### Added
- Grid distances, part 3: random walks. `ls.cost.commute_distance` (expected commute
  distances between points, gdistance's `commuteDistance`) and `ls.cost.passage` without
  `theta` (net passages of a random walk from origins to goals, points or mask rasters,
  gdistance's `passage`). The Laplacian of the transition's graph is grounded and
  factorised per connected part with the circuit module's multifrontal Cholesky; the
  results agree with gdistance (CHOLMOD) to 1e-14 typically, 8e-12 at worst, on 5
  reference setups (projected, longitude/latitude and global grids, 4/8/16 directions).
  `gdistance_compat=True` (default) keeps gdistance's potential of the grounded cell (the
  one before it), which changes the passages around that cell; `False` gives the exact
  currents. Where gdistance fails, landschaft answers: points on the grounded cell, several
  parts (inf between parts, each part's volume), points off the graph (NaN).

### Changed
- `HabitatGraph.from_raster/from_vector(distance="commute-time")` and `distance_table`
  solve with the same engine (multifrontal Cholesky instead of SuperLU); the distances stay
  within 1e-11 of Makurhini/gdistance.

## [0.76.0] - 2026-09-29

### Added
- Grid distances, part 2: least-cost analysis over a transition. `ls.cost.acc_cost` (cost
  surface from the nearest of several origins, points or a mask raster),
  `ls.cost.cost_distance` (costs between points, or from one set to another; asymmetric
  transitions give direction-dependent costs) and `ls.cost.shortest_path` (the path as a
  transition, one transition per goal, the cells, or line strings). Identical bit for bit to
  gdistance's `accCost`, `costDistance` and `shortestPath` on 10 reference setups (among
  them uniform grids with 4, 8 and 16 directions, where many paths cost the same: the path
  is the one gdistance returns, as igraph's indexed heap is reproduced). Dijkstra runs on
  the grid itself (no graph is built), several origins in parallel; 4 million cells take
  0.6 s from one origin. gdistance fails on `shortestPath(output = "TransitionStack")` with
  several goals and on `output = "SpatialLines"` with an unreachable goal; landschaft
  returns the paths (empty ones for unreachable goals).

## [0.75.0] - 2026-09-29

### Added
- Grid distances (`ls.cost`), a port of the R package gdistance, part 1: transition
  matrices on a raster grid. `ls.cost.transition` connects neighbouring cells (4, 8 or 16
  directions, symmetric or one value per direction) with a built-in function computed in
  C++ with R's arithmetic (`"mean"`, `"min"`, `"max"`, `"sum"`, `"prod"`, `"harmonic"`,
  `"geometric"`, `"from"`, `"to"`, `"diff"`), any vectorised callable, or the Mahalanobis
  distance between the layers of a multi-band raster (`"mahal"`; R's `cov`, LAPACK's
  `dgesv` and the reference BLAS reproduced). `Transition` objects combine with `*`, `+`,
  `-`, numbers and `map()` (also over every pair of neighbours, as gdistance's
  `tr[adjacent(...)] <-`), convert to scipy sparse matrices and rasters (`to_raster`,
  gdistance's reduce methods), and `geo_correction` (types `"c"` and `"r"`, `scale`,
  `factors`), `normalize` and `sum_reciprocal` work as in gdistance. On longitude/latitude
  grids the distances between cells are the R package raster's ellipsoidal ones
  (GeographicLib with mingw's maths) and global grids wrap around the 180th meridian.
  Identical bit for bit to gdistance 1.6 on 43 reference cases (projected, non-square,
  longitude/latitude and global grids, 4/8/16 directions, the Tobler hiking example);
  4 million cells take 0.15 s. Example data of gdistance in `ls.cost.example_data`.
- gdistance's own `transition(brick, "mahal")` fails (it leaves `transitionCells` empty)
  and `normalize(tr, "symm")` gives NaN or fails on grids with missing cells; landschaft
  gives the values their code intends.
- Wheels: cibuildwheel configuration (`pyproject.toml`) for CPython 3.10-3.14 on Linux
  (x86_64, aarch64), Windows (x86_64) and macOS (arm64, x86_64), with the OpenMP runtime
  bundled (conda-forge's `libomp` on macOS 11+, delvewheel on Windows); a GitHub workflow
  that builds and tests them and publishes to PyPI on a release (run by hand, never on
  push); `tools/build_macos_wheels.sh` for macOS wheels without CI. The sdist holds only
  the sources.

### Changed
- `fad` 4 times faster (8000 x 8000 on an M4: 0.69 s to 0.17 s): density, raw density,
  classes and statistics come from one pass in C++ over integral images built once;
  `fad_multiscale` classifies in the same pass as the average (1.4 times faster). Results
  are unchanged.

### Fixed
- macOS: the build finds the OpenMP runtime of the conda environment (`llvm-openmp`) or
  Homebrew's `libomp` with any CMake. With a CMake from outside the environment
  (Homebrew's, for example) it used to build a single-threaded core (on an M4: landscape
  metrics 4.7 times, distances 3.3 times and MSPA 1.6 times faster once found).
- pandas 3: label and group properties kept their unset values as `None` (the new string
  dtype turned them into NaN and spatial analyses with properties failed).
- `nlm.mosaic_tess` / `nlm.mosaic_gibbs` warn when GEOS is older than 3.11, which orders the
  Voronoi cells differently from NLMR.
- The test suite passes on ARM64 (Apple silicon): the reference comparisons stay bit for
  bit in x86 builds (`_core.x87_math`) and allow a few units in the last place elsewhere,
  and the R arithmetic and FFT cases draw their inputs with the x86-64 bits (numpy's
  generators use fused multiply-adds on ARM64).

## [0.74.0] - 2026-09-28

### Added
- Biodiversity analysis: the inter-event interval indices of BiodiverseX (`iei_stats`,
  `iei_data`), `SpatialResult.to_biodiverse_csv` (Biodiverse's "Delimited text" export,
  the same text byte for byte), `ClusterResult.cut` (a dendrogram into k clusters),
  Biodiverse's example data (`ls.bio.example_data`) and a biodiversity tutorial.

### Changed
- Faster randomisations: spatially structured allocation (`rand_diffusion`,
  `rand_random_walk`...) 5-10 times faster, and the next iteration of the
  `rand_structured` family is randomised while the current one is analysed. Results are
  unchanged.

## [0.73.0] - 2026-09-28

### Added
- Biodiversity analysis: hexagonal and H3 grids (`BaseData.from_hex`, `BaseData.from_h3`,
  `ls.bio.hex_centres`, `ls.bio.h3_cells`, hexagons in `BaseData.to_geodataframe`), a native
  file format (`BaseData.save`/`load`, `ls.bio.save_session`/`load_session`: zips of
  Parquet tables with basedatas, trees, matrices and results), and tree conversions to and
  from Bio.Phylo, DendroPy and ete. New optional extra `bio` (h3, dendropy).

## [0.72.0] - 2026-09-28

### Added
- Biodiversity analysis, cluster analyses: `ls.bio.cluster` (agglomerative clustering of
  groups by Sorenson, Jaccard, S2, Bray-Curtis, phylogenetic Sorenson and other cluster
  indices; average, unweighted average, minimum, maximum and recalculated linkages;
  spatially constrained clustering; random tie breaking) and `ls.bio.region_grower`,
  with the indices of each node's groups, identical to Biodiverse.

## [0.71.0] - 2026-09-28

### Added
- Biodiversity analysis, randomisations: spatially structured allocation
  (`rand_spatially_structured`, `rand_diffusion`, `rand_random_walk`, or `rand_structured`
  with `spatial_conditions_for_label_allocation`: diffusion, random walk, random and
  proximity orders, backtracking, reseeding, seed location conditions), randomising within
  subsets of groups (`spatial_conditions_for_subset`, `definition_query`), labels kept in
  place (`labels_not_to_randomise`), randomised group properties
  (`randomise_group_props_by`) and shuffled tree terminal names (`randomise_trees_by`),
  identical to Biodiverse.

## [0.70.0] - 2026-09-28

### Added
- Biodiversity analysis: swap randomisations `rand_independent_swaps`,
  `rand_independent_swaps_modified` and `rand_curveball` (options `swap_count`,
  `max_swap_attempts`, `stop_on_all_swapped`, `use_max_swap`), identical to Biodiverse.

## [0.69.0] - 2026-09-28

### Added
- Biodiversity analysis: `rand_structured` randomisation (Biodiverse's default: labels to
  random groups keeping richness and ranges, with its swap phase and the
  `richness_multiplier` and `richness_addition` options), identical to Biodiverse.

## [0.68.0] - 2026-09-28

### Added
- Biodiversity analysis, randomisations: `ls.bio.randomise` randomises the basedata of a
  spatial analysis (`rand_csr_by_group`, `rand_nochange`) with Biodiverse's Mersenne
  Twister (`seed`), reruns the analysis with the same neighbourhoods and gives the
  comparison counts, p-ranks, z-scores and CANAPE categories, identical to Biodiverse.
  Runs can be continued (`Randomisation.run`).

### Changed
- `ls.bio.spatial_analysis` no longer reports indices that need two neighbour sets when
  only one is given (such as `RICHNESS_SET2`), as Biodiverse.

## [0.67.0] - 2026-09-28

### Added
- Biodiversity analysis: label and group properties (`ls.bio.read_properties`,
  `BaseData.set_label_properties`, `BaseData.set_group_properties`) and their indices
  (lists, data, value frequencies, summary statistics, quantiles and Getis-Ord Gi*, also
  weighted by local ranges); numeric label indices (statistics, quantiles, harmonic and
  geometric means, data, dissimilarity between neighbour sets, Gi*); matrix indices
  (`spatial_analysis(..., matrix=...)`: matrix statistics, Rao's quadratic entropy with a
  matrix, comparison of the values between two neighbour sets). Identical to Biodiverse.

## [0.66.0] - 2026-09-28

### Added
- Biodiversity analysis: relative phylogenetic diversity and endemism (RPD1/2, RPE1/2,
  central RPE), labels on and off the trimmed tree, RWiBaLD, and the PhyloCom indices
  (MPD and MNTD unweighted, local range and abundance weighted; NRI and NTI with their
  expected values; net VPD) in `ls.bio.spatial_analysis`, identical to Biodiverse,
  including its sampled NTI and VPD expectations (new `prng_seed` argument: Biodiverse's
  Mersenne Twister) and its path length caching on ultrametric trees.

## [0.65.0] - 2026-09-28

### Added
- Biodiversity analysis, phylogenetic indices: `ls.bio.spatial_analysis(..., tree=...)`
  computes Biodiverse's phylogenetic diversity (PD, PD per taxon, local PD, included
  node and terminal lists), phylogenetic endemism (PE, central PE, PE single, PE lists,
  PD endemism, corrected weighted endemism), clade contributions, clade losses and
  ancestral losses (PD and PE), last shared ancestor properties, evolutionary
  distinctiveness (ES, ED, AED, AED-T and its weights, corrected weighted rarity),
  phylogenetic abundance, labels on and off the tree, and phylogenetic turnover
  (Sorenson, Jaccard, S2, range-weighted turnover), identical to Biodiverse, including
  its trimming of the tree to the labels.

## [0.64.0] - 2026-09-28

### Added
- Biodiversity analysis, spatial analysis: `ls.bio.spatial_analysis` computes
  Biodiverse's indices over the neighbourhood of every group, with values identical
  to Biodiverse's: richness, redundancy and label/sample count summaries (lists,
  statistics, quantiles), endemism (weighted, corrected, absolute, central and whole,
  normalised, hierarchical partitions), rarity, turnover and dissimilarity (Sorenson,
  Jaccard, Kulczynski 2, S2, beta diversity, nestedness, Bray-Curtis, range-weighted
  turnover), Simpson and Shannon diversity, Rao's quadratic entropy and richness
  estimators (Chao1, Chao2, ACE, ICE, Hurlbert's ES). `ls.bio.list_calculations` lists
  them; results are a table of indices plus list indices, and map with `to_raster`.
  The engine runs in C++ with OpenMP.

## [0.63.0] - 2026-09-28

### Added
- Biodiversity analysis, neighbourhoods: `ls.bio.neighbours` finds the neighbour sets
  of every group from Biodiverse spatial conditions written in its Perl syntax
  (`sp_circle`, `sp_block`, `sp_rectangle`, `sp_annulus`, `sp_ellipse`, sidedness,
  text and regular expression matches, polygons and shapefiles, richness and
  redundancy thresholds, `$D`, `$C`, `$d[0]`, `$nbr_y > $y`...), with an optional
  definition query (`ls.bio.definition_query`). The lists are Biodiverse's own,
  including its reuse of lists between groups; conditions run in C++ with OpenMP
  and a grid search (half a million groups in half a second).

## [0.62.0] - 2026-09-28

### Added
- Biodiversity analysis, modern data: `ls.bio.read_occurrences` reads GBIF downloads
  (the "simple CSV" zip or file) and Darwin Core Archives (through `meta.xml`) in
  chunks and cleans them (absences, missing or out-of-range coordinates, coordinate
  uncertainty, basis of record, GBIF issues, duplicates);
  `BaseData.from_occurrences` puts the records in cells, optionally reprojected.
  `BaseData.from_raster` and `from_xarray` read stacks of species distribution models
  a block of rows at a time (Cloud-Optimised GeoTIFFs, NetCDF or Zarr through dask),
  with an optional presence threshold.

### Changed
- `BaseData` construction works on integer codes and names only the distinct groups:
  large imports are several times faster.

## [0.61.0] - 2026-09-28

### Added
- Biodiversity analysis (a port of Biodiverse), first part: the `ls.bio` module with
  `BaseData` (sample counts of labels per group, read from delimited text as
  Biodiverse does, or from DataFrames, sparse matrices, xarray stacks, rasters and
  vector files: points, and lines/polygons laid over the cell grid by cell, area or
  length), `read_tree`/`read_trees` (Newick, Phylip, Nexus with translate tables) and
  `read_matrix` (full and sparse label matrices). Groups, labels, coordinates, counts,
  tree nodes and matrices are identical to Biodiverse's; `to_raster` and
  `to_geodataframe` map per-group values.
- Tools: `tools/bio_harness.py`, `tools/bio_reference.pl` and
  `tools/make_reference_bio.py` run the original Biodiverse (Strawberry Perl) with its
  hashes iterated in sorted order, for bit-for-bit references.

## [0.60.0] - 2026-09-28

### Added
- `ls.nlm.merge` and `ls.nlm.binarize`, landscapetools' `util_merge` (a primary
  landscape plus weighted secondary ones) and `util_binarize` (habitat/matrix maps at
  given habitat shares, through `classify`).
- Documentation: a *Neutral landscapes* tutorial (a gallery of the models, habitat
  maps from continuous surfaces, a null-model test of the example forest's
  aggregation against random and fractal landscapes, reproducing NLMR).

## [0.59.0] - 2026-09-28

### Added
- Neutral landscape models, sixth part: `ls.nlm.mosaic_tess` (Voronoi tiles of random
  germs) and `ls.nlm.mosaic_gibbs` (Voronoi tiles of a simple sequential inhibition
  process with shuffled classes), identical bit for bit to NLMR with the same seed
  when shapely is installed (NLMR's value order is GEOS's order of the Voronoi cells,
  asked from GEOS through shapely; without it a warning and each germ its own value).
  25 to 70 times faster than NLMR. Every NLMR model is now available in `ls.nlm`.

## [0.58.0] - 2026-09-28

### Added
- Neutral landscape models, fifth part: `ls.nlm.mosaic_field`, sums of Poisson line
  mosaics (NLMR's `nlm_mosaicfield` with spatstat's `rpoislinetess` and
  `rMosaicField`), with NLMR's `collect` (every step) and `infinit` (the limiting
  Gaussian field of spatstat's `rLGCP`), identical bit for bit to NLMR with the same
  seed and about 1000 times faster (200 × 200 with 20 mosaics: 9.95 s → 0.01 s).
- R's `rpois()` and `rexp()` in the reproduced R generator.

## [0.57.0] - 2026-09-28

### Added
- Neutral landscape models, fourth part: `ls.nlm.gaussian_field` (a Gaussian field with
  exponential covariance by circulant embedding, nugget and mean) and `ls.nlm.fbm`
  (fractional Brownian motion from a power-law spectrum). NLMR 1.4 disabled
  `nlm_gaussianfield` and `nlm_fbm` and ships these methods as internal functions;
  Landschaft gives their values bit for bit with the same seed, 6 to 8 times faster
  (`fbm` 1000 × 1000: 0.40 s → 0.05 s).
- R's `fft()` ported to C++ (the same bits as R, with the transforms of the rows and
  columns in parallel), and R's `exp()` and `x^y` on Windows (mingw-w64's x87 `exp`
  and `powl`) for the metrics code as well (`lsm::r_exp`, `lsm::r_pow`).

## [0.56.0] - 2026-09-28

### Added
- Neutral landscape models, third part: `ls.nlm.random_cluster` (Saura and
  Martínez-Millán's random clusters) and `ls.nlm.curds` (hierarchical curdling, with
  or without wheying), identical bit for bit to NLMR with the same seed; clusters are
  numbered as `terra::patches` does and the gaps filled in NLMR's order, with the
  neighbour counts in parallel. `nlm_randomcluster` 200 × 200 takes 19 s in NLMR and
  3 ms here; `curds` is 5 to 10 times faster.

## [0.55.0] - 2026-09-28

### Added
- Neutral landscape models, second part: `ls.nlm.mpd` (midpoint displacement,
  diamond-square), `ls.nlm.neigh` (neighbourhood-based clusters) and
  `ls.nlm.random_rectangular_cluster`, NLMR's C++ models, identical bit for bit to
  NLMR with the same seed (`std::mt19937` with GCC's libstdc++ distributions).
  `mpd` keeps NLMR's grid size by default (square, `max(nrow, ncol) ± 1` cells, with
  a warning); `nlmr_compat=False` gives the size asked for. 2 to 6 times faster than
  NLMR (`mpd` 1025 × 1025: 0.13 s → 0.043 s).

## [0.54.0] - 2026-09-28

### Added
- Neutral landscape models, a port of the R package NLMR (`ls.nlm`), first part:
  `random`, `percolation`, `planar_gradient`, `edge_gradient`, `distance_gradient`,
  `classify` (class proportions, a real landscape's proportions or Jenks' natural
  breaks) and `rescale`. With the same `seed` they give NLMR's landscapes bit for bit
  (R's random number generator, `set.seed` / `runif` / `rnorm` by inversion /
  `sample`, and the libstdc++ distributions behind NLMR's C++ code are reproduced),
  3 to 110 times faster (`nlm_percolation` 1000 × 1000: 1.54 s → 0.014 s).
  - `distance_gradient` keeps NLMR's behaviour by default (`nlmr_compat=True`): terra
    treats a raster of at most 360 × 90 cells as longitude/latitude, so NLMR returns
    great-circle distances in metres there, and GDAL's proximity algorithm (float32)
    above; `nlmr_compat=False` gives exact Euclidean distances.
  - API reference page *Neutral landscape models*; `tools/make_reference_nlm.py`
    regenerates the NLMR references.
- Documentation website: a *Learn* section, a course on landscape ecology for newcomers
  (concepts, theory and academic references behind every tool, and when to use each).
- `Dockerfile`: a multi-stage image with the package built from source and every
  optional feature (`docker build -t landschaft .`); `--target test` builds an image
  that runs the test suite.

### Changed
- README reduced to the purpose, installation (conda, pip, Docker), a quick start and
  a summary of validation and speed; the details live on the documentation website.
- The project is written Landschaft in the README and on the website.
- The home page highlights the reproduced tools and out-of-core processing.

### Fixed
- The C++ core did not compile with GCC on Linux, where `int64_t` is `long`: three
  `std::max` calls mixed it with `long long` literals (grayspatcon, contortion).

## [0.53.0] - 2026-09-27

### Changed
- MSPA is 3 times faster than in 0.52 (10-17 times faster than GuidosToolbox's `mspa`),
  with the same bits: on the 2000×2000 CORINE map 95 ms instead of 312 ms with one
  thread and 78 ms instead of 239 ms with the default threads (performance cores).
  - The disk dilation behind every thresholded distance works on rows packed into
    64-bit words for radii up to 24 (4× faster).
  - The per-pixel operations between the stages (masks, saturated arithmetic, shifts,
    levels, thresholds, the missing-data and input checks) use SSE2; MSVC left them
    scalar.
  - Large buffers are reused within a call (`gtb::PoolScope`) instead of being
    requested from the operating system, which zeroed fresh pages for each of the
    ~85 intermediate images.
  - The geodesic distance is skipped for edge width 1, where it provably selects no
    pixel, touches only the pixels its front reaches, returns MSPA's 0/1 band directly
    and keeps each queued pixel's distance instead of recomputing it every round.
  - The external boundary of the core is computed once for the openings and the
    perforations; independent steps of the preprocessing, the edges and the corridors
    run as parallel tasks; the watershed skips blocks of seeds without empty
    neighbours; the thinning's initial scan runs in parallel from 1 Mpx.
- `tools/benchmark.py` alternates the one-thread and default-thread runs and reports
  the fastest of each (`--repeats`). Run one after the other, the operating system
  moving a busy process to the efficiency cores of a hybrid CPU made the later runs
  look slower, e.g. more threads slower than one; side by side, the default threads
  were faster in 178 of 180 paired runs.

## [0.52.0] - 2026-09-27

### Changed
- Threads: every parallel function now uses all cores but one by default (it used all
  of them), so the machine stays responsive, and takes `n_threads` (thread pools:
  `n_workers`). `ls.default_threads()` gives the default and `ls.set_default_threads(n)`
  changes it for the process. The rule reaches the compiled code even where a function
  has no thread argument (every call sets the OpenMP thread count of the calling thread,
  in any Python thread), the thread pools of `batch`, `RasterCollection`, `mspa_tiled`
  and the one-to-all driver, and GDAL's block decoding in out-of-core metrics.
- `n_threads` added to `fad`, `fad_multiscale`, `landscape_mosaic`,
  `landscape_mosaic_stats`, `spatial_entropy`, `spatial_entropy_map`, `contagion_pff`,
  `clustering_fac`, `euclidean_distance` (which keep `threads` as an alias),
  `morphological_change`, `fos_change`, `parcellation`, `isochrone_map_ab`,
  `distance_to_resistance`, `raster_boundaries`, `conn.protconn_raster`,
  `conn.settlement_probabilities` and `conn.gdal_areas`.
- `ls.max_threads()` returns the processors OpenMP can use.

## [0.51.0] - 2026-09-27

### Changed
- Circuit connectivity: large components are factorised in parallel (subtrees of the
  nested dissection on separate threads, the fronts above them by heights, large fronts
  by a blocked Cholesky over fixed blocks), the same bits with any number of threads.
  1000×1000 raster, 20 threads: factorisation 3.4 s -> 0.7 s, advanced mode 5.2 s ->
  2.6 s; the 1 M-node network pairwise run of 0.50.0: 7.2 s -> 5.0 s. Eigen's own GEMM
  threads are turned off (`EIGEN_DONT_PARALLELIZE`).

### Fixed
- Documentation: the network benchmark compared the resistances with CHOLMOD's to
  "3e-15" by dividing by the largest node id in the matrix; they agree to 5e-10
  (relative), the conditioning of that 1 M-node problem.

## [0.50.0] - 2026-09-27

### Added
- Circuit connectivity: Circuitscape's network (graph) mode.
  `ls.circuit.circuitscape_network_pairwise`, `circuitscape_network_advanced` and
  `circuitscape_ini` with `data_type = network`: edge lists numbered from 0 or 1,
  duplicate edges and cancelling edges as Circuitscape builds its graph, focal nodes,
  include / exclude pairs, sources and grounds by node, per-pair voltages, node and branch
  currents, cumulative currents and Circuitscape's output files. Matched with
  Circuitscape's CHOLMOD runs of its network test cases and six synthetic ones
  (`make_reference_circuitscape.py network`) to 1.1e-14.

### Changed
- The circuit solver orders graphs without geometry by a nested dissection of their own
  (breadth-first level separators, split in parallel, the same tree with any number of
  threads) instead of one dense front. On a 1 M-node network with 20 focal nodes:
  7.2 s against Circuitscape's 27.7 s (CHOLMOD) and 182 s (cg+amg).
- Pairwise solves without maps keep only the voltages of the focal nodes per pair (they
  kept the whole component's), and batches with maps are capped at about 256 MB.

## [0.49.0] - 2026-09-27

### Added
- The rest of landscapemetrics' API: `full_name=True` in `landscape_metrics`,
  `Landscape.metrics`, `sample_metrics`, `scale_sample` and `extract_metrics` (the
  columns `name`, `type` and `function_name` of `calculate_lsm(full_name = TRUE)`);
  `sample_metrics(return_raster=True)` (each plot's raster, `raster_sample_plots`, on
  terra's crop extent); the public helpers `construct_buffer`, `pad_raster`,
  `unpad_raster`, `raster_to_points` and `get_unique_values` (parallel C++). Matched
  bit for bit with landscapemetrics on four maps (`make_reference_lsm.py api`,
  `tests/data/reference/lsm_api`).

### Changed
- Circular sample plots use R's `sin`/`cos` bits (the x87 `fsin`/`fcos` of mingw-w64;
  `np.sin` differed in the last bit at 5 of the 100 vertices). The `siei` catalogue
  name is landscapemetrics' ("simspon's evenness index"); `list_metrics(name=)` takes
  both spellings.

## [0.48.0] - 2026-09-26

### Added
- Circuit connectivity, part 5d/5e: Circuitscape's one-to-all and all-to-one modes on
  rasters (`ls.circuit.circuitscape_one_to_all`, `circuitscape_ini` with those
  scenarios): focal points and regions merged into the short-circuit regions, include
  files, variable source strengths, per-point voltage and current maps, cumulative and
  maximum maps, with Circuitscape's quirks; the points are solved in parallel on one
  graph. `circuitscape_ini` now runs every raster scenario and writes Circuitscape's
  outputs. Matched with Circuitscape's CHOLMOD runs of its 25 one-to-all / all-to-one
  test cases to 5e-14.

### Changed
- One-to-all / all-to-one factorise every component once without the focal nodes and
  solve each point through the Schur complement on them (Circuitscape factorises once
  per point); ASCII grids are written by a parallel C++ formatter. On a 1000x1000 grid
  with 20 focal points, against Circuitscape with 20 threads: pairwise 2.7 s (8.7 s),
  pairwise with the cumulative map 9.5 s (76.5 s), one-to-all with maps 4.6 s (21.8 s).

## [0.47.0] - 2026-09-26

### Added
- Circuit connectivity, part 5c: Circuitscape's pairwise mode on rasters.
  `ls.circuit.circuitscape_pairwise` and `circuitscape_ini` with `scenario = pairwise`:
  focal points and focal regions (a graph per pair, the regions merged with the polygons
  they touch), include/exclude pair files, resistances with Circuitscape's shortcut and
  its -1 / 0 rules, per-pair voltage and current maps, cumulative and maximum current
  maps with Circuitscape's null, log and focal-zero options, and the resistance files.
  One factorisation per component (a node grounded) and parallel batches of pair solves;
  focal-region pairs in parallel. Matched with Circuitscape's CHOLMOD runs of its 17
  pairwise test cases (resistances to 1e-12, maps to 3e-13).

## [0.46.0] - 2026-09-26

### Added
- Circuit connectivity, part 5b: Circuitscape 6.1's raster core and advanced mode.
  `ls.circuit.circuitscape_advanced` and `ls.circuit.circuitscape_ini` (INI files with
  `scenario = advanced`): Circuitscape's node map with short-circuit regions, average
  conductance or resistance edges, masks, reclass tables, sources and grounds from maps
  or text lists, grounds as resistances, unit currents, direct grounds and the four
  source/ground conflict policies; voltage and current maps written as Circuitscape
  names them. ASCII grids are parsed in double as Circuitscape does. The multifrontal
  solver now also takes general graphs (merged nodes eliminated at the root). Matched
  with Circuitscape's CHOLMOD runs of its own advanced test cases to 5e-12.

## [0.45.0] - 2026-09-26

### Added
- Circuit connectivity, part 5a: `ls.omniscape(memory_limit=...)` (default: half of the
  physical memory) runs fewer threads when each thread's factorisation of a full window
  would not fit (large radii); `attrs["threads"]` reports the threads used. An iterative
  solver was considered and dropped: the direct one is faster up to radius 200 at least,
  and memory is the real limit.

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
