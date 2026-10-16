# Changelog

All notable changes to Landschaft are documented here. The project follows
[Semantic Versioning](https://semver.org/): from 1.0 on, the public API changes incompatibly only
in a new major version.

## [Unreleased]

### Added
- `ls.hydro.swat` (in progress): SWAT+ 62.0.1 ported to C++, read from its own TxtInOut files
  (`load_txtinout` -> `SwatProject`, `run` -> `SwatResult`), bit for bit with SWAT+ itself. So far the daily
  weather: measured series, the weather generator with its random numbers, the stations and the daily Hargreaves
  PET, checked against a dump of the reference on the Hubbard Brook and Ames examples.
- `ls.hydro.swat` routing: the daily command loop over the spatial objects, routing units with their delivery
  ratios, 1-D aquifers and SWAT-DEG channels (rating curves, variable storage or Muskingum routing, channel
  sediment, in-stream water quality, water temperature), bit for bit with SWAT+ on Hubbard Brook given the
  reference's HRU outputs.
- `ls.hydro.swat` HRUs: the land phase of each day (snow, canopy, soil temperature, PET, ET, curve number
  runoff, MUSLE, percolation and lateral flow, residue and nutrient cycling, plant growth, the lag storages and the
  HRU hydrographs), so the whole Hubbard Brook simulation, HRUs feeding the routing, is bit for bit with SWAT+.
- `ls.hydro.swat` management: scheduled operations (plant, grain and residue harvest, kill, harvest and kill,
  tillage, irrigation, mineral fertilizer, skip) and land use decision tables with HRU conditions and irrigation
  actions, bit for bit with SWAT+ on the Ames example (scheduled corn and automatic irrigation).
- `ls.hydro.swat` from the landscape: `watershed` divides a DEM with land use and soil maps as QSWAT+ does
  (subbasins and channels from stream links split at the outlets, HRUs by land use, soil and slope band with QSWAT+'s
  dominant / filter / area selection, aquifers, routing; QSWAT+'s tables), `from_tables` writes the SWAT+ project as
  the SWAT+ Editor 4.0.2 does (the same files, token for token, also from a QSWAT+ project database with
  `load_qswat_project`), `from_rasters` does both; `climate` and `wgn_from_series` make the weather files and the
  weather generator from daily tables; `hru_map`, `channel_map` and `reach_discharge` put HRU and channel results
  back on the map. A synthetic landscape and a real DEM simulate bit for bit as SWAT+ on the editor's files.
- `ls.hydro.swat` results: `run` gives what SWAT+ prints (print.prt's HRU, basin, channel and aquifer tables by day,
  month, year and average annual, with SWAT+'s sums and means) as `SwatResult.tables` / `result.table(name, step)`,
  DataFrames with SWAT+'s column names and units, bit for bit with SWAT+'s own records; `SwatProject.set_print` and
  `run(..., tables=, nyskip=)` choose them. `hru_map` and `channel_map` map SWAT+'s average annual values.
- `ls.hydro.swat`: the SWAT+ Editor's automatic crop rotations (plant, harvest, kill and reset actions of the decision
  tables) and urban land uses (impervious runoff, build-up/wash-off and USGS regression loads); `run_many` runs
  several projects at once; `load_txtinout` reads a zip; example project `ls.example_data("swat_hubbard_brook")`.
- `ls.hydro.stream_links`, `ls.hydro.watersheds` and `ls.hydro.snap_outlets`: the links of a stream network (split
  at junctions and outlets, with length, drop and Strahler order), the area draining to each seed, outlets moved
  onto the streams.

## [1.49.0] - 2026-10-16

### Added
- `ls.parallel.map`: any tool of `ls.parallel.TOOLS` (the analyses of `ls`, `ls.surface`, `ls.hydro`,
  `ls.services`, `ls.nlm`, `ls.circuit`) on many inputs, here or on helpers; arguments and results travel as data
  (rasters with their georeferencing, tables, GeoDataFrames, arrays, landschaft's result classes), never pickled.
- `ls.parallel.tiled`: a neighbourhood tool on a large (or lazily read) raster tile by tile, each tile with the margin
  the tool needs, here or on helpers: the same bits as on the whole raster for geomorphons, FAD, window metrics, TRI,
  slope, curvature, dissection, TRASP and carbon (the running-sum window means of TPI, VRM, SRR, HSP and the spherical
  standard deviation to about 1e-12, with an explicit halo).
- `on=ls.helpers()` in `sdm.fit` (each model on each cross-validation fold; the estimator travels as its scikit-learn
  or `ls.sdm` class and parameters, the scores come back, the same scikit-learn required), `uncertainty.monte_carlo`
  (chunks of draws of `ls.parallel.call(tool, ..., param(i))`), the `ls.prioritize` calibrations (`calibrate`,
  `calibrate_penalty`, `calibrate_budget`, `pareto_front`: the compiled model travels, the solution is made here) and
  `eval_replacement_importance` (without zones). The same results as on one computer.

## [1.48.0] - 2026-10-16

### Added
- `ls.services.sdr` (InVEST's sediment delivery ratio): RUSLE soil loss with the LS factor of Desmet & Govers from
  the flow accumulation, the index of connectivity of Borselli / Cavalli (D_up from the accumulated C factor and
  slope, D_dn down the flow weighted by 1 / (W S)), the SDR, sediment export, the deposition and flux of the sediment
  that does not reach a stream (InVEST's sediment_deposition, MFD and D8), avoided erosion and total retention,
  drainage given by the user, what drains to a stream, and the totals per watershed. Every raster of InVEST's chain
  in its float32 / float64 type, on pygeoprocessing's routing (`ls.hydro`) and its percent slope (ported in C++):
  InVEST 3.20.2's rasters bit for bit (MFD, D8, with drainage), the watershed totals to 3e-6 (InVEST adds the cells
  in float32). 3000 x 3000 cells: 9.8 s against 114 s.
- `ls.hydro.connectivity_index`: that index of connectivity for any weight (the C factor, or any impedance from
  0.001 to 1), with the streams given or traced from a threshold.
- `ls.hydro`, terrain hydrology: `fill` (Priority-Flood; `drain_flats=True` for Priority-Flood+epsilon), `breach`
  (least-cost channels through the rims of depressions, as WhiteboxTools' `BreachDepressionsLeastCost`, its
  min-heap reproduced so equal costs resolve alike),
  `flow_direction` (D8 as ESRI codes, D-infinity azimuths), `flow_accumulation` (D8, D-infinity, FD8 of Freeman,
  Quinn's multiple flow with contour lengths; cells, area or specific contributing area; weights), `streams`, `hand`
  (height above the nearest drainage) and the indices `twi`, `spi`, `sti`; flat areas drain by Barnes, Lehman &
  Mulla (2014). `method="mfd"` and `distance_to_channel`: the multiple flow routing of pygeoprocessing 2.4 (the
  InVEST models'), its 4-bit packed directions, accumulation, mean distance to a channel and stream tracing
  (`streams(trace=, dem=)`, its `extract_streams_mfd`), identical bit for bit
  (`tools/make_reference_pygeoprocessing.py`; its `fill_pits` gives the same DEM as `fill`). C++ (`src/cpp/hydro`);
  the directions, accumulations, HAND and indices identical to WhiteboxTools
  2.4 on four DEMs (`tools/make_reference_hydro.py`), 1.2 to 3.1 times faster on 4 cores (`tools/benchmark_hydro.py`).
- `ls.surface.geomorphons`: landforms by geomorphons (Jasiewicz & Stepinski 2013), a C++ port of GRASS GIS's
  `r.geomorphon` 8.4: from the zenith and nadir angles of the lines of sight in 8 directions, the 10 forms (flat,
  peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit), the 498 ternary patterns, the binary
  positive and negative patterns and, with `measures=True`, intensity, exposition, range, variance, elongation,
  azimuth, extend and width; search, skip, flatness threshold and distance in cells or map units, the three
  comparison modes and the extended correction. Bit for bit with GRASS on 9 stored cases (its mingw-w64 atan2,
  float32 DEM, coordinates as `Rast_row_to_northing`); `GEOMORPHON_FORMS`, `Geomorphons.share()`. 2000 x 2000
  cells, search 10: 2.5 s on 3 threads (6.1x GRASS, single-threaded; 2.5x on one thread), the zenith and nadir
  kept by their height / distance ratio and atan2 computed only at the end or at near ties.
- `ls.services`, InVEST's ecosystem service models reimplemented (natcap.invest 3.20, Apache-2.0), C++ with
  OpenMP: `carbon` (the four pools by class, storage of a baseline and a future map, sequestration, its net present
  value with InVEST's valuation constant, `valuation_constant`; landscape totals in tons) and `habitat_quality`
  (threats decaying linearly or exponentially with the distance to their nearest cell, weights, sensitivities,
  access from a raster or polygons, total degradation, quality h (1 - D^2.5 / (D^2.5 + k^2.5)), rarity against a
  base map). The maps are InVEST's float32 maps bit for bit (the exponential decay to 1 unit in the last place:
  numpy's float32 `exp`), checked against stored outputs of natcap.invest (`tools/make_reference_invest.py`).
- `ls.services.water_yield` (InVEST's annual water yield): the Budyko curve in Fu and Zhang's form by cell (PET =
  Kc ET0, w = Z AWC / P + 1.25 capped at 5, non-vegetated classes at min(Kc ET0, P)), water yield, actual and
  potential evapotranspiration, demand, and per watershed and sub-watershed (`ls.zonal_stats`) the means, the yield
  volume, consumption, realized supply and the hydropower energy and net present value. The maps are InVEST's bit
  for bit in its float32 or float64 arithmetic (float32, int16 and float64 inputs checked); the watershed tables
  agree to 1.2e-7 (InVEST adds the cells in float32). 3000 x 3000 cells: 0.8 s against 11 s.

### Fixed (against WhiteboxTools)
- `fill` lets water out at the map's edge and at missing cells: WhiteboxTools' `FillDepressions` raises a
  depression that touches the edge (or a missing cell) together with the edge cells in it, when the depression grows
  from an interior pit.
- `breach` cuts a channel that reaches the map's edge: WhiteboxTools' `BreachDepressionsLeastCost` walks back from
  the cell beyond the edge, which has no way back, and leaves the channel uncut.

### Fixed (against GRASS GIS's r.geomorphon)
- Lines of sight reach the DEM's last row (`r.geomorphon`'s row buffer leaves it out for the cells of its bottom
  `search` rows); the binary positive and negative patterns set a bit per direction (`r.geomorphon` adds the
  direction numbers); the extended correction's second, shorter search starts next to the cell (`r.geomorphon`
  keeps the inner radius, though it sets it to 0); a geomorphon whose first direction has no point keeps the shape
  of its other points (`r.geomorphon` gives none).

### Fixed (against InVEST)
- Carbon: a change of exactly -1 t/ha is a change, not a missing cell (InVEST writes the change map with -1 as its
  nodata, so those cells drop out of the change total and get no net present value).
- Habitat quality: a threat without any cell degrades nothing (InVEST's distance transform puts every cell at about
  half the raster's perimeter from it, which a long enough maximum distance turns into degradation everywhere).
- Habitat quality: a base map with threat maps gets its degradation and quality (InVEST 3.20.2 stops with
  `KeyError: 'deg_sum_b'`, an output it never registers).
- Water yield: the hydropower value with a discount rate of 0 is the time span times the annual value (InVEST
  gives 0: its geometric sum is left at 0 when the ratio is 1).
- Water yield: `rsupply_mn`, the realized supply per cell, is in m3 per cell, the yield of a cell (mm over its
  area) minus its demand (InVEST subtracts the demand in m3 per cell from the yield in mm).

## [1.47.0] - 2026-10-16

### Added
- Work shared with the other computers of a lab (`ls.parallel`, adapted from zeit's tile runner): a computer lends
  itself with `landschaft worker` (the new `landschaft` command; the first time `--join <code>`, with the code that
  `landschaft helpers invite` shows), and `batch`, `mspa_batch`, `simulate.scenarios` and `simulate.calibrate`
  (grid and random searches) take `on=ls.helpers()`: their tasks run here and on the helpers, with the same
  results as on one computer (seeds set here, outputs put together in task order, the same version and code
  required). A task is a registered operation, its JSON parameters and its input files, never code: helpers run
  only landschaft's own operations, in child processes, connect out over TLS and prove the group's secret (HMAC
  bound to the certificate). Discovery by UDP broadcast (`--connect` otherwise), input files from the helper's
  cache (by the hash of their bytes), a shared folder (NAS) or the network, outputs straight into a shared output
  folder, retries on another computer, stragglers run twice near the end, helpers that leave give their tasks back.
  `landschaft helpers list|firewall|leave`; tutorial "Several computers".
- `ls.load_result`: a result folder written by `ls.save` read back as a dict of its maps, tables and numbers.
- Documentation of the 1.46 tools: the course chapters on continuous surfaces (summits, saddles and protected
  hilltops; the gravity model, with checks by hand) and on land change (stocks and endogenous roads); the surface
  texture tutorial's hilltop protection areas; the land change tutorial's gravity pull, roads that grow with the
  model and stocks taken by patches; their speed and accuracy against Dinamica EGO on the benchmarks page, with
  `tools/benchmark_dinamica.py` (4 cores).

## [1.46.0] - 2026-10-16

### Added
- `ls.surface.gravity`: the gravitational interaction potential of source cells, sum of m / d^beta (Dinamica EGO's
  `CalcInteractionPotentialMap`, clean-room): distances in cells (a source's own cell at 1) or map units, a mask,
  direct sums in C++ (one table of weights by offset, the same bits with any thread count) or by FFT for many
  sources. Within 3e-6 of Dinamica EGO 8.13 (which sums float32 terms in float32), 3.7x faster on 4 cores.
- `ls.simulate.gravity_of`: the potential of the cells of some classes (optionally weighted by a mass map) as a
  dynamic variable, recomputed from the simulated map at every step. `Model.save` does not write it yet.
- `ls.simulate.ContinuousPatcher`: new patches that take a stock (biomass, carbon, timber) up to a quota (Dinamica
  EGO's `PatcherContinuousState`, clean-room): `allocate` (stock left, depleted probabilities, stock taken, patch
  numbers, quota left) and `run` (a quota per step, optional regrowth). Candidates, pruning (summed stock up to
  prune factor x quota), seeds drawn in proportion to the probability and the quota rule measured in Dinamica EGO
  8.13 (exact on the cases that do not depend on the draw, statistical on the draw); patch growth as the Patcher's
  (C++ `sim/continuous.cpp`). 3000 x 3000 cells: 0.41 s against 2.9 s.
- Endogenous roads (Dinamica EGO's road constructor, clean-room): `ls.simulate.friction_cost` (`CalcCostMap`: a
  step costs the friction of the cell entered, sqrt(2) on diagonals; Dijkstra in C++, the same bits as Dinamica's
  push-broom run to convergence, 1.2 s against 308 s on 3000 x 3000 cells; directions to the predecessors),
  `destinations` (`CalcDestinationMap`: draws by attraction in quadrants, a minimum distance between them),
  `pathways` (`CalcPathwayMap`: down the costs to the network, ties to the first in row order or by lottery, or
  along the directions; the same maps as Dinamica), `RoadBuilder` (the `ClassicRoadConstructor`: destinations at a
  range of distances from the roads, joined to the network each step; `step`, `run`) and `RoadDistance`
  (`RoadBuilder.dynamic`: the distance to the roads as a dynamic variable that builds a step of roads each time
  the simulated map changes).
- `ls.surface.hilltop`: hilltops as permanent protection areas (APP de topo de morro, Lei 12.651/2012 art. 4 IX):
  every summit with its base (the saddle where it meets higher terrain, the plain or water body it reaches, or the
  lowest elevation of its area), height, mean slope and line slope; the hills of at least 100 m and mean slope above
  25 degrees (configurable, e.g. CONAMA 303/2002's 50 m and 30 % along the line) and their upper third. A merge tree
  of the cells swept from the highest down in C++ (radix sort, 32-bit union-find with a forest of links); 3000 x
  3000 cells in 1.7 s on 4 cores. An open replacement of Dinamica EGO's `CalcHilltop` (CONAMA 303 on a DEM cut into
  elevation bands, whose dip slope reaches to the map's edge or other features: 5 degrees for an isolated cone of
  21.8).

### Fixed (against Dinamica EGO)
- `destinations` covers the whole map with its quadrants (Dinamica gives each `n // q` rows or columns: the last
  `n % q` never get a destination) and draws in exact proportion to the attraction (Dinamica rounds its draw to an
  integer of the summed attraction: the first cell gets half its share and a draw past the end takes the
  quadrant's last cell, even of no attraction).

### Fixed
- The names of the landscape mosaic's classes 6 to 12 (`LM19_CLASSES`): they did not match the classes
  `landscape_mosaic` gives (two of them had one name). Each name lists the dominant cover first (above 60%), then
  those at 10% or more: 6 Developed-Forest, 7 Developed-Agriculture, 8 Forest-Agriculture, 9 Forest-Developed,
  10 Agriculture-Forest-Developed, 11 Developed-Agriculture-Forest, 12 Forest-Agriculture-Developed.

## [1.45.0] - 2026-10-13

### Added
- `Model.save` writes potentials Terraflux has no form for (an `Estimator` of scikit-learn, `MultiCriteria`,
  MOLUSCE's weights or any piece with `predict`, or `probability` for the Expander and Patcher) as their
  predictions on the starting map, the layers `Model.run` reads from them; `terraflux run` gives the same maps.
  A model with dynamic variables needs a potential Terraflux computes (logistic regressions, weights of evidence).
- `Model.save` writes regional demands (`Regional`: the regions map and each region's area demand, Markov chain
  or rate table) for the CLUE-S, ordered, CARS and Expander/Patcher allocations; `Model.load` reads them back.
  Terraflux (`demand.regional`) allocates region by region as `Model.run` does: the same maps.
- `Model.save` writes the dynamic variable `distance_to(classes, cost=transition)`: the transition's values, a
  layer per direction, and its layout; Terraflux computes the same accumulated costs (gdistance's `accCost`, the
  C++ of `ls.cost`).
- `Model.save` writes MOLUSCE's simulator (`MolusceCA`): `MolusceWoE`'s fitted maps, or its weights when the
  drivers move (dynamic variables), or a potential per class; numpy's generator and the cross table. Terraflux
  (`rng.numpy`, `molusce.total`, `molusce.step`) gives the same maps.
- `Model.save` writes CLUMondo models: the settings as a run compiles them (supply, possibly by step, demand for
  goods and services, conversion order and resistances, the conversion matrix with its codes and maps, the 2017
  executable's demand kinds, deviations per service and patches, land system roles' caps, conversion orders and
  successions), the regressions or suitabilities, neighbourhood functions, location preferences and ages.
  Terraflux (`clumondo.ages`, `clumondo.step`) runs them as `Model.run` does, a year without solution included
  (the simulation stops there). Drivers given per step are not written.
- `Model.save` writes FUTURES models (`PatchGrowing` with `DevelopmentPotential` and `DevelopmentDemand`):
  subregions, potential subregions, patch libraries, the starting development pressure (computed when not given),
  weights, zoning and its effects, restrictions and the flood submodel (`Flooding`: depths or flood probabilities
  and HAND, damage curves and regions, response, adaptation, redistribution); `Model.load` reads them back.
  Terraflux (`futures.simulation`) gives the same maps.

### Changed
- MOLUSCE's prediction, total weights and simulator step run in C++ (`src/cpp/sim/molusce.cpp`, with numpy's
  legacy `RandomState.random_sample`), so that Terraflux shares them; the same bits as before and as the patched
  MOLUSCE. Among cells of equal confidence and draw the later cells change first (numpy's quicksort left the
  order to its implementation).
- The neighbourhood functions and location preferences of CLUMondo's yearly suitability run in C++
  (`clumondo_add_neighbourhood`, `clumondo_add_preference`), the same floats as before.

## [1.44.0] - 2026-10-10

### Added
- `Model.save(path, landscape, drivers, seed=...)` writes a fitted model as a
  [Terraflux](https://github.com/sacridini/terraflux) graph (JSON): the fitted pieces as they are (weights of
  evidence, logistic regressions, Markov chains, area demands, Expander/Patcher settings, lulcc's rules) and the
  data it runs on as GeoTIFF files next to it (or the paths given). `terraflux run` gives the same maps as
  `Model.run` with the same seed, bit for bit: CLUE-S, the ordered allocation, the Expander and Patcher and CARS,
  with weights of evidence, logistic regressions or given maps as potential, and the dynamic variables
  `distance_to` (Euclidean) and `share_of`. `Model.load(path)` reads such a graph back as `(model, inputs)`.
- `ls.simulate.terraflux(path)` runs a model with the `terraflux` command and returns the simulated maps.
## [1.43.0] - 2026-10-10

### Added
- `LANDSCHAFT_PYTHON=OFF` (CMake) builds and installs only the static C++ libraries and their headers, without
  Python, pybind11 or the MILP solver: for native programs that link them, such as
  [Terraflux](https://github.com/sacridini/terraflux), which runs `ls.simulate`'s models without Python.

### Changed
- The steps of `ls.simulate` that were numpy code run in C++ (`src/cpp/sim/rules.cpp`, `ordered.cpp`,
  `demand.cpp` and `woe.cpp`), so that native programs share them: lulcc's conversion rules, automatic
  conversions and neighbourhood rules, the ordered allocation, the linear demand (`approxExtrap`, `roundSum`),
  Markov chains (transition matrices from counts, the matrix root with Eigen's matrix logarithm and exponential
  instead of SciPy's, projections), the ranges, weights and probabilities of weights of evidence, and the
  patches observed between two maps (`presets.observed_patches`, without `scipy.ndimage`). The results are the
  same bits as before (lulcc's reference cases included), the Markov roots within 1e-14 as before.
## [1.42.0] - 2026-10-10

### Added
- Evaluation of solutions with management zones: `eval_n_summary`, `eval_cost_summary`,
  `eval_boundary_summary` (`edge_factor` per zone, `zones=` matrix), `eval_connectivity_summary` and
  `eval_asym_connectivity_summary` (`zones=`), `eval_feature_representation_summary` and
  `eval_target_coverage_summary` give a row `overall` and one per zone (the coverage a `zone` column), bit
  for bit as prioritizr 8.0.6's (4 more reference cases, proportion decisions included);
  `eval_replacement_importance` (every selected unit-zone column left out in turn) and
  `eval_rank_importance` (`by_zone`: a budget per zone in equal steps) give a layer per zone;
  `eval_extinction_risk` sums the zones; `eval_landscape_metrics` takes the map of the zones (a class per
  zone, with a `zone` column) or one `zone`, `eval_mspa` one `zone`; `report()` works with zones. Ferrier and
  rarity-weighted importance refuse zones, as prioritizr does.
- `ls.plot` of a solution with zones: the map of the zones with their names and colours (the zone map is a
  `ZONES` product, also in static figures), the coverage of every target with its zones, the cost and units
  of every zone; portfolios with zones map how often every unit goes to every zone.
- Documentation: a section on zones in the tutorial "Optimized conservation plans" (Serra Verde: strict
  protection, sustainable use and restoration of pasture and cropland), "Zones, actions or restoration" in
  chapter 12.8 of the course, zones in the API reference and the migration guide.

## [1.41.0] - 2026-10-10

### Added
- Landschaft's own engines for problems with management zones, used by the HiGHS solver (HiGHS's MIP stalls on
  them from 1e5 units: the zones make n_zones times more columns and a row per unit):
  - minimum sets (`src/cpp/opt/zones.cpp`): the Lagrangian of the targets decomposes by planning unit (every
    unit takes the zone of least reduced cost, or none; with mandatory allocation the best zone), a bundle
    method finds its duals, sifting over units solves the linear relaxation exactly (its bound), fractional
    units are rounded to their largest share, the targets left unmet repaired and the units not needed dropped
    or moved to cheaper zones; it stops as soon as a repaired Lagrangian solution is within the gap of the
    bound, and HiGHS's MIP goes on from the solution after Lagrangian fixing only when the gap is not reached.
    1e6 units x 3 zones at a 1% gap in 4 s (Gurobi 220 s; HiGHS no solution in 300 s);
  - budgeted objectives (`src/cpp/opt/zones_wide.cpp`: max utility, min shortfall, max features, one budget or
    one per zone...): duals from the LP on a 2% sample of the units, sifting over units with the few other
    columns always in, the fractional units rounded down and up with the other columns re-optimised, units
    added by reduced cost while the budgets allow. Max utility on 1e6 units x 3 zones in 13 s (Gurobi 49 s);
  - boundary penalties and connectivity rewards (`src/cpp/opt/zones_cut.cpp`, zones matrices >= 0): the
    Lagrangian of the targets is one minimum cut over every unit-zone column (with multipliers of the
    one-zone-per-unit rows), its solutions repaired and improved by local search and by expansion moves (every
    unit keeps its zone or moves to a given zone, by a minimum cut over the units; Boykov, Veksler & Zabih).
    1e5 units x 3 zones within 1.1% of the bound in 22 s (Gurobi 121 s to 0.25%; HiGHS no answer in 1500 s).
- Portfolios with zones (cuts, top, gap and shuffle): every solution's zone map and how often every unit goes to
  every zone.
## [1.40.0] - 2026-10-09

### Added
- Spatial penalties and constraints with management zones: boundary penalties (`edge_factor` one or one per
  zone; `zones`: `"same"`, the default, `"any"`, or a zones x zones matrix of how much the boundary between two
  zones counts, from -1 to 1), symmetric and asymmetric connectivity penalties (`zones=` likewise), linear
  penalties (a penalty per zone; the same data in every zone, a mapping zone -> data or a raster with a layer
  per zone) and constraints (summed over every zone), neighbour constraints (`k` one or one per zone, `zones` a
  binary matrix of the zones that count as neighbours) and contiguity constraints (`zones`: the clusters of
  linked zones whose units form one patch; exact, as without zones). Compiled bit for bit as prioritizr
  8.0.6's `compile()` (10 more reference cases) with the same optima as prioritizr on Gurobi. Feature
  contiguity and corridors are not available with zones.

### Fixed
- prioritizr's default adjacency with zones (neighbour and contiguity constraints) takes only the cells with a
  cost in every zone, so a unit missing from one zone loses all its neighbours: Landschaft takes the cells
  with a cost in any zone, as prioritizr's boundary matrix does.
- prioritizr's linear penalties with zones leave out the penalties of every zone when one zone's penalty is 0:
  Landschaft applies the others.

## [1.39.0] - 2026-10-09

### Added
- Every objective of `ls.prioritize` with management zones: minimum shortfall, minimum largest shortfall,
  maximum utility, cover and features, maximum phylogenetic diversity and endemism, minimum penalties and
  the minimum extinction risk (ours: every feature's amounts summed over its zones by their contributions).
  Budgets are one number or one per zone (a list in the zones' order or a mapping zone -> budget); feature
  weights are one per target, or, without targets, one per feature (the same in every zone) or a features x
  zones matrix. Sub-regional targets (an extension) take a `zone` column. Compiled bit for bit as
  prioritizr 8.0.6's `compile()` (11 more reference cases; the phylogenetic objectives with the tip-order
  fix) with the same optima as prioritizr on Gurobi.

## [1.38.0] - 2026-10-09

### Added
- Management zones in `ls.prioritize` (prioritizr's zones): every planning unit goes to at most one of
  several zones (strict protection, sustainable use, restoration...), each with its own cost and its own
  contribution to the features. `problem()` takes the zones' costs (a mapping zone -> raster, a raster with
  a band per zone, or a mapping zone -> cost column for tables and polygons) and either the features once
  with a `contribution=` per zone (a number, one per feature, or a raster / column of factors) or every
  zone's own features (a mapping, prioritizr's `zones()`); Marxan-style tables take a `zone` column and a
  `zones=` table. Targets are summed over the zones by default (`add_relative_targets(0.3)`: 30% of every
  feature's total, its amounts counted by their zone's contribution); `zone=` adds targets of one zone or a
  list of zones, a features x zones matrix gives every zone its own targets, and `add_manual_targets`
  takes a `zone` column (a zone or a list). `add_mandatory_allocation_constraints()`, locked units per zone
  (a mapping zone -> units, or a raster with a band per zone), manual bounded and locked constraints with
  a `zone` column (or `zone=`), binary, proportion and semicontinuous decisions. The compiled model is
  prioritizr 8.0.6's `compile()` bit for bit (13 reference cases: 2 and 3 zones, rasters, tables, polygons,
  Marxan tables, costs missing in some zones) and the solver reaches prioritizr's optimum on Gurobi. The
  solution has `values` (units x zones), a layer per zone (`selection`), the zone of every unit (`zones`, a
  map of zone numbers or a column of names), `zone_costs` and the coverage per target with its zones.
  `category_layer` and `binary_stack` convert between zone maps and stacks of zone layers (for example to
  lock an existing zoning plan). This first part covers the minimum set objective on HiGHS's MIP and Gurobi;
  the other objectives, spatial penalties, Landschaft's own engines and the evaluations follow.
- The feature matrix of raster planning units builds every zone's layers in one pass in C++ (the features
  times each zone's constant or per-cell contribution, in parallel over the layers).
- Reference group `zones` (tools/make_reference_optimize.py) and three zone families in
  tools/benchmark_optimize.py (`zones_minset`, `zones_maxutil`, `zones_boundary`), with the built-in HiGHS
  as a solver (`builtin`).

### Fixed
- prioritizr's manual bounded and locked constraints open a zone where a unit has no cost (at cost 0):
  with zones, Landschaft keeps such a unit out of the zone, and a lower bound above 0 there is an error.

## [1.37.0] - 2026-10-09

### Added
- Conservation and conversion in restoration plans (plangea's `C`, `D` and `R + D`):
  `ls.prioritize.restoration_problem(restorable, conservable=, convertible=, threat=)`. Every action is a
  block of shares (from 0 to what can be acted on, times `limit`, now also per action); conserving keeps
  the habitat that the baseline loses (`threat` of the natural land, 1 as plangea), converting takes it
  away (with a positive share converted instead of plangea's negative x), so the benefit stays concave and
  the exact method proves the optimum of every form. Area targets of one action, of all of them or of the
  `"net"` change of natural land (`action=`); linear values and costs per action (a mapping); the
  intactness benefit takes the footprint's `increase` per unit converted; the solution has `conserved`
  and `converted` maps and the area of every action (`areas`); `evaluate(restored=, conserved=,
  converted=)`; `eval_risk_reduction` reports the `baseline` habitat. `ls.sdm.area_of_habitat(...,
  loss=True)` and `ecosystem_areas` give the habitat lost per unit of natural land lost (the current
  natural cover). plangea's procedure (`method="steps"`) for C and D matches its `scen` loop (plans to
  1e-12, refreshed layers to 1e-7).
- Targets per region in restoration plans (plangea's sub-region targets): `add_projected_change` (the
  econometric projections of natural land per region: a plan that restores and converts acts in every
  region in the projected direction only, its net change as projected; a plan that only restores takes
  the land demanded elsewhere from the restoration, as plangea's `harmonize_sr_sparable`, or restores what
  each region spares); `add_area_target(..., relative=True)` (shares of the area that can be acted on, the
  flat targets); `add_linear_target` (targets on production, opportunity cost or any value);
  `ls.prioritize.ecosystem_targets` (a target shared among ecosystems for the least risk of collapse,
  exactly, or by plangea's steps of `harmonize_sr_ecosys`, which it matches to 1e-12).
- Trade-off curves of restoration plans: `RestorationProblem.tradeoff(levels)` solves the plans of a growing
  target, nested (every plan contains the previous, each the exact best given it: a priority order) or as
  the frontier (each level optimal on its own), or by plangea's gradient steps (`method="steps"`), with the
  gradient map of `post_gradient_output` (how many levels act on a cell). `compare_weights` solves with
  sets of the benefits' weights and ranks them by plangea's `post_choose_weights` metric, also as
  `ls.prioritize.mean_sd` beside `topsis`.
- The exact ecological intactness in restoration plans: `add_intactness_benefit(..., form="exact")` takes
  the total intactness itself instead of its derivative at the plan of nothing. It is not concave (a sum
  of exponentials of the footprint), so the exact method climbs by successive concave minorants (the
  tangent of every pair's exponential in the footprint's reduction, each step solved exactly, the benefit
  never falling) to a local optimum, and bounds the best plan by the secant of every pair over the
  footprints the plans can reach (new C++ kernel `rest_intactness_secant`); the status is `"local"` unless
  the two meet. Where the footprint can be cleared the plan gains much over the linearised one (2.5 times
  the intactness in a test of 2000 cells); the bound's gap is 2-20 % in the tests.
- Restoration plans at a global scale: the exact method proves the optimum of 9 x 10^6 cells and 5000 species
  (4.3 x 10^8 habitat gains) in 10 s, 4 x 10^6 x 4000 in 3.4 s instead of 8.3 s (11 threads). Its vertices
  under one area target or budget are fractional knapsacks solved by weighted selection in C++ instead of a
  linear program, the products of the gain matrix with plans and slopes run in parallel (C++), and the gain
  matrix is cut to the planning units by a remap of its columns (5.6 s to 0.6 s). The HiGHS wrapper no
  longer builds an array of strings for models of continuous columns only. `tools/benchmark_restoration.py`
  times it on synthetic data.
- The cheapest restoration plan for a benefit (`add_min_cost_objective`) searches the budget by regula falsi
  with Illinois' modification instead of a bisection (the most benefit is concave in the budget): 11
  solves where the bisection took 26, about 20 instead of 25 for a benefit within 0.1 % of the most.
- Corrected in plangea (only the correct form): in its trade-off loop the habitat is refreshed from the
  current gradient step's restoration only, forgetting the earlier steps, and a region's "<=" target
  caps every gradient step instead of the whole plan; `post_choose_weights` takes the mean of the
  criteria together with their standard deviation.
- Corrected in plangea (only the correct form): its econometric targets of R + D do not work (the
  per-region bounds are set only for C + R, which its checks forbid, and the compensation of demanded land
  runs for R + D too, zeroing the conversion targets); with more land demanded than spared it gives
  restoration negative targets (here none).
- Corrected in plangea (only the correct form): under C its habitat update subtracts the natural share
  (0-1) from an area in km², and its first slopes are taken at the current habitat instead of the
  baseline's; under C and D the original cover (and so the potential habitat) is computed on the
  problem's pixels only, so a pixel converted in full adds no potential habitat.

## [1.36.0] - 2026-10-09

### Added
- Maps of out-of-core landscapes (`ls.Landscape(path, block_rows=...)`): `labels()` and `to_raster()`, which
  needed the whole raster, now return lazy (dask) rasters on the file's grid, one chunk per block: each block
  is read and labelled again when computed (in any order), with the same ids and values as in memory. With
  `path=` (new for `labels()` too) the map is written to the GeoTIFF block by block, never whole in memory.

### Changed
- `ls.window_metrics` is about twice as fast, with the same values bit for bit: every landscape metric
  in a 21 x 21 window over a 1000 x 1000 map takes 3.7 s instead of 7.5 s (11 threads), the number of
  patches 0.60 s instead of 1.45 s. Each thread now refills one landscape window after window and keeps
  its memory (patches, core areas, adjacencies, patch values), the copy of the window hands the landscape
  its classes ready, patches are labelled with the decision tree of Wu, Otoo and Suzuki (2009) (a union
  only where two pieces meet), cores at the default `edge_depth` = 1 come from the four neighbours without
  distances, circles find the hull of their corners without sorting them, and small landscapes no longer
  open (serialised) OpenMP regions. Whole maps gain too.
- `ls.reproject` with `resampling="cubic"` is 1.6 times as fast, with the same values bit for bit (still
  GDAL's with `tolerance=0`): 8000 x 8000 float32 cells from UTM to EPSG:4326 in 0.27 s instead of 0.43 s
  (11 threads), now close to bilinear (0.21 s). The four rows of the 4 x 4 kernel go two at a time through
  SSE2 registers, each summed in GDAL's order, and the cells are checked for NoData four at a time (NaN on
  the values themselves).
- Out-of-core landscapes (`block_rows`) work on several blocks at once: a pool of threads labels the blocks
  (one thread each) while the next ones are read ahead, and the blocks are merged in order, so the results
  are the same bits as in memory. On 100 million cells, building the landscape takes 0.56 s instead of
  1.21 s and the core, enn, gyrate and circle pass 0.86 s instead of 1.93 s (a 1000-column raster: 0.74 s
  and 0.66 s instead of 1.13 s and 2.09 s), close to the whole map in memory. `block_rows="auto"` is now
  64 rows (at least 32,768 cells), worked on one block per thread; larger blocks you choose still go one at
  a time, on every thread. Core areas out of core also take the four-neighbour shortcut at `edge_depth` = 1,
  and the class values of the file are found in C++.
- `ls.write_raster` computes the chunks of a dask array several at a time (one per thread, at most 32 million
  cells) before writing them in order, instead of one after the other.

### Fixed
- `ls.window_metrics` with FRAGSTATS contrast or similarity tables (`contrast=` for `cwed` and `teci`,
  `similarity=` for `simi`) looked the tables up by the window's class indices instead of the class values,
  so most windows got NaN or the wrong weights.

## [1.35.0] - 2026-10-09

### Added
- Tutorial *Restoration priorities* (docs/tutorials/restoration-plans.md): a complete restoration plan for
  Serra Verde, from the original vegetation and the species' areas of habitat to plans for an area target,
  several benefits (species, ecosystems, carbon, intactness) and their trade-offs, budgets, the cheapest plan
  for a benefit, regional targets and uncertainty, with ten figures (`rest_*` in
  tools/make_prioritize_figures.py).
- Example data for restoration plans in `ls.prioritize.example_data`: `land_cover` (shares of six classes),
  `past_cover`, `footprint`, `ecoregions` and `carbon_potential`, derived from the Serra Verde landscape (the
  other layers are unchanged).

### Fixed
- The exact restoration solver (`RestorationProblem.solve()`) did not close its bound when a species was
  restored to its whole potential habitat (the benefit has a kink there): on the tutorial's data a plan of
  300 km² stopped after 5000 vertices 1.8 % from its bound, reported as `"time_limit"`. Such species are now
  lifted (their habitat gained a variable of its own, at most the potential), and the plan is proven optimal
  in 2 vertices; a budgeted plan that stopped after 18 s without a proof now takes 0.01 s. Running out of
  iterations is now reported as `"iteration_limit"`.

## [1.34.0] - 2026-10-08

### Added
- `ls.plot` of a stack (a time series) while Python draws detail: once the view rests, the other bands
  are asked for ahead, the nearest first, eight at a time over threads (the 41 bands of a 5000 x 5000
  MSPA series in 0.37 s of Python), so the band slider (even at a step every 16 ms) and the play button
  show every band sharp at once: no coarse frame in the tests, forwards, backwards and after a pan. The
  details of the bands seen are kept: decoded up to 256 MB (the bands around the one shown first), the
  others as their PNG, decoded when shown.

### Changed
- Rasters are reprojected for maps once per grid: GDAL works out the source cell of every pixel (the
  same pixels as before) and the bands of a stack, or images of the same grid, only look their pixels up
  (C++): a stack of 47 bands in 0.56 s instead of 0.81 s, the 41 bands of the series above in 0.2 s
  instead of 1.4 s.
- The detail of a view is that view's pixels on the screen: their source cells worked out in C++ from
  PROJ every 16 pixels and linear between, within a hundredth of a percent of PROJ pixel by pixel (GDAL's
  approximate transformer was off at 3 to 6 % of the cells' edges), with PROJ's transformation made once
  per CRS (setting up a CRS read from a file, such as EPSG:10603, took 50 ms, every time). A detail takes
  15 to 40 ms instead of 65 to 300 ms.

## [1.33.2] - 2026-10-08

### Fixed
- The band slider of a stack was slow while Python draws detail (1.33.1): every position it went through
  asked for that band's detail and waited for it, up to 400 ms, so a drag over 20 bands put one on the map,
  after 1.2 s. The slider shows a band at once again (20 bands in a 0.9 s drag, each as it is reached) and
  the detail is asked for once it rests; the details of the last 12 bands seen are kept for their view, so
  a band seen again shows sharp at once. Playing still waits for each band's detail (no coarse frames).

## [1.33.1] - 2026-10-08

### Fixed
- `ls.plot` of polygons with over 20 000 features or a million vertices (Brazil's municipalities) failed with
  "too many values to unpack": simplified without its topology, a ring that crosses itself is mended by GEOS
  into a MultiPolygon, which the WebGL layer and static figures now take part by part.
- Playing the bands of a stack (or moving its slider) while Python draws detail showed every band coarse,
  then sharp: a band is now put on the map together with its detail, made ready beforehand (a frame of the
  animation waits for it, 0 coarse frames instead of about a third). A detail replaces the one before it once
  drawn, never leaving a gap, and a zoom asks for the detail of the view it goes to as it starts.
- The detail of a view came in 0.3 s for a raster in a projection such as EPSG:10603 (Brazil's Albers): GDAL
  set the transformation up again in every thread. Images of up to 4 million pixels are reprojected in one
  thread: 0.06 to 0.1 s, within the zoom's animation. Around the view a quarter of it is asked for, not half.

## [1.33.0] - 2026-10-08

### Added
- `ls.plot`: detail as you zoom in. While Python runs, a map asks for the part in view at the screen's
  resolution once it is zoomed in past the overview its page holds: a raster's own cells (read through a
  window of the raster, a lazy one from its file) in the overview's colours and classes, reprojected at the
  screen's resolution, the value under the cursor read from them; vector layers drawn as an image burnt
  again for the view; WebGL layers given the features in view simplified to the screen (a spatial index made
  at the first question). The questions go through pywebview's JavaScript API in a script's window, and in a
  notebook (Jupyter, JupyterLab, VS Code) through a widget of no size shown under the map (anywidget, now a
  dependency): no server, no port. The map itself stays an HTML output, so a saved notebook or page keeps
  the overview. A detail takes 10 to 100 ms for a raster in memory. `LANDSCHAFT_PLOT_LIVE=0` turns it off.

## [1.32.0] - 2026-10-08

### Added
- `ls.plot`: vector layers of 20 000 to 2 million features are drawn on the map by the graphics card
  (WebGL) instead of as an image: sharp at every zoom, the value of the feature under the cursor shown as
  for rasters (its colouring column, or read back from its colour), its columns in a popup on a click,
  with the opacity slider, the layer control and the swipe of `compare=True`. The features are
  simplified to an eighth of a pixel in their own CRS, only their coordinates reprojected, polygons cut
  into triangles in C++ (mapbox's earcut, vendored, ISC) over threads, and packed into the page as
  quantized deltas in byte planes, deflated in pieces in parallel (the browser inflates them with
  `DecompressionStream`). 200 000 polygons with three columns: 0.99 s and a 5.3 MB page (the image took
  as long and could not be read); 2 million points: 0.70 s. Larger layers are still drawn as an image.

### Changed
- Vector layers are simplified for drawing over threads (GEOS lets go of the GIL): 4 times faster on
  200 000 polygons.

## [1.31.1] - 2026-10-08

### Changed
- `ls.plot(..., static=True)` draws a vector layer as one collection per kind of geometry, made from the
  coordinates of all its features at once, instead of geopandas's patch per feature: 50 000 polygons in
  0.76 s instead of 3.7 s. More than 100 000 features (or 2 million vertices) are burnt into an image at the
  figure's resolution (C++ rasterisation), as on interactive maps. Holes are drawn whatever the orientation
  of the rings.
- Vector layers are simplified for drawing without keeping their topology (6 to 20 times faster; a feature
  that would vanish is simplified keeping it, so small polygons are still there when zoomed in): 10 000
  polygons on a map in 0.15 s instead of 0.24 s.
- The bands of a stack on a map share 16 million pixels (each at least 512 on its long side), instead of a
  fixed size per number of bands: a page of 47 bands of 1000 x 1000 is 14 MB instead of 54 MB, lighter in a
  notebook.

## [1.31.0] - 2026-10-07

### Added
- `ls.prioritize.restoration_problem(restorable, limit=, cell_area=)`: plangea's restoration problem
  (Strassburg et al. 2020), how much of every converted cell to restore. Benefits: the reduction of the
  species' extinction risk 1 - (A / P)^z with the habitat restored (`add_extinction_risk_benefit`, from
  `ls.sdm.area_of_habitat`; z one number for all species, default plangea's 0.25, or one per species;
  with a `restoration_discount`), the risk of ecosystem collapse
  (`ls.prioritize.ecosystem_areas`), linear benefits such as carbon and ecological intactness by its
  gradient; area targets in the landscape or per region, a budget, or the cheapest plan for a benefit.
  The risk curves are concave and the problem is solved to a proven tolerance (simplicial decomposition
  with the Frank-Wolfe bound; 9 x 10⁴ cells and 500 species in 0.6 s), not by plangea's heuristic of
  benefit per cost and steps of the target, which is `method="steps"`, the same plans as plangea's loop
  (to 1e-15). `RestorationProblem.uniform()` is plangea's uniform scenario.
- `ls.prioritize.eval_risk_reduction(problem, solution, z_sd=)`: the reduction of the extinction risk of
  a plan, per species and in total (plangea's `multi_exrisk_aggregate`, with its square root after the
  loop over species), and its standard deviation from the uncertainty of z (one deviation or one per
  species): by default the species' z err together (|sum d reduction / dz sd|), or plangea's two forms
  (`"independent"`, `"linear"`).
### Fixed
- pandas 3: R data frames read from `.rda` / `.rds` files (example data) keep their text columns as objects
  with `None` for NA (pandas 3 inferred its string dtype, NA as NaN), and
  `DevelopmentDemand.from_population` no longer takes evenly spaced times in the index (to which pandas 3
  gives a `RangeIndex`) for a default index ("no observed population for subregion ..."). The same for a
  graph's per-node values given as a Series indexed by node ids (`HabitatGraph`) and for the tables batch
  runs write (their index kept): only pandas's default index counts as no index.
- Tests on Linux: the two simulate test inputs drawn with numpy's `sin`, `cos` and `exp` (whose last bits
  differ between platforms) are stored as drawn on Windows, where the references were made; MOLUSCE's
  correlation, which went through numpy's dgemm on Windows, is compared within a few units in the last
  place elsewhere (`conftest.WINDOWS_BLAS`).

## [1.30.0] - 2026-10-07

### Added
- `ls.original_vegetation(land_cover, natural, anthropic, past=, ecoregions=, failsafe=)`: the natural
  vegetation of converted land (the anthropic share of each pixel given back to the natural classes in
  their past proportions, else those of the pixel's ecoregion, else the classes' mean ones) and the
  vegetation restoration brings back, as plangea prepares them for its restoration model (Strassburg et
  al. 2020); bit for bit plangea's `harmonize_oa`, with equal ecoregion shares when there are no
  ecoregion maps (plangea divides the pixel's natural share instead) and no stop on NA in the failsafe.
- `ls.sdm.area_of_habitat(ranges, land_cover, natural, suitability=, original=, elevation=, ...)`: the
  area of habitat of species (Brooks et al. 2019) from ranges, habitat classes and elevation limits,
  current and potential, and the habitat each species gains per unit of a cell restored (a sparse
  matrix, `gain_map()`); bit for bit plangea's `harmonize_bd`. `ls.sdm.cell_area_km2`.
- `ls.intactness(hfi, radius)` and `ls.intactness_gradient(hfi, reduction, radius)`: ecological
  intactness from the human footprint (Beyer et al. 2020), after restoration too, and the complete
  derivative of the total intactness with respect to restoring each cell (C++, OpenMP); bit for bit
  plangea's `multi_calc_it`, with the window clipped at the raster's edges (plangea skips the cells near
  the edges and gives them the mean).
- `ls.uncertainty`: the standard deviation of totals of uncertain values (quadratic, linear, with a
  correlation), of products (Goodman's exact variance), the delta method (central or forward
  differences, a covariance matrix), Monte Carlo with a fixed seed and normal confidence intervals.

### Changed
- Maximum features, phylogenetic diversity and phylogenetic endemism on HiGHS: after the wide engine, a
  local search over the set of targets met (additions and swaps, the most valuable first, each priced
  exactly by the covering engine). Synthetic problems (30 features, targets of 30 %, a budget of 10 %):
  phylogenetic diversity at 10⁵ units 47.9 -> 53.7 (Gurobi in 300 s: 46.7), at 10⁶ 42.4 -> 52.2 in 60 s;
  phylogenetic endemism at 10⁵ 841 (Gurobi in 30 s: 179).
- Maximum features: the bound is rounded down to a whole number of targets (the cost tie-break adds
  nothing to a maximisation), which proves gaps the linear relaxation alone does not (10⁴ units, targets
  of 20 %: a 7.7 % gap proven in 0.5 s instead of 10.5 % after 120 s).
- Multi-action problems with boundary penalties: one product column per pair of neighbours instead of one
  per direction (the two were a symmetry of the MIP): 38 % fewer columns at 10³ units.
- The cut engine (minimum sets with boundary penalties or connectivity rewards) repairs its selections on
  `n_threads` threads while the bundle goes on, its decisions a fixed number of evaluations behind (the
  same result on any number of threads): 10⁶ units, gap 1 %, 17.6 s -> 7.5 s (90 % of the time was the
  repairs, 7 % the maximum flows, so dynamic flows would not pay).

### Fixed
- The wide engine's sifting cycled on degenerate LPs (phylogenetic objectives at 10⁶ units: the units let
  in at one round were dropped at the next, and the LP never converged within the time limit): the working
  set now grows while more units price in than a round admits (5 rounds, 8 s).
- HiGHS's MIP kept running up to 20 s past the time limit on wide models (10⁵ columns, 10⁶ non-zeros):
  two changes to its sources (applied at build time by `src/cpp/opt/highs_patch.cmake`) make the central
  rounding's line search and the analytic centre keep the limits.
- `Problem.add_manual_bounded_constraints` and `add_manual_locked_constraints` (prioritizr 8.0.6's, and
  prioripy's `add_continuous_bounds`): bounds or a fixed share for given planning units, from prioritizr's
  table (`pu` among every unit, `lower` / `upper` or `status`) or as values (a number, a raster on the
  units' grid, a column or one value per unit). The compiled model is prioritizr's bit for bit; bounds
  outside the decisions' bounds, or shares with binary decisions, are an error.
- `ls.prioritize.eval_landscape_identification` and `eval_landscape_comparison` (prioripy's, after
  Zonation's landscape identification and comparison) over any priority surface: a ranking, an importance,
  a selection frequency or a solution. The separate areas of a top fraction (minimum size, merging of
  nearby areas, 8 or 4 neighbours) with their cells, mean value, cost and the features they hold, and the
  overlap of two surfaces' top fractions (Jaccard index, comparison maps). The top fraction follows
  Zonation's rule (`position_rank`), so on a Zonation ranking the areas are Zonation's own, bit for bit.
  Both results plot with `ls.plot`.
- `ls.prioritize.eval_proxirr(p, triage)`: ProxIRR (Baisero et al. 2022), alpha per target and unit,
  beta and gamma per unit, for every `>=` target of a problem; `eval_proxirr_marxan(pu, spec, puvspr)`
  for Marxan's tables. The same bits as the CRAN package proxirr 0.5 (its 80-bit `prod()` included), in
  C++; proxirr's Marxan functions look features up by position instead of id, and stop on a feature
  missing from the spec: here by id, the feature left out.

## [1.29.0] - 2026-10-07

### Fixed
- `add_contiguity_constraints` and `add_feature_contiguity_constraints` now use an exact model (a root, a
  spanning arborescence on both directions of every edge and a single-commodity flow). prioritizr's
  formulation, which the previous versions compiled, orients every edge from a unit to one of a lower
  index: a restriction that misses patches (three units in a line with the middle one first have none)
  and can return a costlier patch than the best one or call a feasible problem infeasible. `compile()`
  still matches prioritizr row by row in every other case.

### Changed
- Contiguity and corridor constraints on HiGHS are solved in three steps: a first patch from prioritizr's
  restriction, a bound from directed cuts (node cuts for corridors) separated by maximum flows, then the
  exact MIP from the first patch, within the time limit.

## [1.28.0] - 2026-10-07

### Added
- Migration guide "From prioritizr, prioriactions and prioripy" in the API reference: the equivalent of
  every function, a script side by side in R and Python, what to expect when comparing results
  (prioritizr 9's rescaled boundary data, dropped tie-break terms and renamed objectives) and what is not
  available.
- `Problem.add_max_wtd_sum_objective`, `add_max_n_targets_met_objective` and `add_neighbor_constraints`:
  prioritizr's newer names for `add_max_utility_objective`, `add_max_features_objective` and
  `add_neighbors_constraints` (the same models).

## [1.27.0] - 2026-10-05

### Added
- `ls.prioritize.example_data(name)`: **Serra Verde**, an imaginary coastal region of 60 x 90 cells of 1 km
  (4825 planning units) with land cost, eight features from widespread to rare, protected areas, cities,
  habitat condition, regions, carbon, corridor terminals, deforestation and fire threats, elevation, rivers
  and roads. Every public function of the optimization side has a runnable example on it.
- Learn: "Conservation priorities" is now a part of ten chapters (12.1 to 12.10): why prioritize, key
  concepts, ranking with Zonation, optimization and solvers, spatial design, trade-offs and alternatives,
  evaluation, management actions, ranking and optimization together, and the planning process (good
  practices, mistakes, recipes, a prioritizr / prioriactions / Marxan / Zonation correspondence), with
  figures of Serra Verde and runnable code throughout. 33 new glossary terms.
- Tutorial "Optimized conservation plans": a complete plan for Serra Verde, from the gap analysis to the
  report.

### Changed
- Minimisations with a budget row and pair terms (minimum penalties with boundary or connectivity
  penalties) go to the closure engine before HiGHS's MIP: 2400 units, more than 150 s -> 12 s at a 10 % gap.
- `pareto_front` without `budgets` spaces them above the cost of the locked-in units (which every budget
  includes) instead of from a tenth of the plan's cost, which made problems with reserves infeasible.

## [1.26.0] - 2026-10-04

### Added
- `ls.prioritize.coarsen(cost, features, factor)` and `refine(selection, like)`: solve a continental
  problem on blocks of cells and bring the selection back to the fine grid.

### Changed
- The covering engine fixes the units dearer than a greedy cover of what is left to cover (exact), and
  the replacement importance re-solves on arrays prepared once: 10^5 units with 12,929 selected in 143 s
  (was 192 s). A min set problem of 10^7 cells solves to optimality in 60 s.

## [1.25.0] - 2026-10-04

### Added
- `ls.prioritize.zonation_features(features, condition=..., connectivity=..., retention=..., impact=...)`:
  Zonation 5's feature transforms (condition, connectivity kernels, interactions, matrix connectivity,
  retention, impacts) without the ranking, back in the features' units, for an optimization problem.
- `add_min_extinction_risk_objective(budget, z=0.25, segments=16)`: the least species-area extinction
  risk, 1 - (held / total)^z, within a budget (a piecewise-linear objective), and
  `eval_extinction_risk(problem, solution, z)`.
- `compare_zonation(problem, solution)`: a Zonation ranking cut at the optimization solution's cost,
  feature by feature and on maps (`ZonationComparison`).
- `sdm_features(results, env, threshold=None)`: features from `ls.sdm` predictions.
- `report(problem, solution)`: the solution, its summaries, representation, targets, irreplaceability
  and (optionally) extinction risk and the Zonation comparison in one object; `ls.plot(report)` draws
  it and `.save("report.html")` writes a page that works offline. `ls.plot` also draws solutions,
  portfolios, trade-offs, multi-action solutions and portfolios and Zonation comparisons.
- `eval_boundary_summary` measures polygon planning units without a boundary matrix.

## [1.24.0] - 2026-10-04

### Added
- Calibrating trade-offs (prioritizr's tutorial): `calibrate_penalty` (blended approach; by default
  penalties around Cohon's), `calibrate_budget` (hierarchical approach: the least penalty within cost
  thresholds), `cohon_penalty` (Cohon et al. 1979), `pareto_front` (cost vs the targets' shortfall) and
  `calibrate(make, values, criterion)` for any parameter (feature weights...), solved in parallel; a
  `TradeOffs` table marks the candidates `topsis` (Hwang & Yoon) and `knee_point` choose. The optima
  are prioritizr's on Gurobi. `iterative_solve(problem, update, steps)` re-solves as the problem
  changes (e.g. between `ls.simulate` steps; `protected_cells(solution)` gives the simulation's
  restrictions), keeping earlier selections with `lock_previous=True`.

## [1.23.0] - 2026-10-04

### Added
- Evaluation of solutions (prioritizr 8.0.6's `eval_*`), bit for bit with prioritizr: `eval_n_summary`,
  `eval_cost_summary`, `eval_boundary_summary`, `eval_connectivity_summary`,
  `eval_asym_connectivity_summary`, `eval_feature_representation_summary`,
  `eval_target_coverage_summary`, `eval_ferrier_importance` (with R's `pnorm` in C++) and
  `eval_rare_richness_importance`; `eval_replacement_importance` (re-solves in parallel: 1,300 selected
  units of 10^4 in 2.4 s) and `eval_rank_importance` (the same optima as prioritizr's on Gurobi). Also
  `eval_landscape_metrics` and `eval_mspa` of a raster solution (landscape metrics and MSPA of the
  selected cells). They take a `Solution`, its selection or plain values.
- `ls.prioritize.problem(cost, ls.load_raster([...]))`: the features as a `RasterCollection`, read
  block by block.

## [1.22.0] - 2026-10-04

### Added
- Multi-action planning solves on an own closure engine behind `add_highs_solver()`: a Lagrangian
  bound as strong as the linear relaxation (each action needs its unit: closed form per unit, or a
  minimum cut with boundary penalties), repaired selections, and core MIPs on HiGHS over the
  Lagrangian's near ties. At 10^5 units the 1% gap takes 0.7 s (HiGHS alone 57 s, Gurobi 3 s) and the
  proven optimum 6 s (Gurobi 13 s). `ActionsProblem.solve(start=...)` takes a solution to start from;
  `add_highs_solver(engine=False)` runs HiGHS's MIP alone.
- `ls.prioritize.actions_raster_data(monitoring_cost, features, threats, action_cost, ...)`: multi-action
  problems from rasters (the cells with a monitoring cost are the planning units; one layer per feature
  and per threat; action costs as numbers or rasters per threat; relative or absolute recovery and
  conservation targets; locked cells; the grid's boundary), and `ActionsSolution.maps()` with the
  monitored cells and every threat's actions as rasters (also written by `ls.save`).
- `ActionsProblem.add_directional_dependencies(flow, penalty, hard=False)`: upstream-downstream
  dependencies between planning units (after prioripy): monitoring a target unit without its source
  costs `strength x penalty`, or is not allowed with `hard=True`; on the closure engine as pair terms or
  requirements (10^5 units with a river-like flow at 1% gap in 1 s soft, 5.5 s hard; Gurobi 6 s and
  8 s). Unlike prioripy, the penalty is not counted in the budget of `maximizeBenefits`.
- Multi-action portfolios: `ActionsProblem.add_cuts_portfolio`, `add_top_portfolio`,
  `add_gap_portfolio` and `add_shuffle_portfolio` (no-good cuts on the units and the actions) give an
  `ActionsPortfolio` with every solution, a summary and how often each unit is monitored and each action
  taken (as rasters for raster problems).

### Changed
- The multi-action model, its figures and the equivalent model for solving are built in C++:
  `problem()` and `compile()` at 10^5 units from 4.5 s to 0.15 s.

## [1.21.0] - 2026-10-04

### Added
- MILP prioritization in `ls.prioritize` (prioritizr 8.0.6 and prioriactions 0.5.0), with HiGHS 1.15.1
  built into the extension and Gurobi optional:
  - `ls.prioritize.problem(cost, features)` on raster cells, polygons, tables or H3 cells
    (`h3_units`; polygon amounts by an exactextract port, bit for bit with exactextractr) and the
    prioritizr `add_*` grammar. Objectives: min set, min shortfall, min largest shortfall, max
    utility, max features, max cover, min penalties, max phylogenetic diversity and endemism. Targets:
    relative, absolute, manual, loglinear and sub-regional, plus feature weights. Penalties: boundary,
    connectivity, asymmetric connectivity and linear. Constraints: linear, contiguity, feature
    contiguity, neighbours and corridors. Locked units, binary, proportion and semi-continuous
    decisions, and cuts, top, gap and shuffle portfolios. `compile()` gives prioritizr's model bit
    for bit; `solve(start=...)` accepts a Zonation ranking.
  - Own engines behind the HiGHS solver: a sifted-LP covering engine (min set at 10^6 units), a
    Lagrangian wide engine (budgeted objectives), and a cut engine (boundary and connectivity
    penalties: Lagrangian bound by minimum cuts with an own Boykov-Kolmogorov max-flow; 10^6 units at
    a proven 0.8% gap in 27 s).
  - Matrix functions `boundary_matrix`, `adjacency_matrix`, `connectivity_matrix`,
    `proximity_matrix`, `decay_connectivity` and `unit_points`.
  - `ls.prioritize.marxan(...)`: a Marxan 4.0.6 port with the same runs unit by unit as the
    executable; `add_anneal_solver()` runs it on min-set problems with target repair.
  - Multi-action planning (prioriactions): `actions_data(...)`, `.problem(model, budget, blm, curve,
    segments)`, `compile()` bit for bit, `solve()` reaching prioriactions's optimum, and the
    solution's cost, benefits, connectivity penalty and actions as prioriactions computes them.

### Fixed
- Bugs found in the references are implemented only in their correct form (listed in IDEAS.md):
  prioritizr's phylogenetic tip order and forbid-solution cut, Marxan's uninitialised amounts,
  heuristic loop and all-locked hang, and prioriactions's status reset, conservation benefit and
  budget check.

## [1.20.0] - 2026-10-04

### Added
- `ls.simulate` phase 10, documentation: the course chapter "Simulating land change" (demand,
  transition potential, allocation, validation against persistence, calibration by patterns,
  scenarios), the tutorial "Land change simulation" (Sibuyan Island from the change analysis to
  scenarios with replicates), the land change tables of the benchmarks and accuracy page, and
  `tools/benchmark_simulate.py` (against lulcc, MOLUSCE and Dinamica EGO).

### Fixed
- Docstrings of `ls.simulate` that the documentation build could not parse (notes after the
  parameters).

## [1.19.0] - 2026-10-04

### Added
- `ls.simulate` phase 9, pattern-oriented calibration: `calibrate(model or build function, start,
  observed, params=..., targets=..., method=...)` searches the settings (paths into the model's
  pieces, or the arguments of a function building it) whose simulations best reproduce the observed
  map by figure of merit, fuzzy similarity, kappa, quantity or allocation disagreement, landscape
  metrics or any loss function; Nelder-Mead, differential evolution, grid or Latin hypercube, each
  evaluation the mean of replicates with common seeds; `Calibration` with the history, the best
  model and simulation, `validate()` and a plot.

## [1.18.0] - 2026-10-04

### Added
- `ls.simulate` phase 8b: `fuzzy_similarity` (Hagen 2003, as Dinamica EGO's
  `CalcReciprocalSimilarityMap`: exponential or constant decay in a square window, two-way where both
  maps have a class), `differences` and `similarity_of_differences` (Dinamica's validation of the
  observed against the simulated change, per window), the same maps and means as Dinamica EGO 8;
  `futures_validation_grid` (per tile or region); `validate` gathering every comparison (three
  maps at several resolutions, fuzzy similarity, kappas, error budget, disagreement, landscape
  metrics side by side) into a `Validation` with `summary()` and a plot; `Projection.validate()`.

## [1.17.0] - 2026-10-04

### Added
- `ls.simulate` phase 8a, validation: `three_map_comparison` (the reference maps at the start
  and the end against the simulated end at several resolutions, Pontius et al. 2011) with its
  table, `agreement_budget()` (misses, hits, wrong hits, false alarms, correct rejections; overall,
  per class, per transition) and `figure_of_merit()`, bit for bit with lulcc (the per-block
  equations in C++); MOLUSCE's `compare_maps` (kappa, kappa location and histogram, correctness,
  Cramer's V, joint information uncertainty, correlation) and `error_budget`, bit for bit; FUTURES'
  `futures_validation` (quantity and allocation disagreement, kappa, kappa simulation, hits, misses,
  false alarms, figure of merit), bit for bit with r.futures.validation.

## [1.16.0] - 2026-10-04

### Added
- `ls.simulate` phase 7c, scenarios and replicates: `scenarios(model or projection, ...)` runs every
  scenario (a model, the pieces to replace, or a function of the model) several times in parallel,
  replicate `r` with seed `seed + r` in every scenario (common random numbers), the same maps with
  any number of workers; `Ensemble` with `probability(class, year)`, `change_probability`, `cover`,
  `metrics` (landscape metrics by name, or any function of a map) and `to_dataset`.
- `ls.plot` of land change simulations: the maps with a slider over the steps or years, the cells
  of each class, the flows from the first to the last map (`Simulation`, `Projection` with its
  check), the probability of change and the mean cover per scenario (`Ensemble`).

## [1.15.0] - 2026-10-04

### Added
- `ls.simulate` phase 7b, presets and one-call projections:
  - `MolusceWoE` and `MolusceCA`: MOLUSCE's weights of evidence (bins, persistence codes) and its
    cellular automaton simulator, bit for bit with MOLUSCE 5.3.0 patched (`tools/molusce_fixed.patch`:
    the confidence from the two largest weights; 255 for cells without any weight, which numpy 2
    refuses as -1);
  - `presets`: `clue_s`, `ordered`, `clumondo`, `futures` (the pieces assembled) and `dinamica`,
    `plus`, `molusce` (everything fitted from two maps; `observed_patches` measures the Expander's
    share and the patch sizes of each transition);
  - `run(t0, t1, drivers, until=..., preset=...)`: fit with a step per year, simulate the observed
    period and compare it with the later map (`agreement`: overall agreement, figure of merit,
    quantity per class), project; `Projection` with `maps` by year, `report`, `settings`, `redo`,
    `to_dataset`;
  - `Estimator(stratify=True)`: the sample drawn from presences and absences separately.

## [1.14.0] - 2026-10-04

### Added
- `ls.simulate` phase 7a, dynamic variables and regions:
  - `Model(..., dynamic={name: piece})`: drivers computed again from the map at the start of every
    step, and the potential predicted again (CLUE, Ordered, ExpanderPatcher, CARS, CLUMondo);
    `distance_to(classes)` (exact Euclidean distance, or the accumulated cost over an `ls.cost`
    transition: distances along networks) and `share_of(classes, window)`, also usable on their own;
  - `Regional(regions, {region: demand})`: a demand per region for CLUE, Ordered, CARS (demand and
    adaptive coefficient per region and class, in C++) and ExpanderPatcher (each region's rates);
    `Regional.from_maps` and `Regional.markov`.

## [1.13.0] - 2026-10-04

### Added
- `ls.simulate` phase 6, PLUS: `CARS`, the cellular automaton of multi-type random patch seeds
  (classes compete through growth probability, neighbourhood share and an adaptive coefficient that
  follows the gap to the demand; random seeds start new patches; a decreasing threshold decides the
  changes), in C++, written from Liang et al. (2021) (PLUS has no source code); demand from an
  `AreaDemand` or a projected `Markov` chain, 0/1 conversion matrix and restricted cells; the same
  maps with any number of threads. `Estimator.importance()`: the contribution of each driver to each
  class's model (LEAS's driver contributions), by impurity or permutation.

### Changed
- The xoshiro256** generator of the clean-room allocations moved to `src/cpp/sim/rng.hpp`
  (`ExpanderPatcher` gives the same maps).

## [1.12.0] - 2026-10-03

### Added
- `ls.simulate` phase 5c, Dinamica's transition functions: `ExpanderPatcher` (the Expander grows
  patches next to existing ones of the target class, the Patcher seeds new ones; lognormal patch
  sizes, isometry, prune factor, a share of each transition per function), in C++; written from the
  publications and Dinamica EGO 8's measured seed selection (Expander acceptance `P n_j / 5`, seeds
  drawn by probability among the pruned cells), statistically checked against it with one-cell
  patches (its functors crash on larger ones).

## [1.11.0] - 2026-10-03

### Added
- `ls.simulate` phase 5b, FUTURES' climate submodel and tools:
  - `Flooding` (`PatchGrowing(climate=...)`): floods drawn per catchment from depth maps or from
    flood probability and HAND, depth-damage curves per region, a Gaussian response from adaptive
    capacity (retreat, adapt, stay), abandonment, migration of demand between subregions through a
    redistribution matrix and population densities; `Simulation.adaptation`, `floods`,
    `redistribution`; bit for bit with r.futures.simulation;
  - `patch_library` and `calibrate_patches` (r.futures.calib), bit for bit;
  - `DevelopmentPotential.fit` (r.futures.potential): R's `glmer` with a random intercept or slope
    per subregion (Laplace; within lme4's convergence tolerance), MuMIn's dredge by AIC, R's `glm`
    with one subregion (bit for bit); `DevelopmentPotential.surface` (r.futures.potsurface).
- FUTURES defects corrected (tested against the patched module): the flood level samples the whole
  catchment (the module reads only the first column of its bounding box); only simulated cells set a
  catchment's flood probability.

## [1.10.0] - 2026-10-03

### Added
- `ls.simulate` phase 5a, urban growth by patches (FUTURES, GRASS GIS's r.futures):
  - `PatchGrowing`: r.futures.simulation's allocation in C++ (seeds drawn on the development
    probability, patch sizes from a library per subregion, compact growth, development pressure
    updated around new cells, overflow between subregions and steps, incentive, potential weights,
    zoning with stringency, potential subregions); the same maps as GRASS bit for bit with the
    same seed; 480,000 cells over 5 steps in 0.20 s (GRASS 0.35 s);
  - `development_pressure` (r.futures.devpressure), `DevelopmentDemand` with `from_population`
    (r.futures.demand's regressions), `DevelopmentPotential`, `Simulation.development`.
- FUTURES defect corrected (tested against a patched module): every cell of a patch that crosses
  into another subregion adds its development pressure.

## [1.9.0] - 2026-10-03

### Added
- `ls.simulate` phase 4, land systems (CLUMondo):
  - `CLUMondo`: the demand-driven allocation of land systems (goods and services per cell, conversion
    order, demand weights, deviations in % or absolute, resistance, CLUMondo's conversion codes with
    maps, minimum and maximum ages and autonomous succession, restricted areas, random or given
    ages, neighbourhood effects, location preferences, drivers and supply per step), in C++; the
    same maps as CLUMondo bit for bit (6 cases), 400 x 500 cells over 8 years in 1.7 s (CLUMondo
    16.2 s);
  - `ServiceDemand`, `Supply`, `NeighbourhoodEffect`, `LogisticRegression.from_coefficients`;
  - from the CLUMondo executable of 2017 (manual): minimum and maximum demands, a deviation per
    demand, patches (97-100% of cells in agreement with it);
  - `LandSystemStack`: land systems as overlapping binary layers (bit masks; overlap policies,
    priority, rare combinations merged, new mosaics), supply and suitability from the layers;
    `Roles` (expansion, regeneration, intensification, de-intensification), rules by role
    (forbidden roles, transition resistance, conversion order), `Cap` (most cells or area a year),
    `Succession`;
  - `Simulation.layers`, `cover`, `supply`, `shortfall`, `trajectory`, `outputs`, `ages`.
- `Rules(elasticity=...)` accepts one value for all classes.
- CLUMondo defect corrected (tested against a patched CLUMondo): restricted cells keep their age
  (CLUMondo sets it to -9999 after the first year, which stops autonomous changes there).

## [1.8.0] - 2026-10-03

### Added
- `ls.simulate` phase 3, allocation (lulcc):
  - `Model(potential, demand, allocation, rules).run(landscape, drivers, seed=)`: the simulation loop of
    lulcc's `allocate` (elasticities, neighbourhood and conversion rules, automatic conversions, history),
    giving a `Simulation` (the map of every step, a table per step, the seed, the history map);
  - `AreaDemand`: cells per class and step, from a table or interpolated/extrapolated from observed
    maps (`approxExtrapDemand`);
  - `Rules` (conversion matrix with lulcc's codes, elasticities, restricted areas, history) and
    `Neighbourhood` (focal thresholds, in C++);
  - `CLUE` (CLUE-S's iterative allocation, in C++) and `Ordered` (the ordered procedure, stochastic or
    not). The same maps as lulcc bit for bit with the same `set.seed`; Plum Island's 14 steps in
    4.1 s (lulcc 10.0 s) and 0.38 s (3.4 s).
- lulcc defects corrected (tested against a corrected lulcc): the CLUE-S mean criterion averages
  the absolute differences (lulcc's signed ones always sum to 0); neighbourhood rules work on more
  than one class and through the models (lulcc stops with an error); cells released in the ordered
  allocation that no class takes keep their class (lulcc leaves -1).

## [1.7.0] - 2026-10-03

### Added
- `ls.simulate` phase 2, transition potential:
  - `WeightsOfEvidence` / `weights_of_evidence`: Dinamica EGO's weights of evidence: ranges of
    continuous variables by its line generalisation of nested buffers (increment, minimum and
    maximum delta, tolerance angle), categorical variables, W+, contrast and significance, the fix
    of abnormal weights, both "potential transition" criteria, per-transition settings, the
    correlation report (chi-square, Cramer's V, contingency, joint entropy and uncertainty) and the
    probability maps (multinomial over the transitions of a class). The same ranges, counts and
    weights as Dinamica EGO 8 (its 15 printed digits); counting and probabilities in C++.
  - `LogisticRegression`: lulcc's `glmModels`, R's `glm(family = binomial)` reproduced operation for
    operation (coefficients, deviance, predictions bit for bit, aliased drivers as NaN).
  - `Estimator`: any scikit-learn classifier per class, trained on presence or on expansion between
    two maps (PLUS's LEAS), with sampling and balancing.
  - `MultiCriteria`: MOLUSCE's multi-criteria evaluation (Saaty's AHP weights and consistency ratio).
  - `RasterPotential`: a potential given as maps.
- Variables with negative values: Dinamica EGO drops values just below zero and shifts its ranges
  there by one increment; here they are treated like positive ones.

## [1.6.0] - 2026-10-03

### Added
- `ls.simulate` phase 1, land change analysis:
  - `changes()`: the transitions of a series of maps (OpenLand's `contingencyTable`), the class
    areas (lulcc's `total`), the intensity analysis of Aldwaik & Pontius at the interval, category
    and transition levels with the stationarity tests (OpenLand's `intensityAnalysis`), the number
    of changes per cell (`acc_changes`), and links to the components and the Markov chain;
  - `crosstab()` (lulcc's `crossTabulate`, diffeR's `crosstabm`), in C++: 41.7 million cells in
    0.03 s;
  - the components of difference of diffeR: `difference_components` (`diffTablej`),
    `overall_difference`, `difference_resolutions` (`differenceMR`), `composite_matrix`,
    `memberships`, `mad` (`MAD`), `sample_to_population` (`sample2pop`);
  - `Markov`: transition probabilities of a period and per time step (Dinamica EGO's
    DetermineTransitionMatrix), the principal root by `expm(logm(P) / n)`, projections of areas.
  The same numbers as the R packages, bit for bit; Dinamica within its 15 printed digits. On the
  Sao Lourenco basin (5 maps of 41.7 million cells): the transition tables in 0.16 s (OpenLand's
  `contingencyTable`: 120 s), the change counts in 0.25 s (`acc_changes`: 24 s).
- Corrected where the references are wrong (tested against corrected versions): OpenLand drops the
  transitions into a class found only in the last map from the intensity analysis; diffeR's
  `composite` counts a missing value as a category, so `differenceMR` gives NaN at every coarser
  resolution of a map with missing cells; `sample2pop` takes the population of the row's position
  instead of its category; Dinamica gives a class absent at the start a row of zeros (its
  multi-step rates come out wrong) and gives up on matrices with repeated or complex eigenvalues.

## [1.5.0] - 2026-10-03

### Added
- `ls.simulate`, the land use and land cover change module (plan and phases in IDEAS.md), starts with
  its reference harness and example data. `ls.simulate.example_data()`: Plum Island Ecosystems and
  Sibuyan Island from lulcc (land use, drivers, restrictions, demand scenarios) and the Sao Lourenco
  River Basin from OpenLand (five land use maps of 6372 x 6546 cells, downloaded and converted on
  first use). `tools/make_reference_simulate.py` runs the cases of `tests/simulate_cases.py` on R
  (lulcc, OpenLand, diffeR), CLUMondo, GRASS's r.futures (in WSL), Dinamica EGO's console and
  MOLUSCE's algorithms. The generators of CLUMondo (the Microsoft C runtime's `rand`) and r.futures
  (GRASS's `G_drand48`) are in C++, the same values as the originals.

### Fixed
- Reading R data files (`.rda`, `.rds`) with compiled functions (bytecode), such as the raster
  package's objects, failed with "unsupported R object type 21".

## [1.4.3] - 2026-10-03

### Fixed
- `ls.plot` maps in VS Code notebooks: the base map could stay invisible, its tiles loaded but
  transparent, as Leaflet fades tiles in on animation frames that a notebook's output frame does
  not always get. Tiles now show as soon as they load, without the fade.

## [1.4.2] - 2026-10-03

### Fixed
- `ls.plot` maps in notebooks: a map whose output loaded while hidden (JupyterLab opening a
  notebook with saved outputs, an output not yet on screen) stayed on the whole world at zoom 0,
  its data a speck and often no base map. The view is now fitted again once the map has a size.

## [1.4.1] - 2026-10-02

### Changed
- `ls.plot` is faster on stacks and makes smaller pages. The bands of a stack are rendered a band
  per thread (47 bands of 1000 x 1000: 3.9 s -> 1.1 s); the images and data go into the page in
  one pass (a replace per image went over the page as it grew); PNGs are compressed at level 2
  (about 1 ms more per image, pages about 30 % smaller: a class map of 10 000 x 10 000,
  2.5 -> 1.6 MB; 2 million points, 7.8 -> 4.8 MB). Measured and left as they are: lossless WebP
  (40 % smaller than PNG but 10 to 30 times slower to encode) and landschaft's own warp kernel for
  the reprojection to Web Mercator (no faster than GDAL's on images of the screen's size).

## [1.4.0] - 2026-10-02

### Added
- `ls.plot` of a stack: one entry in the layer control and, on its legend card, a slider choosing
  the band shown and a button playing through them (a time series), instead of a layer per band.
- `ls.plot` of a species distribution model (`ls.sdm.fit`): the response curves of every
  predictor and model, the cross-validated scores with their spread, the thresholds; with
  `env=` the predictor rasters, the suitability of each model, the ensemble and its spread
  (`ls.sdm.predict`) with the presences.
- `ls.plot` of a clustergram (`SpatialSignatureAnalysis.clustergram`), interactive: the clusters
  of each k at their means, sized by their rows, and the rows moving between them.
- `ls.plot` of a least-cost path: the path outlined over the combined cost surface (it stays
  visible however far the map zooms out) and its numbers.

### Changed
- `ls.plot` of a habitat graph with `distance=`: the links on a colour ramp of their distance (the
  shortest bright), with a legend, instead of all in white.
- The tests draw static figures with matplotlib's Agg backend (`MPLBACKEND`, in `conftest.py`).

## [1.3.0] - 2026-10-02

### Added
- `ls.plot(..., static=True)` for publications: a scale bar of a round ground length (corrected for
  latitude on Web Mercator and degrees), a north arrow, longitudes and latitudes on a single map;
  small multiples for a stack's bands (one colour scale and colour bar, `bands=` respected), a
  result's rasters and two dates (`compare=True`); one legend for maps of the same colours;
  options `figsize=`, `ncols=`, `scalebar=False`, `north_arrow=False`.

### Fixed
- Static figures drew only the first band of a stack and the first raster of a result; their
  vector layers ignored `cmap` dictionaries, `labels`, `categorical`, `vmin` / `vmax` / `log`
  (they now take the interactive map's colours and legend); a GeoDataFrame was drawn as a table;
  tables without numbers no longer fail.

## [1.2.0] - 2026-10-02

### Added
- `ls.plot([before, after], compare=True)`: two rasters or vector layers on one map, either side of
  a line dragged across it (each side in a Leaflet pane of its own, clipped). Layers of the same
  colours share one legend card.
- `ls.plot` of a change between two dates: `simple_change` (the flows between background and
  foreground as a Sankey diagram, gain in green, loss in red), `fos_change` (the flows between the
  fragmentation classes of the common foreground), `fad_multiscale_change` (the flows between the
  FAD classes at one scale, `scale=`, default the multiscale map, and the averages by scale) and
  `morphological_change` (the counts of each kind of change), with their maps and numbers.

### Fixed
- `ls.plot`: the legend of a FAD or FOS density map names its classes (`Rare (0-9%)` to
  `Intact (100%)`, `Background`); the class of each density was named by the code of another
  (`Patchy` for 2%). The `classes` map of `fad` / `fad_multiscale` is drawn in the colours of the
  density map (its `landschaft_product` attribute is `"FAD_CLASSES"`), and the change maps name
  their classes (`Gain`, `Loss`, ΔFOS ranges...).

## [1.1.0] - 2026-10-02

### Added
- `ls.plot` maps read their values: under the cursor (with the coordinates, bottom left) and in a
  popup on a click, every raster visible gives its class name or its value (to 1/254 of its
  ramp). The values come from the page itself: each image's palette holds one colour per index,
  read back from the pixel in the browser, so no second image is needed.
- Every layer of a map has a legend card with an opacity slider; a card hides while its layer is
  switched off. Vector layers drawn in one colour and true-colour composites have cards too.

### Changed
- `ls.plot`: a layer's opacity is applied by the map, the images' palettes are opaque (static
  figures apply it when drawing). The base map's credits keep to one line (all of them on hover),
  so they no longer hide the scale bar and the legends.

## [1.0.2] - 2026-10-02

### Fixed
- `ls.plot`: an HTML page opens without a network connection. Leaflet and plotly are in the page
  instead of loaded from CDNs (folium's unused jQuery, Bootstrap and Font Awesome are left out);
  a window loads plotly from the installed package. The full-screen button and the coordinates
  under the cursor are Landschaft's own controls.
- `ls.plot`: NoData of a true-colour composite is transparent and left out of its stretch (an
  integer NoData such as 0 was drawn black and darkened the image); a lazy raster's NoData is left
  out of its block means.
- `ls.plot(..., bands=)` takes the bands by their labels first (the band numbers of a file, from
  1), then by position: `bands=[1]` drew the second band of a file read by `load_raster`.
- `ls.plot`: `vmin`, `vmax` and `log` apply to the column of a vector layer too; a ramp's legend
  marks its ends `≤` / `≥` when values beyond them share their colours (static figures: arrow
  ends on the colour bar); the values of a vector layer's tooltip are escaped.

## [1.0.1] - 2026-10-02

### Changed
- `ls.sdm.autocorrelation_range` computes gstat's sample variogram and fits the models in C++ (9 layers of
  5 000 cells: 11.6 s -> 0.73 s, 24 times faster than blockCV); `ls.sdm.clean_occurrences` only draws the
  circles of reference points that can reach a record, built in one call (5 000 records: 0.70 s -> 0.16 s).
  The results are the same.
- The benchmarks page has a section on species distributions (`tools/benchmark_sdm.py`).

## [1.0.0] - 2026-10-02

The first stable release. It adds species distribution modelling (`ls.sdm`): from occurrence
records and predictor rasters to cross-validated, projected models, every step reproducing the R
package the field relies on for it.

### Added
- `ls.sdm`, species distribution modelling: example data
  (`ls.sdm.example_data`: the *Bradypus* and *Solanum acaule* records and the bioclimatic and biome
  rasters of the R package predicts), `ls.sdm.one_per_cell` (one record per predictor cell, as
  `terra::cellFromXY` + `duplicated()`), `ls.sdm.download_gbif` (records from GBIF's occurrence search
  API, no account or extra package needed), `ls.sdm.accessible_area` (geodesic buffers, convex hulls
  or the regions holding records, cut to a shape), `ls.sdm.thin` (spThin's heuristic, the same
  records as `spThin::thin` for the same seed, in C++ without its n x n matrices: 100 000 clustered
  records in 0.12 s), `ls.sdm.clean_occurrences` (CoordinateCleaner's tests: capitals, centroids,
  institutions, GBIF headquarters, zeros, equal coordinates, sea, towns, country mismatch,
  outliers, duplicates, artificial hotspots; flag or drop, the same flags as CoordinateCleaner, with
  its errors corrected: see the function's notes), `ls.sdm.reference_data` (its reference tables and
  Natural Earth layers, downloaded on first use), `ls.sdm.select_predictors` and `ls.sdm.vif`
  (usdm's `vifstep` / `vifcor` and `vif`, the same values to the last bit: R's `lm` QR and
  `summary.lm` in C++, terra's cell sampling with R's generator; usdm's crashes with one variable
  left or every variable kept are fixed), `ls.sdm.background` (dismo's `randomPoints`, the same cells
  for the same seed: area-weighted on longitude / latitude grids, within the accessible area, a
  distance band around the records (biomod2's "disk") or outside their environmental envelope
  (biomod2's "SRE"), or weighted by a bias surface for target-group background), `ls.sdm.dataset`
  and `ls.sdm.SDMData` (presences, background and predictor values, ready for scikit-learn:
  `X`, `y`, `sample_weight`, `cv`), cross-validation folds: `ls.sdm.spatial_blocks`
  (blockCV's `cv_spatial`: square or hexagonal blocks, random / systematic / checkerboard selection,
  the same folds for the same seed; on longitude / latitude data with straight block edges, and
  offsets in the blocks' own units, where blockCV mixes metres and degrees), `ls.sdm.buffer_folds`
  (blockCV's `cv_buffer`), `ls.sdm.autocorrelation_range` (blockCV's `cv_spatial_autocor`: automap's
  variogram classes and models with gstat's Gauss-Newton fit, ranges to 1e-6),
  `ls.sdm.spatial_clusters` (blockCV's `cv_cluster`: R's `kmeans` ported to C++, Hartigan-Wong,
  Lloyd / Forgy and MacQueen, the same clusters as R for the same seed, starts in parallel: 100 000
  points in 0.25 s against 1.9 s), and ENMeval's partitions `ls.sdm.quadrant_folds` ("block"),
  `ls.sdm.checkerboard_folds`, `ls.sdm.random_folds` and `ls.sdm.jackknife_folds`, evaluation:
  `ls.sdm.evaluate` (predicts' `pa_evaluate`: AUC, kappa, TSS and the rest per threshold, the usual
  thresholds, the same values), `ls.sdm.boyce` (ecospat's continuous Boyce index without its "+1
  trick", which made the index depend on the scale of the predictions), `ls.sdm.threshold` and
  `ls.sdm.omission_rate` (ENMeval's `or.mtp` / `or.10p`), extrapolation: `ls.sdm.mess` (predicts'
  MESS, the same values), `ls.sdm.mop` (the mop package: out-of-range codes and distances to the
  nearest reference conditions in C++; the Mahalanobis distance with the reference's covariance,
  where mop used that of the cells tested) and `ls.sdm.exdet` (dsmextra's NT1 / NT2, with its
  combinatorial "most influential covariate" numbered the right way round), and the pipeline:
  `ls.sdm.prepare` (every step in one call with sensible defaults, each step skipped, set by a
  number, a dict of its arguments or a function; `report`, `settings` and `redo()`), `ls.sdm.fit`
  (any scikit-learn estimator, cross-validated on the folds: AUC, TSS, Boyce, omission rates and
  overfitting per fold), `ls.sdm.predict` (in blocks of rows, lazy rasters included; weighted
  ensemble and the spread between models; clamping to the data's ranges),
  `ls.sdm.response_curves` and `ls.sdm.importance`, `ls.sdm.MaxNet` (Maxent as R's maxnet fits it,
  through glmnet's path, in C++: the same coefficients, lambda and predictions as R to the last bit,
  10 000 background points and 20 predictors in 0.47 s against 6.4 s; scikit-learn's API, the default
  model of `ls.sdm.fit`), and the reference harness
  (`tools/make_reference_sdm.py`).
- R data files (`.rda`, `.rds`) are read without R (`landschaft._rdata`), so the reference tables
  of R packages (CoordinateCleaner's centroids, capitals, institutions, land and sea) are used as
  published.
- Reference data too large to ship are downloaded on first use into a cache folder, pinned by
  checksum (`LANDSCHAFT_CACHE`).

### Changed
- R's weighted sampling without replacement (`sample.int(..., prob = )`, used by the neutral landscape
  models and ls.sdm) runs in closed form over runs of equal weights with a Fenwick tree, the same
  draws as R: 1 million cells and 30 000 draws in 0.24 s instead of 16-20 s.
- `ls.load_table` reads delimited text saved in Windows-1252 / Latin-1 (as Excel on Windows often
  saves it) when it is not valid UTF-8; `encoding=` sets it explicitly.

## [0.101.0] - 2026-10-02

### Changed
- `ls.reclassify` runs in C++, in one parallel pass with no temporary arrays (it now takes
  `n_threads`). 8- and 16-bit rasters go through a table of every value of their type, and 32/64-bit
  integers through a table of the raster's range when that range is small. Ranges and other values
  use a branchless search. On 5000 x 5000 cells (11 threads): uint8 0.17 s -> 0.005 s, int32
  0.97 s -> 0.01 s, float32 with ranges 0.65 s -> 0.02 s, a habitat map 0.27 s -> 0.004 s (with a
  function 0.23 s -> 0.05 s). Peak memory drops from 18 to 40 bytes per cell to the result alone.
  The results are the same, except that an int64 kept value above 2^53 is now exact (it was
  rounded through float64), and a dask-backed habitat map stays lazy.

### Fixed
- `ls.zonal_stats`, `ls.extract`, `ls.rasterize` and `ls.reproject` failed with "unsupported raster
  type" on rasters of the second NumPy dtype of a C type: int32 as `int` on Windows (e.g. the type
  of `np.result_type(np.uint16, np.int16)`, or of a raster computed from int16 and uint16 ones)
  and int64 as `long long` on Linux and macOS. `ls.reproject` also failed on rasters in the
  other byte order.
- `ls.plot` of a stack draws up to 128 bands (was 16: the later years of a 41-year land-cover
  series were left out), 1024 pixels on the long side up to 64 bands; the map's list of layers
  scrolls when it is longer than the map.
- A map fills the window it is shown in and follows its size (a stack with a note put it in a
  frame 640 pixels high); in a notebook its bottom-right corner drags it taller or shorter.

## [0.100.0] - 2026-10-02

### Added
- `ls.reproject`: rasters and vectors to another CRS, as terra's `project()`. A raster goes on
  the grid GDAL suggests, with a cell size (`res=`), or onto the grid of a reference raster
  (`like=`); `res=` alone resamples in the same CRS; a vector, or a raster passed as `crs=`,
  gives its CRS only. Every GDAL resampling method; vectors keep their attributes and can be
  densified first (`densify=`). Rasters are warped by GDAL's resampling kernels ported to C++
  (`gdalwarpkernel.cpp`, every method but `sum`, which runs GDAL's warp itself), in parallel, on
  source coordinates fixed beforehand: GDAL's own transformation (the GDAL rasterio loads) at the
  nodes of a lattice anchored at the result's origin, every 16 cells, finer down to every cell
  where interpolating them would be off by more than `tolerance` (1/8 cell), interpolated in
  between. In memory, lazy (dask) or from a file (read window by window), with any number of
  threads, the values are the same bits (GDAL's own warp depends on how its memory limit splits
  the raster), and faster than GDAL's own warp (8000 x 8000 float32, UTM to EPSG:4326, 11
  threads on 6 cores: nearest 0.13 s against 0.49 s, bilinear 0.21 s against 0.57 s, average
  0.33 s against 0.64 s). `tolerance=0` transforms every cell exactly: GDAL's warp of the
  whole result at once, bit for bit, for every method and type, across the antimeridian, at
  the poles and off the globe, but for two GDAL bugs, fixed here:
  from a geographic source GDAL does not reproject the poles it checks for in the result (and,
  missing one there, estimates the kernel scale from the result's edges only; GDAL 3.12, still
  in its master), and GDAL before 3.12.3 warps Int8 rasters as Int16 (values past -128 then
  become the NoData).
- `ls.write_raster(..., colormap={value: colour})` takes colours as `"#rrggbb"`, names
  (`"forestgreen"`) or `(r, g, b[, a])`, as `ls.plot` does.

### Changed
- `ls.match_grid` and `load_raster(..., like=)` warp as `ls.reproject` does: their values can
  differ slightly from before (in the last digits of interpolated values, and the few cells whose
  nearest source cell lies within 1/8 cell of an edge), and no longer depend on GDAL's memory
  limit. A lazy raster stays lazy instead of being loaded whole.
- The tutorials, the quickstart, the concepts and the home page look at results with `ls.plot`,
  save them with `ls.write_raster`, `ls.write_vector` and `ls.save`, read tables with
  `ls.load_table` and recode classes with `ls.reclassify`.
- `pip install landschaft` (and `pip install -e .`) installs everything every tool uses: dask,
  geopandas, pyarrow, openpyxl, matplotlib, folium, plotly, Pillow, xyzservices, contextily,
  pywebview, h3 and dendropy are now dependencies. The extras `dask`, `plot`, `window`, `vector`,
  `bio` and `all` are gone (only `test` and `docs`, for development, remain), and so is the
  Docker image's `EXTRAS` build argument.
- `ls.plot` draws a floating-point raster of a few whole numbers (at most 12) as classes: a class
  map read with `load_raster(..., masked=True)`, its NoData as NaN. Legend labels of such classes are
  `1`, not `1.0`.
- `ls.plot` lists MSPA's classes in GuidosToolbox's order (Core, Islet, Perforation, Edge...) and
  leaves the products' missing class out of the map (the imagery shows through) and the legend, as
  any class given a transparent colour.

### Performance
- Polygons are burnt in time proportional to the edges each row crosses, not rows times
  vertices (GDAL's rasterizer, and Landschaft's port of it, test every edge on every row); the
  cells are the same. `rasterize`, `clip`, `zonal_stats` and `extract(buffer=)` on a border of
  200 000 vertices over 8000 x 8000 cells: 0.07-0.12 s, against 2.7-4 s for rasterio,
  rioxarray and rasterstats.
- `ls.clip` masks with Landschaft's rasterizer instead of rasterio's `geometry_mask` (the same
  cells) and in one NumPy pass; `clip`, `rasterize`, `match_grid` and `load_raster` no longer copy
  whole rasters to write their CRS, transform or NoData (rioxarray's `write_*` do unless asked to
  work in place). On 8000 x 8000 cells, `rasterize` of 2000 polygons takes 0.03 s instead of
  0.09 s, and `clip` to a circle 0.08 s instead of 0.14 s. A crop with bounds only, and
  `match_grid` onto the grid a raster already has, return views of the raster's values, as
  xarray's selections do.
- The rasters every other tool returns (`reclassify`, habitat maps, `ls.nlm`, `ls.lsm` maps,
  `ls.surface`, `ls.cost`, `ls.circuit`, `ls.conn`, `ls.prioritize`, `ls.bio`) get their CRS,
  transform and NoData without a copy of their values.
- `zonal_stats`: the median and the percentiles of a zone of a million cells or more are selected
  in parallel from the raster (a histogram of the values' top bits, then only the values of the
  bins that hold the ranks asked for) instead of gathering and partitioning all its values on one
  thread; the results are the same bits. A zone of 40 million cells: the median adds 0.02-0.05 s
  instead of 0.45-0.55 s, five percentiles take 0.2-0.4 s instead of 0.8-1.1 s.

### Fixed
- The rasters of `ls.cost` (`acc_cost`, `passage`, the transition summaries...) keep the CRS and
  transform of the raster the transition came from; they had the coordinates only.
- `ls.plot` of a list that is not only georeferenced rasters and vectors (tables, arrays) draws
  each part as it calls for instead of failing.
- Windows: a plain `pip install -e .` builds with Visual Studio's compiler (MSVC) whatever else is
  on `PATH`. A CMake found first that did not know the installed Visual Studio (Strawberry Perl's,
  for example) made the build pick a MinGW gcc, whose module failed with "DLL load failed while
  importing _core". CMake 4.2 or newer is now required on Windows (pip fetches it), and a
  non-MSVC compiler on Windows stops the build with a message.
- `ls.plot`: the NoData of a raster on an interactive map is transparent again; reprojected to
  Web Mercator, it was drawn in the first colour of the legend.
- Apple silicon (and aarch64 Linux): `pip install -e .` builds again; the prioritization module
  failed to compile ("no matching function for call to 'round'"), Eigen having no rounding of
  NEON double packets. With older Xcode (before 16) a C++20 lambda capture in the least-cost
  search stopped the build too.
- `pip install -e ".[test]"` installs threadpoolctl, which the thread tests need.

## [0.99.0] - 2026-10-01

### Added
- `ls.plot(obj)`: the plot a piece of data calls for, in one call. A raster or a vector layer goes
  on an interactive map (folium) over satellite imagery: the products of the tools in GuidosToolbox's
  colours with their class names, other class maps (integers with at most 64 values, booleans) in
  distinct colours, continuous values on a ramp chosen by the raster's name and values (priority
  ranks, currents on a log scale, elevation, NDVI, diverging around zero), stacks band by band or as
  true colour, vector layers coloured by a column. Tables become charts (plotly); the results of
  Zonation, habitat graphs, Biodiverse, PCA and the tools that return several rasters show their
  maps, charts and numbers together. `cmap`, `labels`, `categorical`, `vmin`, `vmax`, `log`,
  `column` and `basemap` change the defaults. It shows inline in notebooks and in a window of its
  own elsewhere (pywebview, else the web browser); `static=True` draws a matplotlib figure (the
  imagery with contextily). `Plot.save` writes a self-contained HTML page, or PNG, PDF, SVG.
- Fast at any size: rasters are reduced to the screen's resolution before anything else (block
  means in C++ with OpenMP, `io_block_mean`; lazy rasters read reduced, in strips or through their
  overviews), reprojected once and written as 8-bit PNGs with a palette; the images and GeoJSON go
  into the page after folium's templates. A 10 000 × 10 000 raster draws in 0.2 s, 2 million points
  in 0.8 s; see the benchmarks page.
- Extras `plot` (now matplotlib, folium, plotly, Pillow) and `window` (pywebview); the environment
  variable `LANDSCHAFT_PLOT_SHOW=0` keeps windows from opening.
- `ls.save` and `ls.plot` take an `xarray.Dataset` (the result of `circuit.omniscape`, for example):
  one file per variable, its attributes as a table.

### Changed
- `ls.rasterize` burns polygons in C++ (`io_burn_runs`, from the GDAL port) and points with NumPy:
  50 000 polygons in 0.07 s instead of 2.1 s, 200 000 points in 0.1 s instead of 3.5 s. Large
  layers are reprojected in parallel.

### Fixed
- `ls.rasterize` and `ls.zonal_stats` orient every polygon ring clockwise as GDAL does, so that the
  cells whose centre lies exactly on a horizontal edge are those GDAL burns.
- The files of lazy rasters are closed before the interpreter exits, which ended with an "Error in
  sys.excepthook" message.

## [0.98.1] - 2026-10-01

### Fixed
- Linux (and any build whose C library is not Windows' UCRT) gives the same results as Windows,
  and as R for Windows, for grid distances on longitude/latitude, the geodesic cell sizes and the
  surfaces' autocorrelation: `hypot` is now computed as the UCRT computes it (`sqrt(x*x + y*y)`,
  rescaled by a power of two near overflow and underflow) on every platform; glibc's differed in
  the last bit. Off Windows, only the slopes of `surface.terrain` can still differ from R's, by one
  unit in the last place of the arctangent (the UCRT's code is not public); see the benchmarks
  page.
- Test inputs drawn with platform functions are stored with the references: the fft draws (which
  take `log`) and the test DEMs (`sin`, `cos`) differed in the last bit on Linux.

## [0.98.0] - 2026-10-01

### Added
- `ls.zonal_stats(raster, zones, stats)`: statistics of a raster in every zone, polygons (overlaps
  allowed; reprojected to the raster) or the labels of a raster, on every band of a stack: count,
  nodata, sum, mean, min, max, range, std, var, median, percentiles (`"p90"`), majority, minority,
  unique. C++ with OpenMP (`src/cpp/io`): the polygons' cells are those GDAL burns (centre rule or
  `all_touched`, from the GDAL port of `lsm`, now able to burn into a window), percentiles by
  linear-time selection with NumPy's interpolation; the result does not depend on the threads.
  13-15 times faster than rasterstats and 22-46 times than exactextract on 25 million cells
  (`tools/benchmark_zonal.py`, benchmarks page).
- `ls.extract(raster, points)`: the values of every band at points (nearest cell or bilinear),
  or the statistics of the cells around them (`buffer=`); points reprojected to the raster; a
  dask-backed raster is read only in the blocks of rows that hold points. A million points in
  0.014 s.
- `ls.clip(raster, region)` crops and masks to polygons or bounds (class maps keep their type and
  NoData), and `ls.load_raster(path, clip=region)` reads only the region's window.
- `ls.load_raster([files], like=raster)` (or a folder or pattern) stacks rasters of any grids as one
  `(band, y, x)` array named by the files.
- `ls.reclassify(raster, table)` gives new values from a table: a dict, pairs `(old, new)`, ranges
  `(low, high, new)` or a DataFrame, with `others=` for the values it does not hold; lookup tables
  for small integer types, lazy on dask. With `foreground=` (or a list of classes) it still builds
  habitat maps.

### Fixed
- The Docker test image (`docker build --target test`) copies the example data of every module
  (`bio`, `cost`, `surface`), whose tests failed to load it.

## [0.97.0] - 2026-10-01

### Added
- Loading and saving everything in one call each, so that analyses need no rasterio, xarray or
  geopandas code around them (new page *Getting started > Loading and saving data*):
  - `ls.load_raster(path, like=raster)` puts a raster on the grid of another as it is read
    (reprojected and resampled with GDAL in parallel: nearest cell for classes, bilinear for floats,
    or `resampling=`), with the coordinates of `like` so that the two combine cell by cell;
    `ls.match_grid(data, like)` does it in memory.
  - `ls.load_vector` reads any vector format, GeoParquet and tables of point coordinates (columns
    found by name: `lon`/`lat`, `decimalLongitude`...), reprojected to a raster's CRS with `like=`;
    `ls.write_vector` writes GeoPackage, Shapefile, GeoJSON, FlatGeobuf, GeoParquet or CSV (WKT).
  - `ls.rasterize(vector, like=raster)` burns features on the grid of a raster: a mask, the values
    of an attribute, or counts of overlapping features (`merge="add"`).
  - `ls.load_table` reads CSV and text files (separator and decimal comma detected), Excel,
    Parquet, Feather, JSON and vector attribute tables; `ls.write_table` is now general (it was a
    metric-table writer) and adds `.tsv`, `.xlsx`, `.feather`, `.json`, keeping a named index.
  - `ls.save(obj, path)` writes anything in the format that suits it: rasters as GeoTIFF, tables as
    CSV (or `table_format=`), GeoDataFrames as GeoPackage, trees as Newick, and results of the tools
    as a folder with one GeoTIFF per map, one table per table and `summary.json` with the numbers
    and settings. Every result class has a `save(folder)` method; Zonation's and Biodiverse's
    results keep their own formats.
- `ls.example_data(name)` returns the example data of every module (surfaces, grid distances,
  biodiversity), not only the habitat and land-cover maps.

### Changed
- Biodiverse's vector inputs (`BaseData.from_vector`, shape-file spatial conditions) are read with
  `load_vector`.

### Fixed
- `write_raster` (and every writer built on it) took the affine transform rioxarray caches, which
  keeps the old cell size after a strided slice such as `data[::2, ::2]`: such rasters were written
  with the wrong cell size. The transform is now computed from the coordinates.

## [0.96.0] - 2026-10-01

### Changed
- `ls.load_raster` reads any raster without extra arguments: no habitat checks by default
  (`validate="mspa"` runs them; the habitat tools check their input anyway); every band of a
  multi-band file as a `(band, y, x)` array whose `band` coordinate holds the band descriptions
  (`band=N` one band, `band=[...]` several); NoData of floating-point rasters as NaN while integer
  class maps keep their codes (`masked=True` / `masked=False` to always / never mask). Collections
  mask the same way.
- `ls.write_raster` writes what the data say about themselves: several bands with their names,
  NaN as the NoData of floats (or `nodata=`, which replaces the NaN), integers with their
  `.rio.nodata`, `bool` as `uint8`, 64-bit integers as the smallest type that holds them, `dtype=`
  to cast; `like=` georeferences a numpy array from a raster on the same grid; the format follows
  the extension (`.tif`, `.asc`, `.img`, `.nc`...; `driver="COG"`); compression with the predictor
  that suits the type; dask arrays written block by block; missing folders created.
- Every tool reads and writes rasters through these two functions: Zonation's rank and
  post-processing maps, Circuitscape's and Omniscape's grids, the grid-distance inputs.

### Fixed
- Reading inputs from files no longer checks them as MSPA habitat maps where they are not:
  `fos_change` (FOS maps), `parcellation` and `gray_spatial_convolution` (class and grey-scale
  maps), the resistance of `cost_distance`, `least_cost_path` and `restoration_corridor` and the
  source mask of `cost_distance` were rejected when given as paths.

## [0.95.0] - 2026-09-30

### Added
- Spatial prioritization, part 7: the tutorial "Conservation priorities" (features from land cover,
  performance curves, complementarity against richness hotspots, rules, weights, costs, existing
  reserves, connectivity, ties) and chapter 12 of the course, "Conservation priorities"
  (systematic conservation planning, complementarity and irreplaceability, how Zonation ranks,
  performance curves, costs, reserves, connectivity, data and uncertainty), with glossary terms and
  further reading. `tools/benchmark_prioritize.py` times z5 and Landschaft on synthetic features and
  checks the rank maps; the benchmarks page has the results: on 12 threads 4 to 11 times faster than
  z5 on 12 threads, with z5's rank map on one thread bit for bit.

### Changed
- `ls.prioritize.zonation` reads the features in large chunks of rows and closes Zonation's per-block
  sums at the files' block boundaries inside the chunk: GeoTIFFs in strips of one or two rows were
  read row by row (loading 30 features of 1000 × 1000: 3.7 s to 0.3 s).
- The course's "From question to tool" is now chapter 13; the home page and README list the
  biodiversity and prioritization modules.

## [0.94.0] - 2026-09-30

### Added
- Spatial prioritization, part 6 (large maps): `ls.prioritize.zonation(..., storage=dir)` keeps the
  large arrays (quantized features, ranking rows and work arrays) in memory-mapped temporary files
  that the operating system pages to disk, for maps larger than the memory; same results bit for
  bit, the files deleted when no longer needed.

### Changed
- The quantized features are stored as runs of NoData, zeros and values (one or two bytes per
  value): a feature costs about its occurrences, not one entry per cell with data (4 bytes before).
- The ranking works on compact rows (a bitmap of the layers or their gaps, whichever is smaller,
  and one byte per 8-bit value), sorts 16-byte items with a radix pass inside the sample-sort
  buckets, specializes the marginal-loss pass per rule and sums only the nonzero ranking-error
  terms; the reading and quantization run in parallel (per layer, in cell order for Zonation's
  sums), files are read in parallel threads and dask blocks computed in one graph. Benchmark
  (2000 x 2000 cells, 30 features, 75 iterations, 12 threads): 13.1 s to 6.9 s (z5 v2.4: 31.9 s
  on 12 threads, 52.8 s on one), with the same rank map as z5 on one thread; about 130 bytes of
  memory per cell with data during the ranking.

## [0.93.0] - 2026-09-30

### Added
- Spatial prioritization, part 5: Zonation 5's impact assessment in `ls.prioritize.zonation`
  (`impact_groups`, `impact_bins`, `impact`, `cumulative_impact`): impact sources with footprint
  and buffer zones by Euclidean distance (`ImpactScenario`, `ImpactSource`), and direct habitat
  loss and cumulative pressures with response functions, both variants and aggregations
  (`CumulativeScenario`, `Pressure`); the losses per area, impact group, source, zone and value
  class (`ZonationResult.impacts`), the impact rasters, and the features reduced before the
  ranking. `read_project` reads the scenario, impact buffer, binning, raster list, matrix and
  response function files; `write` writes Zonation's impact files. Bit for bit with z5 on 5 more
  runs: every one of their 742 impact files and the rank maps.

## [0.92.0] - 2026-09-30

### Added
- Spatial prioritization, part 4: Zonation 5's post-processing as methods of `ZonationResult`:
  `area_statistics` (external zones), `area_statistics_rank_range`, `area_statistics_separate_areas`
  (sizes, mean ranks, feature coverage, relative richness and cost per area, per weight class and
  output group; `AreaStatistics`), `solution_comparison`, `histogram` and `flexibility_map`;
  `hierarchic_mask_ranks`; `postprocess` runs Zonation's post-processing files and writes its files
  (`run_project` with `-p`). Area statistics, comparisons and histograms digit for digit with z5
  (on its LOAD run of the ranking); flexibility maps bit for bit where z5's curves are those of its
  ranking, and against a direct port of its algorithm elsewhere (z5 computes them from the curves
  of the order before its last sort).

## [0.91.0] - 2026-09-30

### Added
- Spatial prioritization, part 3: Zonation 5's connectivity transforms in `ls.prioritize.zonation`:
  single-feature connectivity with dispersal kernels (`connectivity`, `self_multiply`; `-d`),
  interaction connectivity with a resource, positive or negative (`interaction`, `interaction2`;
  `-i`, `-I`), and matrix connectivity between features (`matrix_connectivity`, `matrices`; `-m`);
  `ls.prioritize.Kernel` (uniform, triangular, negative exponential, Gaussian, with cut-off),
  `Interaction`, and `convolve` (FFT convolution in blocks, as Zonation). `read_project` reads the
  connectivity link files and matrices. Bit for bit with z5 on 6 more runs (smoothed layers, rank
  maps, tables), on a grid spanning several FFT blocks.

## [0.90.0] - 2026-09-30

### Added
- Spatial prioritization, part 2: in `ls.prioritize.zonation`, condition layers (`condition`,
  `renormalize`; Zonation's `-c`), retention (`retention`: constant fractions, fraction rasters or
  absolute amounts; `-r`), Zonation's pre-processing maps (`preprocessing`: richness and range-size
  rarity per weight class and output group, weighted; `-b`) and the feature layers as ranked
  (`transformed`; `-f`). `ls.prioritize.read_project` and `run_project` read and run Zonation 5
  projects: the settings file, feature list (value extraction included) and link files, with z5's
  command-line switches. Checked bit for bit against z5 on 9 more runs (rank maps, tables,
  pre-processing maps, transformed layers), and every reference run through its project files.
- Warnings for weight and output groups mixing positive and negative weights, as Zonation.

## [0.89.0] - 2026-09-30

### Added
- Spatial prioritization, part 1: `ls.prioritize.zonation`, a port of Zonation 5 v2.4
  (Lehtinen, Moilanen et al., University of Helsinki; GPL-3). Feature layers (files,
  NumPy, xarray or dask stacks) quantized to 8 or 16 bits; the marginal loss rules CAZ1,
  CAZ2, CAZMAX, CAZP, CAZPW, ABF (per-feature exponents), RAND and LOAD; weights, weight
  groups, output groups, thresholds, zero modes, value extraction from categorical rasters,
  cost (tracked or in the ranking), hierarchic masks, analysis areas; the iteration
  settings (maximum iterations, ranking error, smoothing, plateau). The result holds the
  rank map, the feature, summary, density and group curves, the feature information and
  the iteration history, and writes Zonation's files (`ZonationResult.write`). Same rank
  maps as the z5 program on one thread, bit for bit, and the same tables digit for digit,
  checked on 41 runs (`tools/make_reference_prioritize.py`, `tests/prioritize_cases.py`);
  vdt's fast_log/fast_exp reproduced as Zonation's `-Ofast` build computes them.
- API page *Spatial prioritization*.

### Changed
- Zonation 5 reports the curves and the area under the curve of the order before its last
  sort, and the rank map of the sort after it; `zonation` gives the curves of the ranking
  it returns (checked against Zonation's LOAD mode on its own rank map).

### Performance
- The ranking walks the feature occurrences in the current order, kept in that order
  between iterations, with the losses and their sums per cell in one parallel pass from
  the levels of every feature at the start of each block of cells, and a parallel sample
  sort. On 3.1 million cells with 30 features (75 iterations): 13 s against 54 s for z5 on
  one thread and 33 s on 12 threads (6 performance cores).

## [0.88.0] - 2026-09-30

### Added
- Learn, chapter 11 *Continuous surfaces*: the patch-mosaic and gradient models (what a
  threshold does to a continuous map), surface metrics by hand, the bearing area curve,
  autocorrelation, anisotropy and wavelength, texture images, terrain indices, pitfalls;
  figures `learn_threshold` and `learn_aacf`, six glossary terms. *From question to tool*
  becomes chapter 12.

### Changed
- Faster focal means, sums and standard deviations in `ls.surface` (`tpi`, `hsp`, `srr`, `vrm`,
  `spherical_sd`, Bolstad's `curvature`): compensated prefix sums along the rows of the
  values minus their mean, one step per window row instead of one per cell, and for `hsp`
  one set of sums for all scales. Results within about 1e-13 of summing each window (no
  longer bit for bit); `hsp` on 2000 x 2000 with scales 3-27: 1.9 s to 0.44 s.
- `ls.surface` outputs on a DataArray's grid reuse its coordinates and spatial reference
  instead of writing them again (about 1 ms less per call).

## [0.87.0] - 2026-09-30

### Added
- Tutorial *Surface texture*: trend removal, whole-surface metrics, the bearing area curve,
  texture images in moving windows, spatial structure and terrain indices, on the Oregon
  NDVI and elevation rasters (figures `surf_texture`, `surf_bearing`, `surf_terrain`).
- Surface-metric benchmarks against geodiv and spatialEco (`tools/benchmark_surface.py`,
  `tools/surface_benchmark.R`): 2 to 1100 times faster for whole-raster metrics, 6000 times
  for texture images of the bearing-area metrics; on the benchmarks page and its chart.

## [0.86.0] - 2026-09-30

### Added
- Surface metrics, part 6: terrain indices of DEMs, from the R package spatialEco 2.0-5
  over terra 1.9-50. `ls.surface.terrain` (slope and aspect as terra's `terrain`: Horn's
  8 or 4 neighbours, geodesic cell sizes on longitude/latitude rasters), `tpi` (square or
  circular windows, normalised), `tri`, `vrm`, `srr`, `dissection`, `curvature`
  (planform, profile, total, McNab, Bolstad), `spherical_sd`, `hsp`, `trasp`, `hli`. Slope,
  aspect, tri and dissection identical to terra / spatialEco bit for bit; vrm, srr and
  spherical_sd within 1e-11 (terra slides running sums whose rounding depends on its
  blocks; here each window is summed directly); on longitude/latitude rasters slope and
  aspect within 1e-12 (terra's geodesics use the UCRT's sin and cos).
- Where spatialEco errs, the method's intent, checked bit for bit against spatialEco with
  the errors corrected in R: curvature's second derivatives over L^2 (spatialEco halves
  them), missing edges (spatialEco pads with 0), no rounding to 6 decimals, the cell height
  for y, McNab's mean over 8 neighbours (spatialEco: / 9 - 1) and Bolstad's with the cell
  width (it fails in spatialEco); hsp's z-scores ((tpi - mean) / sd; spatialEco:
  tpi - mean / sd); trasp with pi (spatialEco: 3.142); hli's southern hemisphere folded in
  radians with absolute values and the absolute latitude, and radians left as they are;
  circular windows by exact distances (terra's focalMat measures in single precision with
  the cell width in both directions). spatialEco's `zero.correct` (the same mean, and a
  standard deviation of 0 when normalising) is not offered.

## [0.85.0] - 2026-09-30

### Added
- Surface metrics, part 5: texture images. `ls.surface.window_metrics` gives every surface
  metric in a moving square or circular window around each cell (missing cells beyond the
  edges), each window computed as a raster of its own, rows of cells in parallel; `sv`,
  `sph` and `sds` on square windows by running extremes and a summed-area table of the
  peaks. Checked against R on 8 setups (square and circular windows, projected and
  longitude/latitude, every metric). geodiv's own `texture_image` does not work with
  current terra (it reads the raster as one column), its `focal_metrics` transposes every
  window and its circular windows mask nothing.

### Changed
- `ls.surface` no longer reproduces geodiv's errors: `convention` is gone, and where geodiv
  errs the method's intent is computed. Sq is ISO 25178's root mean square (over N, not
  N - 1), also inside `ssk` and `sku`; `sdq` averages over the pairs with both values;
  `sdq6` has the right seven-point stencil (`+ 9 z[-2]`) and takes the mean; valleys are
  compared with their own neighbours (also in `s10z`); `ssc` is ISO's -1/(2n) sum of second
  derivatives; the bearing area curve works on the heights themselves, `svi` / `sci` are the
  areas between the curve and its top (geodiv's could turn negative) by a complete
  Simpson's rule, and the core line's end slopes are one-sided differences over h;
  `scl`, `stxr`, `std` and `srw` measure from zero lag / frequency (geodiv's images leave
  its row out) with exact distances in map units, `std` counts orientations 0-179.5
  degrees without zero frequency; `sfd` has its prism typo, extremes and indices fixed (it
  now works on any raster); `best_fit_plane` / `remove_plane` choose the order by BIC
  (geodiv's criterion, the mean residual, is 0 for every fit). Everything is checked bit
  for bit against geodiv, or against geodiv with these errors corrected in R
  (`tools/surface_fixed.R`).
- The core line of the bearing area curve picks its window from prefix sums and R's mean
  only for the candidates (the same window; 1000 windows: 19.6 s to 0.2 s).

## [0.84.0] - 2026-09-30

### Added
- Surface metrics, part 4: spatial parameters and fractal dimension, as geodiv.
  `ls.surface.aacf` (areal autocorrelation image), `scl` (correlation lengths), `stxr`
  (texture aspect ratio), `std` (texture direction and index), `srw` (dominant radial
  wavelength, its index, half-power wavelength), `sfd` (triangular-prism fractal
  dimension); in `ls.surface.metrics` as `sal`, `str`, `std`, `stdi`, `srw`, `srwi`, `shw`,
  `sfd`. Identical to geodiv, bit for bit, on 12 reference setups, projected and
  longitude/latitude: R's `fft`, e1071's Hann window, terra 1.9's `crop` (snapping to the
  grid) and `distance` (GDAL 3.9's proximity in single precision on projected rasters,
  the haversine on longitude/latitude ones).
- Where geodiv fails, the intended result: `scl` / `stxr` on rasters with missing cells
  (geodiv looks for the centre of the cropped image with the size of the whole raster).
  `sfd` raises an error where geodiv reads past the raster (its central square reaching
  the last row or column); `metrics` gives NaN there.
- geodiv's search for the largest square without missing cells runs on a summed-area
  table (2000 x 2000 with one missing cell: 0.26 s).

## [0.83.0] - 2026-09-30

### Added
- Surface metrics, part 3: the bearing area (Abbott-Firestone) curve, as geodiv.
  `ls.surface.bearing_area` (the curve's points), `height_ba` (heights at bearing
  fractions), `sdc`, `sbi`, `svi`, `sci`, `sk`, `spk`, `svk`, also in `ls.surface.metrics`
  (one sorted sample for all). Identical to geodiv, bit for bit, on 9 reference setups:
  R's `ecdf`, type-7 `quantile`, `seq`, geodiv's Simpson's rule, and its core line
  (`find_flat`: central-difference slopes at 10000 points, the mean of each 40 % window,
  `lm()` through the flattest one).

### Changed
- The port of R's `lm()` (`src/cpp/lsm/rlm.cpp`) also returns the intercept, and that of a
  rank-deficient fit, as R's coefficients.

## [0.82.0] - 2026-09-30

### Added
- Surface metrics, part 2: slopes, developed area and local features, as geodiv. `ls.surface.sdq`,
  `sdq6`, `surface_area`, `flat_area`, `sdr`; `find_peaks`, `find_valleys` (tables of cells
  with coordinates, value, row and column), `sds`, `ssc`, `s10z`; the new metrics in
  `ls.surface.metrics`. Identical to geodiv, bit for bit, on 19 reference setups.
- `convention="iso"` (`sdq6`, `surface_area`, `flat_area`, `sdr`, `find_valleys`, `s10z`,
  `metrics`) for ISO 25178's definitions where geodiv departs from them: geodiv's `sdq6`
  stencil has `- 9 z[+2]` twice in place of `+ 9 z[-2]` and it takes the root of the sum of
  the squares, not of their mean (`length()` of a raster is 1 in R); its `findvalleys` reads
  each cell's neighbours from the values arranged by columns (on geodiv's `normforest` it
  finds 33041 valleys where there are 5167); its areas rescale the heights to 0-1. The
  default stays geodiv's, for comparability with studies done in R.

### Changed
- `ls.surface` amplitude functions compute only what they return (`sa` 4x faster);
  `fit_plane` / `best_fit_plane` fill the regression matrix in parallel and run the
  Householder products of each step's columns in parallel (same bits).

## [0.81.0] - 2026-09-30

### Added
- Surface metrics, part 1: the new `ls.surface` module, the texture of continuous rasters
  (ISO 25178 areal parameters as *gradient surface metrics*), ported from the R package
  geodiv 1.1.1. Amplitude parameters `sa`, `sq`, `ssk` (adjusted or not), `sku` (excess or
  not), `sv`, `sph`, `smean`; trend surfaces `fit_plane` (least-squares polynomials of
  order 0-6 as `spatial::surf.ls`), `best_fit_plane` and `remove_plane`; `metrics` (every
  metric in one table, shared computations done once) and `list_metrics`; geodiv's three
  example rasters (`ls.surface.example_data`). Identical to geodiv, bit for bit, on 18
  reference setups, including `bestfitplane`'s choice of order, which rests on rounding
  errors (R sums the residual matrix column by column in 80 bits). With fewer cells than
  coefficients `spatial` reads past its matrix; here that is an error.
- Reference harness for the surface metrics: `tools/make_reference_surface.py` runs the
  cases of `tests/surface_cases.py` through geodiv and spatialEco in R.

## [0.80.0] - 2026-09-30

### Changed
- Randomised shortest paths (`ls.cost.passage(theta=...)`, `rsp_distance`, `path_inc(theta=...)`)
  are 2 to 8 times faster. Asymmetric transitions are factorised with a new multifrontal LU
  without pivoting (`I - W` is an M-matrix) on the circuit module's nested dissection,
  subtrees in parallel and tiled dense kernels, instead of Eigen's `SparseLU` (1000 x 1000:
  10.2 s to 1.5 s; 2000 x 2000: about 60 s to 7.8 s). The factors of `I - W` decay into
  subnormal numbers, whose arithmetic is two orders of magnitude slower: the factorisations
  and solves now flush them to zero (each thread's floating-point control word is restored
  afterwards), which also halves the symmetric (Cholesky) case (1000 x 1000: 2.2 s to
  0.96 s). Method 2, whose `P^theta` rows can add up to more than 1, keeps the pivoting LU.
  Results stay within 1e-10 of gdistance.

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
