# Changelog

All notable changes to landschaft are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0 the API may still change
between minor versions.

## [Unreleased]

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
