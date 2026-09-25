# Changelog

All notable changes to guidospy are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0 the API may still change
between minor versions.

## [Unreleased]

### Added
- GTB's sample maps (`input`, `clc`, `gscinput`) bundled with the package, with
  `example_data()` and `EXAMPLE_DATA`; `tools/benchmark.py` uses them.

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
  (`tests/data/reference`, optional live comparison via `GUIDOSPY_MSPA_EXE`).
