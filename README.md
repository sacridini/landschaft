# landschaft

Fast Python reimplementation of [GuidosToolbox (GTB)](https://forest.jrc.ec.europa.eu/en/activities/lpa/gtb/)
raster image analysis, with a C++17/OpenMP core and a modern numpy / xarray / dask / rasterio API.

**Current Version (v0.19.0):** Complete reimplementation of all major GuidosToolbox tools:
* **MSPA**: Morphological Spatial Pattern Analysis (bit-identical to GTB `mspa` v2.3, 5-7x faster), and GTB's MSPA Tiling for maps larger than memory (out of core, identical to GTB).
* **SPA2/3/5/6**: Simplified Pattern Analysis (identical to GTB).
* **GTN & Connectivity**: Topological Network Components (ECA, DOC), Importance (dPC), and Conefor Sensinode inputs.
* **Fragmentation & Distance**: Forest Area Density (single & multiscale FAD, $O(1)$ integral images), FOS (Fixed Observation Scale: FAD/FED/FAC in 5 or 6 classes, binary or grayscale input, identical to GTB), Euclidean Distance Transform ($O(N)$), and Directional Spatial Entropy.
* **Landscape Mosaic & Contagion**: LM (19 & 103 classes with barycentric stats), Contagion ($P_{ff}$), and Clustering (FAC).
* **Objects**: Accounting (foreground objects in up to 6 size classes, with object ID and area maps; identical to GTB's IDL code).
* **Restoration Planner**: Isochrone maps A and AB with the least-cost path (identical to GTB's `ggeo`), Distance → Resistance.
* **Distance (experimental)**: Influence Zones and Proximity. Not yet identical to GTB, see the CHANGELOG.
* **Restoration & Morphology**: Restoration Status (RSS), Cost Distance, Least-Cost Paths & Corridors, Landscape Parcellation (AWAPS, DIVISION, PARC), and Habitat Contortion.
* **Change Detection & Convolution**: Simple Change, Morphological Change Detection (MCD with geodesic core reconstruction & elasticity), Fragmentation Change (FOS Delta & 6x6 transition matrix), FAD Multiscale Change (7x7 class transitions per observation scale), Landscape Mosaic Heatmap and Heatmap Change (GTB's heatmap/delta CSV, bit-identical), and Continuous Grayscale Spatial Convolution (GraySpatCon).
* **Landscape Metrics (FRAGSTATS)**: 133 patch-, class- and landscape-level metrics with the names, options and values (bit for bit) of the R package landscapemetrics (area/edge, shape, core area, aggregation incl. nearest-neighbour distance, diversity, complexity), plus FRAGSTATS' `_am`/`_md`/`_ra` statistics, 100-500x faster.

📖 **For complete documentation and usage examples for every module, see [DOCUMENTATION.md](DOCUMENTATION.md).**

```python
import landschaft as ls

data = ls.load_raster("forest.tif")        # reads and checks the 0/1/2 coding
result = ls.mspa(data, edge_width=2)        # georeferenced xarray.DataArray
ls.write_raster(result, "forest_mspa.tif")  # GeoTIFF with the GTB colour table
print(ls.mspa_stats(result))                # class areas, as in GTB
```

## Installation

landschaft builds from source with a C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
and OpenMP if available. Recommended with conda:

```bash
conda create -n landschaft -c conda-forge python=3.12 numpy pandas rasterio rioxarray xarray dask \
      scikit-build-core pybind11 cmake ninja pytest scipy
conda activate landschaft
pip install --no-build-isolation -e .
```

On Windows, with only the Visual Studio *Build Tools* installed, pick the generator:
`set CMAKE_GENERATOR=Visual Studio 17 2022` (or `Visual Studio 18 2026`) before `pip install`.

## Usage

### Example data

GTB's sample maps ship with the package, so everything below can be tried right away:

```python
ls.EXAMPLE_DATA                       # name -> description
data = ls.load_raster(ls.example_data("input"))                 # MSPA-ready 1000 x 1000 map
clc = ls.load_raster(ls.example_data("clc"), validate=None)     # CORINE land cover, 2000 x 2000
forest = ls.reclassify(clc, foreground=[23, 24, 25])            # forests as foreground
```

| name | content |
|---|---|
| `input` | GTB MSPA test map, 1000×1000, coded 0/1/2 (EPSG:4326) |
| `clc` | CORINE Land Cover excerpt, 2000×2000, 25 m (EPSG:3035), CLC classes; forests are 23/24/25 |
| `gscinput` | GraySpatCon test map, 1990×1289, 2430 m (EPSG:5070), percentages 0–100, nodata 255 |

### Input data

MSPA uses GTB's coding: a single-band `uint8` map with
`0` = missing, `1` = background, `2` = foreground.
`load_raster` rejects anything else with a clear message (more than one band,
non-byte data, values above 2, no foreground or no background), just like
GTB's *MSPA compliance* check.

`reclassify` builds that input from categorical or continuous maps:

```python
lc = ls.load_raster("corine.tif", validate=None)
forest = ls.reclassify(lc, foreground=[23, 24, 25], missing=[48])  # nodata -> missing too
tree_cover = ls.reclassify(ls.load_raster("cover.tif", validate=None), foreground=lambda a: a >= 30)
```

### MSPA

```python
result = ls.mspa(
    data,
    connectivity=8,   # foreground connectivity, GTB "FGconn" (8 or 4)
    edge_width=1,     # effective edge width in pixels, GTB "EdgeWidth"
    transition=True,  # GTB "Transition" (only changes the colour table)
    internal=True,    # GTB "Intext": +100 for internal features, openings 100/220
)
```

`mspa` accepts numpy arrays, dask arrays and `xarray.DataArray`s and returns the
same type. Arrays with extra dimensions, e.g. a `(time, y, x)` stack, are
processed slice by slice; with dask the slices run in parallel and lazily:

```python
stack = xr.open_mfdataset(...)["forest"].chunk({"time": 1})
ls.mspa(stack).to_netcdf("mspa_stack.nc")
```

Output classes (`ls.MSPA_CLASSES`):

| value | class | value | class |
|---|---|---|---|
| 17 | Core | 65 / 67 / 69 | Loop (/ in edge / in perforation) |
| 9 | Islet | 33 / 35 / 37 | Bridge (/ in edge / in perforation) |
| 5 | Perforation | 1 | Branch |
| 3 | Edge | 0 | Background |
| 100 | Core-opening | 220 | Border-opening |
| 129 | Missing | +100 | the same class inside a core-opening (`internal=True`) |

### Landscape metrics

The FRAGSTATS metrics of the R package
[landscapemetrics](https://r-spatialecology.github.io/landscapemetrics/), with the
same names and the same values (to the last bit for nearly all of them), computed by
a C++ core that labels the patches of all classes in one parallel pass:

```python
lc = ls.Landscape("landcover.tif")               # or a DataArray / numpy array (res=...)
lc.patch_metrics(["area", "shape", "gyrate"])     # one row per patch
lc.class_metrics(["ca", "pland", "np", "ed", "ai"])   # one row per class
lc.landscape_metrics(["shdi", "contag", "lsi"])   # one row
lc.metrics(level="class", type="aggregation metric")  # long format, like calculate_lsm
lc.to_raster("area")                              # patch metric as a raster (spatialize_lsm)

ls.landscape_metrics("landcover.tif", what=["lsm_c_ca", "lsm_l_ta"])  # one call
ls.list_metrics(level="landscape")               # catalogue (like list_lsm)
ls.sample_metrics("landcover.tif", points_xy, size=500, shape="circle")  # sample plots (sample_lsm)
ls.window_metrics("landcover.tif", window=21, what="lsm_l_shdi")         # moving window (window_lsm)
ls.Landscape("huge.tif", block_rows="auto").metrics()                   # larger than memory, same values
lc.class_metrics(["AREA_AM", "shape_index_md", "gyrate_am"])  # FRAGSTATS extras, pylandstats/FRAGSTATS names
lc.centroids(); lc.nearest_neighbours(return_id=True)            # get_centroids, get_nearestneighbour
lc.adjacencies(8); lc.boundaries(edge_depth=2); lc.circumscribing_circles()
ls.extract_metrics("landcover.tif", points_xy)                   # extract_lsm
ls.calculate_correlation(lc.metrics(level="class"))             # calculate_correlation
```

Options are those of landscapemetrics (`directions`, `count_boundary`,
`edge_depth`, `neighbourhood`, …); `n_threads` limits the cores. `extras=True` adds
the FRAGSTATS statistics landscapemetrics lacks (area-weighted mean `_am`, median
`_md`, range `_ra`, and the `perim` statistics) and FRAGSTATS' proximity, similarity,
connectance and edge-contrast metrics (`prox`, `simi`, `connect`, `econ`, `cwed`,
`teci`); `convention="fragstats"` switches to FRAGSTATS' conventions. On landscapemetrics' `augusta_nlcd` (440×678)
all 133 metrics take 0.03 s instead of 15.6 s in R; a 10,000×10,000 map with 7.7
million patches takes about 10 s for all of them (92 million values).

### Many images

`load_raster` also takes a directory, a glob pattern or a list of paths. It then
returns a `RasterCollection`. The collection reads only the file headers, checks
them (existence, band count, `uint8`) and holds no pixels in memory:

```python
tiles = ls.load_raster("tiles/")              # or "tiles/*.tif", or [path1, path2, ...]
tiles                                          # <RasterCollection: 3120 rasters, different grids, ...>

ls.mspa(tiles, edge_width=2, output_dir="out/", max_memory="16GB")   # -> list of BatchResult
small = ls.mspa(tiles[:10])                    # no output_dir: list of DataArrays in memory
```

Passed to `mspa`, the collection is processed one raster per thread. Each thread
loads, processes and writes one file, then takes the next. The C++ core
releases the GIL, so the threads run truly in parallel.

* Memory stays bounded. With `output_dir` each result goes to disk
  (`<name>_<conn>_<edgewidth>_<transition>_<internal>.tif`, as GTB).
  `max_memory` limits how many rasters run at once, from their size (about
  32 bytes per pixel), besides `n_workers` (default: one per core).
* Existing outputs are skipped, so an interrupted run can be resumed.
  `overwrite=True` recomputes them.
* Errors are recorded per file (`result.ok`, `result.error`) without stopping the
  batch. Files with unusable headers are reported by `load_raster`, or left out
  with `errors="skip"` (see `tiles.skipped`).

When all rasters share the same grid (e.g. a time series of one tile),
`load_raster(..., stack=True)` returns a lazy `(file, y, x)` DataArray with one
dask chunk per file, and `ls.mspa(stack)` processes it in parallel with dask.

`ls.mspa_batch(inputs, output_dir, ...)` does the same as
`ls.mspa(collection, output_dir=...)` and accepts paths, directories or collections directly.

## API reference

All functions are available from the top-level package (`import landschaft as ls`);
their docstrings (`help(ls.mspa)`) give every detail.

| function | purpose |
|---|---|
| `load_raster(source, *, band=None, chunks=None, validate="mspa", stack=False, errors="raise", recursive=False)` | One file: read one band as a `(y, x)` DataArray with CRS/transform, checking the MSPA coding (`InvalidInputError` otherwise); `chunks` gives a lazy dask array. Directory, glob or list: a `RasterCollection` (headers only), or with `stack=True` a lazy `(file, y, x)` DataArray. |
| `RasterCollection` | Sequence of `RasterInfo` (path, shape, dtype, CRS, …) with `.paths`, `.same_grid`, `.total_pixels`, `.skipped`, `.load(i)`, `.to_stack()`. |
| `write_raster(data, path, *, crs=None, transform=None, nodata=None, colormap="auto", compress="lzw", description=None)` | Write a 2-D array or DataArray to GeoTIFF. MSPA results get the GTB colour table and their parameters as tags. |
| `reclassify(data, foreground, *, missing=None, background=None)` | Build a 0/1/2 MSPA input from class values or a boolean function; NaN and nodata become missing. |
| `mspa(data, *, connectivity=8, edge_width=1, transition=True, internal=True, n_threads=None, validate=True, output_dir=None, n_workers=None, max_memory=None, overwrite=False)` | Run MSPA on numpy, dask or xarray data (2-D or stacks), or on a `RasterCollection` in parallel (the last four options are for collections). |
| `mspa_batch(inputs, output_dir, *, ..., n_workers=None, threads_per_worker=1, max_memory=None, overwrite=False, raise_errors=False)` | Process many files in parallel; returns one `BatchResult` (`input`, `output`, `seconds`, `error`, `ok`) per file. |
| `accounting(data, thresholds=(50, 250, 1000, 5000, 20000), *, connectivity=8, highlight_largest=False, pixel_resolution=25.0, n_threads=0)` | GTB Accounting: size-class map (GTB colour codes), object IDs, object areas, per-class table and `summary()` report. `thresholds="auto"` uses the classes proposed by GTB's dialog. |
| `fad_multiscale_change(a, b, *, window_sizes=(7, 13, 27, 81, 243), n_threads=0)` | GTB Change → FAD Multiscale: 7×7 fragmentation-class transition tables A → B for every observation scale and the multiscale map, with FAD_AV/AVCON changes. `a`/`b` are FAD-MS results, input maps or GTB `*_fad_mscale.tif` files. |
| `landscape_mosaic_heatmap(data, window_size=27, *, is_lm103=False)`, `landscape_mosaic_change(a, b, window_size=27)` | GTB Landscape Mosaic heatmap (share of the 103 classes, `triangle` grid, `to_csv()` = GTB `heatmap.csv`) and its change B − A (GTB `delta_heatmap.csv`). |
| `isochrone_map(resistance, marker)`, `isochrone_map_ab(resistance, marker)` | GTB Restoration Planner isochrone maps: accumulated cost from object A (marker 1), or the cheapest A–B route (markers 1 and 2) with its least-cost path; marker 3 = missing. GTB display codes 0–100, 101–104. |
| `distance_to_resistance(data)` | Resistance map growing with the distance to the foreground (GTB Setup Tools). |
| `spa(data, variant=5)` | GTB SPA2/3/5/6: map in GTB codes (17 core, 3 edge, 5 perforation, 1 margin, 9 islet, 100 core-opening), class table, object counts, porosity, `summary()`. |
| `batch(tool, inputs, output_dir, *, n_workers=None, threads_per_worker=1, max_memory=None, overwrite=False, **params)` | Any tool (`BATCH_TOOLS`, e.g. `"accounting"`, `"spa"`, `"fad_multiscale"`, or a function) on many rasters in parallel; writes GeoTIFF maps, CSV tables, TXT reports and a JSON record per input. |
| `Landscape(data, *, res=None, nodata=None, directions=8, count_boundary=False, consider_boundary=False, edge_depth=1, cell_center=False, classes_max=None, neighbourhood=4, ordered=True, base="log2", band=None, block_rows=None, n_threads=None)` | Categorical map prepared for landscape metrics: `.patch_metrics()`, `.class_metrics()`, `.landscape_metrics()` (wide tables), `.metrics()` (long table), `.labels()`, `.to_raster(metric)`. Values match landscapemetrics. `block_rows` reads a file by blocks for rasters larger than memory (identical results). Metric names also as FRAGSTATS abbreviations (`"SHDI"`) or pylandstats/FRAGSTATS full names (`"shannon_diversity_index"`). |
| `landscape_metrics(data, level=None, metric=None, name=None, type=None, what=None, *, extras=False, n_threads=None, **options)` | landscapemetrics' `calculate_lsm`: long table of the selected metrics for one raster, a list or a `RasterCollection` (`layer` column). |
| `list_metrics(level=None, metric=None, name=None, type=None, what=None, *, extras=False)` | Catalogue of the metrics (`list_lsm`); `extras=True` adds the FRAGSTATS `_am`/`_md`/`_ra` statistics and `perim_*`. |
| `sample_metrics(data, y, *, size=None, shape="square", plot_id=None, all_classes=False, level=None, metric=None, name=None, type=None, what=None, extras=False, n_threads=None, **options)` | landscapemetrics' `sample_lsm`: metrics of sample plots (buffers around points, or polygons), with `plot_id` and `percentage_inside`. |
| `window_metrics(data, window, level="landscape", metric=None, name=None, type=None, what=None, *, extras=False, n_threads=None, **options)` | landscapemetrics' `window_lsm`: landscape-level metrics in a moving window (odd side or any odd-sized shape), as `{metric: raster}`. |
| `Landscape.centroids`, `.nearest_neighbours`, `.circumscribing_circles`, `.adjacencies`, `.boundaries`, `.patches`, `.extract_metrics` | landscapemetrics' `get_centroids`, `get_nearestneighbour`, `get_circumscribingcircle`, `get_adjacencies`, `get_boundaries`, `get_patches`, `extract_lsm` (same values). |
| `extract_metrics(data, y, *, extract_id=None, ...)`, `raster_boundaries(data, ...)`, `check_landscape(data)`, `calculate_correlation(metrics, method="pearson")` | landscapemetrics' `extract_lsm` (lists of rasters too), `get_boundaries` on raster values, `check_landscape`, `calculate_correlation`. |
| `mspa_stats(result)` | DataFrame of pixels, % of data area and % of foreground per MSPA class group. |
| `validate_mspa_input(data)` | The input checks on their own. |
| `mspa_colormap(transition=True)` | GTB's MSPA colour table (value → RGBA). |
| `MSPA_CLASSES` | Byte value → class name. |
| `example_data(name="input")`, `EXAMPLE_DATA` | Path and descriptions of the bundled GTB sample maps. |
| `max_threads()` | Threads available to the C++ core. |
| `__version__` | Installed version. |

Parameter correspondence with GuidosToolbox:

| GTB | landschaft | values |
|---|---|---|
| Foreground connectivity (FGconn) | `connectivity` | 8 (default), 4 |
| EdgeWidth | `edge_width` | 1 (default) to 255 (GTB's GUI allows up to 100) |
| Transition | `transition` | `True`/`False`: colour table only, as in GTB |
| Intext | `internal` | `True`/`False` |

## Versioning

landschaft follows [Semantic Versioning](https://semver.org/); the current version is
**0.19.0** (`ls.__version__`). While below 1.0, minor versions may change the API.
Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Performance

`python tools/benchmark.py --exe <mspa executable>` (bundled example maps), on a laptop Ryzen 7 7730U
(8 cores, 15 W). The original's time includes its TIFF I/O. landschaft's is in memory:

| map | FGconn_EdgeWidth | GTB mspa | landschaft, 1 thread | landschaft, 16 threads | identical |
|---|---|---|---|---|---|
| GTB input.tif 1000×1000 | 8_1 | 0.64 s | 0.11 s | 0.09 s | yes |
| GTB input.tif 1000×1000 | 4_20 | 0.97 s | 0.14 s | 0.13 s | yes |
| CORINE forest 2000×2000 | 8_1 | 1.64 s | 0.33 s | 0.29 s | yes |
| CORINE forest 2000×2000 | 8_5 | 2.04 s | 0.37 s | 0.38 s | yes |
| CORINE forest 2000×2000 | 4_20 | 2.48 s | 0.45 s | 0.36 s | yes |

Where the speed comes from, and why threads help less than one might expect:

* Every Euclidean distance transform in MSPA is only compared with a threshold.
  `sqrt(float(d²)) <= s` is monotone in `d²`, so it is replaced by the exact
  integer test `d² <= T`. That test is a dilation by a digital disk, computed
  with clipped column distances and two sweeps per row, instead of a full EDT.
* Hole filling, reconstruction and labelling use run-length union-find instead of
  pixel queues.
* The geodesic distance, the anchored thinning and the watershed are
  *order-dependent* FIFO propagations. Their queue order decides ties, so they
  stay sequential to remain bit-exact. They are the critical path of one image.
* Independent stages run concurrently as tasks. OpenMP data-parallel loops are
  enabled only for maps above 16 Mpx: for smaller maps, the short parallel regions
  cost more than they save, because idle OpenMP workers keep spinning and slow
  down the sequential stages that follow.

For many images the throughput scales like independent processes. On the laptop
above, 16 workers give about 4× (the CPU drops its clock under all-core load).
Expect more on workstations and servers.

## Exactness and tests

landschaft is a line-by-line port of GTB's `fsp.c` (the `mspa` executable, run with
`-disk` as GTB does) and of the miallib routines it calls (`sqedt`, `ced`,
`binOIthin_FIFO`, `wsfah`, `label`, `set_regions`, …). It keeps their integer and
float types, neighbour orders and queue orders, and their quirks, e.g. the
replicated frame of `fm_preproc2` that leaves the last row and column empty.

```bash
pytest                                   # 160+ tests
LANDSCHAFT_MSPA_EXE=/path/to/mspa_win64.exe pytest -m live_reference   # live comparison
```

* `tests/test_reference.py` compares pixel by pixel with outputs of the original
  executable stored in `tests/data/reference` (GTB's sample map with 7 parameter
  sets and 10 synthetic scenes with 4 parameter sets). With
  `LANDSCHAFT_MSPA_EXE` set, it also runs the executable live on random scenes and
  parameters. `tools/make_reference.py` regenerates the stored outputs.
* During development, 1,200 random cases (5 scene types, connectivity 4/8,
  edge widths 1 to 10, internal on/off) plus the 2000×2000 CORINE map with edge
  widths up to 100 were compared with the executable. They showed 0 differing pixels.
* Tools that GTB implements in IDL (Accounting, FAD-MS, FAD Multiscale change, LM
  heatmap and heatmap change) are checked against GTB's own code: `tests/gtb_idl` holds the processing statements copied verbatim from
  `guidostoolbox.pro`, run with [GDL](https://github.com/gnudatalanguage/gdl) (the
  free IDL implementation), with GTB's `spatcon` executable where GTB calls it.
  `tools/make_reference_idl.py` stores digests of their
  outputs in `tests/data/reference/idl_reference.json`; with GDL installed
  (`LANDSCHAFT_GDL_EXE`, PATH or the default install), the tests also run GTB's
  code live on random scenes. Maps and integer statistics are identical; GTB's
  single-precision means and standard deviations agree to 1e-4 (landschaft
  computes them in double precision).
* Landscape metrics are compared with the R package landscapemetrics:
  `tools/make_reference_lsm.py` stores every value of `calculate_lsm` for 14 maps
  (its example data, synthetic maps with NA cells, non-square cells and
  degenerate layouts) and 5 option sets in `tests/data/reference/lsm`, and with
  `Rscript` available the tests also run landscapemetrics live on random maps. All
  values are the same bits, except `circle` (randomised in landscapemetrics),
  `gyrate`, `pafrac` and a few `enn` values, which agree to a relative 1e-10: sums,
  means and standard deviations reproduce R's 80-bit accumulation, and `log`/`sqrt`
  R's x87 functions.
* `tests/test_primitives.py` checks each C++ primitive against SciPy or brute
  force. `tests/test_mspa_semantics.py` checks the classes on hand-made patterns.
  The I/O and API tests cover validation, georeferencing, colour tables,
  xarray/dask and batch processing.

Set `LANDSCHAFT_PROFILE=1` to print the time of every MSPA stage.

## Limitations

* MSPA is a global operator: the whole map must fit in memory (≈30 bytes/pixel).
  There is no tiling yet. GTB's "MSPA tiling" is approximate by design.
* The original stores geodesic offsets in 16-bit integers and distances in 32 bits.
  landschaft reproduces them. For maps with a side of 46,000 px or more, the fast
  distance path is disabled and the original arithmetic is used.
* Eigen is not used yet: MSPA's hot paths are queue propagations and byte
  operations that the compiler already vectorises. Eigen will be considered for
  GTB tools with floating-point kernels (moving windows, distances, costs).

## License and credits

GPL-3.0-or-later. The example maps in `src/landschaft/data` come from the
[GTB repository](https://github.com/ec-jrc/GTB) (GPL-3.0). landschaft ports code from GuidosToolbox and miallib, © European
Union (Joint Research Centre), authors Pierre Soille, Peter Vogt et al., both
GPL-3.0-or-later. GTB's `mspa` notes that its interface is made available for
non-commercial purposes. Contact the JRC for commercial use of MSPA.

If you use MSPA, cite:

* Soille, P. & Vogt, P. (2009). Morphological segmentation of binary patterns.
  *Pattern Recognition Letters* 30(4), 456–459. doi:10.1016/j.patrec.2008.10.015
* Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object
  analysis. *European Journal of Remote Sensing* 50(1), 352–361.
  doi:10.1080/22797254.2017.1330650
