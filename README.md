# guidospy

Fast Python reimplementation of [GuidosToolbox (GTB)](https://forest.jrc.ec.europa.eu/en/activities/lpa/gtb/)
raster image analysis, with a C++17/OpenMP core and a small numpy / xarray / dask
/ rasterio API.

**Status (0.1.0):** Morphological Spatial Pattern Analysis (MSPA) is available and
produces **bit-identical** results to GTB's `mspa` v2.3 executable, 5 to 7 times
faster on one thread. The other GTB tools come next.

```python
import guidospy as gp

data = gp.load_raster("forest.tif")        # reads and checks the 0/1/2 coding
result = gp.mspa(data, edge_width=2)        # georeferenced xarray.DataArray
gp.write_raster(result, "forest_mspa.tif")  # GeoTIFF with the GTB colour table
print(gp.mspa_stats(result))                # class areas, as in GTB
```

## Installation

guidospy builds from source with a C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
and OpenMP if available. Recommended with conda:

```bash
conda create -n guidospy -c conda-forge python=3.12 numpy pandas rasterio rioxarray xarray dask \
      scikit-build-core pybind11 cmake ninja pytest scipy
conda activate guidospy
pip install --no-build-isolation -e .
```

On Windows, with only the Visual Studio *Build Tools* installed, pick the generator:
`set CMAKE_GENERATOR=Visual Studio 17 2022` (or `Visual Studio 18 2026`) before `pip install`.

## Usage

### Example data

GTB's sample maps ship with the package, so everything below can be tried right away:

```python
gp.EXAMPLE_DATA                       # name -> description
data = gp.load_raster(gp.example_data("input"))                 # MSPA-ready 1000 x 1000 map
clc = gp.load_raster(gp.example_data("clc"), validate=None)     # CORINE land cover, 2000 x 2000
forest = gp.reclassify(clc, foreground=[23, 24, 25])            # forests as foreground
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
lc = gp.load_raster("corine.tif", validate=None)
forest = gp.reclassify(lc, foreground=[23, 24, 25], missing=[48])  # nodata -> missing too
tree_cover = gp.reclassify(gp.load_raster("cover.tif", validate=None), foreground=lambda a: a >= 30)
```

### MSPA

```python
result = gp.mspa(
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
gp.mspa(stack).to_netcdf("mspa_stack.nc")
```

Output classes (`gp.MSPA_CLASSES`):

| value | class | value | class |
|---|---|---|---|
| 17 | Core | 65 / 67 / 69 | Loop (/ in edge / in perforation) |
| 9 | Islet | 33 / 35 / 37 | Bridge (/ in edge / in perforation) |
| 5 | Perforation | 1 | Branch |
| 3 | Edge | 0 | Background |
| 100 | Core-opening | 220 | Border-opening |
| 129 | Missing | +100 | the same class inside a core-opening (`internal=True`) |

### Thousands of images

```python
results = gp.mspa_batch(sorted(Path("tiles").glob("*.tif")), "out/", edge_width=2)
failed = [r for r in results if not r.ok]
```

Each worker thread loads, processes and writes one file
(`<name>_<conn>_<edgewidth>_<transition>_<internal>.tif`, as GTB). The C++ core
releases the GIL, so threads run truly in parallel. By default one worker per
core with one thread each, which gives the best throughput. Memory: about
30 bytes per pixel per worker.

## API reference

All functions are available from the top-level package (`import guidospy as gp`);
their docstrings (`help(gp.mspa)`) give every detail.

| function | purpose |
|---|---|
| `load_raster(source, *, band=None, chunks=None, validate="mspa")` | Read one band as a `(y, x)` DataArray with CRS/transform. By default checks the MSPA coding and raises `InvalidInputError` otherwise. `chunks` gives a lazy dask array. |
| `write_raster(data, path, *, crs=None, transform=None, nodata=None, colormap="auto", compress="lzw", description=None)` | Write a 2-D array or DataArray to GeoTIFF. MSPA results get the GTB colour table and their parameters as tags. |
| `reclassify(data, foreground, *, missing=None, background=None)` | Build a 0/1/2 MSPA input from class values or a boolean function; NaN and nodata become missing. |
| `mspa(data, *, connectivity=8, edge_width=1, transition=True, internal=True, n_threads=None, validate=True)` | Run MSPA on numpy, dask or xarray data (2-D or stacks). |
| `mspa_batch(inputs, output_dir, *, ..., n_workers=None, threads_per_worker=1, overwrite=False, raise_errors=False)` | Process many files in parallel; returns one `BatchResult` (`input`, `output`, `seconds`, `error`, `ok`) per file. |
| `mspa_stats(result)` | DataFrame of pixels, % of data area and % of foreground per MSPA class group. |
| `validate_mspa_input(data)` | The input checks on their own. |
| `mspa_colormap(transition=True)` | GTB's MSPA colour table (value → RGBA). |
| `MSPA_CLASSES` | Byte value → class name. |
| `example_data(name="input")`, `EXAMPLE_DATA` | Path and descriptions of the bundled GTB sample maps. |
| `max_threads()` | Threads available to the C++ core. |
| `__version__` | Installed version. |

Parameter correspondence with GuidosToolbox:

| GTB | guidospy | values |
|---|---|---|
| Foreground connectivity (FGconn) | `connectivity` | 8 (default), 4 |
| EdgeWidth | `edge_width` | 1 (default) to 255 (GTB's GUI allows up to 100) |
| Transition | `transition` | `True`/`False`: colour table only, as in GTB |
| Intext | `internal` | `True`/`False` |

## Versioning

guidospy follows [Semantic Versioning](https://semver.org/); the current version is
**0.1.0** (`gp.__version__`). While below 1.0, minor versions may change the API.
Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Performance

`python tools/benchmark.py --exe <mspa executable>` (bundled example maps), on a laptop Ryzen 7 7730U
(8 cores, 15 W). The original's time includes its TIFF I/O. guidospy's is in memory:

| map | FGconn_EdgeWidth | GTB mspa | guidospy, 1 thread | guidospy, 16 threads | identical |
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

guidospy is a line-by-line port of GTB's `fsp.c` (the `mspa` executable, run with
`-disk` as GTB does) and of the miallib routines it calls (`sqedt`, `ced`,
`binOIthin_FIFO`, `wsfah`, `label`, `set_regions`, …). It keeps their integer and
float types, neighbour orders and queue orders, and their quirks, e.g. the
replicated frame of `fm_preproc2` that leaves the last row and column empty.

```bash
pytest                                   # 160+ tests
GUIDOSPY_MSPA_EXE=/path/to/mspa_win64.exe pytest -m live_reference   # live comparison
```

* `tests/test_reference.py` compares pixel by pixel with outputs of the original
  executable stored in `tests/data/reference` (GTB's sample map with 7 parameter
  sets and 10 synthetic scenes with 4 parameter sets). With
  `GUIDOSPY_MSPA_EXE` set, it also runs the executable live on random scenes and
  parameters. `tools/make_reference.py` regenerates the stored outputs.
* During development, 1,200 random cases (5 scene types, connectivity 4/8,
  edge widths 1 to 10, internal on/off) plus the 2000×2000 CORINE map with edge
  widths up to 100 were compared with the executable. They showed 0 differing pixels.
* `tests/test_primitives.py` checks each C++ primitive against SciPy or brute
  force. `tests/test_mspa_semantics.py` checks the classes on hand-made patterns.
  The I/O and API tests cover validation, georeferencing, colour tables,
  xarray/dask and batch processing.

Set `GUIDOSPY_PROFILE=1` to print the time of every MSPA stage.

## Limitations

* MSPA is a global operator: the whole map must fit in memory (≈30 bytes/pixel).
  There is no tiling yet. GTB's "MSPA tiling" is approximate by design.
* The original stores geodesic offsets in 16-bit integers and distances in 32 bits.
  guidospy reproduces them. For maps with a side of 46,000 px or more, the fast
  distance path is disabled and the original arithmetic is used.
* Eigen is not used yet: MSPA's hot paths are queue propagations and byte
  operations that the compiler already vectorises. Eigen will be considered for
  GTB tools with floating-point kernels (moving windows, distances, costs).

## License and credits

GPL-3.0-or-later. The example maps in `src/guidospy/data` come from the
[GTB repository](https://github.com/ec-jrc/GTB) (GPL-3.0). guidospy ports code from GuidosToolbox and miallib, © European
Union (Joint Research Centre), authors Pierre Soille, Peter Vogt et al., both
GPL-3.0-or-later. GTB's `mspa` notes that its interface is made available for
non-commercial purposes. Contact the JRC for commercial use of MSPA.

If you use MSPA, cite:

* Soille, P. & Vogt, P. (2009). Morphological segmentation of binary patterns.
  *Pattern Recognition Letters* 30(4), 456–459. doi:10.1016/j.patrec.2008.10.015
* Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object
  analysis. *European Journal of Remote Sensing* 50(1), 352–361.
  doi:10.1080/22797254.2017.1330650
