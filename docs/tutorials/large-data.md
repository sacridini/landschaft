# Large data & batch

Real projects rarely stop at one map: a country cut into tiles, the same
region every year, a map too large for memory. This tutorial shows how to
process many rasters in parallel within a memory budget, how to handle maps
larger than memory, and how to stream very large tables to disk.

**You will learn to:**

- open a folder of rasters as a collection and process it in parallel;
- run any tool on many files with `batch`, with outputs named and logged;
- run MSPA on maps larger than memory, tile by tile;
- compute landscape metrics of a file larger than memory;
- process time stacks lazily with dask;
- control threads and memory.

Let us first make a small folder of rasters to work with, by cutting the
example map into four tiles:

```python
from pathlib import Path
import landschaft as ls

forest = ls.load_raster(ls.example_data("input"))
Path("tiles").mkdir(exist_ok=True)
for row in (0, 500):
    for col in (0, 500):
        ls.write_raster(forest[row:row + 500, col:col + 500], f"tiles/forest_{row}_{col}.tif")
```

## Collections

`load_raster` given a folder, a glob pattern or a list returns a
**collection**: only the file headers are read now, the pixels when each file
is processed.

```python
maps = ls.load_raster("tiles/*.tif")
maps
```

Every file is checked on the way in; `errors="skip"` leaves unusable files
out (listed in `maps.skipped`) instead of stopping. MSPA processes a
collection directly, one raster per worker thread:

```python
results = ls.mspa(maps, edge_width=2, output_dir="mspa_out")
results[0]
```

With `output_dir`, each result is written as soon as it is ready (so memory
stays bounded) and a `BatchResult` per file records the output path, the run
time and any error. Without it, the maps are returned in a list.

## Any tool on many files: batch

```python
runs = ls.batch("fad", "tiles", "fad_out", window_size=27)
[run.output.name for run in runs]
```

`batch` runs a tool on every raster of a folder, pattern, list or collection
and writes everything the tool returns, named after the input:

| Output | File |
|---|---|
| maps | `<name>_<tool>_<field>.tif` (with the colour table where there is one) |
| tables | `<name>_<tool>_<field>.csv`, or `.parquet` with `table_format="parquet"` |
| report | `<name>_<tool>.txt` (the result's `summary()`) |
| record | `<name>_<tool>.json`: parameters, outputs, scalar results, run time |

The tools are listed in `ls.BATCH_TOOLS` (MSPA, fragmentation, accounting,
distance, landscape metrics...); any function that takes a raster as its first
argument works too. Tool options follow as keywords (`window_size=27`).
Inputs whose record already exists are skipped, so an interrupted run resumes
where it stopped (`overwrite=True` recomputes).

| Option | Default | Meaning |
|---|---|---|
| `n_workers` | all cores but one | rasters in flight |
| `threads_per_worker` | 1 | threads inside each raster: best throughput for many small rasters; raise it for a few large ones |
| `max_memory` | none | e.g. `"16GB"`: a raster starts only when its estimated memory fits |
| `raise_errors` | False | stop at the first error instead of recording it |

## MSPA of maps larger than memory

`mspa_tiled` cuts a GeoTIFF into overlapping tiles, runs MSPA on each and
writes the result tile by tile: memory depends on the tile size, not on the
map size.

```python
path = ls.mspa_tiled(ls.example_data("input"), "input_mspa_tiled.tif",
                     tile_size=600, overlap=200)
path
```

Tiles are `tile_size` pixels wide including the `overlap` shared with their
neighbours; the default (6,500 and 2,000) suits most machines. Features larger
than the overlap can come out slightly differently near the seams than on the
whole map: here, with tiny tiles, 10 of a million pixels differ; with 300
pixels of overlap, none. `n_workers` processes several tiles at once.

## Landscape metrics of files larger than memory

Give the file path and `block_rows`: the raster is read in blocks of rows and
only the table of patches is kept. The results are identical to reading the
whole map:

```python
big = ls.Landscape(ls.example_data("clc"), block_rows=256)
big.out_of_core, big.class_metrics(["ca", "np", "ed"]).head(3)
```

`block_rows="auto"` picks about four million cells per block. A few
features need the whole raster (patch maps, `to_raster`); everything else
works.

For tables with millions of patches, write them in chunks instead of building
them in memory:

```python
big.write_patch_metrics("patches.parquet", ["area", "perim", "shape"])
```

## Time stacks with dask

Rasters that share the same grid (e.g. the same region at several dates)
form a lazy stack with one dask chunk per file; MSPA then runs on the slices
in parallel when the result is computed:

```python
Path("years").mkdir(exist_ok=True)
ls.write_raster(forest, "years/forest_2010.tif")
ls.write_raster(forest, "years/forest_2020.tif")

stack = ls.load_raster("years/*.tif", stack=True)   # (file, y, x), lazy
morph_stack = ls.mspa(stack)                       # still lazy
morph_stack.shape
```

`morph_stack.compute()` (or `.to_netcdf(...)`) runs it. The `dask` extra is
needed.

## Threads and memory

- Every parallel function uses all cores but one by default; see
  [configuration](../getting-started/configuration.md#threads).
- Collections and `batch`: `n_workers × threads_per_worker` threads; a
  memory budget with `max_memory`.
- Omniscape: `memory_limit=` caps the memory of its windows.

## Next steps

- The [API reference](../api/batch.md) lists every option.
- The [benchmarks](../benchmarks.md) show the speed you can expect.
