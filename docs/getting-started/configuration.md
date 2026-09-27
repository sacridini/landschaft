# Configuration

Landschaft has no configuration file: a handful of settings, passed to the
functions that use them or set once for the Python session.

## Threads

Every function that runs in parallel uses **all processor cores but one** by
default, so that the computer stays responsive while it works. You can change
this per call or for the whole session:

```python
import landschaft as ls

ls.max_threads()          # processors available, e.g. 20
ls.default_threads()      # threads used when you do not say: 19

forest = ls.load_raster(ls.example_data("input"))
ls.mspa(forest, n_threads=4)       # this call only: 4 threads
ls.set_default_threads(8)          # from now on, every call: 8 threads
ls.set_default_threads(None)       # back to all cores but one
```

`n_threads=None` or `n_threads=0` always means the default. The results never
depend on the number of threads: the same input gives the same bits with 1 or
64 threads.

Functions that process many rasters at once (`ls.batch`, `ls.mspa` on a
collection, `ls.mspa_tiled`) also take `n_workers`, the number of rasters or
tiles in flight (`ls.batch` also `threads_per_worker`); by default the workers
share the same cores (all but one).

!!! note "Jobs that run side by side"
    If you start several Python processes at once (e.g. on a cluster node),
    give each a share of the cores with `ls.set_default_threads(n)` at the top
    of each script, so that together they do not use more cores than the
    machine has.

## Memory

| Setting | Where | What it does |
|---|---|---|
| `max_memory="16GB"` | `ls.batch`, `ls.mspa` on a collection | rasters are started only while their estimated memory fits the budget |
| `block_rows=...` | `ls.Landscape`, `ls.landscape_metrics` | landscape metrics of a file read in blocks of rows, for rasters larger than memory |
| `tile_size=...` | `ls.mspa_tiled` | MSPA of rasters larger than memory, tile by tile |
| `memory_limit=...` | `ls.omniscape` | caps the memory of the moving windows (fewer threads for very large windows) |

## Output files

`ls.write_raster(data, "out.tif")` writes a compressed GeoTIFF with the
georeferencing of `data` and, for class maps, their colour table
(`colormap="auto"`); pass `colormap=None` to leave it out or a
`{value: (r, g, b, a)}` dictionary for your own colours. Tables are pandas
DataFrames; large metric tables can be streamed to Parquet or CSV (see the
landscape metrics tutorial).

## Environment variables

| Variable | Effect |
|---|---|
| `LANDSCHAFT_CIRCUIT_AVX2=0` | use the generic build of the circuit solver instead of the AVX2 one (for comparisons; the default picks AVX2 when the processor has it) |
| `GDAL_*` | GDAL's own settings (cache size, number of threads for decoding...), honoured by all file reading |

## Reproducibility

Landschaft has no random numbers: the same input and options give the same
output on the same platform, whatever the number of threads. Record the
version (`ls.__version__`) with your results.
