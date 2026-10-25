# Configuration

Landschaft has no configuration file: a handful of settings, passed to the
functions that use them or set once for the Python session.

## Threads

Every function that runs in parallel uses **all processor cores but one** by
default, so that the computer stays responsive while it works; inside a batch
job of a cluster (Slurm, PBS, LSF), or when the process may run on fewer
processors than the computer has, it uses all of those it may use. You can
change this per call or for the whole session:

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

Without `n_threads` or `ls.set_default_threads`, the count comes from the
environment, from the first that applies:

| | |
|---|---|
| `LANDSCHAFT_NUM_THREADS` | this process and the processes it starts |
| a dask.distributed worker | the worker's threads |
| `OMP_NUM_THREADS` | as for any OpenMP program |
| a batch job (Slurm, PBS, LSF) | the job's processors (`SLURM_CPUS_PER_TASK`...) |
| the computer | the processors this process may run on (its affinity and CPU quota), all but one |

`landschaft doctor` (or `ls.doctor()`) prints the count and where it came
from; see [on a cluster](../tutorials/hpc-slurm.md).

!!! note "Jobs that run side by side"
    If you start several Python processes at once on one computer, give each a
    share of the cores (`LANDSCHAFT_NUM_THREADS=4 python script.py`, or
    `ls.set_default_threads(4)` at the top of each script), so that together
    they do not use more cores than the machine has.

## Other computers

Functions that run many independent tasks (`batch`, `mspa_batch`,
`simulate.scenarios`, `simulate.calibrate`) take `on=ls.helpers()` and share
the tasks with the computers of a group running `landschaft worker`, with the
same results; see [several computers](../tutorials/many-computers.md). The
group's secret is kept in `%APPDATA%\landschaft` (Windows) or
`~/.config/landschaft` (`LANDSCHAFT_CONFIG_DIR` overrides it).

## Downloads and the cache

A few tools download reference data the first time they need it (each file
pinned by its checksum) into `%LOCALAPPDATA%\landschaft\cache` (Windows),
`~/Library/Caches/landschaft` (macOS) or `~/.cache/landschaft` (Linux;
`XDG_CACHE_HOME` moves it). `LANDSCHAFT_CACHE` puts it elsewhere, and
`landschaft fetch` downloads everything beforehand, for computers without
internet access.

## Memory

| Setting | Where | What it does |
|---|---|---|
| `max_memory="16GB"` | `ls.batch`, `ls.mspa` on a collection | rasters are started only while their estimated memory fits the budget |
| `block_rows=...` | `ls.Landscape`, `ls.landscape_metrics` | landscape metrics of a file read in blocks of rows, for rasters larger than memory |
| `tile_size=...` | `ls.mspa_tiled` | MSPA of rasters larger than memory, tile by tile |
| `memory_limit=...` | `ls.omniscape` | caps the memory of the moving windows (fewer threads for very large windows) |

## Output files

`ls.write_raster(data, "out.tif")` writes a compressed, tiled GeoTIFF with
the georeferencing, NoData and band names of `data` and, for class maps, their
colour table (`colormap="auto"`); pass `colormap=None` to leave it out or a
`{value: (r, g, b, a)}` dictionary for your own colours. It chooses what the
data need: NaN as the NoData of floats (or `nodata=-9999`), `uint8` for
`True`/`False` maps, the smallest integer type for labels, the format from the
extension (`.tif`, `.asc`, `.img`...; `driver="COG"` for a Cloud-Optimised
GeoTIFF). Dask arrays are written block by block, and missing folders are
created. Tables are pandas DataFrames; large metric tables can be streamed to
Parquet or CSV (see the landscape metrics tutorial).

## Environment variables

| Variable | Effect |
|---|---|
| `LANDSCHAFT_CIRCUIT_AVX2=0` | use the generic build of the circuit solver instead of the AVX2 one (for comparisons; the default picks AVX2 when the processor has it) |
| `GDAL_*` | GDAL's own settings (cache size, number of threads for decoding...), honoured by all file reading |

## Reproducibility

Landschaft has no random numbers: the same input and options give the same
output on the same platform, whatever the number of threads. Record the
version (`ls.__version__`) with your results.
