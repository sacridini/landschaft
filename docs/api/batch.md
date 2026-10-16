# Large data & batch

Many rasters at once, maps larger than memory, and tiled MSPA. Tutorial:
[large data and batch processing](../tutorials/large-data.md).

Most tools also accept a [`RasterCollection`](data.md#landschaft.RasterCollection)
or a lazy stack directly (e.g. `ls.mspa(ls.load_raster("tiles/*.tif"),
output_dir="out")`), and [`Landscape`](landscape-metrics.md#landschaft.Landscape)
reads large maps in blocks (`block_rows`).

## Folders of rasters

::: landschaft.batch

::: landschaft.mspa_batch

::: landschaft.BatchResult

### Tools of `batch`

`ls.BATCH_TOOLS` lists the names `batch` accepts; any function taking a raster
as its first argument works too.

`accounting`, `clustering_fac`, `conefor_inputs`, `contagion_pff`, `contortion`, `distance_to_resistance`, `euclidean_distance`, `fad`, `fad_multiscale`, `fos`, `gray_spatial_convolution`, `gtn_components`, `gtn_importance`, `influence_zones`, `landscape_mosaic`, `landscape_mosaic_heatmap`, `landscape_mosaic_stats`, `mspa`, `parcellation`, `proximity`, `reclassify`, `restoration_status`, `spa`, `spatial_entropy`, `spatial_entropy_map`, `landscape_metrics`, `window_metrics`.

## Several computers

`on=ls.helpers()` runs the tasks of `batch`, `mspa_batch`,
`simulate.scenarios`, `simulate.calibrate`, `sdm.fit` (its cross-validation),
`uncertainty.monte_carlo`, the trade-off calibrations and replacement importance
of `ls.prioritize`, `parallel.map` and `parallel.tiled` on this computer and on
the computers of its group running `landschaft worker`. Tutorial:
[several computers](../tutorials/many-computers.md).

::: landschaft.parallel.helpers

::: landschaft.parallel.Helpers

::: landschaft.parallel.map

::: landschaft.parallel.tiled

::: landschaft.parallel.call

::: landschaft.parallel.param

`ls.parallel.TOOLS` lists the tools `map`, `tiled` and `monte_carlo` can run by name:
the analyses of `ls` (`"fad"`, `"mspa"`, ...), `ls.surface` (`"surface.geomorphons"`),
`ls.hydro`, `ls.services`, `ls.nlm` and `ls.circuit`.

::: landschaft.parallel.run_worker

::: landschaft.parallel.invite_helpers

::: landschaft.parallel.list_helpers

::: landschaft.parallel.leave_group

::: landschaft.parallel.open_firewall

### The `landschaft` command

| Command | What it does |
|---|---|
| `landschaft helpers invite [--minutes 10]` | show a code with which other computers join this one's group |
| `landschaft worker --join CODE` | join a group (the first time) and lend this computer |
| `landschaft worker [--cores N] [--connect HOST[:PORT]] [--map SRC=DST] [--cache-gb 20]` | lend this computer until Ctrl+C |
| `landschaft helpers list` | the helpers waiting for work on the local network |
| `landschaft helpers firewall` | Windows: let helpers reach this computer (TCP 47811, UDP 47810, local subnet) |
| `landschaft helpers leave` | take this computer out of its group |

## Maps larger than memory

::: landschaft.mspa_tiled

::: landschaft.tile_layout
