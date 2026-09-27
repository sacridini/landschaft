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

## Maps larger than memory

::: landschaft.mspa_tiled

::: landschaft.tile_layout
