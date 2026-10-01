# Data & configuration

Reading, preparing and writing rasters, vectors and tables, saving and
plotting results, colour tables, collections of files and the thread settings shared by every
tool. The page [Loading and saving data](../getting-started/data.md) shows
them together.

!!! example "Every example runs"
    The examples on these pages use the maps that ship with Landschaft
    (`ls.example_data`) and are run as tests, outputs included.

## Rasters

::: landschaft.load_raster

::: landschaft.write_raster

::: landschaft.match_grid

::: landschaft.clip

::: landschaft.reclassify

## Vectors

::: landschaft.load_vector

::: landschaft.write_vector

::: landschaft.rasterize

## Values by zone and at points

::: landschaft.zonal_stats

::: landschaft.extract

## Tables

::: landschaft.load_table

::: landschaft.write_table

## Saving results

::: landschaft.save

Every result object (of `fos`, `restoration_status`, `zonation`...) also has
a `save(folder)` method that calls `save`.

## Plotting

::: landschaft.plot

::: landschaft.Plot
    options:
      members: [maps, charts, figure, show, save, to_html]

The page [Plotting](../getting-started/plotting.md) shows the plots of every
kind of data, and how to choose their colours.

## Example data

::: landschaft.example_data

## Example data

`ls.EXAMPLE_DATA` describes the habitat and land-cover rasters; the
examples of the other modules are listed under `example_data` above:

| Name | Content |
|---|---|
| `"input"` | habitat map, 1000 × 1000 cells, 0 missing, 1 background, 2 forest (EPSG:4326) |
| `"clc"` | CORINE land cover excerpt, 2000 × 2000 cells of 25 m (EPSG:3035), classes 1–44 |
| `"gscinput"` | percentage map, 1990 × 1289 cells of 2430 m (EPSG:5070), 0–100, 255 missing |

## Collections of rasters

::: landschaft.RasterCollection
    options:
      members: [load, to_stack]

::: landschaft.RasterInfo
    options:
      members: false

## Checking inputs

::: landschaft.validate_mspa_input

::: landschaft.validate_fad_input

::: landschaft.InvalidInputError
    options:
      members: false

## Colour tables

Every class map has a colour table: a `{value: (red, green, blue, alpha)}`
dictionary with 256 entries, embedded by `write_raster`. Each is also
available as a constant (`ls.MSPA_CLASSES` names the MSPA codes;
`ls.FAD_COLORMAP`, `ls.LM_COLORMAP`... are the tables themselves).

::: landschaft.mspa_colormap

::: landschaft.fad_colormap

::: landschaft.fad5_colormap

::: landschaft.lm_colormap

::: landschaft.accounting_colormap

::: landschaft.change_colormap

::: landschaft.foschange_colormap

::: landschaft.dist_colormap

::: landschaft.entropy_colormap

::: landschaft.contortion_colormap

::: landschaft.resistance_colormap

::: landschaft.gtn_components_colormap

::: landschaft.gtn_importance_colormap

## Threads

::: landschaft.default_threads

::: landschaft.set_default_threads

::: landschaft.max_threads
