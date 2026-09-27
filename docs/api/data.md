# Data & configuration

Reading, preparing and writing rasters, colour tables, collections of files
and the thread settings shared by every tool.

!!! example "Every example runs"
    The examples on these pages use the maps that ship with landschaft
    (`ls.example_data`) and are run as tests, outputs included.

## Reading and writing

::: landschaft.load_raster

::: landschaft.write_raster

::: landschaft.reclassify

::: landschaft.example_data

## Example data

`ls.EXAMPLE_DATA` maps the names accepted by `example_data` to a
description:

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
