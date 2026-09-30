# Surface metrics

Measures of the texture of a continuous raster (NDVI, elevation, canopy height,
any gradient) that need no classification into patches: how much it varies,
how its heights are distributed, how its trend can be removed. They are the
areal surface texture parameters of ISO 25178, brought to landscape ecology as
*gradient surface metrics*, and live in the `ls.surface` module.

Every value is the one the R package geodiv gives, bit for bit, so studies done
in R can be repeated and extended here.

<!-- no-run -->
```python
import landschaft as ls

ndvi = ls.surface.example_data("orforest")
z = ls.surface.remove_plane(ndvi)       # residuals from the best trend surface
ls.surface.metrics(z)                   # every metric in one table
```

## All metrics at once

::: landschaft.surface.metrics

::: landschaft.surface.list_metrics

## Amplitude

::: landschaft.surface.sa

::: landschaft.surface.sq

::: landschaft.surface.ssk

::: landschaft.surface.sku

::: landschaft.surface.sv

::: landschaft.surface.sph

::: landschaft.surface.smean

## Trend surfaces

::: landschaft.surface.fit_plane

::: landschaft.surface.best_fit_plane

::: landschaft.surface.remove_plane

## Example data

::: landschaft.surface.example_data
