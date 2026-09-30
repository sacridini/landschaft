# Surface metrics

Measures of the texture of a continuous raster (NDVI, elevation, canopy height,
any gradient) that need no classification into patches: how much it varies,
how its heights are distributed, how its trend can be removed. They are the
areal surface texture parameters of ISO 25178, brought to landscape ecology as
*gradient surface metrics*, and live in the `ls.surface` module.

Every value is the one the R package geodiv gives, bit for bit, so studies done
in R can be repeated and extended here. Where geodiv departs from ISO 25178
(`sdq6`, `sdr`, the valleys of `find_valleys` and `s10z`), `convention="iso"`
gives the standard's definition instead.

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

## Slopes and developed area

::: landschaft.surface.sdq

::: landschaft.surface.sdq6

::: landschaft.surface.sdr

::: landschaft.surface.surface_area

::: landschaft.surface.flat_area

## Peaks and valleys

::: landschaft.surface.find_peaks

::: landschaft.surface.find_valleys

::: landschaft.surface.sds

::: landschaft.surface.ssc

::: landschaft.surface.s10z

## Bearing area curve

::: landschaft.surface.bearing_area

::: landschaft.surface.height_ba

::: landschaft.surface.sdc

::: landschaft.surface.sbi

::: landschaft.surface.svi

::: landschaft.surface.sci

::: landschaft.surface.sk

::: landschaft.surface.spk

::: landschaft.surface.svk

## Autocorrelation, orientation and wavelength

::: landschaft.surface.aacf

::: landschaft.surface.scl

::: landschaft.surface.stxr

::: landschaft.surface.std

::: landschaft.surface.srw

::: landschaft.surface.sfd

## Trend surfaces

::: landschaft.surface.fit_plane

::: landschaft.surface.best_fit_plane

::: landschaft.surface.remove_plane

## Example data

::: landschaft.surface.example_data
