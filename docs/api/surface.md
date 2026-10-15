# Surface metrics

Measures of the texture of a continuous raster (NDVI, elevation, canopy height,
any gradient) that need no classification into patches: how much it varies,
how its heights are distributed, how its trend can be removed. They are the
areal surface texture parameters of ISO 25178, brought to landscape ecology as
*gradient surface metrics*, and live in the `ls.surface` module.

The metrics follow the R package geodiv, bit for bit wherever geodiv is
right; where it errs (Sq and the moments built on it, the seven-point slopes,
the valleys, the summit curvature, the areas of the bearing area curve, the
centre of its Fourier images, the fractal dimension's prisms, the choice of
trend surface), Landschaft computes what the method intends, and each
function's notes say what differs.

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

## Texture images

::: landschaft.surface.window_metrics

## Terrain

Indices of a DEM's shape as the R package spatialEco computes them (over
terra's slope and aspect), corrected where it errs.

::: landschaft.surface.terrain

::: landschaft.surface.tpi

::: landschaft.surface.tri

::: landschaft.surface.vrm

::: landschaft.surface.srr

::: landschaft.surface.dissection

::: landschaft.surface.curvature

::: landschaft.surface.spherical_sd

::: landschaft.surface.hsp

::: landschaft.surface.trasp

::: landschaft.surface.hli

## Spatial interaction

::: landschaft.surface.gravity

## Trend surfaces

::: landschaft.surface.fit_plane

::: landschaft.surface.best_fit_plane

::: landschaft.surface.remove_plane

## Example data

::: landschaft.surface.example_data
