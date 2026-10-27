# Ecosystem services

Models of [InVEST](https://naturalcapitalproject.stanford.edu/software/invest)
(the Natural Capital Project's natcap.invest, Apache-2.0), reimplemented in C++
from land use maps and parameter tables: their maps, bit for bit where the
arithmetic permits. A simulated land use map (`ls.simulate`) or a series of
scenarios goes straight in.

The input rasters may come on different grids (cell sizes, origins, extents,
CRS): each model aligns them first as InVEST does, on the grid of its reference
raster (the DEM, or the land use map), and the models with watersheds or an AOI
clip the maps to the polygons' bounds. Inputs already on that grid are used as
they are, and the results come on the grid they were computed on.

## Carbon storage and sequestration

::: landschaft.services.carbon

::: landschaft.services.Carbon

::: landschaft.services.valuation_constant

## Annual water yield and hydropower

::: landschaft.services.water_yield

::: landschaft.services.WaterYield

## Seasonal water yield

::: landschaft.services.seasonal_water_yield

::: landschaft.services.SeasonalWaterYield

## Sediment delivery ratio

::: landschaft.services.sdr

::: landschaft.services.SDR

## Nutrient delivery ratio

::: landschaft.services.ndr

::: landschaft.services.NDR

## Habitat quality and rarity

::: landschaft.services.habitat_quality

::: landschaft.services.HabitatQuality

## Crop pollination

::: landschaft.services.pollination

::: landschaft.services.Pollination

## Coastal vulnerability

::: landschaft.services.coastal_vulnerability

::: landschaft.services.CoastalVulnerability
