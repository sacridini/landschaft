# Data on demand

Rasters downloaded onto the grid of a map: the climate of WorldClim 2.1 and
CHELSA 2.1 (now and under the CMIP6 scenarios), the Copernicus DEM (30 or 90
m), SoilGrids 2.0, MapBiomas' land cover of Brazil, Hydrography90m's streams,
sub-catchments and terrain, the scenes of any STAC catalogue (a median of the
items; a function to sign the links where they need a token) or any raster a URL or a path
names; and the environment of Hydrography90m's sub-catchments (the climate,
land cover and soil Environment90m sums up for each) as a table or a map. GDAL
reads only the part of each file the grid covers, warps it onto the grid of
`like` (or of an `extent` in longitudes and latitudes and a `cellsize`), and
the result is kept in the cache (`LANDSCHAFT_CACHE`): a second call reads the
file it wrote. Classes are resampled by their most common cell, values
bilinearly. The `ls.download` module.

<!-- no-run -->
```python
import landschaft as ls

forest = ls.load_raster("forest.tif")
clim = ls.download.worldclim(forest, layers=[1, 12])       # bio01 and bio12 on the forest's grid
future = ls.download.chelsa(forest, gcm="MPI-ESM1-2-HR", ssp=370, period="2041-2070")
dem = ls.download.copernicus_dem(forest, resolution=30)
soc = ls.download.soilgrids(forest, property="soc", depth=["0-5cm", "5-15cm"])
cover = ls.download.mapbiomas(forest, years=[1985, 2023])
basins = ls.download.hydrography90m(forest, variable="sub_catchment")

s2 = ls.download.stac("https://earth-search.aws.element84.com/v1", "sentinel-2-l2a", forest,
                      assets=["red", "nir"], datetime="2024-07-01/2024-08-31",
                      query={"eo:cloud_cover": {"lt": 10}})
ndvi = (s2.sel(band="nir") - s2.sel(band="red")) / (s2.sel(band="nir") + s2.sel(band="red"))

env = ls.download.environment90m(basins, ["bio01_1981-2010_observed", "c10_2020"], stats=["mean", "sd"])
env.table                                                   # a row per sub-catchment
```

!!! note "Online"
    These functions need the internet the first time they fetch a layer; the
    examples that download are not run as tests.

## Climate

::: landschaft.download.worldclim

::: landschaft.download.chelsa

## Terrain, soils and land cover

::: landschaft.download.copernicus_dem

::: landschaft.download.soilgrids

::: landschaft.download.mapbiomas

## Hydrography90m and Environment90m

::: landschaft.download.hydrography90m

::: landschaft.download.environment90m

::: landschaft.download.Environment90m

## Any raster, any STAC catalogue

::: landschaft.download.raster

::: landschaft.download.stac
