# Loading and saving data

Landschaft's tools work on data in memory: rasters as `xarray.DataArray`,
vectors as `geopandas.GeoDataFrame`, tables as `pandas.DataFrame`. Files go
in and out through a handful of functions, one call each, that take care of
what usually needs a page of rasterio and xarray code: NoData, band names,
coordinate systems, grids that do not match, data types and formats.

| | Read | Write |
|---|---|---|
| rasters | `ls.load_raster` | `ls.write_raster` |
| vectors | `ls.load_vector` | `ls.write_vector` |
| tables | `ls.load_table` | `ls.write_table` |
| anything, results included | | `ls.save`, `result.save(folder)` |

Two more put data together: `ls.match_grid` (a raster on the grid of
another) and `ls.rasterize` (vector features on the grid of a raster).

## Rasters

```python
import numpy as np
import landschaft as ls

landcover = ls.load_raster(ls.example_data("clc"))
landcover.shape, landcover.rio.crs           # (2000, 2000), EPSG:3035
```

`load_raster` reads any raster GDAL reads and keeps what the file says about
itself:

- **several bands** (one species per band, the bands of a satellite image)
  give a `(band, y, x)` array named by the band descriptions:
  `stack.sel(band="jaguar")`; `band=2` or `band=[1, 3]` read only some;
- **NoData** of floating-point rasters becomes NaN; class maps (integers) keep
  their codes, with the NoData value in `.rio.nodata`, so that they stay small;
  `masked=True` turns every NoData into NaN;
- `chunks="auto"` reads lazily with dask, for rasters larger than memory.

Rasters from different sources rarely share their grid. `like=` puts a raster
on the grid of another as it is read: reprojected, resampled (nearest cell for
classes, bilinear for continuous values, or `resampling="average"`...) and
with the same coordinates, so that the two can be combined cell by cell:

```python
# a coarse layer in longitude/latitude, e.g. a human footprint index
footprint = landcover.isin(range(1, 12)).astype("float32").coarsen(x=8, y=8).mean()
ls.write_raster(footprint.rio.reproject("EPSG:4326"), "footprint.tif")

hfi = ls.load_raster("footprint.tif", like=landcover)    # 25 m cells of EPSG:3035
hfi.shape == landcover.shape                             # True
```

`ls.match_grid(data, like)` does the same for a raster already in memory.

`write_raster` stores a raster so that GIS software, and `load_raster`, read
it back as it was: georeferencing, NoData (NaN of floats included), band
names, the colour table of landschaft's class maps. It picks what the data
need (`uint8` for `True`/`False` maps, the smallest integer type for labels),
the format from the extension (`.tif`, `.asc`, `.img`...; `driver="COG"` for a
Cloud-Optimised GeoTIFF), writes dask arrays block by block and creates
missing folders. A numpy array takes its georeferencing from a raster on the
same grid with `like=`:

```python
towns = np.isin(landcover.values, [1, 2])                  # a plain numpy array
ls.write_raster(towns, "output/towns.tif", like=landcover)
```

## Vectors

`load_vector` reads any vector format (GeoPackage, Shapefile, GeoJSON,
FlatGeobuf, GeoParquet...) and tables of point coordinates (CSV, Excel,
Parquet), whose columns it finds by name: `x`/`y`, `lon`/`lat`,
`longitude`/`latitude`, Darwin Core's `decimalLongitude`/`decimalLatitude`.
With `like=` the features are reprojected to the CRS of a raster:

```python
import pandas as pd

records = pd.DataFrame({"species": ["lynx", "lynx"],
                        "decimalLongitude": [10.04, 10.07], "decimalLatitude": [52.42, 52.44]})
ls.write_table(records, "records.csv")

points = ls.load_vector("records.csv", like=landcover)    # longitude/latitude to EPSG:3035
points.crs.to_epsg()                                     # 3035
```

`rasterize` burns features into a raster on the grid of another: a mask (1
inside, 0 outside), the values of an attribute (`field="id"`), or the number
of overlapping features (`merge="add"`):

```python
from shapely.geometry import box
import geopandas as gpd

x0, y0 = landcover.rio.bounds()[:2]                     # lower-left corner
reserves = gpd.GeoDataFrame({"name": ["bog", "heath"]}, crs=landcover.rio.crs,
                            geometry=[box(x0 + 5000, y0 + 5000, x0 + 9000, y0 + 9000),
                                      box(x0 + 20000, y0 + 30000, x0 + 26000, y0 + 34000)])
ls.write_vector(reserves, "reserves.gpkg")

mask = ls.rasterize(ls.load_vector("reserves.gpkg"), like=landcover)
int(mask.sum()) * 25 * 25 / 1e6                          # 40.0 km2 protected
```

## Tables

`load_table` reads CSV and text files (the separator, `,` `;` tab or `|`, and
a decimal comma are detected), Excel, Parquet, Feather, JSON and the
attribute table of a vector file. `write_table` writes the format of the
extension and keeps a named index as columns:

```python
from pathlib import Path

Path("weights.csv").write_text("species;weight\nlynx;2,5\nwolf;1\n", encoding="utf-8")
weights = ls.load_table("weights.csv")
weights["weight"].tolist()                               # [2.5, 1.0]

metrics = ls.landscape_metrics(landcover[:400, :400], level="class", metric="ca")
ls.write_table(metrics, "output/class_area.parquet")
```

## Saving results

`ls.save` writes anything in the format that suits it: a raster as a
GeoTIFF, a table as a CSV, a GeoDataFrame as a GeoPackage. A result of a
tool is written into a folder: one GeoTIFF per map (numpy maps get the
georeferencing of the result's other maps), one table per table, and the
numbers and settings in `summary.json`. Every result also has a `save`
method:

```python
forest = ls.load_raster(ls.example_data("input"))

ls.save(ls.mspa(forest), "output/mspa")                 # output/mspa.tif
folder = ls.fos(forest).save("output/fragmentation")     # map.tif, stats.csv, summary.json...
sorted(p.name for p in folder.iterdir())
```

Results with a format of their own are written in it: a Zonation ranking as
Zonation 5's files, a Biodiverse basedata in its format, a tree as Newick.

## All together

Species distributions in one multi-band file, a cost layer on another grid,
existing reserves as polygons: ranked for conservation and saved.

```python
import xarray as xr


def share(*codes):
    return landcover[:800, :800].isin(codes).coarsen(x=4, y=4).mean().astype("float32")

species = xr.concat([share(23), share(24), share(25), share(35, 36)], dim="band")
species = species.assign_coords(band=["broadleaf", "conifer", "mixed", "bog"])
ls.write_raster(species, "input/species.tif")

# the four files of a project, each in one line
sdm = ls.load_raster("input/species.tif")                         # (band, y, x), named bands
hfi = ls.load_raster("footprint.tif", like=sdm)                   # on the grid of the species
pa = ls.rasterize(ls.load_vector("reserves.gpkg"), like=sdm)      # 1 in the reserves

res = ls.prioritize.zonation(sdm, cost=1 + hfi.fillna(0), hierarchic_mask=1 + pa)
res.save("output/priorities")                                     # rankmap.tif, curves...
```
