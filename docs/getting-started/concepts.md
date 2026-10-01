# Core concepts

Every Landschaft tool works on rasters and follows the same conventions. This
page explains them once; the tutorials assume them.

## Rasters

A raster is a grid of cells (pixels), each holding a number: a land-cover
class, a percentage of tree cover, a resistance to movement... Landschaft
accepts rasters in three forms:

| Form | Example | Georeferencing |
|---|---|---|
| a file path | `"landcover.tif"` | read from the file |
| an `xarray.DataArray` | `ls.load_raster("landcover.tif")` | kept in the result |
| a numpy array | `np.array([[1, 2], [2, 2]])` | none: results are plain arrays |

Functions return the same form they receive: give a georeferenced DataArray
and you get one back, ready to save with `ls.write_raster`. Most analyses are
easiest with DataArrays, which is what `ls.load_raster` returns:

```python
import landschaft as ls

landcover = ls.load_raster(ls.example_data("clc"))
print(landcover.rio.crs, landcover.rio.resolution())   # EPSG:3035, (25.0, -25.0)
```

`load_raster` reads any raster the same way and keeps what the file says
about itself:

- **several bands** give a `(band, y, x)` array whose `band` coordinate holds
  the band names stored in the file (e.g. one species per band:
  `stack.sel(band="jaguar")`); `band=2` or `band=[1, 3]` read only some;
- **NoData** of floating-point rasters becomes NaN; class maps (integers) keep
  their codes, with the NoData value in `.rio.nodata`, so they stay small
  (`masked=True` turns every NoData into NaN, `masked=False` keeps the stored
  values);
- `chunks="auto"` reads lazily with dask, for rasters larger than memory.

`ls.write_raster` stores a result back so that GIS software, and
`load_raster`, read it as it was: georeferencing, NoData (NaN of floats
included), band names, the colours of class maps. A numpy array takes its
georeferencing from a raster on the same grid with `like=`:

```python
import numpy as np

towns = np.isin(landcover.values, [1, 2])                  # a plain numpy array
ls.write_raster(towns, "output/towns.tif", like=landcover)  # creates the folder; bool is saved as uint8
```

Vectors, tables and whole results have functions of the same kind: see
[Loading and saving data](data.md).

## Habitat maps

Pattern, fragmentation and change tools (MSPA, density, fragmentation classes,
accounting, restoration...) work on **binary habitat maps**: single-band 8-bit
rasters coded

| value | meaning |
|---|---|
| 0 | missing: outside the study area, no data |
| 1 | background: not habitat |
| 2 | foreground: habitat |

`ls.reclassify` builds such a map from any raster, choosing which values are
habitat:

```python
landcover = ls.load_raster(ls.example_data("clc"))

# CORINE classes 23, 24 and 25 are the forests
forest = ls.reclassify(landcover, foreground=[23, 24, 25])

# a rule instead of a list, e.g. on a tree-cover percentage:
#     ls.reclassify(tree_cover, foreground=lambda a: a >= 30)
```

Values that are NaN or equal to the raster's NoData become 0 (missing); other
classes can be sent there with `missing=[...]`. The habitat tools check
their input and explain what is wrong (wrong data type, values above 2, no
habitat at all...); `ls.load_raster(path, validate="mspa")` runs the same
checks on reading.

Other tools take other kinds of maps: **categorical maps** (any integer
classes: landscape metrics), **continuous maps** (percentages for
grayscale fragmentation, resistances or conductances for connectivity), or
**patch maps** (one id per habitat patch: habitat networks). Each tutorial
says what it needs.

## NoData

Cells without data are marked by the raster's NoData value (a DataArray keeps
it in `data.rio.nodata`), by NaN in floating-point rasters, or by 0 in habitat
maps. Landschaft leaves them out of every computation: a moving window near
NoData uses only the valid cells, a landscape metric counts only valid cells,
and results carry NoData where the input had it.

## Cell size and units

Distances, areas and densities use the raster's cell size and units. Use a
**projected** coordinate system in metres (UTM, a national grid, EPSG:3035 in
Europe...) whenever sizes matter: in a geographic CRS (degrees) a cell is not a
square on the ground, and Landschaft warns you where it matters. A numpy
array has cells of size 1 unless a function takes a `res=` argument.

Window sizes (`window_size=27`) and edge widths (`edge_width=2`) are counted in
**cells**, not map units: a 27 × 27 window over 25 m cells covers 675 m.

## Results

Simple analyses return a raster. Richer ones return a **result object**
with the maps and tables as attributes and a `summary()` report:

```python
result = ls.fad(forest, window_size=27)
result.density        # habitat density map, percent
result.classes        # fragmentation classes
print(result.summary())
```

Tables are pandas DataFrames, easy to save (`.to_csv`, `.to_parquet`) or plot.

## Plotting maps

A DataArray plots itself with `data.plot()`. Class maps look best with the
colours of their classes; every tool that returns class codes has a colour
table (`ls.mspa_colormap()`, `ls.fad_colormap()`...) as a
`{value: (red, green, blue, alpha)}` dictionary, embedded automatically by
`ls.write_raster`. To draw one in matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def show_classes(data, colours, ax=None):
    """Draw a class map with a {value: (r, g, b, a)} colour table."""
    lut = np.zeros((256, 4))
    for value, rgba in colours.items():
        lut[value] = np.array(rgba) / 255
    ax = ax or plt.gca()
    ax.imshow(lut[np.asarray(data)], interpolation="nearest")
    ax.set_axis_off()
    return ax


morph = ls.mspa(ls.load_raster(ls.example_data("input")))
show_classes(morph, ls.mspa_colormap())
plt.show()
```

## Large maps and many maps

Everything runs in memory by default and uses all processor cores but one.
For maps larger than memory, collections of files and batch processing, see
[configuration](configuration.md) and the *big data* tutorial.
