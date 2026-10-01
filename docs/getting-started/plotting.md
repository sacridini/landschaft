# Plotting

`ls.plot` draws anything Landschaft reads or makes, as the data call for: a
raster or a vector layer on an interactive map over satellite imagery, a table
as a chart, the result of a tool as its maps, charts and numbers together. One
call, whatever the input:

```python
import landschaft as ls

forest = ls.load_raster(ls.example_data("input"))
p = ls.plot(ls.mspa(forest))
p.save("mspa.html").name                     # 'mspa.html'
```

In a notebook (Jupyter, JupyterLab, VS Code, Colab) the plot shows inline. In a
script it opens in a window of its own ([pywebview](https://pywebview.flowrl.com/),
installed with Landschaft), or in the web browser where no window can open.

## What is drawn, and how

| Input | Plot |
|---|---|
| a product of a tool (MSPA, FAD, FOS, landscape mosaic, accounting, distances...) | its classes in GuidosToolbox's colours, with their names in the legend, in GuidosToolbox's order |
| a raster of integers with at most 64 values, or booleans | classes, each in a distinct colour |
| a floating-point raster of at most 12 whole numbers (a class map read with `masked=True`, NaN for its NoData) | classes, each in a distinct colour |
| any other raster | a colour ramp from the 2nd to the 98th percentile |
| a stack `(band, y, x)` | one layer per band, on one scale; red, green and blue bands as true colour |
| a GeoDataFrame, GeoSeries or geometry | polygons, lines or points, coloured by `column=` |
| a DataFrame or Series | bars, lines or one chart per metric |
| a tool's result (Zonation, habitat graphs, Biodiverse, PCA, results with several rasters) | its maps, charts and summary numbers |
| a list of rasters and vectors | one map, one layer each |

The ramp follows the name and the values of a raster: priority ranks (`rank`,
`priority`) on Zonation's rank colours, currents (`current`, `cum_currmap`,
`flow_potential`) on a log scale, elevations (`dem`, `elevation`) on a terrain
ramp, NDVI from red to green, values on both sides of zero on a diverging ramp
centred on zero. NoData is transparent, and so is the *Missing* class of a
product (the area outside the map): the imagery shows through.

## Choosing the colours

The defaults can all be changed in the call:

```python
dem = ls.load_raster(ls.example_data("orelevation"))
p = ls.plot(dem, cmap="magma", vmin=0, vmax=1500, title="Elevation")
landcover = ls.load_raster(ls.example_data("clc"))[:400, :400]
p = ls.plot(landcover == 23, cmap={0: "#f4f1de", 1: "#2d6a4f"},
            labels={0: "other", 1: "broad-leaved forest"})
```

- `cmap`: a palette name (`"viridis"`, `"magma"`, `"terrain"`, `"rdylgn"`,
  `"rdbu"`, `"blues"`, `"greens"`, `"ylorrd"`, `"rank"`, `"current"`, or any
  matplotlib colormap; `"_r"` reverses it), a list of colours, or a
  `{value: colour}` dictionary for classes;
- `labels`: `{value: name}`, the names of the classes; a class given a
  transparent colour (`"#00000000"`) is left out of the map and the legend;
- `categorical`: `True` draws the values as classes, `False` on a ramp;
- `vmin`, `vmax`, `log`: the range of the ramp and a log scale;
- `column`: the column of a vector layer that colours it;
- `basemap`: `"satellite"` (default), `"osm"`, `"topo"`, `"light"`, `"relief"`
  or `None`; the others stay selectable in the map's layer control.

## Static figures

`static=True` draws a matplotlib figure instead, for papers and reports; the
satellite imagery under it comes from [contextily](https://contextily.readthedocs.io/).
An interactive plot saved as `.png`, `.pdf` or `.svg` is drawn the same way:

```python
p = ls.plot(forest, static=True, basemap=None)
p.figure                                      # the matplotlib figure
p.save("forest.png").name                     # 'forest.png'
```

## Speed

A raster is reduced to the resolution of the screen (`max_size=2048` cells on
the long side) before anything else: block means in C++ for continuous values,
one cell per block for classes, and a raster read lazily (`chunks="auto"`) is
read reduced, in strips, or through its overviews when it has some. It is
reprojected to Web Mercator once and written as an 8-bit PNG with a palette.
Vector layers of up to 20 000 features go to the map as GeoJSON, simplified to
half a pixel; larger ones are drawn as an image. A map of 100 million cells
takes a fraction of a second; see [Benchmarks](../benchmarks.md#plotting).

In tests and in scripts that only save plots, the environment variable
`LANDSCHAFT_PLOT_SHOW=0` keeps windows from opening.
