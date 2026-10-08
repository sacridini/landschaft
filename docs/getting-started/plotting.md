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

A plot saved as `.html` is a single file that opens without a network connection:
the map library and the chart library are in the page, only the satellite imagery
and the other base maps come from the web. Send it to a colleague or open it in
the field; with `basemap=None` nothing at all is downloaded.

## What is drawn, and how

| Input | Plot |
|---|---|
| a product of a tool (MSPA, FAD, FOS, landscape mosaic, accounting, distances...) | its classes in GuidosToolbox's colours, with their names in the legend, in GuidosToolbox's order |
| a raster of integers with at most 64 values, or booleans | classes, each in a distinct colour |
| a floating-point raster of at most 12 whole numbers (a class map read with `masked=True`, NaN for its NoData) | classes, each in a distinct colour |
| any other raster | a colour ramp from the 2nd to the 98th percentile |
| a stack `(band, y, x)` | one layer, on one scale for all its bands, the band shown chosen on its legend card (a slider, and ▶ to play through them); red, green and blue bands as true colour |
| a GeoDataFrame, GeoSeries or geometry | polygons, lines or points, coloured by `column=` |
| a DataFrame or Series | bars, lines or one chart per metric |
| a tool's result (Zonation, habitat graphs, Biodiverse, PCA, clustergrams, least-cost paths, results with several rasters) | its maps, charts and summary numbers |
| a species distribution model (`ls.sdm.fit`) | its response curves and cross-validated scores; with `env=` (the predictors), the suitability map of each model and of the ensemble, with the presences |
| a change between two dates (`simple_change`, `morphological_change`, `fos_change`, `fad_multiscale_change`) | the change map with its classes named, the flows between the classes as a Sankey diagram, the counts |
| a list of rasters and vectors | one map, one layer each |

The ramp follows the name and the values of a raster: priority ranks (`rank`,
`priority`) on Zonation's rank colours, currents (`current`, `cum_currmap`,
`flow_potential`) on a log scale, elevations (`dem`, `elevation`) on a terrain
ramp, NDVI from red to green, values on both sides of zero on a diverging ramp
centred on zero. NoData is transparent, and so is the *Missing* class of a
product (the area outside the map): the imagery shows through.

## Reading the map

Move the cursor over a map and the box at its bottom left gives the coordinates
and, for every raster drawn, the value under the cursor: the name of a class
(`Core`, `Bridge in Edge`...) or, on a colour ramp, the value to 1/254 of the
ramp's range (`≈ 1006`; `≤` / `≥` beyond its ends). Click to keep them in a
popup. The values are read from the image in the page, so they are those of the
map drawn: block means where a large raster was reduced to the screen.

## Detail as you zoom in

While Python runs, a map draws the part in view in full detail as you zoom in:
past the resolution of the overview the page holds, it asks Python for the
cells in view at the screen's resolution, in the colours of the overview (a
raster of 50 000 × 50 000 cells shows its own cells, not block means), and the
value under the cursor is read from them. Vector layers drawn as an image are
burnt again for the view; WebGL layers get the features in view simplified to
the screen. This works in a notebook (Jupyter, JupyterLab, VS Code: a widget of
no size, under the map, carries the questions to the kernel) and in the window
a script opens (pywebview), with no server and no port. A page saved, or a
notebook opened again without its kernel, keeps the overview. The environment
variable `LANDSCHAFT_PLOT_LIVE=0` turns it off.

Every layer has a card in the legend, with a slider for its opacity (to see the
imagery underneath); a card hides while its layer is switched off in the layer
control (top right). The button under the zoom buttons shows the map full
screen.

## Comparing two dates

`compare=True` draws two rasters (or vector layers) on one map, the first on the
left of a line, the second on its right; drag the handle to sweep between them:

```python
forest = ls.load_raster(ls.example_data("input"))
after = forest.where(forest.x < forest.x[400], 1)     # the east cut
p = ls.plot([ls.mspa(forest), ls.mspa(after)], compare=True, title="Before / after")
```

Two maps of the same colours share one legend card. The values under the
cursor are read on both sides.

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
- `vmin`, `vmax`, `log`: the range of the ramp and a log scale, for rasters and
  for the column of a vector layer; values beyond the range take the colour of
  its end, marked `≤` and `≥` in the legend;
- `bands`: the bands of a stack to draw, by their names or the band numbers of
  the file (`bands=[4, 3, 2]`);
- `column`: the column of a vector layer that colours it;
- `basemap`: `"satellite"` (default), `"osm"`, `"topo"`, `"light"`, `"relief"`
  or `None`; the others stay selectable in the map's layer control.

## Static figures

`static=True` draws a matplotlib figure instead, for papers and reports; the
satellite imagery under it comes from [contextily](https://contextily.readthedocs.io/).
An interactive plot saved as `.png`, `.pdf` or `.svg` is drawn the same way.

A map has a scale bar (a round length, measured on the ground: Web Mercator and
degrees are corrected for latitude) and a north arrow; a single map on Web
Mercator or degrees has longitudes and latitudes on its axes. Several maps come
as small multiples: the bands of a stack (on one colour scale, one colour bar),
the rasters of a result, or two dates with `compare=True`. Layers of the same
colours share one legend. Vector layers take the colours of the interactive map
(`column=`, `cmap=`, `labels=`...). The figure's options are keywords:
`figsize=(w, h)` in inches, `ncols=`, `scalebar=False`, `north_arrow=False`.

```python
p = ls.plot(forest, static=True, basemap=None)
p.figure                                      # the matplotlib figure
p.save("forest.png").name                     # 'forest.png'
```

```python
stack = ls.load_raster(ls.example_data("orelevation")).expand_dims(band=["a", "b"])
p = ls.plot(stack, static=True, basemap=None, ncols=2, figsize=(9, 4))
```

## Speed

A raster is reduced to the resolution of the screen (`max_size=2048` cells on
the long side) before anything else: block means in C++ for continuous values,
one cell per block for classes, and a raster read lazily (`chunks="auto"`) is
read reduced, in strips, or through its overviews when it has some. It is
reprojected to Web Mercator once and written as an 8-bit PNG with a palette.
Vector layers of up to 20 000 features go to the map as GeoJSON, simplified to
half a pixel. Larger ones, up to 2 million features, are drawn by the graphics
card (WebGL): simplified to an eighth of a pixel, cut into triangles in C++ and
packed into the page, they stay sharp when zoomed in, and the value of the
feature under the cursor shows as for rasters, its columns in a popup on a
click (200 000 polygons in about a second). Even larger layers are drawn as an
image. WebGL needs a browser of the last few years (any recent Chrome, Edge,
Firefox, Safari or VS Code). A static figure draws all the
features of a layer at once (50 000 polygons in under a second) and burns more
than 100 000 into an image. The bands of a stack share 16 million pixels, so a
page of many bands stays light. A map of 100 million cells takes a fraction of
a second; see [Benchmarks](../benchmarks.md#plotting).

In tests and in scripts that only save plots, the environment variable
`LANDSCHAFT_PLOT_SHOW=0` keeps windows from opening. A detail asked for as
you zoom in takes 10 to 100 ms for a raster in memory.
