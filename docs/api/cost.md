# Grid distances

Distances and routes over a raster seen as a graph of neighbouring cells: each
connection between two cells carries a conductance (how easily something
moves from one to the other), from which come cost surfaces, least-cost
distances and paths, and the distances and flows of random walks. They live in
the `ls.cost` module.

The results are those of the R package gdistance, value for value: the
connections are built with R's arithmetic, and on longitude/latitude grids the
distances between cells are the ellipsoidal ones of the R package raster, to
the last bit. Tutorial: [routes on grids](../tutorials/grid-routes.md).

<!-- no-run -->
```python
import numpy as np
import landschaft as ls

dem = ls.load_raster(ls.cost.example_data("maungawhau"))
slope = ls.cost.geo_correction(ls.cost.transition(dem, "diff", 8, symmetric=False))
speed = slope.map(lambda s: 6 * np.exp(-3.5 * np.abs(s + 0.05)), pairs="all")  # Tobler, m/s
conductance = ls.cost.geo_correction(speed)                                    # 1 / seconds per step
```

## Transitions

::: landschaft.cost.transition

::: landschaft.cost.Transition

::: landschaft.cost.geo_correction

::: landschaft.cost.normalize

::: landschaft.cost.sum_reciprocal

## Least-cost surfaces, distances and paths

The cost of a step between neighbouring cells is the reciprocal of the
transition's value; among equally cheap paths, `shortest_path` returns the one
gdistance returns.

::: landschaft.cost.acc_cost

::: landschaft.cost.cost_distance

::: landschaft.cost.shortest_path

## Random walks

A random walk moves from a cell to a neighbour with a probability proportional
to the conductance of the connection. Its commute distances and passages are
those of an electrical circuit (the transition's conductances as resistors),
computed with the same sparse Cholesky factorisation as the
[circuit tools](circuits.md).

::: landschaft.cost.commute_distance

::: landschaft.cost.passage

## Randomised shortest paths

Between the random walk and the least-cost path: walks that prefer cheap
routes more and more as `theta` grows (Saerens et al. 2009). `passage` with a
`theta` gives their passages; `rsp_distance` their expected costs.

::: landschaft.cost.rsp_distance

## Route overlap, barriers and areas

How much the routes from a common origin to two locations share (van Etten
and Hijmans 2010), optionally weighted by barriers or areas.

::: landschaft.cost.path_inc

::: landschaft.cost.overlap

::: landschaft.cost.nonoverlap

::: landschaft.cost.barriers

::: landschaft.cost.areas

## Example data

::: landschaft.cost.example_data
