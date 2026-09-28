# Neutral landscape models

Simulated landscapes that follow a known rule (complete randomness, a
gradient, clustering, fractal roughness) and nothing else: the null models of
landscape ecology, to test whether a pattern or a process differs from what
chance alone would give. They live in the `ls.nlm` module.

With the same `seed` every model gives the same landscape as the R package
NLMR, value for value: the random number generators of R and of the C++
library NLMR is built with are reproduced, so studies done in R can be
repeated and extended here.

<!-- no-run -->
```python
import landschaft as ls

r = ls.nlm.random(200, 200, seed=42)             # a DataArray of values in 0-1
habitat = ls.nlm.classify(r, weighting=[0.7, 0.3])
```

## Random and percolation

::: landschaft.nlm.random

::: landschaft.nlm.percolation

## Gradients

::: landschaft.nlm.planar_gradient

::: landschaft.nlm.edge_gradient

::: landschaft.nlm.distance_gradient

## Fractal surfaces and random fields

::: landschaft.nlm.mpd

::: landschaft.nlm.fbm

::: landschaft.nlm.gaussian_field

## Clusters

::: landschaft.nlm.random_cluster

::: landschaft.nlm.neigh

::: landschaft.nlm.random_rectangular_cluster

::: landschaft.nlm.curds

## Classification and rescaling

::: landschaft.nlm.classify

::: landschaft.nlm.rescale
