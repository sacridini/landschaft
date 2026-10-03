# Land change simulation

Land use and land cover change models: analysing the changes between maps,
estimating where change is likely, allocating future change and validating
the simulated maps: the `ls.simulate` module. It brings the tools of the
field together as interchangeable pieces of one model (CLUE-S, CLUMondo,
Dinamica EGO, PLUS, FUTURES), each reproducing its reference. The module is
being built phase by phase; this page grows with it.

<!-- no-run -->
```python
import landschaft as ls

lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))   # 1985, 1991, 1999
ch = ls.simulate.changes(lu)                                   # transitions, areas, change counts
ia = ch.intensity(gain=2, loss=1)                              # intensity analysis
ls.simulate.difference_components(ch.matrix())                 # quantity, exchange, shift
mk = ch.markov(steps=14)                                       # transition rates per year
```

## Change between maps

::: landschaft.simulate.changes

::: landschaft.simulate.Changes

::: landschaft.simulate.IntensityAnalysis

::: landschaft.simulate.crosstab

## Components of difference

::: landschaft.simulate.difference_components

::: landschaft.simulate.overall_difference

::: landschaft.simulate.difference_resolutions

::: landschaft.simulate.composite_matrix

::: landschaft.simulate.memberships

::: landschaft.simulate.mad

::: landschaft.simulate.sample_to_population

## Markov chains

::: landschaft.simulate.Markov

## Example data

::: landschaft.simulate.example_data
