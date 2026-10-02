# Species distributions

Preparing occurrence records and environmental predictors for species
distribution models, fitting the models and projecting them: the `ls.sdm`
module. Every step reproduces the R package the field relies on for it, and
each function's notes name that package.

<!-- no-run -->
```python
import landschaft as ls

occ = ls.load_table(ls.sdm.example_data("bradypus"))
env = ls.load_raster(ls.sdm.example_data("bioclim"))
occ = ls.sdm.one_per_cell(occ, env)                     # one record per predictor cell
m = ls.sdm.accessible_area(occ, distance=300_000)       # where the background comes from
```

## Occurrence records

::: landschaft.sdm.download_gbif

::: landschaft.sdm.clean_occurrences

::: landschaft.sdm.reference_data

::: landschaft.sdm.one_per_cell

::: landschaft.sdm.thin

## Study area

::: landschaft.sdm.accessible_area

## Predictors

::: landschaft.sdm.select_predictors

::: landschaft.sdm.vif

## Example data

::: landschaft.sdm.example_data
