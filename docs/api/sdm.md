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
data = ls.sdm.prepare(occ, env)          # clean, one per cell, predictors, background, spatial folds
res = ls.sdm.fit(data, models)           # cross-validated scikit-learn models (or ls.sdm.MaxNet)
suit = ls.sdm.predict(res, env)          # ensemble and its uncertainty
```

## The whole preparation

::: landschaft.sdm.prepare

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

## Background and data

::: landschaft.sdm.background

::: landschaft.sdm.dataset

::: landschaft.sdm.SDMData

## Cross-validation folds

::: landschaft.sdm.spatial_blocks

::: landschaft.sdm.autocorrelation_range

::: landschaft.sdm.spatial_clusters

::: landschaft.sdm.buffer_folds

::: landschaft.sdm.quadrant_folds

::: landschaft.sdm.checkerboard_folds

::: landschaft.sdm.random_folds

::: landschaft.sdm.jackknife_folds

## Models

::: landschaft.sdm.MaxNet

## Evaluation

::: landschaft.sdm.evaluate

::: landschaft.sdm.Evaluation

::: landschaft.sdm.boyce

::: landschaft.sdm.threshold

::: landschaft.sdm.omission_rate

## Extrapolation

::: landschaft.sdm.mess

::: landschaft.sdm.mop

::: landschaft.sdm.exdet

## Models and projections

::: landschaft.sdm.fit

::: landschaft.sdm.SDMResult

::: landschaft.sdm.predict

::: landschaft.sdm.response_curves

::: landschaft.sdm.importance

## Area of habitat

::: landschaft.sdm.area_of_habitat

::: landschaft.sdm.HabitatAreas

::: landschaft.sdm.cell_area_km2

## Example data

::: landschaft.sdm.example_data
