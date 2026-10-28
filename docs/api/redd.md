# Deforestation allocation (VT0007)

Verra's VT0007, the Unplanned Deforestation Allocation (UDef-A) of a
jurisdiction's expected deforestation to its forest, as Clark Labs' UDef-ARP
implements it, the same numbers: the negligible risk threshold of the distance
to the forest's edge, the vulnerability in classes of geometric widths (the
benchmark from that distance, or an alternative map such as a transition
potential), the relative frequency of deforestation per class and subdivision,
the allocation of the activity data never above a cell, and the assessment of
the predicted against the actual deforestation in an assessment grid. The
`ls.redd` module.

<!-- no-run -->
```python
import landschaft as ls

forest = ls.load_raster("forest_2015.tif")            # 1 forest at the start of the calibration
cal = ls.load_raster("deforestation_2015_2020.tif")   # 1 deforested in the calibration period
juris = ls.load_raster("jurisdiction.tif")            # non-zero inside
states = ls.load_raster("states.tif")                 # subdivisions (ids 0-999)

d = ls.redd.edge_distance(forest)
t = ls.redd.nrt(d, cal, mask=juris)                   # negligible risk threshold (m)
classes = ls.redd.benchmark(d, t, mask=juris)
f = ls.redd.fit(classes, cal, subdivisions=states)

# confirm the model on a later period, then allocate the activity data
cnf = ls.redd.allocate(classes_2020, f, deforestation=cnf_2020_2023, subdivisions=states)
ls.redd.assess(cnf.density, cnf_2020_2023, grid_area=10_000, mask=juris).metrics
a = ls.redd.allocate(classes_2023, f, expected=12_000, subdivisions=states)
ls.write_raster(a.density, "expected_deforestation_ha.tif")
```

## Distance and threshold

::: landschaft.redd.edge_distance

::: landschaft.redd.nrt

## Vulnerability

::: landschaft.redd.benchmark

::: landschaft.redd.alternative

## Fit and allocation

::: landschaft.redd.fit

::: landschaft.redd.Fit

::: landschaft.redd.allocate

::: landschaft.redd.Allocation

## Assessment

::: landschaft.redd.assess

::: landschaft.redd.Assessment
