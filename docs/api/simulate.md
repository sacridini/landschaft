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

lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
drivers = ls.load_raster(ls.simulate.example_data("sibuyan_drivers"))
```

## Example data

::: landschaft.simulate.example_data
