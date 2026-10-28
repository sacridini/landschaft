# Brazil's Forest Code

The Áreas de Preservação Permanente (APP) of Brazil's Forest Code (Lei
12.651/2012, art. 4) and the balance of every rural property's Reserva Legal:
the margins of rivers by their width and of natural lakes by their area, the
springs, the veredas, the slopes above 45°, the hilltops
([`ls.surface.hilltop`](surface.md#landschaft.surface.hilltop)), the land above
1,800 m and protected classes; art. 61-A's narrower margins in the areas
consolidated by 22 July 2008 by the property's fiscal modules; art. 12's
shares of the Reserva Legal, art. 15's APP counted and art. 67's small
properties. The `ls.forest_code` module.

<!-- no-run -->
```python
import landschaft as ls

dem = ls.load_raster("dem.tif")
a = ls.forest_code.app(
    rivers=ls.load_raster("rivers.tif"), river_width=ls.load_raster("river_width.tif"),
    lakes=ls.load_raster("lakes.tif"), springs=ls.load_vector("springs.gpkg"),
    slope=ls.surface.terrain(dem, "slope"), elevation=dem, hilltops=ls.surface.hilltop(dem).app,
    properties=ls.load_raster("car.tif"), consolidated=ls.load_raster("consolidated.tif"), fiscal_module=70)
a.table                                               # hectares of each kind of APP

r = ls.forest_code.reserve(ls.load_raster("car.tif"), ls.load_raster("native_vegetation.tif"),
                           share=80, app=a.app, fiscal_module=70,
                           vegetation_2008=ls.load_raster("native_2008.tif"))
r.table[["property", "required_ha", "counted_ha", "deficit_ha", "surplus_ha"]]
```

## Áreas de Preservação Permanente

::: landschaft.forest_code.app

::: landschaft.forest_code.APP

::: landschaft.forest_code.river_margin

::: landschaft.forest_code.consolidated_river

::: landschaft.forest_code.consolidated_lake

## Reserva Legal

::: landschaft.forest_code.reserve

::: landschaft.forest_code.ReservaLegal
