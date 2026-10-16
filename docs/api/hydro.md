# Terrain hydrology

Where water goes on a terrain: depressions filled so that every cell drains,
the direction each cell drains to, how much area drains through it, the
stream network, the height above the nearest drainage (HAND) and the wetness,
stream power and sediment transport indices. They live in the `ls.hydro`
module.

The directions and accumulations are those of WhiteboxTools 2.4 cell for cell
(`D8Pointer`, `DInfPointer`, `D8FlowAccumulation`, `DInfFlowAccumulation`,
`FD8FlowAccumulation`, `ElevationAboveStream`); the filling is its
`FillDepressions` with water let out at the map's edge and at missing cells,
which WhiteboxTools does not do. Flat areas drain by the method of Barnes,
Lehman & Mulla (2014).

<!-- no-run -->
```python
import landschaft as ls

dem = ls.load_raster("srtm.tif")
filled = ls.hydro.fill(dem)
acc = ls.hydro.flow_accumulation(filled)                 # D8, in cells
rivers = ls.hydro.streams(acc, 1000)
height = ls.hydro.hand(filled, rivers)                   # height above the drainage
wetness = ls.hydro.twi(filled)                           # FD8 contributing area
```

## Conditioning

::: landschaft.hydro.fill

## Flow

::: landschaft.hydro.flow_direction

::: landschaft.hydro.flow_accumulation

::: landschaft.hydro.streams

`ls.hydro.D8_CODES` maps the D8 codes to their directions: 1 east, 2
south-east, 4 south, 8 south-west, 16 west, 32 north-west, 64 north, 128
north-east (0: no lower neighbour).

## Height above the nearest drainage

::: landschaft.hydro.hand

## Topographic indices

::: landschaft.hydro.twi

::: landschaft.hydro.spi

::: landschaft.hydro.sti
