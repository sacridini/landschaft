# Terrain hydrology

Where does the water go? A digital elevation model (DEM) answers it once its
spurious depressions are removed: every cell then drains to a neighbour, the
area draining through each cell can be summed, and the cells with the most
area are the streams. From there come the terrain descriptors that soil,
vegetation and species follow: the height above the nearest drainage (HAND),
developed at INPE for the Amazon, and the topographic wetness index.

**You will learn to:**

- condition a DEM by filling or breaching its depressions;
- compute flow directions and flow accumulation with single and multiple
  flow methods;
- extract a stream network;
- map the height above the nearest drainage and the wetness index;
- reproduce the routing of the InVEST models.

Everything lives in `ls.hydro`. The flow directions and accumulations are
those of WhiteboxTools cell for cell; where WhiteboxTools errs (it lets no
water out at the map's edge when filling), Landschaft does what the method
intends.

## A DEM and its depressions

The example DEM is SRTM elevation of about 30 m cells over south-western
Oregon (longitude and latitude: cell sizes are taken in metres at the centre
latitude).

```python
import numpy as np
import landschaft as ls

dem = ls.load_raster(ls.surface.example_data("orelevation"))
print(dem.shape, float(dem.min()), float(dem.max()))
```

```text
(371, 371) 433.0 1390.0
```

Raw DEMs have depressions that interrupt the flow: noise, bridges and roads
across valleys, the limits of the sensor. `fill` raises each to the height
where it would spill over (Barnes, Lehman & Mulla 2014):

```python
filled = ls.hydro.fill(dem)
raised = (filled - dem).values
print(int((raised > 0).sum()), round(float(raised.max()), 1))
```

```text
2030 23.0
```

2030 cells were raised, by up to 23 m. Breaching cuts a channel through the
rim of each depression instead, which often changes the terrain much less
(Lindsay 2016): here the total change is 2786 m against 10046 m by filling.

```python
breached = ls.hydro.breach(dem, max_length=30)
print(int((breached.values != dem.values).sum()))
```

```text
3134
```

Channels longer than `max_length` cells (or costlier than `max_cost`) are not
cut, and what is left is filled.

## Flow directions and accumulation

On the conditioned DEM every cell drains somewhere. D8 sends all of a cell's
water to its steepest neighbour, coded as ESRI does (1 east, 2 south-east, 4
south ... 128 north-east; 0 for the cells that drain off the map):

```python
d8 = ls.hydro.flow_direction(filled)
print(int((d8 == 0).sum()), int((d8 == 4).sum()))
```

```text
114 17373
```

Flat areas left by filling are drained towards lower terrain and away from
higher terrain (Barnes, Lehman & Mulla 2014), so no cell is left without a
direction. The accumulation sums, for each cell, the area that drains through
it, in cells, area or specific contributing area (`units`):

```python
acc = ls.hydro.flow_accumulation(filled)
area = ls.hydro.flow_accumulation(filled, units="area")
print(float(acc.max()), round(float(area.max()) / 1e6, 2))
```

```text
55126.0 36.01
```

The outlet of the map's largest catchment collects 55126 cells, 36 km². D8
concentrates flow in single lines, which draws artificial parallel paths on
hillslopes; the multiple flow methods spread it among all lower neighbours:
`"fd8"` (Freeman 1991), `"quinn"` (Quinn et al. 1991), `"dinf"` (Tarboton
1997) and `"mfd"`, the routing of the InVEST models.

## Streams

A stream starts where enough area drains through a cell:

```python
rivers = ls.hydro.streams(acc, 500)
print(int(np.nansum(rivers.values)))
```

```text
3198
```

The threshold sets the density of the network: about 45 hectares of
catchment here. `streams(acc, threshold, trace=0.7, dem=filled)` traces the
streams of a multiple flow accumulation as InVEST's sediment model does.

## Height above the nearest drainage

HAND is the height of each cell above the stream cell its water reaches
(Rennó et al. 2008; Nobre et al. 2011): near zero on floodplains and valley
floors, large on hilltops. It separates terrain by its relation to the water
table, which elevation alone cannot, since a valley floor high in the
mountains and a coastal plain both have small HAND.

```python
height = ls.hydro.hand(filled, rivers)
print(np.nanpercentile(height.values, [10, 50, 90]).round(1))
```

```text
[ 14.  92. 215.]
```

## Wetness

The topographic wetness index, ln(a / tan β) (Beven & Kirkby 1979), is high
where much area drains through a gentle slope. It uses the specific
contributing area of a multiple flow method (FD8 by default) and the slope of
Horn (1981):

```python
wet = ls.hydro.twi(filled)
low = height.values < 5
print(np.nanpercentile(wet.values, [10, 50, 90]).round(2))
print(round(float(np.nanmean(wet.values[low])), 2), round(float(np.nanmean(wet.values[~low & ~np.isnan(height.values)])), 2))
```

```text
[5.1  6.21 8.82]
10.82 6.48
```

The cells less than 5 m above the drainage are much wetter than the rest.
`spi` (stream power) and `sti` (sediment transport, the LS factor of the
USLE) combine the same parts.

## Validation

The D8 and D-infinity directions, the D8, D-infinity and FD8 accumulations,
HAND and the indices are those of WhiteboxTools 2.4 bit for bit, and the
`"mfd"` routing, the distance to channel and the stream tracing those of
pygeoprocessing 2.4 (InVEST 3.20). The filling and the breaching are those of
WhiteboxTools with the map's edge as an outlet; WhiteboxTools raises the
depressions that touch the edge and leaves channels to the edge uncut. See
the [benchmarks](../benchmarks.md).

## Next steps

- The [API reference](../api/hydro.md) lists every option.
- [Surface texture](surface-texture.md) describes the terrain itself: slope,
  ruggedness, curvature, landforms (geomorphons) and hilltops.
