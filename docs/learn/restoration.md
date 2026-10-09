<p class="ls-chapter">Chapter 9</p>

# Distance, cost and restoration

Protecting what remains is no longer enough: in many regions, habitat has
fallen below the thresholds of [chapter 3](fragmentation.md) and must be
recovered. The United Nations declared 2021–2030 the Decade on Ecosystem
Restoration, and national commitments add up to hundreds of millions of
hectares. Budgets, though, are finite, and where restoration happens changes
its benefits by orders of magnitude (Strassburg et al., 2020). This chapter
covers the spatial tools behind the question *where should we restore?*:
distance maps, cost surfaces, least-cost paths and corridors, and indicators
of what reconnecting habitat would gain.

!!! abstract "In this chapter"
    - distance maps and what they tell about edges and natural regeneration;
    - accumulated cost and the algorithm behind it;
    - least-cost paths and least-cost corridors;
    - measuring the connectivity a landscape has and could gain;
    - comparing restoration options by gain and by efficiency.

## Distance to habitat

The simplest spatial variable is the straight-line (Euclidean) distance
from every cell to the nearest habitat edge. Signed, it describes both
sides of the edge: positive inside the habitat (how deep in the interior),
negative outside (how far from habitat).

```python
import numpy as np
import pandas as pd
import landschaft as ls

forest = ls.load_raster(ls.example_data("input"))       # 0 missing, 1 background, 2 forest
distance = ls.euclidean_distance(forest, pixel_size=25)
signed = np.asarray(distance.signed_distance) * 25      # the map is in cells; 25 m cells
{"forest cells within 100 m of an edge (%)": 100 * np.mean(signed[forest.values == 2] <= 100),
 "open cells within 100 m of forest (%)": 100 * np.mean(-signed[forest.values == 1] <= 100)}
```

About half of the forest lies within 100 m of an edge, and about half of
the open land within 100 m of forest. Both numbers have direct ecological
readings. Inside the forest, the share
of cells within reach of edge effects ([chapter 3](fragmentation.md#edge-effects)).
Outside, the share of open land close to seed sources: the probability of
**natural regeneration** falls quickly with distance from existing forest,
because most seeds are dispersed over short distances. Areas near forest can
often be restored by letting them regrow, at a fraction of the cost of
planting (Chazdon & Guariguata, 2016).

## Accumulated cost

A **cost surface** is the accumulated resistance
([chapter 6](connectivity.md#resistance-surfaces)) of the cheapest route
from a set of source cells to every other cell. It is computed on the grid
seen as a graph (each cell linked to its eight neighbours), with Dijkstra's
shortest-path algorithm (Dijkstra, 1959): starting from the sources,
repeatedly settle the unsettled cell with the smallest accumulated cost and
update its neighbours. In Landschaft a step between two cells costs the mean
of their resistances times the step length (1 for a side, √2 for a
diagonal).

Where does the resistance come from? Any surface of values from 1 to 100
works. Without information on the species, a useful default is resistance
that grows with the distance from habitat, on the assumption that forest
animals avoid moving far from cover:

```python
resistance = ls.distance_to_resistance(forest)        # 2 on forest, 3 to 100 away from it
objects = ls.accounting(forest)
ids = np.asarray(objects.ids.values)
largest = [obj[0] for obj in objects.largest]          # ids of the three largest objects
area_a, area_b = ids == largest[0], ids == largest[1]

cost_from_a = np.asarray(ls.cost_distance(resistance, area_a))
cost_from_b = np.asarray(ls.cost_distance(resistance, area_b))
```

## Least-cost paths and corridors

The **least-cost path** between A and B is the route of minimum accumulated
cost. It passes through the cells where the cost from A plus the cost from
B is minimal, which gives a simple way to go beyond a single line: the
**least-cost corridor** is the set of cells whose total cost is within a
small margin of the minimum (Adriaensen et al., 2003; Beier et al., 2008):

```python
total = cost_from_a + cost_from_b
minimum = total[np.isfinite(total)].min()
background = forest.values == 1

{f"within {int(round(100 * (s - 1)))}% of the minimum": int(((total <= minimum * s) & background).sum())
 for s in (1.02, 1.05, 1.10)}
```

```python
path = ls.least_cost_path(resistance, area_a, area_b)
path.path_length, path.rest_pixels, path.expense
```

The path itself crosses 113 non-forest cells; allowing routes 5% more
costly than the best one widens the band to about 2 000 cells. Corridors are
more robust than single paths: they show the room available for a corridor
on the ground, and routes that are almost as good as the best one.

Least-cost models assume that the organism "knows" the best route, which is
reasonable for directed movement (seasonal migrations, animals familiar with
their home range) and less so for dispersers exploring new ground, for which
circuit models ([chapter 8](circuits.md)) are more realistic (Etherington,
2016). Use both when the choice matters: a corridor that is both a
least-cost corridor and a zone of high current is a robust priority.

## How connected is the habitat, and how much could it gain?

To compare restoration options we need an indicator of connectivity that
increases when habitat is reconnected. The restoration status uses the
equivalent connected area ([chapter 7](networks.md#the-probability-of-connectivity))
of a species that cannot cross non-habitat at all: habitat is reachable only
within the object it belongs to, so

<div class="ls-formula" markdown>
ECA = √( Σ<sub>i</sub> a<sub>i</sub>² ) &nbsp;&nbsp;&nbsp;&nbsp; coherence = 100 × ECA / Σ<sub>i</sub> a<sub>i</sub>
<small>a<sub>i</sub>: area of habitat object i. Coherence is 100% when all habitat is one object; restoration potential = 100 − coherence.</small>
</div>

```python
status = ls.restoration_status(forest)

object_areas = np.bincount(ids[ids > 0])[1:].astype(float)
object_areas = object_areas[object_areas > 0]
by_hand = 100 * np.sqrt((object_areas**2).sum()) / object_areas.sum()
{"coherence": status.doc, "by hand": by_hand, "restoration potential": status.rest_pot}
```

About half of the forest is effectively connected; the other half is the
restoration potential, what reconnecting every piece would add. Joining two
objects of areas *a* and *b* adds 2*ab* to the sum under the square root, so
connecting **large** objects gains the most.

## Comparing restoration options

Each candidate corridor has a gain (the increase in coherence) and a cost
(the non-habitat cells to restore, or their accumulated resistance). Compare
the three possible corridors between the three largest forest objects:

```python
from itertools import combinations

options = []
for first, second in combinations(largest, 2):
    corridor = ls.restoration_corridor(forest, resistance, ids == first, ids == second)
    options.append({"objects": f"{first} and {second}",
                    "cells to restore": corridor.rest_pixels,
                    "coherence gain (points)": corridor.delta_doc,
                    "gain per 100 cells": 100 * corridor.delta_doc / corridor.rest_pixels})

pd.DataFrame(options).round(2)
```

The options differ in both dimensions:

- joining the largest object with the third gives the **largest gain**
  (about 10 points), more than merging those two objects alone would give
  (about 6): the long corridor also joins other forest objects on its way.
  But it needs more than 500 cells of restoration;
- joining the largest object with the second gives a gain of 7 points for
  113 cells: the **most efficient** option;
- joining the second and the third is the cheapest (42 cells) but gains
  little, because neither object is large.

This is the logic of spatial restoration planning: rank options by the
function they add per unit of cost, and look for combinations that
complement each other (Tambosi et al., 2014). The same reasoning, with graph
indices instead of coherence, is available for candidate patches in
`HabitatGraph(..., restoration=flags)` ([chapter 7](networks.md)), and for
corridors through a species-specific resistance with the least-cost tools
above.

## Travel-time maps and zones of influence

Two further tools answer questions of proximity:

- **Isochrone maps** (`ls.isochrone_map`, `ls.isochrone_map_ab`) turn the
  accumulated cost from an area into rings of equal cost, like travel-time
  maps: which areas can a disperser from a reserve reach within a given
  effort?
- **Zones of influence** (`ls.influence_zones`) assign each background cell
  to the nearest large habitat object, and `ls.proximity` measures how close
  neighbouring objects are along the boundaries between zones. They show
  which object each part of the matrix "belongs" to, e.g. for planning
  buffer zones.

## Which tool for which question?

| Question | Tool |
|---|---|
| How far is every point from habitat, or from its edge? | `ls.euclidean_distance` |
| How costly is it to reach every point from given areas? | `ls.cost_distance`, `ls.isochrone_map` |
| What is the best route between two areas, and what would it cost to restore? | `ls.least_cost_path`, `ls.isochrone_map_ab` |
| How much would restoring that route add to connectivity? | `ls.restoration_corridor` |
| How connected is the habitat now, and what could it gain? | `ls.restoration_status` |
| Which patches would add the most connectivity if restored? | `HabitatGraph(..., restoration=...)` |
| Which habitat object is each point of the matrix closest to? | `ls.influence_zones`, `ls.proximity` |
| How much of every converted cell to restore, for species, carbon and intactness? | `ls.prioritize.restoration_problem` ([tutorial](../tutorials/restoration-plans.md)) |

!!! success "Key points"
    - Distance to habitat describes both edge exposure inside habitat and
      the potential for natural regeneration outside it.
    - Cost surfaces accumulate resistance along the cheapest route
      (Dijkstra's algorithm); least-cost corridors widen the single best path
      into a band of near-optimal routes.
    - Least-cost models suit directed movement; circuit models suit
      exploratory movement.
    - Joining large habitat objects gains the most connectivity; compare
      restoration options by gain and by gain per unit of cost.

## Check your understanding

1. Why may a restoration project close to existing forest be cheaper, beyond
   the price of land?
2. Two objects of 100 ha and one of 10 ha are separate. Which pair gains the
   most coherence when connected, and by how much does ECA² increase?
3. When would you trust a least-cost corridor less than a circuit-theory
   current map?

??? success "Answers"
    1. Because natural regeneration is more likely near seed sources, so the
       area may recover with little or no planting.
    2. The two 100 ha objects: ECA² increases by 2 × 100 × 100 = 20 000 ha²,
       against 2 000 ha² for a 100 ha and the 10 ha object.
    3. When the organism explores unfamiliar ground (juvenile dispersal) and
       cannot be assumed to know the best route, or when several routes are
       almost equally good and redundancy matters.

## References

<div class="ls-refs" markdown>

- Adriaensen, F., Chardon, J. P., De Blust, G., Swinnen, E., Villalba, S., Gulinck, H., & Matthysen, E. (2003). The application of 'least-cost' modelling as a functional landscape model. *Landscape and Urban Planning*, 64(4), 233–247.
- Beier, P., Majka, D. R., & Spencer, W. D. (2008). Forks in the road: choices in procedures for designing wildland linkages. *Conservation Biology*, 22(4), 836–851.
- Chazdon, R. L., & Guariguata, M. R. (2016). Natural regeneration as a tool for large-scale forest restoration in the tropics: prospects and challenges. *Biotropica*, 48(6), 716–730.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.
- Etherington, T. R. (2016). Least-cost modelling and landscape ecology: concepts, applications, and opportunities. *Current Landscape Ecology Reports*, 1, 40–53.
- Strassburg, B. B. N., Iribarrem, A., Beyer, H. L., et al. (2020). Global priority areas for ecosystem restoration. *Nature*, 586, 724–729.
- Tambosi, L. R., Martensen, A. C., Ribeiro, M. C., & Metzger, J. P. (2014). A framework to optimize biodiversity restoration efforts based on habitat amount and landscape connectivity. *Restoration Ecology*, 22(2), 169–177.

</div>
