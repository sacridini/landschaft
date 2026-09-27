<p class="ls-chapter">Chapter 7</p>

# Habitat networks

Step back far enough from a landscape and the habitat patches become dots,
and the possibility of moving between them becomes lines. That picture is a
**graph**, the object of a branch of mathematics with two centuries of
results about paths, components and flows. Since the turn of the century,
graphs have become the most widely used framework to measure landscape
connectivity and to decide which patches and links to protect or restore
(Urban & Keitt, 2001; Saura & Pascual-Hortal, 2007).

!!! abstract "In this chapter"
    - graphs for ecologists: nodes, links, paths and components;
    - habitat availability: why connectivity should be measured as
      *reachable habitat*;
    - the integral index of connectivity (IIC), the probability of
      connectivity (PC) and the equivalent connected area (EC), computed by
      hand on a small network;
    - the importance of every patch and link, and what its three fractions
      mean;
    - choosing distances, and protected-area connectivity (ProtConn).

## Graphs for ecologists

A **habitat graph** has two ingredients (Urban & Keitt, 2001; Galpern et
al., 2011):

- **Nodes**: the habitat patches, each with an **attribute**, usually its
  area, sometimes its quality-weighted area or carrying capacity.
- **Links**: the connections between pairs of patches, weighted by a
  **distance**: edge to edge, centre to centre, or a cost distance through a
  resistance surface ([chapter 6](connectivity.md#resistance-surfaces)).

In a **binary** graph, two patches are linked when their distance is below a
threshold (the dispersal distance of the species). A **path** is a chain of
links, and a **component** is a group of patches all joined by paths: a
cluster of habitat an organism could, in principle, traverse by stepping
from patch to patch. In a **probabilistic** graph, every link carries a
probability of direct dispersal, derived from a kernel
([chapter 6](connectivity.md#dispersal)), and the connection between two
patches is the best product of probabilities along any path.

The strength of the approach is abstraction: once the graph is built,
connectivity questions become questions about the graph, quick to answer
even for thousands of patches, and easy to repeat with a patch or link
removed or added.

## Measuring connectivity as reachable habitat

Early graph indices counted structural properties: the number of
components, the size of the largest one, the length of the longest path.
They have a serious flaw: they can **improve when habitat is lost**. Remove
a small isolated patch and the number of components falls, which reads as
"more connected". Pascual-Hortal and Saura (2006) proposed instead to
measure **habitat availability** or **reachability**: the amount of habitat
an organism can reach, counting the habitat of the patch it is in and the
habitat of every other patch it can get to. This single idea integrates the
amount of habitat and its connectivity, consistent with the lessons of
[chapter 3](fragmentation.md#two-processes-behind-one-word), and has the
desirable property that losing habitat never increases connectivity.

### The integral index of connectivity

<div class="ls-formula" markdown>
IIC = Σ<sub>i</sub> Σ<sub>j</sub> a<sub>i</sub> a<sub>j</sub> / (1 + nl<sub>ij</sub>) &nbsp;/&nbsp; A<sub>L</sub>²
<small>a<sub>i</sub>, a<sub>j</sub>: attributes (areas) of patches i and j; nl<sub>ij</sub>: number of links in the shortest path between them (0 when i = j; pairs in different components contribute nothing); A<sub>L</sub>: area of the landscape. The sum runs over all ordered pairs, including i = j.</small>
</div>

IIC works on a binary graph: two patches either are or are not connected,
and pairs further apart in the network count less.

### The probability of connectivity

<div class="ls-formula" markdown>
PC = Σ<sub>i</sub> Σ<sub>j</sub> a<sub>i</sub> a<sub>j</sub> p<sup>∗</sup><sub>ij</sub> &nbsp;/&nbsp; A<sub>L</sub>²
<small>p<sup>∗</sup><sub>ij</sub>: the maximum product probability of all paths between i and j (1 when i = j), from direct probabilities p<sub>ij</sub> = e<sup>−k d<sub>ij</sub></sup>.</small>
</div>

PC is the probability that two points placed at random in the landscape
fall in habitat areas that are reachable from each other (Saura &
Pascual-Hortal, 2007). Its numerator, PCnum, is the reachable habitat.
Because PC depends on the area of the whole landscape, its values are
often tiny and hard to read. The **equivalent connected area** is more
intuitive (Saura et al., 2011a):

<div class="ls-formula" markdown>
EC(PC) = √PCnum
<small>The area of a single patch that would provide the same amount of reachable habitat as the whole network. It lies between the area of the largest patch and the total habitat area.</small>
</div>

### Try it: PC by hand

Four patches with areas in hectares and edge-to-edge distances in metres,
and a species with a 50% chance of crossing 1 km:

```python
import numpy as np
import pandas as pd
import landschaft as ls

areas = np.array([100.0, 50.0, 20.0, 80.0])
d = np.array([
    [0,    800,  1500, 5000],
    [800,  0,    600,  4000],
    [1500, 600,  0,    3000],
    [5000, 4000, 3000, 0],
], dtype=float)

k = -np.log(0.5) / 1000.0
p = np.exp(-k * d)                       # direct dispersal probabilities
```

The best path between two patches may go through others (1 → 2 → 3 is
more probable than 1 → 3 directly). The maximum product probability is a
shortest-path problem on −ln p, solved here with the Floyd–Warshall
algorithm:

```python
cost = -np.log(p)
for m in range(len(areas)):                      # allow paths through patch m
    cost = np.minimum(cost, cost[:, [m]] + cost[[m], :])
p_star = np.exp(-cost)

pc_num = (areas[:, None] * areas[None, :] * p_star).sum()
landscape_area = 10_000.0                        # hectares
{"PCnum": pc_num, "PC": pc_num / landscape_area**2, "EC(PC)": np.sqrt(pc_num)}
```

```python
toy = ls.HabitatGraph(areas, d, landscape_area=landscape_area)
toy.overall(["PC"], distance=1000, probability=0.5)[["PCnum", "PC", "EC(PC)"]]
```

The library agrees. EC(PC) = 172 ha: the network of 250 ha behaves like a
single patch of 172 ha, larger than the largest patch (100 ha) because
patches 1, 2 and 3 are close enough to share much of their habitat, smaller
than the total because the links are imperfect and patch 4 is far away.

## Which patches matter?

The importance of a patch is the percentage of connectivity lost when it is
removed (Keitt et al., 1997; Saura & Pascual-Hortal, 2007):

<div class="ls-formula" markdown>
dPC<sub>k</sub> = 100 × (PC − PC<sub>remove k</sub>) / PC
</div>

```python
def pc_numerator(areas, cost):
    for m in range(len(areas)):
        cost = np.minimum(cost, cost[:, [m]] + cost[[m], :])
    return (areas[:, None] * areas[None, :] * np.exp(-cost)).sum()


full = pc_numerator(areas, -np.log(p))
by_hand = []
for k_removed in range(len(areas)):
    keep = np.arange(len(areas)) != k_removed
    rest = pc_numerator(areas[keep], -np.log(p[np.ix_(keep, keep)]))
    by_hand.append(100 * (full - rest) / full)

importance = toy.node_importance("PC", distance=1000, probability=0.5)
importance.assign(dPC_by_hand=by_hand)[["id", "dPC", "dPC_by_hand", "dPCintra", "dPCflux", "dPCconnector"]]
```

Saura and Rubio (2010) showed that dPC splits into three fractions, which
are three different ways in which a patch contributes:

| Fraction | Contribution | Depends on |
|---|---|---|
| **dPCintra** | the habitat inside the patch | its area only |
| **dPCflux** | the dispersal flow starting or ending in the patch | its area and position |
| **dPCconnector** | its role as a stepping stone on the best paths between *other* patches | its position only |

In the toy network, patch 1, the largest, matters mostly through its own
area (dPCintra). Patch 3, the smallest, has the largest connector fraction:
the best paths from patches 1 and 2 to the distant patch 4 go through it.
Patch 2 is also a connector, on the best path from 1 to 3; patches 1 and 4
lie at the ends of the chain and connect nobody.

### On a real landscape

Build the forest network of the example map, with edge-to-edge distances:

```python
landcover = ls.load_raster(ls.example_data("clc"), validate=None)
forest_net = ls.HabitatGraph.from_raster(
    landcover, habitat=[23, 24, 25], area_unit="ha",
    distance="edge", max_distance=10_000,
    landscape_area=landcover.size * 25 * 25 / 1e4,
)
total_forest = forest_net.habitat_area                  # 51 538 ha

overall = forest_net.overall(["NC", "PC"], distance=[250, 500, 1000, 2000, 5000, 10_000], probability=0.5)
overall.assign(**{"EC as % of forest": 100 * overall["EC(PC)"] / total_forest})[
    ["distance", "NC", "EC(PC)", "EC as % of forest", "PCstep(%)"]].round(1)
```

For a species that crosses 250 m of farmland half of the time, the 158
forest patches form 110 components and the reachable forest is less than
30% of the total; for one that crosses 5 km, two components and two thirds.
`PCstep(%)` is the share of connectivity that depends on stepping stones: it
grows from 4% to more than half. The same forest is, for different species,
a fragmented set of woodlots or a well-connected network.

Rank the patches at 2 km:

```python
patch_area = dict(zip(forest_net.ids.tolist(), forest_net.attributes.tolist()))
ranking = forest_net.node_importance("PC", distance=2000, probability=0.5)
ranking["area_ha"] = ranking["id"].map(patch_area)
ranking.sort_values("dPCconnector", ascending=False)[
    ["id", "area_ha", "dPC", "dPCintra", "dPCflux", "dPCconnector"]].head().round(2)
```

Sorted by connector fraction, the list shows patches that an area-based
ranking would overlook: patch 84, of 47 ha (smaller than the median forest
patch), carries 3.5% of the connectivity of the whole network, almost all
of it as a stepping stone. Losing it would cost more connectivity than
losing any of eight patches more than ten times its size. Maps of these fractions
(`forest_net.to_raster(ranking, "dPCconnector")`) are among the most
useful products for conservation planning.

`link_importance` does the same for links: the loss of connectivity if a
link were cut, e.g. by a new road. `centrality` returns betweenness and
other network centralities, which rank patches by how many shortest paths
cross them (Bodin & Saura, 2010).

## Choosing distances and probabilities

The distance and probability parameters describe the species:

- The usual choice is the **median dispersal distance** with probability
  0.5, or a **maximum dispersal distance** with a small probability such as
  0.05. They define the same exponential kernel in different ways.
- When no single species is targeted, analyse a range of distances (for
  example 250 m to 10 km, as above) to represent a range of species.
- Use **edge-to-edge** distances for patches that dispersers leave from
  anywhere along their edge, **centroid** distances for point-like nodes, and
  **least-cost** or **commute-time** distances when the matrix is
  heterogeneous (`distance="least-cost"`, `"commute-time"`).
- Set `max_distance` above the largest distance with a non-negligible
  probability, since pairs beyond it are left out.
- Patches cut by the border of the study area lose part of their neighbours;
  where possible, include a buffer around the study area.

## Protected-area connectivity

Protected areas are more effective as a network than as isolated units. The
**ProtConn** indicator is the percentage of a region covered by protected
land that is also connected (Saura et al., 2017, 2018). It is based on the
same equivalent-connected-area logic, with protected areas as nodes and the
region as the landscape, and it splits the unconnected remainder into
fractions (protected areas that are isolated, connected only through
unprotected land, or through other regions). ProtConn is used to report
progress towards international targets that ask for protected-area systems
to be *well connected*, from the Aichi Target 11 (17% of land by 2020) to
Target 3 of the Kunming–Montreal Global Biodiversity Framework (30% by
2030). See `ls.conn.protconn` and the
[habitat networks tutorial](../tutorials/habitat-networks.md#connectivity-of-protected-areas-protconn).

## Structural networks from MSPA

MSPA ([chapter 4](morphology.md)) already contains a graph: cores are nodes
and bridges are links. `ls.gtn_components` and `ls.gtn_importance` analyse
that structural network directly on the map, and `ls.conefor_inputs` exports
it as node and distance files. Combining MSPA's structural connectors with
graph indices is an effective way to identify key corridors (Saura et al.,
2011b).

!!! success "Key points"
    - A habitat graph abstracts patches as nodes with an attribute and
      connections as links with a distance or probability.
    - Measuring connectivity as reachable habitat (IIC, PC) integrates habitat
      amount and connectivity, and never rewards habitat loss.
    - EC(PC) is the size of a single patch equivalent to the whole network,
      between the largest patch and the total habitat.
    - The importance of a patch (dPC) has three fractions: intra (its area),
      flux (its area and position) and connector (its position as a stepping
      stone). Small patches can be critical connectors.
    - Distances and probabilities describe the species; analyse a range when
      in doubt.

## Check your understanding

1. Why can the number of components be a misleading connectivity index?
2. A network has EC(PC) equal to the area of its largest patch. What does
   that mean?
3. A patch has a high dPCconnector and a negligible dPCintra. What would
   you recommend for it, and why?

??? success "Answers"
    1. Because removing an isolated patch reduces the number of components,
       which reads as better connectivity although habitat was lost.
    2. The other patches add (almost) nothing reachable: they are too small
       or too isolated from each other and from the largest patch, so the
       network is effectively a single patch.
    3. Protect it, and consider it for restoration or enlargement: it adds
       little habitat itself but holds the network together as a stepping
       stone, so its loss would disconnect larger patches.

## References

<div class="ls-refs" markdown>

- Bodin, Ö., & Saura, S. (2010). Ranking individual habitat patches as connectivity providers: integrating network analysis and patch removal experiments. *Ecological Modelling*, 221(19), 2393–2405.
- Galpern, P., Manseau, M., & Fall, A. (2011). Patch-based graphs of landscape connectivity: a guide to construction, analysis and application for conservation. *Biological Conservation*, 144(1), 44–55.
- Keitt, T. H., Urban, D. L., & Milne, B. T. (1997). Detecting critical scales in fragmented landscapes. *Conservation Ecology*, 1(1), 4.
- Pascual-Hortal, L., & Saura, S. (2006). Comparison and development of new graph-based landscape connectivity indices: towards the priorization of habitat patches and corridors for conservation. *Landscape Ecology*, 21, 959–967.
- Saura, S., Bastin, L., Battistella, L., Mandrici, A., & Dubois, G. (2017). Protected areas in the world's ecoregions: how well connected are they? *Ecological Indicators*, 76, 144–158.
- Saura, S., Bertzky, B., Bastin, L., Battistella, L., Mandrici, A., & Dubois, G. (2018). Protected area connectivity: shortfalls in global targets and country-level priorities. *Biological Conservation*, 219, 53–67.
- Saura, S., Estreguil, C., Mouton, C., & Rodríguez-Freire, M. (2011a). Network analysis to assess landscape connectivity trends: application to European forests (1990–2000). *Ecological Indicators*, 11(2), 407–416.
- Saura, S., & Pascual-Hortal, L. (2007). A new habitat availability index to integrate connectivity in landscape conservation planning: comparison with existing indices and application to a case study. *Landscape and Urban Planning*, 83(2–3), 91–103.
- Saura, S., & Rubio, L. (2010). A common currency for the different ways in which patches and links can contribute to habitat availability and connectivity in the landscape. *Ecography*, 33(3), 523–537.
- Saura, S., & Torné, J. (2009). Conefor Sensinode 2.2: a software package for quantifying the importance of habitat patches for landscape connectivity. *Environmental Modelling & Software*, 24(1), 135–139.
- Saura, S., Vogt, P., Velázquez, J., Hernando, A., & Tejera, R. (2011b). Key structural forest connectors can be identified by combining landscape spatial pattern and network analyses. *Forest Ecology and Management*, 262(2), 150–160.
- Urban, D., & Keitt, T. (2001). Landscape connectivity: a graph-theoretic perspective. *Ecology*, 82(5), 1205–1218.

</div>
