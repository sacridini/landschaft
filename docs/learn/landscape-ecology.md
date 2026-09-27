<p class="ls-chapter">Chapter 1</p>

# What is a landscape?

A forest bird, a beetle and a wolf can live in the same valley and yet
inhabit very different worlds. To the beetle, a road is an impassable
desert; to the wolf it is a nuisance crossed in a few seconds. What they share
is that **where** things are matters as much as **how much** of them there
is. Landscape ecology is the science of that "where": of how the spatial
arrangement of the environment shapes ecological processes, and of how those
processes, in turn, reshape the arrangement.

!!! abstract "In this chapter"
    - the origins and the central paradigm of landscape ecology;
    - the patch–corridor–matrix model and its alternatives;
    - two theories from biogeography that explain why spatial structure
      matters: island biogeography and metapopulations;
    - why *habitat* depends on the organism;
    - how a landscape becomes a map, and the kinds of maps Landschaft reads.

## A science of spatial heterogeneity

The term *Landschaftsökologie*, landscape ecology, was coined by the German
geographer Carl Troll in 1939, while interpreting aerial photographs: seen
from above, vegetation, soils, water and land use formed mosaics whose parts
clearly influenced one another (Troll, 1939). The name of this package comes
from the same word, *Landschaft*.

For decades the field grew mostly in Europe, close to geography, land-use
planning and the study of cultural landscapes. In the 1980s a second
tradition emerged in North America, centred on **spatial heterogeneity**: how
to quantify it, what causes it and what its consequences are (Risser et al.,
1984; Forman & Godron, 1986; Turner, 1989). The two traditions have since
converged (Wu & Hobbs, 2002), and the quantitative toolbox of this course
comes mostly from the second one.

What, then, is a landscape? Not a fixed size. A widely used definition is
*an area that is spatially heterogeneous in at least one factor of interest*
(Turner & Gardner, 2015). A landscape can be a few square metres for a
ground beetle and thousands of square kilometres for a migratory ungulate.
The organism or process under study, not the observer, sets the scale
(Wiens, 1989), an idea we develop in [chapter 2](scale.md).

## Pattern, process and scale

The central paradigm of landscape ecology links three words (Turner, 1989):

- **Pattern** is the spatial arrangement of the landscape: the amount of
  each land cover and how it is distributed, broken up, clumped or
  interspersed.
- **Process** is what happens in it: the dispersal of seeds and animals, the
  spread of fire, pests and invasive species, the flow of water and nutrients,
  the colonisation and extinction of populations.
- **Scale** is the spatial and temporal dimension at which pattern and
  process are observed.

Pattern and process form a feedback loop. Pattern constrains process: a
continuous forest lets a fire run while a mosaic of fields and woods stops
it; a line of hedges channels the movement of small mammals. Process creates
pattern: fire, logging, agriculture and urban growth are what break a forest
into pieces in the first place. Both depend on scale: a landscape that looks
fragmented at 10 m resolution can look continuous at 1 km.

Most of this course is about describing **pattern** with numbers and maps.
Keep in mind why we do it: pattern matters because of the processes it
controls. A metric that cannot be linked to a process is a number without an
ecological meaning (Li & Wu, 2004).

## Patches, corridors and the matrix

The most influential mental model of a landscape is the
**patch–corridor–matrix** model (Forman & Godron, 1986; Forman, 1995):

| Element | Definition | Example |
|---|---|---|
| **Patch** | a relatively homogeneous area that differs from its surroundings | a woodlot in farmland, a pond, a town |
| **Corridor** | a strip that differs from the land on both sides | a hedgerow, a riparian forest, a road |
| **Matrix** | the most extensive and connected element, which controls the dynamics of the landscape | the farmland around the woodlots |

The model is simple, visual and useful: it names the elements whose size,
shape, number and arrangement landscape metrics quantify, and it frames
conservation questions in terms everyone understands (keep large patches,
maintain corridors, soften the matrix).

It also has limits. It treats the landscape as sharp-edged pieces of
homogeneous habitat in a hostile background, as islands in an ocean. Real
landscapes often change gradually: tree cover thins out, grazing pressure
varies continuously, the matrix itself offers food and shelter to many
species. The **gradient model** (McGarigal & Cushman, 2005) and the
**variegation model** (McIntyre & Hobbs, 1999) describe landscapes as
continuous surfaces rather than patches. Landschaft covers both views:
patch-based tools on categorical maps, and tools that work on continuous
maps (percentage of tree cover, resistance to movement) where the gradient
view fits better.

## Two theories that explain why structure matters

Why should the arrangement of habitat affect how many species live in a
landscape, or whether a population persists? Two theories from biogeography
and population ecology give the classic answers.

### Island biogeography

MacArthur and Wilson (1967) proposed that the number of species on an
island is a dynamic balance between **immigration** of new species and
**extinction** of those present. Immigration falls with the island's
**isolation** (distance from the mainland); extinction falls with its
**area** (larger islands hold larger populations, which are less likely to
die out by chance). Large, near islands therefore hold more species than
small, remote ones.

Conservation biologists soon read habitat fragments as islands in a sea of
farmland, and derived rules for reserve design (large rather than small,
close rather than far, connected rather than isolated). The analogy is
imperfect: the matrix is not an ocean, it can be crossed, it supplies
species of its own and it drives edge effects deep into the fragments
(Laurance, 2008). But the two variables of the theory, **area** and
**isolation**, remain the backbone of fragmentation research, and many
landscape metrics measure one or the other.

### Metapopulations

A metapopulation is a set of local populations in separate habitat patches,
linked by dispersal (Levins, 1969; Hanski, 1998). Local populations go
extinct from time to time; empty patches are recolonised by dispersers from
occupied ones. The metapopulation persists as long as colonisation keeps pace
with extinction, which depends on the **size** of the patches (extinction)
and their **connectivity** (colonisation). A patch that is too isolated stays
empty even when it is perfectly good habitat.

This is the theoretical root of the connectivity chapters of this course
([6](connectivity.md), [7](networks.md), [8](circuits.md)): the indices used
to rank patches and links for conservation are, in essence, estimates of how
much habitat an organism can reach.

## Habitat is species-specific

"Habitat" is not a land-cover class. It is the set of resources and
conditions that allow a given species to live and reproduce (Hall et al.,
1997). A broad-leaved forest is habitat for a woodpecker, a barrier for a
grassland butterfly and irrelevant for a generalist fox. Two consequences
follow for every analysis in this course:

1. **The habitat map is a hypothesis.** Deciding which classes count as
   habitat, and for which species, is the first and most consequential
   decision of a study. Justify it with the natural history of the species.
2. **One landscape, many patterns.** The same land-cover map gives different
   patches, fragmentation and connectivity for different species, as the
   example below shows.

## From a landscape to a map

Landscape analyses work on **maps**, and a map is itself a model: somebody
decided the classes, the resolution and the extent, and a classifier or an
interpreter assigned every cell to a class with some error. Errors in the
map propagate into every metric computed from it, sometimes strongly
(Langford et al., 2006). Know your map: its source, date, resolution, legend
and accuracy.

Landschaft works on **raster** maps: grids of cells, each holding a number.
Four kinds appear in this course ([core concepts](../getting-started/concepts.md)
has the technical details):

| Map | Cells hold | Used for | Chapters |
|---|---|---|---|
| **categorical** | a class code (land cover) | landscape metrics | 5, 10 |
| **binary habitat** | 0 missing, 1 background, 2 habitat | patterns, fragmentation, change | 3, 4, 9, 10 |
| **continuous** | a quantity (% tree cover, resistance) | grayscale fragmentation, connectivity | 3, 6, 8 |
| **patch** | one id per habitat patch | habitat networks | 7 |

### Try it: one map, two species

The example land-cover map is a 50 × 50 km extract of CORINE Land Cover, a
European map with 44 classes, at 25 m resolution:

```python
import numpy as np
import landschaft as ls

landcover = ls.load_raster(ls.example_data("clc"), validate=None)
landcover.shape, landcover.rio.resolution(), landcover.rio.crs
```

Which classes are present, and how much of the map does each cover?

```python
codes, cells = np.unique(landcover.values, return_counts=True)
share = dict(zip(codes.tolist(), (100 * cells / cells.sum()).round(1).tolist()))
share
```

Half of the map is class 12 (non-irrigated arable land): an agricultural
landscape. Forests (23 broad-leaved, 24 coniferous, 25 mixed) cover about
20%, pastures (18) about 7%.

Now build the habitat of two imaginary species. A forest bird lives in the
three forest classes; a grassland insect in pastures (18) and natural
grassland (26):

```python
forest = ls.reclassify(landcover, foreground=[23, 24, 25])
grassland = ls.reclassify(landcover, foreground=[18, 26])

for name, habitat in [("forest bird", forest), ("grassland insect", grassland)]:
    table = ls.Landscape(habitat).class_metrics(["pland", "np", "area_mn"])
    row = table[table["class"] == 2].iloc[0]          # class 2 = habitat
    print(f"{name:17s} {row.pland:5.1f}% of the map in {row.np:.0f} patches "
          f"of {row.area_mn:.0f} ha on average")
```

The same landscape offers the forest bird three times as much habitat as the
grassland insect, in patches almost three times larger. Every later analysis
(fragmentation, connectivity, restoration) would reach different conclusions
for the two species. `pland`, `np` and `area_mn` are
[landscape metrics](metrics.md): the percentage of the landscape, the number
of patches and their mean area.

## Three ways of seeing a landscape

The tools of Landschaft look at a landscape in three complementary ways,
which organise the rest of the course:

| View | The landscape is... | Questions | Chapters |
|---|---|---|---|
| **cells and neighbourhoods** | a grid; each cell is described by what surrounds it | Is this cell interior or edge? How dense is habitat around it? | [3](fragmentation.md), [4](morphology.md) |
| **patches** | a set of discrete objects with area, shape and position | How many patches? How large, complex, aggregated, diverse? | [5](metrics.md) |
| **networks and flows** | nodes linked by movement, or a surface that conducts it | How much habitat is reachable? Which patches and routes matter? | [6](connectivity.md)–[9](restoration.md) |

No view is the right one in general: each answers different questions, and
a good study often combines them.

!!! success "Key points"
    - Landscape ecology studies how spatial pattern affects ecological
      process, and vice versa, at scales set by the organism or process.
    - The patch–corridor–matrix model names the elements that landscape
      metrics quantify; gradient models describe landscapes that change
      continuously.
    - Island biogeography and metapopulation theory explain why the **area**
      and **isolation** of habitat, and hence its arrangement, matter.
    - Habitat is species-specific: the habitat map is the first hypothesis
      of any study.
    - Maps are models with errors; know their source, resolution and legend.

## Check your understanding

1. A colleague says a 5 km² area "is too small to be a landscape". How would
   you answer?
2. Why is a patch of farmland not equivalent to an ocean for a forest
   species? Give two reasons.
3. In the example above, which species' habitat is more fragmented? Which
   single metric would you look at first, and why is it not enough?

??? success "Answers"
    1. A landscape is defined by heterogeneity relevant to the organism or
       process, not by size. For a small mammal or an insect, 5 km² contains
       many habitat patches and a complex matrix; it is a landscape.
    2. The matrix can be crossed (with some cost or risk), and it is not
       neutral: it provides resources to some species, it is a source of
       predators, competitors and invasive species, and it causes edge effects
       inside the fragments.
    3. The grassland insect has less habitat, in more patches relative to its
       area and in smaller ones. Mean patch area alone mixes two things: the
       amount of habitat and how it is subdivided. [Chapter 3](fragmentation.md)
       explains why those two must be separated.

## References

<div class="ls-refs" markdown>

- Forman, R. T. T. (1995). *Land Mosaics: The Ecology of Landscapes and Regions*. Cambridge University Press.
- Forman, R. T. T., & Godron, M. (1986). *Landscape Ecology*. Wiley.
- Hall, L. S., Krausman, P. R., & Morrison, M. L. (1997). The habitat concept and a plea for standard terminology. *Wildlife Society Bulletin*, 25(1), 173–182.
- Hanski, I. (1998). Metapopulation dynamics. *Nature*, 396, 41–49.
- Langford, W. T., Gergel, S. E., Dietterich, T. G., & Cohen, W. (2006). Map misclassification can cause large errors in landscape pattern indices: examples from habitat fragmentation. *Ecosystems*, 9, 474–488.
- Laurance, W. F. (2008). Theory meets reality: how habitat fragmentation research has transcended island biogeographic theory. *Biological Conservation*, 141(7), 1731–1744.
- Levins, R. (1969). Some demographic and genetic consequences of environmental heterogeneity for biological control. *Bulletin of the Entomological Society of America*, 15(3), 237–240.
- Li, H., & Wu, J. (2004). Use and misuse of landscape indices. *Landscape Ecology*, 19, 389–399.
- MacArthur, R. H., & Wilson, E. O. (1967). *The Theory of Island Biogeography*. Princeton University Press.
- McGarigal, K., & Cushman, S. A. (2005). The gradient concept of landscape structure. In J. A. Wiens & M. R. Moss (Eds.), *Issues and Perspectives in Landscape Ecology* (pp. 112–119). Cambridge University Press.
- McIntyre, S., & Hobbs, R. (1999). A framework for conceptualizing human effects on landscapes and its relevance to management and research models. *Conservation Biology*, 13(6), 1282–1292.
- Risser, P. G., Karr, J. R., & Forman, R. T. T. (1984). *Landscape Ecology: Directions and Approaches*. Special Publication 2, Illinois Natural History Survey.
- Troll, C. (1939). Luftbildplan und ökologische Bodenforschung. *Zeitschrift der Gesellschaft für Erdkunde zu Berlin*, 241–298.
- Turner, M. G. (1989). Landscape ecology: the effect of pattern on process. *Annual Review of Ecology and Systematics*, 20, 171–197.
- Turner, M. G., & Gardner, R. H. (2015). *Landscape Ecology in Theory and Practice: Pattern and Process* (2nd ed.). Springer.
- Wiens, J. A. (1989). Spatial scaling in ecology. *Functional Ecology*, 3(4), 385–397.
- Wu, J., & Hobbs, R. (2002). Key issues and research priorities in landscape ecology: an idiosyncratic synthesis. *Landscape Ecology*, 17, 355–365.

</div>
