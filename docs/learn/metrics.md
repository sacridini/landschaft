<p class="ls-chapter">Chapter 5</p>

# Landscape metrics

Landscape metrics are numbers that summarise the pattern of a categorical
map: how much of each class there is, how many patches, how large, how
complex, how clumped, how diverse. They let us compare landscapes, follow
them over time and relate pattern to ecological responses. They are also
easy to misuse: hundreds exist, many measure the same thing, and all depend
on the map. This chapter explains what the main metrics measure, computes a
few by hand, and gives rules for choosing and interpreting them.

!!! abstract "In this chapter"
    - composition and configuration; patches, classes and landscapes;
    - the families of metrics, with the formulas of the most used ones,
      checked by hand on a tiny map;
    - redundancy: why a handful of metrics carries most of the information;
    - common pitfalls, and a parsimonious set to start from;
    - sampling designs: whole maps, plots, windows and zones.

## A short history

Quantitative indices of landscape pattern appeared in the 1980s (O'Neill et
al., 1988). The program FRAGSTATS (McGarigal & Marks, 1995; McGarigal et
al., 2012) gathered them into a standard library with standard definitions,
and became the reference for thousands of studies. The R package
landscapemetrics reimplemented them in open source (Hesselbarth et al.,
2019), and added information-theory metrics of complexity (Nowosad &
Stepinski, 2019). Landschaft reproduces the definitions of landscapemetrics
and FRAGSTATS ([accuracy](../benchmarks.md#accuracy)), so its values can be
compared with the literature.

## Composition and configuration

Every metric describes one of two aspects of pattern (McGarigal & Marks,
1995; Gustafson, 1998):

- **Composition**: *what* is in the landscape and in which amounts, with no
  reference to where. The share of each class, the number of classes, their
  diversity. Composition does not change if you shuffle the cells.
- **Configuration**: *how* it is arranged in space. The size and shape of
  patches, their isolation, their aggregation or interspersion. Configuration
  changes when you shuffle the cells.

[Chapter 3](fragmentation.md) gave a reason to keep them apart: amount of
habitat (composition) and fragmentation *per se* (configuration) have
different ecological effects, and configuration metrics are often
correlated with composition.

## Patches, classes and landscapes

A **patch** is a group of connected cells of the same class. "Connected"
needs a rule: the **4-neighbour rule** joins cells sharing a side; the
**8-neighbour rule** also joins cells touching at a corner. The rule changes
the number and size of patches ([chapter 3](fragmentation.md#thresholds-and-percolation)),
so report it. FRAGSTATS and landscapemetrics default to 8, as does
Landschaft (`directions=8`).

Metrics exist at three **levels**:

| Level | One value per | Examples |
|---|---|---|
| patch | patch | area, perimeter, shape index, core area, nearest-neighbour distance |
| class | class | class area, percentage of landscape, number of patches, edge density, aggregation |
| landscape | map | total edge, diversity, contagion |

Class and landscape metrics are either computed directly from the map
(`ed`, `ai`, `shdi`) or summarise the patch metrics with a statistic:
mean (`_mn`), area-weighted mean (`_am`), median (`_md`), standard deviation
(`_sd`), coefficient of variation (`_cv`) and range (`_ra`); the
area-weighted mean, median and range need `extras=True`. The
area-weighted mean deserves attention: it describes the patch in which a
random cell of the class falls, which is often more meaningful ecologically
than the mean over patches, dominated by the many small ones.

## Metrics by hand

A tiny map of 6 × 6 cells of 10 m (0.01 ha each) with three classes is
enough to see how the main metrics work:

```python
import numpy as np
import pandas as pd
import landschaft as ls

toy = np.array([
    [1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
    [1, 1, 3, 3, 2, 2],
    [1, 1, 3, 3, 2, 2],
    [3, 1, 1, 1, 1, 1],
    [3, 3, 1, 1, 1, 1],
])
small = ls.Landscape(toy, res=10)
small.patch_metrics(["area", "perim"])
```

Four patches: one of class 1, one of class 2 and two of class 3 (the block
in the middle and the three cells in the corner, which do not touch).
Areas are in hectares, perimeters in metres.

### Area and edge

The percentage of the landscape (PLAND) is composition in its simplest form;
edge density (ED) is the length of the boundaries between classes per unit
area:

<div class="ls-formula" markdown>
PLAND<sub>i</sub> = 100 × A<sub>i</sub> / A  &nbsp;&nbsp;&nbsp;&nbsp; ED = E / A
<small>A<sub>i</sub>: area of class i; A: total area (ha); E: total length of edge (m) between cells of different classes, not counting the map border.</small>
</div>

Counting the edges between horizontally and vertically adjacent cells of
different classes gives the total edge:

```python
horizontal = (toy[:, 1:] != toy[:, :-1]).sum()
vertical = (toy[1:, :] != toy[:-1, :]).sum()
total_edge = 10 * (horizontal + vertical)                 # metres
total_area = toy.size * 0.01                              # hectares

pland_1 = 100 * (toy == 1).sum() / toy.size
{"TE": int(total_edge), "ED": total_edge / total_area, "PLAND class 1": pland_1}
```

```python
small.landscape_metrics(["te", "ed"])
```

The values match: 16 cell sides lie between different classes, 160 m of
edge in 0.36 ha, 444 m/ha. Edge density is among the most used metrics
because edges are where edge effects happen.

### Diversity

The **Shannon diversity index** (SHDI), from information theory (Shannon,
1948), combines the number of classes (richness) and how evenly the area is
shared among them:

<div class="ls-formula" markdown>
SHDI = − Σ<sub>i</sub> P<sub>i</sub> ln P<sub>i</sub>  &nbsp;&nbsp;&nbsp;&nbsp; SHEI = SHDI / ln m
<small>P<sub>i</sub>: proportion of the landscape in class i; m: number of classes. SHDI is 0 with a single class and grows with more, and more evenly shared, classes. SHEI, the evenness, is 1 when all classes have the same area.</small>
</div>

```python
p = np.array([(toy == c).mean() for c in (1, 2, 3)])
shdi = -(p * np.log(p)).sum()
{"SHDI by hand": shdi, "SHEI by hand": shdi / np.log(3)}
```

```python
small.landscape_metrics(["shdi", "shei", "pr"])
```

Diversity metrics are pure composition: shuffle the cells of the toy map
and SHDI stays 1.01.

### Aggregation

The **aggregation index** (AI) of a class compares the number of adjacencies
between its own cells with the maximum possible for that number of cells
(He et al., 2000):

<div class="ls-formula" markdown>
AI<sub>i</sub> = 100 × g<sub>ii</sub> / max g<sub>ii</sub>
<small>g<sub>ii</sub>: number of shared sides between cells of class i; max g<sub>ii</sub>: the number it would have if its cells formed the most compact shape possible.</small>
</div>

```python
small.class_metrics(["np", "ai"])
```

Class 2 scores 100: its ten cells are as compact as ten cells can be.
Class 3, split in two, scores lowest. AI is independent of the amount of the
class, which makes it useful to compare aggregation between classes and
landscapes with different amounts ([chapter 3](fragmentation.md)).

## The families of metrics

Beyond the examples above, metrics fall into six families. Landschaft
lists them with `ls.list_metrics()`, and every tool selects them by the
same `level`, `metric` and `type` filters.

```python
catalogue = ls.list_metrics()
catalogue.groupby(["type", "level"]).size().unstack(fill_value=0)
```

| Family | Measures | Key metrics |
|---|---|---|
| **area and edge** | size of patches and classes, amount of edge | `area`, `ca`, `pland`, `lpi`, `te`, `ed` |
| **shape** | complexity of patch outlines | `shape` (perimeter over the most compact perimeter, 1 = square), `frac` (fractal dimension), `para`, `circle`, `contig` |
| **core area** | interior habitat after removing an edge depth | `core`, `cai` (core as % of the patch), `tca`, `ndca` |
| **aggregation** | subdivision, isolation, clumping, interspersion | `np`, `pd`, `enn` (nearest-neighbour distance), `ai`, `clumpy`, `cohesion`, `division`, `mesh`, `split`, `contag`, `iji` |
| **diversity** | richness and evenness of classes | `pr`, `shdi`, `shei`, `sidi`, `siei` |
| **complexity** | information-theory measures of the whole pattern | `ent`, `condent`, `joinent`, `mutinf`, `relmutinf` |

A few deserve a sentence each:

- **Largest patch index** (`lpi`): the share of the landscape in the largest
  patch. Sensitive to percolation ([chapter 3](fragmentation.md)).
- **Euclidean nearest-neighbour distance** (`enn`): edge-to-edge distance
  from each patch to the nearest patch of the same class. The simplest
  isolation metric; it ignores everything but the nearest neighbour.
- **Patch cohesion** (`cohesion`): the physical connectedness of a class,
  shown to relate to the dispersal success of simulated organisms
  (Schumaker, 1996).
- **Contagion** (`contag`) and **interspersion and juxtaposition** (`iji`):
  whether classes form large clumps or are finely mixed (O'Neill et al.,
  1988; Li & Reynolds, 1993; McGarigal & Marks, 1995).
- **Complexity metrics** separate the diversity of classes (marginal
  entropy, `ent`) from the complexity of their arrangement (conditional
  entropy, `condent`) and their aggregation (mutual information, `mutinf`),
  in one consistent framework (Nowosad & Stepinski, 2019).
- **Core-area metrics** depend on an edge depth (`edge_depth`, in cells):
  choose it from the depth of edge effects, as for MSPA
  ([chapter 4](morphology.md#choosing-the-edge-width)).

## Redundancy: fewer metrics than you think

With more than 130 metrics available, it is tempting to compute them all.
But many are mathematically related, and in real landscapes most are
strongly correlated. Factor analyses of dozens of metrics over many
landscapes found that a handful of independent dimensions (about six or
seven) explain most of the variation: roughly amount, subdivision, edge,
shape, aggregation, isolation and diversity (Riitters et al., 1995; Cushman
et al., 2008).

We can see it on the example map, cut into a grid of 100 zones of 5 km:

```python
landcover = ls.load_raster(ls.example_data("clc"), validate=None)
zones = ls.grid_zones(landcover, 5000)
names = ["np", "pd", "te", "ed", "lsi", "lpi", "ai", "contag",
         "division", "mesh", "shdi", "sidi", "pr"]
per_zone = ls.zonal_metrics(landcover, zones, level="landscape", metric=names)

corr = per_zone[names].corr(method="spearman")
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print(f"{(upper.abs() > 0.7).sum()} of {len(upper)} pairs have |r| > 0.7")
upper[upper.abs() > 0.95].round(3)
```

Some pairs are identical by definition over zones of equal area: `np` and
`pd` (patch density is the number of patches per area), `te`, `ed` and
`lsi`, `division` and `mesh`. Others are nearly so because they measure
the same property: aggregation (`ai`) is the mirror image of edge
density, the largest patch (`lpi`) of division. Reporting all of them adds
columns, not information. `ls.calculate_correlation` and
`ls.show_correlation` compute and draw such matrices for any metric table.

## Pitfalls

Landscape metrics are easily misused (Li & Wu, 2004; Kupfer, 2012). The most
frequent problems:

1. **Metrics without a question.** A metric is useful when it measures a
   property linked to the process studied. Start from the process (dispersal,
   edge effects, disturbance spread), then pick the metric; not the reverse.
2. **Scale dependence.** Values change with grain, extent and legend
   ([chapter 2](scale.md)). Compare only maps with the same grain, extent
   and classes.
3. **Confounding with amount.** Many configuration metrics change with the
   amount of the class alone (Neel et al., 2004). Include the amount (`pland`)
   in analyses and interpret configuration conditional on it.
4. **Map error.** Misclassified cells create spurious patches and edges;
   patch counts and edge metrics are the most affected (Langford et al.,
   2006). Know the accuracy of the map, and smooth or filter very small
   patches if they are mostly noise.
5. **The map border.** Patches cut by the border look small and simple. Buffer
   the study area, or be consistent in how the border is treated
   (`count_boundary`, `consider_boundary`).
6. **No sampling distribution.** A metric is a descriptive value of one map;
   whether two values differ "significantly" requires replicate landscapes
   or a model of the pattern (Remmel & Csillag, 2003).
7. **Binary patch thinking.** Patch metrics assume sharp boundaries. For
   gradual landscapes, consider continuous (surface) measures (McGarigal et
   al., 2009) or the moving-window tools of [chapter 3](fragmentation.md).

## A parsimonious starting set

No list fits every study, but one metric per independent dimension is a
defensible start, to be adapted to the question:

| Dimension | Class level (one habitat) | Landscape level (the mosaic) |
|---|---|---|
| amount | `pland` | – |
| subdivision | `np` or `pd`, `mesh` | `np`, `mesh` |
| edge | `ed` | `ed` |
| aggregation | `ai` or `cohesion` | `contag` or `ai` |
| isolation | `enn_mn` (or `enn_am`) | – |
| shape | `shape_am` | `shape_am` |
| interior | `cai_am` (with a justified `edge_depth`) | – |
| diversity | – | `pr`, `shdi` |

Area-weighted means (`_am`) are FRAGSTATS statistics that landscapemetrics
does not have; ask for them with `extras=True`, e.g.
`Landscape(landcover).class_metrics(["shape_am", "cai_am"], extras=True)`.

## Sampling designs

The same metrics can be computed over different spatial units, each suited
to a kind of question:

| Design | Unit | Question | Tool |
|---|---|---|---|
| whole map | the study area | How does this landscape compare with others, or with itself later? | `Landscape`, `landscape_metrics` |
| plots | buffers around sites | How does the landscape around each site relate to what was observed there? | `sample_metrics`, `scale_sample`, `extract_metrics` |
| moving window | a window around every cell | Where does the structure change across the region? | `window_metrics` |
| zones | polygons or a grid | How do administrative units, watersheds or grid cells compare? | `zonal_metrics`, `grid_zones` |
| many landscapes | a set of maps or zones | What types of landscape exist, and what gradients separate them? | `SpatialSignatureAnalysis` |

Site-based designs are the core of landscape-scale ecological studies, and
their design deserves as much care as the metrics. Landscapes around sites
should not overlap (overlapping buffers share information, so they are not
independent samples), should span a wide range of the variables of
interest, and should decorrelate them as far as possible, for example
selecting landscapes where habitat amount and fragmentation vary
independently (Eigenbrod et al., 2011; Pasher et al., 2013). The buffer
size should be chosen by the multi-scale approach of
[chapter 2](scale.md#the-scale-of-effect).

!!! success "Key points"
    - Composition (how much) and configuration (how arranged) are distinct
      aspects of pattern; measure both, and interpret configuration given
      composition.
    - Patches depend on the neighbour rule; metrics exist at patch, class and
      landscape levels.
    - Most metrics are simple counts and ratios; computing a few by hand is
      the best way to understand them.
    - Metrics are highly redundant: choose a small set, one per dimension,
      guided by the process studied.
    - Metrics are scale-dependent and sensitive to map error; compare like
      with like.

## Check your understanding

1. Shuffling the cells of a map at random: which of `pland`, `shdi`, `ed`,
   `ai` and `np` change?
2. A study reports mean patch area (`area_mn`) of forest falling over time.
   Give two different landscape changes that could explain it.
3. Why is `area_am` often more informative than `area_mn`?

??? success "Answers"
    1. `pland` and `shdi` (composition) stay the same; `ed` increases, `ai`
       falls and `np` usually increases: configuration changes.
    2. Loss of habitat from large patches (shrinkage, or large patches split
       in two), or the appearance of many new small patches (new plantations,
       regrowth) that pull the mean down while the large patches stay intact.
       The mean alone cannot tell them apart; look at `np`, `lpi` and `pland`.
    3. The mean over patches is dominated by the many small patches; the
       area-weighted mean describes the patch where a random cell of habitat
       lies, i.e. the landscape as experienced by the habitat itself.

## References

<div class="ls-refs" markdown>

- Cushman, S. A., McGarigal, K., & Neel, M. C. (2008). Parsimony in landscape metrics: strength, universality, and consistency. *Ecological Indicators*, 8(5), 691–703.
- Eigenbrod, F., Hecnar, S. J., & Fahrig, L. (2011). Sub-optimal study design has major impacts on landscape-scale inference. *Biological Conservation*, 144(1), 298–305.
- Gustafson, E. J. (1998). Quantifying landscape spatial pattern: what is the state of the art? *Ecosystems*, 1(2), 143–156.
- He, H. S., DeZonia, B. E., & Mladenoff, D. J. (2000). An aggregation index (AI) to quantify spatial patterns of landscapes. *Landscape Ecology*, 15(7), 591–601.
- Hesselbarth, M. H. K., Sciaini, M., With, K. A., Wiegand, K., & Nowosad, J. (2019). landscapemetrics: an open-source R tool to calculate landscape metrics. *Ecography*, 42(10), 1648–1657.
- Kupfer, J. A. (2012). Landscape ecology and biogeography: rethinking landscape metrics in a post-FRAGSTATS landscape. *Progress in Physical Geography*, 36(3), 400–420.
- Langford, W. T., Gergel, S. E., Dietterich, T. G., & Cohen, W. (2006). Map misclassification can cause large errors in landscape pattern indices: examples from habitat fragmentation. *Ecosystems*, 9, 474–488.
- Li, H., & Reynolds, J. F. (1993). A new contagion index to quantify spatial patterns of landscapes. *Landscape Ecology*, 8(3), 155–162.
- Li, H., & Wu, J. (2004). Use and misuse of landscape indices. *Landscape Ecology*, 19, 389–399.
- McGarigal, K., & Marks, B. J. (1995). *FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure*. General Technical Report PNW-GTR-351, USDA Forest Service, Pacific Northwest Research Station.
- McGarigal, K., Cushman, S. A., & Ene, E. (2012). *FRAGSTATS v4: Spatial Pattern Analysis Program for Categorical and Continuous Maps*. University of Massachusetts, Amherst.
- McGarigal, K., Tagil, S., & Cushman, S. A. (2009). Surface metrics: an alternative to patch metrics for the quantification of landscape structure. *Landscape Ecology*, 24, 433–450.
- Neel, M. C., McGarigal, K., & Cushman, S. A. (2004). Behavior of class-level landscape metrics across gradients of class aggregation and area. *Landscape Ecology*, 19, 435–455.
- Nowosad, J., & Stepinski, T. F. (2019). Information theory as a consistent framework for quantification and classification of landscape patterns. *Landscape Ecology*, 34, 2091–2101.
- O'Neill, R. V., Krummel, J. R., Gardner, R. H., et al. (1988). Indices of landscape pattern. *Landscape Ecology*, 1(3), 153–162.
- Pasher, J., Mitchell, S. W., King, D. J., Fahrig, L., Smith, A. C., & Lindsay, K. E. (2013). Optimizing landscape selection for estimating relative effects of landscape variables on ecological responses. *Landscape Ecology*, 28, 371–383.
- Remmel, T. K., & Csillag, F. (2003). When are two landscape pattern indices significantly different? *Journal of Geographical Systems*, 5, 331–351.
- Riitters, K. H., O'Neill, R. V., Hunsaker, C. T., et al. (1995). A factor analysis of landscape pattern and structure metrics. *Landscape Ecology*, 10(1), 23–39.
- Schumaker, N. H. (1996). Using landscape indices to predict habitat connectivity. *Ecology*, 77(4), 1210–1225.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379–423.

</div>
