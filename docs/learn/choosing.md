<p class="ls-chapter">Chapter 15</p>

# From question to tool

The previous chapters went from concepts to tools. A real project goes the
other way: it starts with a question, often vague, and has to arrive at a
few well-chosen analyses whose results can be defended. This chapter gives a
workflow, a decision guide from questions to tools, a worked study design
and a checklist of what to report.

!!! abstract "In this chapter"
    - a seven-step workflow for landscape analyses;
    - a decision guide: which question, which tool, which chapter;
    - how the main approaches compare;
    - a worked example, from the question to the report;
    - a reporting checklist and the most common mistakes.

## A workflow in seven steps

1. **The question and the process.** State what you want to know and which
   ecological process links pattern to the answer: dispersal, edge effects,
   disturbance spread, habitat selection. The process decides everything
   else ([chapter 1](landscape-ecology.md#pattern-process-and-scale)).
2. **The organism and its habitat.** Define habitat for the species or
   group studied, with references to its natural history
   ([chapter 1](landscape-ecology.md#habitat-is-species-specific)), or
   model it from the species' records ([chapter 13](sdm.md)).
3. **The map.** Choose a map whose legend separates that habitat, whose
   grain resolves the features that matter and whose extent contains the
   process. Know its date, minimum mapping unit and accuracy
   ([chapter 2](scale.md)).
4. **The scales.** Translate the biology into parameters: edge depth
   (edge widths), dispersal distances (network distances, circuit radii),
   neighbourhood sizes (windows, buffers). Convert them to cells at the
   map's grain.
5. **The analyses.** Pick the fewest tools that answer the question, using
   the guide below. Measure habitat amount together with configuration
   ([chapter 3](fragmentation.md)).
6. **Sensitivity.** Repeat the key analyses with the uncertain parameters
   varied over a plausible range: edge width, dispersal distance, resistance
   values, map classes. Keep conclusions that are robust to them.
7. **Interpretation and report.** Interpret results through the process of
   step 1, and report everything needed to reproduce them (see the
   [checklist](#a-reporting-checklist)).

## A decision guide

| If you want to know... | Use | Chapter |
|---|---|---|
| **Amount and composition** | | |
| how much habitat, and of which kinds | `pland`, `ca` (landscape metrics) | [5](metrics.md) |
| how diverse the land cover is | `pr`, `shdi`, `shei` | [5](metrics.md) |
| how much habitat surrounds every point, at one or many scales | `ls.fad`, `ls.fad_multiscale` | [3](fragmentation.md) |
| what land uses surround the habitat | `ls.landscape_mosaic` | [3](fragmentation.md) |
| **Configuration and fragmentation** | | |
| how many patches, how large, how isolated | `np`, `area_mn`/`area_am`, `lpi`, `enn` | [5](metrics.md) |
| how divided the habitat is | `mesh`, `division`, `ls.parcellation` | [4](morphology.md) |
| how aggregated or edge-exposed it is | `ai`, `ed`, `ls.fos`, `ls.contagion_pff` | [3](fragmentation.md), [5](metrics.md) |
| how much interior habitat there is | MSPA core, core-area metrics | [4](morphology.md), [5](metrics.md) |
| which pixels are corridors, edges or fragments | `ls.mspa`, `ls.spa` | [4](morphology.md) |
| how habitat is shared among small and large objects | `ls.accounting` | [4](morphology.md) |
| **Connectivity** | | |
| how much habitat is reachable for a species | `HabitatGraph.overall` (PC, EC) | [7](networks.md) |
| which patches and links matter most | `node_importance`, `link_importance` | [7](networks.md) |
| how well protected areas are connected | `ls.conn.protconn` | [7](networks.md) |
| where movement concentrates; pinch points and barriers | `ls.omniscape`, `circuit.circuitscape_pairwise` | [8](circuits.md) |
| how isolated populations are from each other | effective resistances (pairwise) | [8](circuits.md) |
| **Planning** | | |
| the best route between two areas | `ls.least_cost_path`, least-cost corridors | [9](restoration.md) |
| what restoring a corridor or a patch would gain | `ls.restoration_corridor`, `HabitatGraph(restoration=...)` | [9](restoration.md), [7](networks.md) |
| where natural regeneration is likely | `ls.euclidean_distance` | [9](restoration.md) |
| which places matter most for many species at once | `ls.prioritize.zonation` | [12](prioritization.md) |
| how much of every species a share of the land would protect | performance curves (`feature_curves`) | [12](prioritization.md) |
| what existing reserves miss, and how to complete them | hierarchic mask, gap analysis | [12](prioritization.md) |
| **Species distributions** | | |
| where a species could live, from records and environmental maps | `ls.sdm.prepare`, `ls.sdm.fit`, `ls.sdm.predict` | [13](sdm.md) |
| how well a distribution model predicts new places | spatial folds, `ls.sdm.evaluate` (AUC, TSS, Boyce, omission) | [13](sdm.md) |
| where a projection to another climate or region extrapolates | `ls.sdm.mess`, `ls.sdm.mop`, `ls.sdm.exdet` | [13](sdm.md) |
| **Change** | | |
| what was gained and lost, and whether it is real | `ls.simple_change`, `ls.morphological_change` | [10](change.md) |
| how fragmentation classes shifted | `ls.fos_change`, `ls.fad_multiscale_change` | [10](change.md) |
| whether connectivity fell faster than habitat | `ls.conn.eca_change` | [10](change.md) |
| **Many places or dates** | | |
| how landscapes around sites relate to field data | `ls.sample_metrics`, `ls.scale_sample` | [2](scale.md), [5](metrics.md) |
| how structure varies across a region | `ls.window_metrics`, `ls.zonal_metrics` | [5](metrics.md) |
| which types of landscape exist | `ls.SpatialSignatureAnalysis` | [5](metrics.md) |
| how metrics evolve over a time series | `ls.SpatioTemporalAnalysis` | [10](change.md) |
| **Future landscapes** | | |
| where land use is heading if the observed trend continues | `ls.simulate.run` (Markov demand, potential, allocation) | [14](simulation.md) |
| how scenarios (demands, protected areas, policies) differ in space | `ls.simulate.scenarios`, `Ensemble.probability` | [14](simulation.md) |
| whether a simulation reproduces the observed change | `ls.simulate.validate` (figure of merit, three maps, fuzzy similarity) | [14](simulation.md) |

## How the main approaches compare

| Approach | Input | Output | Main assumption | Strength | Limitation |
|---|---|---|---|---|---|
| landscape metrics | categorical map | numbers per patch, class, landscape | patches are homogeneous, with sharp edges | standard, comparable, fast | redundant, scale-dependent, no process |
| moving windows | binary or continuous map | maps | the neighbourhood of a cell sets its condition | shows *where*; multi-scale | window size must be justified |
| morphology (MSPA) | binary map | class map | structure reflects function | intuitive roles of every pixel | structural only; edge width choice |
| graphs | patches and distances | indices, rankings | movement depends on distance between patches | ranks patches and links; fast | patch definition; ignores routes within the matrix unless least-cost distances are used |
| circuits | resistance map | current and resistance maps | random-walk movement | all routes, pinch points | needs a resistance surface; computation for large maps |
| least-cost | resistance map | paths, corridors, costs | movement along the optimal route | simple, mappable corridors | one route; assumes knowledge of the landscape |
| distribution models | occurrence records, environmental rasters | suitability maps | records sample the conditions the species uses; equilibrium with the environment | habitat maps for species without one; projections to other climates | sampling bias; relative suitability only from presence-only data; extrapolation |

## A worked example

**Question.** A regional agency wants to know how much habitat the
landscape of the CORINE example offers to a forest-interior bird, how it is
arranged, and which forest patches are most important to keep its
population connected.

**Process and organism.** The species avoids forest edges (edge effects
reach about 100 m) and disperses through farmland with a median distance of
about 1 km. Both values are uncertain; we test half and double.

**Map.** CORINE Land Cover, 25 m cells, minimum mapping unit 25 ha. It
resolves woodlots but not hedges, which suits a species of forest interiors.
Habitat: broad-leaved, coniferous and mixed forest (classes 23–25).

**Amount and configuration:**

```python
import numpy as np
import pandas as pd
import landschaft as ls

landcover = ls.load_raster(ls.example_data("clc"))
forest = ls.reclassify(landcover, foreground=[23, 24, 25])

structure = ls.Landscape(forest).class_metrics(["pland", "np", "area_mn", "lpi", "ed", "enn_mn"])
structure.set_index("class").loc[2].round(1)
```

About 21% of the landscape is forest, in 158 patches of 326 ha on average;
the nearest patch is on average about 540 m away.

**Interior habitat, with the edge depth varied:**

```python
interior = {}
for depth_m in (50, 100, 200):
    morph = ls.mspa(forest, edge_width=depth_m // 25)
    interior[f"edge effects {depth_m} m"] = ls.mspa_stats(morph).loc["Core", "pct_foreground"]
pd.Series(interior).round(1)
```

Between about half and four fifths of the forest is interior, depending on
the edge depth. The number is sensitive to the assumption, but the
conclusion "most of the forest is interior habitat" holds for all three.

**Connectivity, with the dispersal distance varied:**

```python
network = ls.HabitatGraph.from_raster(
    landcover, habitat=[23, 24, 25], area_unit="ha", distance="edge",
    max_distance=10_000, landscape_area=landcover.size * 25 * 25 / 1e4,
)
reach = network.overall(["NC", "PC"], distance=[500, 1000, 2000], probability=0.5)
reach.assign(**{"EC as % of forest": 100 * reach["EC(PC)"] / network.habitat_area})[
    ["distance", "NC", "EC(PC)", "EC as % of forest"]].round(1)
```

Whatever the dispersal distance, the reachable forest is less than half of
the forest: for this species the forest works as partly isolated clusters,
and connectivity, more than amount, limits the habitat available.

**Priority patches, and whether the ranking is robust:**

```python
top = {}
for d in (500, 1000, 2000):
    ranking = network.node_importance("PC", distance=d, probability=0.5)
    top[d] = set(ranking.nlargest(10, "dPC")["id"])

robust = top[500] & top[1000] & top[2000]
sorted(robust), len(robust)
```

Eight of the ten most important patches are the same at the three
dispersal distances. These are robust priorities; those that appear only at some distances depend on the
assumption and deserve a closer look (field data on dispersal, or a
resistance-based distance).

**Report.** A results table would state: forest amount (21%) and
configuration; interior habitat for each edge depth; reachable forest (EC)
for each dispersal distance; the robust priority patches on a map
(`network.to_raster(ranking, "dPC")`); and every parameter and data source
listed in the checklist below. Where the conclusions change with the
parameters, say so.

## A reporting checklist

- **Data**: map source, date, legend, cell size, minimum mapping unit,
  accuracy; coordinate reference system; study area and its buffer.
- **Habitat definition**: classes counted as habitat, and why.
- **Patch rule**: 4 or 8 neighbours.
- **Scale parameters in map units**: window sizes, edge widths and depths,
  buffer sizes, dispersal distances and probabilities, circuit radii.
- **Resistance surface**: values, their source, and the sensitivity test.
- **Metric choices**: which metrics, why, and how redundancy was handled.
- **Border treatment**: whether the map border counts as edge, how patches
  cut by the border were handled.
- **Software and version**: e.g. `landschaft.__version__`, with the
  reference implementation it reproduces where relevant.
- **Sensitivity**: which parameters were varied and whether conclusions held.

```python
ls.__version__
```

## Common mistakes

| Mistake | Better |
|---|---|
| computing dozens of metrics and reporting those that "work" | choose a few metrics from the process, before looking at results |
| interpreting fragmentation metrics without habitat amount | report amount and configuration together |
| comparing metrics from maps of different resolution or legend | harmonise maps first, or compare changes within one map series |
| using window sizes and edge widths in cells without converting | reason in metres; convert for each map |
| one dispersal distance chosen arbitrarily | a range of distances, or one justified by data |
| resistance values from a single expert guess, untested | several plausible sets; report which conclusions hold |
| change maps from independent maps, without accuracy | consistent map series; accuracy of change; filter boundary noise |
| treating MSPA bridges as proven corridors | structural hypotheses to validate with species data |
| a species' habitat taken from a land-cover class without checking | a distribution model from its records, or at least a check of the class against them |

!!! success "Key points"
    - Start from the question and the process, then define habitat, choose
      the map, set the scales and only then pick the tools.
    - Use the fewest analyses that answer the question, and test the
      sensitivity of conclusions to uncertain parameters.
    - Report enough to reproduce every number: data, habitat definition,
      parameters in map units, software and version.

## Where to go next

- The [tutorials](../tutorials/index.md) show every tool with its options.
- The [API reference](../api/index.md) documents every function.
- The [further reading](reading.md) lists textbooks and key papers to go
  deeper, and the [glossary](glossary.md) defines the terms of this course.
