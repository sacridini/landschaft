<p class="ls-chapter">Chapter 14</p>

# Simulating land change

[Chapter 10](change.md) looked back: what changed between two maps, by
which process, and with what effect on habitat and connectivity. This
chapter looks forward. A **land change model** takes what happened in an
observed period, encodes it as rules, and asks where the landscape is going
if those rules hold, or if a policy changes them. Its maps of the future are
the inputs of the analyses of the previous chapters: the fragmentation, the
connectivity or the habitat of a species in 2050 under one scenario or
another. Done carelessly, it produces confident maps of nothing; done well,
it is one of the few ways to compare the spatial consequences of decisions
before they are taken (Verburg et al. 2004; Brown et al. 2013).

!!! abstract "In this chapter"
    - the anatomy of a land change model: how much changes (demand), where
      (transition potential) and with what shape (allocation), under which
      rules;
    - the main families of models and what each assumes;
    - validation: the persistence null model, the figure of merit, the
      comparison of three maps at several resolutions, fuzzy similarity, and
      why kappa misleads;
    - calibration by patterns, and the separation of calibration and
      validation periods;
    - scenarios, replicates and the uncertainty of a single simulated map.

## The anatomy of a land change model

Almost every model of land change answers three questions, and the clearest
way to read one is to ask how it answers each (Verburg et al. 2004):

1. **How much** will change? The *demand* or *quantity* of change: hectares
   of cropland in 2030, a rate of deforestation, the housing a growing
   population needs. It usually comes from outside the map: trends, a
   transition matrix, economic models, a storyline.
2. **Where** will it happen? The *transition potential* (or *suitability*):
   for each cell, how likely each class, or each transition, is, given the
   *drivers*: distance to roads and to existing patches, slope, soil,
   protected areas.
3. With **what shape**? The *allocation*: how the demanded change is placed
   on the most likely cells: one cell at a time, in competition between
   classes, as new patches or as the growth of old ones.

Around them sit the **rules**: transitions that cannot happen (built land
does not turn into primary forest in ten years), areas where nothing may
change, the inertia of each class (an orchard resists conversion more than a
pasture), succession after abandonment.

| Family | Demand | Potential | Allocation | Typical use |
|---|---|---|---|---|
| Competition among classes (CLUE-S, Verburg et al. 2002) | areas per class and year | logistic regression per class | iterative: every cell takes the class of highest suitability plus an adjustable term, until the areas match | regional land use under scenarios |
| Land systems (CLUMondo, van Asselen & Verburg 2013) | goods and services | per land system | land systems that supply the demand, with intensification | agriculture, intensification against expansion |
| Transition-based cellular automata (DINAMICA, Soares-Filho et al. 2002) | Markov rates per transition | weights of evidence per transition | new patches and expansion of old ones, with size and shape | deforestation frontiers |
| Patch-generating CA (PLUS, Liang et al. 2021) | areas per class | random forests on the areas that expanded | competition with random seeds and a decreasing threshold | multi-class urban and rural change |
| Patch growing (FUTURES, Meentemeyer et al. 2013) | population per region | mixed-effects regression | patches of observed sizes grown with compactness | urban growth |

Landschaft's `ls.simulate` builds all of them from the same pieces: a
**demand**, a **potential**, an **allocation** and **rules**, put together in
a `Model`. The pieces can be mixed: weights of evidence with the competition
of CLUE-S, a random forest with Expander and Patcher.

## How much: demand

The simplest demand is the trend of the observed period. A **transition
matrix** (chapter 10) gives, for each class, the share of its cells that
became each other class. As a **Markov chain** it projects the areas forward,
assuming that the rates stay as they were. For yearly steps from a 14-year
period, the yearly matrix is the 14th root of the period's matrix (a matrix
whose 14th power is the observed one), not the period's rates divided by 14.

Sibuyan Island in the Philippines, mapped in 1997 and 2011 with five classes
(forest, coconut, grassland, rice, other), is the classic example of
CLUE-S:

```python
import numpy as np
import landschaft as ls

S = ls.simulate
lu = ls.load_raster(S.example_data("sibuyan_landuse"))      # 1997 and 2011
drivers = ls.load_raster(S.example_data("sibuyan_drivers"))
drivers = drivers.sel(band=["dist_sea", "dist_roads", "elevation", "slope"])
t0, t1 = lu[0].where(lu[0] != 255), lu[1].where(lu[1] != 255)

chain = S.Markov.from_maps(t0, t1, steps=14)        # yearly rates
print(chain.multi.round(4))
print(chain.project({c: int((t1 == c).sum()) for c in chain.classes}, steps=10).round(0).iloc[[0, -1]])
```

The projection is only as good as the assumption that the next decade
repeats the last. Most studies replace it with **scenarios**: demands from
an economic model, a policy target (zero deforestation), or a storyline
("agricultural expansion", "protection"), each a table of areas per year
(`S.AreaDemand`). The location and pattern parts of the model stay the same,
which is what makes the scenarios comparable.

## Where: transition potential

The potential of a cell for a class or transition is estimated from where
change happened in the observed period. Three approaches dominate:

- **Weights of evidence** (Bonham-Carter 1994) cut each driver into ranges
  and give each range a weight: the log of how much more frequent the
  transition is there than elsewhere. The weights add up (assuming the
  drivers independent given the transition) into a probability. They are
  easy to read: the curve of weight against distance to roads *is* the
  effect of roads.
- **Logistic regression** of presence (or of change) on the drivers, the
  workhorse of CLUE-S and of many studies since.
- **Machine learning** (random forests, boosting): flexible responses and
  interactions, at the price of interpretability.

What the model learns matters as much as how. Fitted to where a class *is*,
the model describes its current distribution, which mixes old and new land
uses. Fitted to where the class *appeared* in the observed period (the
*land expansion analysis* of PLUS), it describes where it *expands*, which is
what a simulation needs:

```python
from sklearn.ensemble import RandomForestClassifier

leas = S.Estimator(RandomForestClassifier(n_estimators=50, random_state=0),
                   target="expansion", classes=[2, 3], sample=0.3, stratify=True, seed=1)
leas.fit(t0, drivers, t1=t1)
print(leas.importance().round(3))     # which drivers explain where coconut (2) and grassland (3) expanded
```

Variables that the simulation itself changes, such as the distance to the
nearest forest, the share of built land around a cell or the gravitational
pull of the towns, should be recomputed at every step: these are the
**dynamic variables** of `Model(dynamic={...})` (`S.distance_to`,
`S.share_of`, `S.gravity_of`). Roads can grow with the simulation too: a
`S.RoadBuilder` (Dinamica's road constructor) adds roads from the network
towards attractive places at every step, and its `dynamic(network)` is the
distance to that growing network.

## With what shape: allocation

The allocation decides the *pattern* of the simulated change, and two
models with the same demand and potential can produce very different
landscapes. The competitive allocation of CLUE-S gives each cell to the
class of highest potential, which tends to scatter change over the many
cells of near-equal potential. Patch-based allocations place change as
**new patches** and as the **growth of existing ones**, with sizes taken
from the observed change: the Expander and Patcher of DINAMICA, or the
random patch seeds of PLUS. The share of the change that extends existing
patches, and the size distribution of the new ones, can be measured on the
observed maps:

```python
patches = S.presets.observed_patches(t0, t1)
print(patches[["from", "to", "cells", "expansion", "new_patches", "new_mean"]].round(2))
```

A model whose patches are too small fragments the simulated landscape; one
whose patches are too large consolidates it. Since the next steps of most
studies are landscape metrics or connectivity, the shape of the simulated
change is not a detail: it decides the answer.

Two kinds of change escape the class-by-class picture. Logging, fire and
harvest take biomass, carbon or timber from the forest without turning it
into another class: the **Patcher of a continuous state** of Dinamica EGO
(`S.ContinuousPatcher`) opens patches that take the stock of their cells,
drawn by probability, until a quota of stock is met. And change builds the
infrastructure that drives more change: on Amazonian frontiers clearing
follows the roads, and new roads follow the clearing (Soares-Filho et al.,
2004, 2006). A **road constructor** (`S.RoadBuilder`) draws destinations
among attractive cells at a range of distances from the roads and joins
each to the network down the cheapest way over a friction surface, every
step; the distance to the growing network is then a dynamic variable of
the model.

## Validation: is the simulation any good?

A simulation of the observed period, from the earlier map, can be compared
with the later observed map. The comparison is less obvious than it seems.

**Persistence is the null model.** Most of a landscape does not change in a
decade. A simulation that changes nothing at all agrees with the observed
map in most cells (often more than 90 %), and a model that changes the right
amount in the wrong places can agree *less* than doing nothing. Agreement
must be measured on the **change**: the **figure of merit** (Pontius et al.
2008) is the share of correctly simulated change among all change, observed
or simulated: *hits / (hits + misses + wrong hits + false alarms)*. It
ranges from 0 to 1; values of 0.1-0.3 are common for published models at the
resolution of the data (Pontius et al. 2008).

**Kappa misleads.** It compares agreement with a random baseline that no one
proposes as a model, and mixes errors of quantity and of location (Pontius &
Millones 2011). Report instead the **quantity disagreement** (wrong amounts
of each class) and the **allocation disagreement** (right amounts in the
wrong places), and, if kappa is needed, **kappa simulation**, which measures
agreement on the transitions (van Vliet et al. 2011).

**Near misses are not misses.** A patch of deforestation simulated two cells
away from the real one is a better simulation than one on the other side of
the map, but cell-by-cell measures score both as errors. Two answers: compare
the maps at **several resolutions** (blocks of 2, 4, 8... cells; Pontius et
al. 2011), or score each cell by how near the same change is in the other
map, with a **fuzzy similarity** that decays with distance (Hagen 2003).
Both show how far apart the simulated and observed change are:

```python
model = S.presets.dinamica(t0, t1, drivers, steps=14, woe=500)
sim = model.run(t0, drivers, seed=1)
check = S.validate(t0, t1, sim.maps[-1], factors=[1, 2, 4, 8, 16], windows=[1, 3, 5, 7, 9, 11])
print(check.summary().round(3))
print(check.three_map.figure_of_merit()["overall"].round(3))
```

The figure of merit and the similarity grow with the resolution and the
window: the model gets the general location of change right more often than
the exact cells. That growth is information, not consolation: it tells at
which resolution the simulated maps can be used.

## Calibration: patterns, not parameters

The settings of a model (the size of its patches, their compactness, the
inertia of each class, the weights of the drivers) are rarely measurable.
**Pattern-oriented modelling** (Grimm et al. 2005) chooses them so that the
simulation reproduces several observed patterns at once: the location of
change (figure of merit), its neighbourhood (fuzzy similarity), and the
structure of the resulting landscape (number of patches, edge density,
aggregation). Matching one pattern is easy and uninformative; matching
several independent ones constrains the model.

```python
def build(isometry):
    return S.presets.dinamica(t0, t1, drivers, steps=14, woe=500, isometry=isometry)

cal = S.calibrate(build, t0, t1, drivers, params={"isometry": [0.5, 1.0, 1.5, 2.0]},
                  targets={"similarity_5": 1, "lsm": (["np", "ed"], 1)},
                  method="grid", replicates=2, seed=1)
print(cal.history.round(3))
```

Two rules keep calibration honest. **Calibrate and validate on different
periods**: a model tuned to reproduce 1997-2011 and then "validated" on
1997-2011 has only been fitted. And **average replicates**: a stochastic
model run once gives one draw; comparing two settings on single runs mostly
compares their random numbers. Running each setting with the same seeds
(common random numbers) makes the comparison fair.

## Scenarios and uncertainty

A single simulated map of 2050 hides how much of it is chance. Running the
model many times with different seeds gives, for each cell, the
**probability** of each class: cells that change in every run are
consequences of the drivers and the demand; cells that change in a few runs
are where the allocation happened to fall. Scenarios are then compared on
these probabilities, or on the distribution of a landscape metric across
runs, rather than on one map each:

```python
plus = S.presets.plus(t0, t1, drivers, steps=14, seed=1, step_size=20)     # LEAS + CARS
twice = S.Markov.from_maps(t0, t1, steps=7)        # the same change in half the time
ens = S.scenarios(plus, t1, drivers, scenarios={"trend": {}, "faster": {"demand": twice}},
                  replicates=10, steps=5, years=2011, seed=1)
coconut = ens.probability(2, year=2016)            # share of the runs where each cell is coconut
print(coconut.sel(scenario="trend").sum().item(), coconut.sel(scenario="faster").sum().item())
change = ens.change_probability().sel(year=2016)
print((change > 0.5).sum(dim=change.dims[1:]).values)     # cells that change in most runs
```

## Good practice

1. **Describe the three parts** of the model separately (demand, potential,
   allocation) and justify each; most disagreements about a model are about
   one of them.
2. **Compare with persistence** and report the figure of merit, quantity and
   allocation disagreement, at several resolutions.
3. **Use the shape of the observed change** (patch sizes, expansion against
   new patches) and check the landscape metrics of the simulated maps.
4. **Calibrate and validate on different periods**, and on several patterns.
5. **Run replicates**, and report probabilities or distributions, not one
   map per scenario.
6. **Treat demand as a scenario**, not a forecast: the model says where and
   how, given how much.

!!! success "Key points"
    - A land change model answers how much (demand), where (potential) and
      with what shape (allocation), under rules.
    - Persistence is the null model: measure agreement on change (figure of
      merit), separate quantity from allocation disagreement, avoid kappa.
    - Agreement depends on resolution; fuzzy similarity and multiple
      resolutions show how near the simulated change is to the observed.
    - Calibrate on patterns, with replicates and common random numbers, and
      validate on another period.
    - One simulated map is one draw: use replicates and probabilities.

## Check your understanding

1. A simulation of 2001-2011 agrees with the observed 2011 map in 94 % of the
   cells. The observed change covered 5 % of the map. Is the model good?
2. Two models have the same figure of merit at the cell level; at blocks of
   8 × 8 cells one reaches 0.6 and the other 0.25. What does that say?
3. Why fit the potential to where a class appeared rather than to where it
   is?

??? success "Answers"
    1. Not necessarily: a model that changes nothing agrees in 95 % of the
       cells. Look at the figure of merit, the quantity and allocation
       disagreement, and compare with persistence.
    2. The first places its change near the observed change (it misses by a
       few cells), the second far from it: the first is useful at coarser
       resolutions, the second is not.
    3. Because the distribution of a class mixes old and new land use; the
       simulation needs the conditions under which the class *expands*,
       which can differ (new cropland on flatter land than the old, near the
       newest roads).

## References

<div class="ls-refs" markdown>

- Bonham-Carter, G. F. (1994). *Geographic Information Systems for Geoscientists: Modelling with GIS*. Pergamon.
- Brown, D. G., Verburg, P. H., Pontius, R. G., Jr., & Lange, M. D. (2013). Opportunities to improve impact, integration, and evaluation of land change models. *Current Opinion in Environmental Sustainability*, 5(5), 452–457.
- Grimm, V., Revilla, E., Berger, U., Jeltsch, F., Mooij, W. M., Railsback, S. F., Thulke, H.-H., Weiner, J., Wiegand, T., & DeAngelis, D. L. (2005). Pattern-oriented modeling of agent-based complex systems: lessons from ecology. *Science*, 310(5750), 987–991.
- Hagen, A. (2003). Fuzzy set approach to assessing similarity of categorical maps. *International Journal of Geographical Information Science*, 17(3), 235–249.
- Liang, X., Guan, Q., Clarke, K. C., Liu, S., Wang, B., & Yao, Y. (2021). Understanding the drivers of sustainable land expansion using a patch-generating land use simulation (PLUS) model: a case study in Wuhan, China. *Computers, Environment and Urban Systems*, 85, 101569.
- Meentemeyer, R. K., Tang, W., Dorning, M. A., Vogler, J. B., Cunniffe, N. J., & Shoemaker, D. A. (2013). FUTURES: multilevel simulations of emerging urban–rural landscape structure using a stochastic patch-growing algorithm. *Annals of the Association of American Geographers*, 103(4), 785–807.
- Pontius, R. G., Jr., Boersma, W., Castella, J.-C., Clarke, K., de Nijs, T., Dietzel, C., et al. (2008). Comparing the input, output, and validation maps for several models of land change. *The Annals of Regional Science*, 42(1), 11–37.
- Pontius, R. G., Jr., & Millones, M. (2011). Death to Kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. *International Journal of Remote Sensing*, 32(15), 4407–4429.
- Pontius, R. G., Jr., Peethambaram, S., & Castella, J.-C. (2011). Comparison of three maps at multiple resolutions: a case study of land change simulation in Cho Don District, Vietnam. *Annals of the Association of American Geographers*, 101(1), 45–62.
- Soares-Filho, B. S., Alencar, A., Nepstad, D., Cerqueira, G. C., Diaz, M. del C. V., Rivero, S., Solórzano, L., & Voll, E. (2004). Simulating the response of land-cover changes to road paving and governance along a major Amazon highway: the Santarém–Cuiabá corridor. *Global Change Biology*, 10(5), 745–764.
- Soares-Filho, B. S., Cerqueira, G. C., & Pennachin, C. L. (2002). DINAMICA: a stochastic cellular automata model designed to simulate the landscape dynamics in an Amazonian colonization frontier. *Ecological Modelling*, 154(3), 217–235.
- Soares-Filho, B. S., Nepstad, D. C., Curran, L. M., Cerqueira, G. C., Garcia, R. A., Ramos, C. A., Voll, E., McDonald, A., Lefebvre, P., & Schlesinger, P. (2006). Modelling conservation in the Amazon basin. *Nature*, 440, 520–523.
- van Asselen, S., & Verburg, P. H. (2013). Land cover change or land-use intensification: simulating land system change with a global-scale land change model. *Global Change Biology*, 19(12), 3648–3667.
- van Vliet, J., Bregt, A. K., & Hagen-Zanker, A. (2011). Revisiting Kappa to account for change in the accuracy assessment of land-use change models. *Ecological Modelling*, 222(8), 1367–1375.
- Verburg, P. H., Schot, P. P., Dijst, M. J., & Veldkamp, A. (2004). Land use change modelling: current practice and research priorities. *GeoJournal*, 61(4), 309–324.
- Verburg, P. H., Soepboer, W., Veldkamp, A., Limpiada, R., Espaldon, V., & Mastura, S. S. A. (2002). Modeling the spatial dynamics of regional land use: the CLUE-S model. *Environmental Management*, 30(3), 391–405.

</div>
