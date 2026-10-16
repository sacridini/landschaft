# Land change simulation

Where is a landscape heading? A land change model learns from two maps how
much changed, where and with what shape, and projects it forward, under
the observed trend or under scenarios. `ls.simulate` builds such models from
interchangeable pieces (a demand, a transition potential, an allocation and
rules), runs them, compares them with what really happened, calibrates them
and runs scenarios with replicates. The ideas behind it are in the course
chapter [Simulating land change](../learn/simulation.md).

**You will learn to:**

- describe the change between two maps and turn it into a demand;
- estimate where each class is likely to appear;
- allocate the change with four different allocations, and see how the
  shape of the simulated change differs;
- fit, check and project a model in one call;
- validate a simulation against the observed map;
- calibrate a setting that cannot be measured;
- run scenarios with replicates and map the probability of change;
- let variables, roads and stocks change with the simulation.

The pieces reproduce the established land change models, value for value
where their code is open: CLUE-S and the ordered procedure as the R package
lulcc runs them, CLUMondo, FUTURES (GRASS's `r.futures`) and MOLUSCE; the
weights of evidence, transition rates, Expander and Patcher, fuzzy
similarity, gravity potential, road constructor and continuous Patcher of
Dinamica EGO, measured on the program; PLUS's random patch
seeds, written from its paper. The [benchmarks page](../benchmarks.md)
lists every comparison.

## The data

Sibuyan Island, in the Philippines, was mapped in 1997 and 2011 with five
classes, and comes with a set of explanatory variables: distances to the
sea, roads and towns, elevation, slope, soils and population density.

```python
import numpy as np
import landschaft as ls

S = ls.simulate
lu = ls.load_raster(S.example_data("sibuyan_landuse"))
drivers = ls.load_raster(S.example_data("sibuyan_drivers"))
print(lu.band.values, drivers.band.values)

t0, t1 = lu[0].where(lu[0] != 255), lu[1].where(lu[1] != 255)   # 255: outside the island
drivers = drivers.sel(band=["dist_sea", "dist_roads", "elevation", "slope"])
names = {1: "forest", 2: "coconut", 3: "grassland", 4: "rice", 5: "other"}
```

## What changed

`S.changes` tabulates the change: the cells of each transition, the gains
and losses of each class and the intensity analysis of Aldwaik & Pontius,
which says whether a class gained or lost faster than the landscape changed
as a whole, and which classes a transition targets or avoids.

```python
ch = S.changes([t0, t1], times=[1997, 2011])
print(ch.matrix())                      # cells from each class (rows) to each class (columns)
ia = ch.intensity(gain=2, loss=1)       # coconut's gains, forest's losses
print(ia.category_gain)                 # which classes gained faster than the landscape changed
print(ia.transition_loss)               # which classes took forest more than uniformly
```

## How much: the demand

A Markov chain projects the observed transition rates forward. With
`steps=14` its yearly matrix is the 14th root of the period's matrix:

```python
chain = S.Markov.from_maps(t0, t1, steps=14)
print(chain.multi.round(4))
areas = {c: int((t1 == c).sum()) for c in chain.classes}
print(chain.project(areas, steps=10).round(0).iloc[[0, 5, 10]])
```

A demand can also be any table of areas per step (`S.AreaDemand`), the
areas of observed maps interpolated (`S.AreaDemand.from_maps`), goods and
services for land systems (`S.ServiceDemand`), or population for urban
growth (`S.DevelopmentDemand`); `S.Regional` gives a demand per region.

## Where: the transition potential

Weights of evidence cut each variable into ranges and weigh each range by
how much more often a transition happens there:

```python
woe = S.WeightsOfEvidence({"dist_roads": 250, "dist_sea": 250, "elevation": 50, "slope": 2},
                          transitions=[(1, 2), (1, 3), (3, 2)]).fit(t0, t1, drivers)
w = woe.table
print(w[(w.Transition_From == 1) & (w.Transition_To == 2) & (w.Variable == "dist_roads")]
      [["Range_Lower_Limit", "Range_Upper_Limit", "Weight_Coefficient"]].head(6).round(3))
```

A logistic regression per class, or any scikit-learn classifier, gives a
potential per class instead of per transition. Trained on the cells where a
class *appeared* (`target="expansion"`), it describes where the class
expands:

```python
from sklearn.ensemble import RandomForestClassifier

rf = S.Estimator(RandomForestClassifier(n_estimators=50, random_state=0), target="expansion",
                 classes=[2, 3, 4], sample=0.3, stratify=True, seed=1).fit(t0, drivers, t1=t1)
print(rf.importance().round(3))
lr = S.LogisticRegression({c: list(drivers.band.values) for c in names}).fit(t1, drivers)
```

## With what shape: four allocations

The same demand and the same potential, allocated four ways, from 2011 to
2016. The competitive allocation (`S.CLUE`) and the ordered one
(`S.Ordered`) take a potential per class; Expander and Patcher place the
transitions as new patches and as the growth of old ones; `S.CARS` grows
patches from random seeds:

```python
yearly = np.floor(chain.project(areas, steps=5).to_numpy())    # 2011 (step 0) to 2016
yearly[:, 0] += int(t1.notnull().sum()) - yearly.sum(axis=1)   # forest takes the rounding: whole island
demand = S.AreaDemand(yearly, classes=chain.classes)

models = {
    "competition": S.Model(lr, demand, S.CLUE(jitter=0.0002, scale=1e-6, max_diff=5, mean_diff=5),
                           S.Rules(elasticity={1: 0.6, 2: 0.4})),
    "ordered": S.Model(lr, demand, S.Ordered(order=[2, 3, 4, 5, 1])),
    "random seeds": S.Model(S.RasterPotential({c: lr.predict(drivers).sel({"class": c}).values
                                               for c in names}), demand, S.CARS(step_size=10)),
}
rates = {tr: chain.multi.loc[tr] for tr in [(1, 2), (1, 3), (3, 2)]}   # the transitions of the weights
models["expand and patch"] = S.Model(woe, rates,
                                     S.ExpanderPatcher(steps=5, expansion=0.5, expander=(6, 20, 1.5),
                                                       patcher=(6, 20, 1.5)))
ends = {}
for name, model in models.items():
    sim = model.run(t1, drivers, seed=1)
    ends[name] = sim.maps[-1]
    m = ls.landscape_metrics(sim.maps[-1].fillna(0).astype("int32"), level="class", metric=["np"])
    print(f"{name:18s}", m[m["class"] == 2]["value"].item(), "coconut patches")
print("observed 2011    ", ls.landscape_metrics(t1.fillna(0).astype("int32"), level="class",
                                                metric=["np"]).query("`class` == 2")["value"].item())
```

The amounts are the same; the patterns are not: here the coconut patches
of 2016 number from 17 to 22 depending on the allocation (16 in 2011),
with the ordered allocation, which takes the most suitable cells wherever
they are, the most scattered. On larger maps and longer periods the
differences grow, and the landscape metrics or the connectivity computed
on the simulated maps depend on that choice.

## One call: fit, check, project

`S.run` does the whole sequence for the classic models: it fits a preset
on the two maps with one step per year, simulates the observed period from
the earlier map and compares it with the later one, then projects:

```python
proj = S.run(t0, t1, drivers, until=2021, preset="dinamica", woe=250, seed=1)
print(proj)
print(proj.report)
print(proj.maps.year.values)
```

`proj.settings` records every choice and the seed, `proj.redo(...)` runs
again with some of them changed, and `ls.plot(proj)` shows the projected
maps with a slider over the years.

```python
again = proj.redo(seed=2, until=2030)
print(again)
```

## Validation

A simulation of the observed period is judged on the change, not on the
whole map, at several resolutions and with a fuzzy tolerance for near
misses. `S.validate` gathers the comparisons:

```python
check = proj.validate(factors=[1, 2, 4, 8], windows=[1, 3, 5, 9])
print(check.summary().round(3))
budget = check.three_map.agreement_budget()
print(budget[budget.level == "overall"][["factor", "hits", "misses", "wrong_hits", "false_alarms"]].round(3))
```

The figure of merit (correct change over all change) and the similarity
grow with the resolution: the model places change in the right area more
often than in the right cells.

## Calibration

The compactness of the simulated patches (the isometry of Expander and
Patcher) cannot be measured on the maps. `S.calibrate` searches the value
whose simulations best reproduce the observed patterns, here the fuzzy
similarity of the change and the number and edge of the patches, each value
run with the same seeds:

```python
def build(isometry):
    return S.presets.dinamica(t0, t1, drivers, steps=14, woe=250, isometry=isometry)

cal = S.calibrate(build, t0, t1, drivers, params={"isometry": [0.5, 1.0, 1.5, 2.0]},
                  targets={"similarity_5": 1, "lsm": (["np", "ed"], 1)}, method="grid",
                  replicates=2, seed=1)
print(cal.history.round(3))
print(cal.best)
```

`method="nelder-mead"` or `"differential-evolution"` search continuous
ranges, and the settings can also be named as paths into a model
(`"allocation.step_size"`, `"rules.elasticity[1]"`).

## Scenarios and replicates

One simulated map is one draw. `S.scenarios` runs every scenario many
times, in parallel, with the same seeds in every scenario, and maps the
probability of each class:

```python
plus = S.presets.plus(t0, t1, drivers, steps=14, seed=1, step_size=20)
faster = S.Markov.from_maps(t0, t1, steps=7)            # the observed change in half the time
ens = S.scenarios(plus, t1, drivers, scenarios={"trend": {}, "faster": {"demand": faster}},
                  replicates=10, steps=5, years=2011, seed=1)
print(ens)
p = ens.change_probability().sel(year=2016)
print((p > 0.5).sum(dim=p.dims[1:]).to_series())          # cells that change in most runs
cover = ens.cover().groupby(["scenario", "year"]).mean().drop(columns="replicate")
print(cover.xs(2016, level="year").round(0))
```

`ens.metrics(["np", "ed"])` measures every run and year, and `ls.plot(ens)`
maps the probability of change in each scenario.

## Variables that change with the map

Distances to a class, or its share around each cell, change as the
simulation goes on. Given as dynamic variables they are recomputed at every
step, and the potential with them:

```python
drivers_t1 = {k: drivers.sel(band=k) for k in drivers.band.values}
drivers_t1["dist_forest"] = S.distance_to(1)(t1)
lr2 = S.LogisticRegression({2: ["dist_roads", "dist_forest", "slope"], 1: ["elevation", "dist_forest"],
                            3: ["dist_roads", "slope"], 4: ["elevation", "slope"], 5: ["dist_sea"]})
lr2 = lr2.fit(t1, drivers_t1)
model = S.Model(lr2, demand, S.Ordered(order=[2, 3, 4, 5, 1]), dynamic={"dist_forest": S.distance_to(1)})
sim = model.run(t1, drivers_t1, seed=1)
print(sim.steps.filter(like="cells_"))
```

`S.gravity_of` is another: the pull of a class (here the rice fields, as a
stand-in for villages) summed over the map with a distance decay, the
gravity model of spatial interaction, recomputed as the class grows:

```python
pull = S.gravity_of(4, exponent=2)          # sum over rice cells of 1 / distance^2 (in cells)
print(float(pull(t1).max()), float(pull(t1).mean()))
```

## Roads that grow with the model

Roads open new land to clearing, and clearing calls for new roads. A road
builder adds roads each step: it keeps the attraction (here, the forest)
at 500-2000 m from the roads, draws destinations in quadrants of the map,
and joins each to the network down the cheapest way over a friction (here
the slope):

```python
roads = (drivers.sel(band="dist_roads") == 0).astype(float).where(~np.isnan(t1))   # the road cells of 2011
friction = (1 + drivers.sel(band="slope")).where(~np.isnan(t1))
rb = S.RoadBuilder(friction, lambda m: (m == 1).astype(float), quadrants=(2, 4), distance=(500, 2000),
                   spacing=1000)
nets = rb.run(roads.fillna(0), 3, landscape=t1, seed=1)
print([int((nets[k] > 0).sum()) for k in range(4)])                  # road cells at each step
```

`rb.dynamic(network)` makes the distance to the growing network a dynamic
variable of a `Model`: each step builds roads on the simulated map, and the
potential sees them. The pieces are also functions of their own:
`S.friction_cost`, `S.destinations` and `S.pathways`.

## Taking a stock

Logging and harvest take biomass from the forest without changing its
class. `S.ContinuousPatcher` opens patches, of a lognormal size, that take
all the stock of their cells, drawn by probability, until the quota of each
step is met:

```python
biomass = (t1 == 1) * 150.0                                    # Mg per hectare in the forest
near_roads = (1 / (1 + drivers.sel(band="dist_roads") / 500)).where(t1 == 1, 0)
cp = S.ContinuousPatcher(mean=25, variance=100, isometry=1.5)  # patches of about 25 ha
logging = cp.run(biomass.where(~np.isnan(t1)), near_roads, [20000, 20000, 20000], seed=1)
print(logging.steps)                                           # Mg taken, the quota left, patches, cells
```

The quota left is negative when the last cell took more than was needed,
and positive when the cells with stock and probability ran out.

## Other models

- **Land systems**: `S.CLUMondo` allocates land systems that supply goods
  and services (`S.ServiceDemand`, `S.Supply`), with intensification and
  expansion; overlapping binary layers become land systems with
  `S.LandSystemStack` and their roles with `S.Roles`.
- **Urban growth**: `S.PatchGrowing` grows development patches of observed
  sizes from a population-driven demand, with `S.DevelopmentPotential`
  (mixed-effects regression) and optional flooding (`S.Flooding`).
- **MOLUSCE's simulator**: `S.MolusceWoE` and `S.MolusceCA`, or
  `S.presets.molusce`.

Each has its examples in the [API reference](../api/simulate.md).
