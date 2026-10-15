# Land change simulation

Land use and land cover change models: analysing the changes between maps,
estimating where change is likely, allocating future change and validating
the simulated maps: the `ls.simulate` module. It brings the tools of the
field together as interchangeable pieces of one model (CLUE-S, CLUMondo,
Dinamica EGO, PLUS, FUTURES), each reproducing its reference. The module is
being built phase by phase; this page grows with it.

<!-- no-run -->
```python
import landschaft as ls

lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))   # 1985, 1991, 1999
ch = ls.simulate.changes(lu)                                   # transitions, areas, change counts
ia = ch.intensity(gain=2, loss=1)                              # intensity analysis
ls.simulate.difference_components(ch.matrix())                 # quantity, exchange, shift
mk = ch.markov(steps=14)                                       # transition rates per year

ef = ls.load_raster(ls.simulate.example_data("pie_drivers"))
woe = ls.simulate.weights_of_evidence(lu[0], lu[2], ef, {"dist_built_1985": 100})
p = woe.probability(lu[2], ef)                                 # Dinamica EGO's transition probabilities

forms = {1: ["elevation", "slope"], 2: ["elevation", "slope", "dist_built_1985"], 3: ["elevation", "slope"]}
model = ls.simulate.Model(                                     # CLUE-S, as lulcc
    potential=ls.simulate.LogisticRegression(forms).fit(lu[0], ef),
    demand=ls.simulate.AreaDemand.from_maps(lu, times=[0, 6, 14]),
    allocation=ls.simulate.CLUE(jitter=0.0002, scale=1e-6, max_diff=50, mean_diff=50),
    rules=ls.simulate.Rules(elasticity=[0.2, 0.2, 0.2]))
sim = model.run(lu[0], ef, seed=1)                             # 14 yearly maps
```

Land systems that supply goods and services (CLUMondo), given as overlapping
binary layers:

<!-- no-run -->
```python
S = ls.simulate
st = S.LandSystemStack({"forest": f, "extensive": pe, "intensive": pi, "soy": soy})
st.overlaps()                                                  # the combinations (mosaics) found
roles = S.Roles(natural=["forest"], families={"cattle": ["extensive", "intensive"], "crops": ["soy"]})
model = S.Model(
    potential=st.suitability({"forest": sf, "extensive": se, "intensive": si, "soy": ss}),
    demand=S.ServiceDemand(demand_df),                         # heads of cattle, tons of soy per year
    allocation=S.CLUMondo(st.supply({"extensive": {"cattle": 0.6}, "intensive": {"cattle": 2.5},
                                     "soy": {"soy": 3.4}})),
    rules=S.Rules(elasticity=0.5, roles=roles, conversion_order=["intensification", "expansion"],
                  caps=[S.Cap("expansion", cells=500)], restrictions=parks))
sim = model.run(st, seed=1)
sim.layers["forest"]; sim.shortfall; sim.trajectory()
```

Urban growth by patches (FUTURES):

<!-- no-run -->
```python
demand = S.DevelopmentDemand.from_population(observed_maps, counties, observed_pop, projected_pop,
                                             range(2020, 2051))
model = S.Model(
    potential=S.DevelopmentPotential(coefficients_df),         # intercept, devpressure, predictors per county
    demand=demand,
    allocation=S.PatchGrowing(patch_library, subregions=counties, neighborhood=10,
                              compactness_mean=0.4, compactness_range=0.05, incentive_power=2))
sim = model.run(developed_2020, {"slope": slope, "road_dist": road_dist}, seed=1)
sim.development                                                # step in which each cell was developed
```

## Change between maps

::: landschaft.simulate.changes

::: landschaft.simulate.Changes

::: landschaft.simulate.IntensityAnalysis

::: landschaft.simulate.crosstab

## Components of difference

::: landschaft.simulate.difference_components

::: landschaft.simulate.overall_difference

::: landschaft.simulate.difference_resolutions

::: landschaft.simulate.composite_matrix

::: landschaft.simulate.memberships

::: landschaft.simulate.mad

::: landschaft.simulate.sample_to_population

## Markov chains

::: landschaft.simulate.Markov

## Transition potential

::: landschaft.simulate.WeightsOfEvidence

::: landschaft.simulate.weights_of_evidence

::: landschaft.simulate.LogisticRegression

::: landschaft.simulate.Estimator

::: landschaft.simulate.MultiCriteria

::: landschaft.simulate.RasterPotential

## Allocation

::: landschaft.simulate.Model

::: landschaft.simulate.Simulation

::: landschaft.simulate.AreaDemand

::: landschaft.simulate.Rules

::: landschaft.simulate.Neighbourhood

::: landschaft.simulate.CLUE

::: landschaft.simulate.Ordered

## Dynamic variables and regions

::: landschaft.simulate.distance_to

::: landschaft.simulate.share_of

::: landschaft.simulate.gravity_of

::: landschaft.simulate.Regional

## Land systems (CLUMondo)

::: landschaft.simulate.CLUMondo

::: landschaft.simulate.ServiceDemand

::: landschaft.simulate.Supply

::: landschaft.simulate.NeighbourhoodEffect

::: landschaft.simulate.LandSystemStack

::: landschaft.simulate.Roles

::: landschaft.simulate.Cap

::: landschaft.simulate.Succession

## Expander and Patcher (Dinamica)

::: landschaft.simulate.ExpanderPatcher

## Stock taken by patches (Dinamica's continuous Patcher)

::: landschaft.simulate.ContinuousPatcher

::: landschaft.simulate.StockAllocation

::: landschaft.simulate.StockSimulation

## Roads built by the model (Dinamica's road constructor)

::: landschaft.simulate.RoadBuilder

::: landschaft.simulate.RoadDistance

::: landschaft.simulate.friction_cost

::: landschaft.simulate.destinations

::: landschaft.simulate.pathways

## Multi-type random patch seeds (PLUS)

::: landschaft.simulate.CARS

## MOLUSCE

::: landschaft.simulate.MolusceWoE

::: landschaft.simulate.MolusceCA

## Urban growth (FUTURES)

::: landschaft.simulate.PatchGrowing

::: landschaft.simulate.DevelopmentDemand

::: landschaft.simulate.DevelopmentPotential

::: landschaft.simulate.development_pressure

::: landschaft.simulate.Flooding

::: landschaft.simulate.patch_library

::: landschaft.simulate.calibrate_patches

## Validation

::: landschaft.simulate.three_map_comparison

::: landschaft.simulate.ThreeMapComparison

::: landschaft.simulate.compare_maps

::: landschaft.simulate.error_budget

::: landschaft.simulate.futures_validation

::: landschaft.simulate.futures_validation_grid

::: landschaft.simulate.fuzzy_similarity

::: landschaft.simulate.FuzzySimilarity

::: landschaft.simulate.differences

::: landschaft.simulate.similarity_of_differences

::: landschaft.simulate.validate

::: landschaft.simulate.Validation

## Calibration

::: landschaft.simulate.calibrate

::: landschaft.simulate.Calibration

## Presets and one-call projection

::: landschaft.simulate.run

::: landschaft.simulate.Projection

::: landschaft.simulate.agreement

::: landschaft.simulate.presets

## Running models with Terraflux

[Terraflux](https://github.com/sacridini/terraflux) runs land change models as graphs of operators, with the
same C++ kernels: `Model.save` writes a fitted model as its graph (with the data it runs on), `Model.load` reads
it back, and `terraflux` runs it with the `terraflux` command, giving the same maps as `Model.run` with the same
seed.

::: landschaft.simulate.terraflux

## Scenarios and replicates

::: landschaft.simulate.scenarios

::: landschaft.simulate.Ensemble

## Example data

::: landschaft.simulate.example_data
