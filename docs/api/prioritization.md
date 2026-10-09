# Spatial prioritization

Two ways of setting conservation priorities, side by side: ranking every cell of a landscape
(below), and selecting the best set of planning units by optimization, with targets, budgets,
spatial penalties and constraints, portfolios, multi-action plans and the evaluation of the
solutions ([Optimization](#optimization) onwards). Both read the same rasters and share their
pre-processing, and each can start or check the other.

Ranking the cells of a landscape from the least to the most important for
conservation, given where a set of features (species, habitats, ecosystem
services, carbon...) occur: the core of systematic conservation planning. The
ranking balances all features at once, weighted, with costs and hierarchical
masks (for example, protected areas kept to the end), and its performance curves
tell how much of every feature the best 10 %, 20 %... of the landscape hold.
It lives in the `ls.prioritize` module.

The method is Zonation 5 (Moilanen et al. 2022): the cells are sorted by their
marginal loss, the loss to the features of removing a cell at the point where it
is removed, with the core-area (CAZ, driven by the features that lose the largest
fraction) or additive-benefit (ABF) rules, and sorted again until the order is
consistent. Landschaft gives the rank maps and curves of the z5 program (v2.4)
bit for bit, faster, with the curves of the ranking it returns (Zonation's own
report those of the order before its last sort). The
[tutorial](../tutorials/prioritization.md) walks through a project, and the
course chapter [Conservation priorities](../learn/prioritization.md) explains
the method.

<!-- no-run -->
```python
import landschaft as ls

res = ls.prioritize.zonation(["sp1.tif", "sp2.tif", "sp3.tif"], weights=[2, 1, 1],
                             cost="land_price.tif", hierarchic_mask="protected.tif")
res.rank                      # 0 = removed first ... ~1 = highest priority
res.feature_curves            # fraction of every feature kept, by fraction removed
res.write("zonation_output")  # rankmap.tif and Zonation's CSV files
```

## Ranking

::: landschaft.prioritize.zonation

::: landschaft.prioritize.ZonationResult

## Large maps

The features are read in blocks of rows (files, NumPy, xarray or dask arrays;
dask stacks are computed block by block, never whole) and quantized to one or two
bytes per value, kept as runs of NoData, zeros and values: a feature costs about
its occurrences, however many cells it is zero in, so thousands of species ranges
fit where Zonation stores every feature in every cell. During the ranking every
cell's occurrences are a compact row (a bitmap or layer gaps, whichever is
smaller) and the work arrays take about 130 bytes per cell with data.

For maps larger than the memory, `storage` puts the large arrays in memory-mapped
temporary files in a directory (on a fast disk), which the operating system pages
as needed; the reads and writes of the ranking are in sequence or nearly so. The
result is the same bit for bit; the files are deleted when no longer needed.

<!-- no-run -->
```python
import dask.array as da

stack = da.stack([da.from_zarr(f"ranges.zarr/{s}") for s in species])  # (features, y, x)
res = ls.prioritize.zonation(stack, storage="D:/scratch", n_threads=16)
res.write("global_ranking")
```

The ranking is limited to 2^32 - 1 cells with data and 65 535 features.

## Zonation projects

Existing Zonation 5 projects run as they are: the settings file, its feature
list and link files (weight groups, conditions, retention), with the switches of
the z5 command line.

::: landschaft.prioritize.run_project

::: landschaft.prioritize.read_project

## Connectivity

Dispersal kernels for the connectivity transforms of the features (the
``connectivity``, ``interaction`` and ``matrix_connectivity`` arguments of
`zonation`): a feature smoothed by the dispersal of its species, weighted by the
connectivity of a resource, or mixed with others by a matrix.

::: landschaft.prioritize.Kernel

::: landschaft.prioritize.Interaction

::: landschaft.prioritize.convolve

## Post-processing

Analyses of a ranking (methods of `ZonationResult`): statistics of areas
(external zones, the top fraction, separate high-priority areas), comparison with
another solution, histograms of the features' coverage and the flexibility map;
`ZonationResult.postprocess` runs Zonation's post-processing files.

::: landschaft.prioritize.AreaStatistics

## Impact assessment

Scenarios of land-use change and pressures, evaluated for the features and
applied to them before the ranking (the ``impact``, ``cumulative_impact``,
``impact_groups`` and ``impact_bins`` arguments of `zonation`): the loss around
impact sources, per footprint and buffer zone, and the direct habitat loss and
cumulative pressures of every impact group.

::: landschaft.prioritize.ImpactScenario

::: landschaft.prioritize.ImpactSource

::: landschaft.prioritize.CumulativeScenario

::: landschaft.prioritize.Pressure

## Optimization

A problem of planning units (raster cells, polygons, H3 cells or table rows), features, an
objective, targets, penalties and constraints, compiled to a mixed-integer linear model and solved
to a guaranteed gap. The grammar and the models are prioritizr's (Hanson et al.): `compile()` gives
prioritizr 8.0.6's model bit for bit and the solvers reach the same optima as prioritizr on Gurobi.
HiGHS is built in, with Landschaft's own engines in front of it for the problems it finds hard
(covering problems of millions of units, budgeted objectives, boundary penalties by minimum cuts,
multi-action plans); Gurobi is used when installed and asked for. Coming from R or prioripy? The
[migration guide](prioritization-migration.md) lists the equivalent of every function.

```python
import landschaft as ls

P = ls.prioritize
cost = ls.load_raster(P.example_data("cost"))
features = ls.load_raster(P.example_data("features"))
protected = ls.load_raster(P.example_data("protected"))
p = (P.problem(cost, features)
       .add_min_set_objective()
       .add_relative_targets(0.2)
       .add_locked_in_constraints(protected)
       .add_boundary_penalties(0.0005)
       .add_binary_decisions()
       .add_highs_solver(gap=0.05))
sol = p.solve()
sol.status, round(sol.gap, 3)
```

::: landschaft.prioritize.problem

::: landschaft.prioritize.Problem

::: landschaft.prioritize.Solution

::: landschaft.prioritize.Portfolio

::: landschaft.prioritize.OptimizationProblem

::: landschaft.prioritize.h3_units

::: landschaft.prioritize.example_data

## Spatial relations

The matrices that penalties and constraints read: the length of the boundary units share, which
units are adjacent, the connectivity between them from a raster or from distances.

::: landschaft.prioritize.boundary_matrix

::: landschaft.prioritize.adjacency_matrix

::: landschaft.prioritize.connectivity_matrix

::: landschaft.prioritize.proximity_matrix

::: landschaft.prioritize.decay_connectivity

::: landschaft.prioritize.unit_points

## Evaluating solutions

The figures of a solution (prioritizr's `eval_*`, equal bit for bit) and the importance of every
selected unit; the landscape the solution makes (patches, MSPA) and its species-area extinction risk.
They take a `Solution`, its selection or plain values per planning unit.

::: landschaft.prioritize.eval_n_summary

::: landschaft.prioritize.eval_cost_summary

::: landschaft.prioritize.eval_boundary_summary

::: landschaft.prioritize.eval_connectivity_summary

::: landschaft.prioritize.eval_asym_connectivity_summary

::: landschaft.prioritize.eval_feature_representation_summary

::: landschaft.prioritize.eval_target_coverage_summary

::: landschaft.prioritize.eval_replacement_importance

::: landschaft.prioritize.eval_ferrier_importance

::: landschaft.prioritize.eval_rare_richness_importance

::: landschaft.prioritize.eval_rank_importance

::: landschaft.prioritize.eval_extinction_risk

::: landschaft.prioritize.eval_landscape_metrics

::: landschaft.prioritize.eval_mspa

### Areas of a priority surface

The separate areas of the top fraction of any priority surface (a Zonation ranking, an importance, a
selection frequency) and what each holds, and the overlap of the top fractions of two surfaces
(Zonation's landscape identification and comparison; on a Zonation ranking, the same areas as
`ZonationResult.area_statistics_separate_areas`).

::: landschaft.prioritize.eval_landscape_identification

::: landschaft.prioritize.eval_landscape_comparison

::: landschaft.prioritize.position_rank

::: landschaft.prioritize.LandscapeAreas

::: landschaft.prioritize.LandscapeComparison

### Irreplaceability (ProxIRR)

ProxIRR (Baisero et al. 2022): for every target, the share of the feature's spare amount a unit
holds (alpha), and per unit the chance it is needed for at least one feature (beta) and the number of
features it is essential for plus beta (gamma). The bits are those of the CRAN package proxirr.

::: landschaft.prioritize.eval_proxirr

::: landschaft.prioritize.eval_proxirr_marxan

::: landschaft.prioritize.ProxIRR

## Calibration and trade-offs

Candidate solutions over a trade-off (a penalty's weight, cost thresholds, budgets, any parameter),
the choice of one by TOPSIS or the knee of the curve, and solving again as a problem changes.

::: landschaft.prioritize.calibrate_penalty

::: landschaft.prioritize.calibrate_budget

::: landschaft.prioritize.cohon_penalty

::: landschaft.prioritize.pareto_front

::: landschaft.prioritize.calibrate

::: landschaft.prioritize.TradeOffs

::: landschaft.prioritize.topsis

::: landschaft.prioritize.knee_point

::: landschaft.prioritize.iterative_solve

::: landschaft.prioritize.protected_cells

## Ranking and optimization together

Zonation's feature transforms as the features of a problem, the two paradigms compared at the same
cost, features from species distribution models, a report of a solution, and problems solved on
coarser blocks.

::: landschaft.prioritize.zonation_features

::: landschaft.prioritize.compare_zonation

::: landschaft.prioritize.ZonationComparison

::: landschaft.prioritize.sdm_features

::: landschaft.prioritize.report

::: landschaft.prioritize.PrioritizationReport

::: landschaft.prioritize.coarsen

::: landschaft.prioritize.refine

## Multi-action planning

Which units to monitor and which actions to take against which threats in them, so that the
features recover (or are kept) to their targets at the least cost, or recover the most within a
budget: prioriactions's models (Salgado-Rojas et al. 2020), bit for bit, from tables or rasters.

::: landschaft.prioritize.actions_data

::: landschaft.prioritize.actions_raster_data

::: landschaft.prioritize.ActionsData

::: landschaft.prioritize.ActionsProblem

::: landschaft.prioritize.ActionsSolution

::: landschaft.prioritize.ActionsPortfolio

## Restoration planning

How much of every converted cell to restore, for the most reduction of the species' extinction risk
(and of the ecosystems' risk of collapse), carbon and ecological intactness within area targets and a
budget, or the cheapest plan for a benefit: the restoration problem of plangea (Strassburg et al. 2020).
The species-area curves are concave, so the optimum is found and proven, by simplicial decomposition
(linear programs over the cells, mixed by a small concave problem); plangea's own procedure (benefit per cost per cell, the target in steps) is
`method="steps"`. The inputs come from `ls.original_vegetation`, `ls.sdm.area_of_habitat` and
`ls.intactness_gradient`. Tutorial: [restoration priorities](../tutorials/restoration-plans.md).

::: landschaft.prioritize.restoration_problem

::: landschaft.prioritize.RestorationProblem

::: landschaft.prioritize.RestorationSolution

::: landschaft.prioritize.ecosystem_areas

::: landschaft.prioritize.eval_risk_reduction

::: landschaft.prioritize.RiskReduction

## Simulated annealing

Marxan 4's runs, unit by unit as the program, and its annealing as a solver of any problem
(`Problem.add_anneal_solver`) for scales where a gap matters less than an answer.

::: landschaft.prioritize.marxan

::: landschaft.prioritize.MarxanResult
