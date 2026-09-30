# Spatial prioritization

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
report those of the order before its last sort).

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
