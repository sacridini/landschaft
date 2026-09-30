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
