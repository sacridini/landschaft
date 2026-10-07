# Uncertainty

Propagating the uncertainty of inputs to results: the standard deviation of
a total of uncertain pixel values, of a product of two uncertain layers, of a
result with respect to uncertain parameters (the delta method, checked by
Monte Carlo), and confidence intervals. The `ls.uncertainty` module; the
forms are those plangea uses in its restoration plans, with their errors
corrected (each function's notes say which).

<!-- no-run -->
```python
import landschaft as ls

ls.uncertainty.sum_sd(carbon_sd, weights=area * restored)          # a total's deviation
ls.uncertainty.product_sd(x, sd_x, v, sd_v)                         # each product's
d = ls.uncertainty.delta_method(f, params, sd)                      # d.value, d.sd
low, high = ls.uncertainty.interval(d.value, d.sd, level=0.95)
```

## Totals and products

::: landschaft.uncertainty.sum_sd

::: landschaft.uncertainty.product_sd

## Parameters

::: landschaft.uncertainty.delta_method

::: landschaft.uncertainty.DeltaResult

::: landschaft.uncertainty.monte_carlo

::: landschaft.uncertainty.MonteCarloResult

## Intervals

::: landschaft.uncertainty.interval
