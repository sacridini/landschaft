# Sensitivity analysis

Which parameters of a model matter, and how much: Morris's elementary effects
(a screening of many factors with few runs) and Sobol's variance-based indices
(the share of the output's variance due to each factor alone, with its
interactions, and to pairs). The runs and the indices are those of SALib 1.6,
the same numbers; the runs of a landschaft tool can be spread over the
helpers of `ls.parallel`. The `ls.sensitivity` module.

<!-- no-run -->
```python
import numpy as np
import landschaft as ls

d = ls.sensitivity.design("sobol", {"rate": (0.01, 0.1), "threshold": (0.2, 0.8)}, n=256)
y = ls.sensitivity.evaluate(lambda p: my_model(rate=p[0], threshold=p[1]), d)
ls.sensitivity.analyze(d, y).indices                 # s1, st and their intervals

# a landschaft tool, its runs on the helpers of this computer's group
f = ls.parallel.call("surface.gravity", sources, exponent=ls.parallel.param(0))
d = ls.sensitivity.design("morris", {"exponent": (1, 3)}, n=20)
y = ls.sensitivity.evaluate(f, d, on=ls.helpers(), summary=np.mean)
```

## Designs

::: landschaft.sensitivity.design

::: landschaft.sensitivity.Design

## Running the model

::: landschaft.sensitivity.evaluate

## Indices

::: landschaft.sensitivity.analyze

::: landschaft.sensitivity.Sensitivity
