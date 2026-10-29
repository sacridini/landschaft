<p class="ls-chapter">Chapter 15</p>

# Animal movement

The connectivity chapters of this course assumed movement: a dispersal
distance for habitat networks, a random walker for circuits, a cost of
crossing each cell for least-cost paths. Each of those assumptions is a
statement about how animals move, and each is usually set by expert opinion.
Tracking data measure movement instead. A GPS collar records where an animal
was every few minutes for weeks or months, and with some thousands of such
locations come the questions of **movement ecology**: how far animals move
and why, where they go and what they avoid, and what the landscape does to
their routes (Nathan et al. 2008; Kays et al. 2015). This chapter is about
turning tracks into steps, the unit of most analyses, what the sampling rate
does to them, how to describe a track as a whole, and how step selection
functions estimate which parts of the landscape animals choose as they move.

!!! abstract "In this chapter"
    - tracks, their records and their errors: failed fixes, location error,
      outliers, duplicates;
    - steps and turn angles, and the correlated random walk;
    - why step lengths and turn angles depend on the sampling rate, and
      what resampling and bursts are for;
    - the shape of a track: net squared displacement, straightness,
      sinuosity;
    - used and available: from resource selection to step selection
      functions, random steps and conditional logistic regression;
    - integrated step selection, and from selection to resistance maps.

## Tracks

A **track** is the sequence of locations of one animal, ordered in time. A
modern GPS collar computes its position from satellites on a schedule and
stores or transmits it with the time, often with the number of satellites,
an estimate of the error, and the readings of other sensors (temperature,
acceleration). Movebank, the main repository of animal tracking data, stores
them as one table of records, each with a time, an animal (and the tag it
wore) and a location (Kranstauber et al. 2011). Four problems come with the
records.

**Failed fixes.** The collar wakes up but does not see enough satellites:
the record has a time and no location. Failures are not random: they are
more frequent under dense canopy, in deep valleys and when the animal rests
in a den, so dropping them biases a study towards open places (Frair et al.
2010). In the example fishers of the tutorial, 14443 of 47347 records have no
location.

**Location error.** A fix is off by a few metres to tens of metres, more with
few satellites. Two fixes a minute apart can then show a step of 20 m from
an animal that did not move, and a turn of any angle. Errors matter when the
steps are short compared with them.

**Outliers.** Occasionally a fix is kilometres off: a step at an impossible
speed out and straight back. Movebank's users and filters mark them as not
visible.

**Duplicates.** The same time recorded twice for one animal, from a
re-transmission or a merged file, gives a step of zero duration and infinite
speed.

Cleaning a track means dropping the records without a location, the
outliers and the duplicates, and knowing what the failed fixes may hide.

## Steps and turns

Between two consecutive locations, the animal made a **step**. Its length
and its duration give a speed, its direction a heading, and the change of
heading from one step to the next is the **turn angle**:

<div class="ls-formula" markdown>
l<sub>t</sub> = √((x<sub>t+1</sub> − x<sub>t</sub>)² + (y<sub>t+1</sub> − y<sub>t</sub>)²) &nbsp;&nbsp;&nbsp;&nbsp; θ<sub>t</sub> = atan2(y<sub>t+1</sub> − y<sub>t</sub>, x<sub>t+1</sub> − x<sub>t</sub>) &nbsp;&nbsp;&nbsp;&nbsp; α<sub>t</sub> = θ<sub>t</sub> − θ<sub>t−1</sub>, wrapped to (−π, π]
<small>l: step length (in a projected CRS; in longitude and latitude, the distance on the sphere or the ellipsoid); θ: heading; α: turn angle, 0 for an animal that keeps its direction and ±π for one that turns back.</small>
</div>

Checked by hand on the first three locations of a fisher, projected to metres:

```python
import numpy as np
import pandas as pd
import landschaft as ls

tracks = ls.load_tracks(ls.move.example_data("fishers"))
tracks = tracks[~tracks.geometry.is_empty]
f2 = tracks[tracks["individual-local-identifier"] == "F2"].to_crs("EPSG:5070")

x, y = f2.geometry.x.to_numpy()[:3], f2.geometry.y.to_numpy()[:3]
length = np.hypot(np.diff(x), np.diff(y))
heading = np.arctan2(np.diff(y), np.diff(x))
turn = (heading[1] - heading[0] + np.pi) % (2 * np.pi) - np.pi
s = ls.move.steps(f2)
print(f"lengths by hand {length.round(3)}, ls.move.steps {s['sl_'].to_numpy()[:2].round(3)}")
print(f"turn by hand {turn:.6f}, ls.move.steps {s['ta_'].iloc[1]:.6f}")
```

```text
lengths by hand [472.745 620.115], ls.move.steps [472.745 620.115]
turn by hand -0.698614, ls.move.steps -0.698614
```

The simplest model of movement with these two ingredients is the
**correlated random walk**: an animal draws each step length from one
distribution and each turn angle from another, independently (Kareiva &
Shigesada 1983; Turchin 1998). Turn angles concentrated around 0 make a
persistent, directed walk; spread evenly, a diffusive one. Step lengths are
usually skewed, many short and few long, and fitted with gamma, exponential
or log-normal distributions; turn angles with von Mises (the circular normal)
or wrapped Cauchy distributions. The correlated random walk is the null
model of movement analyses, and the source of the random steps of step
selection functions below.

## The sampling rate

A step is not a property of the animal alone: it is the straight line
between two fixes, and the animal's real path between them is longer and
more tortuous. Sampled every 2 minutes, a fisher's zigzags show; every 2
hours, they are summed into one straight line. Step lengths, speeds and turn
angles therefore depend on the sampling rate (Codling & Hill 2005). The same
fisher at five rates:

```python
rows = {}
for rate, tolerance in [("2min", "20s"), ("10min", "1min"), ("30min", "3min"), ("1h", "6min"), ("2h", "12min")]:
    st = ls.move.steps(ls.move.resample(f2, rate, tolerance))
    hours = st["dt_"].dt.total_seconds().sum() / 3600
    rows[rate] = {"steps": len(st), "metres_per_hour": st["sl_"].sum() / hours,
                  "mean_abs_turn_deg": np.degrees(st["ta_"].abs().mean())}
print(pd.DataFrame(rows).T.round(1))
```

```text
        steps  metres_per_hour  mean_abs_turn_deg
2min   1497.0           1532.2               72.9
10min   534.0            923.5               78.9
30min   269.0            672.7               85.0
1h      221.0            341.4              103.2
2h      106.0            251.9              108.0
```

Measured every 2 minutes, F2 covered about 1.5 km per hour; every 2 hours,
a sixth of that. The turns grow sharper on average as the rate coarsens:
over longer intervals the animal has more often come back towards where it
was, the signature of movement within a home range. No rate is the true one.
What matters is to use the same rate for every animal and period compared,
and to choose it for the question: fine for behaviour and fine-scale
selection, coarse for daily ranging and dispersal.

Collars rarely deliver a regular schedule: fixes fail, schedules change, and
different animals carry different settings (the fishers of the example were
recorded every 2, 10 or 15 minutes). **Resampling** keeps the locations
nearest a regular rate, within a tolerance, and splits the track into
**bursts** wherever a location is missing: within a burst all steps last
about the same, and no step bridges a gap.

## The shape of a track

Three summaries describe a whole track. The **net squared displacement**
(NSD) is the squared distance of each location from the first:

<div class="ls-formula" markdown>
NSD<sub>t</sub> = (x<sub>t</sub> − x<sub>0</sub>)² + (y<sub>t</sub> − y<sub>0</sub>)² &nbsp;&nbsp;&nbsp;&nbsp; straightness = D / L
<small>D: the distance from the first to the last location; L: the length of the path, the sum of its steps.</small>
</div>

Plotted against time, NSD levels off for a resident in a home range, rises
steadily for a disperser, and rises, plateaus and falls back for a migrant,
which makes it a classifier of movement strategies (Bunnefeld et al. 2011).
**Straightness** is 1 for a straight line and near 0 for a path that comes
back on itself. It depends on the length of the track, however, so Benhamou
(2004) recommended the **sinuosity** of a correlated random walk instead,
computed from the steps:

<div class="ls-formula" markdown>
S = 2 [p ((1 − c² − s²) / ((1 − c)² + s²) + b²)]<sup>−1/2</sup>
<small>p: the mean step length; c, s: the mean cosine and sine of the turn angles (with s = 0, the fraction is (1 + c) / (1 − c)); b: the coefficient of variation of the step lengths. S has units of length<sup>−1/2</sup>: compare tracks sampled at the same rate.</small>
</div>

Checked by hand for the same fisher, every 30 minutes (the steps of the
whole resampled track, gaps included):

```python
b = ls.move.resample(f2, "30min", "3min").drop(columns="burst_")
st = ls.move.steps(b)
p, cv = st["sl_"].mean(), st["sl_"].std() / st["sl_"].mean()
c, s = np.nanmean(np.cos(st["ta_"])), np.nanmean(np.sin(st["ta_"]))
by_hand = 2 * (p * ((1 - c**2 - s**2) / ((1 - c) ** 2 + s**2) + cv**2)) ** -0.5
print(f"sinuosity by hand {by_hand:.6f}, ls.move.sinuosity {ls.move.sinuosity(b).iloc[0]:.6f}")
```

```text
sinuosity by hand 0.089913, ls.move.sinuosity 0.089913
```

## Used and available

The core question of habitat selection is whether an animal uses a resource
more than its availability would predict. **Resource selection functions**
(RSF) answer it by contrasting used locations with available ones, sampled
at random from an area the animal could have used, in a logistic regression
(Manly et al. 2002). The answer depends on what counts as available, and
Johnson (1980) distinguished orders of selection: of the geographic range,
of the home range within it, of patches within the home range, of items
within patches. An RSF with availability drawn from a whole study area
mixes them, and treats as available places the animal could not have reached
in the time it had.

A **step selection function** (SSF) sets availability by movement (Fortin et
al. 2005; Thurfjell et al. 2014). Each observed step is compared with a set
of **random steps** from the same start, of lengths and turn angles drawn
from the distributions of the observed steps: the places the animal could
have reached in that one step. Each observed step and its random steps form
a **stratum**, and the model is a conditional logistic regression, the
likelihood of a case-control design:

<div class="ls-formula" markdown>
P(step j is the observed one | stratum s) = exp(β · x<sub>j</sub>) / Σ<sub>k ∈ s</sub> exp(β · x<sub>k</sub>)
<small>x<sub>j</sub>: the covariates at the end of step j (or along it); β: the selection coefficients. Each stratum compares only steps from the same place and time, so what differs between places and times (the home range, the season) cancels out.</small>
</div>

exp(β) for a covariate is the **relative selection strength**: how many
times more likely a step is to end where the covariate is one unit higher,
with everything else equal (Avgar et al. 2017; Fieberg et al. 2021). A toy
stratum of one observed step and three random ones, with a coefficient of 1.2
for forest:

```python
forest = np.array([1, 0, 1, 0])   # the observed step (first) and three random steps
w = np.exp(1.2 * forest)
print((w / w.sum()).round(3), round(float(np.exp(1.2)), 2))
```

```text
[0.384 0.116 0.384 0.116] 3.32
```

Each forest step is 3.32 times as likely as each non-forest step to be the
one chosen. The coefficients are estimated by maximising the product of
these probabilities for the observed steps over all strata.

Three practical points matter. The **number of random steps** per observed
step sets the precision: estimates stabilise with tens of random steps, more
for rare covariates (Thurfjell et al. 2014). The random steps must be drawn
in a **projected CRS**, in metres, since a step of 200 m in a random
direction is not a fixed change in longitude and latitude. And the
covariates of a step can be read at its **end** (where the animal chose to
go), at its start (the conditions in which it moved: night, deep snow), or
along it (what it had to cross).

## Integrated step selection

The step lengths and turn angles of the observed steps reflect both how the
animal moves and what it selects: an animal that dashes across fields and
lingers in forest has long steps in the open and short ones in the forest.
Drawing the random steps from the observed distributions, as if movement
were independent of the landscape, biases the selection coefficients
(Forester et al. 2009). The **integrated step selection analysis** (iSSA)
adds the step length, its logarithm and the cosine of the turn angle to the
model (Avgar et al. 2016). Their coefficients correct the distributions the
random steps were drawn from: with gamma step lengths of shape k and scale θ
and a von Mises turn angle of concentration κ,

<div class="ls-formula" markdown>
k' = k + β<sub>ln l</sub> &nbsp;&nbsp;&nbsp;&nbsp; θ' = 1 / (1/θ − β<sub>l</sub>) &nbsp;&nbsp;&nbsp;&nbsp; κ' = κ + β<sub>cos α</sub>
<small>The movement of the animal, freed from the effect of the landscape: its *selection-free* movement kernel. Interactions with covariates (log step length by forest, by night) show how movement changes with conditions.</small>
</div>

The result is a model of both habitat selection and movement, which can
simulate tracks and predict where animals will be (Signer et al. 2017;
Fieberg et al. 2021).

## From selection to resistance

Step selection estimates what the earlier chapters assumed. The relative
selection strength of each cell, exp(β · x), is a measure of how readily an
animal moves into it; its inverse, or a negative exponential transformation
of it, is a **resistance surface** for cost distances, corridors and
circuits ([chapters 6](connectivity.md), [8](circuits.md) and
[9](restoration.md)). Resistances estimated from movement, and from steps in
particular, reflect movement better than ones from habitat suitability or
expert opinion (Zeller et al. 2012; Zeller et al. 2016; Keeley et al. 2016).
Two cautions: selection measured where the animals live (within home
ranges) may not describe dispersal through unfamiliar land, which is what
connectivity is often about; and a few collared animals are a sample of
individuals that differ, so coefficients should be estimated per animal
and their variation reported.

## Common mistakes

| Mistake | Better |
|---|---|
| dropping failed fixes without thought | know where fixes fail (dense canopy, dens); report the fix success |
| comparing steps of animals sampled at different rates | resample all tracks to one rate, in bursts |
| steps across gaps of hours, from a missing fix | bursts: no step bridges a missing location |
| random steps drawn in longitude and latitude | project the tracks to a metric CRS first |
| availability drawn from the whole study area to study fine-scale choices | step selection: availability from each start |
| interpreting selection without the movement terms | log step length, step length and cos turn angle in the model (iSSA) |
| pooling all animals into one model | coefficients per animal, then their mean and variation |
| a resistance map from within-home-range selection used for dispersal | say so, or use dispersal tracks |

!!! success "Key points"
    - A track is a sequence of fixes, with failed fixes (biased towards
      closed places), location error, outliers and duplicates.
    - Steps (length, heading) and turn angles are the units of movement
      analysis; the correlated random walk is their null model.
    - Step lengths, speeds and turn angles depend on the sampling rate:
      compare animals at one rate, resampled in bursts.
    - Net squared displacement separates residents, dispersers and
      migrants; sinuosity measures tortuosity better than straightness.
    - Step selection functions compare each observed step with random steps
      from the same start, in a conditional logistic regression;
      integrated SSA adds the movement terms and corrects the movement
      kernel.
    - Selection coefficients give resistance surfaces for connectivity,
      with care about the behaviour they were measured in.

## Check your understanding

1. Two populations of the same species are tracked, one with fixes every 5
   minutes and one every hour. The first moves 900 m per hour, the second
   300 m. Does the first move more?
2. Why does a step selection function need no definition of the home range,
   when a resource selection function does?
3. An SSF of a deer gives exp(β) = 0.5 for distance to roads, in km. What
   does it mean?

??? success "Answers"
    1. Not necessarily: at 5 minutes the zigzags of the path are measured,
       at an hour they are cut short. Resample the first to one hour and
       compare again.
    2. Because availability is set step by step, by the places the animal
       could reach from where it was; the comparison within each stratum
       removes everything that is the same for the steps of a stratum,
       including where the home range is.
    3. A step is half as likely to end 1 km further from roads, other things
       being equal: the deer selects places near roads (perhaps for the
       vegetation along them). Two kilometres further, a quarter.

## References

<div class="ls-refs" markdown>

- Avgar, T., Lele, S. R., Keim, J. L., & Boyce, M. S. (2017). Relative selection strength: quantifying effect size in habitat- and step-selection inference. *Ecology and Evolution*, 7(14), 5322–5330.
- Avgar, T., Potts, J. R., Lewis, M. A., & Boyce, M. S. (2016). Integrated step selection analysis: bridging the gap between resource selection and animal movement. *Methods in Ecology and Evolution*, 7(5), 619–630.
- Benhamou, S. (2004). How to reliably estimate the tortuosity of an animal's path: straightness, sinuosity, or fractal dimension? *Journal of Theoretical Biology*, 229(2), 209–220.
- Bunnefeld, N., Börger, L., van Moorter, B., Rolandsen, C. M., Dettki, H., Solberg, E. J., & Ericsson, G. (2011). A model-driven approach to quantify migration patterns: individual, regional and yearly differences. *Journal of Animal Ecology*, 80(2), 466–476.
- Codling, E. A., & Hill, N. A. (2005). Sampling rate effects on measurements of correlated and biased random walks. *Journal of Theoretical Biology*, 233(4), 573–588.
- Fieberg, J., Signer, J., Smith, B., & Avgar, T. (2021). A 'How to' guide for interpreting parameters in habitat-selection analyses. *Journal of Animal Ecology*, 90(5), 1027–1043.
- Forester, J. D., Im, H. K., & Rathouz, P. J. (2009). Accounting for animal movement in estimation of resource selection functions: sampling and data analysis. *Ecology*, 90(12), 3554–3565.
- Fortin, D., Beyer, H. L., Boyce, M. S., Smith, D. W., Duchesne, T., & Mao, J. S. (2005). Wolves influence elk movements: behavior shapes a trophic cascade in Yellowstone National Park. *Ecology*, 86(5), 1320–1330.
- Frair, J. L., Fieberg, J., Hebblewhite, M., Cagnacci, F., DeCesare, N. J., & Pedrotti, L. (2010). Resolving issues of imprecise and habitat-biased locations in ecological analyses using GPS telemetry data. *Philosophical Transactions of the Royal Society B*, 365(1550), 2187–2200.
- Johnson, D. H. (1980). The comparison of usage and availability measurements for evaluating resource preference. *Ecology*, 61(1), 65–71.
- Kareiva, P. M., & Shigesada, N. (1983). Analyzing insect movement as a correlated random walk. *Oecologia*, 56(2–3), 234–238.
- Kays, R., Crofoot, M. C., Jetz, W., & Wikelski, M. (2015). Terrestrial animal tracking as an eye on life and planet. *Science*, 348(6240), aaa2478.
- Keeley, A. T. H., Beier, P., & Gagnon, J. W. (2016). Estimating landscape resistance from habitat suitability: effects of data source and nonlinearities. *Landscape Ecology*, 31(9), 2151–2162.
- Kranstauber, B., Cameron, A., Weinzerl, R., Fountain, T., Tilak, S., Wikelski, M., & Kays, R. (2011). The Movebank data model for animal tracking. *Environmental Modelling & Software*, 26(6), 834–835.
- Manly, B. F. J., McDonald, L. L., Thomas, D. L., McDonald, T. L., & Erickson, W. P. (2002). *Resource Selection by Animals: Statistical Design and Analysis for Field Studies* (2nd ed.). Kluwer.
- Nathan, R., Getz, W. M., Revilla, E., Holyoak, M., Kadmon, R., Saltz, D., & Smouse, P. E. (2008). A movement ecology paradigm for unifying organismal movement research. *Proceedings of the National Academy of Sciences*, 105(49), 19052–19059.
- Signer, J., Fieberg, J., & Avgar, T. (2017). Estimating utilization distributions from fitted step-selection functions. *Ecosphere*, 8(4), e01771.
- Thurfjell, H., Ciuti, S., & Boyce, M. S. (2014). Applications of step-selection functions in ecology and conservation. *Movement Ecology*, 2, 4.
- Turchin, P. (1998). *Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants*. Sinauer.
- Zeller, K. A., McGarigal, K., & Whiteley, A. R. (2012). Estimating landscape resistance to movement: a review. *Landscape Ecology*, 27(6), 777–797.
- Zeller, K. A., McGarigal, K., Cushman, S. A., Beier, P., Vickers, T. W., & Boyce, W. M. (2016). Using step and path selection functions for estimating resistance to movement: pumas as a case study. *Landscape Ecology*, 31(6), 1319–1335.

</div>
