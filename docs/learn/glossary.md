# Glossary

The terms of the course, with the chapter where each is introduced.

Accessible area (M)
:   The area a species has been able to reach by dispersal over the relevant period; in a distribution model, where its background is drawn. [Ch. 13](sdm.md#records-and-their-problems)

Additive benefit function (ABF)
:   A Zonation rule: the marginal loss of a cell is the sum over the features of the loss of a benefit that saturates with the share kept (R<sup>z</sup>). [Ch. 12.3](prioritization/ranking.md#how-zonation-ranks)

Aggregation
:   The tendency of cells or patches of a class to be clumped together rather
    than dispersed. Measured by `ai`, `clumpy`, `cohesion`, `contag`.
    [Ch. 5](metrics.md#aggregation)

Area-weighted mean (`_am`)
:   The mean of a patch metric in which every patch counts in proportion to
    its area: the value of the patch in which a random cell of the class
    falls. [Ch. 5](metrics.md#patches-classes-and-landscapes)

Attrition
:   The disappearance of patches, typically small ones. [Ch. 10](change.md)

AUC
:   The area under the ROC curve: the probability that a random presence gets a higher prediction than a random background point or absence (the Mann-Whitney statistic); depends on the extent of the background. [Ch. 13](sdm.md#auc)

Autocorrelation length (Sal)
:   The shortest distance at which a surface's autocorrelation falls to 0.2: the size of the grain of its texture. [Ch. 11](surfaces.md#spatial-structure)

Background points
:   A sample of the conditions available in the study area, contrasted with the presences by presence-only distribution models; not absences. [Ch. 13](sdm.md#presences-absences-and-background)

BAM diagram
:   The places where abiotic (A) and biotic (B) conditions suit a species and that it can reach (M); it occurs where the three overlap. [Ch. 13](sdm.md#niche-and-distribution)

Bearing area curve
:   The heights of a surface sorted from the highest, against the share of the surface at or above each; its core, peak and valley zones give Sk, Spk and Svk. [Ch. 11](surfaces.md#the-bearing-area-curve)

Boundary penalty (BLM)
:   A term added to the objective of an optimized plan, proportional to the length of its outline (Marxan's boundary length modifier): it makes plans compact. Its value has units and must be calibrated. [Ch. 12.5](prioritization/spatial.md#boundary-penalties)

Boyce index (continuous)
:   The rank correlation between predicted suitability and the ratio of presences to background predicted in moving windows of suitability: near 1 for a good presence-only model. [Ch. 13](sdm.md#tss-the-boyce-index-and-omission)

Branch and bound
:   How a MILP solver searches: it solves the linear relaxation, splits on a unit with a fractional value (in or out), and discards every branch whose bound is worse than the best plan found. [Ch. 12.4](prioritization/optimization.md#inside-the-solver)

Bridge
:   In MSPA, a non-core strip of habitat connecting two different core
    areas; a structural corridor. [Ch. 4](morphology.md)

Budget
:   The most a plan may cost, in the units of the cost layer; the constraint of the budget objectives (maximum utility, minimum shortfall, ...). [Ch. 12.4](prioritization/optimization.md#budget-objectives)

Calibration (of a penalty)
:   Solving a problem with a range of penalty values and choosing one on the curve of cost against the penalized quantity, e.g. at its knee or by TOPSIS. [Ch. 12.6](prioritization/tradeoffs.md#calibrating-a-penalty)

Circuit theory
:   The modelling of movement as electric current through a landscape of
    resistances; equivalent to random walks. [Ch. 8](circuits.md)

Clamping
:   Holding each predictor at the edge of its training range before projecting a model, so that responses stay flat beyond the data. [Ch. 13](sdm.md#projection-and-extrapolation)

Class
:   A category of a categorical map (a land-cover type); class-level metrics
    summarise all patches of one class. [Ch. 5](metrics.md)

Complementarity
:   The value of an area for what it adds to the areas already chosen: a site with species found elsewhere adds little, one with species found nowhere else adds much. [Ch. 12.1](prioritization.md#core-ideas)

Component
:   A group of habitat patches joined, directly or through others, by links
    shorter than a threshold distance. [Ch. 7](networks.md)

Composition
:   What a landscape contains and in which amounts, regardless of spatial
    arrangement. [Ch. 5](metrics.md#composition-and-configuration)

Configuration
:   The spatial arrangement of the classes: size, shape, isolation,
    aggregation of patches. [Ch. 5](metrics.md#composition-and-configuration)

Connectivity
:   The degree to which the landscape facilitates or impedes movement among
    resource patches; structural, potential or actual.
    [Ch. 6](connectivity.md)

Contagion
:   The tendency of classes to form large, contiguous areas; at the cell
    level, the probability that a neighbour of a habitat cell is habitat
    (*P*<sub>ff</sub>). [Ch. 3](fragmentation.md), [ch. 5](metrics.md)

Contiguity constraint
:   A rule that the selected planning units form one connected network. [Ch. 12.5](prioritization/spatial.md#rules-on-the-shape-of-the-plan)

Core area
:   The part of a patch further from its edge than a given depth, where
    interior conditions prevail. [Ch. 4](morphology.md)

Core-area Zonation (CAZ)
:   A Zonation rule: the marginal loss of a cell follows the features that would lose the largest fraction of what remains of them (`CAZMAX`; `CAZ1` and `CAZ2` combine the fractions as sums or sums of squares). [Ch. 12.3](prioritization/ranking.md#how-zonation-ranks)

Corridor constraint
:   A rule that given areas (terminals) are selected and joined by selected units, through low resistance. [Ch. 12.5](prioritization/spatial.md#rules-on-the-shape-of-the-plan)

Cost distance
:   The accumulated resistance along the cheapest route between two places.
    [Ch. 6](connectivity.md), [ch. 9](restoration.md)

Cost (of a planning unit)
:   What it takes to include a planning unit in a plan: acquisition, opportunity, management cost, or area. Missing values leave the unit out of the problem. [Ch. 12.2](prioritization/concepts.md#cost)

Current density
:   In circuit models, the current through a cell: the net number of
    passages of random walkers. High values mark corridors and pinch points.
    [Ch. 8](circuits.md)

Decision type
:   What can be done with a planning unit: select it or not (binary), any share of it (proportion), or any share up to a limit (semi-continuous). [Ch. 12.2](prioritization/concepts.md#decisions)

Directional dependency
:   A one-way link between planning units, such as a river reach and the reach upstream: managing the first without the second costs a penalty or is not allowed. [Ch. 12.8](prioritization/actions.md#dependencies-between-places)

Dispersal kernel
:   The probability of moving a given distance, often a negative exponential.
    [Ch. 6](connectivity.md#dispersal)

Dissection
:   The subdivision of habitat by thin lines such as roads. [Ch. 10](change.md)

dPC (and its fractions)
:   The percentage of the probability of connectivity lost when a patch or
    link is removed; split into intra, flux and connector fractions.
    [Ch. 7](networks.md#which-patches-matter)

Edge effect
:   Any change in conditions or processes near the boundary between habitat
    and another land cover. [Ch. 3](fragmentation.md#edge-effects)

Edge width
:   In MSPA, the depth of edge effects expressed in cells.
    [Ch. 4](morphology.md#choosing-the-edge-width)

Effective resistance
:   The resistance between two points of a circuit, combining the length and
    the number of routes between them. [Ch. 8](circuits.md)

Efficiency
:   Reaching conservation goals with the least area or cost. [Ch. 12.1](prioritization.md#core-ideas)

Ensemble (of models)
:   An average of several distribution models, often weighted by their cross-validated scores; the spread between them maps uncertainty. [Ch. 13](sdm.md#the-algorithms-in-brief)

Equivalent connected area (EC, ECA)
:   The area of a single patch that would provide the same reachable habitat
    as a whole habitat network. [Ch. 7](networks.md)

Extent
:   The overall size of the area studied. [Ch. 2](scale.md)

Feature (prioritization)
:   Anything a plan should conserve that has an amount in every planning unit: a species, a habitat, an ecosystem service. [Ch. 12.2](prioritization/concepts.md#features)

Fragmentation *per se*
:   The breaking apart of habitat independently of habitat loss: more,
    smaller, more isolated patches for the same amount.
    [Ch. 3](fragmentation.md)

Gap analysis
:   The measure of how much of every feature existing protected areas hold, and of what they miss. [Ch. 12.3](prioritization/ranking.md#weights-costs-and-what-is-already-protected)

Gradient model
:   A view of landscape structure as continuous variation in space, of which patches are a special case; the alternative to the patch-mosaic model. [Ch. 11](surfaces.md#patches-or-gradients)

Grain
:   The finest spatial resolution of the data: the cell size of a raster.
    [Ch. 2](scale.md)

Graph
:   A set of nodes (patches) and links (connections) with attributes and
    weights. [Ch. 7](networks.md)

Habitat
:   The resources and conditions that allow a given species to live and
    reproduce; species-specific, not a land-cover class.
    [Ch. 1](landscape-ecology.md#habitat-is-species-specific)

Habitat amount hypothesis
:   The proposal that species richness at a site depends mainly on the amount
    of habitat around it, not on how it is divided into patches.
    [Ch. 3](fragmentation.md)

Habitat availability (reachability)
:   The amount of habitat reachable from within the habitat, counting the
    habitat of each patch and of the patches connected to it. [Ch. 7](networks.md)

Hierarchic mask
:   In Zonation, levels of cells removed in order (all cells of a level before any of the next): existing reserves ranked on top of the rest. [Ch. 12.3](prioritization/ranking.md#weights-costs-and-what-is-already-protected)

IIC
:   Integral index of connectivity: habitat availability on a binary graph,
    discounting pairs by the number of links between them. [Ch. 7](networks.md)

Irreplaceability
:   How much the conservation goals depend on an area; an area holding the only population of a species cannot be replaced. [Ch. 12.1](prioritization.md#core-ideas)

Islet
:   In MSPA, a patch too small or thin to contain any core. [Ch. 4](morphology.md)

Knee point
:   The point of a trade-off curve farthest from the line between its ends: where improving one criterion starts to cost much more of the other. [Ch. 12.6](prioritization/tradeoffs.md#the-blended-approach)

Landscape
:   An area spatially heterogeneous in at least one factor of interest, at a
    scale set by the organism or process studied. [Ch. 1](landscape-ecology.md)

Landscape mosaic
:   The classification of each neighbourhood by its shares of agriculture,
    natural and developed land. [Ch. 3](fragmentation.md)

Least-cost path
:   The route of minimum accumulated cost between two places; a least-cost
    corridor includes near-optimal routes. [Ch. 9](restoration.md)

Linear relaxation
:   An optimization problem with its whole-number decisions allowed to take fractional values: easy to solve, and its optimum is a bound on the original problem's. [Ch. 12.4](prioritization/optimization.md#the-optimality-gap)

Locked-in, locked-out unit
:   A planning unit that must (existing reserves) or must not (cities, mines) be selected. [Ch. 12.2](prioritization/concepts.md#decisions)

Marginal loss
:   In Zonation, the loss to the features of removing a cell at the point where it is removed, given the cells still remaining. [Ch. 12.3](prioritization/ranking.md#how-zonation-ranks)

Matrix
:   The dominant, most connected land cover in which habitat patches are
    embedded. [Ch. 1](landscape-ecology.md#patches-corridors-and-the-matrix)

Maxent
:   A presence-background distribution model: a log-linear model of the relative occurrence rate with a lasso penalty, equivalent to a Poisson point process model and to infinitely weighted logistic regression. [Ch. 13](sdm.md#the-algorithms-in-brief)

Maximum utility
:   The budget objective that maximizes the total (weighted) amount of the features held, without targets. [Ch. 12.4](prioritization/optimization.md#budget-objectives)

MESS
:   Multivariate environmental similarity surface: for every cell, the similarity of its least similar predictor to the data a model was fitted to; negative where a projection extrapolates. [Ch. 13](sdm.md#projection-and-extrapolation)

Metapopulation
:   A set of local populations in separate patches, linked by dispersal,
    persisting through a balance of local extinction and recolonisation.
    [Ch. 1](landscape-ecology.md)

Minimum mapping unit (MMU)
:   The smallest area delineated on a map; it can make the effective grain
    coarser than the cell size. [Ch. 2](scale.md)

Minimum set problem
:   Meeting a target for every feature at the least cost: the classic formulation of systematic conservation planning. [Ch. 12.4](prioritization/optimization.md#the-minimum-set)

Minimum shortfall
:   The budget objective that minimizes the sum of the targets' shortfalls, each as a share of its target. [Ch. 12.4](prioritization/optimization.md#budget-objectives)

Mixed integer linear program (MILP)
:   An optimization problem with a linear objective, linear constraints and some whole-number variables, such as a decision of 0 or 1 per planning unit; the model of every `ls.prioritize` problem. [Ch. 12.4](prioritization/optimization.md#inside-the-solver)

MOP
:   Mobility-oriented parity: the distance from each cell to its nearest reference conditions, and whether it is outside their ranges. [Ch. 13](sdm.md#projection-and-extrapolation)

Moving window
:   A neighbourhood of fixed size and shape centred on every cell in turn, in
    which a measure is computed to produce a map. [Ch. 3](fragmentation.md)

MSPA
:   Morphological Spatial Pattern Analysis: the classification of every
    habitat cell into core, edge, perforation, bridge, loop, branch or islet.
    [Ch. 4](morphology.md)

Multi-action planning
:   Choosing which management action to take where (monitoring, actions against each threat) to meet recovery and conservation targets at least cost, or with the most benefit. [Ch. 12.8](prioritization/actions.md)

Neutral landscape model
:   An artificial map with known properties, used as a baseline to test the
    effects of pattern. [Ch. 3](fragmentation.md); tools:
    [neutral landscapes](../tutorials/neutral-landscapes.md)

Niche (fundamental, realized)
:   The environmental conditions in which a species can maintain populations (fundamental), and the part of them it occupies given competitors, predators and dispersal (realized). [Ch. 13](sdm.md#niche-and-distribution)

Objective (prioritization)
:   The quantity an optimized plan minimizes or maximizes: cost, shortfall, utility, extinction risk, ... [Ch. 12.4](prioritization/optimization.md)

Omission rate
:   The share of test presences predicted below a threshold set on the training presences (minimum or 10th percentile); much above 0 and 10 % signals overfitting. [Ch. 13](sdm.md#tss-the-boyce-index-and-omission)

Omnidirectional connectivity
:   Connectivity mapped between all places within a moving window, without
    choosing start and end points (Omniscape). [Ch. 8](circuits.md)

Optimality gap
:   The relative difference between the plan a solver returns and the bound no plan can beat: a gap of 0.01 proves the plan is within 1 % of the best possible. [Ch. 12.4](prioritization/optimization.md#the-optimality-gap)

Pareto front
:   The plans for which no other plan is better in one criterion without being worse in another, such as cost and shortfall. [Ch. 12.6](prioritization/tradeoffs.md#money-against-representation)

Patch
:   A group of connected cells of the same class; "connected" by the 4- or
    8-neighbour rule. [Ch. 5](metrics.md)

PC
:   Probability of connectivity: the probability that two random points of
    the landscape fall in habitat areas reachable from each other.
    [Ch. 7](networks.md)

Percolation threshold
:   The proportion of habitat at which a cluster spanning the whole map
    appears in random maps: about 59% with 4 neighbours, 41% with 8.
    [Ch. 3](fragmentation.md#thresholds-and-percolation)

Perforation
:   Holes inside habitat; also the MSPA class of the boundaries of such holes.
    [Ch. 4](morphology.md), [ch. 10](change.md)

Performance curve
:   The share of a feature remaining as the cells of a ranking are removed, from the least to the most important. [Ch. 12.3](prioritization/ranking.md#reading-the-results)

Pinch point
:   A narrow place where movement is funnelled, marked by high current
    density. [Ch. 8](circuits.md)

Planning unit
:   The smallest area a plan can select or not: a grid cell, a property, a watershed, a hexagon. [Ch. 12.2](prioritization/concepts.md#planning-units)

Portfolio (of plans)
:   A set of distinct plans of nearly equal quality; how often a unit is selected in it is a simple irreplaceability. [Ch. 12.6](prioritization/tradeoffs.md#alternative-plans)

Presence-only data
:   Records of where a species was found, without records of where it was looked for and not found: museum specimens, herbarium sheets, most GBIF data. [Ch. 13](sdm.md#presences-absences-and-background)

ProtConn
:   The percentage of a region that is protected and connected.
    [Ch. 7](networks.md#protected-area-connectivity)

Rarity-weighted richness
:   The sum over the features of a unit's amount divided by the feature's total: a quick score of how much rare biodiversity a unit holds. [Ch. 12.7](prioritization/evaluation.md#which-places-matter-most)

Replacement cost
:   How much more a plan costs when a unit is left out and the rest optimized again; zero when an equally good substitute exists. [Ch. 12.7](prioritization/evaluation.md#which-places-matter-most)

Resistance surface
:   A map of the cost, risk or reluctance of crossing each cell.
    [Ch. 6](connectivity.md#resistance-surfaces)

Scale of effect
:   The extent around a site at which landscape structure best predicts a
    biological response. [Ch. 2](scale.md#the-scale-of-effect)

Shrinkage
:   The reduction of patch size, typically from the edges. [Ch. 10](change.md)

Simulated annealing
:   A heuristic that improves a plan by random swaps, sometimes accepting worse plans to escape local optima; Marxan's method. It gives no guarantee of quality. [Ch. 12.4](prioritization/optimization.md#exact-methods-and-heuristics)

Solver
:   A program that solves optimization models, such as HiGHS or Gurobi. [Ch. 12.4](prioritization/optimization.md#inside-the-solver)

Spatial cross-validation
:   Cross-validation with folds made of spatial blocks, so that test records are not next to training ones; random folds overestimate how well a model predicts new places. [Ch. 13](sdm.md#evaluating-a-model)

Spatial prioritization
:   The ranking or selection of areas for conservation actions, balancing many features, costs and constraints. [Ch. 12](prioritization.md)

Species distribution model (SDM)
:   A model relating the records of a species to environmental predictors, projected onto maps of them; also ecological niche or habitat suitability model. [Ch. 13](sdm.md)

Species-area relationship
:   The number of species, or a species' persistence, grows with habitat area as a power, S ∝ A<sup>z</sup>, z near 0.25; the base of the minimum extinction risk objective. [Ch. 12.4](prioritization/optimization.md#extinction-risk-instead-of-targets)

Structuring element
:   In mathematical morphology, the small reference shape used to probe the
    habitat (erosion, dilation). [Ch. 4](morphology.md)

Subregional targets
:   Targets that must be met within each of several regions, such as states, to share the effort among them. [Ch. 12.10](prioritization/process.md#plans-across-regions)

Surface metrics
:   Measures of a continuous raster (ISO 25178, *gradient surface metrics*): amplitude, height distribution, slopes, features and spatial structure, computed without a threshold. [Ch. 11](surfaces.md)

Systematic conservation planning
:   Choosing areas for conservation with explicit goals, measuring what exists, and selecting areas for what they add to the whole (Margules & Pressey 2000). [Ch. 12.1](prioritization.md#systematic-conservation-planning)

Target (prioritization)
:   How much of a feature a plan should hold: a share of its total (relative), an amount (absolute), or one that falls with the feature's range (log-linear). [Ch. 12.2](prioritization/concepts.md#targets)

Target-group background
:   Background points drawn with the same sampling bias as the presences, from the records of the whole taxonomic group, so that the bias cancels. [Ch. 13](sdm.md#records-and-their-problems)

Texture image
:   A map of a surface metric computed in a moving window around every cell. [Ch. 11](surfaces.md#surfaces-in-moving-windows)

Thematic resolution
:   How finely the classes of a map are defined. [Ch. 2](scale.md)

Thinning (spatial)
:   Removing occurrence records until none is closer than a given distance to another, to reduce sampling bias and pseudo-replication. [Ch. 13](sdm.md#records-and-their-problems)

Threat
:   Anything that harms features in a planning unit, with an intensity, and that an action can remove: deforestation, fire, invasive species. [Ch. 12.8](prioritization/actions.md#from-places-to-actions)

Topographic position index (TPI)
:   Elevation minus the mean elevation of a window: positive on ridges, negative in valleys. [Ch. 11](surfaces.md#elevation-terrain-indices)

TOPSIS
:   A multi-criteria method that chooses the alternative closest to the ideal and farthest from the anti-ideal, after scaling the criteria. [Ch. 12.6](prioritization/tradeoffs.md#the-blended-approach)

Transition matrix
:   A table of the cells that moved from each class at one date to each class
    at another. [Ch. 10](change.md)

TSS
:   True skill statistic: sensitivity + specificity − 1 at a threshold, from −1 to 1 (0 for a random model). [Ch. 13](sdm.md#tss-the-boyce-index-and-omission)

VIF
:   Variance inflation factor: 1 / (1 − R²) of a predictor regressed on the others; above 5–10, collinearity makes its effect unstable. [Ch. 13](sdm.md#predictors-and-collinearity)
