# Glossary

The terms of the course, with the chapter where each is introduced.

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

Autocorrelation length (Sal)
:   The shortest distance at which a surface's autocorrelation falls to 0.2: the size of the grain of its texture. [Ch. 11](surfaces.md#spatial-structure)

Bearing area curve
:   The heights of a surface sorted from the highest, against the share of the surface at or above each; its core, peak and valley zones give Sk, Spk and Svk. [Ch. 11](surfaces.md#the-bearing-area-curve)

Bridge
:   In MSPA, a non-core strip of habitat connecting two different core
    areas; a structural corridor. [Ch. 4](morphology.md)

Circuit theory
:   The modelling of movement as electric current through a landscape of
    resistances; equivalent to random walks. [Ch. 8](circuits.md)

Class
:   A category of a categorical map (a land-cover type); class-level metrics
    summarise all patches of one class. [Ch. 5](metrics.md)

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

Core area
:   The part of a patch further from its edge than a given depth, where
    interior conditions prevail. [Ch. 4](morphology.md)

Cost distance
:   The accumulated resistance along the cheapest route between two places.
    [Ch. 6](connectivity.md), [ch. 9](restoration.md)

Current density
:   In circuit models, the current through a cell: the net number of
    passages of random walkers. High values mark corridors and pinch points.
    [Ch. 8](circuits.md)

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

Equivalent connected area (EC, ECA)
:   The area of a single patch that would provide the same reachable habitat
    as a whole habitat network. [Ch. 7](networks.md)

Extent
:   The overall size of the area studied. [Ch. 2](scale.md)

Fragmentation *per se*
:   The breaking apart of habitat independently of habitat loss: more,
    smaller, more isolated patches for the same amount.
    [Ch. 3](fragmentation.md)

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

IIC
:   Integral index of connectivity: habitat availability on a binary graph,
    discounting pairs by the number of links between them. [Ch. 7](networks.md)

Islet
:   In MSPA, a patch too small or thin to contain any core. [Ch. 4](morphology.md)

Landscape
:   An area spatially heterogeneous in at least one factor of interest, at a
    scale set by the organism or process studied. [Ch. 1](landscape-ecology.md)

Landscape mosaic
:   The classification of each neighbourhood by its shares of agriculture,
    natural and developed land. [Ch. 3](fragmentation.md)

Least-cost path
:   The route of minimum accumulated cost between two places; a least-cost
    corridor includes near-optimal routes. [Ch. 9](restoration.md)

Matrix
:   The dominant, most connected land cover in which habitat patches are
    embedded. [Ch. 1](landscape-ecology.md#patches-corridors-and-the-matrix)

Metapopulation
:   A set of local populations in separate patches, linked by dispersal,
    persisting through a balance of local extinction and recolonisation.
    [Ch. 1](landscape-ecology.md)

Minimum mapping unit (MMU)
:   The smallest area delineated on a map; it can make the effective grain
    coarser than the cell size. [Ch. 2](scale.md)

Moving window
:   A neighbourhood of fixed size and shape centred on every cell in turn, in
    which a measure is computed to produce a map. [Ch. 3](fragmentation.md)

MSPA
:   Morphological Spatial Pattern Analysis: the classification of every
    habitat cell into core, edge, perforation, bridge, loop, branch or islet.
    [Ch. 4](morphology.md)

Neutral landscape model
:   An artificial map with known properties, used as a baseline to test the
    effects of pattern. [Ch. 3](fragmentation.md); tools:
    [neutral landscapes](../tutorials/neutral-landscapes.md)

Omnidirectional connectivity
:   Connectivity mapped between all places within a moving window, without
    choosing start and end points (Omniscape). [Ch. 8](circuits.md)

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

Pinch point
:   A narrow place where movement is funnelled, marked by high current
    density. [Ch. 8](circuits.md)

ProtConn
:   The percentage of a region that is protected and connected.
    [Ch. 7](networks.md#protected-area-connectivity)

Resistance surface
:   A map of the cost, risk or reluctance of crossing each cell.
    [Ch. 6](connectivity.md#resistance-surfaces)

Scale of effect
:   The extent around a site at which landscape structure best predicts a
    biological response. [Ch. 2](scale.md#the-scale-of-effect)

Shrinkage
:   The reduction of patch size, typically from the edges. [Ch. 10](change.md)

Structuring element
:   In mathematical morphology, the small reference shape used to probe the
    habitat (erosion, dilation). [Ch. 4](morphology.md)

Surface metrics
:   Measures of a continuous raster (ISO 25178, *gradient surface metrics*): amplitude, height distribution, slopes, features and spatial structure, computed without a threshold. [Ch. 11](surfaces.md)

Texture image
:   A map of a surface metric computed in a moving window around every cell. [Ch. 11](surfaces.md#surfaces-in-moving-windows)

Thematic resolution
:   How finely the classes of a map are defined. [Ch. 2](scale.md)

Topographic position index (TPI)
:   Elevation minus the mean elevation of a window: positive on ridges, negative in valleys. [Ch. 11](surfaces.md#elevation-terrain-indices)

Transition matrix
:   A table of the cells that moved from each class at one date to each class
    at another. [Ch. 10](change.md)
