# Learn landscape ecology

This is a short course on the ideas behind every tool of Landschaft. It is
written for readers who have never studied landscape ecology: students,
analysts who were handed a land-cover map and a question, researchers from
neighbouring fields. By the end you should be able to say **what** each
analysis measures, **why** ecologists measure it, **when** it is the right
tool, and **how** to read and defend its results.

The chapters follow the order of a university course. Each one starts from
the ecological question, introduces the concepts and, where it helps, the
mathematics behind them, then puts them to work on the maps that ship with
Landschaft. Everything is cited to the primary literature, so you can follow
any thread further.

!!! tip "How to read this course"
    - **Read in order the first time.** Later chapters rely on the vocabulary
      of the earlier ones (patch, matrix, grain, extent, edge...).
    - **Run the code.** Every code block runs as it is, in order within a
      chapter: open a notebook next to the page, with Landschaft
      [installed](../getting-started/installation.md).
    - **Answer the questions** at the end of each chapter before opening the
      answers.
    - **Come back to [chapter 14](choosing.md)** when you start a project: it
      goes from questions to tools.

## Chapters

<div class="grid cards" markdown>

-   **1 · [What is a landscape?](landscape-ecology.md)**

    ---

    Pattern, process and scale; patches, corridors and the matrix; island
    biogeography and metapopulations; from a landscape to a map.

-   **2 · [Scale](scale.md)**

    ---

    Grain, extent and thematic resolution; why every measurement depends on
    them; the scale of effect; choosing windows and resolutions.

-   **3 · [Habitat loss and fragmentation](fragmentation.md)**

    ---

    Loss versus fragmentation *per se*, edge effects, thresholds and
    percolation; density, fragmentation classes and landscape mosaics.

-   **4 · [The shape of habitat](morphology.md)**

    ---

    Mathematical morphology in plain words; cores, edges, bridges and islets
    (MSPA); objects by size; parcels and contorted boundaries.

-   **5 · [Landscape metrics](metrics.md)**

    ---

    Composition and configuration; patch, class and landscape levels; the
    families of metrics, their redundancy and their pitfalls; sampling designs.

-   **6 · [Landscape connectivity](connectivity.md)**

    ---

    Structural and functional connectivity, dispersal, resistance surfaces and
    cost distance; the three families of connectivity models.

-   **7 · [Habitat networks](networks.md)**

    ---

    Graph theory for ecologists: IIC, PC and the equivalent connected area,
    the importance of patches and links, protected-area connectivity.

-   **8 · [Circuit theory](circuits.md)**

    ---

    Random walkers and electric current: effective resistance, current
    density, pinch points and omnidirectional connectivity.

-   **9 · [Distance, cost and restoration](restoration.md)**

    ---

    Distance and cost surfaces, least-cost paths and corridors, how to rank
    restoration options by the connectivity they add.

-   **10 · [Landscape change](change.md)**

    ---

    Spatial processes of transformation, map error versus real change,
    transition matrices and connectivity through time.

-   **11 · [Continuous surfaces](surfaces.md)**

    ---

    Gradients instead of patches: surface metrics, the bearing area curve,
    autocorrelation and texture, terrain indices of elevation models.

-   **12 · [Conservation priorities](prioritization.md)**

    ---

    Systematic conservation planning: complementarity and irreplaceability,
    how Zonation ranks a landscape, performance curves, costs, reserves and
    connectivity.

-   **13 · [Species distributions](sdm.md)**

    ---

    Niches and distributions, cleaning and thinning records, background and
    the accessible area, the main algorithms, spatial cross-validation,
    thresholds and extrapolation.

-   **14 · [From question to tool](choosing.md)**

    ---

    A decision guide from ecological questions to analyses, a worked study
    design and a checklist of what to report.

-   **Appendices**

    ---

    A [glossary](glossary.md) of the terms of the course and an annotated
    list of [further reading](reading.md).

</div>

## What the course assumes

Nothing beyond secondary-school mathematics and a first contact with Python:
variables, functions, a `for` loop. Where formulas appear they are explained
term by term, and the code shows the same computation on real data. Readers
who want the software side first can start with the
[quickstart](../getting-started/quickstart.md) and
[core concepts](../getting-started/concepts.md), then come back.

## How the course relates to the rest of the site

| You want to... | Go to |
|---|---|
| understand what an analysis means and when to use it | **this course** |
| run an analysis step by step, with all its options | the [tutorials](../tutorials/index.md) |
| look up a function's parameters and outputs | the [API reference](../api/index.md) |
| know how the results compare with other software | [benchmarks & accuracy](../benchmarks.md) |
