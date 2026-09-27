# Circuit connectivity

Connectivity from electrical circuit theory: omnidirectional current with
moving windows (Omniscape), current and effective resistance between focal
points, sources and grounds, on rasters and on networks (Circuitscape).
Tutorial: [circuit connectivity](../tutorials/circuit-connectivity.md).

`ls.omniscape` is at the top level; the other functions are in the
`ls.circuit` module (`from landschaft import circuit`).

!!! tip "Resistance, conductance and current"
    A resistance map says how hard each cell is to cross (1 = easy, larger =
    harder); conductance is its inverse. Current is injected at sources and
    leaves at grounds; the current through a cell shows how much movement
    funnels through it. The effective resistance between two points falls
    when there are more, shorter or easier paths between them.

## Omnidirectional current

::: landschaft.circuit.omniscape.omniscape
    options:
      show_root_full_path: false

::: landschaft.circuit.omniscape_ini

::: landschaft.circuit.source_from_resistance

::: landschaft.circuit.read_reclass_table

::: landschaft.circuit.reclassify

::: landschaft.circuit.parse_ini

## Focal points on rasters

::: landschaft.circuit.circuitscape_pairwise

::: landschaft.circuit.circuitscape_one_to_all

::: landschaft.circuit.circuitscape_ini

::: landschaft.circuit.CircuitscapeResult
    options:
      members: false

::: landschaft.circuit.FocalPoints

::: landschaft.circuit.IncludedPairs

## Sources and grounds

::: landschaft.circuit.circuitscape_advanced

::: landschaft.circuit.advanced.advanced
    options:
      show_root_full_path: false

::: landschaft.circuit.AdvancedResult
    options:
      members: false

## Networks

::: landschaft.circuit.circuitscape_network_pairwise

::: landschaft.circuit.circuitscape_network_advanced

::: landschaft.circuit.NetworkResult
    options:
      members: false

::: landschaft.circuit.read_graph
