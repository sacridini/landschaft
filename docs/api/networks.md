# Habitat networks

Graphs of habitat patches linked by distance or by dispersal probability,
their connectivity indices (IIC, PC, ECA...), the importance of every patch
and link, protected-area connectivity (ProtConn), grids and time series, and
the network of MSPA cores and bridges. Tutorial:
[habitat networks](../tutorials/habitat-networks.md).

The graph class is `ls.HabitatGraph`; the other functions are in the
`ls.conn` module (`import landschaft as ls`, then `ls.conn.protconn(...)`).

## Habitat graphs

::: landschaft.HabitatGraph
    options:
      members:
        - from_raster
        - from_vector
        - from_samc
        - from_conefor
        - n_nodes
        - overall
        - node_importance
        - link_importance
        - link_change
        - components
        - centrality
        - to_raster
        - to_geodataframe
        - to_conefor

## Nodes and distances

::: landschaft.conn.node_table

::: landschaft.conn.distance_table

::: landschaft.conn.settlement_probabilities

::: landschaft.conn.boundary_vertices

## Protected-area connectivity

::: landschaft.conn.protconn

::: landschaft.conn.protconn_regions

::: landschaft.conn.protconn_raster

::: landschaft.conn.protconn_nodes

::: landschaft.conn.ProtConnNodes
    options:
      members: false

::: landschaft.conn.protconn_indicators

## Grids

::: landschaft.conn.make_grid

::: landschaft.conn.connectivity_grid

::: landschaft.conn.eca_change_grid

## Change and neighbourhoods

::: landschaft.conn.eca_change

::: landschaft.conn.focal_nodes

## Metapopulations

Hanski's incidence function model as R's MetaLandSim (`spom`) iterates it: the colonisation and extinction of the
patches of a network over time, the same run as R's for a seed.

::: landschaft.conn.metapopulation

::: landschaft.conn.Metapopulation

## Networks of MSPA cores and bridges

::: landschaft.gtn_components

::: landschaft.GTNComponentsResult

::: landschaft.gtn_importance

::: landschaft.GTNImportanceResult

::: landschaft.conefor_inputs
