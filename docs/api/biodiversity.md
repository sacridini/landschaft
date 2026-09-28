# Biodiversity analysis

Species (labels) counted in groups (grid cells, sites or other units): the
data behind analyses of diversity, endemism and phylogenetic patterns. They
live in the `ls.bio` module, a port of Biodiverse (Laffan, Lubarsky and
Rosauer 2010).

Reading data gives the same groups, labels and sample counts as Biodiverse
with the same settings: coordinates are snapped to the cell grid the same
way and groups and labels get the same names (coordinates or name parts joined
with `:`), so results can be compared one to one.

<!-- no-run -->
```python
import landschaft as ls

bd = ls.bio.BaseData.from_table("sites.csv", group_columns=["x", "y"],
                                label_columns=["genus", "species"],
                                cell_sizes=[100000, 100000])
tree = ls.bio.read_tree("tree.nex")
bd.to_raster(bd.group_richness)       # species richness map
```

## Sample counts per group

::: landschaft.bio.BaseData

::: landschaft.bio.snap_coordinates

## Phylogenies

::: landschaft.bio.Tree

::: landschaft.bio.read_tree

::: landschaft.bio.read_trees

## Dissimilarity matrices

::: landschaft.bio.Matrix

::: landschaft.bio.read_matrix
