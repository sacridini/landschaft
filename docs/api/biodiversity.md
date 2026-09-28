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

# a GBIF download, in 50 km cells of an equal-area projection
bd = ls.bio.BaseData.from_occurrences("0012345-260928.zip", [50_000, 50_000],
                                      to_crs="EPSG:6933", max_uncertainty=10_000)
# a stack of species distribution models, presence above 0.5
bd = ls.bio.BaseData.from_raster("sdm_stack.tif", threshold=0.5)
```

## Sample counts per group

::: landschaft.bio.BaseData

::: landschaft.bio.snap_coordinates

## Occurrence records

GBIF downloads (the "simple CSV" zip) and Darwin Core Archives are read in
chunks and cleaned (absences, missing coordinates, uncertainty, basis of
record, GBIF issues, duplicates) before they become sample counts per cell
with `BaseData.from_occurrences`.

::: landschaft.bio.read_occurrences

## Neighbourhoods

Spatial conditions are written as in Biodiverse, in its Perl syntax:
`sp_circle (radius => 200000)`, `sp_block (size => 500000)`,
`$D <= 3 && $nbr_y > $y`, `sp_match_text (text => 'wet', axis => 2)`... They
select the neighbours of each processing group (up to two sets, the second
leaving out the first), optionally only for the groups passing a definition
query. The neighbour lists are the ones Biodiverse uses, including its reuse
of lists between groups for non-overlapping conditions.

::: landschaft.bio.neighbours

::: landschaft.bio.definition_query

::: landschaft.bio.SpatialCondition

::: landschaft.bio.Neighbourhoods

## Spatial analysis

Indices computed over the neighbourhood of every group, as in Biodiverse's
spatial analysis: richness, endemism (weighted, corrected, absolute,
hierarchical), rarity, taxonomic turnover (Sorenson, Jaccard, Bray-Curtis,
Kulczynski, S2, beta diversity, nestedness, range-weighted turnover),
Simpson and Shannon diversity, Rao's quadratic entropy, richness estimators
(Chao1, Chao2, ACE, ICE, Hurlbert's rarefaction) and label and sample count
summaries. With a tree (`tree=`): phylogenetic diversity (PD, local PD,
terminal lists), phylogenetic endemism (PE, central, single, lists, PD
endemism, corrected weighted endemism), clade contributions and losses,
evolutionary distinctiveness (ES, ED, AED, AED-T, corrected weighted rarity),
phylogenetic abundance, phylogenetic turnover (Sorenson, Jaccard, S2,
range-weighted), relative PD and PE (RPD, RPE, against trees with equalised
branch lengths), RWiBaLD and the PhyloCom indices (MPD, MNTD, NRI, NTI, net
VPD, with the exact expected values of Tsirogiannis et al. or, where
Biodiverse samples random sets of tips, its Mersenne Twister draws: pass
`prng_seed`). Values are identical to Biodiverse's.

::: landschaft.bio.spatial_analysis

::: landschaft.bio.list_calculations

::: landschaft.bio.SpatialResult

## Phylogenies

::: landschaft.bio.Tree

::: landschaft.bio.read_tree

::: landschaft.bio.read_trees

## Dissimilarity matrices

::: landschaft.bio.Matrix

::: landschaft.bio.read_matrix
