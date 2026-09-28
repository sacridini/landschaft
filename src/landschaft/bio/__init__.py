# SPDX-License-Identifier: GPL-3.0-or-later
"""Biodiversity analyses: a port of Biodiverse (Laffan et al. 2010).

Species (labels) are counted in groups (grid cells or other units). The
:class:`BaseData` table holds the sample counts, :class:`Tree` a phylogeny and
:class:`Matrix` a label dissimilarity matrix; the analyses compute diversity,
endemism and phylogenetic indices over neighbourhoods of groups, randomise the
data to test them, and cluster the groups.

Results match Biodiverse (the Perl original) with the same inputs: the
reference runs use a copy of Biodiverse that iterates its hashes in sorted key
order, and landschaft numbers labels, groups and tree nodes in that order.

Typical use::

    import landschaft as ls

    bd = ls.bio.BaseData.from_table("sites.csv", group_columns=["x", "y"],
                                    label_columns=["genus", "species"],
                                    cell_sizes=[100000, 100000])
    tree = ls.bio.read_tree("tree.nex")
"""

from ._basedata import BaseData, snap_coordinates
from ._gbif import read_occurrences
from ._matrix import Matrix, read_matrix
from ._tree import Tree, read_tree, read_trees

__all__ = [
    "BaseData",
    "Matrix",
    "Tree",
    "read_matrix",
    "read_occurrences",
    "read_tree",
    "read_trees",
    "snap_coordinates",
]
