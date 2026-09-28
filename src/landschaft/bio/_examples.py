# SPDX-License-Identifier: GPL-3.0-or-later
"""Example data of Biodiverse shipped with the package."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLES = {"sites": "example_sites.csv", "tree": "example_tree.nex"}


def example_data(name: str = "sites") -> Path:
    """Path of Biodiverse's example data (GPL-3, Laffan et al.).

    Parameters
    ----------
    name : {"sites", "tree"}
        ``"sites"``: 365 records of 31 species (columns ``num``, ``genus``,
        ``species``, ``x``, ``y``; projected metres, Australia);
        ``"tree"``: a Nexus phylogeny of those species (``Genus:sp1``...).

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import landschaft as ls
    >>> bd = ls.bio.BaseData.from_table(ls.bio.example_data("sites"), group_columns=["x", "y"],
    ...                                 label_columns=["genus", "species"], cell_sizes=[100000, 100000])
    >>> bd
    BaseData(127 groups, 31 labels, cell_sizes=[100000.0, 100000.0], 365 records)
    >>> len(ls.bio.read_tree(ls.bio.example_data("tree")).terminal_names)
    31
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return Path(str(resources.files("landschaft.bio") / "data" / EXAMPLES[name]))
