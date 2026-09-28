# SPDX-License-Identifier: GPL-3.0-or-later
"""Inputs of the Biodiverse reference runs (tools/make_reference_bio.py).

Each case gives the arguments of Biodiverse's import (``biodiverse``) and of the
landschaft call that must reproduce it (``landschaft``). File names are
relative to ``tests/data/bio``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "bio"
REFERENCE = Path(__file__).resolve().parent / "data" / "reference" / "bio"

SITE = "Example_site_data.csv"


def _bd(file, group_columns, label_columns, cell_sizes, cell_origins=None, **extra):
    biodiverse = {
        "file": file,
        "group_columns": group_columns,
        "label_columns": label_columns,
        "cell_sizes": cell_sizes,
        **extra,
    }
    if cell_origins is not None:
        biodiverse["cell_origins"] = cell_origins
    landschaft = {
        "group_columns": group_columns,
        "label_columns": label_columns,
        "cell_sizes": cell_sizes,
        "cell_origins": cell_origins,
    }
    rename = {"sample_count_columns": "sample_count_columns", "data_in_matrix_form": "matrix_form",
              "label_start_col": "label_start_col", "binarise_counts": "binarise_counts",
              "allow_empty_groups": "allow_empty_groups", "allow_empty_labels": "allow_empty_labels",
              "include_columns": "include_columns", "exclude_columns": "exclude_columns",
              "input_sep_char": "sep", "input_quote_char": "quote"}
    for k, v in extra.items():
        landschaft[rename[k]] = v
    return {"biodiverse": biodiverse, "landschaft": landschaft, "file": file}


BASEDATA_CASES = {
    "site_100k": _bd(SITE, [3, 4], [1, 2], [100000, 100000]),
    "site_origin": _bd(SITE, [3, 4], [1, 2], [200000, 150000], [50000, 25000]),
    "site_raw": _bd(SITE, [3, 4], [1, 2], [0, 0]),
    "site_text_axis": _bd(SITE, [2, 4], [1], [-1, 100000]),
    "site_counts": _bd(SITE, [3, 4], [2], [50000, 50000], sample_count_columns=[0]),
    "site_binarise": _bd(SITE, [3, 4], [1, 2], [300000, 300000], sample_count_columns=[0], binarise_counts=1),
    "site_one_axis": _bd(SITE, [3], [1, 2], [100000]),
    "numeric": _bd("Example_numeric_data.txt", [1, 2], [0], [10, 10]),
    "site_matrix_form": _bd("Example_site_data_matrix_form.csv", [1, 2], [], [25000, 25000],
                            data_in_matrix_form=1, label_start_col=3),
    "decimal": _bd("synthetic_decimal.csv", [1, 2], [0], [0.1, 0.1], [0.03, -0.02], sample_count_columns=[3]),
    "decimal_empty": _bd("synthetic_decimal.csv", [1, 2], [0], [0.25, 0.25], sample_count_columns=[3],
                         allow_empty_groups=1, allow_empty_labels=1),
    "quoted": _bd("synthetic_quoted.csv", [2, 3], [0, 1], [-1, 1.5], exclude_columns=[4],
                  input_sep_char=",", input_quote_char='"'),
}

# Biodiverse's feature import (import_data_shapefile); landschaft: BaseData.from_vector
VECTOR_CASES = {
    "points": {"files": ["site_points.shp"], "labels": ["genus", "species"], "cell_sizes": [100000, 100000]},
    "points_counts": {"files": ["site_points.shp"], "labels": ["species"], "cell_sizes": [50000, 50000],
                      "counts": ["num"], "cell_origins": [10000, 20000]},
    "points_raw": {"files": ["site_points.shp"], "labels": ["species"], "cell_sizes": [0, 0]},
    "polygons_cells": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000]},
    "polygons_cells_origin": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [200000, 150000],
                              "cell_origins": [50000, 25000]},
    "polygons_area": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000],
                      "measure": "area"},
    "lines_cells": {"files": ["polyline data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000]},
    "lines_length": {"files": ["polyline data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000],
                     "measure": "length"},
}

# spatial conditions on the site data (100 km cells); polygons in site coordinates
_POLY = "[[3000000,1000000],[3000000,3000000],[4000000,3000000],[3500000,2200000],[4000000,1000000],[3000000,1000000]]"
NEIGHBOUR_CASES = {
    "D_le": ["$D <= 300000"],
    "C_le": ["$C <= 2"],
    "circle": ["sp_circle (radius => 250000)"],
    "circle_axis": ["sp_circle (radius => 250000, axes => [0])"],
    "circle_cell": ["sp_circle_cell (radius => 2)"],
    "circle_cell_axes": ["sp_circle_cell (radius => 1.5, axes => [1])"],
    "rectangle": ["sp_rectangle (sizes => [300000, 500000])"],
    "rectangle_axes": ["sp_rectangle (sizes => [500000, 300000], axes => [1, 0])"],
    "annulus": ["sp_annulus (inner_radius => 100000, outer_radius => 300000)"],
    "annulus_axes": ["sp_annulus (inner_radius => 100000, outer_radius => 300000, axes => [0])"],
    "square": ["sp_square (size => 500000)"],
    "square_cell": ["sp_square_cell (size => 3)"],
    "ellipse": ["sp_ellipse (major_radius => 400000, minor_radius => 150000, rotate_angle_deg => 30)"],
    "ellipse_rad": ["sp_ellipse (major_radius => 300000, minor_radius => 100000, rotate_angle => 1.5714)"],
    "block": ["sp_block (size => 300000)"],
    "block_axis": ["sp_block (size => [300000, undef])"],
    "block_origin": ["sp_block (size => 400000, origin => [50000, 150000])"],
    "self_only": ["sp_self_only()"],
    "select_all": ["sp_select_all()"],
    "left": ["sp_is_left_of (vector_angle_deg => 45)"],
    "right": ["sp_is_right_of (vector_angle => 1.2)"],
    "in_line": ["sp_in_line_with ()"],
    "d_list": ["$d[0] >= 0 && $D[1] <= 200000"],
    "nbr_y": ["$nbr_y > $y && $D <= 400000"],
    "match_text": ["sp_match_text (text => '3250000', axis => 0)"],
    "match_text_proc": ["sp_match_text (text => $coord[1], axis => 1) && $D < 1000000"],
    "match_regex": ["sp_match_regex (re => qr/^3[0-4]/, axis => 0, type => 'nbr')"],
    "richness": ["sp_richness_greater_than (threshold => 5)"],
    "redundancy": ["sp_redundancy_greater_than (threshold => 0.3)"],
    "not_empty": ["sp_group_not_empty() && $D <= 200000"],
    "select_element": ["sp_select_element (element => '3350000:850000')"],
    "point_in_poly": [f"sp_point_in_poly (polygon => {_POLY})"],
    "point_in_poly_proc": [f"sp_point_in_poly (polygon => {_POLY}, point => \\@coord) && $D <= 200000"],
    "poly_shape": ["sp_point_in_poly_shape (file => 'SHAPES')"],
    "poly_shape_field": ["sp_point_in_poly_shape (file => 'SHAPES', field_name => 'zone', field_val => 2)"],
    "same_poly": ["sp_points_in_same_poly_shape (file => 'SHAPES')"],
    "sequence": ["sp_select_sequence (frequency => 3)"],
    "sequence_offset": ["sp_select_sequence (frequency => 4, first_offset => 2, reverse_order => 1)"],
    "my_var": ["my $r = 150000; $D <= $r * 2"],
    "coord_ids": ["$coord_id1 eq $coord_id2"],
    "abs_d": ["abs($d[0]) <= 200000 and abs($d[1]) < 300000"],
    "or": ["sp_circle (radius => 200000) || sp_block (size => 600000)"],
    "not": ["!sp_circle (radius => 150000) && $D <= 350000"],
    "ternary": ["$D <= 200000 ? 1 : $nbr_x > $x && $D <= 300000"],
    "pow": ["$d[0]**2 + $d[1]**2 <= 250000**2"],
    "two_sets": ["sp_self_only()", "sp_circle (radius => 300000)"],
    "two_sets_block": ["sp_block (size => 300000)", "sp_block (size => 900000)"],
    "defq": ["sp_circle (radius => 200000)", "$D <= 500000"],
}
NEIGHBOUR_DEFQ = {"defq": "$y > 2000000 && sp_richness_greater_than (threshold => 2)"}

# spatial analyses (phase 3 calculations); "calcs": "one" = those needing one set, "all" = every one
SPATIAL_CASES = {
    "self_all": {"basedata": "site_100k", "conditions": ["sp_self_only()"], "calcs": "one"},
    "circle_all": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"], "calcs": "one"},
    "two_sets": {"basedata": "site_100k", "conditions": ["sp_self_only()", "sp_circle (radius => 300000)"],
                 "calcs": "all"},
    "block": {"basedata": "site_100k", "conditions": ["sp_block (size => 400000)"], "calcs": "one"},
    "defq": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 300000)"], "calcs": "one",
             "definition_query": "$y > 2000000"},
    "counts_circle": {"basedata": "site_counts", "conditions": ["sp_circle (radius => 150000)"], "calcs": "one"},
    "counts_two": {"basedata": "site_counts",
                   "conditions": ["sp_circle (radius => 100000)", "sp_circle (radius => 250000)"], "calcs": "all"},
    "decimal_two": {"basedata": "decimal", "conditions": ["sp_circle (radius => 0.25)", "$D <= 0.6"],
                    "calcs": "all"},
    # Biodiverse's ACE divides by zero on some of these blocks (a2 == 1)
    "decimal_block": {"basedata": "decimal", "conditions": ["sp_block (size => 0.5)"], "calcs": "one",
                      "exclude": ["calc_ace", "calc_ice"]},
}
# calculations of the index engine (phase 3)
PHASE3_CALCS = [
    "calc_richness", "calc_redundancy", "calc_abc_counts", "calc_d", "calc_elements_used",
    "calc_nonempty_elements_used", "calc_element_lists_used", "calc_element_lists_used_as_arrays",
    "calc_label_count_quantile_position", "calc_local_range_lists", "calc_local_range_stats",
    "calc_local_sample_count_lists", "calc_local_sample_count_quantiles", "calc_local_sample_count_stats",
    "calc_beta_diversity", "calc_bray_curtis", "calc_bray_curtis_norm_by_gp_counts", "calc_jaccard",
    "calc_kulczynski2", "calc_nestedness_resultant", "calc_rw_turnover", "calc_s2", "calc_simpson_shannon",
    "calc_sorenson", "calc_tx_rao_qe", "calc_endemism_central", "calc_endemism_central_lists",
    "calc_endemism_central_normalised", "calc_endemism_whole", "calc_endemism_whole_lists",
    "calc_endemism_whole_normalised", "calc_endemism_absolute", "calc_endemism_absolute_lists",
    "calc_rarity_central", "calc_rarity_central_lists", "calc_rarity_whole", "calc_rarity_whole_lists",
    "calc_chao1", "calc_chao2", "calc_ace", "calc_ice", "calc_hurlbert_es",
    "calc_endemism_central_hier_part", "calc_endemism_whole_hier_part", "calc_hierarchical_label_ratios",
]
TWO_SET_CALCS = {
    "calc_abc_counts", "calc_beta_diversity", "calc_bray_curtis", "calc_bray_curtis_norm_by_gp_counts",
    "calc_jaccard", "calc_kulczynski2", "calc_nestedness_resultant", "calc_rw_turnover", "calc_s2", "calc_sorenson",
    "calc_hierarchical_label_ratios",
}


def spatial_calcs(case: dict) -> list[str]:
    calcs = list(PHASE3_CALCS) if case["calcs"] == "all" else [c for c in PHASE3_CALCS if c not in TWO_SET_CALCS]
    return [c for c in calcs if c not in case.get("exclude", [])]


# phylogenetic indices (phase 4)
PHYLO_CALCS = [
    "calc_count_labels_on_tree", "calc_labels_not_on_tree", "calc_labels_on_tree", "calc_last_shared_ancestor_props",
    "calc_pd", "calc_pd_clade_contributions", "calc_pd_clade_loss", "calc_pd_clade_loss_ancestral",
    "calc_pd_endemism", "calc_pd_local", "calc_pd_node_list", "calc_pd_terminal_node_count",
    "calc_pd_terminal_node_list", "calc_pe", "calc_pe_central", "calc_pe_central_cwe", "calc_pe_central_lists",
    "calc_pe_clade_contributions", "calc_pe_clade_loss", "calc_pe_clade_loss_ancestral", "calc_pe_lists",
    "calc_pe_single", "calc_phylo_abc", "calc_phylo_abundance", "calc_phylo_aed", "calc_phylo_aed_t",
    "calc_phylo_aed_t_wtlists", "calc_phylo_corrected_weighted_endemism", "calc_phylo_corrected_weighted_rarity",
    "calc_phylo_jaccard", "calc_phylo_rw_turnover", "calc_phylo_s2", "calc_phylo_sorenson",
    # relative (phase 5a)
    "calc_labels_on_trimmed_tree", "calc_labels_not_on_trimmed_tree", "calc_phylo_rpd1", "calc_phylo_rpd2",
    "calc_phylo_rpe1", "calc_phylo_rpe2", "calc_phylo_rpe_central", "calc_rwibald",
]
PHYLO_TWO_SET = {"calc_phylo_abc", "calc_phylo_jaccard", "calc_phylo_rw_turnover", "calc_phylo_s2",
                 "calc_phylo_sorenson"}
PHYLO_CASES = {
    "circle": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_circle (radius => 200000)"],
               "calcs": "one"},
    "self": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_self_only()"], "calcs": "one"},
    "two_sets": {"basedata": "site_100k", "tree": "Example_tree.nex",
                 "conditions": ["sp_self_only()", "sp_circle (radius => 300000)"], "calcs": "all"},
    "block": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_block (size => 400000)"],
              "calcs": "one"},
    "trimmed": {"basedata": "site_100k", "tree": "tree_trim.nex",
                "conditions": ["sp_circle (radius => 150000)", "sp_circle (radius => 400000)"], "calcs": "all"},
    "counts": {"basedata": "site_counts", "tree": "tree_species.nex",
               "conditions": ["sp_circle (radius => 100000)", "sp_circle (radius => 250000)"], "calcs": "all"},
    # the same calculations mixed with non-phylogenetic ones (changes the pre_calc order)
    "mixed": {"basedata": "site_counts", "tree": "tree_species.nex", "conditions": ["sp_circle (radius => 200000)"],
              "calcs": "one", "extra": ["calc_richness", "calc_local_range_lists", "calc_endemism_whole"]},
}


# PhyloCom indices (phase 5b)
PHYLOCOM_CALCS = [
    "calc_phylo_mpd_mntd1", "calc_phylo_mpd_mntd2", "calc_phylo_mpd_mntd3", "calc_nri_nti1", "calc_nri_nti2",
    "calc_nri_nti3", "calc_nri_nti_expected_values",
]
PHYLOCOM_VPD = ["calc_net_vpd", "calc_vpd_expected_values"]
PHYLOCOM_CASES = {
    # ultrametric (Example_tree): exact expected values, path cache filled as groups are processed
    "um_circle": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_circle (radius => 200000)"]},
    "um_two_sets": {"basedata": "site_counts", "tree": "tree_species.nex",
                    "conditions": ["sp_self_only()", "sp_circle (radius => 150000)"]},
    # variances sampled on an ultrametric tree (their draws go through the path cache)
    "um_vpd": {"basedata": "site_counts", "tree": "tree_species.nex", "conditions": ["sp_circle (radius => 150000)"],
               "vpd": True, "seed": 1234},
    # ultrametric with polytomies: only adjacent children share cached lengths
    "polytomy": {"basedata": "site_100k", "tree": "tree_polytomy.nex", "conditions": ["sp_circle (radius => 300000)"]},
    "polytomy_block": {"basedata": "site_100k", "tree": "tree_polytomy.nex",
                       "conditions": ["sp_block (size => 300000)"], "vpd": True, "seed": 5},
    # not ultrametric: sampled NTI
    "sampled": {"basedata": "site_100k", "tree": "tree_trim.nex", "conditions": ["sp_circle (radius => 200000)"],
                "seed": 42},
    "sampled_vpd": {"basedata": "site_100k", "tree": "tree_nu.nex", "conditions": ["sp_block (size => 300000)"],
                    "vpd": True, "seed": 7},
    # every tip in the sample: mean nearest neighbour distance of the tree
    "all_tips": {"basedata": "site_100k", "tree": "tree_nu.nex", "conditions": ["sp_select_all()"], "seed": 3},
}


# element properties, numeric labels and matrices (phase 6)
LABEL_PROPS = {"file": "label_props.csv", "element_columns": [0, 1], "properties": {"prop_a": 2, "prop_b": 3, "prop_c": 4}}
GROUP_PROPS = {"file": "gp_props.csv", "element_columns": [0], "properties": {"num_mean": 3, "num_min": 4, "num_range": 5}}
PROP_CALCS = [
    "calc_lbprop_lists", "calc_lbprop_data", "calc_lbprop_hashes", "calc_lbprop_stats", "calc_lbprop_quantiles",
    "calc_lbprop_gistar", "calc_lbprop_hashes_abc2", "calc_lbprop_stats_abc2", "calc_lbprop_quantiles_abc2",
    "calc_lbprop_gistar_abc2", "calc_gpprop_lists", "calc_gpprop_hashes", "calc_gpprop_stats", "calc_gpprop_quantiles",
    "calc_gpprop_gistar",
]
NUMERIC_CALCS = [
    "calc_numeric_label_stats", "calc_numeric_label_other_means", "calc_numeric_label_quantiles",
    "calc_numeric_label_data", "calc_num_labels_gistar",
]
MATRIX_CALCS = ["calc_matrix_stats", "calc_mx_rao_qe"]
PROPERTY_CASES = {
    "props_circle": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"], "calcs": PROP_CALCS},
    "props_two_sets": {"basedata": "site_100k", "conditions": ["sp_self_only()", "sp_circle (radius => 300000)"],
                       "calcs": PROP_CALCS},
    # without the calculations that sort the values first
    "props_unsorted": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 150000)"],
                       "calcs": ["calc_lbprop_stats", "calc_lbprop_gistar", "calc_gpprop_stats", "calc_gpprop_gistar",
                                 "calc_lbprop_data"]},
    "numeric_circle": {"basedata": "numeric", "conditions": ["sp_circle (radius => 15)"], "calcs": NUMERIC_CALCS},
    "numeric_two_sets": {"basedata": "numeric", "conditions": ["sp_self_only()", "sp_circle (radius => 25)"],
                         "calcs": NUMERIC_CALCS + ["calc_numeric_label_dissimilarity"]},
    "numeric_stats_only": {"basedata": "numeric", "conditions": ["sp_circle (radius => 15)"],
                           "calcs": ["calc_numeric_label_stats", "calc_num_labels_gistar"]},
    "matrix_circle": {"basedata": "site_100k", "matrix": "full", "conditions": ["sp_circle (radius => 200000)"],
                      "calcs": MATRIX_CALCS},
    "matrix_two_sets": {"basedata": "site_100k", "matrix": "full",
                        "conditions": ["sp_self_only()", "sp_circle (radius => 300000)"],
                        "calcs": MATRIX_CALCS + ["calc_compare_dissim_matrix_values"]},
    "matrix_sparse": {"basedata": "site_100k", "matrix": "sparse", "conditions": ["sp_circle (radius => 300000)"],
                      "calcs": MATRIX_CALCS},
}


def property_case_uses(case: dict) -> tuple[bool, bool]:
    """Whether a case needs the label and the group properties."""
    calcs = case["calcs"]
    return any(c.startswith("calc_lbprop") for c in calcs), any(c.startswith("calc_gpprop") for c in calcs)


def write_label_props() -> None:
    """Synthetic label properties for the site data (decimals, repeated values, gaps)."""
    import csv

    import numpy as np

    rng = np.random.default_rng(11)
    labels = [f"sp{i}" for i in range(1, 32)]
    with open(DATA / "label_props.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["genus", "species", "prop_a", "prop_b", "prop_c"])
        pool_a = [f"{v:.3f}" for v in rng.uniform(0.5, 40, 12)]
        for k, sp in enumerate(labels):
            a = pool_a[int(rng.integers(len(pool_a)))]
            b = str(int(rng.integers(1, 6)))
            c = "" if k % 5 == 0 else ("n/a" if k % 7 == 0 else f"{rng.normal(100, 30):.6g}")
            w.writerow(["Genus", sp, a, b, c])


# randomisations (phase 7)
RAND_CALCS = ["calc_richness", "calc_endemism_whole", "calc_pe", "calc_phylo_rpe2", "calc_pe_central",
              "calc_phylo_rpe_central", "calc_pd"]
RAND_CASES = {
    "csr_circle": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_circle (radius => 200000)"],
                   "calcs": RAND_CALCS + ["calc_pe_lists", "calc_endemism_whole_lists", "calc_local_range_lists"],
                   "function": "rand_csr_by_group", "iterations": 9, "seed": 1234},
    "csr_block": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_block (size => 300000)"],
                  "calcs": RAND_CALCS, "function": "rand_csr_by_group", "iterations": 5, "seed": 7},
    "nochange": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_self_only()"],
                 "calcs": ["calc_richness", "calc_pe"], "function": "rand_nochange", "iterations": 3, "seed": 1},
    "structured": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_circle (radius => 200000)"],
                   "calcs": RAND_CALCS, "function": "rand_structured", "iterations": 5, "seed": 42},
    "structured_counts": {"basedata": "site_counts", "conditions": ["sp_self_only()"],
                          "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
                          "function": "rand_structured", "iterations": 5, "seed": 3},
    "structured_half": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                        "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_structured",
                        "iterations": 4, "seed": 9, "args": {"richness_multiplier": 0.5}},
    "structured_more": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                        "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_structured",
                        "iterations": 4, "seed": 11, "args": {"richness_multiplier": 1.2, "richness_addition": 1}},
    "swaps": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
              "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
              "function": "rand_independent_swaps", "iterations": 3, "seed": 5},
    "swaps_all": {"basedata": "site_counts", "conditions": ["sp_self_only()"],
                  "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
                  "function": "rand_independent_swaps", "iterations": 2, "seed": 6,
                  "args": {"stop_on_all_swapped": 1}},
    "swaps_modified": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
                       "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
                       "function": "rand_independent_swaps_modified", "iterations": 3, "seed": 8},
    "curveball": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
                  "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
                  "function": "rand_curveball", "iterations": 3, "seed": 13},
    "curveball_counts": {"basedata": "site_counts", "conditions": ["sp_self_only()"],
                         "calcs": ["calc_richness", "calc_endemism_whole", "calc_rarity_whole"],
                         "function": "rand_curveball", "iterations": 3, "seed": 14,
                         "args": {"stop_on_all_swapped": 1}},
    "curveball_max": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                      "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_curveball",
                      "iterations": 2, "seed": 15, "args": {"use_max_swap": 1, "swap_count": 500}},
    # spatially structured (7d)
    "sp_default": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                   "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_spatially_structured",
                   "iterations": 2, "seed": 21},
    "sp_diffusion": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
                     "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_diffusion",
                     "iterations": 3, "seed": 22,
                     "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 150000)"}},
    "sp_random_walk": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                       "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_random_walk",
                       "iterations": 3, "seed": 23,
                       "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 150000)"}},
    "sp_walk_random_back": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                            "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_spatially_structured",
                            "iterations": 2, "seed": 24,
                            "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 150000)",
                                     "spatial_allocation_order": "random_walk",
                                     "label_allocation_backtracking": "random"}},
    "sp_walk_from_start": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                           "calcs": ["calc_richness", "calc_endemism_whole"],
                           "function": "rand_spatially_structured", "iterations": 2, "seed": 25,
                           "args": {"spatial_conditions_for_label_allocation":
                                        ["sp_circle (radius => 100000)", "sp_circle (radius => 250000)"],
                                    "spatial_allocation_order": "random_walk",
                                    "label_allocation_backtracking": "from_start"}},
    "sp_proximity": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                     "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_spatially_structured",
                     "iterations": 3, "seed": 26,
                     "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 250000)",
                              "spatial_allocation_order": "proximity"}},
    "sp_random_reseed": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                         "calcs": ["calc_richness", "calc_endemism_whole"],
                         "function": "rand_spatially_structured", "iterations": 2, "seed": 27,
                         "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 200000)",
                                  "spatial_allocation_order": "random", "spatial_allocation_reseed_prob": 0.2}},
    "subset_block": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
                     "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_structured",
                     "iterations": 3, "seed": 31, "args": {"spatial_conditions_for_subset": "sp_block (size => 500000)"}},
    "subset_defq": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                    "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_structured",
                    "iterations": 3, "seed": 32, "args": {"definition_query": "$y > 1000000"}},
    "subset_curveball": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                         "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_curveball",
                         "iterations": 2, "seed": 33,
                         "args": {"spatial_conditions_for_subset": "sp_block (size => 400000)"}},
    "subset_spatial": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                       "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_spatially_structured",
                       "iterations": 2, "seed": 34,
                       "args": {"spatial_conditions_for_subset": "sp_block (size => 600000)",
                                "spatial_conditions_for_label_allocation": "sp_circle (radius => 150000)"}},
    "constant_labels": {"basedata": "site_100k", "conditions": ["sp_circle (radius => 200000)"],
                        "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_structured",
                        "iterations": 3, "seed": 35,
                        "args": {"labels_not_to_randomise": ["Genus:sp1", "Genus:sp5", "Genus:sp11"]}},
    "gp_by_set": {"basedata": "site_100k", "group_props": True, "conditions": ["sp_circle (radius => 200000)"],
                  "calcs": ["calc_richness", "calc_gpprop_stats"], "function": "rand_structured",
                  "iterations": 3, "seed": 41, "args": {"randomise_group_props_by": "by_set"}},
    "gp_by_item": {"basedata": "site_100k", "group_props": True, "conditions": ["sp_circle (radius => 200000)"],
                   "calcs": ["calc_richness", "calc_gpprop_stats"], "function": "rand_structured",
                   "iterations": 3, "seed": 42, "args": {"randomise_group_props_by": "by_item"}},
    "gp_subset_defq": {"basedata": "site_100k", "group_props": True, "conditions": ["sp_self_only()"],
                       "calcs": ["calc_richness", "calc_gpprop_stats"], "function": "rand_structured",
                       "iterations": 3, "seed": 43,
                       "args": {"randomise_group_props_by": "by_set", "definition_query": "$y > 1000000",
                                "spatial_conditions_for_subset": "sp_block (size => 500000)"}},
    "tree_shuffle": {"basedata": "site_100k", "tree": "Example_tree.nex", "conditions": ["sp_circle (radius => 200000)"],
                     "calcs": ["calc_pd", "calc_pe", "calc_phylo_rpe2"], "function": "rand_structured",
                     "iterations": 3, "seed": 44, "args": {"randomise_trees_by": "shuffle_terminal_names"}},
    "sp_seeded": {"basedata": "site_100k", "conditions": ["sp_self_only()"],
                  "calcs": ["calc_richness", "calc_endemism_whole"], "function": "rand_spatially_structured",
                  "iterations": 3, "seed": 28,
                  "args": {"spatial_conditions_for_label_allocation": "sp_circle (radius => 150000)",
                           "spatial_conditions_for_seed_location": "$x > 2500000"}},
}


def phylocom_calcs(case: dict) -> list[str]:
    return PHYLOCOM_CALCS + (PHYLOCOM_VPD if case.get("vpd") else [])


def phylo_calcs(case: dict) -> list[str]:
    calcs = list(PHYLO_CALCS) if case["calcs"] == "all" else [c for c in PHYLO_CALCS if c not in PHYLO_TWO_SET]
    return calcs + list(case.get("extra", []))


def _newick(t, length, drop=()) -> str:
    """Newick of a Tree with other branch lengths, dropping (collapsing) some internal nodes."""
    import re

    def fmt(name: str) -> str:
        if name.endswith("___"):
            return ""
        return "'" + name + "'" if re.search(r"[\s(),:;'\[\]]", name) else name

    def items(i: int, extra: float) -> list[str]:
        if i in drop:
            out = []
            for c in t.children[i]:
                out.extend(items(c, extra + length[i]))
            return out
        s = ""
        if t.children[i]:
            s = "(" + ",".join(x for c in t.children[i] for x in items(c, 0.0)) + ")"
        return [s + fmt(t.names[i]) + ":" + repr(float(length[i] + extra))]

    return items(t.root, 0.0)[0] + ";"


def _write_tree_variants(t) -> None:
    """An ultrametric tree with polytomies and a tree that is not ultrametric."""
    import numpy as np

    length = np.asarray(t.length, dtype=float)
    internal = [i for i in range(len(t.names)) if t.children[i] and i != t.root]
    # collapse every third internal node (by name) into its parent
    by_name = sorted(internal, key=lambda i: t.names[i])
    drop = set(by_name[1::3])
    (DATA / "tree_polytomy.nex").write_text(
        "#NEXUS\nbegin trees;\n\ttree polytomy = [&R] " + _newick(t, length, drop) + "\nend;\n", encoding="utf-8")
    nu = length.copy()
    tips = sorted((i for i in range(len(t.names)) if not t.children[i]), key=lambda i: t.names[i])
    for k, i in enumerate(tips):
        nu[i] = nu[i] * (1 + 0.15 * (k % 4))
    (DATA / "tree_nu.nex").write_text(
        "#NEXUS\nbegin trees;\n\ttree nu = [&R] " + _newick(t, nu) + "\nend;\n", encoding="utf-8")


def write_trees() -> None:
    """Variants of the example tree: pruning needed, and species-only names."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import landschaft as ls

    t = ls.bio.read_tree(DATA / "Example_tree.nex")
    # species-only names, for the site data read with the species column as label
    nwk = t.to_newick()
    for name in t.names:
        if name.startswith("Genus:"):
            nwk = nwk.replace(f"'{name}'", name.split(":", 1)[1]).replace(name, name.split(":", 1)[1])
    (DATA / "tree_species.nex").write_text("#NEXUS\nbegin trees;\n\ttree species = [&R] " + nwk + "\nend;\n",
                                           encoding="utf-8")
    # a tree needing a trim: one label missing, extra tips and a named clade without labels
    nwk = t.to_newick().rstrip(";")
    nwk = nwk.replace("'Genus:sp1':", "'Other:tip0':")
    nwk = f"({nwk},(('Other:tip1':0.3,'Other:tip2':0.2)other_clade:0.4,'Other:tip3':0.1):0.25);"
    _write_tree_variants(t)
    (DATA / "tree_trim.nex").write_text("#NEXUS\nbegin trees;\n\ttree trim = [&R] " + nwk + "\nend;\n",
                                        encoding="utf-8")


TREE_FILES = ["Example_tree.nex", "Example_tree_with_colours.nex", "tree_with_nonmatching_names.nex"]

NEWICK_CASES = {
    "simple": "((a:1,b:2):1,c:3);",
    "unnamed": "(((a,b),c),d);",
    "internal_names": "((a:0.1,b:0.2)ab:0.05,(c:0.3,d:0.4)cd:0.06)root:0;",
    "duplicates": "((a:1,a:2)x:1,(b:1,x:1):2);",
    "quoted": "(('a b':1,'c''d':2):1,'e:f':3);",
    "forest": "(a:1,b:2),(c:1,d:1)",
    "all_zero": "((a:0,b:0):0,c:0);",
    "exponent": "((a:1e-3,b:2.5E2):-0.5,c:.25);",
    "boot": "((a:1,b:2)[90]:1,c:3)[x];",
    "phylip_two": "(a:1,b:2);((a:1,b:1):1,c:1);",
}

MATRIX_CASES = {
    "full": {"file": "Example_matrix.txt", "params": {"ELEMENT_COLUMNS": [1, 2]}, "sparse": 0,
             "landschaft": {"element_columns": [1, 2]}},
    "sparse": {"file": "Example_matrix_sparse.txt", "params": {}, "sparse": 1,
               "args": {"label_row_columns": [0], "label_col_columns": [1], "value_column": 2},
               "landschaft": {"sparse": True, "row_columns": [0], "col_columns": [1], "value_column": 2}},
}

RASTER_CASES = {
    "bands_single": {"files": ["raster_single.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
    "bands_multi": {"files": ["raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
    "bands_coarse": {"files": ["raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [30, 20], "cell_origins": [5, 3]},
    "values": {"files": ["raster_values.tif"], "labels_as_bands": 0, "cell_sizes": [10, 10]},
    "values_coarse": {"files": ["raster_values.tif"], "labels_as_bands": 0, "cell_sizes": [25, 25]},
    "two_files": {"files": ["raster_single.tif", "raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
}


def write_synthetic_data() -> None:
    """Deterministic synthetic inputs (written once, committed)."""
    rng = np.random.default_rng(20260928)
    lines = ["label,x,y,count"]
    for i in range(400):
        lab = f"sp{rng.integers(0, 25)}"
        # coordinates near cell edges of 0.1 and 0.25 grids, both signs
        x = round(float(rng.integers(-30, 30)) * 0.05 + float(rng.choice([0.0, 1e-12, -1e-12, 0.0249999, 0.03])), 13)
        y = round(float(rng.integers(-30, 30)) * 0.05 + float(rng.choice([0.0, 0.02, -0.02, 0.07])), 13)
        c = rng.choice(["1", "2", "0.5", "2.25", "0", "3", "", "1"])
        lines.append(f"{lab},{x!r},{y!r},{c}")
    (DATA / "synthetic_decimal.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = ["genus,species,region,y,skip"]
    genera = ["Acacia", "Eucalyptus sp.", "Genus:odd", "Quote'd"]
    regions = ["north", "south", "east west", "region10", "region9", "Region2"]
    for i in range(200):
        g = genera[rng.integers(0, len(genera))]
        s = f"sp {rng.integers(0, 6)}" if i % 3 else f"sp{rng.integers(0, 6)}"
        r = regions[rng.integers(0, len(regions))]
        y = float(rng.integers(0, 40)) * 0.75
        skip = "1" if i % 17 == 0 else ("0" if i % 5 else "")
        gq = '"' + g + '"' if ("," in g or '"' in g or " " in g) else g
        rows.append(f"{gq},{s},{r},{y!r},{skip}")
    (DATA / "synthetic_quoted.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_points() -> None:
    """Point shapefile of the Biodiverse site data."""
    import geopandas as gpd
    import pandas as pd

    df = pd.read_csv(DATA / SITE, dtype={"x": float, "y": float})
    gdf = gpd.GeoDataFrame(df[["num", "genus", "species"]], geometry=gpd.points_from_xy(df["x"], df["y"]))
    gdf.to_file(DATA / "site_points.shp")


def write_site_polygons() -> None:
    """Polygons over the site data (overlapping, with an attribute)."""
    import geopandas as gpd
    from shapely.geometry import Polygon, box

    geoms = [
        box(3000000, 1000000, 3600000, 2500000),
        box(3400000, 2000000, 4200000, 3200000),
        Polygon([(2000000, 1000000), (2600000, 1200000), (2400000, 2000000), (2000000, 1000000)]),
        box(3300000, 900000, 3500000, 1100000),
    ]
    gdf = gpd.GeoDataFrame({"zone": [1, 2, 2, 3]}, geometry=geoms)
    gdf.to_file(DATA / "site_polygons.shp")


def write_rasters() -> None:
    """Small GeoTIFFs for the raster import."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    tf = from_origin(100.0, 250.0, 10.0, 10.0)
    z = rng.integers(0, 4, size=(12, 9)).astype(np.float32)
    z[0, 0] = -9999
    with rasterio.open(DATA / "raster_single.tif", "w", driver="GTiff", width=9, height=12, count=1,
                       dtype="float32", transform=tf, nodata=-9999) as dst:
        dst.write(z, 1)
    zz = rng.integers(0, 3, size=(3, 12, 9)).astype(np.int16)
    zz[1, 2, 3] = -1
    with rasterio.open(DATA / "raster_multi.tif", "w", driver="GTiff", width=9, height=12, count=3,
                       dtype="int16", transform=tf, nodata=-1) as dst:
        dst.write(zz)
        dst.set_band_description(1, "sp_a")
        dst.set_band_description(2, "sp_b")
    v = rng.choice([1, 2, 5, 7], size=(12, 9)).astype(np.int32)
    v[3, 3] = 0
    with rasterio.open(DATA / "raster_values.tif", "w", driver="GTiff", width=9, height=12, count=1,
                       dtype="int32", transform=tf, nodata=0) as dst:
        dst.write(v, 1)
