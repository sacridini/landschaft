// SPDX-License-Identifier: GPL-3.0-or-later
//
// Landscape metrics (FRAGSTATS) for categorical rasters, matching the R
// package landscapemetrics.
//
// The raster is read once into dense class indices. Everything else is built
// lazily and cached: the patches of all classes are labelled in a single
// union-find pass that also accumulates the per-patch statistics (cells,
// edges, adjacencies, coordinates), and one more pass counts the 4-neighbour
// (and, when needed, diagonal) adjacencies between classes. Every metric is a
// cheap function of these tables.
#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace lsm {

using i64 = std::int64_t;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Options {
    int directions = 8;             // patch connectivity: 4 (rook) or 8 (queen)
    bool count_boundary = false;    // landscape boundary (and NA) counts as edge
    bool consider_boundary = false; // cells touching only the boundary can be core
    int edge_depth = 1;             // core: distance (cells) from the patch edge
    bool cell_center = false;       // gyrate: centroid forced to a patch cell
    double classes_max = kNaN;      // rpr: maximum number of classes (NaN = unknown)
    int neighbourhood = 4;          // complexity metrics: 4 or 8 neighbours
    bool ordered = true;            // complexity metrics: ordered pairs
    std::string base = "log2";      // complexity metrics: log2, log or log10
};

// Per-patch statistics, in landscapemetrics' patch order (by class, then by
// the first cell in column-major order).
//
// Patches are found strip by strip (kStripRows rows each, in parallel): every
// pixel keeps the id of its patch piece within the strip ("local" id), and
// local_to_patch joins the pieces cut by strip seams.
struct Patches {
    i64 n = 0;
    std::vector<std::int32_t> cls;   // class index
    std::vector<i64> cells;
    std::vector<i64> edge_h;         // cell sides towards left/right non-patch cells (incl. boundary)
    std::vector<i64> edge_v;         // cell sides towards top/bottom non-patch cells (incl. boundary)
    std::vector<i64> adj_orth;       // same-patch orthogonal cell pairs
    std::vector<i64> adj_diag;       // same-patch diagonal cell pairs
    std::vector<i64> sum_x;          // sum of column indices
    std::vector<i64> sum_y;          // sum of row indices
    std::vector<i64> first_cm;       // first cell in column-major order (x * ny + y)

    i64 strip_rows = 0;
    std::vector<i64> strip_offset;        // first local id of each strip (+ total at the end)
    std::vector<std::int32_t> code;       // per pixel: -1 = NA, else -(local index in its strip) - 2
    std::vector<std::int32_t> local_to_patch;

    // Local id of a (valid) pixel in row y.
    i64 local(i64 p, i64 y) const { return strip_offset[y / strip_rows] + (-static_cast<i64>(code[p]) - 2); }
    // Patch index of pixel p in row y, -1 for NA.
    std::int32_t patch(i64 p, i64 y) const { return code[p] < 0 && code[p] != -1 ? local_to_patch[local(p, y)] : -1; }
};

// Core areas: cells farther than edge_depth (city-block distance) from any
// cell of another class or NA (and from the map edge unless consider_boundary).
struct Core {
    std::vector<i64> cells;   // core cells per patch
    std::vector<i64> areas;   // disjunct core areas per patch (connectivity = directions)
};

// Directed cell-pair counts: m[a * dim + b] = number of (cell of class a,
// neighbour b) pairs. Index K is NA, K + 1 is outside the landscape.
struct Adjacency {
    int dim = 0;
    std::vector<i64> h;    // left/right neighbours
    std::vector<i64> v;    // top/bottom neighbours
    std::vector<i64> diag; // diagonal neighbours (real classes only, K x K), if requested

    i64 orth(int a, int b) const { return h[a * dim + b] + v[a * dim + b]; }
};

class Landscape {
public:
    Landscape(const std::int32_t* values, const std::uint8_t* na_mask, i64 nx, i64 ny, double resx, double resy,
              Options options);

    i64 nx() const { return nx_; }
    i64 ny() const { return ny_; }
    int n_classes() const { return static_cast<int>(class_values_.size()); }
    const std::vector<std::int32_t>& class_values() const { return class_values_; }
    const std::vector<i64>& class_cells() const { return class_cells_; }
    i64 valid_cells() const { return valid_cells_; }
    const Options& options() const { return options_; }

    const Patches& patches();
    const Adjacency& adjacency();
    const Core& core();
    // Patch id (1-based, landscapemetrics' id) of every pixel, 0 = NA.
    std::vector<std::int32_t> labels();

    // Metric values. Patch: one per patch; class: one per class; landscape:
    // a single value. Unknown names throw std::invalid_argument.
    std::vector<double> patch_metric(const std::string& name);
    std::vector<double> class_metric(const std::string& name);
    double landscape_metric(const std::string& name);

private:
    // metrics.cpp
    const std::vector<double>& patch_base(const std::string& name);
    std::vector<double> gyrate();
    // geometry.cpp
    std::vector<double> enn();
    std::vector<double> circle();
    double total_area();
    double edge_length(i64 h, i64 v) const;
    std::vector<double> class_te();
    double landscape_te();
    std::vector<double> pafrac_by(const std::vector<std::int32_t>& groups, int n_groups);

    i64 nx_ = 0;
    i64 ny_ = 0;
    double resx_ = 1.0;
    double resy_ = 1.0;
    Options options_;
    std::vector<std::int32_t> cls_; // class index per pixel, -1 = NA
    std::vector<std::int32_t> class_values_;
    std::vector<i64> class_cells_;
    i64 valid_cells_ = 0;

    std::optional<Patches> patches_;
    std::optional<Adjacency> adjacency_;
    std::optional<Core> core_;
    std::map<std::string, std::vector<double>> patch_cache_; // node-based: references stay valid
};

}  // namespace lsm
