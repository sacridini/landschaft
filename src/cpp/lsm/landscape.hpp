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

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
    int n_threads = 0;              // OpenMP threads of the Python calls (0: all)
    // FRAGSTATS
    bool fragstats = false;          // convention = "fragstats": physical edge lengths, dcore sizes
    double search_radius = kNaN;     // prox, simi (map units)
    double connect_threshold = kNaN; // connect (map units)
    double boundary_contrast = 0.0;  // econ, cwed, teci: weight of edges to NA and the map edge
    std::vector<std::array<double, 3>> contrast;    // (class a, class b, weight): edge contrast
    std::vector<std::array<double, 3>> similarity;  // (class a, class b, weight): simi
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
    // every disjunct core area: cells and class index, sorted by class
    // (in-memory landscapes only)
    std::vector<i64> area_cells;
    std::vector<std::int32_t> area_class;
    bool has_area_cells = false;
};

// Pairs of patches a < b within some distance, and their distance.
struct PatchPairs {
    std::vector<std::int32_t> a, b;
    std::vector<double> dist;
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

// Height of the strips labelled in parallel (and of the blocks of an
// out-of-core landscape, which are multiples of it).
constexpr i64 kStripRows = 64;

// Patches of a raster of dense class indices (-1 = NA), labelled in strips
// of kStripRows rows, in landscapemetrics' order.
Patches label_patches(const std::int32_t* cls, i64 nx, i64 ny, int directions);

// City-block distance of every valid cell to the nearest cell of another
// class or NA, and to the edges of the array, which count as `top`, `bottom`
// and `sides` (the distance just outside them). Capped at a large value.
std::vector<std::int32_t> city_block_distance(const std::int32_t* cls, i64 nx, i64 ny, std::int32_t top,
                                              std::int32_t bottom, std::int32_t sides);

// R's signif(x, digits) for ordinary magnitudes.
double r_signif(double x, int digits);

// R's sum, mean, sd, median of x, and log and sqrt of x[0] (for the tests).
std::vector<double> r_statistics(const std::vector<double>& x);

// Slope of R's lm(y ~ x) over n points, computed as R does (LINPACK dqrls
// and R's reference BLAS); NaN where R gives NA (rlm.cpp).
double r_lm_slope(const double* x, const double* y, int n);

// FRAGSTATS' am (weights w), md or ra of x[0..n) (metrics.cpp).
double reduce_statistic(const std::string& stat, const double* x, i64 n, const double* w);

enum Level : std::int8_t { kPatch = 0, kClass = 1, kLandscape = 2 };

// Smallest circle around a patch or a class: centre in map units from the
// top-left corner of the raster (y downwards), and radius.
struct EnclosingCircle {
    double cx, cy, r;
};

// The cells (pixel indices) of every patch in raster order: patch i has
// cell[offset[i] .. offset[i + 1]).
struct PatchCells {
    std::vector<i64> offset;
    std::vector<i64> cell;
};

// get_centroids: one row per patch (several with cell_center ties).
struct CentroidTable {
    std::vector<std::int32_t> patch;
    std::vector<double> x, y;
};

// get_nearestneighbour: distance per patch (NaN when the class has one
// patch), and with return_id one row per neighbour patch reached (-1: none).
struct NeighbourTable {
    std::vector<std::int32_t> patch;
    std::vector<double> dist;
    std::vector<std::int32_t> neighbour;
};

struct MetricSpec {
    Level level;
    std::string name;
};

class Landscape {
public:
    Landscape(const std::int32_t* values, const std::uint8_t* na_mask, i64 nx, i64 ny, double resx, double resy,
              Options options);
    // A landscape given by its tables (out-of-core): the metrics that need
    // the raster itself (core areas, gyrate, circle, enn) must be supplied
    // with set_core / set_patch_values before they are asked for.
    Landscape(i64 nx, i64 ny, double resx, double resy, Options options, std::vector<std::int32_t> class_values,
              std::vector<i64> class_cells, Patches patches, Adjacency adjacency);

    bool has_raster() const { return !tables_only_; }
    void set_core(Core core) { core_ = std::move(core); }
    void set_patch_values(const std::string& name, std::vector<double> values);

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
    // Compute the summary statistics (_mn, _sd, ...) of these metrics in
    // parallel ahead of time; optional, it only saves time.
    void prepare(const std::vector<MetricSpec>& specs);
    std::vector<double> class_metric(const std::string& name);
    double landscape_metric(const std::string& name);

    // landscapemetrics' utilities (utils.cpp, geometry.cpp)
    PatchCells patch_cells();
    CentroidTable centroids(double xmin, double xmax, double ymin, double ymax, bool cell_center);
    NeighbourTable nearest_neighbours(bool return_id);
    const std::vector<EnclosingCircle>& patch_circles();
    std::vector<EnclosingCircle> class_circles();
    std::vector<std::int8_t> patch_boundaries(bool consider_boundary, int edge_depth);
    std::vector<i64> cooccurrence(const std::vector<std::pair<int, int>>& offsets);

private:
    // metrics.cpp
    const std::vector<double>& patch_base(const std::string& name);
    std::vector<double> gyrate();
    std::vector<double> class_metric_impl(const std::string& name);
    double landscape_metric_impl(const std::string& name);
    std::vector<double> cohesion_terms();
    std::vector<double> stat_values(const std::string& base, const std::string& stat, bool by_class);
    int stat_slot(const std::string& base, const std::string& stat, bool by_class);
    const std::vector<i64>& class_start();
    double cohesion_factor() const;
    // utils.cpp
    void order_ties_as_r(std::vector<std::vector<i64>>& nearest);
    // fragstats.cpp
    std::vector<double> class_weights(const std::vector<std::array<double, 3>>& table, double diagonal) const;
    std::pair<double, double> side_lengths() const;
    std::vector<double> econ();
    const PatchPairs& patch_pairs(double radius, bool same_class);
    std::vector<double> proximity(bool similarity);
    std::vector<double> connectance(bool by_class);
    std::vector<double> contrast_edges(bool teci, bool by_class);
    std::vector<double> dcore_sizes(const std::string& stat, bool by_class);
    // geometry.cpp
    std::vector<double> enn();
    std::vector<double> circle();
    std::vector<EnclosingCircle> hull_circles();
    double total_area();
    double edge_length(i64 h, i64 v) const;
    bool res_all_equal() const;
    std::vector<double> class_te();
    double landscape_te();
    std::vector<double> pafrac_by(const std::vector<i64>& start, bool hectares);

    i64 nx_ = 0;
    i64 ny_ = 0;
    double resx_ = 1.0;
    double resy_ = 1.0;
    Options options_;
    std::vector<std::int32_t> cls_; // class index per pixel, -1 = NA
    std::vector<std::int32_t> class_values_;
    std::vector<i64> class_cells_;
    i64 valid_cells_ = 0;

    bool tables_only_ = false;
    std::optional<Patches> patches_;
    std::optional<Adjacency> adjacency_;
    std::optional<Core> core_;
    std::map<std::string, std::vector<double>> patch_cache_; // node-based: references stay valid
    // summary statistics (stat_values): per patch metric, statistic and level
    static constexpr int kStatBases = 16;
    static constexpr int kStatKinds = 6;
    std::array<std::vector<double>, kStatBases * kStatKinds * 2> stat_cache_;
    std::array<bool, kStatBases * kStatKinds * 2> stat_done_{};
    std::optional<double> total_area_;
    std::map<std::string, std::vector<double>> class_cache_;
    std::map<std::string, double> landscape_cache_;
    std::vector<i64> class_start_;
    std::optional<std::vector<EnclosingCircle>> patch_circles_;
    std::map<std::string, PatchPairs> pair_cache_;
};

// landscapemetrics' get_boundaries on the values of a raster (see utils.cpp).
std::vector<std::int32_t> raster_boundaries(const std::int32_t* values, const std::uint8_t* na_mask, i64 nx, i64 ny,
                                            bool consider_boundary, int edge_depth, bool patch_id, std::int32_t na);

// Mask (ny x nx, row-major) of polygon rings (map coordinates) burnt as
// GDAL's rasterizer does with ALL_TOUCHED=TRUE; gt: GDAL geotransform
// (rasterize.cpp).
std::vector<std::uint8_t> rasterize_all_touched(const std::vector<std::vector<std::pair<double, double>>>& rings,
                                                const double* gt, int nx, int ny);

// R's cor(x, method = "pearson" | "kendall" | "spearman") of the columns of
// the n x m column-major matrix x; returns m x m.
std::vector<double> r_cor(const std::vector<double>& x, i64 n, i64 m, const std::string& method);

// -- many metrics, many rasters (table.cpp) --------------------------------

// Long table as landscapemetrics' calculate_lsm: one row per value, ordered
// by item, then by metric (in the order requested), class and patch id.
struct LongTable {
    std::vector<std::int32_t> item;
    std::vector<std::int16_t> metric;  // index of the MetricSpec
    std::vector<std::int32_t> cls;     // class value
    std::vector<std::uint8_t> cls_na;
    std::vector<std::int32_t> id;      // patch id (1-based)
    std::vector<std::uint8_t> id_na;
    std::vector<double> value;

    std::size_t size() const { return value.size(); }
    void append(const LongTable& other);
};

// Rows of the metrics `specs` of one landscape, as item `item`. As in
// landscapemetrics, an all-NA landscape gives one NA row per metric, and a
// class-level enn_* whose values are all NA gives one row without class.
void append_long_table(Landscape& landscape, const std::vector<MetricSpec>& specs, std::int32_t item,
                       LongTable& out);

// A raster given to the batch functions (not owned).
struct RasterView {
    const std::int32_t* values;
    const std::uint8_t* na_mask;  // may be null
    i64 nx, ny;
    double resx, resy;
};

// Long table of several rasters (items 0..n-1): small rasters side by side
// in parallel, large ones one after the other (each in parallel).
LongTable batch_long_table(const std::vector<RasterView>& rasters, const Options& options,
                           const std::vector<MetricSpec>& specs);

// ENN (Euclidean nearest-neighbour distance) of every patch from the m edge
// cells of the classes with several patches: Morton code (x, y) and patch
// per cell, in any order (the arrays are taken over). patch_cls: class
// index of every patch.
std::vector<double> enn_from_edge_cells(std::unique_ptr<std::uint64_t[]> key, std::unique_ptr<std::int32_t[]> patch,
                                        i64 m, const std::vector<std::int32_t>& patch_cls, int n_classes, i64 nx,
                                        i64 ny, double resx, double resy, int directions);
std::uint64_t morton_code(std::uint32_t x, std::uint32_t y);

// 1 - area / area of the smallest circle around the points (x0, y0, x1, y1,
// ...), found with Welzl's algorithm shuffled by `seed` (the patch index).
double related_circle(const std::vector<double>& xy, std::uint64_t seed, double area_ha);

// Smallest circle around the given cell corners (grid units: x to the right,
// y down), scaled by (rx, ry): Welzl's algorithm (seeded) on their convex
// hull, which is exact on the grid. `corners` is sorted in place.
EnclosingCircle grid_circle(std::vector<std::pair<i64, i64>>& corners, double rx, double ry, std::uint64_t seed);

// Landscape-level metrics in a moving window (window: wy rows x wx columns,
// odd sizes, non-zero = part of the window) centred on every cell; cells
// outside the raster are NA. Returns names.size() x ny x nx values.
std::vector<double> window_metrics(const RasterView& raster, const std::uint8_t* window, i64 wx, i64 wy,
                                   const std::vector<std::string>& names, const Options& options);

// -- out-of-core landscapes (tiled.cpp) ----------------------------------------
//
// A raster read by blocks of rows (each a multiple of kStripRows rows, but
// the last), given with `above` / `below` rows of context. The first pass
// labels every block, joins the patch pieces cut by the block seams and
// counts the class adjacencies; finish() then gives a Landscape of tables
// with the same patches, in the same order, as the in-memory one. Core
// areas, gyrate, circle and enn need a detail pass over the same blocks
// (and gyrate with cell_center a pass for the ties); their results are
// identical to the in-memory computation.
class TiledLandscape {
public:
    TiledLandscape(i64 nx, i64 ny, double resx, double resy, Options options, std::vector<std::int32_t> class_values);
    ~TiledLandscape();

    i64 nx() const;
    i64 ny() const;
    int n_threads() const;

    // pass 1
    void add_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows, i64 above, i64 below);
    void finish();
    Landscape& landscape();

    // detail pass: rows of context the blocks need, then the blocks again
    i64 detail_halo(bool core, bool enn) const;
    void begin_detail(bool core, bool gyrate, bool circle, bool enn);
    void detail_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows, i64 above,
                      i64 below);
    // true when gyrate with cell_center needs the tie pass
    bool end_detail();
    void tie_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows, i64 above, i64 below);
    void end_ties();

private:
    struct State;
    std::unique_ptr<State> s_;
};

}  // namespace lsm
