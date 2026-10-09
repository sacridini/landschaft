// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the landschaft C++ core.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "gtb/accounting.hpp"
#include "gtb/change.hpp"
#include "gtb/contagion.hpp"
#include "gtb/distance.hpp"
#include "gtb/entropy.hpp"
#include "gtb/fad.hpp"
#include "gtb/fos.hpp"
#include "gtb/grayspatcon.hpp"
#include "gtb/influence.hpp"
#include "gtb/isochrone.hpp"
#include "gtb/gtn.hpp"
#include "gtb/morphology.hpp"
#include "gtb/mosaic.hpp"
#include "gtb/mspa.hpp"
#include "gtb/primitives.hpp"
#include "gtb/restoration.hpp"
#include "gtb/spa.hpp"
#include "circuit/python.hpp"
#include "nlm/python.hpp"
#include "cost/python.hpp"
#include "bio/python.hpp"
#include "surf/python.hpp"
#include "io/python.hpp"
#include "prio/python.hpp"
#include "sdm/python.hpp"
#include "sim/python.hpp"
#include "opt/python.hpp"
#include "rest/python.hpp"
#include "conn/cost.hpp"
#include "conn/graph.hpp"
#include "conn/nodes.hpp"
#include "lsm/landscape.hpp"
#include "lsm/rmath.hpp"

namespace py = pybind11;
using gtb::i64;

namespace {

template <class T>
using CArray = py::array_t<T, py::array::c_style>;

template <class T>
gtb::Image<T> to_image(const CArray<T>& a) {
    if (a.ndim() != 2) throw py::value_error("expected a 2-D array");
    gtb::Image<T> im(a.shape(1), a.shape(0));
    if (im.size() > 0) std::memcpy(im.data(), a.data(), sizeof(T) * static_cast<std::size_t>(im.size()));
    return im;
}

// Hand a vector (std::vector or gtb::Buffer) over to numpy without copying it.
template <class T, class A>
py::array_t<T> to_array(std::vector<T, A>&& v, i64 nx, i64 ny) {
    using V = std::vector<T, A>;
    auto* holder = new V(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<V*>(p); });
    return py::array_t<T>({ny, nx}, holder->data(), owner);
}

template <class T>
py::array_t<T> to_array(gtb::Image<T>&& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    return to_array(std::move(im.px), nx, ny);
}

// Runs `f` with the GIL released and, if n_threads > 0, with that many
// OpenMP threads for parallel regions started by this thread.
template <class F>
auto run_parallel(int n_threads, F&& f) {
    py::gil_scoped_release release;
#if defined(_OPENMP)
    const int previous = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(n_threads);
    struct Restore {
        int n;
        ~Restore() { omp_set_num_threads(n); }
    } restore{previous};
#else
    (void)n_threads;
#endif
    return f();
}

py::array_t<std::uint8_t> mspa(const CArray<std::uint8_t>& input, int connectivity, int edge_width, bool internal,
                               int n_threads) {
    const gtb::ImageU8 im = to_image(input);
    gtb::MspaParams params;
    params.connectivity = connectivity;
    params.edge_width = edge_width;
    params.internal = internal;
    params.threads = n_threads;
    gtb::ImageU8 out = run_parallel(n_threads, [&] { return gtb::mspa(im, params); });
    return to_array(std::move(out));
}

py::list extract_node_link_contacts(const CArray<std::int64_t>& nodes_arr, const CArray<std::int64_t>& links_arr) {
    if (nodes_arr.ndim() != 2 || links_arr.ndim() != 2) throw py::value_error("expected 2-D arrays");
    if (nodes_arr.shape(0) != links_arr.shape(0) || nodes_arr.shape(1) != links_arr.shape(1)) {
        throw py::value_error("shape mismatch between nodes and links");
    }
    const i64 nx = nodes_arr.shape(1);
    const i64 ny = nodes_arr.shape(0);
    gtb::ImageI64 nodes(nx, ny);
    gtb::ImageI64 links(nx, ny);
    std::memcpy(nodes.data(), nodes_arr.data(), sizeof(std::int64_t) * static_cast<std::size_t>(nodes.size()));
    std::memcpy(links.data(), links_arr.data(), sizeof(std::int64_t) * static_cast<std::size_t>(links.size()));

    auto contacts = gtb::extract_node_link_contacts(nodes, links);
    py::list res;
    for (const auto& p : contacts) {
        res.append(py::make_tuple(p.first, p.second));
    }
    return res;
}

py::array_t<std::uint8_t> compute_fad_density(
    const CArray<std::uint8_t>& input,
    int window_size,
    bool apply_gtb_mask,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::ImageU8 out = run_parallel(n_threads, [&] {
        return gtb::compute_fad_density(im, window_size, apply_gtb_mask);
    });
    return to_array(std::move(out));
}

py::dict compute_fad(const CArray<std::uint8_t>& input, int window_size, int n_threads) {
    const gtb::ImageU8 im = to_image(input);
    gtb::FADSingleResultCpp res = run_parallel(n_threads, [&] { return gtb::compute_fad(im, window_size); });
    py::dict out;
    out["density"] = to_array(std::move(res.density));
    out["raw"] = to_array(std::move(res.raw));
    out["classes"] = to_array(std::move(res.classes));
    out["foreground"] = res.foreground;
    out["class_counts"] = std::vector<std::int64_t>(std::begin(res.class_counts), std::end(res.class_counts));
    out["density_sum"] = res.density_sum;
    return out;
}

py::dict compute_fad_multiscale(
    const CArray<std::uint8_t>& input,
    const std::vector<int>& window_sizes,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::FADMultiscaleResultCpp res = run_parallel(n_threads, [&] {
        return gtb::compute_fad_multiscale(im, window_sizes);
    });

    py::dict out;
    out["window_sizes"] = res.window_sizes;

    py::list single_scale;
    for (auto& sm : res.single_scale_maps) {
        single_scale.append(to_array(std::move(sm)));
    }
    out["single_scale_maps"] = single_scale;

    py::list raw_dens;
    for (auto& rm : res.raw_densities) {
        raw_dens.append(to_array(std::move(rm)));
    }
    out["raw_densities"] = raw_dens;
    out["multiscale_map"] = to_array(std::move(res.multiscale_map));
    out["classes"] = to_array(std::move(res.multiscale_classes));

    py::list stats;
    for (const auto& row : res.stats) {
        py::dict d;
        d["scale"] = row.scale;
        d["intact_pct"] = row.intact_pct;
        d["interior_pct"] = row.interior_pct;
        d["dominant_pct"] = row.dominant_pct;
        d["transitional_pct"] = row.transitional_pct;
        d["patchy_pct"] = row.patchy_pct;
        d["rare_pct"] = row.rare_pct;
        d["average_fad"] = row.average_fad;
        stats.append(d);
    }
    out["stats"] = stats;

    return out;
}

py::dict compute_euclidean_distance(
    const CArray<std::uint8_t>& input,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::EuclideanDistanceResultCpp res = run_parallel(n_threads, [&] {
        return gtb::compute_euclidean_distance(im);
    });

    py::dict out;
    out["signed_distance"] = to_array(std::move(res.signed_distance), im.nx, im.ny);
    out["byte_map"] = to_array(std::move(res.byte_map));

    py::dict st;
    st["mean_fg_distance"] = res.stats.mean_fg_distance;
    st["max_fg_distance"] = res.stats.max_fg_distance;
    st["mean_bg_distance"] = res.stats.mean_bg_distance;
    st["max_bg_distance"] = res.stats.max_bg_distance;
    st["fg_pixel_count"] = res.stats.fg_pixel_count;
    st["bg_pixel_count"] = res.stats.bg_pixel_count;
    out["stats"] = st;

    return out;
}

py::dict compute_spatial_entropy(
    const CArray<std::uint8_t>& input,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::SpatialEntropyResultCpp res = run_parallel(n_threads, [&] {
        return gtb::compute_spatial_entropy(im);
    });

    py::dict out;
    out["entropy_index"] = res.entropy_index;
    py::list hist;
    for (std::uint64_t c : res.histogram) hist.append(c);
    out["histogram"] = hist;
    return out;
}

py::array_t<float> compute_spatial_entropy_map(
    const CArray<std::uint8_t>& input,
    int window_size,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::Image<float> out = run_parallel(n_threads, [&] {
        return gtb::compute_spatial_entropy_map(im, window_size);
    });
    return to_array(std::move(out));
}

py::array_t<std::uint8_t> compute_contagion_pff(
    const CArray<std::uint8_t>& input,
    int window_size,
    bool apply_gtb_mask,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::ImageU8 out = run_parallel(n_threads, [&] {
        return gtb::compute_contagion_pff(im, window_size, apply_gtb_mask);
    });
    return to_array(std::move(out));
}

py::array_t<float> compute_contagion_pff_float(
    const CArray<std::uint8_t>& input,
    int window_size,
    bool apply_gtb_mask,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::Image<float> out = run_parallel(n_threads, [&] {
        return gtb::compute_contagion_pff_float(im, window_size, apply_gtb_mask);
    });
    return to_array(std::move(out));
}

py::array_t<std::uint8_t> compute_clustering_fac(
    const CArray<std::uint8_t>& input,
    int window_size,
    bool apply_gtb_mask,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::ImageU8 out = run_parallel(n_threads, [&] {
        return gtb::compute_clustering_fac(im, window_size, apply_gtb_mask);
    });
    return to_array(std::move(out));
}

py::array_t<float> compute_clustering_fac_float(
    const CArray<std::uint8_t>& input,
    int window_size,
    bool apply_gtb_mask,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::Image<float> out = run_parallel(n_threads, [&] {
        return gtb::compute_clustering_fac_float(im, window_size, apply_gtb_mask);
    });
    return to_array(std::move(out));
}

py::array_t<std::uint8_t> compute_landscape_mosaic(
    const CArray<std::uint8_t>& input,
    int window_size,
    int output_classes,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::ImageU8 out = run_parallel(n_threads, [&] {
        return gtb::compute_landscape_mosaic(im, window_size, output_classes);
    });
    return to_array(std::move(out));
}

py::dict compute_landscape_mosaic_stats(
    const CArray<std::uint8_t>& im103
) {
    const gtb::ImageU8 im = to_image(im103);
    gtb::LandscapeMosaicStats stats = run_parallel(0, [&] { return gtb::compute_landscape_mosaic_stats(im); });  // releases the GIL
    py::dict out;
    out["class_codes"] = stats.class_codes;
    out["pixel_counts"] = stats.pixel_counts;
    out["percentages"] = stats.percentages;
    return out;
}

py::dict compute_restoration_status(const CArray<std::uint8_t>& input) {
    const gtb::ImageU8 im = to_image(input);
    gtb::RestorationStatusResult res = run_parallel(0, [&] { return gtb::compute_restoration_status(im); });  // releases the GIL
    py::dict d;
    d["data_area"] = res.data_area;
    d["nr_comp"] = res.nr_comp;
    d["eca_max"] = res.eca_max;
    d["larg_obj"] = res.larg_obj;
    d["aps"] = res.aps;
    d["cnoa"] = res.cnoa;
    d["eca"] = res.eca;
    d["doc"] = res.doc;
    d["rest_pot"] = res.rest_pot;
    d["rac"] = res.rac;
    return d;
}

py::array_t<float> compute_cost_distance(
    const CArray<std::uint8_t>& resistance,
    const CArray<std::uint8_t>& source_mask
) {
    const gtb::ImageU8 res = to_image(resistance);
    const gtb::ImageU8 src = to_image(source_mask);
    gtb::Image<float> out = run_parallel(0, [&] { return gtb::compute_cost_distance(res, src); });  // releases the GIL
    return to_array(std::move(out));
}

py::dict compute_least_cost_path(
    const CArray<std::uint8_t>& resistance,
    const CArray<std::uint8_t>& marker_a,
    const CArray<std::uint8_t>& marker_b,
    py::object dist_map_obj
) {
    const gtb::ImageU8 res = to_image(resistance);
    const gtb::ImageU8 ma = to_image(marker_a);
    const gtb::ImageU8 mb = to_image(marker_b);
    std::unique_ptr<gtb::Image<float>> dist_map_ptr;
    if (!dist_map_obj.is_none()) {
        auto d_arr = dist_map_obj.cast<CArray<float>>();
        dist_map_ptr = std::make_unique<gtb::Image<float>>(to_image(d_arr));
    }
    gtb::LeastCostPathResult lcp = run_parallel(0, [&] { return gtb::compute_least_cost_path(res, ma, mb, dist_map_ptr.get()); });  // releases the GIL
    py::dict d;
    d["combined_cost"] = to_array(std::move(lcp.combined_cost));
    d["lcp_mask"] = to_array(std::move(lcp.lcp_mask));
    d["path_length"] = lcp.path_length;
    d["rest_pixels"] = lcp.rest_pixels;
    d["expense"] = lcp.expense;
    d["avg_distance"] = lcp.avg_distance;
    return d;
}

py::tuple compute_parcellation(
    const CArray<std::uint8_t>& input,
    int connectivity,
    bool return_labels
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::ParcellationResult res = run_parallel(0, [&] { return gtb::compute_parcellation(im, connectivity, return_labels); });  // releases the GIL

    py::list class_stats_list;
    for (const auto& cs : res.class_stats) {
        py::dict cd;
        cd["class_id"] = cs.class_id;
        cd["count"] = cs.count;
        cd["area"] = cs.area;
        cd["aps"] = cs.aps;
        cd["awaps_rel"] = cs.awaps_rel;
        cd["awaps_data"] = cs.awaps_data;
        cd["division"] = cs.division;
        cd["parc"] = cs.parc;
        class_stats_list.append(cd);
    }

    py::dict gd;
    gd["total_parcels"] = res.global_stats.total_parcels;
    gd["data_area"] = res.global_stats.data_area;
    gd["aps"] = res.global_stats.aps;
    gd["awaps"] = res.global_stats.awaps;
    gd["division"] = res.global_stats.division;
    gd["parc"] = res.global_stats.parc;

    py::object labels_arr = py::none();
    if (return_labels) {
        labels_arr = to_array(std::move(res.parcel_labels));
    }

    return py::make_tuple(class_stats_list, gd, labels_arr);
}

py::dict compute_accounting(const CArray<std::uint8_t>& input, int connectivity, const std::vector<i64>& thresholds,
                            bool highlight_largest, int n_threads) {
    const gtb::ImageU8 im = to_image(input);
    gtb::AccountingResult res = run_parallel(n_threads, [&] {
        return gtb::compute_accounting(im, connectivity, thresholds, highlight_largest);
    });
    py::dict d;
    d["thresholds"] = py::cast(res.thresholds);
    d["object_area"] = py::array_t<i64>(static_cast<py::ssize_t>(res.object_area.size()), res.object_area.data());
    py::list largest;
    for (const auto& ids : res.largest) largest.append(py::cast(ids));
    d["largest"] = largest;
    d["classes"] = to_array(std::move(res.classes));
    d["ids"] = to_array(std::move(res.ids));
    d["area"] = to_array(std::move(res.area));
    return d;
}

py::array_t<std::uint32_t> compute_contortion(
    const CArray<std::uint8_t>& input,
    i64 min_area
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::Image<std::uint32_t> out = run_parallel(0, [&] { return gtb::compute_contortion(im, min_area); });  // releases the GIL
    return to_array(std::move(out));
}

py::dict compute_simple_change(
    const CArray<std::uint8_t>& im1,
    const CArray<std::uint8_t>& im2
) {
    const gtb::ImageU8 a = to_image(im1);
    const gtb::ImageU8 b = to_image(im2);
    gtb::SimpleChangeResult res = run_parallel(0, [&] { return gtb::compute_simple_change(a, b); });  // releases the GIL
    py::dict d;
    d["change_map"] = to_array(std::move(res.change_map));
    d["stable_bg"] = res.stable_bg;
    d["gain"] = res.gain;
    d["loss"] = res.loss;
    d["stable_fg"] = res.stable_fg;
    d["missing"] = res.missing;
    d["total_valid"] = res.total_valid;
    return d;
}

py::dict compute_morphological_change(
    const CArray<std::uint8_t>& im1,
    const CArray<std::uint8_t>& im2
) {
    const gtb::ImageU8 a = to_image(im1);
    const gtb::ImageU8 b = to_image(im2);
    gtb::MorphologicalChangeResult res = run_parallel(0, [&] { return gtb::compute_morphological_change(a, b); });  // releases the GIL
    py::dict d;
    d["change_map"] = to_array(std::move(res.change_map));
    d["fg_area_t1"] = res.fg_area_t1;
    d["fg_area_t2"] = res.fg_area_t2;
    d["net_change"] = res.net_change;
    d["stable_bg"] = res.stable_bg;
    d["stable_fg"] = res.stable_fg;
    d["mcd_gain"] = res.mcd_gain;
    d["mcd_loss"] = res.mcd_loss;
    d["spurious"] = res.spurious;
    d["missing"] = res.missing;
    d["elasticity"] = res.elasticity;
    return d;
}

py::dict compute_fos_change(
    const CArray<std::uint8_t>& fos1,
    const CArray<std::uint8_t>& fos2
) {
    const gtb::ImageU8 a = to_image(fos1);
    const gtb::ImageU8 b = to_image(fos2);
    gtb::FOSChangeResult res = run_parallel(0, [&] { return gtb::compute_fos_change(a, b); });  // releases the GIL
    py::dict d;
    d["change_map"] = to_array(std::move(res.change_map));
    d["fg_gain"] = res.fg_gain;
    d["fg_loss"] = res.fg_loss;
    d["stable_bg"] = res.stable_bg;
    d["forcom"] = res.forcom;
    d["diag_same_class"] = res.diag_same_class;
    d["diff_class"] = res.diff_class;
    d["frag_decrease_pct"] = res.frag_decrease_pct;
    d["frag_increase_pct"] = res.frag_increase_pct;
    d["matrix_6x6"] = res.matrix_6x6;
    return d;
}

py::dict compute_fad_change_counts(const CArray<std::uint8_t>& a, const CArray<std::uint8_t>& b, int n_threads) {
    const gtb::ImageU8 ia = to_image(a);
    const gtb::ImageU8 ib = to_image(b);
    const gtb::FADChangeCounts c = run_parallel(n_threads, [&] { return gtb::compute_fad_change_counts(ia, ib); });
    py::array_t<i64> m({7, 7});
    std::copy(c.matrix.begin(), c.matrix.end(), m.mutable_data());
    py::dict d;
    d["matrix"] = m;
    d["fg_area_a"] = c.fg_area_a;
    d["fg_area_b"] = c.fg_area_b;
    d["forcom"] = c.forcom;
    d["fg_sum_a"] = c.fg_sum_a;
    d["fg_sum_b"] = c.fg_sum_b;
    d["ru_area_a"] = c.ru_area_a;
    d["ru_area_b"] = c.ru_area_b;
    return d;
}

gtb::DistanceMetric parse_metric(const std::string& metric) {
    if (metric == "euclidean") return gtb::DistanceMetric::Euclidean;
    if (metric == "quasi-euclidean") return gtb::DistanceMetric::QuasiEuclidean;
    throw py::value_error("metric must be 'euclidean' or 'quasi-euclidean'");
}

py::dict compute_influence(const CArray<std::uint8_t>& input, i64 min_area, int core_buffer, int buffer, bool proximity,
                           const std::string& metric, int n_threads) {
    const gtb::ImageU8 im = to_image(input);
    const gtb::DistanceMetric m = parse_metric(metric);
    gtb::InfluenceResult r = run_parallel(n_threads, [&] {
        return gtb::compute_influence(im, min_area, core_buffer, buffer, proximity, m);
    });
    py::dict d;
    d["n_objects"] = r.n_objects;
    d["object_area"] = py::cast(r.object_area);
    d["classes"] = to_array(std::move(r.classes));
    d["ids"] = to_array(std::move(r.ids));
    d["zones"] = to_array(std::move(r.zones));
    if (proximity) {
        d["distance"] = to_array(std::move(r.distance));
        if (r.cag.size() > 0) d["cag"] = to_array(std::move(r.cag));
        py::list lines;
        for (const auto& l : r.lines) lines.append(py::make_tuple(l.x, l.y, l.nbr, l.clength, l.cag, l.ids, l.areas));
        d["lines"] = lines;
        d["prox_min"] = r.prox_min;
        d["prox_min_count"] = r.prox_min_count;
    }
    return d;
}

py::array_t<std::uint32_t> compute_geodesic_time(const CArray<std::uint8_t>& mask, const CArray<std::uint8_t>& ref) {
    const gtb::ImageU8 m = to_image(mask);
    const gtb::ImageU8 r = to_image(ref);
    gtb::Image<std::uint32_t> t;
    {
        py::gil_scoped_release release;
        t = gtb::geodesic_time(m, r);
    }
    return to_array(std::move(t));
}

py::dict compute_spa(const CArray<std::uint8_t>& input, int variant, int n_threads) {
    const gtb::ImageU8 im = to_image(input);
    gtb::SPAResult r = run_parallel(n_threads, [&] { return gtb::compute_spa(im, variant); });
    py::dict d;
    d["map"] = to_array(std::move(r.map));
    d["nr_fg"] = r.nr_fg;
    d["nr_cores"] = r.nr_cores;
    d["nr_margin"] = r.nr_margin;
    d["nr_holes"] = r.nr_holes;
    d["nr_edge"] = r.nr_edge;
    d["nr_perf"] = r.nr_perf;
    d["nr_islet"] = r.nr_islet;
    d["nr_contiguous"] = r.nr_contiguous;
    return d;
}

py::array_t<float> compute_gray_convolution(
    const CArray<std::uint8_t>& input,
    const std::string& metric,
    int window_size,
    bool mask_missing,
    int n_threads
) {
    const gtb::ImageU8 im = to_image(input);
    gtb::Image<float> out = run_parallel(n_threads, [&] {
        return gtb::compute_gray_convolution(im, metric, window_size, mask_missing, n_threads);
    });
    return to_array(std::move(out));
}

int max_threads() {
#if defined(_OPENMP)
    return omp_get_num_procs();
#else
    return 1;
#endif
}

// Threads of a parallel region opened now by the calling thread (tests).
int team_size() {
    int n = 1;
#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp single
        n = omp_get_num_threads();
    }
#endif
    return n;
}

// OpenMP thread count of the calling thread's next parallel regions (the
// Python side sets it before every compiled call; see landschaft/_threads.py).
void set_num_threads(int n) {
#if defined(_OPENMP)
    omp_set_num_threads(n > 0 ? n : 1);
#else
    (void)n;
#endif
}

void bind_detail(py::module_& m) {
    m.doc() = "Low-level morphological primitives (exposed for testing).";

    m.def("sqedt", [](const CArray<std::uint8_t>& a) {
        const auto im = to_image(a);
        return to_array(gtb::sqedt(im), im.nx, im.ny);
    });
    m.def("squared_threshold", &gtb::squared_threshold);
    m.def("dilate_disk", [](const CArray<std::uint8_t>& a, std::int64_t T) { return to_array(gtb::dilate_disk(to_image(a), T)); });
    m.def("ced", [](const CArray<std::uint8_t>& ref, const CArray<std::uint8_t>& mask) {
        const auto r = to_image(ref);
        return to_array(gtb::ced(r, to_image(mask)), r.nx, r.ny);
    });
    m.def("fill_holes", [](const CArray<std::uint8_t>& a, int graph) { return to_array(gtb::fill_holes(to_image(a), graph)); });
    m.def("reconstruct_dilation", [](const CArray<std::uint8_t>& marker, const CArray<std::uint8_t>& mask, int graph) {
        auto mk = to_image(marker);
        gtb::reconstruct_dilation(mk, to_image(mask), graph);
        return to_array(std::move(mk));
    });
    m.def("label_components", [](const CArray<std::uint32_t>& a, int graph) {
        auto im = to_image(a);
        gtb::label_components(im, graph);
        return to_array(std::move(im));
    });
    m.def("label_regions", [](const CArray<std::uint8_t>& a, std::uint8_t value, int graph, bool bottom_up) {
        const auto im = to_image(a);
        gtb::Image<std::int32_t> labels;
        std::vector<i64> areas;
        gtb::label_regions(im, value, graph, bottom_up, labels, areas);
        return py::make_tuple(to_array(std::move(labels)), py::cast(areas));
    });
    m.def("morph_distance", [](const CArray<std::uint8_t>& a, bool background, const std::string& metric) {
        const auto im = to_image(a);
        return to_array(gtb::morph_distance(im, background, parse_metric(metric)), im.nx, im.ny);
    });
    m.def("hist_equal", [](const CArray<float>& a) {
        if (a.ndim() != 2) throw py::value_error("expected a 2-D array");
        std::vector<float> v(a.data(), a.data() + a.size());
        return to_array(gtb::hist_equal(v, a.shape(1), a.shape(0)));
    });
    m.def("watershed", [](const CArray<std::uint8_t>& a, int graph) {
        std::int32_t nregions = 0;
        auto ws = gtb::watershed(to_image(a), graph, nregions);
        return py::make_tuple(to_array(std::move(ws)), nregions);
    });
    m.def("wsfah", [](const CArray<std::uint32_t>& labels, const CArray<std::uint8_t>& ref, int graph, int maxfl) {
        auto im = to_image(labels);
        gtb::wsfah(im, to_image(ref), graph, maxfl);
        return to_array(std::move(im));
    });
    m.def("thin_anchored", [](const CArray<std::uint8_t>& a, int stype, const CArray<std::uint8_t>& anchor) {
        auto im = to_image(a);
        auto an = to_image(anchor);
        gtb::thin_anchored(im, stype, an);
        return to_array(std::move(im));
    });
}

// ---- landscape metrics ---------------------------------------------------

lsm::Options make_options(const py::dict& o) {
    lsm::Options opt;
    opt.directions = o["directions"].cast<int>();
    opt.count_boundary = o["count_boundary"].cast<bool>();
    opt.consider_boundary = o["consider_boundary"].cast<bool>();
    opt.edge_depth = o["edge_depth"].cast<int>();
    opt.cell_center = o["cell_center"].cast<bool>();
    opt.classes_max = o["classes_max"].is_none() ? lsm::kNaN : o["classes_max"].cast<double>();
    opt.neighbourhood = o["neighbourhood"].cast<int>();
    opt.ordered = o["ordered"].cast<bool>();
    opt.base = o["base"].cast<std::string>();
    if (o.contains("n_threads") && !o["n_threads"].is_none()) opt.n_threads = o["n_threads"].cast<int>();
    auto number = [&](const char* key, double& target) {
        if (o.contains(key) && !o[key].is_none()) target = o[key].cast<double>();
    };
    auto table = [&](const char* key, std::vector<std::array<double, 3>>& target) {
        if (o.contains(key) && !o[key].is_none()) target = o[key].cast<std::vector<std::array<double, 3>>>();
    };
    if (o.contains("convention")) opt.fragstats = o["convention"].cast<std::string>() == "fragstats";
    number("search_radius", opt.search_radius);
    number("connect_threshold", opt.connect_threshold);
    number("boundary_contrast", opt.boundary_contrast);
    table("contrast", opt.contrast);
    table("similarity", opt.similarity);
    return opt;
}

// A raster and its NA mask (None or uint8 of the same shape); the arrays
// must outlive the view.
lsm::RasterView raster_view(const CArray<std::int32_t>& values, const py::object& na_mask,
                            CArray<std::uint8_t>& mask_arr, double resx, double resy) {
    if (values.ndim() != 2) throw py::value_error("expected a 2-D array");
    const std::uint8_t* mask = nullptr;
    if (!na_mask.is_none()) {
        mask_arr = na_mask.cast<CArray<std::uint8_t>>();
        if (mask_arr.ndim() != 2 || mask_arr.shape(0) != values.shape(0) || mask_arr.shape(1) != values.shape(1)) {
            throw py::value_error("mask shape does not match the raster");
        }
        mask = mask_arr.data();
    }
    return lsm::RasterView{values.data(), mask, values.shape(1), values.shape(0), resx, resy};
}

std::unique_ptr<lsm::Landscape> make_landscape(const CArray<std::int32_t>& values, const py::object& na_mask,
                                               double resx, double resy, const py::dict& options) {
    CArray<std::uint8_t> mask_arr;
    const lsm::RasterView r = raster_view(values, na_mask, mask_arr, resx, resy);
    lsm::Options opts = make_options(options);
    const int n_threads = opts.n_threads;
    try {
        return run_parallel(n_threads, [&] {
            return std::make_unique<lsm::Landscape>(r.values, r.na_mask, r.nx, r.ny, resx, resy, std::move(opts));
        });
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

std::vector<lsm::MetricSpec> metric_specs(const std::vector<int>& levels, const std::vector<std::string>& names) {
    if (levels.size() != names.size()) throw py::value_error("levels and names differ in length");
    std::vector<lsm::MetricSpec> specs;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        if (levels[i] < 0 || levels[i] > 2) throw py::value_error("level must be 0, 1 or 2");
        specs.push_back({static_cast<lsm::Level>(levels[i]), names[i]});
    }
    return specs;
}

template <class T>
py::array_t<T> vector_array(const std::vector<T>& v) {
    return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

py::dict long_table_dict(const lsm::LongTable& t) {
    py::dict d;
    d["item"] = vector_array(t.item);
    d["metric"] = vector_array(t.metric);
    d["class"] = vector_array(t.cls);
    d["class_na"] = vector_array(t.cls_na).attr("astype")("bool");
    d["id"] = vector_array(t.id);
    d["id_na"] = vector_array(t.id_na).attr("astype")("bool");
    d["value"] = vector_array(t.value);
    return d;
}

void tiled_check(const lsm::TiledLandscape& t, const CArray<std::int32_t>& values, const CArray<std::uint8_t>& na,
                 i64 rows) {
    if (values.ndim() != 2 || values.shape(1) != t.nx() || values.shape(0) != rows) {
        throw py::value_error("block shape does not match the raster width and the rows given");
    }
    if (na.ndim() != 2 || na.shape(0) != values.shape(0) || na.shape(1) != values.shape(1)) {
        throw py::value_error("mask shape does not match the block");
    }
}

template <class F>
auto lsm_call(int n_threads, F&& f) {
    try {
        return run_parallel(n_threads, std::forward<F>(f));
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

void bind_landscape(py::module_& m) {
    py::class_<lsm::Landscape>(m, "LsmLandscape")
        .def(py::init(&make_landscape), py::arg("values"), py::arg("na_mask"), py::arg("resx"), py::arg("resy"),
             py::arg("options"))
        .def_property_readonly("class_values", [](const lsm::Landscape& l) {
            return py::array_t<std::int32_t>(l.class_values().size(), l.class_values().data());
        })
        .def_property_readonly("class_cells", [](const lsm::Landscape& l) {
            return py::array_t<std::int64_t>(l.class_cells().size(), l.class_cells().data());
        })
        .def_property_readonly("valid_cells", &lsm::Landscape::valid_cells)
        .def("n_patches", [](lsm::Landscape& l) { return lsm_call(l.options().n_threads, [&] { return l.patches().n; }); })
        .def("patch_class", [](lsm::Landscape& l) {
            lsm_call(l.options().n_threads, [&] { return l.patches().n; });
            const lsm::Patches& p = l.patches();
            return py::array_t<std::int32_t>(p.cls.size(), p.cls.data());
        })
        .def("labels", [](lsm::Landscape& l) {
            auto v = lsm_call(l.options().n_threads, [&] { return l.labels(); });
            return to_array(std::move(v), l.nx(), l.ny());
        })
        .def("patch_metric", [](lsm::Landscape& l, const std::string& name) {
            auto v = lsm_call(l.options().n_threads, [&] { return l.patch_metric(name); });
            return py::array_t<double>(v.size(), v.data());
        })
        .def("class_metric", [](lsm::Landscape& l, const std::string& name) {
            auto v = lsm_call(l.options().n_threads, [&] { return l.class_metric(name); });
            return py::array_t<double>(v.size(), v.data());
        })
        .def("landscape_metric", [](lsm::Landscape& l, const std::string& name) {
            return lsm_call(l.options().n_threads, [&] { return l.landscape_metric(name); });
        })
        .def("centroids", [](lsm::Landscape& l, double xmin, double xmax, double ymin, double ymax, bool cell_center) {
            const lsm::CentroidTable t =
                lsm_call(l.options().n_threads, [&] { return l.centroids(xmin, xmax, ymin, ymax, cell_center); });
            py::dict d;
            d["patch"] = vector_array(t.patch);
            d["x"] = vector_array(t.x);
            d["y"] = vector_array(t.y);
            return d;
        })
        .def("nearest_neighbours", [](lsm::Landscape& l, bool return_id) {
            const lsm::NeighbourTable t = lsm_call(l.options().n_threads, [&] { return l.nearest_neighbours(return_id); });
            py::dict d;
            d["patch"] = vector_array(t.patch);
            d["dist"] = vector_array(t.dist);
            d["neighbour"] = vector_array(t.neighbour);
            return d;
        })
        .def("circles", [](lsm::Landscape& l, bool by_class) {
            const std::vector<lsm::EnclosingCircle> c = lsm_call(l.options().n_threads, [&] {
                return by_class ? l.class_circles() : l.patch_circles();
            });
            std::vector<double> cx(c.size()), cy(c.size()), r(c.size());
            for (std::size_t i = 0; i < c.size(); ++i) {
                cx[i] = c[i].cx;
                cy[i] = c[i].cy;
                r[i] = c[i].r;
            }
            py::dict d;
            d["cx"] = vector_array(cx);
            d["cy"] = vector_array(cy);
            d["r"] = vector_array(r);
            return d;
        })
        .def("patch_boundaries", [](lsm::Landscape& l, bool consider_boundary, int edge_depth) {
            auto v = lsm_call(l.options().n_threads, [&] { return l.patch_boundaries(consider_boundary, edge_depth); });
            return to_array(std::move(v), l.nx(), l.ny());
        })
        .def("cooccurrence", [](lsm::Landscape& l, const std::vector<std::pair<int, int>>& offsets) {
            const std::vector<std::int64_t> v = lsm_call(l.options().n_threads, [&] { return l.cooccurrence(offsets); });
            const auto k = static_cast<py::ssize_t>(l.n_classes());
            py::array_t<std::int64_t> a({k, k});
            std::copy(v.begin(), v.end(), a.mutable_data());
            return a;
        })
        .def("prepare", [](lsm::Landscape& l, const std::vector<int>& levels, const std::vector<std::string>& names) {
            const std::vector<lsm::MetricSpec> specs = metric_specs(levels, names);
            lsm_call(l.options().n_threads, [&] { l.prepare(specs); });
        })
        .def("long_table", [](lsm::Landscape& l, const std::vector<int>& levels, const std::vector<std::string>& names) {
            const std::vector<lsm::MetricSpec> specs = metric_specs(levels, names);
            const lsm::LongTable t = lsm_call(l.options().n_threads, [&] {
                lsm::LongTable out;
                lsm::append_long_table(l, specs, 0, out);
                return out;
            });
            return long_table_dict(t);
        });

    py::class_<lsm::TiledLandscape>(m, "LsmTiled")
        .def(py::init([](i64 nx, i64 ny, double resx, double resy, const py::dict& options,
                         const std::vector<std::int32_t>& class_values) {
                 try {
                     return std::make_unique<lsm::TiledLandscape>(nx, ny, resx, resy, make_options(options),
                                                                  class_values);
                 } catch (const std::invalid_argument& e) {
                     throw py::value_error(e.what());
                 }
             }),
             py::arg("nx"), py::arg("ny"), py::arg("resx"), py::arg("resy"), py::arg("options"),
             py::arg("class_values"))
        .def("add_block", [](lsm::TiledLandscape& t, const CArray<std::int32_t>& values,
                             const CArray<std::uint8_t>& na, i64 y0, i64 rows, i64 above, i64 below) {
            tiled_check(t, values, na, rows + above + below);
            lsm_call(t.n_threads(), [&] { t.add_block(values.data(), na.data(), y0, rows, above, below); });
        })
        .def("finish", [](lsm::TiledLandscape& t) { lsm_call(t.n_threads(), [&] { t.finish(); }); })
        .def("landscape", [](lsm::TiledLandscape& t) -> lsm::Landscape& { return t.landscape(); },
             py::return_value_policy::reference_internal)
        .def("detail_halo", &lsm::TiledLandscape::detail_halo, py::arg("core"), py::arg("enn"))
        .def("begin_detail", [](lsm::TiledLandscape& t, bool core, bool gyrate, bool circle, bool enn) {
            lsm_call(t.n_threads(), [&] { t.begin_detail(core, gyrate, circle, enn); });
        })
        .def("detail_block", [](lsm::TiledLandscape& t, const CArray<std::int32_t>& values,
                                const CArray<std::uint8_t>& na, i64 y0, i64 rows, i64 above, i64 below) {
            tiled_check(t, values, na, rows + above + below);
            lsm_call(t.n_threads(), [&] { t.detail_block(values.data(), na.data(), y0, rows, above, below); });
        })
        .def("end_detail", [](lsm::TiledLandscape& t) { return lsm_call(t.n_threads(), [&] { return t.end_detail(); }); })
        .def("tie_block", [](lsm::TiledLandscape& t, const CArray<std::int32_t>& values,
                             const CArray<std::uint8_t>& na, i64 y0, i64 rows, i64 above, i64 below) {
            tiled_check(t, values, na, rows + above + below);
            lsm_call(t.n_threads(), [&] { t.tie_block(values.data(), na.data(), y0, rows, above, below); });
        })
        .def("end_ties", [](lsm::TiledLandscape& t) { lsm_call(t.n_threads(), [&] { t.end_ties(); }); })
        .def("block_labels", [](const lsm::TiledLandscape& t, const CArray<std::int32_t>& values,
                                const CArray<std::uint8_t>& na, i64 y0, i64 rows, i64 above, i64 below) {
            tiled_check(t, values, na, rows + above + below);
            auto v = lsm_call(t.n_threads(),
                              [&] { return t.block_labels(values.data(), na.data(), y0, rows, above, below); });
            return to_array(std::move(v), t.nx(), rows);
        });

    // rasters: list of (values, na_mask or None, resx, resy)
    m.def(
        "lsm_batch",
        [](const py::list& rasters, const py::dict& options, const std::vector<int>& levels,
           const std::vector<std::string>& names) {
            const std::vector<lsm::MetricSpec> specs = metric_specs(levels, names);
            const lsm::Options opt = make_options(options);
            std::vector<CArray<std::int32_t>> values;
            std::vector<CArray<std::uint8_t>> masks(rasters.size());
            std::vector<lsm::RasterView> views;
            values.reserve(rasters.size());
            for (std::size_t i = 0; i < rasters.size(); ++i) {
                const py::tuple t = rasters[i].cast<py::tuple>();
                values.push_back(t[0].cast<CArray<std::int32_t>>());
                views.push_back(raster_view(values.back(), t[1], masks[i], t[2].cast<double>(), t[3].cast<double>()));
            }
            const lsm::LongTable t = lsm_call(opt.n_threads, [&] { return lsm::batch_long_table(views, opt, specs); });
            return long_table_dict(t);
        },
        py::arg("rasters"), py::arg("options"), py::arg("levels"), py::arg("names"));

    m.def(
        "lsm_raster_boundaries",
        [](const CArray<std::int32_t>& values, const py::object& na_mask, bool consider_boundary, int edge_depth,
           bool patch_id) {
            CArray<std::uint8_t> mask_arr;
            const lsm::RasterView r = raster_view(values, na_mask, mask_arr, 1.0, 1.0);
            constexpr std::int32_t kNa = std::numeric_limits<std::int32_t>::min();
            auto v = lsm_call(0, [&] {
                return lsm::raster_boundaries(r.values, r.na_mask, r.nx, r.ny, consider_boundary, edge_depth, patch_id,
                                              kNa);
            });
            return to_array(std::move(v), r.nx, r.ny);
        },
        py::arg("values"), py::arg("na_mask"), py::arg("consider_boundary"), py::arg("edge_depth"),
        py::arg("patch_id"));
    m.def(
        "lsm_unique_values",
        [](const CArray<std::int32_t>& values, const py::object& na_mask) {
            CArray<std::uint8_t> mask_arr;
            const lsm::RasterView r = raster_view(values, na_mask, mask_arr, 1.0, 1.0);
            std::vector<std::int32_t> v;
            {
                py::gil_scoped_release release;
                v = lsm::unique_values(r.values, r.na_mask, r.nx * r.ny);
            }
            py::array_t<std::int32_t> a(static_cast<py::ssize_t>(v.size()));
            std::copy(v.begin(), v.end(), a.mutable_data());
            return a;
        },
        py::arg("values"), py::arg("na_mask"), "The distinct non-NA values of a raster, sorted.");
    m.def(
        "lsm_r_cor",
        [](const CArray<double>& x, const std::string& method) {
            if (x.ndim() != 2) throw py::value_error("expected a 2-D array (observations x variables)");
            const auto n = static_cast<lsm::i64>(x.shape(0));
            const auto m = static_cast<lsm::i64>(x.shape(1));
            std::vector<double> col(static_cast<std::size_t>(n * m));  // column-major, as R
            for (lsm::i64 i = 0; i < n; ++i) {
                for (lsm::i64 j = 0; j < m; ++j) col[j * n + i] = x.data()[i * m + j];
            }
            const std::vector<double> r = lsm_call(0, [&] { return lsm::r_cor(col, n, m, method); });
            py::array_t<double> a({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(m)});
            std::copy(r.begin(), r.end(), a.mutable_data());
            return a;
        },
        py::arg("x"), py::arg("method"), "R's cor() of the columns of x.");
    m.def(
        "lsm_rasterize",
        [](const std::vector<CArray<double>>& ring_arrays, const std::array<double, 6>& gt, int nx, int ny) {
            std::vector<std::vector<std::pair<double, double>>> rings;
            for (const auto& r : ring_arrays) {
                if (r.ndim() != 2 || r.shape(1) < 2) throw py::value_error("rings must be (n, 2) arrays");
                std::vector<std::pair<double, double>> ring(static_cast<std::size_t>(r.shape(0)));
                for (py::ssize_t i = 0; i < r.shape(0); ++i) ring[i] = {r.at(i, 0), r.at(i, 1)};
                rings.push_back(std::move(ring));
            }
            auto v = lsm_call(0, [&] { return lsm::rasterize_all_touched(rings, gt.data(), nx, ny); });
            return to_array(std::move(v), nx, ny);
        },
        py::arg("rings"), py::arg("geotransform"), py::arg("nx"), py::arg("ny"),
        "Mask of polygon rings burnt as GDAL's rasterizer with ALL_TOUCHED=TRUE.");
    m.def(
        "lsm_r_sincos",
        [](const CArray<double>& x) {
            const auto n = static_cast<py::ssize_t>(x.size());
            py::array_t<double> s(n), c(n);
            for (py::ssize_t i = 0; i < n; ++i) {
                s.mutable_data()[i] = lsm::r_sin(x.data()[i]);
                c.mutable_data()[i] = lsm::r_cos(x.data()[i]);
            }
            return py::make_tuple(s, c);
        },
        py::arg("x"), "R's sin(x) and cos(x) (the x87 bits of R on x86-64 Windows).");
    m.def("lsm_r_statistics", &lsm::r_statistics, py::arg("x"),
          "R's sum, mean, sd and median of x, log(x[0]) and sqrt(x[0]), as landschaft computes them.");
    m.def(
        "lsm_r_lm_slope",
        [](const std::vector<double>& x, const std::vector<double>& y) {
            if (x.size() != y.size()) throw py::value_error("x and y must have the same length");
            return lsm::r_lm_slope(x.data(), y.data(), static_cast<int>(x.size()));
        },
        py::arg("x"), py::arg("y"), "The slope of R's lm(y ~ x), as landschaft computes it (NaN for R's NA).");

    m.def(
        "lsm_window",
        [](const CArray<std::int32_t>& values, const py::object& na_mask, double resx, double resy,
           const CArray<std::uint8_t>& window, const std::vector<std::string>& names, const py::dict& options) {
            if (window.ndim() != 2) throw py::value_error("the window must be a 2-D array");
            CArray<std::uint8_t> mask_arr;
            const lsm::RasterView r = raster_view(values, na_mask, mask_arr, resx, resy);
            const lsm::Options opt = make_options(options);
            const std::vector<double> out = lsm_call(opt.n_threads, [&] {
                return lsm::window_metrics(r, window.data(), window.shape(1), window.shape(0), names, opt);
            });
            py::array_t<double> a({static_cast<py::ssize_t>(names.size()), static_cast<py::ssize_t>(r.ny),
                                   static_cast<py::ssize_t>(r.nx)});
            std::copy(out.begin(), out.end(), a.mutable_data());
            return a;
        },
        py::arg("values"), py::arg("na_mask"), py::arg("resx"), py::arg("resy"), py::arg("window"), py::arg("names"),
        py::arg("options"));
}

// One scenario of a habitat graph: the link and/or probabilistic analyses.
py::dict conn_analyse(const CArray<std::int32_t>& pi, const CArray<std::int32_t>& pj, const CArray<double>& value,
                      const CArray<double>& area, const std::string& kind, double threshold, double probability,
                      double pij_min, double expand_limit, bool conefor, bool links, bool probabilities,
                      bool importance, bool edges, bool betweenness, int n_threads) {
    if (pi.ndim() != 1 || pj.ndim() != 1 || value.ndim() != 1 || area.ndim() != 1) {
        throw py::value_error("expected 1-D arrays");
    }
    const auto n = static_cast<std::int32_t>(area.shape(0));
    conn::Pairs pairs;
    pairs.i.assign(pi.data(), pi.data() + pi.shape(0));
    pairs.j.assign(pj.data(), pj.data() + pj.shape(0));
    pairs.value.assign(value.data(), value.data() + value.shape(0));
    std::vector<double> a(area.data(), area.data() + n);
    conn::Scenario s;
    if (kind == "distance") s.kind = conn::Kind::Distance;
    else if (kind == "probability") s.kind = conn::Kind::Probability;
    else if (kind == "link") s.kind = conn::Kind::Link;
    else throw py::value_error("kind must be 'distance', 'probability' or 'link'");
    s.threshold = threshold;
    s.probability = probability;
    s.pij_min = pij_min;
    s.conefor = conefor;

    py::dict out;
    if (links) {
        std::vector<std::int32_t> eu, ev;
        conn::LinkResult r = lsm_call(n_threads, [&] {
            const conn::Csr g = conn::make_link_graph(n, pairs, s);
            eu = g.eu;
            ev = g.ev;
            return conn::analyse_links(g, a.data(), importance, edges, betweenness);
        });
        out["nl"] = r.nl;
        out["nc"] = r.nc;
        out["h"] = r.h;
        out["iic_reach"] = r.iic_reach;
        out["iic_direct"] = r.iic_direct;
        out["unreach_aa"] = r.unreach_aa;
        out["comp_sq"] = r.comp_sq;
        out["component"] = vector_array(r.component);
        if (importance) {
            out["iic_row"] = vector_array(r.iic_row);
            out["unreach_a"] = vector_array(r.unreach_a);
            out["h_row"] = vector_array(r.h_row);
            out["iic_conn"] = vector_array(r.iic_conn);
            out["h_conn"] = vector_array(r.h_conn);
            out["degree"] = vector_array(r.degree);
            out["pieces"] = vector_array(r.pieces);
            out["pieces_sq"] = vector_array(r.pieces_sq);
        }
        if (betweenness) {
            out["bc"] = vector_array(r.bc);
            out["bc_iic"] = vector_array(r.bc_iic);
        }
        if (edges) {
            out["link_i"] = vector_array(eu);
            out["link_j"] = vector_array(ev);
            out["edge_iic"] = vector_array(r.edge_iic);
            out["edge_h"] = vector_array(r.edge_h);
        }
    }
    if (probabilities) {
        std::vector<double> f_row, awf_row;
        std::vector<std::int32_t> eu, ev;
        conn::ProbResult r = lsm_call(n_threads, [&] {
            const conn::Csr g = conn::make_prob_graph(n, pairs, s, a.data(), &f_row, &awf_row);
            eu = g.eu;
            ev = g.ev;
            return conn::analyse_probabilities(g, a.data(), expand_limit, importance, edges, betweenness);
        });
        out["pc_num"] = r.pc_num;
        out["pc_direct"] = r.pc_direct;
        out["f_row"] = vector_array(f_row);
        out["awf_row"] = vector_array(awf_row);
        if (importance) {
            out["pc_row"] = vector_array(r.pc_row);
            out["pc_conn"] = vector_array(r.pc_conn);
        }
        if (betweenness) out["bc_pc"] = vector_array(r.bc_pc);
        if (edges) {
            out["edge_i"] = vector_array(eu);
            out["edge_j"] = vector_array(ev);
            out["edge_pc"] = vector_array(r.edge_pc);
        }
    }
    return out;
}

conn::Scenario conn_scenario(const std::string& kind, double threshold, double probability, double pij_min,
                             bool conefor) {
    conn::Scenario s;
    if (kind == "distance") s.kind = conn::Kind::Distance;
    else if (kind == "probability") s.kind = conn::Kind::Probability;
    else if (kind == "link") s.kind = conn::Kind::Link;
    else throw py::value_error("kind must be 'distance', 'probability' or 'link'");
    s.threshold = threshold;
    s.probability = probability;
    s.pij_min = pij_min;
    s.conefor = conefor;
    return s;
}

conn::Pairs conn_pairs(const CArray<std::int32_t>& pi, const CArray<std::int32_t>& pj, const CArray<double>& value) {
    conn::Pairs pairs;
    pairs.i.assign(pi.data(), pi.data() + pi.shape(0));
    pairs.j.assign(pj.data(), pj.data() + pj.shape(0));
    pairs.value.assign(value.data(), value.data() + value.shape(0));
    return pairs;
}

// Shortest path lengths between all nodes of a scenario (hops, or -log p*).
py::array_t<double> conn_all_pairs(const CArray<std::int32_t>& pi, const CArray<std::int32_t>& pj,
                                   const CArray<double>& value, std::int32_t n, const std::string& kind,
                                   double threshold, double probability, double pij_min, double expand_limit,
                                   bool conefor, bool links, int n_threads) {
    const conn::Pairs pairs = conn_pairs(pi, pj, value);
    const conn::Scenario s = conn_scenario(kind, threshold, probability, pij_min, conefor);
    std::vector<double> d = lsm_call(n_threads, [&] {
        std::vector<double> zero(static_cast<std::size_t>(n), 0.0);
        const conn::Csr g = links ? conn::make_link_graph(n, pairs, s)
                                  : conn::make_prob_graph(n, pairs, s, zero.data(), nullptr, nullptr);
        return conn::all_pairs(g, links, expand_limit);
    });
    return to_array(std::move(d), n, n);
}

// The length of the edge each pair makes (1 for a link, -log p), NaN for none.
py::array_t<double> conn_pair_lengths(const CArray<double>& value, const std::string& kind, double threshold,
                                      double probability, double pij_min, bool conefor, bool links, bool cut) {
    const conn::Scenario s = conn_scenario(kind, threshold, probability, pij_min, conefor);
    std::vector<double> out(static_cast<std::size_t>(value.shape(0)));
    for (py::ssize_t k = 0; k < value.shape(0); ++k) out[k] = conn::pair_length(value.data()[k], s, links, cut);
    return vector_array(out);
}

py::dict gains_dict(conn::Gains&& g) {
    py::dict out;
    out["num"] = vector_array(g.num);
    out["h"] = vector_array(g.h);
    return out;
}

py::dict conn_addition_gains(const CArray<double>& D, const CArray<double>& area, const CArray<std::int64_t>& cand_start,
                             const CArray<std::int32_t>& cand_node, const CArray<double>& cand_len,
                             const CArray<double>& cand_area, bool hops, int n_threads) {
    const auto n = static_cast<std::int32_t>(area.shape(0));
    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n) throw py::value_error("D must be n x n");
    std::vector<double> d(D.data(), D.data() + static_cast<std::size_t>(n) * n);
    const auto nc = static_cast<std::int32_t>(cand_area.shape(0));
    return gains_dict(lsm_call(n_threads, [&] {
        return conn::addition_gains(d, n, area.data(), cand_start.data(), cand_node.data(), cand_len.data(), nc,
                                    cand_area.data(), hops);
    }));
}

py::dict conn_edge_gains(const CArray<double>& D, const CArray<double>& area, const CArray<std::int32_t>& ea,
                         const CArray<std::int32_t>& eb, const CArray<double>& elen, bool hops, int n_threads) {
    const auto n = static_cast<std::int32_t>(area.shape(0));
    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n) throw py::value_error("D must be n x n");
    std::vector<double> d(D.data(), D.data() + static_cast<std::size_t>(n) * n);
    const auto ne = static_cast<std::int32_t>(ea.shape(0));
    return gains_dict(lsm_call(n_threads, [&] {
        return conn::edge_gains(d, n, area.data(), ea.data(), eb.data(), elen.data(), ne, hops);
    }));
}

py::dict conn_centrality(const CArray<std::int32_t>& pi, const CArray<std::int32_t>& pj, const CArray<double>& length,
                         std::int32_t n, bool unit, int n_threads) {
    std::vector<std::int32_t> ei(pi.data(), pi.data() + pi.shape(0)), ej(pj.data(), pj.data() + pj.shape(0));
    std::vector<double> len(length.data(), length.data() + length.shape(0));
    conn::Centrality c = lsm_call(n_threads, [&] {
        const conn::Csr g = conn::make_length_graph(n, ei, ej, len);
        return conn::centrality(g, unit);
    });
    py::dict out;
    out["betweenness"] = vector_array(c.betweenness);
    out["closeness_sum"] = vector_array(c.closeness_sum);
    out["reached"] = vector_array(c.reached);
    return out;
}

py::dict pair_dict(conn::PairList&& p) {
    py::dict out;
    out["i"] = vector_array(p.i);
    out["j"] = vector_array(p.j);
    out["distance"] = vector_array(p.distance);
    return out;
}

// Nodes of a raster of patch ids (Makurhini's nodesfile / distancefile).
py::dict conn_raster_nodes(const CArray<std::int32_t>& values, const py::object& na, double xmin, double xmax,
                           double ymin, double ymax, bool centroids, bool keep_index, int n_threads) {
    if (values.ndim() != 2) throw py::value_error("expected a 2-D raster");
    conn::RasterGrid grid;
    grid.ny = values.shape(0);
    grid.nx = values.shape(1);
    grid.xmin = xmin;
    grid.xmax = xmax;
    grid.ymin = ymin;
    grid.ymax = ymax;
    CArray<std::uint8_t> mask;
    const std::uint8_t* na_ptr = nullptr;
    if (!na.is_none()) {
        mask = na.cast<CArray<std::uint8_t>>();
        if (mask.ndim() != 2 || mask.shape(0) != grid.ny || mask.shape(1) != grid.nx) {
            throw py::value_error("the mask does not match the raster");
        }
        na_ptr = mask.data();
    }
    conn::RasterNodes r =
        lsm_call(n_threads, [&] { return conn::raster_nodes(values.data(), na_ptr, grid, centroids, keep_index); });
    py::dict out;
    out["ids"] = vector_array(r.ids);
    out["cells"] = vector_array(r.cells);
    if (centroids) {
        out["cx"] = vector_array(r.cx);
        out["cy"] = vector_array(r.cy);
    }
    if (keep_index) out["node"] = to_array(std::move(r.node), grid.nx, grid.ny);
    return out;
}

py::dict conn_point_distances(const CArray<double>& x, const CArray<double>& y, double max_distance, int n_threads) {
    if (x.ndim() != 1 || y.ndim() != 1 || x.shape(0) != y.shape(0)) throw py::value_error("expected two 1-D arrays");
    const auto n = static_cast<std::int32_t>(x.shape(0));
    return pair_dict(lsm_call(n_threads, [&] { return conn::point_distances(x.data(), y.data(), n, max_distance); }));
}

py::dict conn_raster_edge_distances(const CArray<std::int32_t>& node, std::int32_t n_nodes, double resx, double resy,
                                    double max_distance, int n_threads) {
    if (node.ndim() != 2) throw py::value_error("expected a 2-D node raster");
    return pair_dict(lsm_call(n_threads, [&] {
        return conn::raster_edge_distances(node.data(), n_nodes, node.shape(1), node.shape(0), resx, resy, max_distance);
    }));
}

py::array_t<double> conn_gdal_areas(const CArray<double>& x, const CArray<double>& y, const CArray<std::int64_t>& ring_start,
                                    const CArray<std::int64_t>& poly_start, const CArray<std::int64_t>& geom_start) {
    const i64 n_geoms = geom_start.shape(0) - 1;
    if (n_geoms < 0) throw py::value_error("geom_start needs at least one entry");
    std::vector<double> a = lsm_call(0, [&] {
        return conn::gdal_areas(x.data(), y.data(), ring_start.data(), poly_start.data(), geom_start.data(), n_geoms);
    });
    return vector_array(a);
}

py::dict conn_least_cost(const CArray<double>& resistance, double xmin, double xmax, double ymin, double ymax,
                         const CArray<std::int64_t>& node_cell, int directions, const std::string& function,
                         double max_cost, int n_threads, int max_threads) {
    if (resistance.ndim() != 2) throw py::value_error("expected a 2-D resistance raster");
    conn::CostGrid grid;
    grid.ny = resistance.shape(0);
    grid.nx = resistance.shape(1);
    grid.xmin = xmin;
    grid.xmax = xmax;
    grid.ymin = ymin;
    grid.ymax = ymax;
    grid.directions = directions;
    if (function == "makurhini") grid.function = conn::CostFunction::Destination;
    else if (function == "mean") grid.function = conn::CostFunction::Mean;
    else throw py::value_error("cost_function must be 'makurhini' or 'mean'");
    const auto n = static_cast<std::int32_t>(node_cell.shape(0));
    return pair_dict(lsm_call(n_threads, [&] {
        return conn::least_cost_distances(resistance.data(), grid, node_cell.data(), n, max_cost, max_threads);
    }));
}

py::dict conn_conductance(const CArray<double>& resistance, double xmin, double xmax, double ymin, double ymax,
                          int directions, const std::string& function) {
    if (resistance.ndim() != 2) throw py::value_error("expected a 2-D resistance raster");
    conn::CostGrid grid;
    grid.ny = resistance.shape(0);
    grid.nx = resistance.shape(1);
    grid.xmin = xmin;
    grid.xmax = xmax;
    grid.ymin = ymin;
    grid.ymax = ymax;
    grid.directions = directions;
    if (function == "makurhini") grid.function = conn::CostFunction::Destination;
    else if (function == "mean") grid.function = conn::CostFunction::Mean;
    else throw py::value_error("cost_function must be 'makurhini' or 'mean'");
    conn::ConductanceEdges e;
    {
        py::gil_scoped_release release;
        e = conn::conductance_edges(resistance.data(), grid);
    }
    py::dict out;
    out["a"] = vector_array(e.a);
    out["b"] = vector_array(e.b);
    out["conductance"] = vector_array(e.conductance);
    return out;
}

py::dict conn_samc_transitions(const CArray<double>& resistance, const CArray<double>& absorption, int directions,
                               const std::string& function) {
    if (resistance.ndim() != 2 || absorption.ndim() != 2 || resistance.shape(0) != absorption.shape(0) ||
        resistance.shape(1) != absorption.shape(1))
        throw py::value_error("expected a resistance and an absorption raster of the same shape");
    conn::CostFunction f;
    if (function == "makurhini") f = conn::CostFunction::Destination;
    else if (function == "mean") f = conn::CostFunction::Mean;
    else throw py::value_error("cost_function must be 'makurhini' or 'mean'");
    conn::SamcTransitions t;
    {
        py::gil_scoped_release release;
        t = conn::samc_transitions(resistance.data(), absorption.data(), resistance.shape(1), resistance.shape(0),
                                   directions, f);
    }
    py::dict out;
    out["cell"] = vector_array(t.cell);
    out["from"] = vector_array(t.from);
    out["to"] = vector_array(t.to);
    out["q"] = vector_array(t.q);
    return out;
}

// AVX2 and FMA usable (CPU and OS): landschaft._circuit_avx2 may be loaded.
bool cpu_supports_avx2_fma() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0, osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
    if (!(fma && osxsave && avx)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // the OS saves the YMM registers
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "landschaft C++ core (MSPA, GTN, FAD, distance, entropy and landscape metrics).";
    bind_landscape(m);
    m.def("conn_analyse", &conn_analyse, py::arg("pi"), py::arg("pj"), py::arg("value"), py::arg("area"),
          py::arg("kind"), py::arg("threshold"), py::arg("probability"), py::arg("pij_min"), py::arg("expand_limit"),
          py::arg("conefor"), py::arg("links"), py::arg("probabilities"), py::arg("importance"), py::arg("edges") = false,
          py::arg("betweenness") = false, py::arg("n_threads") = 0, "Connectivity indices of one habitat-graph scenario (see src/cpp/conn/graph.hpp).");
    m.def("conn_centrality", &conn_centrality, py::arg("pi"), py::arg("pj"), py::arg("length"), py::arg("n"),
          py::arg("unit"), py::arg("n_threads") = 0, "igraph's betweenness and closeness sums over given edges.");
    m.def("conn_all_pairs", &conn_all_pairs, py::arg("pi"), py::arg("pj"), py::arg("value"), py::arg("n"),
          py::arg("kind"), py::arg("threshold"), py::arg("probability"), py::arg("pij_min"), py::arg("expand_limit"),
          py::arg("conefor"), py::arg("links"), py::arg("n_threads") = 0, "All-pairs shortest lengths (n x n).");
    m.def("conn_pair_lengths", &conn_pair_lengths, py::arg("value"), py::arg("kind"), py::arg("threshold"),
          py::arg("probability"), py::arg("pij_min"), py::arg("conefor"), py::arg("links"), py::arg("cut") = true,
          "Edge length of each pair (NaN when it makes no edge).");
    m.def("conn_addition_gains", &conn_addition_gains, py::arg("D"), py::arg("area"), py::arg("cand_start"),
          py::arg("cand_node"), py::arg("cand_len"), py::arg("cand_area"), py::arg("hops"), py::arg("n_threads") = 0,
          "Gains of adding each candidate node.");
    m.def("conn_edge_gains", &conn_edge_gains, py::arg("D"), py::arg("area"), py::arg("ea"), py::arg("eb"),
          py::arg("elen"), py::arg("hops"), py::arg("n_threads") = 0, "Gains of adding each edge.");
    circuit::register_python(m);
    nlm::register_python(m);
    cost::register_python(m);
    bio::register_python(m);
    surf::register_python(m);
    prio::register_python(m);
    lsio::register_python(m);
    sdm::register_python(m);
    sim::register_python(m);
    opt::register_python(m);
    rest::register_python(m);
    m.def("cpu_supports_avx2_fma", &cpu_supports_avx2_fma, "AVX2 and FMA usable (landschaft._circuit_avx2).");
    m.def("conn_raster_nodes", &conn_raster_nodes, py::arg("values"), py::arg("na"), py::arg("xmin"), py::arg("xmax"),
          py::arg("ymin"), py::arg("ymax"), py::arg("centroids"), py::arg("keep_index"), py::arg("n_threads") = 0,
          "Patch ids, cell counts, centroids and node index of a raster of patch ids.");
    m.def("conn_gdal_areas", &conn_gdal_areas, py::arg("x"), py::arg("y"), py::arg("ring_start"),
          py::arg("poly_start"), py::arg("geom_start"), "Polygon areas as GDAL (sf::st_area) computes them.");
    m.def("conn_least_cost", &conn_least_cost, py::arg("resistance"), py::arg("xmin"), py::arg("xmax"),
          py::arg("ymin"), py::arg("ymax"), py::arg("node_cell"), py::arg("directions"), py::arg("function"),
          py::arg("max_cost"), py::arg("n_threads") = 0, py::arg("max_threads") = 0,
          "Least-cost distances between node cells over a resistance raster (gdistance's costDistance).");
    m.def("conn_conductance", &conn_conductance, py::arg("resistance"), py::arg("xmin"), py::arg("xmax"),
          py::arg("ymin"), py::arg("ymax"), py::arg("directions"), py::arg("function"),
          "Conductances between neighbouring cells (gdistance's transition and geoCorrection).");
    m.def("conn_samc_transitions", &conn_samc_transitions, py::arg("resistance"), py::arg("absorption"),
          py::arg("directions"), py::arg("function"), "Transient transition matrix Q of samc's random walk.");
    m.def("conn_point_distances", &conn_point_distances, py::arg("x"), py::arg("y"), py::arg("max_distance"),
          py::arg("n_threads") = 0, "Distances between points, pairs up to max_distance.");
    m.def("conn_raster_edge_distances", &conn_raster_edge_distances, py::arg("node"), py::arg("n_nodes"),
          py::arg("resx"), py::arg("resy"), py::arg("max_distance"), py::arg("n_threads") = 0,
          "Edge-to-edge distances between the patches of a node index raster.");
    m.def("mspa", &mspa, py::arg("input"), py::arg("connectivity") = 8, py::arg("edge_width") = 1,
          py::arg("internal") = true, py::arg("n_threads") = 0,
          "Run MSPA on a uint8 array coded 0=missing, 1=background, 2=foreground.");
    m.def("extract_node_link_contacts", &extract_node_link_contacts, py::arg("nodes"), py::arg("links"),
          "Extract unique (link_id, node_id) contact pairs from labeled node and link rasters.");
    m.def("compute_fad_density", &compute_fad_density, py::arg("input"), py::arg("window_size"),
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Forest Area Density (FAD) for a single window size.");
    m.def("compute_fad", &compute_fad, py::arg("input"), py::arg("window_size"), py::arg("n_threads") = 0,
          "FAD for one window size in one pass: density (GTB's codes), raw density, classes and class counts.");
    m.def("compute_fad_multiscale", &compute_fad_multiscale, py::arg("input"),
          py::arg("window_sizes") = std::vector<int>{7, 13, 27, 81, 243}, py::arg("n_threads") = 0,
          "Compute multiscale FAD across multiple window sizes.");
    m.def("compute_euclidean_distance", &compute_euclidean_distance, py::arg("input"), py::arg("n_threads") = 0,
          "Compute Euclidean Distance transform (signed distance, byte map, and stats).");
    m.def("compute_spatial_entropy", &compute_spatial_entropy, py::arg("input"), py::arg("n_threads") = 0,
          "Compute spatial joint directional entropy index.");
    m.def("compute_spatial_entropy_map", &compute_spatial_entropy_map, py::arg("input"), py::arg("window_size") = 50,
          py::arg("n_threads") = 0, "Compute moving-window spatial entropy map.");
    m.def("compute_contagion_pff", &compute_contagion_pff, py::arg("input"), py::arg("window_size") = 7,
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Contagion (Pff) moving-window metric in byte format [0, 100].");
    m.def("compute_contagion_pff_float", &compute_contagion_pff_float, py::arg("input"), py::arg("window_size") = 7,
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Contagion (Pff) moving-window metric in float format [0.0, 1.0].");
    m.def("compute_clustering_fac", &compute_clustering_fac, py::arg("input"), py::arg("window_size") = 7,
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Foreground Area Clustering (FAC) moving-window metric in byte format [0, 100].");
    m.def("compute_clustering_fac_float", &compute_clustering_fac_float, py::arg("input"), py::arg("window_size") = 7,
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Foreground Area Clustering (FAC) moving-window metric in float format [0.0, 1.0].");
    m.def("compute_landscape_mosaic", &compute_landscape_mosaic, py::arg("input"), py::arg("window_size") = 27,
          py::arg("output_classes") = 19, py::arg("n_threads") = 0,
          "Compute Landscape Mosaic (LM) 19-class or 103-class moving-window metric.");
    m.def("compute_landscape_mosaic_stats", &compute_landscape_mosaic_stats, py::arg("im103"),
          "Compute 103-class distribution and percentages for Landscape Mosaic.");
    m.def("compute_restoration_status", &compute_restoration_status, py::arg("input"),
          "Compute Restoration Status (RSS) metrics (ECA, DOC, CNOA, RAC, etc.).");
    m.def("compute_cost_distance", &compute_cost_distance, py::arg("resistance"), py::arg("source_mask"),
          "Compute 8-neighbor anisotropic cost distance from source mask across resistance surface.");
    m.def("compute_least_cost_path", &compute_least_cost_path, py::arg("resistance"),
          py::arg("marker_a"), py::arg("marker_b"), py::arg("dist_map") = py::none(),
          "Compute two-ended least cost path between marker A and marker B.");
    m.def("compute_parcellation", &compute_parcellation, py::arg("input"), py::arg("connectivity") = 8,
          py::arg("return_labels") = false, "Compute Parcellation class-level and global statistics.");
    m.def("compute_accounting", &compute_accounting, py::arg("input"), py::arg("connectivity") = 8,
          py::arg("thresholds") = std::vector<i64>{50, 250, 1000, 5000, 20000}, py::arg("highlight_largest") = false,
          py::arg("n_threads") = 0, "GTB Accounting: foreground objects grouped into size classes.");
    m.def("compute_contortion", &compute_contortion, py::arg("input"), py::arg("min_area") = 1,
          "Compute Contortion boundary shape tortuosity metric for foreground habitat components.");
    m.def("compute_simple_change", &compute_simple_change, py::arg("im1"), py::arg("im2"),
          "Compute simple binary change between two binary rasters (stable BG, gain, loss, stable FG).");
    m.def("compute_morphological_change", &compute_morphological_change, py::arg("im1"), py::arg("im2"),
          "Compute Morphological Change Detection (MCD) with geodesic reconstruction, closing, and elasticity.");
    m.def("compute_fos_change", &compute_fos_change, py::arg("fos1"), py::arg("fos2"),
          "Compute Fragmentation (FOS) change map, 6x6 transition matrix, and shift percentages.");
    m.def("compute_fad_change_counts", &compute_fad_change_counts, py::arg("a"), py::arg("b"), py::arg("n_threads") = 0,
          "GTB FAD Multiscale change: 7x7 class transition counts and foreground areas of two FAD maps.");
    m.def("compute_influence", &compute_influence, py::arg("input"), py::arg("min_area") = 5000,
          py::arg("core_buffer") = 0, py::arg("buffer") = 0, py::arg("proximity") = false,
          py::arg("metric") = "euclidean", py::arg("n_threads") = 0,
          "GTB Influence Zones / Proximity of the foreground objects.");
    m.def("compute_morph_distance", [](const CArray<std::uint8_t>& mask, bool background, const std::string& metric) {
        const auto im = to_image(mask);
        return to_array(gtb::morph_distance(im, background, parse_metric(metric)), im.nx, im.ny);
    }, py::arg("mask"), py::arg("background") = true, py::arg("metric") = "quasi-euclidean",
          "IDL MORPH_DISTANCE(mask, /BACKGROUND, NEIGHBOR_SAMPLING=3) as ported for GTB.");
    m.def("compute_thin_anchored", [](const CArray<std::uint8_t>& a, int stype, const CArray<std::uint8_t>& anchor) {
        auto im = to_image(a);
        auto an = to_image(anchor);
        gtb::thin_anchored(im, stype, an);
        return to_array(std::move(im));
    }, py::arg("image"), py::arg("stype"), py::arg("anchor"),
          "miallib binOIthin_FIFO: anchored order-independent homotopic thinning (2-pixel border cleared).");
    m.def("compute_geodesic_time", &compute_geodesic_time, py::arg("mask"), py::arg("ref"),
          "miallib uc_sqtgsym (GTB ggeo): geodesic time from ref == 1 across mask; ref == 3 blocks.");
    m.def("compute_grayspatcon_fos", [](const CArray<std::uint8_t>& input, int metric, int window, int n_threads) {
        const auto im = to_image(input);
        gtb::ImageU8 out = run_parallel(n_threads, [&] { return gtb::grayspatcon_fos(im, metric, window); });
        return to_array(std::move(out));
    }, py::arg("input"), py::arg("metric"), py::arg("window"), py::arg("n_threads") = 0,
          "GraySpatCon metric 1 (mean, byte stretch 6) or 52 (clustering, byte stretch 1) as GTB's FOS runs them.");
    m.def("compute_spa", &compute_spa, py::arg("input"), py::arg("variant"), py::arg("n_threads") = 0,
          "GTB SPA2/3/5/6: simplified pattern analysis of the 8-connected foreground.");
    m.def("compute_gray_convolution", &compute_gray_convolution, py::arg("input"), py::arg("metric") = "mean",
          py::arg("window_size") = 7, py::arg("mask_missing") = true, py::arg("n_threads") = 0,
          "Compute moving-window spatial convolution over continuous grayscale raster.");

    m.def("max_threads", &max_threads, "Processors OpenMP can use (1 if built without OpenMP).");
    m.def("team_size", &team_size, "Threads a parallel region opened now gets (tests).");
    m.def("set_num_threads", &set_num_threads, py::arg("n"),
          "Thread count of the calling thread's next OpenMP parallel regions.");
#if defined(_OPENMP)
    m.attr("openmp") = true;
#else
    m.attr("openmp") = false;
#endif
    // R's and Perl's log, sqrt, sin, cos, exp and atan2 on the x87 unit (x86 builds, lsm/rmath.hpp);
    // elsewhere the system library, which can differ in the last bits.
#if defined(LSM_X87_MASM) || defined(LSM_X87_INLINE)
    m.attr("x87_math") = true;
#else
    m.attr("x87_math") = false;
#endif
    py::module_ detail = m.def_submodule("detail");
    bind_detail(detail);
}

