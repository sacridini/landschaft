// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the landschaft C++ core.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "gtb/accounting.hpp"
#include "gtb/change.hpp"
#include "gtb/contagion.hpp"
#include "gtb/distance.hpp"
#include "gtb/entropy.hpp"
#include "gtb/fad.hpp"
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

// Hand a std::vector over to numpy without copying it.
template <class T>
py::array_t<T> to_array(std::vector<T>&& v, i64 nx, i64 ny) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
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
    gtb::LandscapeMosaicStats stats = gtb::compute_landscape_mosaic_stats(im);
    py::dict out;
    out["class_codes"] = stats.class_codes;
    out["pixel_counts"] = stats.pixel_counts;
    out["percentages"] = stats.percentages;
    return out;
}

py::dict compute_restoration_status(const CArray<std::uint8_t>& input) {
    const gtb::ImageU8 im = to_image(input);
    gtb::RestorationStatusResult res = gtb::compute_restoration_status(im);
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
    gtb::Image<float> out = gtb::compute_cost_distance(res, src);
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
    gtb::LeastCostPathResult lcp = gtb::compute_least_cost_path(res, ma, mb, dist_map_ptr.get());
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
    gtb::ParcellationResult res = gtb::compute_parcellation(im, connectivity, return_labels);

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
    gtb::Image<std::uint32_t> out = gtb::compute_contortion(im, min_area);
    return to_array(std::move(out));
}

py::dict compute_simple_change(
    const CArray<std::uint8_t>& im1,
    const CArray<std::uint8_t>& im2
) {
    const gtb::ImageU8 a = to_image(im1);
    const gtb::ImageU8 b = to_image(im2);
    gtb::SimpleChangeResult res = gtb::compute_simple_change(a, b);
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
    gtb::MorphologicalChangeResult res = gtb::compute_morphological_change(a, b);
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
    gtb::FOSChangeResult res = gtb::compute_fos_change(a, b);
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
    return omp_get_max_threads();
#else
    return 1;
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

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "landschaft C++ core (MSPA, GTN, FAD, distance, and entropy).";
    m.def("mspa", &mspa, py::arg("input"), py::arg("connectivity") = 8, py::arg("edge_width") = 1,
          py::arg("internal") = true, py::arg("n_threads") = 0,
          "Run MSPA on a uint8 array coded 0=missing, 1=background, 2=foreground.");
    m.def("extract_node_link_contacts", &extract_node_link_contacts, py::arg("nodes"), py::arg("links"),
          "Extract unique (link_id, node_id) contact pairs from labeled node and link rasters.");
    m.def("compute_fad_density", &compute_fad_density, py::arg("input"), py::arg("window_size"),
          py::arg("apply_gtb_mask") = true, py::arg("n_threads") = 0,
          "Compute Forest Area Density (FAD) for a single window size.");
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
    m.def("compute_spa", &compute_spa, py::arg("input"), py::arg("variant"), py::arg("n_threads") = 0,
          "GTB SPA2/3/5/6: simplified pattern analysis of the 8-connected foreground.");
    m.def("compute_gray_convolution", &compute_gray_convolution, py::arg("input"), py::arg("metric") = "mean",
          py::arg("window_size") = 7, py::arg("mask_missing") = true, py::arg("n_threads") = 0,
          "Compute moving-window spatial convolution over continuous grayscale raster.");

    m.def("max_threads", &max_threads, "Number of OpenMP threads available (1 if built without OpenMP).");
#if defined(_OPENMP)
    m.attr("openmp") = true;
#else
    m.attr("openmp") = false;
#endif
    py::module_ detail = m.def_submodule("detail");
    bind_detail(detail);
}

