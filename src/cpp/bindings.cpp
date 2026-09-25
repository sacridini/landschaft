// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the guidospy C++ core.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "gtb/contagion.hpp"
#include "gtb/distance.hpp"
#include "gtb/entropy.hpp"
#include "gtb/fad.hpp"
#include "gtb/gtn.hpp"
#include "gtb/mosaic.hpp"
#include "gtb/mspa.hpp"
#include "gtb/primitives.hpp"

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
    m.doc() = "guidospy C++ core (MSPA, GTN, FAD, distance, and entropy).";
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

    m.def("max_threads", &max_threads, "Number of OpenMP threads available (1 if built without OpenMP).");
#if defined(_OPENMP)
    m.attr("openmp") = true;
#else
    m.attr("openmp") = false;
#endif
    py::module_ detail = m.def_submodule("detail");
    bind_detail(detail);
}

