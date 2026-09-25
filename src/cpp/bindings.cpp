// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the guidospy C++ core.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
    m.doc() = "guidospy C++ core (MSPA and morphological primitives).";
    m.def("mspa", &mspa, py::arg("input"), py::arg("connectivity") = 8, py::arg("edge_width") = 1,
          py::arg("internal") = true, py::arg("n_threads") = 0,
          "Run MSPA on a uint8 array coded 0=missing, 1=background, 2=foreground.");
    m.def("max_threads", &max_threads, "Number of OpenMP threads available (1 if built without OpenMP).");
#if defined(_OPENMP)
    m.attr("openmp") = true;
#else
    m.attr("openmp") = false;
#endif
    py::module_ detail = m.def_submodule("detail");
    bind_detail(detail);
}
