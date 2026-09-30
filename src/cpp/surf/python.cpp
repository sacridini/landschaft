// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the surface metrics (landschaft._core.surf_*).

#include "surf/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "surf/surface.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Extent = std::tuple<double, double, double, double>;

surf::Grid grid_of(const Array& z, const Extent& extent) {
    if (z.ndim() != 2) throw std::invalid_argument("expected a 2-D raster");
    surf::Grid g{};
    g.z = z.data();
    g.ny = static_cast<int>(z.shape(0));
    g.nx = static_cast<int>(z.shape(1));
    std::tie(g.xmin, g.xmax, g.ymin, g.ymax) = extent;
    return g;
}

py::array_t<double> to_raster(std::vector<double>&& v, int ny, int nx) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)}, holder->data(), owner);
}

py::dict py_amplitude(const Array& z, unsigned parts) {
    surf::Amplitude a{};
    {
        py::gil_scoped_release release;
        a = surf::amplitude(z.data(), static_cast<surf::i64>(z.size()), parts);
    }
    py::dict d;
    d["sa"] = a.sa;
    d["sq"] = a.sq;
    d["ssk"] = a.ssk;
    d["ssk_unadj"] = a.ssk_unadj;
    d["sku"] = a.sku;
    d["sku_raw"] = a.sku_raw;
    d["sv"] = a.sv;
    d["sph"] = a.sph;
    d["smean"] = a.smean;
    return d;
}

std::tuple<py::array_t<double>, int> py_fit_plane(const Array& z, const Extent& extent, int order) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<double> out;
    int ifail = 0;
    {
        py::gil_scoped_release release;
        out = surf::fit_plane(g, order, &ifail);
    }
    return {to_raster(std::move(out), g.ny, g.nx), ifail};
}

std::tuple<py::array_t<double>, int> py_best_fit_plane(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<double> out;
    int order = 0;
    {
        py::gil_scoped_release release;
        out = surf::best_fit_plane(g, &order);
    }
    return {to_raster(std::move(out), g.ny, g.nx), order};
}

// A metric of the grid, with the GIL released.
template <class F>
double grid_metric(const Array& z, const Extent& extent, F&& f) {
    const surf::Grid g = grid_of(z, extent);
    py::gil_scoped_release release;
    return f(g);
}

py::dict py_hybrid(const Array& z, const Extent& extent, bool iso) {
    const surf::Grid g = grid_of(z, extent);
    double sdq, sdq6, area, flat;
    {
        py::gil_scoped_release release;
        sdq = surf::sdq(g);
        sdq6 = surf::sdq6(g, iso);
        area = surf::surface_area(g, iso);
        flat = surf::flat_area(g, iso);
    }
    py::dict d;
    d["sdq"] = sdq;
    d["sdq6"] = sdq6;
    d["surface_area"] = area;
    d["flat_area"] = flat;
    d["sdr"] = ((area - flat) / flat) * 100;
    return d;
}

py::array_t<std::int64_t> cells_array(std::vector<surf::i64>&& v) {
    auto* holder = new std::vector<surf::i64>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<surf::i64>*>(p); });
    return py::array_t<std::int64_t>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

py::array_t<std::int64_t> py_find_peaks(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<surf::i64> cells;
    {
        py::gil_scoped_release release;
        cells = surf::find_peaks(g);
    }
    return cells_array(std::move(cells));
}

py::array_t<std::int64_t> py_find_valleys(const Array& z, const Extent& extent, bool transposed) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<surf::i64> cells;
    {
        py::gil_scoped_release release;
        cells = surf::find_valleys(g, transposed);
    }
    return cells_array(std::move(cells));
}

}  // namespace

namespace surf {

void register_python(py::module_& m) {
    m.def("surf_hybrid", &py_hybrid, py::arg("z"), py::arg("extent"), py::arg("iso"),
          "geodiv's sdq, sdq6, surface_area, flatsa and sdr (iso: the corrected definitions).");
    m.def("surf_sdq", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::sdq); },
          py::arg("z"), py::arg("extent"), "geodiv's sdq.");
    m.def("surf_sdq6", [](const Array& z, const Extent& e, bool iso) {
        return grid_metric(z, e, [iso](const surf::Grid& g) { return surf::sdq6(g, iso); });
    }, py::arg("z"), py::arg("extent"), py::arg("iso"), "geodiv's sdq6.");
    m.def("surf_find_peaks", &py_find_peaks, py::arg("z"), py::arg("extent"),
          "geodiv's findpeaks: the peak cells (row-order indices).");
    m.def("surf_find_valleys", &py_find_valleys, py::arg("z"), py::arg("extent"), py::arg("transposed"),
          "geodiv's findvalleys: the valley cells (transposed: as geodiv reads the neighbours).");
    m.def("surf_sds", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::sds); },
          py::arg("z"), py::arg("extent"), "geodiv's sds.");
    m.def("surf_ssc", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::ssc); },
          py::arg("z"), py::arg("extent"), "geodiv's ssc.");
    m.def("surf_s10z", [](const Array& z, const Extent& e, bool transposed) {
        return grid_metric(z, e, [transposed](const surf::Grid& g) { return surf::s10z(g, transposed); });
    }, py::arg("z"), py::arg("extent"), py::arg("transposed_valleys"), "geodiv's s10z.");
    m.def("surf_amplitude", &py_amplitude, py::arg("z"), py::arg("parts") = 63u,
          "geodiv's sa, sq, ssk, sku, sv, sph and smean of the values (NaN = NA); parts: "
          "1 sa, 2 sq, 4 ssk, 8 sku, 16 sv and sph, 32 smean.");
    m.def("surf_fit_plane", &py_fit_plane, py::arg("z"), py::arg("extent"), py::arg("order"),
          "geodiv's fitplane (spatial::surf.ls): the surface and spatial's ifail.");
    m.def("surf_best_fit_plane", &py_best_fit_plane, py::arg("z"), py::arg("extent"),
          "geodiv's bestfitplane: the surface and its order.");
}

}  // namespace surf
