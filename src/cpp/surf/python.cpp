// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the surface metrics (landschaft._core.surf_*).

#include "surf/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

py::dict py_amplitude(const Array& z) {
    surf::Amplitude a{};
    {
        py::gil_scoped_release release;
        a = surf::amplitude(z.data(), static_cast<surf::i64>(z.size()));
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

}  // namespace

namespace surf {

void register_python(py::module_& m) {
    m.def("surf_amplitude", &py_amplitude, py::arg("z"),
          "geodiv's sa, sq, ssk, sku, sv, sph and smean of the values (NaN = NA).");
    m.def("surf_fit_plane", &py_fit_plane, py::arg("z"), py::arg("extent"), py::arg("order"),
          "geodiv's fitplane (spatial::surf.ls): the surface and spatial's ifail.");
    m.def("surf_best_fit_plane", &py_best_fit_plane, py::arg("z"), py::arg("extent"),
          "geodiv's bestfitplane: the surface and its order.");
}

}  // namespace surf
