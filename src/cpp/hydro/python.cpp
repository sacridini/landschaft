// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the terrain hydrology (landschaft._core.hydro_*).

#include "hydro/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "hydro/hydro.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Dirs = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

void check2d(const py::array& a, const char* what) {
    if (a.ndim() != 2) throw std::invalid_argument(std::string("expected a 2-D raster: ") + what);
}

void same_shape(const py::array& a, const py::array& b, const char* what) {
    if (b.ndim() != 2 || a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument(std::string("differs in shape from the DEM: ") + what);
}

hydro::Grid grid_of(const Array& z, double dx, double dy) {
    check2d(z, "dem");
    hydro::Grid g;
    g.z = z.data();
    g.ny = static_cast<int>(z.shape(0));
    g.nx = static_cast<int>(z.shape(1));
    g.dx = dx;
    g.dy = dy;
    return g;
}

template <class T>
py::array_t<T> to_raster(std::vector<T>&& v, int ny, int nx) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)}, holder->data(), owner);
}

}  // namespace

namespace hydro {

void register_python(py::module_& m) {
    m.def("hydro_fill", [](const Array& z, bool epsilon) {
        const Grid g = grid_of(z, 1.0, 1.0);
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = fill(g, epsilon);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("epsilon"), "Priority-Flood depression filling (NaN = missing).");

    m.def("hydro_breach", [](const Array& z, double dx, double dy, int max_length, double max_cost, bool min_dist,
                             double small_num, bool fill_rest) {
        const Grid g = grid_of(z, dx, dy);
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = breach(g, max_length, max_cost, min_dist, small_num, fill_rest);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dx"), py::arg("dy"), py::arg("max_length"), py::arg("max_cost"), py::arg("min_dist"),
       py::arg("small_num"), py::arg("fill_rest"), "Least-cost depression breaching.");

    m.def("hydro_d8", [](const Array& z, double dx, double dy, bool resolve_flats, int n_threads) {
        const Grid g = grid_of(z, dx, dy);
        std::vector<std::int8_t> out;
        {
            py::gil_scoped_release release;
            out = d8(g, resolve_flats, n_threads);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dx"), py::arg("dy"), py::arg("resolve_flats"), py::arg("n_threads"),
       "D8 directions: 0 NE, 1 E, ... 7 N; -1 no flow; -2 missing.");

    m.def("hydro_flat_directions", [](const Array& z, const Dirs& dir, double dx, double dy) {
        const Grid g = grid_of(z, dx, dy);
        same_shape(z, dir, "directions");
        std::vector<std::int8_t> d(dir.data(), dir.data() + dir.size());
        std::vector<std::int8_t> out;
        {
            py::gil_scoped_release release;
            out = flat_directions(g, d);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dir"), py::arg("dx"), py::arg("dy"), "D8 directions of the flat cells (Barnes 2014).");

    m.def("hydro_d8_accumulation", [](const Dirs& dir, std::optional<Array> weight) {
        check2d(dir, "directions");
        if (weight) same_shape(dir, *weight, "weights");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = d8_accumulation(dir.data(), weight ? weight->data() : nullptr, ny, nx);
        }
        return to_raster(std::move(out), ny, nx);
    }, py::arg("dir"), py::arg("weight"), "D8 flow accumulation.");

    m.def("hydro_dinf", [](const Array& z, double dx, double dy, std::optional<Dirs> flats, int n_threads) {
        const Grid g = grid_of(z, dx, dy);
        if (flats) same_shape(z, *flats, "flat directions");
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = dinf(g, flats ? flats->data() : nullptr, n_threads);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dx"), py::arg("dy"), py::arg("flats"), py::arg("n_threads"),
       "D-infinity azimuths (degrees clockwise from north; -1 no flow; NaN missing).");

    m.def("hydro_dinf_accumulation", [](const Array& angle, std::optional<Array> weight) {
        check2d(angle, "directions");
        if (weight) same_shape(angle, *weight, "weights");
        const int ny = static_cast<int>(angle.shape(0)), nx = static_cast<int>(angle.shape(1));
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = dinf_accumulation(angle.data(), weight ? weight->data() : nullptr, ny, nx);
        }
        return to_raster(std::move(out), ny, nx);
    }, py::arg("angle"), py::arg("weight"), "D-infinity flow accumulation.");

    m.def("hydro_mfd_accumulation", [](const Array& z, double dx, double dy, double exponent, bool quinn,
                                       std::optional<Dirs> flats, std::optional<Array> weight) {
        const Grid g = grid_of(z, dx, dy);
        if (flats) same_shape(z, *flats, "flat directions");
        if (weight) same_shape(z, *weight, "weights");
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = mfd_accumulation(g, exponent, quinn, flats ? flats->data() : nullptr,
                                   weight ? weight->data() : nullptr);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dx"), py::arg("dy"), py::arg("exponent"), py::arg("quinn"), py::arg("flats"),
       py::arg("weight"), "Multiple flow direction accumulation (FD8 / Quinn).");

    using Packed = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
    m.def("hydro_mfd_directions", [](const Array& z, int block) {
        const Grid g = grid_of(z, 1.0, 1.0);
        std::vector<std::int32_t> out;
        {
            py::gil_scoped_release release;
            out = mfd_directions(g, block);
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("block"), "pygeoprocessing's multiple flow directions (packed 4-bit weights).");

    m.def("hydro_mfd_packed_accumulation", [](const Packed& dir, std::optional<Array> weight) {
        check2d(dir, "directions");
        if (weight) same_shape(dir, *weight, "weights");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = mfd_packed_accumulation(dir.data(), weight ? weight->data() : nullptr, ny, nx);
        }
        return to_raster(std::move(out), ny, nx);
    }, py::arg("dir"), py::arg("weight"), "pygeoprocessing's multiple flow accumulation.");

    m.def("hydro_mfd_distance_to_channel", [](const Packed& dir, const Mask& channel, std::optional<Array> weight) {
        check2d(dir, "directions");
        same_shape(dir, channel, "channels");
        if (weight) same_shape(dir, *weight, "weights");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = mfd_distance_to_channel(dir.data(), channel.data(), weight ? weight->data() : nullptr, ny, nx);
        }
        return to_raster(std::move(out), ny, nx);
    }, py::arg("dir"), py::arg("channel"), py::arg("weight"), "pygeoprocessing's distance to channel (MFD).");

    m.def("hydro_hand", [](const Array& z, const Dirs& dir, const Mask& streams) {
        const Grid g = grid_of(z, 1.0, 1.0);
        same_shape(z, dir, "directions");
        same_shape(z, streams, "streams");
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            out = hand(g, dir.data(), streams.data());
        }
        return to_raster(std::move(out), g.ny, g.nx);
    }, py::arg("z"), py::arg("dir"), py::arg("streams"), "Height above the nearest drainage along the D8 directions.");
}

}  // namespace hydro
