// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the zonal statistics and the extraction at points
// (landschaft._core.io_*).

#include "io/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "io/io.hpp"

namespace py = pybind11;

namespace {

using lsio::i64;

// Calls f with the typed data of a C-contiguous array of one of the raster types.
template <class F>
auto with_typed(const py::array& a, F&& f) {
    if (!(a.flags() & py::array::c_style)) throw py::value_error("expected a C-contiguous array");
    const py::dtype dt = a.dtype();
    if (dt.is(py::dtype::of<std::uint8_t>())) return f(static_cast<const std::uint8_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int8_t>())) return f(static_cast<const std::int8_t*>(a.data()));
    if (dt.is(py::dtype::of<std::uint16_t>())) return f(static_cast<const std::uint16_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int16_t>())) return f(static_cast<const std::int16_t*>(a.data()));
    if (dt.is(py::dtype::of<std::uint32_t>())) return f(static_cast<const std::uint32_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int32_t>())) return f(static_cast<const std::int32_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int64_t>())) return f(static_cast<const std::int64_t*>(a.data()));
    if (dt.is(py::dtype::of<float>())) return f(static_cast<const float*>(a.data()));
    if (dt.is(py::dtype::of<double>())) return f(static_cast<const double*>(a.data()));
    throw py::type_error("unsupported raster type: use (u)int8/16/32, int64, float32 or float64");
}

py::array_t<double> to_array(std::vector<double>&& v, std::vector<py::ssize_t> shape) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, holder->data(), owner);
}

// (bands, ny, nx) of a 2-D or 3-D raster
std::array<int, 3> raster_shape(const py::array& a) {
    if (a.ndim() == 2) return {1, static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1))};
    if (a.ndim() == 3) return {static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)), static_cast<int>(a.shape(2))};
    throw py::value_error("expected a 2-D (y, x) or 3-D (band, y, x) raster");
}

}  // namespace

namespace lsio {

void register_python(py::module_& m) {
    py::class_<Runs>(m, "IoRuns", "The cells of zones as runs of rows (internal).")
        .def_property_readonly("zones", &Runs::zones)
        .def_property_readonly("runs", [](const Runs& r) { return static_cast<i64>(r.y.size()); })
        .def(
            "cells",
            [](const Runs& r) {
                std::vector<double> c(static_cast<std::size_t>(r.zones()), 0.0);
                for (i64 z = 0; z < r.zones(); ++z) {
                    for (i64 k = r.offsets[z]; k < r.offsets[z + 1]; ++k) c[z] += r.x1[k] - r.x0[k];
                }
                return to_array(std::move(c), {r.zones()});
            },
            "Cells of every zone.");

    m.def(
        "io_polygon_runs",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& xy,
           const py::array_t<i64, py::array::c_style | py::array::forcecast>& ring_offsets,
           const py::array_t<i64, py::array::c_style | py::array::forcecast>& geom_rings, const std::array<double, 6>& gt,
           int nx, int ny, bool all_touched) {
            if (xy.ndim() != 2 || xy.shape(1) != 2) throw py::value_error("xy must be (n, 2)");
            if (geom_rings.size() < 1) throw py::value_error("geom_rings must have ngeoms + 1 entries");
            const i64 ngeoms = static_cast<i64>(geom_rings.size()) - 1;
            py::gil_scoped_release release;
            return polygon_runs(xy.data(), ring_offsets.data(), geom_rings.data(), ngeoms, gt.data(), nx, ny,
                                all_touched);
        },
        py::arg("xy"), py::arg("ring_offsets"), py::arg("geom_rings"), py::arg("geotransform"), py::arg("nx"),
        py::arg("ny"), py::arg("all_touched"), "Cells of polygons as GDAL's rasterizer burns them.");

    m.def(
        "io_label_runs",
        [](const py::array& labels, bool has_nodata, double nodata) {
            if (labels.ndim() != 2) throw py::value_error("zones must be a 2-D raster");
            const int ny = static_cast<int>(labels.shape(0)), nx = static_cast<int>(labels.shape(1));
            std::vector<double> labs;
            Runs runs = with_typed(labels, [&](const auto* p) {
                py::gil_scoped_release release;
                return label_runs(p, ny, nx, has_nodata, nodata, &labs);
            });
            const auto n = static_cast<py::ssize_t>(labs.size());
            return py::make_tuple(std::move(runs), to_array(std::move(labs), {n}));
        },
        py::arg("labels"), py::arg("has_nodata"), py::arg("nodata"), "Cells of every label of a raster of zones.");

    m.def(
        "io_zonal_stats",
        [](const py::array& values, bool has_nodata, double nodata, const Runs& runs, const std::vector<int>& stats,
           const std::vector<double>& percentiles) {
            const auto [bands, ny, nx] = raster_shape(values);
            if (percentiles.size() != stats.size()) throw py::value_error("one percentile per statistic");
            for (i64 r = 0; r < static_cast<i64>(runs.y.size()); ++r) {
                if (runs.y[r] >= ny || runs.x1[r] > nx) throw py::value_error("zones larger than the raster");
            }
            std::vector<double> out = with_typed(values, [&](const auto* p) {
                py::gil_scoped_release release;
                return zonal_stats(p, bands, ny, nx, has_nodata, nodata, runs, stats, percentiles);
            });
            return to_array(std::move(out), {runs.zones(), bands, static_cast<py::ssize_t>(stats.size())});
        },
        py::arg("values"), py::arg("has_nodata"), py::arg("nodata"), py::arg("runs"), py::arg("stats"),
        py::arg("percentiles"), "Statistics of every zone and band: (zones, bands, stats).");

    m.def(
        "io_extract",
        [](const py::array& values, bool has_nodata, double nodata, const std::array<double, 6>& gt,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& x,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& y, bool bilinear, int row0,
           int ny_total) {
            const auto [bands, ny, nx] = raster_shape(values);
            if (ny_total < 0) ny_total = ny;
            if (x.size() != y.size()) throw py::value_error("x and y must have the same length");
            const auto n = static_cast<i64>(x.size());
            std::vector<double> out = with_typed(values, [&](const auto* p) {
                py::gil_scoped_release release;
                return extract(p, bands, ny, nx, has_nodata, nodata, gt.data(), x.data(), y.data(), n, bilinear, row0,
                               ny_total);
            });
            return to_array(std::move(out), {static_cast<py::ssize_t>(n), bands});
        },
        py::arg("values"), py::arg("has_nodata"), py::arg("nodata"), py::arg("geotransform"), py::arg("x"),
        py::arg("y"), py::arg("bilinear"), py::arg("row0") = 0, py::arg("ny_total") = -1,
        "Values of every band at points: (n, bands); values may be the rows [row0, row0 + ny) of the grid.");
}

}  // namespace lsio
