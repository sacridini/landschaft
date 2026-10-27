// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the coastal vulnerability kernels (landschaft._core.serv_coastal_*).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "serv/coastal.hpp"

namespace py = pybind11;

namespace {

using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style>;

void check_xy(const py::array& a, const char* what) {
    if (a.ndim() != 2 || a.shape(1) != 2) throw std::invalid_argument(std::string(what) + " must be (n, 2)");
}

void check_gt(const F64& gt) {
    if (gt.size() != 6) throw std::invalid_argument("a geotransform of 6 numbers");
}

template <class T, class A>
py::object ray_depth(const F64& starts, const F64& ends, const A& raster, const F64& gt, bool has_nodata,
                     double nodata, int threads) {
    check_xy(starts, "the starts");
    check_xy(ends, "the ends");
    check_gt(gt);
    if (starts.shape(0) != ends.shape(0)) throw std::invalid_argument("as many starts as ends");
    if (raster.ndim() != 2) throw std::invalid_argument("a 2-D raster");
    const serv::i64 r = starts.shape(0);
    F64 depth(r);
    serv::i64 status;
    {
        py::gil_scoped_release release;
        status = serv::coastal_ray_depth<T>(starts.data(), ends.data(), r, raster.data(), raster.shape(0),
                                            raster.shape(1), gt.data(), has_nodata, nodata, depth.mutable_data(),
                                            threads);
    }
    return py::make_tuple(depth, status);
}

template <class T, class A>
py::object aggregate(const F64& points, const A& raster, const F64& gt, bool has_nodata, double nodata,
                     bool exact_nodata, const U8& kernel, serv::i64 pixel_dist, int threads) {
    check_xy(points, "the points");
    check_gt(gt);
    if (raster.ndim() != 2) throw std::invalid_argument("a 2-D raster");
    if (kernel.ndim() != 2 || kernel.shape(0) != kernel.shape(1)) throw std::invalid_argument("a square kernel");
    const serv::i64 n = points.shape(0);
    F64 sum(n);
    I64 count(n);
    U8 nonzero(n);
    {
        py::gil_scoped_release release;
        serv::coastal_aggregate<T>(points.data(), n, raster.data(), raster.shape(0), raster.shape(1), gt.data(),
                                   has_nodata, nodata, exact_nodata, kernel.data(), kernel.shape(0), pixel_dist,
                                   sum.mutable_data(), count.mutable_data(), nonzero.mutable_data(), threads);
    }
    return py::make_tuple(sum, count, nonzero);
}

}  // namespace

namespace serv {

void register_coastal(py::module_& m) {
    m.def("serv_coastal_rays", [](const F64& points, const U8& on_land, const F64& deltas, double max_fetch,
                                  const F64& segments, int threads) {
        check_xy(points, "the points");
        check_xy(deltas, "the directions");
        if (segments.ndim() != 2 || segments.shape(1) != 4) throw std::invalid_argument("segments must be (m, 4)");
        const i64 n = points.shape(0);
        const int ndir = static_cast<int>(deltas.shape(0));
        if (on_land.size() != n * ndir) throw std::invalid_argument("on_land must be (n, directions)");
        F64 fdist({n, static_cast<i64>(ndir)});
        F64 ends({n, static_cast<i64>(ndir), i64(2)});
        {
            py::gil_scoped_release release;
            coastal_rays(points.data(), n, on_land.data(), deltas.data(), ndir, max_fetch, segments.data(),
                         segments.shape(0), fdist.mutable_data(), ends.mutable_data(), threads);
        }
        return py::make_tuple(fdist, ends);
    }, py::arg("points"), py::arg("on_land"), py::arg("deltas"), py::arg("max_fetch"), py::arg("segments"),
       py::arg("threads"),
       "The fetch rays of InVEST's coastal vulnerability: (length, end) of each point's rays.");
    m.def("serv_coastal_ray_depth", [](const F64& starts, const F64& ends, const py::array& raster, const F64& gt,
                                       bool has_nodata, double nodata, int threads) -> py::object {
        if (py::isinstance<py::array_t<float>>(raster))
            return ray_depth<float>(starts, ends, F32::ensure(raster), gt, has_nodata, nodata, threads);
        return ray_depth<double>(starts, ends, F64::ensure(raster), gt, has_nodata, nodata, threads);
    }, py::arg("starts"), py::arg("ends"), py::arg("raster"), py::arg("gt"), py::arg("has_nodata"),
       py::arg("nodata"), py::arg("threads"),
       "InVEST's mean bathymetry along each ray (float32 rasters in float32, others float64): (depth, status).");
    m.def("serv_coastal_aggregate", [](const F64& points, const py::array& raster, const F64& gt, bool has_nodata,
                                       double nodata, bool exact_nodata, const U8& kernel, i64 pixel_dist,
                                       int threads) -> py::object {
        if (py::isinstance<py::array_t<float>>(raster))
            return aggregate<float>(points, F32::ensure(raster), gt, has_nodata, nodata, exact_nodata, kernel,
                                    pixel_dist, threads);
        return aggregate<double>(points, F64::ensure(raster), gt, has_nodata, nodata, exact_nodata, kernel,
                                 pixel_dist, threads);
    }, py::arg("points"), py::arg("raster"), py::arg("gt"), py::arg("has_nodata"), py::arg("nodata"),
       py::arg("exact_nodata"), py::arg("kernel"), py::arg("pixel_dist"), py::arg("threads"),
       "InVEST's raster values within a radius of each point: (sum, count, any non-zero).");
}

}  // namespace serv
