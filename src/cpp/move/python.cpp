// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of movement ecology (landschaft._core.move_*).

#include "move/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>

#include "move/geo.hpp"
#include "move/track.hpp"
#include "nlm/rng.hpp"

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

std::int64_t check(const F64& x, const F64& y, const I64& track) {
    if (x.ndim() != 1 || y.size() != x.size() || track.size() != x.size())
        throw std::invalid_argument("x, y and track: 1-D arrays of the same length");
    return static_cast<std::int64_t>(x.size());
}

}  // namespace

namespace move {

void register_python(py::module_& m) {
    m.def(
        "move_distances",
        [](const F64& x, const F64& y, const I64& track, bool lonlat, int threads) {
            const std::int64_t n = check(x, y, track);
            F64 out(n);
            {
                py::gil_scoped_release release;
                distances(x.data(), y.data(), track.data(), n, lonlat ? Coords::LonLat : Coords::Planar, threads,
                          out.mutable_data());
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("track"), py::arg("lonlat"), py::arg("threads") = 1,
        "Distance from each location to the next of its track (move2's mt_distance; NaN at the last).");

    m.def(
        "move_azimuths",
        [](const F64& x, const F64& y, const I64& track, bool lonlat, double a, double inv_flattening,
           int threads) {
            const std::int64_t n = check(x, y, track);
            F64 out(n);
            {
                py::gil_scoped_release release;
                azimuths(x.data(), y.data(), track.data(), n, lonlat ? Coords::LonLat : Coords::Planar, a,
                         inv_flattening, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("track"), py::arg("lonlat"), py::arg("a") = 6378137.0,
        py::arg("inv_flattening") = 298.257223563, py::arg("threads") = 1,
        "Azimuth (radians) of the step from each location to the next (move2's mt_azimuth).");

    m.def(
        "move_turn_angles",
        [](const F64& azimuth, int threads) {
            const auto n = static_cast<std::int64_t>(azimuth.size());
            F64 out(n);
            {
                py::gil_scoped_release release;
                turn_angles(azimuth.data(), n, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("azimuth"), py::arg("threads") = 1, "Turn angles from the azimuths (move2's mt_turnangle).");

    m.def(
        "move_interpolate",
        [](const F64& x0, const F64& y0, const F64& x1, const F64& y1, const F64& fraction, bool lonlat) {
            const auto n = static_cast<std::int64_t>(x0.size());
            if (y0.size() != n || x1.size() != n || y1.size() != n || fraction.size() != n)
                throw std::invalid_argument("x0, y0, x1, y1 and fraction: the same length");
            F64 x(n), y(n);
            double* px = x.mutable_data();
            double* py_ = y.mutable_data();
            for (std::int64_t i = 0; i < n; ++i) {
                if (lonlat)
                    s2_interpolate(x0.data()[i], y0.data()[i], x1.data()[i], y1.data()[i], fraction.data()[i], px[i],
                                   py_[i]);
                else
                    planar_sample(x0.data()[i], y0.data()[i], x1.data()[i], y1.data()[i], fraction.data()[i], px[i],
                                  py_[i]);
            }
            return py::make_tuple(x, y);
        },
        py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"), py::arg("fraction"), py::arg("lonlat"),
        "Points at a fraction of the steps: s2's interpolation in lon / lat, sf's st_line_sample otherwise.");

    m.def(
        "move_s2_centroid",
        [](const F64& lon, const F64& lat) {
            if (lat.size() != lon.size()) throw std::invalid_argument("lon and lat: the same length");
            double x = 0, y = 0;
            s2_centroid(lon.data(), lat.data(), static_cast<long long>(lon.size()), x, y);
            return py::make_tuple(x, y);
        },
        py::arg("lon"), py::arg("lat"), "s2's centroid of points (degrees).");

    m.def("move_r_fmod", &r_fmod, py::arg("x1"), py::arg("x2"), "R's x %% y for doubles.");
    m.def("move_r_fdiv", &r_fdiv, py::arg("x1"), py::arg("x2"), "R's x %/% y for doubles.");

    m.def(
        "move_brownian",
        [](const std::vector<F64>& times, const std::vector<F64>& sigmas, const F64& starts, std::int32_t seed) {
            if (times.size() != sigmas.size() || starts.ndim() != 2 || starts.shape(0) != py::ssize_t(times.size()) ||
                starts.shape(1) != 2)
                throw std::invalid_argument("a time vector, a sigma and a start (x, y) per track");
            nlm::RRng rng(seed);
            py::list out;
            for (std::size_t k = 0; k < times.size(); ++k) {
                const auto n = static_cast<std::int64_t>(times[k].size());
                F64 x(n), y(n);
                brownian(&rng, times[k].data(), n, sigmas[k].data(), static_cast<std::int64_t>(sigmas[k].size()),
                         starts.at(k, 0), starts.at(k, 1), x.mutable_data(), y.mutable_data());
                out.append(py::make_tuple(x, y));
            }
            return out;
        },
        py::arg("times"), py::arg("sigmas"), py::arg("starts"), py::arg("seed"),
        "mt_sim_brownian_motion: the tracks one after the other from set.seed(seed).");
}

}  // namespace move
