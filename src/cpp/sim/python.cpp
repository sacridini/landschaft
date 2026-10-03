// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of ls.simulate (landschaft._core.sim_*).

#include "sim/python.hpp"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

#include "sim/rng.hpp"

namespace py = pybind11;

namespace {

using i64 = std::int64_t;

void check_count(i64 n) {
    if (n < 0) throw std::invalid_argument("n must be >= 0");
}

py::array_t<std::int32_t> sim_crt_rand(std::uint32_t seed, i64 n) {
    check_count(n);
    py::array_t<std::int32_t> out(n);
    auto* p = out.mutable_data();
    sim::CrtRand r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = r.next();
    return out;
}

py::array_t<double> sim_drand48(i64 seed, i64 n) {
    check_count(n);
    py::array_t<double> out(n);
    auto* p = out.mutable_data();
    sim::Drand48 r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = r.next_double();
    return out;
}

py::array_t<i64> sim_lrand48(i64 seed, i64 n, bool is_signed) {
    check_count(n);
    py::array_t<i64> out(n);
    auto* p = out.mutable_data();
    sim::Drand48 r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = is_signed ? r.next_signed() : r.next_long();
    return out;
}

}  // namespace

namespace sim {

void register_python(py::module_& m) {
    m.def("sim_crt_rand", &sim_crt_rand, py::arg("seed"), py::arg("n"),
          "n values of the Microsoft C runtime's rand() after srand(seed) (CLUMondo's generator).");
    m.def("sim_drand48", &sim_drand48, py::arg("seed"), py::arg("n"),
          "n values of G_drand48() after G_srand48(seed) (GRASS GIS, r.futures).");
    m.def("sim_lrand48", &sim_lrand48, py::arg("seed"), py::arg("n"), py::arg("signed") = false,
          "n values of G_lrand48() (or G_mrand48() with signed=True) after G_srand48(seed).");
}

}  // namespace sim
