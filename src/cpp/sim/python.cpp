// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of ls.simulate (landschaft._core.sim_*).

#include "sim/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "sim/change.hpp"
#include "sim/rng.hpp"

namespace py = pybind11;

namespace {

using i64 = std::int64_t;

// A C-contiguous numpy array of a numeric type as a map (no copy).
sim::MapView map_view(const py::array& a, std::optional<double> nodata) {
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument("the map must be C-contiguous");
    sim::MapView m;
    m.data = a.data();
    m.n = static_cast<i64>(a.size());
    m.has_nodata = nodata.has_value();
    m.nodata = nodata.value_or(0.0);
    const auto dt = a.dtype();
    const char kind = dt.kind();
    const auto size = dt.itemsize();
    if (kind == 'u' && size == 1) m.type = sim::DType::u8;
    else if (kind == 'i' && size == 1) m.type = sim::DType::i8;
    else if (kind == 'b' && size == 1) m.type = sim::DType::u8;
    else if (kind == 'u' && size == 2) m.type = sim::DType::u16;
    else if (kind == 'i' && size == 2) m.type = sim::DType::i16;
    else if (kind == 'u' && size == 4) m.type = sim::DType::u32;
    else if (kind == 'i' && size == 4) m.type = sim::DType::i32;
    else if (kind == 'i' && size == 8) m.type = sim::DType::i64;
    else if (kind == 'f' && size == 4) m.type = sim::DType::f32;
    else if (kind == 'f' && size == 8) m.type = sim::DType::f64;
    else throw std::invalid_argument("unsupported map type " + std::string(py::str(dt)));
    return m;
}

py::array_t<double> sim_map_classes(const py::array& map, std::optional<double> nodata) {
    const auto view = map_view(map, nodata);
    std::vector<double> c;
    {
        py::gil_scoped_release release;
        c = sim::map_classes(view);
    }
    py::array_t<double> out(static_cast<py::ssize_t>(c.size()));
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

py::array_t<std::int64_t> sim_crosstab(const py::array& a, const py::array& b,
                                       const py::array_t<double, py::array::c_style | py::array::forcecast>& ca,
                                       const py::array_t<double, py::array::c_style | py::array::forcecast>& cb,
                                       std::optional<double> nodata_a, std::optional<double> nodata_b) {
    const auto va = map_view(a, nodata_a), vb = map_view(b, nodata_b);
    std::vector<double> classes_a(ca.data(), ca.data() + ca.size()), classes_b(cb.data(), cb.data() + cb.size());
    std::vector<std::int64_t> c;
    {
        py::gil_scoped_release release;
        c = sim::crosstab(va, vb, classes_a, classes_b);
    }
    py::array_t<std::int64_t> out({static_cast<py::ssize_t>(classes_a.size()), static_cast<py::ssize_t>(classes_b.size())});
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

py::array_t<std::int32_t> sim_change_count(const std::vector<py::array>& maps, std::optional<double> nodata) {
    std::vector<sim::MapView> views;
    for (const auto& m : maps) views.push_back(map_view(m, nodata));
    std::vector<std::int32_t> c;
    {
        py::gil_scoped_release release;
        c = sim::change_count(views);
    }
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(c.size()));
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

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
    m.def("sim_map_classes", &sim_map_classes, py::arg("map"), py::arg("nodata") = py::none(),
          "Sorted distinct values of a map's cells that are not missing (NaN or nodata).");
    m.def("sim_crosstab", &sim_crosstab, py::arg("a"), py::arg("b"), py::arg("classes_a"), py::arg("classes_b"),
          py::arg("nodata_a") = py::none(), py::arg("nodata_b") = py::none(),
          "Cells of each class of a (rows) holding each class of b (columns).");
    m.def("sim_change_count", &sim_change_count, py::arg("maps"), py::arg("nodata") = py::none(),
          "Per cell, the number of consecutive maps that differ (-1 where any map is missing).");
}

}  // namespace sim
