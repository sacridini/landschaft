// SPDX-License-Identifier: GPL-3.0-or-later

#include "rest/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "rest/habitat.hpp"
#include "rest/intactness.hpp"

namespace py = pybind11;

namespace {

using i64 = std::int64_t;
using Vec = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using Index = py::array_t<i64, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> to_array(std::vector<T>&& v, std::vector<py::ssize_t> shape) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(shape, holder->data(), owner);
}

const double* optional(const py::object& o, std::vector<Vec>& keep, std::size_t size) {
    if (o.is_none()) return nullptr;
    keep.push_back(o.cast<Vec>());
    if (static_cast<std::size_t>(keep.back().size()) != size) throw std::invalid_argument("array of the wrong size");
    return keep.back().data();
}

py::tuple py_original_vegetation(const Vec& natural, const Vec& anthropic, const py::object& past,
                                 const py::object& ecoregion, const py::object& domain, const Mask& fill,
                                 double threshold, bool restore_current, int threads) {
    if (natural.ndim() != 2 || anthropic.ndim() != 2 || natural.shape(1) != anthropic.shape(1))
        throw std::invalid_argument("natural and anthropic must be (classes, pixels) of the same pixels");
    const int k = static_cast<int>(natural.shape(0)), m = static_cast<int>(anthropic.shape(0));
    const i64 n = static_cast<i64>(natural.shape(1));
    if (static_cast<int>(fill.size()) != k) throw std::invalid_argument("fill: one flag per natural class");
    std::vector<Vec> keep;
    const double* p = optional(past, keep, static_cast<std::size_t>(k) * n);
    const double* e = optional(ecoregion, keep, static_cast<std::size_t>(k) * n);
    Mask dom;
    const std::uint8_t* d = nullptr;
    if (!domain.is_none()) {
        dom = domain.cast<Mask>();
        if (static_cast<i64>(dom.size()) != n) throw std::invalid_argument("domain: one flag per pixel");
        d = dom.data();
    }
    rest::Vegetation v;
    {
        py::gil_scoped_release release;
        v = rest::original_vegetation(natural.data(), k, anthropic.data(), m, n, p, e, d, fill.data(), threshold,
                                      restore_current, threads);
    }
    return py::make_tuple(to_array(std::move(v.original), {k, n}), to_array(std::move(v.restore), {k, n}),
                          to_array(std::move(v.source), {n}), to_array(std::move(v.fill), {k}));
}

py::tuple py_habitat(const Vec& current, const Vec& original, const Vec& area, const Index& range_ptr,
                     const Index& range_idx, const Mask& suitable, int threads) {
    if (current.ndim() != 2 || original.ndim() != 2 || current.shape(0) != original.shape(0) ||
        current.shape(1) != original.shape(1))
        throw std::invalid_argument("current and original must be (classes, pixels) alike");
    const int k = static_cast<int>(current.shape(0));
    const i64 n = static_cast<i64>(current.shape(1));
    const i64 ns = static_cast<i64>(range_ptr.size()) - 1;
    if (static_cast<i64>(area.size()) != n) throw std::invalid_argument("area: one value per pixel");
    if (ns < 0 || static_cast<i64>(suitable.size()) != ns * k) throw std::invalid_argument("suitable: species x classes");
    for (i64 q = 0; q < static_cast<i64>(range_idx.size()); ++q)
        if (range_idx.data()[q] < 0 || range_idx.data()[q] >= n) throw std::invalid_argument("range index out of range");
    std::vector<double> now(static_cast<std::size_t>(ns)), pot(static_cast<std::size_t>(ns));
    {
        py::gil_scoped_release release;
        rest::habitat_areas(current.data(), original.data(), area.data(), n, k, range_ptr.data(), range_idx.data(),
                            suitable.data(), ns, now.data(), pot.data(), threads);
    }
    return py::make_tuple(to_array(std::move(now), {ns}), to_array(std::move(pot), {ns}));
}

py::tuple py_habitat_gain(const Vec& restore, const Vec& area, const Index& range_ptr, const Index& range_idx,
                          const Mask& suitable, const py::object& domain, int threads) {
    if (restore.ndim() != 2) throw std::invalid_argument("restore must be (classes, pixels)");
    const int k = static_cast<int>(restore.shape(0));
    const i64 n = static_cast<i64>(restore.shape(1));
    const i64 ns = static_cast<i64>(range_ptr.size()) - 1;
    if (static_cast<i64>(area.size()) != n) throw std::invalid_argument("area: one value per pixel");
    if (ns < 0 || static_cast<i64>(suitable.size()) != ns * k) throw std::invalid_argument("suitable: species x classes");
    Mask dom;
    const std::uint8_t* d = nullptr;
    if (!domain.is_none()) {
        dom = domain.cast<Mask>();
        if (static_cast<i64>(dom.size()) != n) throw std::invalid_argument("domain: one flag per pixel");
        d = dom.data();
    }
    std::vector<i64> indptr, index;
    std::vector<double> gain;
    {
        py::gil_scoped_release release;
        rest::habitat_gain(restore.data(), area.data(), n, k, range_ptr.data(), range_idx.data(), suitable.data(), ns,
                           d, indptr, index, gain, threads);
    }
    const auto nnz = static_cast<py::ssize_t>(index.size());
    return py::make_tuple(to_array(std::move(indptr), {ns + 1}), to_array(std::move(index), {nnz}),
                          to_array(std::move(gain), {nnz}));
}

rest::IntactnessParams params(double gamma, double beta, double z, int radius, double dx, double dy) {
    return rest::IntactnessParams{gamma, beta, z, radius, dx, dy};
}

py::array_t<double> py_footprint(const Vec& hfi, const py::object& reduction, const py::object& restored,
                                 int threads) {
    if (hfi.ndim() != 2) throw std::invalid_argument("the footprint must be 2-D");
    const i64 ny = hfi.shape(0), nx = hfi.shape(1);
    std::vector<Vec> keep;
    const double* a = optional(reduction, keep, static_cast<std::size_t>(ny * nx));
    const double* x = optional(restored, keep, static_cast<std::size_t>(ny * nx));
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = rest::restored_footprint(hfi.data(), a, x, ny, nx, threads);
    }
    return to_array(std::move(out), {ny, nx});
}

py::array_t<double> py_intactness(const Vec& hfi, double gamma, double beta, double z, int radius, double dx,
                                  double dy, int threads) {
    if (hfi.ndim() != 2) throw std::invalid_argument("the footprint must be 2-D");
    const i64 ny = hfi.shape(0), nx = hfi.shape(1);
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = rest::intactness(hfi.data(), ny, nx, params(gamma, beta, z, radius, dx, dy), threads);
    }
    return to_array(std::move(out), {ny, nx});
}

py::array_t<double> py_intactness_gradient(const Vec& hfi, const Vec& reduction, double gamma, double beta, double z,
                                           int radius, double dx, double dy, int threads) {
    if (hfi.ndim() != 2 || reduction.ndim() != 2 || hfi.shape(0) != reduction.shape(0) ||
        hfi.shape(1) != reduction.shape(1))
        throw std::invalid_argument("the footprint and the reduction must be 2-D and alike");
    const i64 ny = hfi.shape(0), nx = hfi.shape(1);
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = rest::intactness_gradient(hfi.data(), reduction.data(), ny, nx, params(gamma, beta, z, radius, dx, dy),
                                        threads);
    }
    return to_array(std::move(out), {ny, nx});
}

py::tuple py_intactness_secant(const Vec& h0, const Vec& tlo, const Vec& thi, double gamma, double beta, double z,
                               int radius, double dx, double dy, int threads) {
    if (h0.ndim() != 2 || tlo.ndim() != 2 || thi.ndim() != 2 || h0.shape(0) != tlo.shape(0) ||
        h0.shape(1) != tlo.shape(1) || h0.shape(0) != thi.shape(0) || h0.shape(1) != thi.shape(1))
        throw std::invalid_argument("the footprint and the bounds of its reduction must be 2-D and alike");
    const i64 ny = h0.shape(0), nx = h0.shape(1);
    std::vector<double> sigma;
    double u0;
    {
        py::gil_scoped_release release;
        u0 = rest::intactness_secant(h0.data(), tlo.data(), thi.data(), ny, nx,
                                     params(gamma, beta, z, radius, dx, dy), threads, sigma);
    }
    return py::make_tuple(u0, to_array(std::move(sigma), {ny, nx}));
}

}  // namespace

namespace rest {

void register_python(py::module_& m) {
    m.def("rest_original_vegetation", &py_original_vegetation, py::arg("natural"), py::arg("anthropic"),
          py::arg("past"), py::arg("ecoregion"), py::arg("domain"), py::arg("fill"), py::arg("threshold"),
          py::arg("restore_current"), py::arg("threads"),
          "plangea's original areas: (original, restore, source, failsafe shares).");
    m.def("rest_habitat", &py_habitat, py::arg("current"), py::arg("original"), py::arg("area"),
          py::arg("range_ptr"), py::arg("range_idx"), py::arg("suitable"), py::arg("threads"),
          "Current and potential area of habitat of every species.");
    m.def("rest_habitat_gain", &py_habitat_gain, py::arg("restore"), py::arg("area"), py::arg("range_ptr"),
          py::arg("range_idx"), py::arg("suitable"), py::arg("domain"), py::arg("threads"),
          "Habitat gained per unit of a pixel restored, by species (CSR).");
    m.def("rest_footprint", &py_footprint, py::arg("hfi"), py::arg("reduction"), py::arg("restored"),
          py::arg("threads"), "The human footprint after restoration (at least 0).");
    m.def("rest_intactness", &py_intactness, py::arg("hfi"), py::arg("gamma"), py::arg("beta"), py::arg("z"),
          py::arg("radius"), py::arg("dx"), py::arg("dy"), py::arg("threads"),
          "Beyer et al.'s ecological intactness of every cell.");
    m.def("rest_intactness_gradient", &py_intactness_gradient, py::arg("hfi"), py::arg("reduction"),
          py::arg("gamma"), py::arg("beta"), py::arg("z"), py::arg("radius"), py::arg("dx"), py::arg("dy"),
          py::arg("threads"), "d(total intactness) / d(share of each cell restored).");
    m.def("rest_intactness_secant", &py_intactness_secant, py::arg("h0"), py::arg("tlo"), py::arg("thi"),
          py::arg("gamma"), py::arg("beta"), py::arg("z"), py::arg("radius"), py::arg("dx"), py::arg("dy"),
          py::arg("threads"), "A concave upper bound of the total intactness: (u0, sigma).");
}

}  // namespace rest
