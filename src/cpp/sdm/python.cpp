// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of ls.sdm (landschaft._core.sdm_*) and of R's generator.

#include "sdm/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "nlm/rng.hpp"

namespace py = pybind11;

namespace {

using i64 = std::int64_t;

template <class T>
py::array_t<T> to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(static_cast<py::ssize_t>(holder->size()), holder->data(), owner);
}

}  // namespace

namespace sdm {

void register_python(py::module_& m) {
    // R's generator after set.seed(seed) (Mersenne-Twister, Inversion,
    // Rejection), drawn from step by step: the same numbers as the same calls
    // in R.
    py::class_<nlm::RRng>(m, "RRandom")
        .def(py::init<std::int32_t>(), py::arg("seed"))
        .def("set_seed", &nlm::RRng::set_seed, py::arg("seed"))
        .def("unif_rand", &nlm::RRng::unif_rand)
        .def("exp_rand", &nlm::RRng::exp_rand)
        .def(
            "runif",
            [](nlm::RRng& rng, i64 n, double a, double b) {
                std::vector<double> v(static_cast<std::size_t>(n));
                rng.runif(v.data(), n, a, b);
                return to_array(std::move(v));
            },
            py::arg("n"), py::arg("a") = 0.0, py::arg("b") = 1.0)
        .def(
            "rnorm",
            [](nlm::RRng& rng, i64 n, double mu, double sigma) {
                std::vector<double> v(static_cast<std::size_t>(n));
                rng.rnorm(v.data(), n, mu, sigma);
                return to_array(std::move(v));
            },
            py::arg("n"), py::arg("mu") = 0.0, py::arg("sigma") = 1.0)
        .def(
            "sample_int",
            [](nlm::RRng& rng, i64 n, i64 size, bool replace, std::optional<std::vector<double>> prob) {
                return to_array(rng.sample_int(n, size, replace, prob ? &*prob : nullptr));
            },
            py::arg("n"), py::arg("size"), py::arg("replace") = false, py::arg("prob") = py::none(),
            "sample.int(n, size, replace, prob): 1-based integers")
        .def(
            "unif_index",
            [](nlm::RRng& rng, double dn) { return rng.unif_index(dn); }, py::arg("dn"));
}

}  // namespace sdm
