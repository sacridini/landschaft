// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the neutral landscape models (landschaft._core.nlm_*).

#include "nlm/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "nlm/rng.hpp"

namespace py = pybind11;

namespace {

using nlm::i64;

// Runs `f` with the GIL released and, if n_threads > 0, with that many
// OpenMP threads for parallel regions started by this thread.
template <class F>
auto run_parallel(int n_threads, F&& f) {
    py::gil_scoped_release release;
#if defined(_OPENMP)
    const int previous = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(n_threads);
    struct Restore {
        int n;
        ~Restore() { omp_set_num_threads(n); }
    } restore{previous};
#else
    (void)n_threads;
#endif
    return f();
}

template <class T>
py::array_t<T> vector_to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

// A sequence of draws from R's generator after set.seed(seed): each
// operation is a tuple ("runif", n, a, b), ("rnorm", n, mu, sigma),
// ("sample", n, size, replace, prob or None) or ("unif_index", n, dn).
py::list r_sequence(std::int32_t seed, const py::list& ops, int n_threads) {
    nlm::RRng rng(seed);
    py::list out;
    for (const py::handle& h : ops) {
        const py::tuple op = h.cast<py::tuple>();
        const std::string kind = op[0].cast<std::string>();
        if (kind == "runif" || kind == "rnorm") {
            const i64 n = op[1].cast<i64>();
            const double a = op[2].cast<double>(), b = op[3].cast<double>();
            std::vector<double> v(static_cast<std::size_t>(n));
            run_parallel(n_threads, [&] {
                if (kind == "runif")
                    rng.runif(v.data(), n, a, b);
                else
                    rng.rnorm(v.data(), n, a, b);
                return 0;
            });
            out.append(vector_to_array(std::move(v)));
        } else if (kind == "sample") {
            const i64 n = op[1].cast<i64>(), size = op[2].cast<i64>();
            const bool replace = op[3].cast<bool>();
            std::optional<std::vector<double>> prob;
            if (!op[4].is_none()) prob = op[4].cast<std::vector<double>>();
            std::vector<i64> v = rng.sample_int(n, size, replace, prob ? &*prob : nullptr);
            out.append(vector_to_array(std::move(v)));
        } else if (kind == "unif_index") {
            const i64 n = op[1].cast<i64>();
            const double dn = op[2].cast<double>();
            std::vector<double> v(static_cast<std::size_t>(n));
            for (auto& x : v) x = rng.unif_index(dn);
            out.append(vector_to_array(std::move(v)));
        } else {
            throw py::value_error("unknown operation: " + kind);
        }
    }
    return out;
}

// A sequence of draws from std::mt19937(seed) with libstdc++'s
// distributions: ("canonical", n), ("uniform_int", n, a, b),
// ("normal", n, mean, sd) (one distribution object), ("normal_fresh", n,
// mean, sd) (a new distribution per draw, as NLMR's mpd) or ("shuffle", n)
// (the permutation of 0..n-1).
py::list std_sequence(std::uint32_t seed, const py::list& ops) {
    nlm::StdMt19937 g(seed);
    py::list out;
    for (const py::handle& h : ops) {
        const py::tuple op = h.cast<py::tuple>();
        const std::string kind = op[0].cast<std::string>();
        const i64 n = op[1].cast<i64>();
        if (kind == "canonical") {
            std::vector<double> v(static_cast<std::size_t>(n));
            for (auto& x : v) x = nlm::std_canonical(g);
            out.append(vector_to_array(std::move(v)));
        } else if (kind == "uniform_int") {
            const i64 a = op[2].cast<i64>(), b = op[3].cast<i64>();
            std::vector<i64> v(static_cast<std::size_t>(n));
            for (auto& x : v) x = nlm::std_uniform_int(g, a, b);
            out.append(vector_to_array(std::move(v)));
        } else if (kind == "normal" || kind == "normal_fresh") {
            const double mean = op[2].cast<double>(), sd = op[3].cast<double>();
            std::vector<double> v(static_cast<std::size_t>(n));
            nlm::StdNormal dist(mean, sd);
            for (auto& x : v) {
                if (kind == "normal_fresh") dist = nlm::StdNormal(mean, sd);
                x = dist(g);
            }
            out.append(vector_to_array(std::move(v)));
        } else if (kind == "shuffle") {
            std::vector<i64> v(static_cast<std::size_t>(n));
            for (i64 i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i;
            nlm::std_shuffle(v.data(), n, g);
            out.append(vector_to_array(std::move(v)));
        } else {
            throw py::value_error("unknown operation: " + kind);
        }
    }
    return out;
}

}  // namespace

namespace nlm {

void register_python(py::module_& m) {
    m.def("nlm_r_sequence", &r_sequence, py::arg("seed"), py::arg("ops"), py::arg("n_threads") = 0,
          "Draws from R's generator after set.seed(seed) (Mersenne-Twister, Inversion, Rejection).");
    m.def("nlm_std_sequence", &std_sequence, py::arg("seed"), py::arg("ops"),
          "Draws from std::mt19937(seed) with libstdc++'s distributions.");
}

}  // namespace nlm
