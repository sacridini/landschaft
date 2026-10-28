// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the statistics of maps (landschaft._core.stats_*).

#include "stats/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "stats/special.hpp"
#include "stats/stats.hpp"

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

stats::Lattice lattice_of(const F64& map, bool queen) {
    if (map.ndim() != 2) throw std::invalid_argument("expected a 2-D map");
    return stats::lattice(map.data(), map.shape(0), map.shape(1), queen);
}

// values of the lattice's cells on the map (NaN elsewhere)
F64 on_map(const F64& map, const stats::Lattice& l, const std::vector<double>& v) {
    F64 out({map.shape(0), map.shape(1)});
    double* o = out.mutable_data();
    for (py::ssize_t i = 0; i < map.size(); ++i) o[i] = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 0; k < l.cells.size(); ++k) o[l.cells[k]] = v[k];
    return out;
}

std::vector<double> values(const F64& map, const stats::Lattice& l) {
    std::vector<double> y(l.cells.size());
    for (std::size_t k = 0; k < l.cells.size(); ++k) y[k] = map.data()[l.cells[k]];
    return y;
}

}  // namespace

namespace stats {

void register_python(py::module_& m) {
    m.def("stats_ndtr", [](const F64& x) {
        F64 out(x.request().shape);
        for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = ndtr(x.data()[i]);
        return out;
    }, py::arg("x"), "scipy's normal distribution function (Cephes' ndtr), element by element.");

    m.def("stats_ndtri", [](const F64& x) {
        F64 out(x.request().shape);
        for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = ndtri(x.data()[i]);
        return out;
    }, py::arg("x"), "scipy's normal quantile (Cephes' ndtri), element by element.");

    m.def("stats_mann_kendall", [](const F64& series, double alpha) {
        const MannKendall r = mann_kendall(series.data(), static_cast<int>(series.size()), alpha);
        return std::vector<double>{r.s, r.var_s, r.z, r.p, r.tau, r.slope, r.intercept, r.h, r.trend};
    }, py::arg("series"), py::arg("alpha"),
       "pymannkendall's original_test of a series: s, var_s, z, p, tau, slope, intercept, h, trend.");

    m.def("stats_trend", [](const F64& layers, double alpha, int min_values, int threads) {
        if (layers.ndim() != 2) throw std::invalid_argument("layers must be dates x cells");
        const int t = static_cast<int>(layers.shape(0));
        const i64 n = static_cast<i64>(layers.shape(1));
        F64 out({static_cast<py::ssize_t>(6), static_cast<py::ssize_t>(n)});
        {
            py::gil_scoped_release release;
            trend(layers.data(), t, n, alpha, min_values, out.mutable_data(), threads);
        }
        return out;
    }, py::arg("layers"), py::arg("alpha"), py::arg("min_values"), py::arg("threads"),
       "Every cell's Mann-Kendall test and Sen's slope: 6 x cells (slope, intercept, p, z, tau, trend).");

    m.def("stats_moran", [](const F64& map, bool queen, int perms, std::uint64_t seed, int threads) {
        std::vector<double> out;
        {
            py::gil_scoped_release release;
            const Lattice l = lattice_of(map, queen);
            if (l.size() < 4) throw std::invalid_argument("four cells with a value at least");
            const std::vector<double> y = values(map, l);
            const GlobalMoran g = global_moran(y.data(), l);
            const double p_sim = moran_p_sim(y.data(), l, g.i, perms, seed, threads);
            out = {g.i, g.ei, g.vi_norm, g.z_norm, g.p_norm, g.vi_rand, g.z_rand, g.p_rand, p_sim,
                   static_cast<double>(l.size())};
        }
        return out;
    }, py::arg("map"), py::arg("queen"), py::arg("permutations"), py::arg("seed"), py::arg("threads"),
       "Global Moran's I: I, EI, VI_norm, z_norm, p_norm, VI_rand, z_rand, p_rand, p_sim, n.");

    m.def("stats_lisa", [](const F64& map, bool queen, int perms, std::uint64_t seed, int threads) {
        const Lattice l = lattice_of(map, queen);
        if (l.size() < 4) throw std::invalid_argument("four cells with a value at least");
        const std::vector<double> y = values(map, l);
        const std::size_t n = l.cells.size();
        std::vector<double> li(n), q(n), ei(n), vi(n), z(n), p(n);
        {
            py::gil_scoped_release release;
            local_moran(y.data(), l, perms, seed, li.data(), q.data(), ei.data(), vi.data(), z.data(), p.data(),
                        threads);
        }
        return py::make_tuple(on_map(map, l, li), on_map(map, l, q), on_map(map, l, ei), on_map(map, l, vi),
                              on_map(map, l, z), on_map(map, l, p));
    }, py::arg("map"), py::arg("queen"), py::arg("permutations"), py::arg("seed"), py::arg("threads"),
       "Local Moran's I maps: I, quadrant, EI, VI, z, pseudo p.");

    m.def("stats_getis_ord", [](const F64& map, bool queen, int threads) {
        const Lattice l = lattice_of(map, queen);
        if (l.size() < 3) throw std::invalid_argument("three cells with a value at least");
        const std::vector<double> y = values(map, l);
        const std::size_t n = l.cells.size();
        std::vector<double> g(n), z(n), p(n), s(n);
        {
            py::gil_scoped_release release;
            getis_ord(y.data(), l, g.data(), z.data(), p.data(), s.data(), threads);
        }
        return py::make_tuple(on_map(map, l, g), on_map(map, l, z), on_map(map, l, p), on_map(map, l, s));
    }, py::arg("map"), py::arg("queen"), py::arg("threads"), "Getis and Ord's Gi* maps: G, z, p, spots.");

    m.def("stats_numpy_choice", [](std::uint64_t seed, std::uint64_t pop, std::uint64_t size, int draws) {
        NumpyRng rng(seed);
        std::vector<std::uint64_t> out = rng.choice_distinct(pop, size);
        std::vector<std::uint64_t> after;
        for (int k = 0; k < draws; ++k) after.push_back(rng.integers(1000));
        return py::make_tuple(out, after);
    }, py::arg("seed"), py::arg("pop"), py::arg("size"), py::arg("draws") = 1,
       "numpy's default_rng(seed).choice(pop, size, replace=False) and integers(1000) after it (for tests).");
}

}  // namespace stats
