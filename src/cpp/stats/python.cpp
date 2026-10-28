// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the statistics of maps (landschaft._core.stats_*).

#include "stats/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include "stats/interp.hpp"
#include "stats/raster.hpp"
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

std::int64_t same_size(std::initializer_list<const F64*> arrays) {
    const py::ssize_t n = (*arrays.begin())->size();
    for (const F64* a : arrays)
        if (a->size() != n) throw std::invalid_argument("the coordinates and values differ in length");
    return static_cast<std::int64_t>(n);
}

stats::VarioModel model_of(int shape, double nugget, double psill, double range) {
    if (shape < 0 || shape > 2) throw std::invalid_argument("shape: 0 spherical, 1 exponential, 2 gaussian");
    return {static_cast<stats::Shape>(shape), nugget, psill, range};
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

    m.def("stats_default_cutoff", [](const F64& x, const F64& y) {
        return default_cutoff(x.data(), y.data(), same_size({&x, &y}));
    }, py::arg("x"), py::arg("y"), "gstat's default cutoff: a third of the diagonal of the points' bounding box.");

    m.def("stats_variogram", [](const F64& x, const F64& y, const F64& z, double cutoff, double width) {
        const SampleVariogram s = sample_variogram(x.data(), y.data(), z.data(), same_size({&x, &y, &z}), cutoff,
                                                   width);
        return py::make_tuple(F64(static_cast<py::ssize_t>(s.np.size()), s.np.data()),
                              F64(static_cast<py::ssize_t>(s.dist.size()), s.dist.data()),
                              F64(static_cast<py::ssize_t>(s.gamma.size()), s.gamma.data()));
    }, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("cutoff"), py::arg("width"),
       "gstat's sample variogram: np, dist, gamma of the classes with pairs.");

    m.def("stats_fit_variogram", [](const F64& np_, const F64& dist, const F64& gamma, int shape, double nugget,
                                    double psill, double range, bool fit_nugget, bool fit_psill, bool fit_range) {
        SampleVariogram s;
        const std::int64_t n = same_size({&np_, &dist, &gamma});
        s.np.assign(np_.data(), np_.data() + n);
        s.dist.assign(dist.data(), dist.data() + n);
        s.gamma.assign(gamma.data(), gamma.data() + n);
        const VarioFit f = fit_variogram(s, model_of(shape, nugget, psill, range), fit_nugget, fit_psill, fit_range);
        return py::make_tuple(f.model.nugget, f.model.psill, f.model.range, f.sserr, f.singular, f.converged);
    }, py::arg("np"), py::arg("dist"), py::arg("gamma"), py::arg("shape"), py::arg("nugget"), py::arg("psill"),
       py::arg("range"), py::arg("fit_nugget"), py::arg("fit_psill"), py::arg("fit_range"),
       "gstat's fit.variogram: nugget, psill, range, sserr, singular, converged.");

    m.def("stats_vario_gamma", [](const F64& h, int shape, double nugget, double psill, double range) {
        const VarioModel md = model_of(shape, nugget, psill, range);
        F64 out(h.request().shape);
        for (py::ssize_t i = 0; i < h.size(); ++i) out.mutable_data()[i] = md.gamma(h.data()[i]);
        return out;
    }, py::arg("h"), py::arg("shape"), py::arg("nugget"), py::arg("psill"), py::arg("range"),
       "A variogram model's semivariance at distances.");

    m.def("stats_idw", [](const F64& x, const F64& y, const F64& z, const F64& px, const F64& py_, double power,
                          std::int64_t nmax, int threads) {
        const std::int64_t n = same_size({&x, &y, &z});
        const std::int64_t m_ = same_size({&px, &py_});
        F64 out(static_cast<py::ssize_t>(m_));
        {
            py::gil_scoped_release release;
            idw(x.data(), y.data(), z.data(), n, px.data(), py_.data(), m_, power, nmax, out.mutable_data(),
                threads);
        }
        return out;
    }, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("px"), py::arg("py"), py::arg("power"), py::arg("nmax"),
       py::arg("threads"), "gstat's inverse distance weighting at places (nmax <= 0: every point).");

    m.def("stats_krige", [](const F64& x, const F64& y, const F64& z, int shape, double nugget, double psill,
                            double range, const F64& px, const F64& py_, std::int64_t nmax, int threads) {
        const std::int64_t n = same_size({&x, &y, &z});
        const std::int64_t m_ = same_size({&px, &py_});
        const VarioModel md = model_of(shape, nugget, psill, range);
        F64 pred(static_cast<py::ssize_t>(m_)), var(static_cast<py::ssize_t>(m_));
        bool ok;
        {
            py::gil_scoped_release release;
            ok = krige(x.data(), y.data(), z.data(), n, md, px.data(), py_.data(), m_, nmax, pred.mutable_data(),
                       var.mutable_data(), threads);
        }
        if (!ok)
            throw std::invalid_argument(
                "the covariances of the points are singular (points at one place, or no nugget among neighbours)");
        return py::make_tuple(pred, var);
    }, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("shape"), py::arg("nugget"), py::arg("psill"),
       py::arg("range"), py::arg("px"), py::arg("py"), py::arg("nmax"), py::arg("threads"),
       "gstat's ordinary kriging at places: prediction, variance (nmax <= 0: every point).");

    m.def("stats_cell_stats", [](const F64& layers, int stat, double q, int threads) {
        if (layers.ndim() != 2) throw std::invalid_argument("layers must be layers x cells");
        const std::int64_t k = layers.shape(0), n = layers.shape(1);
        F64 out(static_cast<py::ssize_t>(n));
        {
            py::gil_scoped_release release;
            cell_stats(layers.data(), k, n, stat, q, out.mutable_data(), threads);
        }
        return out;
    }, py::arg("layers"), py::arg("stat"), py::arg("q"), py::arg("threads"),
       "A statistic of the layers at every cell (codes of io_zonal_stats; 14 argmax, 15 argmin).");

    m.def("stats_local_correlation", [](const F64& x, const F64& y, int radius, bool circle, int min_cells,
                                        int threads) {
        if (x.ndim() != 2 || y.ndim() != 2 || x.shape(0) != y.shape(0) || x.shape(1) != y.shape(1))
            throw std::invalid_argument("x and y: 2-D maps of one shape");
        const int ny = static_cast<int>(x.shape(0)), nx = static_cast<int>(x.shape(1));
        F64 out({static_cast<py::ssize_t>(4), static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
        {
            py::gil_scoped_release release;
            local_correlation(x.data(), y.data(), ny, nx, radius, circle, min_cells, out.mutable_data(), threads);
        }
        return out;
    }, py::arg("x"), py::arg("y"), py::arg("radius"), py::arg("circle"), py::arg("min_cells"), py::arg("threads"),
       "Correlation, slope, intercept and r2 of y on x in a moving window: 4 x ny x nx.");

    m.def("stats_viewshed", [](const F64& dem, double rw, double rh,
                               const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& rows,
                               const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& cols,
                               double eye, double target, double reach, bool curvature, int threads) {
        if (dem.ndim() != 2) throw std::invalid_argument("expected a 2-D map");
        if (rows.size() != cols.size()) throw std::invalid_argument("rows and cols differ in length");
        const int ny = static_cast<int>(dem.shape(0)), nx = static_cast<int>(dem.shape(1));
        F64 out({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
        {
            py::gil_scoped_release release;
            viewshed(dem.data(), ny, nx, rw, rh, rows.data(), cols.data(), rows.size(), eye, target, reach, curvature,
                     out.mutable_data(), threads);
        }
        return out;
    }, py::arg("dem"), py::arg("rw"), py::arg("rh"), py::arg("rows"), py::arg("cols"), py::arg("eye"),
       py::arg("target"), py::arg("reach"), py::arg("curvature"), py::arg("threads"),
       "How many observers see every cell (Franklin and Ray's R2).");

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
