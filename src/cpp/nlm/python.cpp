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

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"
#include "nlm/rfft.hpp"
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

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// A row-major raster handed to numpy without a copy.
py::array_t<double> raster_to_array(std::vector<double>&& v, i64 nrow, i64 ncol) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({nrow, ncol}, holder->data(), owner);
}

py::array_t<double> py_random_field(std::int32_t seed, i64 nrow, i64 ncol) {
    nlm::RRng rng(seed);
    std::vector<double> v = run_parallel(0, [&] { return nlm::random_field(rng, nrow, ncol); });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_percolation(std::int32_t seed, i64 nrow, i64 ncol, double prob) {
    nlm::RRng rng(seed);
    std::vector<double> v = run_parallel(0, [&] { return nlm::percolation(rng, nrow, ncol, prob); });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_planar_gradient(i64 nrow, i64 ncol, double direction, bool rescale, bool edge) {
    std::vector<double> v = run_parallel(0, [&] {
        std::vector<double> g = nlm::planar_gradient(nrow, ncol, direction);
        const i64 n = nrow * ncol;
        if (edge) {
            // nlm_edgegradient(): a rescaled planar gradient folded around 0.5
            nlm::rescale(g.data(), n);
            nlm::edge_from_planar(g.data(), n);
        }
        if (rescale) nlm::rescale(g.data(), n);
        return g;
    });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_rescale(const CArray<double>& x) {
    py::array_t<double> out(x.request().shape);
    std::copy(x.data(), x.data() + x.size(), out.mutable_data());
    double* p = out.mutable_data();
    const i64 n = static_cast<i64>(x.size());
    run_parallel(0, [&] {
        nlm::rescale(p, n);
        return 0;
    });
    return out;
}

py::array_t<double> py_distance(const CArray<std::uint8_t>& origin, const std::string& method, double res, double xmin,
                             double ymax, double xres, double yres) {
    if (origin.ndim() != 2) throw py::value_error("expected a 2-D origin mask");
    const i64 nrow = origin.shape(0), ncol = origin.shape(1);
    std::vector<std::uint8_t> o(origin.data(), origin.data() + origin.size());
    std::vector<double> v = run_parallel(0, [&] {
        if (method == "gdal") return nlm::distance_gdal(o, nrow, ncol, res);
        if (method == "haversine") return nlm::distance_haversine(o, nrow, ncol, xmin, ymax, xres, yres);
        if (method == "exact") return nlm::distance_exact(o, nrow, ncol, res);
        throw std::invalid_argument("unknown distance method: " + method);
    });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_jenks_breaks(const CArray<double>& sorted, int k) {
    std::vector<double> d(sorted.data(), sorted.data() + sorted.size());
    if (k < 1) throw py::value_error("k must be >= 1");
    std::vector<double> b = run_parallel(0, [&] { return nlm::jenks_breaks(d, k); });
    return vector_to_array(std::move(b));
}

py::array_t<double> py_cut(const CArray<double>& x, const CArray<double>& breaks) {
    py::array_t<double> out(x.request().shape);
    std::vector<double> b(breaks.data(), breaks.data() + breaks.size());
    double* p = out.mutable_data();
    run_parallel(0, [&] {
        nlm::cut_include_lowest(x.data(), static_cast<i64>(x.size()), b, p);
        return 0;
    });
    return out;
}

py::array_t<double> py_mpd(i64 size, const std::vector<double>& rand_dev, std::uint32_t seed, bool torus) {
    std::vector<double> v = run_parallel(0, [&] { return nlm::mpd_square(size, rand_dev, seed, torus); });
    return raster_to_array(std::move(v), size, size);
}

py::array_t<double> py_neigh(i64 nrow, i64 ncol, int n_categories, const std::vector<std::int64_t>& cells_per_cat,
                             int neighbourhood, float p_neigh, float p_empty, std::uint32_t seed) {
    if (static_cast<i64>(cells_per_cat.size()) != n_categories + 1)
        throw py::value_error("cells_per_cat must have n_categories + 1 values");
    std::vector<double> v = run_parallel(0, [&] {
        return nlm::neigh(nrow, ncol, n_categories, cells_per_cat, neighbourhood, p_neigh, p_empty, seed);
    });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_random_rectangular_cluster(i64 nrow, i64 ncol, int minl, int maxl, std::uint32_t seed) {
    std::vector<double> v = run_parallel(0, [&] { return nlm::random_rectangular_cluster(nrow, ncol, minl, maxl, seed); });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_random_cluster(std::int32_t seed, i64 nrow, i64 ncol, double p, const std::vector<double>& ai,
                                      int neighbourhood) {
    nlm::RRng rng(seed);
    std::vector<double> v = run_parallel(0, [&] { return nlm::random_cluster(rng, nrow, ncol, p, ai, neighbourhood); });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_curds(std::int32_t seed, const std::vector<double>& curds_p, const std::vector<std::int64_t>& steps,
                             std::optional<std::vector<double>> wheyes) {
    if (curds_p.size() != steps.size() || (wheyes && wheyes->size() != steps.size()))
        throw py::value_error("curds, recursion_steps and wheyes must have the same length");
    nlm::RRng rng(seed);
    i64 size = 0;
    std::vector<double> v = run_parallel(0, [&] { return nlm::curds(rng, curds_p, steps, wheyes ? &*wheyes : nullptr, size); });
    return raster_to_array(std::move(v), size, size);
}

py::array_t<double> py_gaussian_field(std::int32_t seed, i64 nrow, i64 ncol, double autocorr_range, double mag_var,
                                      double nug, double mean) {
    nlm::RRng rng(seed);
    std::vector<double> v =
        run_parallel(0, [&] { return nlm::gaussian_field(rng, nrow, ncol, autocorr_range, mag_var, nug, mean); });
    return raster_to_array(std::move(v), nrow, ncol);
}

py::array_t<double> py_fbm(std::int32_t seed, i64 nrow, i64 ncol, double fract_dim) {
    nlm::RRng rng(seed);
    std::vector<double> v = run_parallel(0, [&] { return nlm::fbm(rng, nrow, ncol, fract_dim); });
    return raster_to_array(std::move(v), nrow, ncol);
}

// R's fft() of a complex matrix given as real and imaginary parts
// (column-major, R's layout); returns the interleaved result.
py::array_t<double> py_r_fft(const CArray<double>& re, const CArray<double>& im, int nrow, int ncol, bool inverse,
                             bool parallel) {
    const i64 n = static_cast<i64>(nrow) * ncol;
    if (re.size() != n || im.size() != n) throw py::value_error("re and im must have nrow * ncol values");
    std::vector<double> z(2 * static_cast<std::size_t>(n));
    for (i64 k = 0; k < n; ++k) {
        z[2 * static_cast<std::size_t>(k)] = re.data()[k];
        z[2 * static_cast<std::size_t>(k) + 1] = im.data()[k];
    }
    run_parallel(0, [&] {
        nlm::r_fft2(z, nrow, ncol, inverse, parallel);
        return 0;
    });
    return vector_to_array(std::move(z));
}

// R's sum() and cumsum() of doubles (LDOUBLE accumulation).
double py_r_sum(const CArray<double>& x) { return lsm::r_sum(x.data(), static_cast<i64>(x.size())); }

py::array_t<double> py_r_cumsum(const CArray<double>& x) {
    py::array_t<double> out(static_cast<py::ssize_t>(x.size()));
    double* p = out.mutable_data();
    lsm::Extended s = 0.0;
    for (py::ssize_t i = 0; i < x.size(); ++i) {
        s += x.data()[i];
        p[i] = static_cast<double>(s);
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
    m.def("nlm_random", &py_random_field, py::arg("seed"), py::arg("nrow"), py::arg("ncol"), "nlm_random() before rescaling.");
    m.def("nlm_percolation", &py_percolation, py::arg("seed"), py::arg("nrow"), py::arg("ncol"), py::arg("prob"),
          "nlm_percolation() (1/0).");
    m.def("nlm_planar_gradient", &py_planar_gradient, py::arg("nrow"), py::arg("ncol"), py::arg("direction"),
          py::arg("rescale"), py::arg("edge") = false, "nlm_planargradient() / nlm_edgegradient().");
    m.def("nlm_rescale", &py_rescale, py::arg("x"), "NLMR's util_rescale() (a copy).");
    m.def("nlm_distance", &py_distance, py::arg("origin"), py::arg("method"), py::arg("res") = 1.0, py::arg("xmin") = 0.0,
          py::arg("ymax") = 0.0, py::arg("xres") = 1.0, py::arg("yres") = 1.0,
          "Distance to the origin cells: 'gdal' / 'haversine' (terra::distance) or 'exact'.");
    m.def("nlm_jenks_breaks", &py_jenks_breaks, py::arg("sorted"), py::arg("k"), "NLMR's rcpp_get_jenksbreaks().");
    m.def("nlm_mpd", &py_mpd, py::arg("size"), py::arg("rand_dev"), py::arg("seed"), py::arg("torus"),
          "rcpp_mpd()'s diamond-square map (size x size).");
    m.def("nlm_neigh", &py_neigh, py::arg("nrow"), py::arg("ncol"), py::arg("n_categories"), py::arg("cells_per_cat"),
          py::arg("neighbourhood"), py::arg("p_neigh"), py::arg("p_empty"), py::arg("seed"), "rcpp_neigh().");
    m.def("nlm_random_rectangular_cluster", &py_random_rectangular_cluster, py::arg("nrow"), py::arg("ncol"),
          py::arg("minl"), py::arg("maxl"), py::arg("seed"), "rcpp_randomrectangularcluster().");
    m.def("nlm_random_cluster", &py_random_cluster, py::arg("seed"), py::arg("nrow"), py::arg("ncol"), py::arg("p"),
          py::arg("ai"), py::arg("neighbourhood"), "nlm_randomcluster() before rescaling.");
    m.def("nlm_curds", &py_curds, py::arg("seed"), py::arg("curds"), py::arg("steps"), py::arg("wheyes") = py::none(),
          "nlm_curds() (1 curds, 0 matrix).");
    m.def(
        "nlm_r_math",
        [](const std::string& fn, const CArray<double>& x, double y) {
            py::array_t<double> out(x.request().shape);
            double* p = out.mutable_data();
            const double* v = x.data();
            for (py::ssize_t i = 0; i < x.size(); ++i) {
                if (fn == "exp")
                    p[i] = lsm::r_exp(v[i]);
                else if (fn == "pow")
                    p[i] = lsm::r_pow(v[i], y);
                else if (fn == "sqrt")
                    p[i] = lsm::r_sqrt(v[i]);
                else
                    throw py::value_error("unknown function: " + fn);
            }
            return out;
        },
        py::arg("fn"), py::arg("x"), py::arg("y") = 0.0, "R's exp(x), x^y or vector sqrt(x) on x86-64 Windows.");
    m.def("nlm_gaussian_field", &py_gaussian_field, py::arg("seed"), py::arg("nrow"), py::arg("ncol"),
          py::arg("autocorr_range"), py::arg("mag_var"), py::arg("nug"), py::arg("mean"),
          "NLMR's simulate_gaussian_field().");
    m.def("nlm_fbm", &py_fbm, py::arg("seed"), py::arg("nrow"), py::arg("ncol"), py::arg("fract_dim"),
          "NLMR's simulate_fbm().");
    m.def("nlm_r_fft", &py_r_fft, py::arg("re"), py::arg("im"), py::arg("nrow"), py::arg("ncol"),
          py::arg("inverse") = false, py::arg("parallel") = true, "R's fft() of a complex matrix (interleaved result).");
    m.def("nlm_r_sum", &py_r_sum, py::arg("x"), "R's sum() of doubles (80-bit accumulation).");
    m.def("nlm_r_cumsum", &py_r_cumsum, py::arg("x"), "R's cumsum() of doubles (80-bit accumulation).");
    m.def("nlm_cut", &py_cut, py::arg("x"), py::arg("breaks"),
          "base::cut(x, breaks, include.lowest = TRUE, labels = FALSE) (NaN outside).");
}

}  // namespace nlm
