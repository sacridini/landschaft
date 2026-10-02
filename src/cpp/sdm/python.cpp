// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of ls.sdm (landschaft._core.sdm_*) and of R's generator.

#include "sdm/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <algorithm>
#include <optional>
#include <string>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlm/rng.hpp"
#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "sdm/clean.hpp"
#include "sdm/collinear.hpp"
#include "sdm/extrapolation.hpp"
#include "sdm/geocircle/geocircle.hpp"
#include "sdm/rstats.hpp"
#include "sdm/kmeans.hpp"
#include "sdm/rformat.hpp"
#include "sdm/thin.hpp"
#include "sdm/variogram.hpp"

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

namespace {

using Vec = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple py_thin_spthin(const Vec& lon, const Vec& lat, double max_km, int reps, std::int32_t seed) {
    if (lon.size() != lat.size()) throw std::invalid_argument("lon and lat differ in length");
    if (reps < 0) throw std::invalid_argument("reps must be >= 0");
    const i64 n = static_cast<i64>(lon.size());
    ThinResult r;
    {
        py::gil_scoped_release release;
        r = thin_spthin(lon.data(), lat.data(), n, max_km, reps, seed);
    }
    py::array_t<std::uint8_t> keep({static_cast<py::ssize_t>(reps), static_cast<py::ssize_t>(n)});
    std::copy(r.keep.begin(), r.keep.end(), keep.mutable_data());
    return py::make_tuple(keep, to_array(std::move(r.kept)));
}

py::tuple py_earth_neighbours(const Vec& lon, const Vec& lat, double max_km) {
    if (lon.size() != lat.size()) throw std::invalid_argument("lon and lat differ in length");
    Neighbours nb;
    {
        py::gil_scoped_release release;
        nb = earth_neighbours(lon.data(), lat.data(), static_cast<i64>(lon.size()), max_km);
    }
    return py::make_tuple(to_array(std::move(nb.start)), to_array(std::move(nb.index)));
}

py::array_t<double> py_outlier_statistic(const Vec& lon, const Vec& lat, bool use_min) {
    if (lon.size() != lat.size()) throw std::invalid_argument("lon and lat differ in length");
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = outlier_statistic(lon.data(), lat.data(), static_cast<i64>(lon.size()), use_min);
    }
    return to_array(std::move(out));
}

py::array_t<double> py_outlier_statistic_weighted(const Vec& lon, const Vec& lat, const Vec& count, double within,
                                                  bool use_min) {
    if (lon.size() != lat.size() || lon.size() != count.size()) throw std::invalid_argument("lengths differ");
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = outlier_statistic_weighted(lon.data(), lat.data(), count.data(), static_cast<i64>(lon.size()), within,
                                         use_min);
    }
    return to_array(std::move(out));
}

// VIF of some columns of a table (observations x variables).
py::array_t<double> py_vif(const py::array_t<double, py::array::c_style | py::array::forcecast>& table,
                           const std::vector<i64>& which) {
    if (table.ndim() != 2) throw std::invalid_argument("expected a 2-D table (observations x variables)");
    const i64 n = static_cast<i64>(table.shape(0)), m = static_cast<i64>(table.shape(1));
    for (const i64 w : which)
        if (w < 0 || w >= m) throw std::invalid_argument("column out of range");
    std::vector<double> col(static_cast<std::size_t>(n * m));  // column-major, as R
    for (i64 i = 0; i < n; ++i)
        for (i64 j = 0; j < m; ++j) col[static_cast<std::size_t>(j * n + i)] = table.data()[i * m + j];
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = vif(col.data(), n, m, which);
    }
    return to_array(std::move(out));
}

// The circles of radius d around each point: for each point a list of closed rings (n x 2 arrays).
py::list py_circles(const Vec& lon, const Vec& lat, double d, const std::string& kind) {
    if (lon.size() != lat.size()) throw std::invalid_argument("lon and lat differ in length");
    const i64 n = static_cast<i64>(lon.size());
    std::vector<GeoParts> parts(static_cast<std::size_t>(n));
    {
        py::gil_scoped_release release;
        for (i64 i = 0; i < n; ++i) {
            const double x = lon.data()[i], y = lat.data()[i];
            GeoParts& p = parts[static_cast<std::size_t>(i)];
            if (kind == "geosphere") {
                p = geod_polys_fixed(x, y, d);
            } else if (kind == "terra") {
                p = terra_point_buffer(x, y, d);
            } else if (kind == "planar") {
                p = {geos_circle(x, y, d)};
            } else {
                throw std::invalid_argument("kind must be 'geosphere', 'terra' or 'planar'");
            }
        }
    }
    py::list out;
    for (const GeoParts& p : parts) {
        py::list rings;
        for (const GeoRing& r : p) {
            py::array_t<double> a({static_cast<py::ssize_t>(r.size() / 2), static_cast<py::ssize_t>(2)});
            std::copy(r.begin(), r.end(), a.mutable_data());
            rings.append(a);
        }
        out.append(rings);
    }
    return out;
}

py::dict kmeans_dict(KmeansResult&& r, i64 p) {
    const i64 k = static_cast<i64>(r.size.size());
    py::array_t<double> centers({static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(p)});
    double* c = centers.mutable_data();
    for (i64 l = 0; l < k; ++l)
        for (i64 j = 0; j < p; ++j) c[l * p + j] = r.centers[static_cast<std::size_t>(j * k + l)];
    py::dict out;
    out["cluster"] = to_array(std::move(r.cluster));
    out["centers"] = centers;
    out["totss"] = r.totss;
    out["withinss"] = to_array(std::move(r.withinss));
    out["tot_withinss"] = r.tot_withinss;
    out["betweenss"] = r.betweenss;
    out["size"] = to_array(std::move(r.size));
    out["iter"] = r.iter;
    out["ifault"] = r.ifault < 0 ? py::object(py::none()) : py::object(py::int_(r.ifault));
    out["warnings"] = r.warnings;
    return out;
}

KmeansAlgorithm kmeans_algorithm(const std::string& name) {
    if (name == "Hartigan-Wong") return KmeansAlgorithm::HartiganWong;
    if (name == "Lloyd" || name == "Forgy") return KmeansAlgorithm::Lloyd;
    if (name == "MacQueen") return KmeansAlgorithm::MacQueen;
    throw std::invalid_argument("algorithm must be one of 'Hartigan-Wong', 'Lloyd', 'Forgy', 'MacQueen'");
}

// R's kmeans() of a table (observations x variables). The starts come from
// `rng` when given (its state advances as R's does), else from set.seed(seed);
// `centers` (k x p) replaces the random starts.
py::dict py_kmeans(const py::array_t<double, py::array::c_style | py::array::forcecast>& x, std::optional<int> k,
                   int iter_max, int nstart, std::optional<std::int32_t> seed, const std::string& algorithm,
                   nlm::RRng* rng,
                   std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> centers) {
    if (x.ndim() != 2) throw std::invalid_argument("expected a 2-D table (observations x variables)");
    const i64 m = static_cast<i64>(x.shape(0)), p = static_cast<i64>(x.shape(1));
    const KmeansAlgorithm alg = kmeans_algorithm(algorithm);
    std::vector<double> col(static_cast<std::size_t>(m * p));  // column-major, as R
    for (i64 i = 0; i < m; ++i)
        for (i64 j = 0; j < p; ++j) col[static_cast<std::size_t>(j * m + i)] = x.data()[i * p + j];
    KmeansResult r;
    if (centers) {
        if (centers->ndim() != 2 || centers->shape(1) != p)
            throw std::invalid_argument("must have same number of columns in 'x' and 'centers'");
        const i64 kc = static_cast<i64>(centers->shape(0));
        std::vector<double> ccol(static_cast<std::size_t>(kc * p));
        for (i64 i = 0; i < kc; ++i)
            for (i64 j = 0; j < p; ++j) ccol[static_cast<std::size_t>(j * kc + i)] = centers->data()[i * p + j];
        py::gil_scoped_release release;
        r = kmeans(col.data(), m, p, ccol.data(), static_cast<int>(kc), iter_max, alg);
    } else {
        if (!k) throw std::invalid_argument("'centers' must be a number or a matrix");
        if (!rng && !seed) throw std::invalid_argument("give a seed or an RRandom generator");
        nlm::RRng own(seed.value_or(0));
        nlm::RRng& g = rng ? *rng : own;
        py::gil_scoped_release release;
        r = kmeans(col.data(), m, p, *k, iter_max, nstart, alg, g);
    }
    return kmeans_dict(std::move(r), p);
}

}  // namespace

void register_python(py::module_& m) {
    register_maxnet_python(m);
    m.def("sdm_circles", &py_circles, py::arg("lon"), py::arg("lat"), py::arg("d"), py::arg("kind"),
          "Circles of radius d around points as CoordinateCleaner draws them: 'geosphere' (73 vertices, "
          "corrected at poles and the antimeridian), 'terra' (terra::buffer), 'planar' (GEOS buffer)");
    m.def("sdm_vif", &py_vif, py::arg("table"), py::arg("which"),
          "usdm's VIF (1 / (1 - R^2), R's lm and summary.lm) of the columns `which`");
    m.def("sdm_outlier_statistic", &py_outlier_statistic, py::arg("lon"), py::arg("lat"), py::arg("use_min"),
          "cc_outl: mean (R's 80-bit mean) or minimum haversine km to the other records at other locations");
    m.def("sdm_outlier_statistic_weighted", &py_outlier_statistic_weighted, py::arg("lon"), py::arg("lat"),
          py::arg("count"), py::arg("within"), py::arg("use_min"), "cc_outl's raster variant with weights");
    m.def(
        "sdm_haversine_km",
        [](double lon1, double lat1, double lon2, double lat2) { return haversine_km(lon1, lat1, lon2, lat2); },
        py::arg("lon1"), py::arg("lat1"), py::arg("lon2"), py::arg("lat2"));
    m.def(
        "sdm_r_cos",
        [](const Vec& x) {
            std::vector<double> out(static_cast<std::size_t>(x.size()));
            for (py::ssize_t i = 0; i < x.size(); ++i) out[static_cast<std::size_t>(i)] = lsm::r_cos(x.data()[i]);
            return to_array(std::move(out));
        },
        py::arg("x"), "R's cos() of a vector (x87 fcos on Windows)");
    m.def(
        "sdm_knn_mean_distance",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& ref,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& g, i64 k, bool squared) {
            if (ref.ndim() != 2 || g.ndim() != 2 || ref.shape(1) != g.shape(1))
                throw std::invalid_argument("expected two tables with the same columns");
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = knn_mean_distance(ref.data(), ref.shape(0), g.data(), g.shape(0), g.shape(1), k, squared);
            }
            return to_array(std::move(out));
        },
        py::arg("ref"), py::arg("g"), py::arg("k"), py::arg("squared") = false,
        "mop: mean of the k smallest distances of each row of g to the rows of ref");
    m.def(
        "sdm_r_colmeans",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& x) {
            if (x.ndim() != 2) throw std::invalid_argument("expected a 2-D table");
            const i64 n = x.shape(0), p = x.shape(1);
            std::vector<double> out(static_cast<std::size_t>(p));
            for (i64 j = 0; j < p; ++j) {  // colMeans: one 80-bit pass, divided by n in 80 bits
                lsm::Extended s = 0.0;
                for (i64 i = 0; i < n; ++i) s += x.data()[i * p + j];
                out[static_cast<std::size_t>(j)] = static_cast<double>(s / lsm::Extended(n));
            }
            return to_array(std::move(out));
        },
        py::arg("x"), "R's colMeans()");
    m.def(
        "sdm_sample_variogram",
        [](const Vec& x, const Vec& y, const Vec& z, const std::vector<double>& bounds, bool lonlat) {
            if (x.size() != y.size() || x.size() != z.size()) throw std::invalid_argument("lengths differ");
            SampleVariogram v;
            {
                py::gil_scoped_release release;
                v = sample_variogram(x.data(), y.data(), z.data(), static_cast<i64>(x.size()), bounds, lonlat);
            }
            return py::make_tuple(to_array(std::move(v.np)), to_array(std::move(v.dist)), to_array(std::move(v.gamma)));
        },
        py::arg("x"), py::arg("y"), py::arg("z"), py::arg("bounds"), py::arg("lonlat"),
        "gstat's sample variogram with given class bounds: (np, mean distance, semivariance) per class");
    m.def(
        "sdm_fit_variogram",
        [](const std::string& model, double kappa, const Vec& dist, const Vec& gamma, const Vec& np, double nugget,
           double psill, double range) -> py::object {
            const VariogramFit f = fit_variogram(model.c_str(), kappa, dist.data(), gamma.data(), np.data(),
                                                 static_cast<i64>(dist.size()), nugget, psill, range);
            if (!f.ok) return py::none();
            return py::make_tuple(f.nugget, f.psill, f.range, f.sserr);
        },
        py::arg("model"), py::arg("kappa"), py::arg("dist"), py::arg("gamma"), py::arg("np"), py::arg("nugget"),
        py::arg("psill"), py::arg("range"), "gstat's fit.variogram (method 7) of a nugget plus one model");
    m.def(
        "sdm_r_cor2",
        [](const Vec& x, const Vec& y, bool kendall) {
            if (x.size() != y.size()) throw std::invalid_argument("x and y differ in length");
            return r_cor2(x.data(), y.data(), static_cast<i64>(x.size()), kendall);
        },
        py::arg("x"), py::arg("y"), py::arg("kendall") = false, "R's cor(x, y) of two vectors (cov_na_2)");
    m.def(
        "sdm_r_sd",
        [](const Vec& x) {
            const i64 n = static_cast<i64>(x.size());
            return lsm::r_sd(x.data(), n, lsm::r_mean(x.data(), n));
        },
        py::arg("x"), "R's sd() (80-bit mean and sum of squares)");
    m.def(
        "sdm_r_mean", [](const Vec& x) { return lsm::r_mean(x.data(), static_cast<i64>(x.size())); }, py::arg("x"),
        "R's mean() (80-bit accumulation)");
    m.def("sdm_kmeans", &py_kmeans, py::arg("x"), py::arg("k") = py::none(), py::arg("iter_max") = 10,
          py::arg("nstart") = 1, py::arg("seed") = py::none(), py::arg("algorithm") = "Hartigan-Wong",
          py::arg("rng") = py::none(), py::arg("centers") = py::none(),
          "R's stats::kmeans (Hartigan-Wong, Lloyd/Forgy, MacQueen): dict of cluster (1-based), centers, totss, "
          "withinss, tot_withinss, betweenss, size, iter, ifault, warnings");
    m.def(
        "sdm_r_as_character",
        [](const Vec& x) {
            std::vector<std::string> out(static_cast<std::size_t>(x.size()));
            for (py::ssize_t i = 0; i < x.size(); ++i) out[static_cast<std::size_t>(i)] = r_as_character(x.data()[i]);
            return out;
        },
        py::arg("x"), "R's as.character() of doubles (R 4.4.2 on Windows: formatReal in long double, trio's printing)");
    m.def(
        "sdm_r_unique_rows",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& x) {
            if (x.ndim() != 2) throw std::invalid_argument("expected a 2-D array");
            const i64 m = static_cast<i64>(x.shape(0)), p = static_cast<i64>(x.shape(1));
            std::vector<double> col(static_cast<std::size_t>(m * p));
            for (i64 i = 0; i < m; ++i)
                for (i64 j = 0; j < p; ++j) col[static_cast<std::size_t>(j * m + i)] = x.data()[i * p + j];
            return to_array(r_unique_rows(col.data(), m, p));
        },
        py::arg("x"), "Rows (0-based) that R's unique.matrix() keeps (exact ==, 0 == -0)");
    m.def("sdm_thin_spthin", &py_thin_spthin, py::arg("lon"), py::arg("lat"), py::arg("max_km"), py::arg("reps"),
          py::arg("seed"), "spThin::thin.algorithm: (keep[reps, n], kept[reps])");
    m.def("sdm_earth_neighbours", &py_earth_neighbours, py::arg("lon"), py::arg("lat"), py::arg("max_km"),
          "Neighbours closer than max_km (fields::rdist.earth): CSR (start, index)");
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
