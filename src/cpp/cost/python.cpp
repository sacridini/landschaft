// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the grid distances (landschaft._core.cost_*).

#include "cost/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cost/geodesic.hpp"
#include "cost/shortest.hpp"
#include "cost/transition.hpp"
#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"

namespace py = pybind11;

namespace {

using cost::i64;

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> vector_to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

// The grid of a transition: (ny, nx, xmin, xmax, ymin, ymax, wrap).
cost::Grid grid_of(const py::tuple& t) {
    if (t.size() != 7) throw std::invalid_argument("grid: (ny, nx, xmin, xmax, ymin, ymax, wrap)");
    cost::Grid g;
    g.ny = t[0].cast<i64>();
    g.nx = t[1].cast<i64>();
    g.xmin = t[2].cast<double>();
    g.xmax = t[3].cast<double>();
    g.ymin = t[4].cast<double>();
    g.ymax = t[5].cast<double>();
    g.wrap = t[6].cast<bool>();
    return g;
}

i64 n_directions(int directions, bool symmetric) { return static_cast<i64>(cost::offsets(directions, symmetric).size()); }

cost::Builtin builtin_of(const std::string& name) {
    if (name == "mean") return cost::Builtin::Mean;
    if (name == "min") return cost::Builtin::Min;
    if (name == "max") return cost::Builtin::Max;
    if (name == "sum") return cost::Builtin::Sum;
    if (name == "from") return cost::Builtin::From;
    if (name == "to") return cost::Builtin::To;
    if (name == "diff") return cost::Builtin::Diff;
    if (name == "harmonic") return cost::Builtin::Harmonic;
    if (name == "geometric") return cost::Builtin::Geometric;
    if (name == "prod") return cost::Builtin::Product;
    throw std::invalid_argument("unknown transition function '" + name + "'");
}

void check_values(const cost::Grid& g, int directions, bool symmetric, const py::buffer_info& b) {
    if (b.size != n_directions(directions, symmetric) * g.ncell())
        throw std::invalid_argument("values do not match the grid and directions");
}

}  // namespace

namespace cost {

void register_python(py::module_& m) {
    m.def(
        "cost_offsets",
        [](int directions, bool symmetric) {
            std::vector<std::pair<int, int>> out;
            for (const Offset o : offsets(directions, symmetric)) out.emplace_back(o.dr, o.dc);
            return out;
        },
        py::arg("directions"), py::arg("symmetric"), "Directions (row, column offsets) of a transition layout.");

    m.def(
        "cost_builtin",
        [](const CArray<double>& x, const py::tuple& grid, int directions, bool symmetric, const std::string& f) {
            const Grid g = grid_of(grid);
            if (x.size() != g.ncell()) throw std::invalid_argument("values do not match the grid");
            const Builtin fn = builtin_of(f);
            const i64 nd = n_directions(directions, symmetric);
            py::array_t<double> out({nd, g.ncell()});
            double* o = out.mutable_data();
            const double* px = x.data();
            {
                py::gil_scoped_release release;
                builtin_values(px, g, directions, symmetric, fn, o);
            }
            return out;
        },
        py::arg("x"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("function"),
        "Transition values of a built-in function of the two cells' values (gdistance's transition()).");

    m.def(
        "cost_neighbours",
        [](const py::tuple& grid, int directions, bool symmetric) {
            const Grid g = grid_of(grid);
            const i64 nd = n_directions(directions, symmetric);
            py::array_t<i64> out({nd, g.ncell()});
            i64* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                neighbours(g, directions, symmetric, o);
            }
            return out;
        },
        py::arg("grid"), py::arg("directions"), py::arg("symmetric"),
        "The neighbour cell of every (direction, cell) of a transition layout (-1: none).");

    m.def(
        "cost_geo_correction",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric, bool lonlat,
           const std::string& type, bool resistance, bool scale, bool factors) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            if (type.size() != 1) throw std::invalid_argument("type must be 'c' or 'r'");
            py::array_t<double> out({n_directions(directions, symmetric), g.ncell()});
            std::copy(values.data(), values.data() + values.size(), out.mutable_data());
            double* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                geo_correction(g, directions, symmetric, lonlat, type[0], resistance, scale, factors, o);
            }
            return out;
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("lonlat"),
        py::arg("type"), py::arg("resistance"), py::arg("scale"), py::arg("factors"),
        "gdistance's geoCorrection on transition values.");

    m.def(
        "cost_margin_sums",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric, int margin,
           bool count) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            py::array_t<double> out(g.ncell());
            double* o = out.mutable_data();
            const double* v = values.data();
            {
                py::gil_scoped_release release;
                margin_sums(g, directions, symmetric, v, margin, count, o);
            }
            return out;
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("margin"),
        py::arg("count") = false, "Row (0) or column (1) sums of a transition matrix, as the Matrix package.");

    m.def(
        "cost_to_csc",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            const double* v = values.data();
            Csc m;
            {
                py::gil_scoped_release release;
                m = to_csc(g, directions, symmetric, v);
            }
            return py::make_tuple(vector_to_array(std::move(m.p)), vector_to_array(std::move(m.i)),
                                  vector_to_array(std::move(m.x)));
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"),
        "The transition matrix in compressed sparse columns (p, i, x), both triangles.");

    m.def("cost_scale_value", [](const py::tuple& grid, bool lonlat) { return scale_value(grid_of(grid), lonlat); },
          py::arg("grid"), py::arg("lonlat"), "geoCorrection's scale (scl = TRUE).");

    m.def(
        "cost_geodesic",
        [](const CArray<double>& lon1, const CArray<double>& lat1, const CArray<double>& lon2,
           const CArray<double>& lat2) {
            const i64 n = lon1.size();
            if (lat1.size() != n || lon2.size() != n || lat2.size() != n)
                throw std::invalid_argument("coordinates of different lengths");
            py::array_t<double> out(n);
            double* o = out.mutable_data();
            const double *a = lon1.data(), *b = lat1.data(), *c = lon2.data(), *d = lat2.data();
            {
                py::gil_scoped_release release;
                const Geodesic geod;
#pragma omp parallel for schedule(static)
                for (i64 k = 0; k < n; ++k) o[k] = geod.distance(a[k], b[k], c[k], d[k]);
            }
            return out;
        },
        py::arg("lon1"), py::arg("lat1"), py::arg("lon2"), py::arg("lat2"),
        "Geodesic distances on WGS84 as the R package raster computes them (metres).");

    m.def(
        "cost_rmath",
        [](const std::string& name, const CArray<double>& x, double y) {
            const i64 n = x.size();
            py::array_t<double> out(n);
            double* o = out.mutable_data();
            const double* px = x.data();
            int f = name == "sqrt" ? 0 : name == "exp" ? 1 : name == "log" ? 2 : name == "pow" ? 3 : -1;
            if (f < 0) throw std::invalid_argument("unknown function '" + name + "'");
            {
                py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
                for (i64 k = 0; k < n; ++k) {
                    switch (f) {
                        case 0: o[k] = lsm::r_sqrt(px[k]); break;
                        case 1: o[k] = lsm::r_exp(px[k]); break;
                        case 2: o[k] = lsm::r_log(px[k]); break;
                        default: o[k] = lsm::r_pow(px[k], y); break;
                    }
                }
            }
            return out;
        },
        py::arg("name"), py::arg("x"), py::arg("y") = 0.0,
        "R's vectorised sqrt, exp, log and x^y (mingw's x87 maths) on an array.");

    m.def(
        "cost_acc_cost",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric,
           const std::vector<i64>& sources) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            const Layout t{&g, directions, symmetric, values.data()};
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = accumulated_cost(t, sources);
            }
            return vector_to_array(std::move(out));
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("sources"),
        "Accumulated cost from the nearest source cell (gdistance's accCost).");

    m.def(
        "cost_cost_matrix",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric,
           const std::vector<i64>& from, const std::vector<i64>& to) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            const Layout t{&g, directions, symmetric, values.data()};
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = cost_matrix(t, from, to);
            }
            py::array_t<double> a = vector_to_array(std::move(out));
            return a.reshape({static_cast<py::ssize_t>(from.size()), static_cast<py::ssize_t>(to.size())});
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("from_cells"),
        py::arg("to_cells"), "Least costs between cells (gdistance's costDistance), sources in parallel.");

    m.def(
        "cost_shortest_paths",
        [](const CArray<double>& values, const py::tuple& grid, int directions, bool symmetric, i64 origin,
           const std::vector<i64>& goals) {
            const Grid g = grid_of(grid);
            py::buffer_info b = values.request();
            check_values(g, directions, symmetric, b);
            const Layout t{&g, directions, symmetric, values.data()};
            std::vector<std::vector<i64>> paths;
            {
                py::gil_scoped_release release;
                paths = shortest_paths(t, origin, goals);
            }
            py::list out;
            for (auto& p : paths) out.append(vector_to_array(std::move(p)));
            return out;
        },
        py::arg("values"), py::arg("grid"), py::arg("directions"), py::arg("symmetric"), py::arg("origin"),
        py::arg("goals"), "Cells of the least-cost paths from origin to each goal (igraph's ties).");

    m.def(
        "cost_r_mean", [](const CArray<double>& x) { return lsm::r_mean(x.data(), x.size()); }, py::arg("x"),
        "R's mean() (long double, two passes).");

    m.def(
        "cost_r_cov",
        [](const CArray<double>& x) {
            // R's cov(x) (cov.c, cov_na_1, no missing values): column means in
            // two passes, sums of products of deviations, all in LDOUBLE.
            if (x.ndim() != 2) throw std::invalid_argument("x must be 2-D (observations x variables)");
            const i64 n = x.shape(0), k = x.shape(1);
            const double* px = x.data();
            using E = lsm::Extended;
            std::vector<double> mean(static_cast<std::size_t>(k));
            for (i64 j = 0; j < k; ++j) {
                E s = 0.0;
                for (i64 r = 0; r < n; ++r) s += px[r * k + j];
                E t = s / E(static_cast<double>(n));
                if (std::isfinite(static_cast<double>(t))) {
                    s = 0.0;
                    for (i64 r = 0; r < n; ++r) s += E(px[r * k + j]) - t;
                    t = t + s / E(static_cast<double>(n));
                }
                mean[j] = static_cast<double>(t);
            }
            py::array_t<double> out({k, k});
            double* o = out.mutable_data();
            for (i64 i = 0; i < k; ++i) {
                for (i64 j = 0; j <= i; ++j) {
                    E s = 0.0;
                    const E xm = mean[i], ym = mean[j];
                    for (i64 r = 0; r < n; ++r) s += (E(px[r * k + i]) - xm) * (E(px[r * k + j]) - ym);
                    o[i * k + j] = o[j * k + i] = static_cast<double>(s / E(static_cast<double>(n - 1)));
                }
            }
            return out;
        },
        py::arg("x"), "R's cov() of a complete matrix (observations x variables).");

    m.def("cost_mingw_cbrt", &mingw_cbrt, py::arg("x"), "mingw-w64's cbrt (Cephes).");
}

}  // namespace cost
