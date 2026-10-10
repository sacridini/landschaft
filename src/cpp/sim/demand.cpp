// SPDX-License-Identifier: GPL-3.0-or-later
//
// Demands and Markov chains (see demand.hpp). The matrix logarithm and
// exponential are Eigen's (Schur-Parlett and scaling and squaring with Padé
// approximants), as accurate as SciPy's logm and expm.

#include "sim/demand.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "lsm/rmath.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// The indices of v by a stable sort, increasing, NaN last (numpy's argsort).
std::vector<std::size_t> stable_order(const std::vector<double>& v) {
    std::vector<std::size_t> o(v.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
        if (std::isnan(v[b])) return !std::isnan(v[a]);
        return v[a] < v[b];
    });
    return o;
}

Matrix to_matrix(const std::vector<double>& p, int k) {
    if (static_cast<i64>(p.size()) != i64{k} * k) throw std::invalid_argument("the matrix must be k x k");
    return Eigen::Map<const Matrix>(p.data(), k, k);
}

std::vector<double> from_matrix(const Matrix& m) { return {m.data(), m.data() + m.size()}; }

}  // namespace

std::vector<double> approx_extrap(const std::vector<double>& x_in, const std::vector<double>& y_in,
                                  const std::vector<double>& xout) {
    if (x_in.size() != y_in.size()) throw std::invalid_argument("x and y differ in length");
    std::vector<double> xu, yu;
    for (std::size_t k = 0; k < x_in.size(); ++k) {
        if (std::find(xu.begin(), xu.end(), x_in[k]) != xu.end()) continue;  // the first of duplicated x
        xu.push_back(x_in[k]);
        yu.push_back(y_in[k]);
    }
    const auto o = stable_order(xu);
    std::vector<double> x(xu.size()), y(yu.size());
    for (std::size_t k = 0; k < o.size(); ++k) {
        x[k] = xu[o[k]];
        y[k] = yu[o[k]];
    }
    const std::size_t n = x.size();
    if (n < 2) throw std::invalid_argument("approxExtrap needs two distinct x");
    std::vector<double> out(xout.size());
    for (std::size_t k = 0; k < xout.size(); ++k) {
        const double v = xout[k];
        if (v < x[0]) {
            out[k] = (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0]) + y[0];
        } else if (v > x[n - 1]) {
            out[k] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) * (v - x[n - 2]) + y[n - 2];
        } else {  // R's approx1: bisection, then y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
            std::size_t i = 0, j = n - 1;
            while (i < j - 1) {
                const std::size_t ij = (i + j) / 2;
                if (v < x[ij]) j = ij;
                else i = ij;
            }
            if (v == x[j]) out[k] = y[j];
            else if (v == x[i]) out[k] = y[i];
            else out[k] = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]));
        }
    }
    return out;
}

std::vector<double> round_sum(const std::vector<double>& x, std::int64_t nrow, std::int64_t ncol, std::int64_t ncell) {
    if (static_cast<i64>(x.size()) != nrow * ncol) throw std::invalid_argument("x must be nrow x ncol");
    std::vector<double> out(x.size()), y(static_cast<std::size_t>(ncol)), fl(static_cast<std::size_t>(ncol)),
        frac(static_cast<std::size_t>(ncol));
    for (i64 r = 0; r < nrow; ++r) {
        const double* row = x.data() + r * ncol;
        double* o = out.data() + r * ncol;
        const double total = lsm::r_sum(row, ncol);
        for (i64 c = 0; c < ncol; ++c) {
            y[static_cast<std::size_t>(c)] = row[c] / total * static_cast<double>(ncell);
            fl[static_cast<std::size_t>(c)] = std::floor(y[static_cast<std::size_t>(c)]);
            frac[static_cast<std::size_t>(c)] = y[static_cast<std::size_t>(c)] - fl[static_cast<std::size_t>(c)];
            o[c] = fl[static_cast<std::size_t>(c)];
        }
        const auto ix = stable_order(frac);
        const double tot = static_cast<double>(ncell) - lsm::r_sum(fl.data(), ncol);
        if (tot > 0) {
            // ix[int(len(y) - tot):], with Python's slicing of a negative start
            i64 start = static_cast<i64>(static_cast<double>(ncol) - tot);
            if (start < 0) start = std::max<i64>(0, start + ncol);
            for (i64 c = start; c < ncol; ++c) o[ix[static_cast<std::size_t>(c)]] += 1;
        }
    }
    return out;
}

std::vector<double> markov_from_counts(const std::vector<double>& counts, int k, std::vector<int>& kept) {
    if (static_cast<i64>(counts.size()) != i64{k} * k) throw std::invalid_argument("the counts must be k x k");
    std::vector<bool> start(static_cast<std::size_t>(k)), reached(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        double s = 0;
        for (int j = 0; j < k; ++j) s += counts[static_cast<std::size_t>(i * k + j)];
        start[static_cast<std::size_t>(i)] = s > 0;
    }
    for (int j = 0; j < k; ++j) {
        double s = 0;
        for (int i = 0; i < k; ++i)
            if (start[static_cast<std::size_t>(i)]) s += counts[static_cast<std::size_t>(i * k + j)];
        reached[static_cast<std::size_t>(j)] = start[static_cast<std::size_t>(j)] || s > 0;
    }
    kept.clear();
    for (int i = 0; i < k; ++i)
        if (reached[static_cast<std::size_t>(i)]) kept.push_back(i);
    const int m = static_cast<int>(kept.size());
    std::vector<double> p(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
    for (int a = 0; a < m; ++a) {
        double r = 0;  // counts are whole numbers: exact in any order
        for (int b = 0; b < m; ++b) r += counts[static_cast<std::size_t>(kept[a] * k + kept[b])];
        if (r > 0) {
            for (int b = 0; b < m; ++b)
                p[static_cast<std::size_t>(a * m + b)] = counts[static_cast<std::size_t>(kept[a] * k + kept[b])] / r;
        } else {
            p[static_cast<std::size_t>(a * m + a)] = 1.0;  // without cells at the start: stays as it is
        }
    }
    return p;
}

std::vector<double> markov_root(const std::vector<double>& p, int k, int n) {
    if (n < 1) throw std::invalid_argument("steps must be >= 1");
    if (n == 1) return p;
    const Matrix m = to_matrix(p, k);
    Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(m), true);
    if (es.info() != Eigen::Success) throw std::runtime_error("the eigendecomposition of the transition matrix failed");
    const Eigen::VectorXcd w = es.eigenvalues();
    const double wmax = w.cwiseAbs().maxCoeff();
    bool singular = false;
    for (int i = 0; i < k; ++i) {
        const bool on_axis = std::fabs(w[i].imag()) <= 1e-12 * std::max(1.0, wmax);
        if (on_axis && w[i].real() < -1e-12)
            throw std::invalid_argument("the transition matrix has a negative eigenvalue: no real matrix of rates per "
                                        "time step gives it (the change is not a Markov chain of equal steps)");
        singular = singular || std::abs(w[i]) <= 1e-12;
    }
    if (singular) {
        const Eigen::MatrixXcd v = es.eigenvectors();
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(v);
        const auto& sv = svd.singularValues();
        if (sv(sv.size() - 1) == 0 || sv(0) / sv(sv.size() - 1) > 1e8)
            throw std::invalid_argument("the transition matrix is singular and not diagonalisable: no root computed");
        Eigen::VectorXcd wr(k);
        for (int i = 0; i < k; ++i)
            wr[i] = std::abs(w[i]) <= 1e-12 ? std::complex<double>(0.0, 0.0) : std::pow(w[i], 1.0 / n);
        const Eigen::MatrixXcd root = v * wr.asDiagonal() * v.inverse();
        return from_matrix(Matrix(root.real()));
    }
    const Eigen::MatrixXd log = Eigen::MatrixXd(m).log();
    const Eigen::MatrixXd root = (log / static_cast<double>(n)).exp();
    return from_matrix(Matrix(root));
}

std::vector<double> markov_project(const std::vector<double>& areas, const std::vector<double>& p, int k, int steps) {
    if (static_cast<int>(areas.size()) != k) throw std::invalid_argument("one area per class");
    const Matrix m = to_matrix(p, k);
    std::vector<double> out(static_cast<std::size_t>(steps + 1) * static_cast<std::size_t>(k));
    std::copy(areas.begin(), areas.end(), out.begin());
    for (int s = 1; s <= steps; ++s) {
        const double* prev = out.data() + static_cast<std::size_t>(s - 1) * static_cast<std::size_t>(k);
        double* cur = out.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(k);
        for (int j = 0; j < k; ++j) {
            double v = 0;
            for (int i = 0; i < k; ++i) v += prev[i] * m(i, j);
            cur[j] = v;
        }
    }
    return out;
}

}  // namespace sim
