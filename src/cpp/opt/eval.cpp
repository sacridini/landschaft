// SPDX-License-Identifier: GPL-3.0-or-later
//
// pnorm: R 4.4.2's nmath/pnorm.c (pnorm_both, Cody's algorithm; GPL >= 2), with R's exp on Windows
// (lsm::r_exp, mingw-w64's x87 exp). The rest: prioritizr 8.0.6's rcpp_boundary.cpp,
// rcpp_connectivity.cpp, rcpp_asym_connectivity.cpp and rcpp_ferrier_score.cpp (GPL-3).

#include "opt/eval.hpp"

#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "lsm/rmath.hpp"

namespace opt::eval {

double pnorm(double x) {
    if (std::isnan(x)) return x;
    if (!std::isfinite(x)) return x < 0 ? 0.0 : 1.0;
    static const double a[5] = {2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582,
                                18154.981253343561249, 0.065682337918207449113};
    static const double b[4] = {47.20258190468824187, 976.09855173777669322, 10260.932208618978205,
                                45507.789335026729956};
    static const double c[9] = {0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979,
                                597.27027639480026226,  2494.5375852903726711, 6848.1904505362823326,
                                11602.651437647350124,  9842.7148383839780218, 1.0765576773720192317e-8};
    static const double d[8] = {22.266688044328115691, 235.38790178262499861, 1519.377599407554805,
                                6485.558298266760755,  18615.571640885098091, 34900.952721145977266,
                                38912.003286093271411, 19685.429676859990727};
    static const double p[6] = {0.21589853405795699,     0.1274011611602473639,  0.022235277870649807,
                                0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303};
    static const double q[5] = {1.28426009614491121, 0.468238212480865118, 0.0659881378689285515,
                                0.00378239633202758244, 7.29751555083966205e-5};
    constexpr double M_SQRT_32 = 5.656854249492380195206754896838;
    constexpr double M_1_SQRT_2PI = 0.398942280401432677939946059934;
    const double eps = DBL_EPSILON * 0.5;
    double cum, ccum, xnum, xden, temp, xsq, del;
    const double y = std::fabs(x);
    auto do_del = [&](double X) {
        xsq = std::ldexp(std::trunc(std::ldexp(X, 4)), -4);
        del = (X - xsq) * (X + xsq);
        cum = lsm::r_exp(-xsq * std::ldexp(xsq, -1)) * lsm::r_exp(-std::ldexp(del, -1)) * temp;
        ccum = 1.0 - cum;
    };
    if (y <= 0.67448975) {
        if (y > eps) {
            xsq = x * x;
            xnum = a[4] * xsq;
            xden = xsq;
            for (int i = 0; i < 3; ++i) {
                xnum = (xnum + a[i]) * xsq;
                xden = (xden + b[i]) * xsq;
            }
        } else {
            xnum = xden = 0.0;
        }
        temp = x * (xnum + a[3]) / (xden + b[3]);
        return 0.5 + temp;
    }
    if (y <= M_SQRT_32) {
        xnum = c[8] * y;
        xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        temp = (xnum + c[7]) / (xden + d[7]);
        do_del(y);
    } else if (-37.5193 < x && x < 8.2924) {
        xsq = 1.0 / (x * x);
        xnum = p[5] * xsq;
        xden = xsq;
        for (int i = 0; i < 4; ++i) {
            xnum = (xnum + p[i]) * xsq;
            xden = (xden + q[i]) * xsq;
        }
        temp = xsq * (xnum + p[4]) / (xden + q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;
        do_del(x);
    } else {
        return x > 0 ? 1.0 : 0.0;
    }
    return x > 0. ? ccum : cum;  // swap_tail
}

double accumulate(const double* x, i64 n) {
    double acc1 = 0.0, acc2 = 0.0;
    i64 j;
    for (j = 1; j < n; j += 2) {
        acc1 += x[j - 1];
        acc2 += x[j];
    }
    if (j - 1 < n) acc1 += x[j - 1];
    return acc1 + acc2;
}

double boundary(double edge_factor, const Csc& lower, const std::vector<double>& exposed,
                const std::vector<double>& total, const std::vector<double>& solution) {
    const std::size_t n = total.size();
    if (exposed.size() != n || solution.size() != n || lower.ncol != static_cast<i64>(n))
        throw std::invalid_argument("boundary: sizes");
    double c1 = 0.0, c2 = 0.0;
    const double zone = 1.0;
    for (std::size_t i = 0; i < n; ++i) c1 += solution[i] * zone * ((total[i] - exposed[i]) + (exposed[i] * edge_factor));
    for (i64 col = 0; col < lower.ncol; ++col)
        for (i64 k = lower.start[static_cast<std::size_t>(col)]; k < lower.start[static_cast<std::size_t>(col) + 1]; ++k) {
            const double v = lower.value[static_cast<std::size_t>(k)] * zone;
            if (std::abs(v) > 1.0e-15)
                c2 -= (2.0 * v * solution[static_cast<std::size_t>(lower.index[static_cast<std::size_t>(k)])] *
                       solution[static_cast<std::size_t>(col)]);
        }
    return c1 + c2 + 0.0;
}

double connectivity(const Csc& data, const std::vector<double>& solution) {
    // row(i) *= s_i, then col(i) *= s_i, for i = 0, 1, ...: an entry (r, c) is scaled first by the
    // smaller of r and c (by its row when r == c); entries that become 0 leave the matrix
    std::vector<double> kept;
    kept.reserve(data.value.size());
    for (i64 col = 0; col < data.ncol; ++col)
        for (i64 k = data.start[static_cast<std::size_t>(col)]; k < data.start[static_cast<std::size_t>(col) + 1]; ++k) {
            const i64 row = data.index[static_cast<std::size_t>(k)];
            const double sr = solution[static_cast<std::size_t>(row)], sc = solution[static_cast<std::size_t>(col)];
            double v = data.value[static_cast<std::size_t>(k)];
            v = row <= col ? (v * sr) * sc : (v * sc) * sr;
            if (v != 0.0) kept.push_back(v);
        }
    return 0.0 + accumulate(kept.data(), static_cast<i64>(kept.size()));
}

double asym_connectivity(const Csc& data, const std::vector<double>& solution) {
    double out = 0.0;
    for (i64 col = 0; col < data.ncol; ++col)
        for (i64 k = data.start[static_cast<std::size_t>(col)]; k < data.start[static_cast<std::size_t>(col) + 1]; ++k) {
            const i64 row = data.index[static_cast<std::size_t>(k)];
            const double v = data.value[static_cast<std::size_t>(k)];
            if (row == col)
                out += v * solution[static_cast<std::size_t>(row)];
            else
                out += (v * solution[static_cast<std::size_t>(row)]) * solution[static_cast<std::size_t>(col)];
        }
    return out;
}

double boundary_zones(const std::vector<double>& edge_factor, const std::vector<double>& zones, const Csc& lower,
                      const std::vector<double>& exposed, const std::vector<double>& total,
                      const std::vector<double>& solution) {
    const std::size_t n = total.size(), nz = edge_factor.size();
    if (exposed.size() != n || solution.size() != n * nz || zones.size() != nz * nz || lower.ncol != static_cast<i64>(n))
        throw std::invalid_argument("boundary_zones: sizes");
    auto s = [&](i64 i, std::size_t z) { return solution[z * n + static_cast<std::size_t>(i)]; };
    double c1 = 0.0, c2 = 0.0, c3 = 0.0;
    for (std::size_t z = 0; z < nz; ++z)
        for (std::size_t i = 0; i < n; ++i)
            c1 += s(static_cast<i64>(i), z) * zones[z * nz + z] * ((total[i] - exposed[i]) + (exposed[i] * edge_factor[z]));
    for (std::size_t z1 = 0; z1 < nz; ++z1)
        for (std::size_t z2 = z1; z2 < nz; ++z2) {
            const double zm = zones[z1 * nz + z2];
            for (i64 col = 0; col < lower.ncol; ++col)
                for (i64 k = lower.start[static_cast<std::size_t>(col)]; k < lower.start[static_cast<std::size_t>(col) + 1]; ++k) {
                    const double v = lower.value[static_cast<std::size_t>(k)] * zm;
                    if (!(std::abs(v) > 1.0e-15)) continue;
                    const i64 row = lower.index[static_cast<std::size_t>(k)];
                    if (z1 == z2) {
                        c2 -= (2.0 * v * s(row, z1) * s(col, z2));
                    } else {
                        c3 -= (2.0 * v * s(row, z1) * s(col, z2));
                        c3 -= (2.0 * v * s(row, z2) * s(col, z1));
                    }
                }
        }
    return c1 + c2 + c3;
}

double connectivity_zones(const Csc& data, const std::vector<double>& zones, const std::vector<double>& solution) {
    const std::size_t n = static_cast<std::size_t>(data.ncol);
    const std::size_t nz = static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(zones.size()))));
    if (nz * nz != zones.size() || solution.size() != n * nz) throw std::invalid_argument("connectivity_zones: sizes");
    double out = 0.0;
    std::vector<double> kept;
    for (std::size_t z1 = 0; z1 < nz; ++z1)
        for (std::size_t z2 = 0; z2 < nz; ++z2) {
            // data * zones[z1, z2] (R), the diagonal zeroed between zones, then row(i) *= s_z1[i] and
            // col(i) *= s_z2[i] for i = 0, 1, ...; the entries left summed (Armadillo's accu)
            const double zm = zones[z1 * nz + z2];
            kept.clear();
            for (i64 col = 0; col < data.ncol; ++col)
                for (i64 k = data.start[static_cast<std::size_t>(col)]; k < data.start[static_cast<std::size_t>(col) + 1]; ++k) {
                    const i64 row = data.index[static_cast<std::size_t>(k)];
                    if (z1 != z2 && row == col) continue;
                    const double sr = solution[z1 * n + static_cast<std::size_t>(row)];
                    const double sc = solution[z2 * n + static_cast<std::size_t>(col)];
                    double v = data.value[static_cast<std::size_t>(k)] * zm;
                    v = row <= col ? (v * sr) * sc : (v * sc) * sr;
                    if (v != 0.0) kept.push_back(v);
                }
            out += accumulate(kept.data(), static_cast<i64>(kept.size()));
        }
    return out;
}

double asym_connectivity_zones(const Csc& data, const std::vector<double>& zones, const std::vector<double>& solution) {
    const std::size_t n = static_cast<std::size_t>(data.ncol);
    const std::size_t nz = static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(zones.size()))));
    if (nz * nz != zones.size() || solution.size() != n * nz) throw std::invalid_argument("asym_connectivity_zones: sizes");
    double out = 0.0;
    for (std::size_t z1 = 0; z1 < nz; ++z1)
        for (std::size_t z2 = 0; z2 < nz; ++z2) {
            const double zm = zones[z1 * nz + z2];
            for (i64 col = 0; col < data.ncol; ++col)
                for (i64 k = data.start[static_cast<std::size_t>(col)]; k < data.start[static_cast<std::size_t>(col) + 1]; ++k) {
                    const i64 row = data.index[static_cast<std::size_t>(k)];
                    const double v = data.value[static_cast<std::size_t>(k)] * zm;
                    if (v == 0.0) continue;  // a zero entry adds nothing
                    const double sr = solution[z1 * n + static_cast<std::size_t>(row)];
                    if (row == col && z1 == z2)
                        out += (v * sr);
                    else
                        out += (v * sr * solution[z2 * n + static_cast<std::size_t>(col)]);
                }
        }
    return out;
}

namespace {

bool approx_equal(double x, double y) { return std::abs(x - y) < 1.0e-15; }

double adjusted_portfolio_size(double n_pu, double portfolio_size) {
    return portfolio_size > (n_pu / 2.0) ? std::sqrt(n_pu - portfolio_size) / portfolio_size
                                         : std::sqrt(portfolio_size) / portfolio_size;
}

double rx_removed(double n_pu, double ps, double stdev, double amount, double target, double mean, double sum) {
    const double mean_target = target / (ps - 1.0);
    const double adj_sd = stdev * adjusted_portfolio_size(n_pu - 1.0, ps - 1.0);
    if ((sum - amount) < target) return 0.0;
    if (adj_sd < 1.0e-11) return mean < mean_target ? 0.0 : 1.0;
    return 1.0 - pnorm((mean_target - mean) / adj_sd);
}

double rx_included(double n_pu, double ps, double stdev, double amount, double target, double mean) {
    const double mean_target = (target - amount) / (ps - 1.0);
    const double adj_sd = stdev * adjusted_portfolio_size(n_pu - 1.0, ps - 1.0);
    if (amount >= target) return 1.0;
    if (adj_sd < 1.0e-11) return mean < mean_target ? 0.0 : 1.0;
    return 1.0 - pnorm((mean_target - mean) / adj_sd);
}

double rx_excluded(double n_pu, double ps, double stdev, double amount, double target, double sum, double mean) {
    const double mean_target = target / ps;
    const double adj_sd = stdev * adjusted_portfolio_size(n_pu - 1, ps);
    if ((sum - amount) < target) return 0;
    if (adj_sd < 1.0e-11) return mean < mean_target ? 0.0 : 1.0;
    return 1.0 - pnorm((mean_target - mean) / adj_sd);
}

double irrep_value(double n_pu, double ps, double mult, double wt_include, double wt_exclude, double amount,
                   double target, double sum, double sum_sq) {
    const double amount_sq = amount * amount;
    sum = (sum - amount) * mult;
    sum_sq = (sum_sq - amount_sq) * mult;
    const double mean = sum / n_pu;
    const double stdev = std::sqrt(sum_sq - ((sum * sum) / n_pu) / n_pu);
    const double removed = rx_removed(n_pu, ps, stdev, amount, target, mean, sum);
    double included = rx_included(n_pu, ps, stdev, amount, target, mean);
    const double excluded = rx_excluded(n_pu, ps, stdev, amount, target, sum, mean);
    if (approx_equal(included + excluded, 0.0)) return 0.0;
    if (approx_equal(included, 0.0) & (amount > 1.0e-15)) included = 1.0;
    if (approx_equal(included + excluded, 0.0)) return 0.0;
    return ((included - removed) * wt_include) / (included * wt_include + excluded * wt_exclude);
}

}  // namespace

Csc ferrier_score(const Csc& rij, const std::vector<double>& targets, const std::vector<std::uint8_t>& selected) {
    if (static_cast<i64>(targets.size()) != rij.nrow || static_cast<i64>(selected.size()) != rij.ncol)
        throw std::invalid_argument("ferrier: sizes");
    double ps = 0;
    for (const auto s : selected) ps += s ? 1.0 : 0.0;
    const double n_pu = static_cast<double>(rij.ncol);
    const double mult = n_pu / (n_pu - 1.0);
    const double wt_include = ps / n_pu;
    const double wt_exclude = 1.0 - wt_include;
    // every feature's total and total of squares, by its row (in the order of the columns)
    const auto nf = static_cast<std::size_t>(rij.nrow);
    std::vector<double> sum(nf, 0.0), sum_sq(nf, 0.0);
    for (i64 col = 0; col < rij.ncol; ++col)
        for (i64 k = rij.start[static_cast<std::size_t>(col)]; k < rij.start[static_cast<std::size_t>(col) + 1]; ++k) {
            const auto r = static_cast<std::size_t>(rij.index[static_cast<std::size_t>(k)]);
            const double v = rij.value[static_cast<std::size_t>(k)];
            sum[r] += v;
            sum_sq[r] += v * v;
        }
    Csc out;
    out.nrow = rij.nrow;
    out.ncol = rij.ncol;
    out.start.assign(1, 0);
    for (i64 col = 0; col < rij.ncol; ++col) {
        if (selected[static_cast<std::size_t>(col)])
            for (i64 k = rij.start[static_cast<std::size_t>(col)]; k < rij.start[static_cast<std::size_t>(col) + 1]; ++k) {
                const double v = rij.value[static_cast<std::size_t>(k)];
                if (v == 0.0) continue;
                const auto r = static_cast<std::size_t>(rij.index[static_cast<std::size_t>(k)]);
                const double s = irrep_value(n_pu, ps, mult, wt_include, wt_exclude, v, targets[r], sum[r], sum_sq[r]);
                if (s == 0.0) continue;  // arma drops the zeros it is assigned
                out.index.push_back(static_cast<i64>(r));
                out.value.push_back(s);
            }
        out.start.push_back(static_cast<i64>(out.value.size()));
    }
    return out;
}

}  // namespace opt::eval
