// SPDX-License-Identifier: GPL-3.0-or-later

#include "rest/intactness.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/rmath.hpp"

namespace rest {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// plangea's focal window: exp(beta d) for the offsets within the radius, in R's column-major order
// (column offset outer, row offset inner), as its as.vector(foc.w) and the window's as.vector().
struct Window {
    std::vector<int> dr, dc;
    std::vector<double> w;
};

Window make_window(const IntactnessParams& p) {
    if (p.radius < 0) throw std::invalid_argument("the radius must be at least 0 cells");
    const int n = p.radius;
    // rad = n_pixels * res
    const double rad = static_cast<double>(n) * p.dx;
    Window win;
    for (int j = -n; j <= n; ++j)
        for (int i = -n; i <= n; ++i) {
            const double a = p.dy * static_cast<double>(i), b = p.dx * static_cast<double>(j);
            const double d = lsm::r_sqrt(a * a + b * b);
            if (d <= rad) {
                win.dr.push_back(i);
                win.dc.push_back(j);
                win.w.push_back(lsm::r_exp(p.beta * d));
            }
        }
    return win;
}

// exp(-gamma HFI), R's way
std::vector<double> quality(const double* hfi, i64 size, double gamma, int threads) {
    std::vector<double> q(static_cast<std::size_t>(size));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < size; ++i) q[i] = std::isnan(hfi[i]) ? kNaN : lsm::r_exp(-gamma * hfi[i]);
    return q;
}

}  // namespace

std::vector<double> restored_footprint(const double* hfi, const double* reduction, const double* restored, i64 ny,
                                       i64 nx, int threads) {
    const i64 size = ny * nx;
    std::vector<double> h(hfi, hfi + size);
    if (!reduction || !restored) return h;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < size; ++i) {
        if (std::isnan(h[i])) continue;
        const double r = reduction[i] * restored[i];  // HFI - prop_convert_weight * x
        const double v = std::isnan(r) ? h[i] : h[i] - r;
        h[i] = v < 0 ? 0.0 : v;
    }
    return h;
}

std::vector<double> intactness(const double* hfi, i64 ny, i64 nx, const IntactnessParams& p, int threads) {
    const Window win = make_window(p);
    const std::vector<double> q = quality(hfi, ny * nx, p.gamma, threads);
    const i64 nw = static_cast<i64>(win.w.size());
    std::vector<double> out(static_cast<std::size_t>(ny * nx), kNaN);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const double qc = q[r * nx + c];
            if (std::isnan(qc)) continue;
            auto at = [&](i64 t) {  // the window's cell t: NaN outside the raster or without a footprint
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) return kNaN;
                return q[rr * nx + cc];
            };
            // sum(((q_rc) * x)^z * dvec, na.rm = TRUE) / sum(dvec[which(!is.na(x))], na.rm = TRUE)
            const double num = lsm::sum_of(0, nw, [&](i64 t) {
                const double x = at(t);
                return std::isnan(x) ? kNaN : lsm::r_pow(qc * x, p.z) * win.w[t];
            }, true);
            const double den = lsm::sum_of(0, nw, [&](i64 t) { return std::isnan(at(t)) ? kNaN : win.w[t]; }, true);
            out[r * nx + c] = num / den;
        }
    return out;
}

std::vector<double> intactness_gradient(const double* hfi, const double* reduction, i64 ny, i64 nx,
                                        const IntactnessParams& p, int threads) {
    const Window win = make_window(p);
    const std::vector<double> q = quality(hfi, ny * nx, p.gamma, threads);
    const i64 nw = static_cast<i64>(win.w.size());
    const i64 size = ny * nx;
    // per cell: the window's weight W_i and Q_i
    std::vector<double> wsum(static_cast<std::size_t>(size), kNaN), qv(static_cast<std::size_t>(size), kNaN);
    std::vector<double> qz(static_cast<std::size_t>(size), kNaN);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < size; ++i)
        if (!std::isnan(q[i])) qz[i] = lsm::r_pow(q[i], p.z);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            if (std::isnan(q[i])) continue;
            double s = 0.0, w = 0.0;
            for (i64 t = 0; t < nw; ++t) {
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx || std::isnan(q[rr * nx + cc])) continue;
                s += qz[rr * nx + cc] * win.w[t];
                w += win.w[t];
            }
            wsum[i] = w;
            qv[i] = qz[i] * s / w;
        }
    // dT/dq_k = z / q_k (Q_k + sum_i q_i^z q_k^z w_ik / W_i), the i within k's window (the window is
    // symmetric); dq_k/dx_k = gamma q_k reduction_k while the footprint is above 0
    std::vector<double> out(static_cast<std::size_t>(size), kNaN);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 k = r * nx + c;
            if (std::isnan(q[k])) continue;
            const double a = reduction ? reduction[k] : 0.0;
            if (std::isnan(a) || a == 0.0 || (!(hfi[k] > 0.0) && a > 0.0)) {
                out[k] = 0.0;
                continue;
            }
            double b = 0.0;
            for (i64 t = 0; t < nw; ++t) {
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
                const i64 i = rr * nx + cc;
                if (std::isnan(q[i])) continue;
                b += qz[i] * win.w[t] / wsum[i];
            }
            out[k] = p.z * p.gamma * a * (qv[k] + qz[k] * b);
        }
    return out;
}

double intactness_secant(const double* h0, const double* tlo, const double* thi, i64 ny, i64 nx,
                         const IntactnessParams& p, int threads, std::vector<double>& sigma) {
    const Window win = make_window(p);
    const i64 nw = static_cast<i64>(win.w.size());
    const i64 size = ny * nx;
    const double gz = p.gamma * p.z;
    std::vector<double> wsum(static_cast<std::size_t>(size), kNaN);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            if (std::isnan(h0[i])) continue;
            double w = 0.0;
            for (i64 t = 0; t < nw; ++t) {
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx || std::isnan(h0[rr * nx + cc])) continue;
                w += win.w[t];
            }
            wsum[i] = w;
        }
    // the secant of the pair (j, k), weight wt of k in j's window: its slope and its value at s_lo
    auto secant = [&](i64 j, i64 k, double wt, double& slope, double& at_lo) {
        const double c = wt / wsum[j];
        const double base = h0[j] + h0[k];
        const double lo = tlo[j] + tlo[k], hi = thi[j] + thi[k];
        const double e_lo = c * std::exp(gz * (lo - base));
        if (hi - lo > 1e-12 * (1.0 + std::fabs(lo) + std::fabs(hi))) {
            slope = (c * std::exp(gz * (hi - base)) - e_lo) / (hi - lo);
        } else {
            slope = gz * e_lo;  // no room: the tangent
        }
        at_lo = e_lo - slope * lo;
    };
    sigma.assign(static_cast<std::size_t>(size), 0.0);
    std::vector<double> u0(static_cast<std::size_t>(ny), 0.0);
    // j's own side: the pairs of its window
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r) {
        double row = 0.0;
        for (i64 c = 0; c < nx; ++c) {
            const i64 j = r * nx + c;
            if (std::isnan(h0[j])) continue;
            double s = 0.0;
            for (i64 t = 0; t < nw; ++t) {
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
                const i64 k = rr * nx + cc;
                if (std::isnan(h0[k])) continue;
                double slope, at_lo;
                secant(j, k, win.w[t], slope, at_lo);
                s += slope;
                row += at_lo;
            }
            sigma[j] += s;
        }
        u0[r] = row;
    }
    // k's side: the pairs (j, k) of the windows that hold k (the window is symmetric)
    std::vector<double> other(static_cast<std::size_t>(size), 0.0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 k = r * nx + c;
            if (std::isnan(h0[k])) continue;
            double s = 0.0;
            for (i64 t = 0; t < nw; ++t) {
                const i64 rr = r + win.dr[t], cc = c + win.dc[t];
                if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
                const i64 j = rr * nx + cc;
                if (std::isnan(h0[j])) continue;
                double slope, at_lo;
                secant(j, k, win.w[t], slope, at_lo);
                s += slope;
            }
            other[k] = s;
        }
    double total = 0.0;
    for (i64 r = 0; r < ny; ++r) total += u0[r];
    for (i64 i = 0; i < size; ++i) sigma[i] += other[i];
    return total;
}

}  // namespace rest
