// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's stats::kmeans(), ported from R 4.4.2 (GPL >= 2):
//   src/library/stats/R/kmeans.R  - the starts, do_one(), errors and warnings
//   src/library/stats/src/kmns.f  - KMNS / OPTRA / QTRAN, Hartigan & Wong's
//                                   AS 136 in double precision (B. D. Ripley,
//                                   M. Maechler: quick-transfer step limit)
//   src/library/stats/src/kmeans.c - kmeans_Lloyd, kmeans_MacQueen
//   src/library/base/R/duplicated.R, src/main/unique.c - unique.matrix()
// The Fortran is kept statement for statement (1-based indices, the same
// loops, comparisons and order of the double operations; the repo builds with
// /fp:precise, so no contraction into FMA), which is what makes the clusters
// and sums of squares R's to the bit.

#include "sdm/kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"

namespace sdm {

namespace {

// DATA BIG /1.E30/: a REAL (single precision) literal stored in a DOUBLE
// PRECISION variable, i.e. 1.0000000150474662e30.
constexpr double kBig = static_cast<double>(1.0e30f);
constexpr double kZero = 0.0, kOne = 1.0;

// One run of R's do_one(): the .Fortran / .C call's outputs.
struct Run {
    std::vector<double> centers;  // K x N column-major
    std::vector<int> cluster;     // IC1 / c1
    std::vector<int> size;        // NC / nc
    std::vector<double> wss;
    int iter = 0;
    int ifault = 0;
};

// -- AS 136 (kmns.f) -----------------------------------------------------------

// Arrays of kmns.f, 1-based as in Fortran: A(M,N) data, C(K,N) centres.
struct Kmns {
    const double* a;
    int m, n, k;
    double* c;
    std::vector<int> ic1, ic2, nc, ncp, itran, live;
    std::vector<double> an1, an2, d;

    double A(int i, int j) const { return a[static_cast<i64>(i - 1) + static_cast<i64>(j - 1) * m]; }
    double& C(int l, int j) { return c[static_cast<i64>(l - 1) + static_cast<i64>(j - 1) * k]; }

    // Moves point I from cluster L1 to L2: the update shared by OPTRA and QTRAN
    // (centres, NC, AN1, AN2, IC1, IC2).
    void transfer(int i, int l1, int l2) {
        const double al1 = nc[l1];
        const double alw = al1 - kOne;
        const double al2 = nc[l2];
        const double alt = al2 + kOne;
        for (int j = 1; j <= n; ++j) {
            C(l1, j) = (C(l1, j) * al1 - A(i, j)) / alw;
            C(l2, j) = (C(l2, j) * al2 + A(i, j)) / alt;
        }
        nc[l1] = nc[l1] - 1;
        nc[l2] = nc[l2] + 1;
        an2[l1] = alw / al1;
        an1[l1] = kBig;
        if (alw > kOne) an1[l1] = alw / (alw - kOne);
        an1[l2] = alt / al2;
        an2[l2] = alt / (alt + kOne);
        ic1[i] = l2;
        ic2[i] = l1;
    }

    // SUBROUTINE OPTRA (AS 136.1): the optimal-transfer stage.
    void optra(int& indx) {
        for (int l = 1; l <= k; ++l)
            if (itran[l] == 1) live[l] = m + 1;
        for (int i = 1; i <= m; ++i) {
            ++indx;
            const int l1 = ic1[i];
            int l2 = ic2[i];
            const int ll = l2;
            // If point I is the only member of cluster L1, no transfer.
            if (nc[l1] != 1) {
                // If L1 has been updated in this stage, re-compute D(I).
                if (ncp[l1] != 0) {
                    double de = kZero;
                    for (int j = 1; j <= n; ++j) {
                        const double df = A(i, j) - C(l1, j);
                        de = de + df * df;
                    }
                    d[i] = de * an1[l1];
                }
                // Find the cluster with minimum R2.
                double da = kZero;
                for (int j = 1; j <= n; ++j) {
                    const double db = A(i, j) - C(l2, j);
                    da = da + db * db;
                }
                double r2 = da * an2[l2];
                for (int l = 1; l <= k; ++l) {
                    // IF (I .GE. LIVE(L1) .AND. I .GE. LIVE(L) .OR. L .EQ. L1 .OR. L .EQ. LL) GO TO 60
                    if ((i >= live[l1] && i >= live[l]) || l == l1 || l == ll) continue;
                    const double rr = r2 / an2[l];
                    double dc = kZero;
                    bool skip = false;
                    for (int j = 1; j <= n; ++j) {
                        const double dd = A(i, j) - C(l, j);
                        dc = dc + dd * dd;
                        if (dc >= rr) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip) continue;
                    r2 = dc * an2[l];
                    l2 = l;
                }
                if (r2 >= d[i]) {
                    // No transfer: L2 is the new IC2(I).
                    ic2[i] = l2;
                } else {
                    indx = 0;
                    live[l1] = m + i;
                    live[l2] = m + i;
                    ncp[l1] = i;
                    ncp[l2] = i;
                    transfer(i, l1, l2);
                }
            }
            // 90
            if (indx == m) return;
        }
        for (int l = 1; l <= k; ++l) {
            itran[l] = 0;     // before entering QTRAN
            live[l] -= m;     // LIVE(L) is decreased by M before re-entering OPTRA
        }
    }

    // SUBROUTINE QTRAN (AS 136.2): the quick-transfer stage. Sets imaxqtr = -1
    // when the steps reach R's limit.
    void qtran(int& indx, int& imaxqtr) {
        int icoun = 0, istep = 0;
        for (;;) {  // 10 CONTINUE ... GO TO 10
            for (int i = 1; i <= m; ++i) {
                ++icoun;
                ++istep;
                if (istep >= imaxqtr) {
                    imaxqtr = -1;
                    return;
                }
                const int l1 = ic1[i];
                const int l2 = ic2[i];
                // If point I is the only member of cluster L1, no transfer.
                if (nc[l1] != 1) {
                    // If ISTEP > NCP(L1), no need to re-compute the distance to L1.
                    if (istep <= ncp[l1]) {
                        double da = kZero;
                        for (int j = 1; j <= n; ++j) {
                            const double db = A(i, j) - C(l1, j);
                            da = da + db * db;
                        }
                        d[i] = da * an1[l1];
                    }
                    // If ISTEP >= both NCP(L1) and NCP(L2) there is no transfer.
                    if (istep < ncp[l1] || istep < ncp[l2]) {
                        const double r2 = d[i] / an2[l2];
                        double dd = kZero;
                        bool skip = false;
                        for (int j = 1; j <= n; ++j) {
                            const double de = A(i, j) - C(l2, j);
                            dd = dd + de * de;
                            if (dd >= r2) {
                                skip = true;
                                break;
                            }
                        }
                        if (!skip) {
                            icoun = 0;
                            indx = 0;
                            itran[l1] = 1;
                            itran[l2] = 1;
                            ncp[l1] = istep + m;
                            ncp[l2] = istep + m;
                            transfer(i, l1, l2);
                        }
                    }
                }
                // 60: if no re-allocation took place in the last M steps, return.
                if (icoun == m) return;
            }
        }
    }
};

// SUBROUTINE KMNS(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D, ITRAN, LIVE,
// ITER, WSS, IFAULT) as R's do_one() calls it: ITRAN(1) = 50 M quick-transfer
// steps at most, IFAULT = 0 (no trace). `c` holds the initial centres and
// returns the final ones.
Run kmns(const double* a, int m, int n, std::vector<double> c, int k, int iter) {
    Run out;
    out.ifault = 3;
    if (k <= 1 || k >= m) return out;
    out.ifault = 0;
    int imaxqtr = static_cast<int>(std::min<i64>(std::numeric_limits<int>::max(), 50 * static_cast<i64>(m)));

    Kmns s{a, m, n, k, c.data(), {}, {}, {}, {}, {}, {}, {}, {}, {}};
    s.ic1.assign(static_cast<std::size_t>(m) + 1, 0);
    s.ic2.assign(static_cast<std::size_t>(m) + 1, 0);
    s.d.assign(static_cast<std::size_t>(m) + 1, 0.0);
    s.nc.assign(static_cast<std::size_t>(k) + 1, 0);
    s.ncp.assign(static_cast<std::size_t>(k) + 1, 0);
    s.itran.assign(static_cast<std::size_t>(k) + 1, 0);
    s.live.assign(static_cast<std::size_t>(k) + 1, 0);
    s.an1.assign(static_cast<std::size_t>(k) + 1, 0.0);
    s.an2.assign(static_cast<std::size_t>(k) + 1, 0.0);

    // For each point I, find its two closest centres, IC1(I) and IC2(I).
    // Assign it to IC1(I).
    for (int i = 1; i <= m; ++i) {
        s.ic1[i] = 1;
        s.ic2[i] = 2;
        double dt[3];
        for (int il = 1; il <= 2; ++il) {
            dt[il] = kZero;
            for (int j = 1; j <= n; ++j) {
                const double da = s.A(i, j) - s.C(il, j);
                dt[il] = dt[il] + da * da;
            }
        }
        if (dt[1] > dt[2]) {
            s.ic1[i] = 2;
            s.ic2[i] = 1;
            const double temp = dt[1];
            dt[1] = dt[2];
            dt[2] = temp;
        }
        for (int l = 3; l <= k; ++l) {
            double db = kZero;
            bool skip = false;
            for (int j = 1; j <= n; ++j) {
                const double dc = s.A(i, j) - s.C(l, j);
                db = db + dc * dc;
                if (db >= dt[2]) {
                    skip = true;  // GO TO 50
                    break;
                }
            }
            if (skip) continue;
            if (db >= dt[1]) {
                dt[2] = db;
                s.ic2[i] = l;
            } else {
                dt[2] = dt[1];
                s.ic2[i] = s.ic1[i];
                dt[1] = db;
                s.ic1[i] = l;
            }
        }
    }

    // Update cluster centres to be the average of points contained within them.
    for (int l = 1; l <= k; ++l) {
        s.nc[l] = 0;
        for (int j = 1; j <= n; ++j) s.C(l, j) = kZero;
    }
    for (int i = 1; i <= m; ++i) {
        const int l = s.ic1[i];
        s.nc[l] = s.nc[l] + 1;
        for (int j = 1; j <= n; ++j) s.C(l, j) = s.C(l, j) + s.A(i, j);
    }

    // Check to see if there is any empty cluster at this stage; initialise
    // AN1 = NC / (NC - 1), AN2 = NC / (NC + 1), ITRAN and NCP.
    for (int l = 1; l <= k; ++l) {
        if (s.nc[l] == 0) {
            out.ifault = 1;
            return out;
        }
        const double aa = s.nc[l];
        for (int j = 1; j <= n; ++j) s.C(l, j) = s.C(l, j) / aa;
        s.an2[l] = aa / (aa + kOne);
        s.an1[l] = kBig;
        if (aa > kOne) s.an1[l] = aa / (aa - kOne);
        s.itran[l] = 1;
        s.ncp[l] = -1;
    }

    int indx = 0;
    int ij = 1;
    bool done = false;
    for (; ij <= iter; ++ij) {
        // Optimal-transfer stage: one pass through the data.
        s.optra(indx);
        // Stop if no transfer took place in the last M optimal transfer steps.
        if (indx == m) {
            done = true;
            break;
        }
        // Quick-transfer stage.
        s.qtran(indx, imaxqtr);
        if (imaxqtr < 0) {
            out.ifault = 4;
            done = true;
            break;
        }
        // With only two clusters there is no need to re-enter OPTRA.
        if (k == 2) {
            done = true;
            break;
        }
        // NCP has to be set to 0 before entering OPTRA.
        for (int l = 1; l <= k; ++l) s.ncp[l] = 0;
    }
    // The DO loop ran out: IJ = ITER + 1 and IFAULT = 2.
    if (!done) out.ifault = 2;
    // 150
    out.iter = ij;

    // Compute within-cluster sum of squares for each cluster.
    out.wss.assign(static_cast<std::size_t>(k), kZero);
    auto wss = [&](int l) -> double& { return out.wss[static_cast<std::size_t>(l) - 1]; };  // WSS(1..K)
    for (int l = 1; l <= k; ++l)
        for (int j = 1; j <= n; ++j) s.C(l, j) = kZero;
    for (int i = 1; i <= m; ++i) {
        const int ii = s.ic1[i];
        for (int j = 1; j <= n; ++j) s.C(ii, j) = s.C(ii, j) + s.A(i, j);
    }
    for (int j = 1; j <= n; ++j) {
        for (int l = 1; l <= k; ++l) s.C(l, j) = s.C(l, j) / static_cast<double>(s.nc[l]);
        for (int i = 1; i <= m; ++i) {
            const int ii = s.ic1[i];
            const double da = s.A(i, j) - s.C(ii, j);
            wss(ii) = wss(ii) + da * da;
        }
    }

    out.cluster.assign(s.ic1.begin() + 1, s.ic1.end());
    out.size.assign(s.nc.begin() + 1, s.nc.end());
    out.centers = std::move(c);
    return out;
}

// -- kmeans.c --------------------------------------------------------------------

// kmeans_Lloyd(x, n, p, cen, k, cl, maxiter, nc, wss)
Run lloyd(const double* x, int n, int p, std::vector<double> cen, int k, int maxiter) {
    Run out;
    std::vector<int> cl(static_cast<std::size_t>(n), -1);
    std::vector<int> nc(static_cast<std::size_t>(k), 0);
    int iter, inew = 0;
    for (iter = 0; iter < maxiter; iter++) {
        bool updated = false;
        for (int i = 0; i < n; i++) {
            // find nearest centre for each point
            double best = std::numeric_limits<double>::infinity();
            for (int j = 0; j < k; j++) {
                double dd = 0.0;
                for (int c = 0; c < p; c++) {
                    const double tmp = x[i + static_cast<i64>(n) * c] - cen[j + static_cast<std::size_t>(k) * c];
                    dd += tmp * tmp;
                }
                if (dd < best) {
                    best = dd;
                    inew = j + 1;
                }
            }
            if (cl[i] != inew) {
                updated = true;
                cl[i] = inew;
            }
        }
        if (!updated) break;
        // update each centre
        for (int j = 0; j < k * p; j++) cen[j] = 0.0;
        for (int j = 0; j < k; j++) nc[j] = 0;
        for (int i = 0; i < n; i++) {
            const int it = cl[i] - 1;
            nc[it]++;
            for (int c = 0; c < p; c++) cen[it + static_cast<std::size_t>(c) * k] += x[i + static_cast<i64>(c) * n];
        }
        for (int j = 0; j < k * p; j++) cen[j] /= nc[j % k];
    }
    out.iter = iter + 1;
    out.wss.assign(static_cast<std::size_t>(k), 0.0);
    for (int i = 0; i < n; i++) {
        const int it = cl[i] - 1;
        for (int c = 0; c < p; c++) {
            const double tmp = x[i + static_cast<i64>(n) * c] - cen[it + static_cast<std::size_t>(k) * c];
            out.wss[it] += tmp * tmp;
        }
    }
    out.cluster = std::move(cl);
    out.size = std::move(nc);
    out.centers = std::move(cen);
    return out;
}

// kmeans_MacQueen(x, n, p, cen, k, cl, maxiter, nc, wss)
Run macqueen(const double* x, int n, int p, std::vector<double> cen, int k, int maxiter) {
    Run out;
    std::vector<int> cl(static_cast<std::size_t>(n), 0);
    std::vector<int> nc(static_cast<std::size_t>(k), 0);
    int iter, inew = 0, iold;
    auto X = [&](int i, int c) { return x[i + static_cast<i64>(n) * c]; };
    auto CEN = [&](int j, int c) -> double& { return cen[j + static_cast<std::size_t>(k) * c]; };

    // first assign each point to the nearest cluster centre
    for (int i = 0; i < n; i++) {
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < k; j++) {
            double dd = 0.0;
            for (int c = 0; c < p; c++) {
                const double tmp = X(i, c) - CEN(j, c);
                dd += tmp * tmp;
            }
            if (dd < best) {
                best = dd;
                inew = j + 1;
            }
        }
        if (cl[i] != inew) cl[i] = inew;
    }
    // and recompute centres as centroids
    for (int j = 0; j < k * p; j++) cen[j] = 0.0;
    for (int j = 0; j < k; j++) nc[j] = 0;
    for (int i = 0; i < n; i++) {
        const int it = cl[i] - 1;
        nc[it]++;
        for (int c = 0; c < p; c++) CEN(it, c) += X(i, c);
    }
    for (int j = 0; j < k * p; j++) cen[j] /= nc[j % k];

    for (iter = 0; iter < maxiter; iter++) {
        bool updated = false;
        for (int i = 0; i < n; i++) {
            double best = std::numeric_limits<double>::infinity();
            for (int j = 0; j < k; j++) {
                double dd = 0.0;
                for (int c = 0; c < p; c++) {
                    const double tmp = X(i, c) - CEN(j, c);
                    dd += tmp * tmp;
                }
                if (dd < best) {
                    best = dd;
                    inew = j;
                }
            }
            if ((iold = cl[i] - 1) != inew) {
                updated = true;
                cl[i] = inew + 1;
                nc[iold]--;
                nc[inew]++;
                // update old and new cluster centres
                for (int c = 0; c < p; c++) {
                    CEN(iold, c) += (CEN(iold, c) - X(i, c)) / nc[iold];
                    CEN(inew, c) += (X(i, c) - CEN(inew, c)) / nc[inew];
                }
            }
        }
        if (!updated) break;
    }
    out.iter = iter + 1;
    out.wss.assign(static_cast<std::size_t>(k), 0.0);
    for (int i = 0; i < n; i++) {
        const int it = cl[i] - 1;
        for (int c = 0; c < p; c++) {
            const double tmp = X(i, c) - CEN(it, c);
            out.wss[it] += tmp * tmp;
        }
    }
    out.cluster = std::move(cl);
    out.size = std::move(nc);
    out.centers = std::move(cen);
    return out;
}

// -- unique.matrix() ---------------------------------------------------------------

// A row as hashable bytes: the doubles with -0 as 0 and one pattern for NA
// and one for the other NaNs (unique.c: rhash / requal).
std::string row_bytes(const double* x, i64 m, i64 p, i64 i) {
    std::string key(static_cast<std::size_t>(p) * sizeof(double), '\0');
    for (i64 j = 0; j < p; ++j) {
        double v = x[j * m + i];
        if (v == 0.0) v = 0.0;
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        if (std::isnan(v)) bits = static_cast<std::uint32_t>(bits) == 1954u ? 1u : 2u;  // NA_real_ / NaN
        std::memcpy(&key[static_cast<std::size_t>(j) * sizeof bits], &bits, sizeof bits);
    }
    return key;
}

// -- kmeans.R ----------------------------------------------------------------------

Run run_one(const double* x, int m, int p, std::vector<double> centers, int k, int iter_max, int nmeth) {
    switch (nmeth) {
        case 1:
            return kmns(x, m, p, std::move(centers), k, iter_max);
        case 2:
            return lloyd(x, m, p, std::move(centers), k, iter_max);
        default:
            return macqueen(x, m, p, std::move(centers), k, iter_max);
    }
}

// The R-level part of do_one(): errors (thrown) and warnings for a finished run.
void finish_one(Run& z, int nmeth, int m, int iter_max, std::vector<std::string>& warnings) {
    const int isteps_qtran = static_cast<int>(std::min<i64>(std::numeric_limits<int>::max(), 50 * static_cast<i64>(m)));
    if (nmeth == 1) {
        switch (z.ifault) {
            case 1:
                throw std::runtime_error("empty cluster: try a better set of initial centers");
            case 2:
                z.iter = std::max(z.iter, iter_max + 1);
                break;
            case 3:
                throw std::invalid_argument("number of cluster centres must lie between 1 and nrow(x)");
            case 4:
                warnings.push_back("Quick-TRANSfer stage steps exceeded maximum (= " + std::to_string(isteps_qtran) + ")");
                break;
            default:
                break;
        }
    } else {
        z.ifault = -1;  // the .C() result has no ifault: NULL
    }
    const bool m23 = nmeth == 2 || nmeth == 3;
    auto any_empty = [&] {
        for (const int s : z.size)
            if (s == 0) return true;
        return false;
    };
    if (m23 && any_empty()) warnings.emplace_back("empty cluster: try a better set of initial centers");
    if (z.iter > iter_max) {
        warnings.push_back("did not converge in " + std::to_string(iter_max) +
                           (iter_max == 1 ? " iteration" : " iterations"));
        if (m23) z.ifault = 2;
    }
    // (kmeans.R tests for empty clusters twice)
    if (m23 && any_empty()) warnings.emplace_back("empty cluster: try a better set of initial centers");
}

void check_data(const double* x, i64 m, i64 p) {
    if (m < 1 || p < 1) throw std::invalid_argument("x must have at least one row and one column");
    if (m > std::numeric_limits<int>::max() || p > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid nrow(x)");
    for (i64 i = 0; i < m * p; ++i)
        if (!std::isfinite(x[i])) throw std::invalid_argument("NA/NaN/Inf in foreign function call (arg 1)");
}

// Rows `rows` (0-based) of the column-major m x p matrix x, as a k x p matrix.
std::vector<double> take_rows(const double* x, i64 m, i64 p, const std::vector<i64>& rows) {
    const i64 k = static_cast<i64>(rows.size());
    std::vector<double> out(static_cast<std::size_t>(k * p));
    for (i64 j = 0; j < p; ++j)
        for (i64 r = 0; r < k; ++r) out[static_cast<std::size_t>(j * k + r)] = x[j * m + rows[static_cast<std::size_t>(r)]];
    return out;
}

// Runs the starts (independent once their centres are drawn, so in parallel)
// and keeps R's choice: the first start, replaced by a later one only if its
// sum(wss) (R's 80-bit sum) is strictly smaller. When a start stops with an
// error, `rng` is put back to `states[s]`, its state after that start's draw,
// where R's stream is when it stops.
KmeansResult run_starts(const double* x, i64 m, i64 p, const std::vector<std::vector<double>>& starts, int k,
                        int iter_max, int nmeth, const std::vector<nlm::RRng>* states = nullptr,
                        nlm::RRng* rng = nullptr) {
    const int ns = static_cast<int>(starts.size());
    std::vector<Run> runs(static_cast<std::size_t>(ns));
#pragma omp parallel for schedule(dynamic, 1) if (ns > 1)
    for (int s = 0; s < ns; ++s)
        runs[static_cast<std::size_t>(s)] = run_one(x, static_cast<int>(m), static_cast<int>(p),
                                                    starts[static_cast<std::size_t>(s)], k, iter_max, nmeth);

    KmeansResult res;
    int best_start = 0;
    double best = 0.0;
    for (int s = 0; s < ns; ++s) {
        Run& z = runs[static_cast<std::size_t>(s)];
        try {
            finish_one(z, nmeth, static_cast<int>(m), iter_max, res.warnings);
        } catch (...) {
            if (states && rng) *rng = (*states)[static_cast<std::size_t>(s)];
            throw;
        }
        // sum(ZZ$wss): R's 80-bit sum
        const double total =
            lsm::sum_of(0, static_cast<i64>(z.wss.size()), [&](i64 l) { return z.wss[static_cast<std::size_t>(l)]; });
        if (s == 0 || total < best) {
            best_start = s;
            best = total;
        }
    }
    Run& z = runs[static_cast<std::size_t>(best_start)];
    res.cluster = std::move(z.cluster);
    res.centers = std::move(z.centers);
    res.withinss = std::move(z.wss);
    res.size = std::move(z.size);
    res.iter = z.iter;
    res.ifault = z.ifault;
    res.tot_withinss = best;

    // totss <- sum(scale(x, scale = FALSE)^2): colMeans() (80-bit sum / n),
    // x - centre, squared, R's 80-bit sum in column-major order.
    std::vector<double> centre(static_cast<std::size_t>(p));
    for (i64 j = 0; j < p; ++j) {
        lsm::Extended sum = 0.0;
        for (i64 i = 0; i < m; ++i) sum += x[j * m + i];
        centre[static_cast<std::size_t>(j)] = static_cast<double>(sum / lsm::Extended(static_cast<std::int64_t>(m)));
    }
    res.totss = lsm::sum_of(0, m * p, [&](i64 t) {
        const double v = x[t] - centre[static_cast<std::size_t>(t / m)];
        return v * v;
    });
    res.betweenss = res.totss - best;
    return res;
}

int method_number(KmeansAlgorithm algorithm, int k) {
    if (k == 1) return 3;  // Hartigan-Wong (Fortran) needs k > 1: MacQueen
    return static_cast<int>(algorithm);
}

}  // namespace

std::vector<bool> r_duplicated_rows(const double* x, i64 m, i64 p) {
    std::vector<bool> dup(static_cast<std::size_t>(m), false);
    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<std::size_t>(m));
    for (i64 i = 0; i < m; ++i) dup[static_cast<std::size_t>(i)] = !seen.insert(row_bytes(x, m, p, i)).second;
    return dup;
}

std::vector<i64> r_unique_rows(const double* x, i64 m, i64 p) {
    const std::vector<bool> dup = r_duplicated_rows(x, m, p);
    std::vector<i64> keep;
    for (i64 i = 0; i < m; ++i)
        if (!dup[static_cast<std::size_t>(i)]) keep.push_back(i);
    return keep;
}

KmeansResult kmeans(const double* x, i64 m, i64 p, int k, int iter_max, int nstart, KmeansAlgorithm algorithm,
                    nlm::RRng& rng) {
    check_data(x, m, p);
    if (nstart < 1) throw std::invalid_argument("'nstart' must be at least 1");
    if (k < 0) throw std::invalid_argument("invalid 'size' argument");

    std::vector<std::vector<double>> starts;
    std::vector<nlm::RRng> states;  // the stream after each start's draw
    std::vector<double> cn;  // unique(x), mm x p
    i64 mm = 0;
    bool have_cn = false;
    auto draw_unique = [&] {
        std::vector<i64> rows = rng.sample_int(mm, k, false);
        for (i64& r : rows) --r;
        return take_rows(cn.data(), mm, p, rows);
    };
    auto make_unique = [&] {
        const std::vector<i64> keep = r_unique_rows(x, m, p);
        mm = static_cast<i64>(keep.size());
        if (mm < k) throw std::invalid_argument("more cluster centers than distinct data points.");
        cn = take_rows(x, m, p, keep);
        have_cn = true;
    };

    // "we need to avoid duplicates here"
    if (nstart == 1) {
        if (k > m) throw std::invalid_argument("cannot take a sample larger than the population when 'replace = FALSE'");
        std::vector<i64> rows = rng.sample_int(m, k, false);
        for (i64& r : rows) --r;
        std::vector<double> centers = take_rows(x, m, p, rows);
        bool dup = false;
        for (const bool d : r_duplicated_rows(centers.data(), k, p)) dup = dup || d;
        if (dup) {
            make_unique();
            centers = draw_unique();
        }
        starts.push_back(std::move(centers));
    } else {
        make_unique();
        starts.push_back(draw_unique());
    }
    states.push_back(rng);
    if (k < 1) throw std::invalid_argument("number of cluster centres must lie between 1 and nrow(x)");
    if (iter_max < 1) throw std::invalid_argument("'iter.max' must be positive");
    if (nstart >= 2 && have_cn)
        for (int i = 2; i <= nstart; ++i) {
            starts.push_back(draw_unique());
            states.push_back(rng);
        }

    return run_starts(x, m, p, starts, k, iter_max, method_number(algorithm, k), &states, &rng);
}

KmeansResult kmeans(const double* x, i64 m, i64 p, const double* centers, int k, int iter_max,
                    KmeansAlgorithm algorithm) {
    check_data(x, m, p);
    if (k < 1) throw std::invalid_argument("number of cluster centres must lie between 1 and nrow(x)");
    for (const bool d : r_duplicated_rows(centers, k, p))
        if (d) throw std::invalid_argument("initial centers are not distinct");
    if (m < k) throw std::invalid_argument("more cluster centers than data points");
    if (iter_max < 1) throw std::invalid_argument("'iter.max' must be positive");
    for (i64 i = 0; i < static_cast<i64>(k) * p; ++i)
        if (!std::isfinite(centers[i])) throw std::invalid_argument("NA/NaN/Inf in foreign function call (arg 4)");
    std::vector<std::vector<double>> starts{std::vector<double>(centers, centers + static_cast<i64>(k) * p)};
    return run_starts(x, m, p, starts, k, iter_max, method_number(algorithm, k));
}

}  // namespace sdm
