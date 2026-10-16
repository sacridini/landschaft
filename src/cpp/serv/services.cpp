// SPDX-License-Identifier: GPL-3.0-or-later
//
// Carbon storage and habitat quality (InVEST), see services.hpp.

#include "serv/services.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace serv {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr i64 kBlock = 1 << 16;  // cells per block of the sums: the same order with any thread count

}  // namespace

i64 classify(const i64* lulc, const std::uint8_t* valid, i64 n, const i64* codes, int k, std::int32_t* out,
             int threads) {
    // A dense table when the codes span a short range, else a binary search.
    const i64 lo = k ? codes[0] : 0, hi = k ? codes[k - 1] : -1;
    const bool dense = k && hi - lo < (i64{1} << 22);
    std::vector<std::int32_t> table;
    if (dense) {
        table.assign(static_cast<std::size_t>(hi - lo + 1), -2);
        for (int c = 0; c < k; ++c) table[static_cast<std::size_t>(codes[c] - lo)] = c;
    }
    int unknown = 0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(| : unknown)
    for (i64 i = 0; i < n; ++i) {
        if (!valid[i]) {
            out[i] = -1;
            continue;
        }
        const i64 v = lulc[i];
        std::int32_t c = -2;
        if (dense) {
            if (v >= lo && v <= hi) c = table[static_cast<std::size_t>(v - lo)];
        } else {
            const i64* p = std::lower_bound(codes, codes + k, v);
            if (p != codes + k && *p == v) c = static_cast<std::int32_t>(p - codes);
        }
        out[i] = c;
        if (c == -2) unknown = 1;
    }
    if (unknown)
        for (i64 i = 0; i < n; ++i)
            if (out[i] == -2) return i;
    return -1;
}

void carbon_pools(const std::int32_t* cls, i64 n, const float* pools, float* const out[5], int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const std::int32_t c = cls[i];
        if (c < 0) {
            for (int p = 0; p < 5; ++p) out[p][i] = kNaN;
            continue;
        }
        const float* v = pools + 4 * static_cast<i64>(c);
        // numpy.sum([above, below, soil, dead], axis=0) on float32: one after the other
        float s = v[0];
        s = s + v[1];
        s = s + v[2];
        s = s + v[3];
        out[0][i] = v[0];
        out[1][i] = v[1];
        out[2][i] = v[2];
        out[3][i] = v[3];
        out[4][i] = s;
    }
}

double total(const float* x, i64 n, int threads) {
    const i64 blocks = (n + kBlock - 1) / kBlock;
    std::vector<double> part(static_cast<std::size_t>(blocks), 0.0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 b = 0; b < blocks; ++b) {
        double s = 0.0;
        const i64 end = std::min(n, (b + 1) * kBlock);
        for (i64 i = b * kBlock; i < end; ++i)
            if (!std::isnan(x[i])) s += static_cast<double>(x[i]);
        part[static_cast<std::size_t>(b)] = s;
    }
    double s = 0.0;
    for (double p : part) s += p;
    return s;
}

void distance_transform(const std::uint8_t* region, int ny, int nx, float* dist, int threads) {
    const i64 n = static_cast<i64>(ny) * nx;
    bool any = false;
    for (i64 i = 0; i < n && !any; ++i) any = region[i] != 0;
    if (!any) {
        std::fill(dist, dist + n, std::numeric_limits<float>::infinity());
        return;
    }
    // Phase 1 (columns): g = vertical distance to the region, in float32.
    const float inf = static_cast<float>(nx + ny);  // sample distances 1: max(1, 1) * max(1, 1) * (nx + ny)
    std::vector<float> g(static_cast<std::size_t>(n));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int x = 0; x < nx; ++x) {
        g[static_cast<std::size_t>(x)] = region[x] == 0 ? inf : 0.0f;
        for (int y = 1; y < ny; ++y) {
            const i64 i = static_cast<i64>(y) * nx + x;
            g[static_cast<std::size_t>(i)] = region[i] ? 0.0f : g[static_cast<std::size_t>(i - nx)] + 1.0f;
        }
        for (int y = ny - 2; y >= 0; --y) {
            const i64 i = static_cast<i64>(y) * nx + x;
            const float below = g[static_cast<std::size_t>(i + nx)];
            if (below < g[static_cast<std::size_t>(i)]) g[static_cast<std::size_t>(i)] = 1.0f + below;
        }
    }
    // Phase 2 (rows): the lower envelope of the parabolas, as pygeoprocessing's Cython does it.
#pragma omp parallel num_threads(threads)
    {
        std::vector<int> s(static_cast<std::size_t>(nx)), t(static_cast<std::size_t>(nx));
#pragma omp for schedule(static)
        for (int y = 0; y < ny; ++y) {
            const float* gr = g.data() + static_cast<i64>(y) * nx;
            const std::uint8_t* mr = region + static_cast<i64>(y) * nx;
            float* dr = dist + static_cast<i64>(y) * nx;
            int q = 0;
            s[0] = 0;
            t[0] = 0;
            int sq = 0;
            float gsq = 0.0f;
            for (int u = 1; u < nx; ++u) {
                const float gu = gr[u] * gr[u];
                while (q >= 0) {
                    const int tq = t[static_cast<std::size_t>(q)];
                    sq = s[static_cast<std::size_t>(q)];
                    gsq = gr[sq] * gr[sq];
                    const float a = static_cast<float>(tq - sq), b = static_cast<float>(tq - u);
                    if (a * a + gsq <= b * b + gu) break;
                    --q;
                }
                if (q < 0) {
                    q = 0;
                    s[0] = u;
                    sq = u;
                    gsq = gr[sq] * gr[sq];
                } else {
                    const float fu = static_cast<float>(u), fs = static_cast<float>(sq);
                    const float w = 1.0f + ((fu * fu - fs * fs + gu - gsq) / (2.0f * static_cast<float>(u - sq)));
                    if (w < static_cast<float>(nx)) {
                        ++q;
                        s[static_cast<std::size_t>(q)] = u;
                        t[static_cast<std::size_t>(q)] = static_cast<int>(w);
                    }
                }
            }
            sq = s[static_cast<std::size_t>(q)];
            gsq = gr[sq] * gr[sq];
            int tq = t[static_cast<std::size_t>(q)];
            for (int u = nx - 1; u >= 0; --u) {
                float d2 = 0.0f;
                if (mr[u] == 0) {
                    const float a = static_cast<float>(u - sq);
                    d2 = a * a + gsq;
                }
                dr[u] = std::sqrt(d2);
                if (u <= tq) {
                    --q;
                    if (q >= 0) {
                        sq = s[static_cast<std::size_t>(q)];
                        gsq = gr[sq] * gr[sq];
                        tq = t[static_cast<std::size_t>(q)];
                    }
                }
            }
        }
    }
}

void threat_degradation(const float* dist, const float* given, const std::int32_t* cls, const float* sens,
                        const float* access, double max_px, bool exponential, double weight, i64 n, float* deg,
                        float* total, int threads) {
    const float m = static_cast<float>(max_px);
    const float k299 = 2.99f;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const std::int32_t c = cls[i];
        const float a = access ? access[i] : 1.0f;
        if (c < 0 || std::isnan(a)) {
            deg[i] = kNaN;
            total[i] = kNaN;
            continue;
        }
        const float d = dist[i];
        float decay;
        if (given) {
            decay = given[i];
        } else if (d > m) {
            decay = 0.0f;
        } else if (exponential) {
            const float e = (-d * k299) / m;
            decay = static_cast<float>(std::exp(static_cast<double>(e)));
        } else {
            decay = (m - d) / m;
        }
        const float fs = decay * sens[c];
        const double t = static_cast<double>(fs) * weight * static_cast<double>(a);
        const float v = static_cast<float>(t);
        deg[i] = v;
        total[i] = total[i] + v;
    }
}

void quality(const float* total, const std::int32_t* cls, const float* habitat, double ksq, i64 n, float* out,
             int threads) {
    const float k = static_cast<float>(ksq);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const std::int32_t c = cls[i];
        const float d = total[i];
        if (c < 0 || std::isnan(d)) {
            out[i] = kNaN;
            continue;
        }
        const float dz = std::pow(d, 2.5f);
        out[i] = habitat[c] * (1.0f - dz / (dz + k));
    }
}

template <typename T>
void water_fraction(const std::int32_t* cls, const float* kc, const float* root, const float* veg, const T* eto,
                    const T* precip, const T* soil, const T* pawc, double z, i64 n, float* fractp, int threads) {
    const T zt = static_cast<T>(z), one = T(1), w0 = static_cast<T>(1.25), cap = T(5);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const std::int32_t c = cls[i];
        const T p = precip[i], e = eto[i], s = soil[i], w = pawc[i];
        // numpy.isclose(precip, 0): |precip| <= 1e-8
        if (c < 0 || std::isnan(p) || std::isnan(e) || std::isnan(s) || std::isnan(w) ||
            std::abs(static_cast<double>(p)) <= 1e-8) {
            fractp[i] = kNaN;
            continue;
        }
        const T k = static_cast<T>(kc[c]), r = static_cast<T>(root[c]);
        const T pet = k * e;
        const T phi = pet / p;
        const T awc = (r < s ? r : s) * w;
        T cw = ((awc / p) * zt) + w0;
        if (cw > cap) cw = cap;
        const T x = pet / p;
        const T aet_p = (one + x) - std::pow(one + std::pow(x, cw), one / cw);
        T result;
        if (static_cast<T>(veg[c]) == one) {
            result = phi < aet_p ? phi : aet_p;
        } else {
            T nonveg = pet;
            if (p < pet) nonveg = p;
            result = nonveg / p;
        }
        fractp[i] = static_cast<float>(result);
    }
}

template <typename U>
void water_yield(const float* fractp, const U* precip, i64 n, U* wyield, U* aet, int threads) {
    const U nan = std::numeric_limits<U>::quiet_NaN();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const float f = fractp[i];
        const U p = precip[i];
        if (std::isnan(f) || std::isnan(p)) {
            wyield[i] = nan;
            aet[i] = nan;
            continue;
        }
        wyield[i] = static_cast<U>(1.0f - f) * p;  // (1 - fractp) in float32, then times precip
        aet[i] = static_cast<U>(f) * p;
    }
}

template <typename V>
void potential_et(const std::int32_t* cls, const float* kc, const V* eto, i64 n, V* pet, int threads) {
    const V nan = std::numeric_limits<V>::quiet_NaN();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const std::int32_t c = cls[i];
        const V e = eto[i];
        pet[i] = (c < 0 || std::isnan(e)) ? nan : e * static_cast<V>(kc[c]);
    }
}

template void water_fraction<float>(const std::int32_t*, const float*, const float*, const float*, const float*,
                                    const float*, const float*, const float*, double, i64, float*, int);
template void water_fraction<double>(const std::int32_t*, const float*, const float*, const float*, const double*,
                                     const double*, const double*, const double*, double, i64, float*, int);
template void water_yield<float>(const float*, const float*, i64, float*, float*, int);
template void water_yield<double>(const float*, const double*, i64, double*, double*, int);
template void potential_et<float>(const std::int32_t*, const float*, const float*, i64, float*, int);
template void potential_et<double>(const std::int32_t*, const float*, const double*, i64, double*, int);

}  // namespace serv
