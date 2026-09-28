// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR 1.4's spectral models (R code, nlm_gaussianfield.R and nlm_fbm.R):
// simulate_gaussian_field (circulant embedding of an exponential covariance)
// and simulate_fbm (a 1/f^beta spectrum). Matrices are kept column-major, as
// in R, because R's mean() and sd() accumulate in that order.

#include <cmath>
#include <stdexcept>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"
#include "nlm/rfft.hpp"

namespace nlm {

namespace {

constexpr i64 kParallel = 1 << 14;

// Column-major nrow x ncol to row-major.
std::vector<double> to_row_major(const std::vector<double>& byc, i64 nrow, i64 ncol) {
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol));
    lsm::parallel_for(nrow * ncol >= kParallel, nrow, [&](i64 r) {
        for (i64 c = 0; c < ncol; ++c)
            out[static_cast<std::size_t>(r * ncol + c)] = byc[static_cast<std::size_t>(c * nrow + r)];
    });
    return out;
}

// sqrt(s) * (re + i im) for the n cells of a real vector s, as R multiplies
// a double by a complex vector: (s + 0i)(re + i im).
void scale_complex(std::vector<double>& z, const std::vector<double>& s, const std::vector<double>& re,
                   const std::vector<double>& im) {
    const i64 n = static_cast<i64>(s.size());
    z.resize(2 * s.size());
    lsm::parallel_for(n >= kParallel, n, [&](i64 k) {
        const std::size_t i = static_cast<std::size_t>(k);
        const double a = lsm::r_sqrt(s[i]);
        z[2 * i] = a * re[i] - 0.0 * im[i];
        z[2 * i + 1] = a * im[i] + 0.0 * re[i];
    });
}

}  // namespace

std::vector<double> gaussian_field(RRng& rng, i64 nrow, i64 ncol, double autocorr_range, double mag_var, double nug,
                                   double mean) {
    // simulate_gaussian_field_exp(): the 2 nrow x 2 ncol circulant embedding.
    const i64 n1 = 2 * nrow, n2 = 2 * ncol, n = n1 * n2;
    if (n1 > 0x7fffffff / 2 || n2 > 0x7fffffff / 2 || n > 0x7fffffff) throw std::invalid_argument("raster too large");
    std::vector<double> z(2 * static_cast<std::size_t>(n));
    lsm::parallel_for(n >= kParallel, n2, [&](i64 j) {
        const double dy = static_cast<double>(std::min(j, n2 - j));
        for (i64 i = 0; i < n1; ++i) {
            const double dx = static_cast<double>(std::min(i, n1 - i));
            const double distance = lsm::r_sqrt(dx * dx + dy * dy);
            const std::size_t k = static_cast<std::size_t>(j * n1 + i);
            z[2 * k] = mag_var * lsm::r_exp(-distance / autocorr_range);
            z[2 * k + 1] = 0.0;
        }
    });
    r_fft2(z, static_cast<int>(n1), static_cast<int>(n2), false, true);
    std::vector<double> eigen(static_cast<std::size_t>(n));
    for (i64 k = 0; k < n; ++k) {
        const double e = z[2 * static_cast<std::size_t>(k)];
        eigen[static_cast<std::size_t>(k)] = e < 0 ? 0.0 : e;
    }
    std::vector<double> re(static_cast<std::size_t>(n)), im(static_cast<std::size_t>(n));
    rng.rnorm(re.data(), n);
    rng.rnorm(im.data(), n);
    scale_complex(z, eigen, re, im);
    r_fft2(z, static_cast<int>(n1), static_cast<int>(n2), true, true);
    const double denom = std::sqrt(static_cast<double>(n));
    std::vector<double> field(static_cast<std::size_t>(nrow * ncol));
    lsm::parallel_for(nrow * ncol >= kParallel, ncol, [&](i64 c) {
        for (i64 r = 0; r < nrow; ++r)
            field[static_cast<std::size_t>(c * nrow + r)] = z[2 * static_cast<std::size_t>(c * n1 + r)] / denom;
    });
    // simulate_gaussian_field(): nugget and mean
    if (nug > 0) {
        std::vector<double> e(field.size());
        rng.rnorm(e.data(), nrow * ncol, 0.0, std::sqrt(nug));
        for (std::size_t k = 0; k < field.size(); ++k) field[k] = field[k] + e[k];
    }
    for (double& v : field) v = v + mean;
    return to_row_major(field, nrow, ncol);
}

std::vector<double> fbm(RRng& rng, i64 nrow, i64 ncol, double fract_dim) {
    const i64 n = nrow * ncol;
    std::vector<double> field(static_cast<std::size_t>(n));
    if (fract_dim >= 2 - 1.4901161193847656e-08) {
        // simulate_fbm_linear_limit()
        double slopes[2];
        rng.rnorm(slopes, 2);
        for (i64 c = 0; c < ncol; ++c)
            for (i64 r = 0; r < nrow; ++r)
                field[static_cast<std::size_t>(c * nrow + r)] =
                    slopes[0] * static_cast<double>(r) + slopes[1] * static_cast<double>(c);
    } else {
        // simulate_fbm_spectral()
        if (n > 0x7fffffff) throw std::invalid_argument("raster too large");
        auto freq = [](i64 len, i64 k) { return k <= len / 2 ? k : k - len; };
        const double expo = -(fract_dim + 1) / 2;
        std::vector<double> spectrum(static_cast<std::size_t>(n));
        lsm::parallel_for(n >= kParallel, ncol, [&](i64 c) {
            const double kx = static_cast<double>(freq(ncol, c));
            for (i64 r = 0; r < nrow; ++r) {
                const double ky = static_cast<double>(freq(nrow, r));
                const double s = kx * kx + ky * ky;
                spectrum[static_cast<std::size_t>(c * nrow + r)] = lsm::r_pow(s < 1 ? 1.0 : s, expo);
            }
        });
        spectrum[0] = 0;
        std::vector<double> re(static_cast<std::size_t>(n)), im(static_cast<std::size_t>(n));
        rng.rnorm(re.data(), n);
        rng.rnorm(im.data(), n);
        std::vector<double> z;
        scale_complex(z, spectrum, re, im);
        r_fft2(z, static_cast<int>(nrow), static_cast<int>(ncol), true, true);
        const double denom = std::sqrt(static_cast<double>(nrow) * static_cast<double>(ncol));
        lsm::parallel_for(n >= kParallel, n, [&](i64 k) {
            field[static_cast<std::size_t>(k)] = z[2 * static_cast<std::size_t>(k)] / denom;
        });
    }
    // field - mean(field), then / sd of the unit increments (row steps, then
    // column steps, each column-major)
    const double m = lsm::r_mean(field.data(), n);
    for (double& v : field) v = v - m;
    std::vector<double> diffs;
    diffs.reserve(static_cast<std::size_t>(2 * n));
    if (nrow > 1)
        for (i64 c = 0; c < ncol; ++c)
            for (i64 r = 0; r + 1 < nrow; ++r)
                diffs.push_back(field[static_cast<std::size_t>(c * nrow + r + 1)] - field[static_cast<std::size_t>(c * nrow + r)]);
    if (ncol > 1)
        for (i64 c = 0; c + 1 < ncol; ++c)
            for (i64 r = 0; r < nrow; ++r)
                diffs.push_back(field[static_cast<std::size_t>((c + 1) * nrow + r)] - field[static_cast<std::size_t>(c * nrow + r)]);
    double sd = 1.0;
    if (!diffs.empty()) {
        const i64 nd = static_cast<i64>(diffs.size());
        sd = lsm::r_sd(diffs.data(), nd, lsm::r_mean(diffs.data(), nd));
    }
    for (double& v : field) v = v / sd;
    return to_row_major(field, nrow, ncol);
}

}  // namespace nlm
