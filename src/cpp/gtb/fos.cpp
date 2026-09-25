// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of GraySpatCon metrics 1 and 52 (GraySpatCon.c, Kurt Riitters and
// Peter Vogt, (C) European Union (JRC), GPL-3.0-or-later) as GuidosToolbox's
// FOS runs them.
#include "fos.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gtb {

namespace {

constexpr int kNum = 102;       // byte values 0..100 and 101 = missing
constexpr int kMissing = 101;

// GraySpatCon's float metric -> byte conversion.
inline u8 to_byte(float metric, int stretch) {
    if (metric < 0.0f) return 255;  // metric is -0.01 if missing
    const double v = stretch == 1 ? static_cast<double>(metric) * 100. : static_cast<double>(metric);
    return static_cast<u8>(std::round(v));
}

inline float round5(float temp_out) {  // roundf(temp_out * 100000) / 100000
    return std::roundf(temp_out * 100000.0f) / 100000.0f;
}

// Adjacency counts of a window with the cells in use kept in bitsets, so the
// metric can visit them in GraySpatCon's (row-major) order without scanning
// all 102 x 102 cells.
struct Adjacencies {
    std::vector<std::int64_t> count = std::vector<std::int64_t>(kNum * kNum, 0);
    std::uint64_t used[kNum][2] = {};
    std::int64_t valid = 0;  // adjacencies without a missing value

    void add(int a, int b, int d) {
        std::int64_t& c = count[static_cast<std::size_t>(a * kNum + b)];
        c += d;
        if (a < kMissing && b < kMissing) valid += d;
        const std::uint64_t bit = std::uint64_t(1) << (b & 63);
        if (c) used[a][b >> 6] |= bit; else used[a][b >> 6] &= ~bit;
    }

    // Metric 52 (clustering, exclude_zero = 0).
    float clustering() const {
        if (valid == 0) return -0.01f;
        const float inv = static_cast<float>(1.0 / (1.0 * static_cast<double>(valid)));
        float numerator = 0.0f;
        for (int i = 0; i < kMissing; ++i) {
            for (int w = 0; w < 2; ++w) {
                std::uint64_t bits = used[i][w];
                while (bits) {
                    const int j = w * 64 + ctz64(bits);
                    bits &= bits - 1;
                    if (j >= kMissing) continue;
                    const std::int64_t n = count[static_cast<std::size_t>(i * kNum + j)];
                    const float tf2 = static_cast<float>((1.0 * (i + j)) / 2.0);
                    const float tf = inv * static_cast<float>(n);
                    numerator += tf * tf2;
                }
            }
        }
        const float temp_out = static_cast<float>(0.01 * static_cast<double>(numerator));
        return round5(temp_out);
    }

    static int ctz64(std::uint64_t v) {  // v != 0
        int n = 0;
        while (!(v & 1)) {
            v >>= 1;
            ++n;
        }
        return n;
    }
};

// Metric 1 (mean, exclude_zero = 0) from the pixel frequencies.
float mean_metric(const std::int64_t* freq, std::int64_t npix) {
    const std::int64_t n = npix - freq[kMissing];
    if (n == 0) return -0.01f;
    const float inv = static_cast<float>(1.0 / (1.0 * static_cast<double>(n)));
    std::int64_t sum = 0;
    for (int v = 0; v < kMissing; ++v) sum += freq[v] * v;
    const float temp_out = static_cast<float>(static_cast<double>(inv) * (1.0 * static_cast<double>(sum)));
    return round5(temp_out);
}

}  // namespace

ImageU8 grayspatcon_fos(const ImageU8& im, int metric, int window) {
    if (metric != 1 && metric != 52) throw std::invalid_argument("grayspatcon_fos: metric must be 1 or 52");
    if (window < 3 || window % 2 == 0) throw std::invalid_argument("window size must be an odd integer >= 3");
    const i64 nx = im.nx, ny = im.ny, b = (window - 1) / 2;
    // buffered input: values 0..100, 255 -> 101, frame of 101
    const i64 bx = nx + 2 * b, by = ny + 2 * b;
    std::vector<u8> buf(static_cast<std::size_t>(bx * by), kMissing);
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const u8 v = im.at(x, y);
            if (v > 100 && v != 255) throw std::invalid_argument("grayspatcon input values must be 0-100 or 255 (missing)");
            buf[static_cast<std::size_t>((y + b) * bx + x + b)] = v == 255 ? kMissing : v;
        }
    }
    auto at = [&](i64 r, i64 c) { return static_cast<int>(buf[static_cast<std::size_t>(r * bx + c)]); };
    const std::int64_t npix = static_cast<std::int64_t>(window) * window;
    const int stretch = metric == 1 ? 6 : 1;
    ImageU8 out(nx, ny);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (i64 row = 0; row < ny; ++row) {
        std::int64_t freq[kNum] = {};
        Adjacencies adj;
        const i64 r_min = row, r_max = row + window;
        i64 c_min = 0, c_max = window;
        if (metric == 1) {
            for (i64 r = r_min; r < r_max; ++r) {
                for (i64 c = c_min; c < c_max; ++c) ++freq[at(r, c)];
            }
        } else {  // (pixel, pixel below) and (pixel, pixel at right) pairs
            for (i64 r = r_min; r < r_max; ++r) {
                for (i64 c = c_min; c < c_max; ++c) {
                    if (r + 1 < r_max) adj.add(at(r, c), at(r + 1, c), 1);
                    if (c + 1 < c_max) adj.add(at(r, c), at(r, c + 1), 1);
                }
            }
        }
        for (i64 col = 0; col < nx; ++col) {
            if (col > 0) {  // slide one column to the right
                if (metric == 1) {
                    for (i64 r = r_min; r < r_max; ++r) {
                        --freq[at(r, c_min)];
                        ++freq[at(r, c_max)];
                    }
                } else {
                    for (i64 r = r_min; r < r_max; ++r) {
                        if (r + 1 < r_max) {
                            adj.add(at(r, c_min), at(r + 1, c_min), -1);
                            adj.add(at(r, c_max), at(r + 1, c_max), 1);
                        }
                        adj.add(at(r, c_min), at(r, c_min + 1), -1);
                        adj.add(at(r, c_max - 1), at(r, c_max), 1);
                    }
                }
                ++c_min;
                ++c_max;
            }
            const u8 in = im.at(col, row);
            const float value = in == 255 ? -0.01f : (metric == 1 ? mean_metric(freq, npix) : adj.clustering());
            out.at(col, row) = to_byte(value, stretch);
        }
    }
    return out;
}

}  // namespace gtb
