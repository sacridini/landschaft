// SPDX-License-Identifier: GPL-3.0-or-later
//
// Many metrics of many landscapes: the long table of calculate_lsm, batches
// of rasters (layers, sample plots) and moving windows (window_lsm).
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

#include "lsm/landscape.hpp"


namespace lsm {

namespace {

constexpr i64 kLargeRaster = i64(1) << 16;  // cells from which a raster is computed in parallel on its own

void push_row(LongTable& t, std::int32_t item, std::int16_t metric, std::int32_t cls, bool cls_na, std::int32_t id,
              bool id_na, double value) {
    t.item.push_back(item);
    t.metric.push_back(metric);
    t.cls.push_back(cls);
    t.cls_na.push_back(cls_na);
    t.id.push_back(id);
    t.id_na.push_back(id_na);
    t.value.push_back(value);
}

// Exceptions must not leave an OpenMP region: the first one is kept and
// thrown again afterwards.
class FirstError {
public:
    template <class F>
    void run(F&& f) {
        try {
            f();
        } catch (...) {
#pragma omp critical(lsm_first_error)
            if (!error_) error_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::exception_ptr error_;
};

}  // namespace

void LongTable::append(const LongTable& o) {
    item.insert(item.end(), o.item.begin(), o.item.end());
    metric.insert(metric.end(), o.metric.begin(), o.metric.end());
    cls.insert(cls.end(), o.cls.begin(), o.cls.end());
    cls_na.insert(cls_na.end(), o.cls_na.begin(), o.cls_na.end());
    id.insert(id.end(), o.id.begin(), o.id.end());
    id_na.insert(id_na.end(), o.id_na.begin(), o.id_na.end());
    value.insert(value.end(), o.value.begin(), o.value.end());
}

void append_long_table(Landscape& l, const std::vector<MetricSpec>& specs, std::int32_t item, LongTable& out) {
    const bool empty = l.n_classes() == 0;
    const std::vector<std::int32_t>& values = l.class_values();
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const auto m = static_cast<std::int16_t>(s);
        const MetricSpec& spec = specs[s];
        if (empty) {
            push_row(out, item, m, 0, true, 0, true, kNaN);
            continue;
        }
        if (spec.level == kPatch) {
            const std::vector<double> v = l.patch_metric(spec.name);
            const Patches& pt = l.patches();
            for (std::size_t i = 0; i < v.size(); ++i) {
                push_row(out, item, m, values[pt.cls[i]], false, static_cast<std::int32_t>(i + 1), false, v[i]);
            }
        } else if (spec.level == kClass) {
            const std::vector<double> v = l.class_metric(spec.name);
            const bool all_na = std::all_of(v.begin(), v.end(), [](double x) { return std::isnan(x); });
            if (spec.name.compare(0, 4, "enn_") == 0 && all_na) {
                // every class has a single patch: landscapemetrics gives one row without class
                push_row(out, item, m, 0, true, 0, true, kNaN);
                continue;
            }
            for (std::size_t c = 0; c < v.size(); ++c) push_row(out, item, m, values[c], false, 0, true, v[c]);
        } else {
            push_row(out, item, m, 0, true, 0, true, l.landscape_metric(spec.name));
        }
    }
}

LongTable batch_long_table(const std::vector<RasterView>& rasters, const Options& options,
                           const std::vector<MetricSpec>& specs) {
    const auto n = static_cast<i64>(rasters.size());
    std::vector<LongTable> parts(static_cast<std::size_t>(n));
    auto compute = [&](i64 i) {
        const RasterView& r = rasters[i];
        Landscape l(r.values, r.na_mask, r.nx, r.ny, r.resx, r.resy, options);
        append_long_table(l, specs, static_cast<std::int32_t>(i), parts[i]);
    };
    FirstError error;
#pragma omp parallel for schedule(dynamic)
    for (i64 i = 0; i < n; ++i) {
        if (rasters[i].nx * rasters[i].ny < kLargeRaster) error.run([&] { compute(i); });
    }
    error.rethrow();
    for (i64 i = 0; i < n; ++i) {
        if (rasters[i].nx * rasters[i].ny >= kLargeRaster) compute(i);
    }
    LongTable out;
    std::size_t total = 0;
    for (const LongTable& p : parts) total += p.size();
    out.item.reserve(total);
    out.metric.reserve(total);
    out.cls.reserve(total);
    out.cls_na.reserve(total);
    out.id.reserve(total);
    out.id_na.reserve(total);
    out.value.reserve(total);
    for (LongTable& p : parts) {
        out.append(p);
        p = LongTable{};
    }
    return out;
}

std::vector<double> window_metrics(const RasterView& raster, const std::uint8_t* window, i64 wx, i64 wy,
                                   const std::vector<std::string>& names, const Options& options) {
    if (wx % 2 == 0 || wy % 2 == 0 || wx < 1 || wy < 1) throw std::invalid_argument("the window must have odd sides");
    const i64 nx = raster.nx;
    const i64 ny = raster.ny;
    const i64 n = nx * ny;
    const auto m = static_cast<i64>(names.size());
    // Unknown names fail here, outside the parallel region.
    {
        const std::int32_t one = 0;
        Landscape probe(&one, nullptr, 1, 1, raster.resx, raster.resy, options);
        for (const std::string& name : names) probe.landscape_metric(name);
    }
    // Class values as dense indices (same order): windows then need no
    // large look-up tables.
    std::vector<std::int32_t> dense(static_cast<std::size_t>(n), 0);
    std::vector<std::uint8_t> na(static_cast<std::size_t>(n), 0);
    {
        std::vector<std::int32_t> classes;
        for (i64 p = 0; p < n; ++p) {
            if (raster.na_mask && raster.na_mask[p]) {
                na[p] = 1;
            } else {
                classes.push_back(raster.values[p]);
            }
        }
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
#pragma omp parallel for schedule(static) if (n >= kLargeRaster)
        for (i64 p = 0; p < n; ++p) {
            if (!na[p]) {
                dense[p] = static_cast<std::int32_t>(std::lower_bound(classes.begin(), classes.end(), raster.values[p]) -
                                                     classes.begin());
            }
        }
    }
    std::vector<double> out(static_cast<std::size_t>(m * n));
    const i64 hx = wx / 2;
    const i64 hy = wy / 2;
    FirstError error;
#pragma omp parallel
    {
        std::vector<std::int32_t> buf(static_cast<std::size_t>(wx * wy));
        std::vector<std::uint8_t> buf_na(static_cast<std::size_t>(wx * wy));
#pragma omp for schedule(dynamic)
        for (i64 y = 0; y < ny; ++y) {
            error.run([&] {
                for (i64 x = 0; x < nx; ++x) {
                    for (i64 j = 0; j < wy; ++j) {
                        const i64 yy = y - hy + j;
                        for (i64 i = 0; i < wx; ++i) {
                            const i64 xx = x - hx + i;
                            const i64 b = j * wx + i;
                            const bool in = window[b] && yy >= 0 && yy < ny && xx >= 0 && xx < nx && !na[yy * nx + xx];
                            buf_na[b] = !in;
                            buf[b] = in ? dense[yy * nx + xx] : 0;
                        }
                    }
                    Landscape l(buf.data(), buf_na.data(), wx, wy, raster.resx, raster.resy, options);
                    for (i64 k = 0; k < m; ++k) out[k * n + y * nx + x] = l.landscape_metric(names[k]);
                }
            });
        }
    }
    error.rethrow();
    return out;
}

}  // namespace lsm
