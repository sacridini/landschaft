// SPDX-License-Identifier: GPL-3.0-or-later
//
// Many metrics of many landscapes: the long table of calculate_lsm, batches
// of rasters (layers, sample plots) and moving windows (window_lsm).
#include <algorithm>
#include <cmath>
#include <exception>
#include <optional>
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
    if (!empty) l.prepare(specs);
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
    // Class values as dense indices (same order, -1 = NA): windows then need
    // no large look-up tables.
    std::vector<std::int32_t> dense(static_cast<std::size_t>(n), -1);
    std::vector<std::int32_t> classes;
    {
        for (i64 p = 0; p < n; ++p) {
            if (!(raster.na_mask && raster.na_mask[p])) classes.push_back(raster.values[p]);
        }
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
#pragma omp parallel for schedule(static) if (n >= kLargeRaster)
        for (i64 p = 0; p < n; ++p) {
            if (!(raster.na_mask && raster.na_mask[p])) {
                dense[p] = static_cast<std::int32_t>(std::lower_bound(classes.begin(), classes.end(), raster.values[p]) -
                                                     classes.begin());
            }
        }
    }
    const auto n_classes = static_cast<i64>(classes.size());
    std::vector<double> out(static_cast<std::size_t>(m * n));
    const i64 hx = wx / 2;
    const i64 hy = wy / 2;
    // window_lsm computes shdi and shei from the class areas of
    // 8-connected patches whatever `directions` is (their _calc functions
    // hard-code it; calculate_lsm hides it with patches it has already
    // labelled). The sums round differently: with 4 directions these metrics
    // come from an 8-direction copy of the window.
    std::vector<char> eight(static_cast<std::size_t>(m), 0);
    bool any_eight = false;
    if (options.directions == 4) {
        for (i64 k = 0; k < m; ++k) {
            eight[k] = names[k] == "shdi" || names[k] == "shei";
            any_eight = any_eight || eight[k];
        }
    }
    Options options8 = options;
    options8.directions = 8;
    // terra::focal hands window_lsm the values row by row and it fills its
    // window matrix column by column: its landscape is the window
    // transposed. Ours is too (with the cell sides swapped, so the geometry
    // stays right): the patches then come in R's order, which pafrac's
    // regression rounds by.
    std::vector<std::uint8_t> shape(static_cast<std::size_t>(wx * wy));
    for (i64 j = 0; j < wy; ++j) {
        for (i64 i = 0; i < wx; ++i) shape[i * wy + j] = window[j * wx + i] != 0;
    }
    FirstError error;
#pragma omp parallel
    {
        // One landscape per thread, refilled window after window; the
        // window's classes come out of the copy (cells per global class,
        // the classes seen), so the landscape gets them ready.
        Landscape l(raster.resy, raster.resx, options);
        std::optional<Landscape> l8;
        if (any_eight) l8.emplace(raster.resy, raster.resx, options8);
        std::vector<std::int32_t> buf(static_cast<std::size_t>(wx * wy));
        std::vector<i64> count(static_cast<std::size_t>(n_classes), 0);
        std::vector<std::int32_t> local(static_cast<std::size_t>(n_classes));
        std::vector<std::int32_t> seen, values;
        std::vector<i64> cells;
#pragma omp for schedule(dynamic)
        for (i64 y = 0; y < ny; ++y) {
            error.run([&] {
                const i64 j0 = std::max<i64>(0, hy - y);
                const i64 j1 = std::min<i64>(wy, ny - y + hy);
                for (i64 x = 0; x < nx; ++x) {
                    seen.clear();
                    for (i64 i = 0; i < wx; ++i) {
                        std::int32_t* col = buf.data() + i * wy;
                        const i64 xx = x - hx + i;
                        if (xx < 0 || xx >= nx) {
                            std::fill(col, col + wy, -1);
                            continue;
                        }
                        const std::uint8_t* in = shape.data() + i * wy;
                        std::fill(col, col + j0, -1);
                        for (i64 j = j0; j < j1; ++j) {
                            const std::int32_t v = in[j] ? dense[(y - hy + j) * nx + xx] : -1;
                            col[j] = v;
                            if (v >= 0 && count[v]++ == 0) seen.push_back(v);
                        }
                        std::fill(col + j1, col + wy, -1);
                    }
                    std::sort(seen.begin(), seen.end());
                    const auto k = seen.size();
                    values.resize(k);
                    cells.resize(k);
                    for (std::size_t c = 0; c < k; ++c) {
                        const std::int32_t g = seen[c];
                        local[g] = static_cast<std::int32_t>(c);
                        values[c] = classes[g];
                        cells[c] = count[g];
                        count[g] = 0;
                    }
                    if (k > 1) {
                        for (std::int32_t& v : buf) {
                            if (v >= 0) v = local[v];
                        }
                    } else if (k == 1) {
                        for (std::int32_t& v : buf) v = v >= 0 ? 0 : -1;
                    }
                    l.reset(buf.data(), wy, wx, values, cells);
                    if (l8) l8->reset(buf.data(), wy, wx, values, cells);
                    for (i64 q = 0; q < m; ++q) {
                        out[q * n + y * nx + x] = eight[q] ? l8->landscape_metric(names[q]) : l.landscape_metric(names[q]);
                    }
                }
            });
        }
    }
    error.rethrow();
    return out;
}

}  // namespace lsm
