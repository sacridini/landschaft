// SPDX-License-Identifier: GPL-3.0-or-later
//
// Metric formulas. Each one follows the corresponding lsm_{p,c,l}_* function
// of landscapemetrics, including its conventions (sample standard deviation,
// NA handling, which neighbours count as edge).
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "lsm/extended.hpp"
#include "lsm/landscape.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace lsm {

using namespace std::string_view_literals;

namespace {

// R's median(): the middle value, or the mean() of the two middle values.
double r_median(const double* x, i64 n) {
    if (n == 0) return kNaN;
    std::vector<double> v(x, x + n);
    if (std::any_of(v.begin(), v.end(), [](double a) { return std::isnan(a); })) return kNaN;
    const i64 half = (n + 1) / 2;
    std::nth_element(v.begin(), v.begin() + (half - 1), v.end());
    if (n % 2 == 1) return v[half - 1];
    const double lo = v[half - 1];
    const double hi = *std::min_element(v.begin() + half, v.end());
    const double two[2] = {lo, hi};
    return r_mean(two, 2);
}

// Summary statistics of the patch values x[0..n) (weights w: patch areas):
// am (area-weighted mean, as R's weighted.mean), md (median) and ra (range)
// as FRAGSTATS. NA propagates. (mn, sd and cv: Landscape::stat_values.)
double reduce(std::string_view stat, const double* x, i64 n, const double* w) {
    if (stat == "am"sv) {
        if (n == 0) return kNaN;
        return sum_of(0, n, [&](i64 i) { return x[i] * w[i]; }) / r_sum(w, n);
    }
    if (stat == "md"sv) return r_median(x, n);
    // ra
    if (n == 0) return kNaN;
    double lo = x[0], hi = x[0];
    for (i64 i = 0; i < n; ++i) {
        if (std::isnan(x[i])) return kNaN;
        lo = std::min(lo, x[i]);
        hi = std::max(hi, x[i]);
    }
    return hi - lo;
}

constexpr const char* kStats[] = {"mn", "sd", "cv", "am", "md", "ra"};

// "<base>_<stat>" split into base and stat (stat empty when the name has no
// summary-statistic suffix).
std::pair<std::string_view, std::string_view> split_stat(std::string_view name) {
    const std::size_t pos = name.rfind('_');
    if (pos == std::string_view::npos || pos == 0 || name.size() - pos != 3) return {};
    const std::string_view suffix = name.substr(pos + 1);
    for (const char* stat : kStats) {
        if (suffix == stat) return {name.substr(0, pos), suffix};
    }
    return {};
}

// The patch metrics with summary statistics (their place in the caches).
constexpr std::string_view kPatchBases[] = {"area", "cai",   "circle", "contig", "core", "dcore", "enn",  "frac",
                                            "gyrate", "ncore", "para", "perim",  "shape", "prox", "simi", "econ"};

int base_index(std::string_view name) {
    for (int i = 0; i < static_cast<int>(std::size(kPatchBases)); ++i) {
        if (name == kPatchBases[i]) return i;
    }
    return -1;
}

// One summary statistic of x[0..n) (mean and sd: the mn and sd already known;
// population: FRAGSTATS' standard deviation over n).
double summary(std::string_view stat, const double* x, i64 n, const double* w, double mean, double sd,
               bool population) {
    if (stat == "mn"sv) return r_mean(x, n);
    if (stat == "sd"sv) return r_sd(x, n, mean, population);
    if (stat == "cv"sv) return sd / mean * 100;
    return reduce(stat, x, n, w);
}

constexpr i64 kParallelPatches = i64(1) << 16;
constexpr i64 kCachedFrom = 4096;  // patches from which class/landscape values are cached

double entropy(const std::vector<double>& x, const std::string& base) {
    const double s = std::accumulate(x.begin(), x.end(), 0.0);
    double r = 0.0;
    for (double v : x) {
        const double p = v / s;
        if (p > 0.0) r -= p * r_log(p);
    }
    if (base == "log10"sv) return r / r_log(10.0);
    if (base == "log2"sv) return r / r_log(2.0);
    return r;
}

// Smallest possible perimeter (in cell sides) of a class of n cells.
double min_edge(i64 n) {
    const i64 s = static_cast<i64>(std::trunc(std::sqrt(static_cast<double>(n))));
    const i64 m = n - s * s;
    if (m == 0) return 4.0 * s;
    if (n <= s * (1 + s)) return 4.0 * s + 2;
    return 4.0 * s + 4;
}

// Largest number of like adjacencies of a class of n cells.
double max_adjacency(i64 n) {
    const i64 s = static_cast<i64>(std::trunc(std::sqrt(static_cast<double>(n))));
    const i64 m = n - s * s;
    if (m == 0) return 2.0 * s * (s - 1);
    if (m <= s) return 2.0 * s * (s - 1) + 2.0 * m - 1;
    return 2.0 * s * (s - 1) + 2.0 * m - 2;
}

}  // namespace

double reduce_statistic(const std::string& stat, const double* x, i64 n, const double* w) {
    return reduce(stat, x, n, w);
}

std::vector<double> r_statistics(const std::vector<double>& x) {
    const auto n = static_cast<i64>(x.size());
    const double mean = r_mean(x.data(), n);
    return {r_sum(x.data(), n), mean, r_sd(x.data(), n, mean), r_median(x.data(), n), r_log(x.empty() ? 1.0 : x[0]),
            r_sqrt(x.empty() ? 1.0 : x[0])};
}

// R's signif(x, digits) (nmath fprec) for ordinary magnitudes.
double r_signif(double x, int digits) {
    if (!std::isfinite(x) || x == 0) return x;
    double sgn = 1.0;
    if (x < 0) {
        sgn = -1.0;
        x = -x;
    }
    const double l10 = std::log10(x);
    const int e10 = static_cast<int>(digits - 1 - std::floor(l10));
    auto pow10i = [](int e) {
        double r = 1.0;
        for (int i = 0; i < e; ++i) r *= 10.0;
        return r;
    };
    if (e10 > 0) {
        const double p = pow10i(e10);
        return sgn * (std::nearbyint(x * p) / p);
    }
    const double p = pow10i(-e10);
    return sgn * (std::nearbyint(x / p) * p);
}

double Landscape::edge_length(i64 h, i64 v) const {
    // left/right sides are vertical: FRAGSTATS takes the cell height for
    // them, landscapemetrics the cell width
    if (options_.fragstats) return static_cast<double>(h) * resy_ + static_cast<double>(v) * resx_;
    if (resx_ == resy_) return static_cast<double>(h + v) * resx_;
    return static_cast<double>(h) * resx_ + static_cast<double>(v) * resy_;
}

// R's isTRUE(all.equal(resx, resy)): landscapemetrics' patch perimeter
// treats nearly equal cell sides as equal.
bool Landscape::res_all_equal() const {
    constexpr double kTolerance = 1.5e-8;
    const double diff = std::fabs(resx_ - resy_);
    const double scale = std::fabs(resx_);
    return (std::isfinite(scale) && scale > kTolerance ? diff / scale : diff) <= kTolerance;
}

double Landscape::total_area() {
    if (!total_area_) {
        const std::vector<double>& area = patch_base("area");
        total_area_ = r_sum(area.data(), static_cast<i64>(area.size()));
    }
    return *total_area_;
}

// First patch of every class (patches are sorted by class), and the total.
const std::vector<i64>& Landscape::class_start() {
    if (class_start_.empty()) {
        const Patches& pt = patches();
        const int k = n_classes();
        class_start_.assign(static_cast<std::size_t>(k) + 1, 0);
        for (i64 i = 0; i < pt.n; ++i) ++class_start_[pt.cls[i] + 1];
        for (int c = 0; c < k; ++c) class_start_[c + 1] += class_start_[c];
    }
    return class_start_;
}

Landscape::CachedValues& Landscape::patch_entry(std::string_view name) {
    const int b = base_index(name);
    if (b >= 0) return base_cache_[b];
    if (auto it = patch_cache_.find(name); it != patch_cache_.end()) return it->second;
    return patch_cache_.emplace(std::string(name), CachedValues{}).first->second;
}

const std::vector<double>& Landscape::patch_base(std::string_view name) {
    // computed in place (other entries may be added meanwhile: the node stays)
    CachedValues& entry = patch_entry(name);
    if (entry.valid) return entry.values;
    const Patches& pt = patches();
    const i64 n = pt.n;
    std::vector<double>& out = entry.values;
    out.assign(static_cast<std::size_t>(n), 0.0);
    const double factor_ha = resx_ * resy_ / 10000;
    const bool parallel = n >= kParallelPatches;
    if (name == "area"sv) {
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(pt.cells[i]) * factor_ha;
    } else if (name == "perim"sv) {
        const bool equal = res_all_equal() && !options_.fragstats;
        const auto [lh, lv] = side_lengths();
        for (i64 i = 0; i < n; ++i) {
            out[i] = equal ? static_cast<double>(pt.edge_h[i] + pt.edge_v[i]) * resx_
                           : static_cast<double>(pt.edge_h[i]) * lh + static_cast<double>(pt.edge_v[i]) * lv;
        }
    } else if (name == "para"sv || name == "shape"sv || name == "frac"sv) {
        const std::vector<double>& area = patch_base("area");
        const std::vector<double>& perim = patch_base("perim");
        const int kind = name == "para"sv ? 0 : name == "shape"sv ? 1 : 2;
        // FRAGSTATS' raster SHAPE: perimeter over the smallest perimeter of
        // as many cells, in cell sides (pylandstats keeps .25 p / sqrt(a)
        // for cells that are not square)
        const bool raster_shape = options_.fragstats && res_all_equal();
        parallel_for(parallel, n, [&](i64 i) {
            if (kind == 0) {
                out[i] = perim[i] / (area[i] * 10000);
            } else if (kind == 1 && raster_shape) {
                out[i] = static_cast<double>(pt.edge_h[i] + pt.edge_v[i]) / min_edge(pt.cells[i]);
            } else if (kind == 1) {
                out[i] = (0.25 * perim[i]) / r_sqrt(area[i] * 10000, n == 1);
            } else {
                const double f = 2 * r_log(0.25 * perim[i]) / r_log(area[i] * 10000);
                out[i] = std::isnan(f) ? 1.0 : f;
            }
        });
    } else if (name == "contig"sv) {
        for (i64 i = 0; i < n; ++i) {
            const double cells = static_cast<double>(pt.cells[i]);
            const double diagonal = 2.0 * static_cast<double>(pt.adj_diag[i]);
            const double straight = 2.0 * static_cast<double>(pt.adj_orth[i]) * 2;
            out[i] = (((diagonal + straight + cells) / cells) - 1) / 12;
        }
    } else if (name == "gyrate"sv || name == "enn"sv || name == "circle"sv) {
        if (tables_only_) throw std::logic_error(std::string(name) + " of an out-of-core landscape needs a detail pass");
        out = name == "gyrate"sv ? gyrate() : name == "enn"sv ? enn() : circle();
    } else if (name == "core"sv) {
        const Core& core = this->core();
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(core.cells[i]) * (resx_ * resy_) / 10000;
    } else if (name == "ncore"sv || name == "dcore"sv) {
        // landscapemetrics' dcore_* summarise the number of core areas
        const Core& core = this->core();
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(core.areas[i]);
    } else if (name == "econ"sv) {
        if (tables_only_) throw std::logic_error("econ is not available for an out-of-core landscape");
        out = econ();
    } else if (name == "prox"sv || name == "simi"sv) {
        if (tables_only_) throw std::logic_error(std::string(name) + " is not available for an out-of-core landscape");
        out = proximity(name == "simi"sv);
    } else if (name == "cai"sv) {
        const std::vector<double> area = patch_base("area");
        const std::vector<double> core = patch_base("core");
        for (i64 i = 0; i < n; ++i) out[i] = core[i] / area[i] * 100;
    } else {
        throw std::invalid_argument("unknown patch metric: " + std::string(name));
    }
    entry.valid = true;
    return out;
}

// Mean distance of the cell centres to the patch centroid; with cell_center
// the centroid is moved to the nearest cell(s) (ties at 6 significant digits
// are averaged, like landscapemetrics' merge of all tied cells).
std::vector<double> Landscape::gyrate() {
    const Patches& pt = patches();
    const i64 n = pt.n;
    std::vector<double> cx(n), cy(n);
    for (i64 i = 0; i < n; ++i) {
        const double c = static_cast<double>(pt.cells[i]);
        cx[i] = static_cast<double>(pt.sum_x[i] + pt.cells[i]) / c * resx_;
        cy[i] = static_cast<double>(pt.sum_y[i] + pt.cells[i]) / c * resy_;
    }
    // Sums per strip piece (in parallel), then per patch.
    auto distance_sums = [&](const std::vector<double>& px, const std::vector<double>& py) {
        const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
        const i64 nx = nx_;
        const i64 ny = ny_;
        const double rx = resx_;
        const double ry = resy_;
        const bool parallel = nx * ny >= (i64(1) << 16) && n_strips > 1;
        std::vector<std::vector<double>> pieces(static_cast<std::size_t>(n_strips));
        parallel_for_dynamic(parallel, n_strips, [&](i64 s) {
            const i64 off = pt.strip_offset[s];
            std::vector<double>& acc = pieces[s];
            acc.assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - off), 0.0);
            const i64 y1 = std::min(ny, (s + 1) * pt.strip_rows);
            for (i64 y = s * pt.strip_rows; y < y1; ++y) {
                for (i64 x = 0; x < nx; ++x) {
                    const std::int32_t c = pt.code[y * nx + x];
                    if (c == -1) continue;
                    const i64 k = -static_cast<i64>(c) - 2;
                    const std::int32_t pid = pt.local_to_patch[off + k];
                    const double dx = (x + 1) * rx - px[pid];
                    const double dy = (y + 1) * ry - py[pid];
                    acc[k] += std::sqrt(dx * dx + dy * dy);
                }
            }
        });
        std::vector<double> sum(n, 0.0);
        for (i64 s = 0; s < n_strips; ++s) {
            for (std::size_t k = 0; k < pieces[s].size(); ++k) {
                sum[pt.local_to_patch[pt.strip_offset[s] + static_cast<i64>(k)]] += pieces[s][k];
            }
        }
        return sum;
    };
    std::vector<double> out(n);
    if (!options_.cell_center) {
        const std::vector<double> sum = distance_sums(cx, cy);
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(sum[i] / pt.cells[i]);
        return out;
    }
    // Nearest cells to the centroid.
    std::vector<double> best(n, std::numeric_limits<double>::infinity());
    std::vector<std::vector<std::pair<double, double>>> ties(n);
    for (int pass = 0; pass < 2; ++pass) {
        for (i64 y = 0; y < ny_; ++y) {
            for (i64 x = 0; x < nx_; ++x) {
                const std::int32_t k = pt.patch(y * nx_ + x, y);
                if (k < 0) continue;
                const double X = (x + 1) * resx_;
                const double Y = (y + 1) * resy_;
                const double d = r_signif(std::sqrt((X - cx[k]) * (X - cx[k]) + (Y - cy[k]) * (Y - cy[k])), 6);
                if (pass == 0) {
                    best[k] = std::min(best[k], d);
                } else if (d == best[k]) {
                    ties[k].emplace_back(X, Y);
                }
            }
        }
    }
    std::vector<double> sum(n, 0.0);
    for (i64 y = 0; y < ny_; ++y) {
        for (i64 x = 0; x < nx_; ++x) {
            const std::int32_t k = pt.patch(y * nx_ + x, y);
            if (k < 0) continue;
            const double X = (x + 1) * resx_;
            const double Y = (y + 1) * resy_;
            for (const auto& c : ties[k]) {
                sum[k] += std::sqrt((X - c.first) * (X - c.first) + (Y - c.second) * (Y - c.second));
            }
        }
    }
    for (i64 i = 0; i < n; ++i) {
        out[i] = static_cast<double>(sum[i] / (static_cast<double>(pt.cells[i]) * ties[i].size()));
    }
    return out;
}

std::vector<double> Landscape::patch_metric(const std::string& name) { return patch_base(name); }

// Summary statistic `stat` of the patch metric `base`, per class or for the
// landscape (one value). Cached for large landscapes: mn is shared by sd and
// cv, sd by cv. The classes are computed in parallel when there are many
// patches (R's sums are sequential within a class).
const std::vector<double>& Landscape::stat_values(std::string_view base, std::string_view stat, bool by_class) {
    const int slot = stat_slot(base, stat, by_class);
    if (slot < 0) {
        patch_base(base);  // unknown metric: throws
        throw std::invalid_argument("unknown summary statistic: " + std::string(base) + "_" + std::string(stat));
    }
    if (stat_done_[slot]) return stat_cache_[slot];
    if (base == "dcore"sv && options_.fragstats) {
        stat_cache_[slot] = dcore_sizes(stat, by_class);
        stat_done_[slot] = true;
        return stat_cache_[slot];
    }
    const std::vector<double>& x = patch_base(base);
    const std::vector<double>& w = patch_base("area");
    // other slots: references into stat_cache_ stay valid
    static const std::vector<double> kNone;
    const std::vector<double>& mean = stat == "sd"sv || stat == "cv"sv ? stat_values(base, "mn", by_class) : kNone;
    const std::vector<double>& sd = stat == "cv"sv ? stat_values(base, "sd", by_class) : kNone;
    const i64 whole[2] = {0, patches().n};
    const i64* start = by_class ? class_start().data() : whole;
    const i64 groups = by_class ? n_classes() : 1;
    std::vector<double>& out = stat_cache_[slot];  // in place: keeps its memory across reset()
    out.resize(static_cast<std::size_t>(groups));
    parallel_for_dynamic(groups > 1 && start[groups] >= kParallelPatches, groups, [&](i64 g) {
        out[g] = summary(stat, x.data() + start[g], start[g + 1] - start[g], w.data() + start[g],
                         mean.empty() ? kNaN : mean[g], sd.empty() ? kNaN : sd[g], options_.fragstats);
    });
    stat_done_[slot] = true;
    return out;
}

// Slot of a summary statistic in stat_cache_, -1 for other patch metrics.
int Landscape::stat_slot(std::string_view base, std::string_view stat, bool by_class) {
    static_assert(std::size(kPatchBases) == kStatBases);
    const int b = base_index(base);
    int st = -1;
    for (int i = 0; i < kStatKinds && st < 0; ++i) {
        if (stat == std::string_view(kStats[i], 2)) st = i;
    }
    if (b < 0 || st < 0) return -1;
    return (b * kStatKinds + st) * 2 + (by_class ? 1 : 0);
}

// Every summary statistic the metrics `specs` need, computed at once: R's
// sums are sequential, so the independent ones (families, levels, classes)
// run side by side, the means before the standard deviations.
void Landscape::prepare(const std::vector<MetricSpec>& specs) {
    struct Task {
        std::string base, stat;
        bool by_class;
    };
    if (patches().n < kCachedFrom) return;
    std::vector<Task> wanted;
    for (const MetricSpec& spec : specs) {
        if (spec.level == kPatch || n_classes() == 0) continue;
        if (options_.fragstats && spec.name.compare(0, 6, "dcore_") == 0) continue;  // per core area
        for (const char* stat : kStats) {
            const std::string suffix = std::string("_") + stat;
            const std::string& name = spec.name;
            if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                wanted.push_back({name.substr(0, name.size() - suffix.size()), stat, spec.level == kClass});
            }
        }
    }
    if (wanted.empty()) return;
    for (const Task& t : wanted) patch_base(t.base);  // throws for unknown names
    patch_base("area");
    const std::vector<i64>& by_class = class_start();
    const std::vector<i64> whole{0, patches().n};
    auto key = [this](const std::string& base, const std::string& stat, bool cls) {
        return stat_slot(base, stat, cls);
    };
    // rounds: 0 = mn, am, md, ra; 1 = sd (needs mn); 2 = cv
    for (int round = 0; round < 3; ++round) {
        std::vector<Task> todo;
        for (const Task& t : wanted) {
            const bool needs_mean = t.stat == "sd"sv || t.stat == "cv"sv;
            const bool needs_sd = t.stat == "cv"sv;
            const int r = t.stat == "cv"sv ? 2 : t.stat == "sd"sv ? 1 : 0;
            if (r == round) todo.push_back(t);
            if (round == 0 && needs_mean) todo.push_back({t.base, "mn", t.by_class});
            if (round == 1 && needs_sd) todo.push_back({t.base, "sd", t.by_class});
        }
        // unique and not cached yet
        std::vector<Task> tasks;
        for (const Task& t : todo) {
            const int k = key(t.base, t.stat, t.by_class);
            if (k < 0 || stat_done_[k]) continue;
            bool dup = false;
            for (const Task& u : tasks) dup = dup || key(u.base, u.stat, u.by_class) == k;
            if (!dup) tasks.push_back(t);
        }
        if (tasks.empty()) continue;
        // one unit per task and group (class), larger groups first
        struct Unit {
            std::size_t task;
            i64 group, size;
        };
        std::vector<Unit> units;
        std::vector<std::vector<double>> results(tasks.size());
        std::vector<const std::vector<double>*> xs(tasks.size()), means(tasks.size(), nullptr), sds(tasks.size(), nullptr);
        for (std::size_t t = 0; t < tasks.size(); ++t) {
            const std::vector<i64>& start = tasks[t].by_class ? by_class : whole;
            const auto groups = static_cast<i64>(start.size()) - 1;
            results[t].assign(static_cast<std::size_t>(groups), kNaN);
            xs[t] = &patch_base(tasks[t].base);
            if (tasks[t].stat == "sd"sv || tasks[t].stat == "cv"sv) {
                means[t] = &stat_cache_[key(tasks[t].base, "mn", tasks[t].by_class)];
            }
            if (tasks[t].stat == "cv"sv) sds[t] = &stat_cache_[key(tasks[t].base, "sd", tasks[t].by_class)];
            for (i64 g = 0; g < groups; ++g) units.push_back({t, g, start[g + 1] - start[g]});
        }
        std::sort(units.begin(), units.end(), [](const Unit& a, const Unit& b) { return a.size > b.size; });
        const std::vector<double>& w = patch_base("area");
        const auto n_units = static_cast<i64>(units.size());
        parallel_for_dynamic(n_units > 1 && whole.back() >= kParallelPatches, n_units, [&](i64 u) {
            const Unit& unit = units[u];
            const Task& t = tasks[unit.task];
            const i64 s = (t.by_class ? by_class : whole)[unit.group];
            results[unit.task][unit.group] =
                summary(t.stat, xs[unit.task]->data() + s, unit.size, w.data() + s,
                        means[unit.task] ? (*means[unit.task])[unit.group] : kNaN,
                        sds[unit.task] ? (*sds[unit.task])[unit.group] : kNaN, options_.fragstats);
        });
        for (std::size_t t = 0; t < tasks.size(); ++t) {
            const int k = key(tasks[t].base, tasks[t].stat, tasks[t].by_class);
            stat_cache_[k] = std::move(results[t]);
            stat_done_[k] = true;
        }
    }
}

// landscapemetrics' cohesion: perimeter * sqrt(cells), the number of cells
// being recovered from the area in hectares.
std::vector<double> Landscape::cohesion_terms() {
    const std::vector<double>& area = patch_base("area");
    const std::vector<double>& perim = patch_base("perim");
    const auto n = static_cast<i64>(area.size());
    std::vector<double> out(area.size());
    const double cell = resx_ * resy_;
    parallel_for(n >= kParallelPatches, n, [&](i64 i) { out[i] = perim[i] * r_sqrt(area[i] * 10000 / cell, n == 1); });
    return out;
}

// (1 - 1 / sqrt(cells of the landscape)) ^ -1
double Landscape::cohesion_factor() const {
    return r_reciprocal(1 - (1 / r_sqrt(static_cast<double>(valid_cells_), true)));
}

std::vector<double> Landscape::class_te() {
    const Adjacency& adj = adjacency();
    const int k = n_classes();
    std::vector<double> te(k);
    for (int c = 0; c < k; ++c) {
        i64 h = 0, v = 0;
        const int last = options_.count_boundary ? k + 2 : k;
        for (int b = 0; b < last; ++b) {
            if (b == c) continue;
            h += adj.h[c * adj.dim + b];
            v += adj.v[c * adj.dim + b];
        }
        te[c] = edge_length(h, v);
    }
    return te;
}

double Landscape::landscape_te() {
    const Adjacency& adj = adjacency();
    const int k = n_classes();
    i64 h = 0, v = 0;
    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < a; ++b) {
            h += adj.h[a * adj.dim + b];
            v += adj.v[a * adj.dim + b];
        }
        if (options_.count_boundary) {
            h += adj.h[a * adj.dim + k] + adj.h[a * adj.dim + k + 1];
            v += adj.v[a * adj.dim + k] + adj.v[a * adj.dim + k + 1];
        }
    }
    return edge_length(h, v);
}

// PAFRAC of the groups of patches [start[g], start[g + 1]): 2 / the slope
// of R's lm(log(area) ~ log(perim)), area in m2 (landscapemetrics' class
// level) or in hectares (its landscape level); NaN below 10 patches.
std::vector<double> Landscape::pafrac_by(const std::vector<i64>& start, bool hectares) {
    static const std::string kLogPerim = "~log perim";
    static const std::string kLogAreaM2 = "~log area m2";
    static const std::string kLogAreaHa = "~log area ha";
    const std::string& key_area = hectares ? kLogAreaHa : kLogAreaM2;
    const std::vector<double>& perim = patch_base("perim");
    const std::vector<double>& area = patch_base("area");
    CachedValues& log_perim = patch_cache_[kLogPerim];
    if (!log_perim.valid) {
        const auto n = static_cast<i64>(perim.size());
        log_perim.values.resize(perim.size());
        parallel_for(n >= kParallelPatches, n, [&](i64 i) { log_perim.values[i] = r_log(perim[i]); });
        log_perim.valid = true;
    }
    CachedValues& log_area = patch_cache_[key_area];
    if (!log_area.valid) {
        const auto n = static_cast<i64>(area.size());
        log_area.values.resize(area.size());
        parallel_for(n >= kParallelPatches, n,
                     [&](i64 i) { log_area.values[i] = r_log(hectares ? area[i] : area[i] * 10000); });
        log_area.valid = true;
    }
    const std::vector<double>& lp = log_perim.values;
    const std::vector<double>& la = log_area.values;
    const auto groups = static_cast<i64>(start.size()) - 1;
    std::vector<double> out(static_cast<std::size_t>(groups), kNaN);
    parallel_for_dynamic(groups > 1 && start.back() >= kParallelPatches, groups, [&](i64 g) {
        const i64 n = start[g + 1] - start[g];
        if (n >= 10) out[g] = 2 / r_lm_slope(lp.data() + start[g], la.data() + start[g], static_cast<int>(n));
    });
    return out;
}

// Class and landscape values are cached, except for small landscapes
// (moving windows, sample plots) where computing them again is cheaper.
std::vector<double> Landscape::class_metric(const std::string& name) {
    if (n_classes() == 0 || patches().n < kCachedFrom) return class_metric_impl(name);
    if (auto it = class_cache_.find(name); it != class_cache_.end()) return it->second;
    std::vector<double> out = class_metric_impl(name);
    class_cache_.emplace(name, out);
    return out;
}

double Landscape::landscape_metric(const std::string& name) {
    if (n_classes() == 0 || patches().n < kCachedFrom) return landscape_metric_impl(name);
    if (auto it = landscape_cache_.find(name); it != landscape_cache_.end()) return it->second;
    const double out = landscape_metric_impl(name);
    landscape_cache_.emplace(name, out);
    return out;
}

std::vector<double> Landscape::class_metric_impl(const std::string& name) {
    const int k = n_classes();
    if (k == 0) return {};
    const std::vector<i64>& start = class_start();
    auto per_class = [&](auto&& f) {
        std::vector<double> out(k);
        for (int c = 0; c < k; ++c) out[c] = f(c, start[c], start[c + 1] - start[c]);
        return out;
    };

    if (const auto [base, stat] = split_stat(name); !stat.empty()) return stat_values(base, stat, true);

    const std::vector<double>& area = patch_base("area");
    const double ta = total_area();
    auto ca = [&](int c) { return r_sum(area.data() + start[c], start[c + 1] - start[c]); };

    if (name == "ca"sv) return per_class([&](int c, i64, i64) { return ca(c); });
    if (name == "pland"sv) {
        std::vector<double> sums = per_class([&](int c, i64, i64) { return ca(c); });
        const double total = r_sum(sums.data(), k);
        for (double& v : sums) v = v / total * 100;
        return sums;
    }
    if (name == "np"sv) return per_class([](int, i64, i64 n) { return static_cast<double>(n); });
    if (name == "pd"sv) return per_class([&](int, i64, i64 n) { return (static_cast<double>(n) / ta) * 100; });
    if (name == "lpi"sv) {
        return per_class([&](int, i64 s, i64 n) {
            double m = -std::numeric_limits<double>::infinity();
            for (i64 i = s; i < s + n; ++i) m = std::max(m, area[i] / ta * 100);
            return m;
        });
    }
    if (name == "te"sv) return class_te();
    if (name == "connect"sv) {
        if (tables_only_) throw std::logic_error("connect is not available for an out-of-core landscape");
        return connectance(true);
    }
    if (name == "cwed"sv || name == "teci"sv) return contrast_edges(name == "teci"sv, true);
    if (name == "ed"sv) {
        std::vector<double> te = class_te();
        for (double& v : te) v = v / ta;
        return te;
    }
    if (name == "division"sv) {
        return per_class([&](int, i64 s, i64 n) {
            return 1 - sum_of(s, s + n, [&](i64 i) {
                const double r = area[i] / ta;
                return r * r;
            });
        });
    }
    if (name == "mesh"sv) {
        const double total = ta * 10000;
        return per_class([&](int, i64 s, i64 n) {
            const double sum = sum_of(s, s + n, [&](i64 i) {
                const double a = area[i] * 10000;
                return a * a;
            });
            return (sum / total) * (1.0 / 10000);
        });
    }
    if (name == "split"sv) {
        return per_class([&](int, i64 s, i64 n) {
            return (ta * ta) / sum_of(s, s + n, [&](i64 i) { return area[i] * area[i]; });
        });
    }
    if (name == "cohesion"sv) {
        const std::vector<double>& perim = patch_base("perim");
        const std::vector<double> den_i = cohesion_terms();
        const double factor = cohesion_factor();
        return per_class([&](int, i64 s, i64 n) {
            const double num = r_sum(perim.data() + s, n);
            const double den = r_sum(den_i.data() + s, n);
            return (1 - (num / den)) * factor * 100;
        });
    }
    if (name == "pafrac"sv) return pafrac_by(start, false);
    if (name == "tca"sv || name == "cpland"sv) {
        const std::vector<double>& core = patch_base("core");
        return per_class([&](int, i64 s, i64 n) {
            const double tca = r_sum(core.data() + s, n);
            return name == "tca"sv ? tca : tca / ta * 100;
        });
    }
    if (name == "ndca"sv || name == "dcad"sv) {
        const std::vector<double>& ncore = patch_base("ncore");
        return per_class([&](int, i64 s, i64 n) {
            const double ndca = r_sum(ncore.data() + s, n);
            return name == "ndca"sv ? ndca : ndca / ta * 100;
        });
    }

    const Adjacency& adj = adjacency();
    const int dim = adj.dim;
    auto row_sum = [&](int c, bool with_na, bool with_out, bool with_self) {
        i64 s = 0;
        for (int b = 0; b < k; ++b) {
            if (b != c || with_self) s += adj.orth(c, b);
        }
        if (with_na) s += adj.orth(c, k);
        if (with_out) s += adj.orth(c, k + 1);
        return s;
    };
    if (name == "ai"sv) {
        return per_class([&](int c, i64, i64) {
            const double like = static_cast<double>(adj.orth(c, c)) / 2;
            const double ai = (like / max_adjacency(class_cells_[c])) * 100;
            return std::isnan(ai) ? kNaN : ai;
        });
    }
    if (name == "pladj"sv) {
        return per_class([&](int c, i64, i64) {
            return static_cast<double>(adj.orth(c, c)) / static_cast<double>(row_sum(c, true, true, true)) * 100;
        });
    }
    if (name == "clumpy"sv) {
        return per_class([&](int c, i64, i64) {
            const double p = static_cast<double>(class_cells_[c]) / static_cast<double>(valid_cells_);
            const double g = static_cast<double>(adj.orth(c, c)) /
                             (static_cast<double>(row_sum(c, true, true, true)) - min_edge(class_cells_[c]));
            if (std::isnan(g) || p == 1) return kNaN;
            if (g >= p) return (g - p) / (1 - p);
            if (p >= 0.5) return (g - p) / (1 - p);
            return (g - p) / -p;
        });
    }
    if (name == "lsi"sv && options_.fragstats && !res_all_equal()) {
        // cells that are not square: .25 E / sqrt(A) with the physical edge
        // lengths (pylandstats; FRAGSTATS needs square cells)
        return per_class([&](int c, i64, i64) {
            i64 h = 0, v = 0;
            for (int b = 0; b < k + 2; ++b) {
                if (b == c) continue;
                h += adj.h[c * dim + b];
                v += adj.v[c * dim + b];
            }
            return 0.25 * edge_length(h, v) / std::sqrt(static_cast<double>(class_cells_[c]) * resx_ * resy_);
        });
    }
    if (name == "lsi"sv || name == "nlsi"sv) {
        const double total = static_cast<double>(valid_cells_);
        const double boundary = static_cast<double>((ny_ * 2) + (nx_ * 2));
        return per_class([&](int c, i64, i64) {
            const double perim = static_cast<double>(row_sum(c, true, true, false));
            const double cells = static_cast<double>(class_cells_[c]);
            const double lo = min_edge(class_cells_[c]);
            if (name == "lsi"sv) return perim / lo;
            const double pi = cells / total;
            const double limit = (0.5 * total + 0.5 * boundary) / total;
            const bool even = std::fmod(total, 2.0) == 0;
            double hi;
            if (pi <= 0.5) {
                hi = 4 * cells;
            } else if (even && pi <= limit) {
                hi = 3 * total - 2 * cells;
            } else if (!even && pi <= limit) {
                hi = 3 * total - 2 * cells + 3;
            } else {
                hi = perim + 4 * (total - cells);
            }
            const double nlsi = (perim - lo) / (hi - lo);
            return std::isfinite(nlsi) ? nlsi : kNaN;
        });
    }
    if (name == "iji"sv) {
        if (k < 3) return std::vector<double>(k, kNaN);
        return per_class([&](int c, i64, i64) {
            const double rs = static_cast<double>(row_sum(c, false, false, false));
            const double s = sum_of(0, k, [&](i64 b) {
                if (b == c) return kNaN;  // diagonal set to 0: 0 * log(0)
                const double r = static_cast<double>(adj.orth(c, static_cast<int>(b))) / rs;
                return r * r_log(r);
            }, true);
            return (-s / r_log(static_cast<double>(k - 1))) * 100;
        });
    }
    (void)dim;
    throw std::invalid_argument("unknown class metric: " + name);
}

double Landscape::landscape_metric_impl(const std::string& name) {
    const int k = n_classes();
    if (k == 0) return kNaN;
    const Patches& pt = patches();

    if (const auto [base, stat] = split_stat(name); !stat.empty()) return stat_values(base, stat, false)[0];

    const std::vector<double>& area = patch_base("area");
    const double ta = total_area();
    const i64 np = pt.n;
    if (name == "ta"sv) return ta;
    if (name == "np"sv) return static_cast<double>(np);
    if (name == "pd"sv) return static_cast<double>(np) / ta * 100;
    if (name == "lpi"sv) {
        double m = -std::numeric_limits<double>::infinity();
        for (double a : area) m = std::max(m, a / ta * 100);
        return m;
    }
    if (name == "te"sv) return landscape_te();
    if (name == "connect"sv) {
        if (tables_only_) throw std::logic_error("connect is not available for an out-of-core landscape");
        return connectance(false)[0];
    }
    if (name == "cwed"sv || name == "teci"sv) return contrast_edges(name == "teci"sv, false)[0];
    if (name == "ed"sv) return landscape_te() / ta;
    if (name == "division"sv) {
        return 1 - sum_of(0, np, [&](i64 i) {
            const double r = area[i] / ta;
            return r * r;
        });
    }
    if (name == "mesh"sv || name == "split"sv) {
        const double s = sum_of(0, np, [&](i64 i) { return area[i] * area[i]; });
        return name == "mesh"sv ? s / ta : (ta * ta) / s;
    }
    if (name == "cohesion"sv) {
        const std::vector<double>& perim = patch_base("perim");
        const std::vector<double> den_i = cohesion_terms();
        const double den = r_sum(den_i.data(), np);
        return (1 - (r_sum(perim.data(), np) / den)) * cohesion_factor() * 100;
    }
    if (name == "tca"sv) {
        const std::vector<double>& core = patch_base("core");
        return r_sum(core.data(), np);
    }
    if (name == "ndca"sv || name == "dcad"sv) {
        const std::vector<double>& ncore = patch_base("ncore");
        const double ndca = r_sum(ncore.data(), np);
        return name == "ndca"sv ? ndca : ndca / ta * 100;
    }
    if (name == "pafrac"sv) {
        if (np < 10) return kNaN;
        return pafrac_by({0, np}, true)[0];
    }

    // Composition.
    std::vector<double> class_area = class_metric("ca");
    const double ca_total = r_sum(class_area.data(), k);
    auto proportions = [&]() {
        std::vector<double> p(k);
        for (int c = 0; c < k; ++c) p[c] = (class_area[c] / ca_total * 100) / 100;
        return p;
    };
    if (name == "pr"sv) return static_cast<double>(k);
    if (name == "prd"sv) return static_cast<double>(k) / ta * 100;
    if (name == "rpr"sv) return std::isnan(options_.classes_max) ? kNaN : static_cast<double>(k) / options_.classes_max * 100;
    if (name == "shdi"sv || name == "shei"sv) {
        const std::vector<double> p = proportions();
        if (name == "shdi"sv) return sum_of(0, k, [&](i64 c) { return -p[c] * r_log(p[c]); });
        if (k == 1) return 0.0;
        return -sum_of(0, k, [&](i64 c) { return p[c] * r_log(p[c]); }) / r_log(static_cast<double>(k));
    }
    if (name == "sidi"sv || name == "siei"sv) {
        const std::vector<double> p = proportions();
        const double sidi = 1 - sum_of(0, k, [&](i64 c) { return p[c] * p[c]; });
        return name == "sidi"sv ? sidi : sidi / (1 - (1 / static_cast<double>(k)));
    }
    if (name == "msidi"sv || name == "msiei"sv) {
        const double msidi = -r_log(sum_of(0, k, [&](i64 c) {
            const double r = class_area[c] / ca_total;
            return r * r;
        }));
        return name == "msidi"sv ? msidi : msidi / r_log(static_cast<double>(k));
    }

    // Configuration.
    const Adjacency& adj = adjacency();
    if (name == "ai"sv) {
        const std::vector<double> ai = class_metric("ai");
        const std::vector<double> pland = class_metric("pland");
        return sum_of(0, k, [&](i64 c) { return ai[c] * (pland[c] / 100); }, true);
    }
    if (name == "pladj"sv) {
        i64 like = 0, total = 0;
        for (int a = 0; a < k; ++a) {
            like += adj.orth(a, a);
            for (int b = 0; b < k; ++b) total += adj.orth(a, b);
            total += adj.orth(a, k + 1);
        }
        return static_cast<double>(like) / static_cast<double>(total) * 100;
    }
    if (name == "lsi"sv) {
        i64 h = 0, v = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < a; ++b) {
                h += adj.h[a * adj.dim + b];
                v += adj.v[a * adj.dim + b];
            }
            h += adj.h[a * adj.dim + k] + adj.h[a * adj.dim + k + 1];
            v += adj.v[a * adj.dim + k] + adj.v[a * adj.dim + k + 1];
        }
        if (options_.fragstats && !res_all_equal()) {
            return 0.25 * edge_length(h, v) / std::sqrt(static_cast<double>(valid_cells_) * resx_ * resy_);
        }
        return static_cast<double>(h + v) / min_edge(valid_cells_);
    }
    if (name == "iji"sv) {
        if (k < 3) return kNaN;
        i64 e_total = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < a; ++b) e_total += adj.orth(a, b);
        }
        // lower triangle in R's (column-major) order
        Extended s = 0.0;
        for (int b = 0; b < k; ++b) {
            for (int a = b + 1; a < k; ++a) {
                const double r = static_cast<double>(adj.orth(a, b)) / static_cast<double>(e_total);
                const double term = r * r_log(r);
                if (!std::isnan(term)) s += term;
            }
        }
        return (-static_cast<double>(s) / r_log(0.5 * (k * (k - 1.0)))) * 100;
    }
    if (name == "contag"sv) {
        if (k < 2) return kNaN;
        i64 total = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) total += adj.orth(a, b);
        }
        Extended esum = 0.0;
        for (int b = 0; b < k; ++b) {
            for (int a = 0; a < k; ++a) {
                const double r = static_cast<double>(adj.orth(a, b)) / static_cast<double>(total);
                const double term = r * r_log(r);
                if (!std::isnan(term)) esum += term;
            }
        }
        const double emax = 2 * r_log(static_cast<double>(k));
        return (1 + static_cast<double>(esum) / emax) * 100;
    }
    if (name == "ent"sv || name == "condent"sv || name == "joinent"sv || name == "mutinf"sv || name == "relmutinf"sv) {
        const bool eight = options_.neighbourhood == 8;
        auto cooc = [&](int a, int b) {
            return static_cast<double>(adj.orth(a, b) + (eight ? adj.diag[a * adj.dim + b] : 0));
        };
        std::vector<double> col(k, 0.0);
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) col[b] += cooc(a, b);
        }
        const double comp = entropy(col, options_.base);
        if (name == "ent"sv) return comp;
        std::vector<double> pairs;
        if (options_.ordered) {
            for (int b = 0; b < k; ++b) {
                for (int a = 0; a < k; ++a) pairs.push_back(cooc(a, b));
            }
        } else {
            for (int a = 0; a < k; ++a) {
                for (int b = 0; b <= a; ++b) pairs.push_back((a == b ? cooc(a, a) : cooc(a, b) + cooc(b, a)) / 2);
            }
        }
        const double cplx = entropy(pairs, options_.base);
        if (name == "joinent"sv) return cplx;
        const double conf = cplx - comp;
        if (name == "condent"sv) return conf;
        const double aggr = comp - conf;
        if (name == "mutinf"sv) return aggr;
        return aggr == 0 ? 1.0 : aggr / comp;
    }
    throw std::invalid_argument("unknown landscape metric: " + name);
}

}  // namespace lsm
