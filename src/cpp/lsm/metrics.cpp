// SPDX-License-Identifier: GPL-3.0-or-later
//
// Metric formulas. Each one follows the corresponding lsm_{p,c,l}_* function
// of landscapemetrics, including its conventions (sample standard deviation,
// NA handling, which neighbours count as edge).
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "lsm/landscape.hpp"

namespace lsm {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// R's mean(): sum, divide, then add the mean residual.
double r_mean(const double* x, i64 n) {
    if (n == 0) return kNaN;
    long double s = 0;
    for (i64 i = 0; i < n; ++i) s += x[i];
    s /= n;
    if (std::isfinite(static_cast<double>(s))) {
        long double t = 0;
        for (i64 i = 0; i < n; ++i) t += x[i] - s;
        s += t / n;
    }
    return static_cast<double>(s);
}

// R's sd(): two-pass sample standard deviation (NA for fewer than 2 values).
double r_sd(const double* x, i64 n) {
    if (n < 2) return kNaN;
    const long double m = r_mean(x, n);
    long double ss = 0;
    for (i64 i = 0; i < n; ++i) {
        const long double d = x[i] - m;
        ss += d * d;
    }
    return std::sqrt(static_cast<double>(ss / (n - 1)));
}

double r_sum(const double* x, i64 n) {
    long double s = 0;
    for (i64 i = 0; i < n; ++i) s += x[i];
    return static_cast<double>(s);
}

double reduce(const std::string& stat, const double* x, i64 n) {
    if (stat == "mn") return r_mean(x, n);
    if (stat == "sd") return r_sd(x, n);
    return r_sd(x, n) / r_mean(x, n) * 100;  // cv
}

double entropy(const std::vector<double>& x, const std::string& base) {
    const double s = std::accumulate(x.begin(), x.end(), 0.0);
    double r = 0.0;
    for (double v : x) {
        const double p = v / s;
        if (p > 0.0) r -= p * std::log(p);
    }
    if (base == "log10") return r / std::log(10.0);
    if (base == "log2") return r / std::log(2.0);
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

// Slope of lm(y ~ x), NaN when x is collinear with the intercept (as R's
// QR with tol = 1e-7 decides).
double slope(const std::vector<double>& x, const std::vector<double>& y) {
    const i64 n = static_cast<i64>(x.size());
    const double xm = r_mean(x.data(), n);
    const double ym = r_mean(y.data(), n);
    long double sxx = 0, sxy = 0, sx2 = 0;
    for (i64 i = 0; i < n; ++i) {
        sxx += (x[i] - xm) * (x[i] - xm);
        sxy += (x[i] - xm) * (y[i] - ym);
        sx2 += static_cast<long double>(x[i]) * x[i];
    }
    if (sxx <= 0 || std::sqrt(static_cast<double>(sxx / sx2)) < 1e-7) return kNaN;
    return static_cast<double>(sxy / sxx);
}

}  // namespace

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
    if (resx_ == resy_) return static_cast<double>(h + v) * resx_;
    return static_cast<double>(h) * resx_ + static_cast<double>(v) * resy_;
}

double Landscape::total_area() {
    const std::vector<double>& area = patch_base("area");
    return r_sum(area.data(), static_cast<i64>(area.size()));
}

const std::vector<double>& Landscape::patch_base(const std::string& name) {
    if (auto it = patch_cache_.find(name); it != patch_cache_.end()) return it->second;
    const Patches& pt = patches();
    const i64 n = pt.n;
    std::vector<double> out(static_cast<std::size_t>(n));
    const double factor_ha = resx_ * resy_ / 10000;
    if (name == "area") {
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(pt.cells[i]) * factor_ha;
    } else if (name == "perim") {
        for (i64 i = 0; i < n; ++i) out[i] = edge_length(pt.edge_h[i], pt.edge_v[i]);
    } else if (name == "para" || name == "shape" || name == "frac") {
        const std::vector<double> area = patch_base("area");
        const std::vector<double> perim = patch_base("perim");
        for (i64 i = 0; i < n; ++i) {
            if (name == "para") {
                out[i] = perim[i] / (area[i] * 10000);
            } else if (name == "shape") {
                out[i] = (0.25 * perim[i]) / std::sqrt(area[i] * 10000);
            } else {
                const double f = 2 * std::log(0.25 * perim[i]) / std::log(area[i] * 10000);
                out[i] = std::isnan(f) ? 1.0 : f;
            }
        }
    } else if (name == "contig") {
        for (i64 i = 0; i < n; ++i) {
            const double cells = static_cast<double>(pt.cells[i]);
            const double diagonal = 2.0 * static_cast<double>(pt.adj_diag[i]);
            const double straight = 2.0 * static_cast<double>(pt.adj_orth[i]) * 2;
            out[i] = (((diagonal + straight + cells) / cells) - 1) / 12;
        }
    } else if (name == "gyrate" || name == "enn" || name == "circle") {
        if (tables_only_) throw std::logic_error(name + " of an out-of-core landscape needs a detail pass");
        out = name == "gyrate" ? gyrate() : name == "enn" ? enn() : circle();
    } else if (name == "core") {
        const Core& core = this->core();
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(core.cells[i]) * (resx_ * resy_) / 10000;
    } else if (name == "ncore" || name == "dcore") {
        // landscapemetrics' dcore_* summarise the number of core areas
        const Core& core = this->core();
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(core.areas[i]);
    } else if (name == "cai") {
        const std::vector<double> area = patch_base("area");
        const std::vector<double> core = patch_base("core");
        for (i64 i = 0; i < n; ++i) out[i] = core[i] / area[i] * 100;
    } else {
        throw std::invalid_argument("unknown patch metric: " + name);
    }
    return patch_cache_.emplace(name, std::move(out)).first->second;
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
        std::vector<std::vector<long double>> pieces(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (parallel)
        for (i64 s = 0; s < n_strips; ++s) {
            const i64 off = pt.strip_offset[s];
            std::vector<long double>& acc = pieces[s];
            acc.assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - off), 0.0L);
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
        }
        std::vector<long double> sum(n, 0.0L);
        for (i64 s = 0; s < n_strips; ++s) {
            for (std::size_t k = 0; k < pieces[s].size(); ++k) {
                sum[pt.local_to_patch[pt.strip_offset[s] + static_cast<i64>(k)]] += pieces[s][k];
            }
        }
        return sum;
    };
    std::vector<double> out(n);
    if (!options_.cell_center) {
        const std::vector<long double> sum = distance_sums(cx, cy);
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
    std::vector<long double> sum(n, 0.0L);
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
        out[i] = static_cast<double>(sum[i] / (static_cast<long double>(pt.cells[i]) * ties[i].size()));
    }
    return out;
}

std::vector<double> Landscape::patch_metric(const std::string& name) { return patch_base(name); }

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

// PAFRAC per group of patches (groups: patch -> group index).
std::vector<double> Landscape::pafrac_by(const std::vector<std::int32_t>& groups, int n_groups) {
    const std::vector<double>& area = patch_base("area");
    const std::vector<double>& perim = patch_base("perim");
    std::vector<std::vector<double>> xs(n_groups), ys(n_groups);
    for (std::size_t i = 0; i < area.size(); ++i) {
        xs[groups[i]].push_back(std::log(perim[i]));
        ys[groups[i]].push_back(std::log(area[i] * 10000));
    }
    std::vector<double> out(n_groups, kNaN);
    for (int g = 0; g < n_groups; ++g) {
        if (xs[g].size() < 10) continue;
        out[g] = 2 / slope(xs[g], ys[g]);
    }
    return out;
}

std::vector<double> Landscape::class_metric(const std::string& name) {
    const int k = n_classes();
    if (k == 0) return {};
    const Patches& pt = patches();
    std::vector<i64> start(k + 1, 0);
    for (i64 i = 0; i < pt.n; ++i) ++start[pt.cls[i] + 1];
    for (int c = 0; c < k; ++c) start[c + 1] += start[c];
    auto per_class = [&](auto&& f) {
        std::vector<double> out(k);
        for (int c = 0; c < k; ++c) out[c] = f(c, start[c], start[c + 1] - start[c]);
        return out;
    };

    for (const char* stat : {"mn", "sd", "cv"}) {
        const std::string suffix = std::string("_") + stat;
        if (ends_with(name, suffix)) {
            const std::vector<double>& base = patch_base(name.substr(0, name.size() - suffix.size()));
            return per_class([&](int, i64 s, i64 n) { return reduce(stat, base.data() + s, n); });
        }
    }

    const std::vector<double>& area = patch_base("area");
    const double ta = total_area();
    auto ca = [&](int c) { return r_sum(area.data() + start[c], start[c + 1] - start[c]); };

    if (name == "ca") return per_class([&](int c, i64, i64) { return ca(c); });
    if (name == "pland") {
        std::vector<double> sums = per_class([&](int c, i64, i64) { return ca(c); });
        const double total = r_sum(sums.data(), k);
        for (double& v : sums) v = v / total * 100;
        return sums;
    }
    if (name == "np") return per_class([](int, i64, i64 n) { return static_cast<double>(n); });
    if (name == "pd") return per_class([&](int, i64, i64 n) { return (static_cast<double>(n) / ta) * 100; });
    if (name == "lpi") {
        return per_class([&](int, i64 s, i64 n) {
            double m = -std::numeric_limits<double>::infinity();
            for (i64 i = s; i < s + n; ++i) m = std::max(m, area[i] / ta * 100);
            return m;
        });
    }
    if (name == "te") return class_te();
    if (name == "ed") {
        std::vector<double> te = class_te();
        for (double& v : te) v = v / ta;
        return te;
    }
    if (name == "division") {
        return per_class([&](int, i64 s, i64 n) {
            long double sum = 0;
            for (i64 i = s; i < s + n; ++i) sum += (area[i] / ta) * (area[i] / ta);
            return 1 - static_cast<double>(sum);
        });
    }
    if (name == "mesh") {
        return per_class([&](int, i64 s, i64 n) {
            long double sum = 0;
            for (i64 i = s; i < s + n; ++i) sum += (area[i] * 10000) * (area[i] * 10000);
            return (static_cast<double>(sum) / (ta * 10000)) * (1.0 / 10000);
        });
    }
    if (name == "split") {
        return per_class([&](int, i64 s, i64 n) {
            long double sum = 0;
            for (i64 i = s; i < s + n; ++i) sum += area[i] * area[i];
            return (ta * ta) / static_cast<double>(sum);
        });
    }
    if (name == "cohesion") {
        const std::vector<double>& perim = patch_base("perim");
        const double n_cells = static_cast<double>(valid_cells_);
        return per_class([&](int, i64 s, i64 n) {
            long double num = 0, den = 0;
            for (i64 i = s; i < s + n; ++i) {
                num += perim[i];
                den += perim[i] * std::sqrt(static_cast<double>(pt.cells[i]));
            }
            return (1 - (static_cast<double>(num) / static_cast<double>(den))) * (1 / (1 - (1 / std::sqrt(n_cells)))) * 100;
        });
    }
    if (name == "pafrac") return pafrac_by(pt.cls, k);
    if (name == "tca" || name == "cpland") {
        const std::vector<double>& core = patch_base("core");
        return per_class([&](int, i64 s, i64 n) {
            const double tca = r_sum(core.data() + s, n);
            return name == "tca" ? tca : tca / ta * 100;
        });
    }
    if (name == "ndca" || name == "dcad") {
        const std::vector<double>& ncore = patch_base("ncore");
        return per_class([&](int, i64 s, i64 n) {
            const double ndca = r_sum(ncore.data() + s, n);
            return name == "ndca" ? ndca : ndca / ta * 100;
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
    if (name == "ai") {
        return per_class([&](int c, i64, i64) {
            const double like = static_cast<double>(adj.orth(c, c)) / 2;
            const double ai = (like / max_adjacency(class_cells_[c])) * 100;
            return std::isnan(ai) ? kNaN : ai;
        });
    }
    if (name == "pladj") {
        return per_class([&](int c, i64, i64) {
            return static_cast<double>(adj.orth(c, c)) / static_cast<double>(row_sum(c, true, true, true)) * 100;
        });
    }
    if (name == "clumpy") {
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
    if (name == "lsi" || name == "nlsi") {
        const double total = static_cast<double>(valid_cells_);
        const double boundary = static_cast<double>((ny_ * 2) + (nx_ * 2));
        return per_class([&](int c, i64, i64) {
            const double perim = static_cast<double>(row_sum(c, true, true, false));
            const double cells = static_cast<double>(class_cells_[c]);
            const double lo = min_edge(class_cells_[c]);
            if (name == "lsi") return perim / lo;
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
    if (name == "iji") {
        if (k < 3) return std::vector<double>(k, kNaN);
        return per_class([&](int c, i64, i64) {
            const double rs = static_cast<double>(row_sum(c, false, false, false));
            double s = 0;
            for (int b = 0; b < k; ++b) {
                if (b == c) continue;
                const double r = static_cast<double>(adj.orth(c, b)) / rs;
                const double term = r * std::log(r);
                if (!std::isnan(term)) s += term;
            }
            return (-s / std::log(static_cast<double>(k - 1))) * 100;
        });
    }
    (void)dim;
    throw std::invalid_argument("unknown class metric: " + name);
}

double Landscape::landscape_metric(const std::string& name) {
    const int k = n_classes();
    if (k == 0) return kNaN;
    const Patches& pt = patches();

    for (const char* stat : {"mn", "sd", "cv"}) {
        const std::string suffix = std::string("_") + stat;
        if (ends_with(name, suffix)) {
            const std::vector<double>& base = patch_base(name.substr(0, name.size() - suffix.size()));
            return reduce(stat, base.data(), static_cast<i64>(base.size()));
        }
    }

    const std::vector<double>& area = patch_base("area");
    const double ta = total_area();
    const i64 np = pt.n;
    if (name == "ta") return ta;
    if (name == "np") return static_cast<double>(np);
    if (name == "pd") return static_cast<double>(np) / ta * 100;
    if (name == "lpi") {
        double m = -std::numeric_limits<double>::infinity();
        for (double a : area) m = std::max(m, a / ta * 100);
        return m;
    }
    if (name == "te") return landscape_te();
    if (name == "ed") return landscape_te() / ta;
    if (name == "division") {
        long double s = 0;
        for (double a : area) s += (a / ta) * (a / ta);
        return 1 - static_cast<double>(s);
    }
    if (name == "mesh" || name == "split") {
        long double s = 0;
        for (double a : area) s += a * a;
        return name == "mesh" ? static_cast<double>(s) / ta : (ta * ta) / static_cast<double>(s);
    }
    if (name == "cohesion") {
        const std::vector<double>& perim = patch_base("perim");
        long double num = 0, den = 0;
        for (i64 i = 0; i < np; ++i) {
            num += perim[i];
            den += perim[i] * std::sqrt(static_cast<double>(pt.cells[i]));
        }
        return (1 - (static_cast<double>(num) / static_cast<double>(den))) *
               (1 / (1 - (1 / std::sqrt(static_cast<double>(valid_cells_))))) * 100;
    }
    if (name == "tca") {
        const std::vector<double>& core = patch_base("core");
        return r_sum(core.data(), np);
    }
    if (name == "ndca" || name == "dcad") {
        const std::vector<double>& ncore = patch_base("ncore");
        const double ndca = r_sum(ncore.data(), np);
        return name == "ndca" ? ndca : ndca / ta * 100;
    }
    if (name == "pafrac") {
        if (np < 10) return kNaN;
        return pafrac_by(std::vector<std::int32_t>(static_cast<std::size_t>(np), 0), 1)[0];
    }

    // Composition.
    std::vector<double> class_area = class_metric("ca");
    const double ca_total = r_sum(class_area.data(), k);
    auto proportions = [&]() {
        std::vector<double> p(k);
        for (int c = 0; c < k; ++c) p[c] = (class_area[c] / ca_total * 100) / 100;
        return p;
    };
    if (name == "pr") return static_cast<double>(k);
    if (name == "prd") return static_cast<double>(k) / ta * 100;
    if (name == "rpr") return std::isnan(options_.classes_max) ? kNaN : static_cast<double>(k) / options_.classes_max * 100;
    if (name == "shdi" || name == "shei") {
        const std::vector<double> p = proportions();
        long double s = 0;
        for (double v : p) s += -v * std::log(v);
        if (name == "shdi") return static_cast<double>(s);
        return k == 1 ? 0.0 : static_cast<double>(s) / std::log(static_cast<double>(k));
    }
    if (name == "sidi" || name == "siei") {
        const std::vector<double> p = proportions();
        long double s = 0;
        for (double v : p) s += v * v;
        const double sidi = 1 - static_cast<double>(s);
        return name == "sidi" ? sidi : sidi / (1 - (1 / static_cast<double>(k)));
    }
    if (name == "msidi" || name == "msiei") {
        long double s = 0;
        for (double a : class_area) s += (a / ca_total) * (a / ca_total);
        const double msidi = -std::log(static_cast<double>(s));
        return name == "msidi" ? msidi : msidi / std::log(static_cast<double>(k));
    }

    // Configuration.
    const Adjacency& adj = adjacency();
    if (name == "ai") {
        const std::vector<double> ai = class_metric("ai");
        const std::vector<double> pland = class_metric("pland");
        long double s = 0;
        for (int c = 0; c < k; ++c) {
            const double v = ai[c] * (pland[c] / 100);
            if (!std::isnan(v)) s += v;
        }
        return static_cast<double>(s);
    }
    if (name == "pladj") {
        i64 like = 0, total = 0;
        for (int a = 0; a < k; ++a) {
            like += adj.orth(a, a);
            for (int b = 0; b < k; ++b) total += adj.orth(a, b);
            total += adj.orth(a, k + 1);
        }
        return static_cast<double>(like) / static_cast<double>(total) * 100;
    }
    if (name == "lsi") {
        i64 perim = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < a; ++b) perim += adj.orth(a, b);
            perim += adj.orth(a, k) + adj.orth(a, k + 1);
        }
        return static_cast<double>(perim) / min_edge(valid_cells_);
    }
    if (name == "iji") {
        if (k < 3) return kNaN;
        i64 e_total = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < a; ++b) e_total += adj.orth(a, b);
        }
        double s = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < a; ++b) {
                const double r = static_cast<double>(adj.orth(a, b)) / static_cast<double>(e_total);
                const double term = r * std::log(r);
                if (!std::isnan(term)) s += term;
            }
        }
        return (-s / std::log(0.5 * (k * (k - 1.0)))) * 100;
    }
    if (name == "contag") {
        if (k < 2) return kNaN;
        i64 total = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) total += adj.orth(a, b);
        }
        double esum = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) {
                const double r = static_cast<double>(adj.orth(a, b)) / static_cast<double>(total);
                const double term = r * std::log(r);
                if (!std::isnan(term)) esum += term;
            }
        }
        const double emax = 2 * std::log(static_cast<double>(k));
        return (1 + esum / emax) * 100;
    }
    if (name == "ent" || name == "condent" || name == "joinent" || name == "mutinf" || name == "relmutinf") {
        const bool eight = options_.neighbourhood == 8;
        auto cooc = [&](int a, int b) {
            return static_cast<double>(adj.orth(a, b) + (eight ? adj.diag[a * adj.dim + b] : 0));
        };
        std::vector<double> col(k, 0.0);
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) col[b] += cooc(a, b);
        }
        const double comp = entropy(col, options_.base);
        if (name == "ent") return comp;
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
        if (name == "joinent") return cplx;
        const double conf = cplx - comp;
        if (name == "condent") return conf;
        const double aggr = comp - conf;
        if (name == "mutinf") return aggr;
        return aggr == 0 ? 1.0 : aggr / comp;
    }
    throw std::invalid_argument("unknown landscape metric: " + name);
}

}  // namespace lsm
