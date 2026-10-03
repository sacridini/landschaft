// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLUMondo's allocation step for step (see clumondo.hpp). The arithmetic
// follows the Win32 build: floats stay floats (SSE2, /fp:precise), a float
// combined with a double is evaluated in double and rounded where it is
// stored, exp(float) is (float)exp(double).
//
// Corrected here (the reference is CLUMondo patched the same way): calc_age
// sets the age of restricted cells to -9999 after the first year, so the
// autonomous changes the manual says also happen there stop after it; here
// restricted cells age like the others.
//
// Extensions of the CLUMondo executable of May 2017, whose source is not
// published, written from its manual (Nov. 2017) and checked against its
// iteration log:
//  - demand types: a minimum demand counts as met when the supply exceeds
//    it, a maximum demand when the supply stays below it (deviation 0, the
//    iteration variable unchanged);
//  - iteration mode 2: each demand has its own largest deviation (%), and
//    the iteration stops when every demand is within its own;
//  - patches: in each iteration, a cell newly allocated to a system with
//    patches (in row order, unless an earlier patch took it) seeds a patch:
//    the cells of the square window of the system's radius that may convert
//    to it (conversion rules, not restricted) and whose suitability reaches
//    the threshold take it, regardless of resistance and demand; when the
//    patch (seed included, cells already in the system counted) stays
//    below the minimum size, nothing is allocated and the seed keeps its
//    old system.
//
// Extensions of ls.simulate (absent, CLUMondo's behaviour is unchanged):
//  - pair_resistance: a resistance per transition, subtracted from the
//    score of changing from one system to another;
//  - caps: at most n cells per year may make the transitions of a cap; when
//    more do, those with the largest margin over staying keep the change
//    (ties: row order) and the others stay.

#include "sim/clumondo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "lsm/parallel.hpp"
#include "sim/rng.hpp"

namespace sim {

namespace {

// The rule of allowed[from][to] at a cell: -30 (not allowed) or not.
inline bool forbidden(int code, int age, const float* allow_maps, std::int64_t ncell, std::int64_t cell) {
    if (code == 1) return false;
    int tempallow = 1;
    bool out = false;
    if (code == 0) out = true;
    if (code > 1 && code < 100) tempallow = static_cast<int>(allow_maps[static_cast<std::int64_t>(code - 2) * ncell + cell]);
    if (tempallow == 0) out = true;
    if (std::abs(code) > 100) tempallow = code;
    if (tempallow > 100 && tempallow < 1000 && age < tempallow - 100) out = true;
    if (tempallow < -100 && age >= std::abs(tempallow) - 100) out = true;
    if (tempallow > 1000 && age < tempallow - 1000) out = true;
    return out;
}

// sum_c v[s(c)] over the counted cells, in cell order: from the counts when
// every partial sum is exact in a double (the values are multiples of a
// common power of two and the total stays below 2^53 of them).
bool exact_from_counts(const std::vector<double>& v, const std::vector<std::int64_t>& counts, double& out) {
    int emin = 1 << 20;
    for (double x : v) {
        if (x == 0) continue;
        int e;
        double m = std::frexp(std::fabs(x), &e);  // x = m 2^e, m in [0.5, 1)
        // lowest set bit of the 53-bit mantissa
        auto q = static_cast<std::uint64_t>(std::ldexp(m, 53));
        int tz = 0;
        while ((q & 1u) == 0) {
            q >>= 1;
            ++tz;
        }
        emin = std::min(emin, e - 53 + tz);
    }
    if (emin == (1 << 20)) {
        out = 0;
        return true;
    }
    const double lim = 9007199254740992.0;  // 2^53
    double total = 0, bound = 0;
    for (std::size_t s = 0; s < v.size(); ++s) {
        if (counts[s] == 0 || v[s] == 0) continue;
        const double q = std::ldexp(v[s], -emin);  // an integer
        if (std::fabs(q) >= lim) return false;
        bound += std::fabs(q) * static_cast<double>(counts[s]);
        if (bound >= lim) return false;
        total += q * static_cast<double>(counts[s]);
    }
    out = std::ldexp(total, emin);
    return true;
}

}  // namespace

ClumondoYear clumondo_year(const ClumondoSettings& s, std::int64_t ncell, const int* oldco, const int* region,
                           const int* age, const float* suit, const float* allow_maps, int nmap,
                           const double* demand, float* dem_elas) {
    const int nsys = s.nsys, ndem = s.ndem;
    for (int a = 0; a < nsys * nsys; ++a) {
        const int code = s.allowed[a];
        if (code > 1 && code < 100 && code - 2 >= nmap) throw std::invalid_argument("conversion code without its map");
    }
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (region[c] > -9900 && (oldco[c] < 0 || oldco[c] >= nsys))
            throw std::invalid_argument("a cell's land system is not one of the model's");
    }
    ClumondoYear out;
    out.newco.assign(static_cast<std::size_t>(ncell), 0);
    // the rules do not change during the iteration: forbidden[i * ncell + c]
    std::vector<unsigned char> forbid(static_cast<std::size_t>(nsys) * static_cast<std::size_t>(ncell), 0);
    std::vector<unsigned char> stuck(static_cast<std::size_t>(ncell), 0);
    const bool par = ncell * nsys >= (1 << 15);
    lsm::parallel_for(par, ncell, [&](std::int64_t c) {
        if (region[c] <= -9900) return;
        const int old = oldco[c];
        bool any = false;
        for (int i = 0; i < nsys; ++i) {
            const bool f = forbidden(s.allowed[old * nsys + i], age[c], allow_maps, ncell, c);
            forbid[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncell) + c] = f;
            any = any || !f;
        }
        stuck[c] = !any;
    });
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (stuck[c])
            throw std::runtime_error("too many constraints: no land system is allowed at cell " + std::to_string(c));
    }
    std::vector<double> lus(static_cast<std::size_t>(nsys) * ndem);
    for (std::size_t k = 0; k < lus.size(); ++k) lus[k] = static_cast<double>(s.lusmatrix[k]);
    std::vector<float> speed(ndem, s.speed), ran(ndem), sumelas(static_cast<std::size_t>(nsys) * nsys);
    std::vector<double> area(ndem);
    std::vector<std::int64_t> counts(nsys);
    std::vector<double> col(nsys);
    CrtRand crt;
    const bool patches = !s.patch_radius.empty();
    if (patches && (s.nx <= 0 || ncell % s.nx != 0)) throw std::invalid_argument("patches need the map's columns");
    const std::int64_t ny = patches ? ncell / s.nx : 0;
    std::vector<unsigned char> in_patch(patches ? static_cast<std::size_t>(ncell) : 0);
    std::vector<std::int64_t> members;
    const bool pairres = !s.pair_resistance.empty();
    std::vector<std::pair<float, std::int64_t>> capped;
    const bool typed = !s.kind.empty();
    const bool per_demand = !s.dev_limit.empty();
    std::vector<double> eff(ndem);
    bool within = true;
    int loop = 0;
    float maxdiff = 0, totdiff = 0;
    do {
        ++loop;
        // calc_change_ch: the iteration variable of each (from, to) pair, then the best system per cell
        for (int o = 0; o < nsys; ++o) {
            for (int i = 0; i < nsys; ++i) {
                float se = 0;
                for (int d = 0; d < ndem; ++d) {
                    const int to = s.lusconv[i * ndem + d];
                    if (to == -1) continue;
                    const int from = s.lusconv[o * ndem + d];
                    if (from < to) se += s.weight[d] * dem_elas[d];
                    if (from > to) se -= s.weight[d] * dem_elas[d];
                }
                sumelas[o * nsys + i] = se;
            }
        }
        lsm::parallel_for(par, ncell, [&](std::int64_t c) {
            const int r = region[c];
            if (r <= -9999) {
                out.newco[c] = kNoData;
                return;
            }
            if (r < -9900) {
                out.newco[c] = oldco[c];
                return;
            }
            const int old = oldco[c];
            float best = -30;
            int maxcov = -1;
            for (int i = 0; i < nsys; ++i) {
                float t;
                if (forbid[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncell) + c]) {
                    t = -30;
                } else {
                    t = suit[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncell) + c] + sumelas[old * nsys + i];
                    if (i == old) t = t + s.resistance[i];
                    else if (pairres) t = t - s.pair_resistance[old * nsys + i];
                }
                if (t > best) {
                    best = t;
                    maxcov = i;
                }
            }
            out.newco[c] = maxcov;  // -1: no suitability above CLUMondo's floor (checked below)
        });
        for (std::int64_t c = 0; c < ncell; ++c) {
            if (out.newco[c] == -1)
                throw std::runtime_error("too many constraints: no land system is allowed at cell " +
                                         std::to_string(c) + " (suitabilities missing?)");
        }
        if (patches) {
            std::fill(in_patch.begin(), in_patch.end(), 0);
            for (std::int64_t c = 0; c < ncell; ++c) {
                const int sys = out.newco[c];
                if (region[c] < 0 || in_patch[c] || sys == oldco[c] || s.patch_radius[sys] <= 0) continue;
                const int rad = s.patch_radius[sys];
                const std::int64_t j = c / s.nx, k = c % s.nx;
                members.clear();
                std::int64_t size = 1;
                for (std::int64_t jj = j - rad; jj <= j + rad; ++jj) {
                    if (jj < 0 || jj >= ny) continue;
                    for (std::int64_t kk = k - rad; kk <= k + rad; ++kk) {
                        if (kk < 0 || kk >= s.nx) continue;
                        const std::int64_t cc = jj * s.nx + kk;
                        if (cc == c || region[cc] < 0 || in_patch[cc]) continue;
                        if (out.newco[cc] == sys) {
                            members.push_back(cc);
                            ++size;
                            continue;
                        }
                        const std::size_t at = static_cast<std::size_t>(sys) * static_cast<std::size_t>(ncell) + cc;
                        if (forbid[at] || suit[at] < s.patch_threshold[sys]) continue;
                        members.push_back(cc);
                        ++size;
                    }
                }
                in_patch[c] = 1;
                if (size < s.patch_min[sys]) {
                    out.newco[c] = oldco[c];
                    continue;
                }
                for (std::int64_t cc : members) {
                    out.newco[cc] = sys;
                    in_patch[cc] = 1;
                }
            }
        }
        for (std::size_t k = 0; k < s.cap_cells.size(); ++k) {
            const auto& pairs = s.cap_pairs[k];
            capped.clear();
            for (std::int64_t c = 0; c < ncell; ++c) {
                if (region[c] < 0) continue;
                const int old = oldco[c], nw = out.newco[c];
                if (nw == old || !pairs[old * nsys + nw]) continue;
                const std::size_t so = static_cast<std::size_t>(old) * static_cast<std::size_t>(ncell) + c;
                if (forbid[so]) continue;  // the cell cannot stay
                const float t_new = suit[static_cast<std::size_t>(nw) * static_cast<std::size_t>(ncell) + c] +
                                    sumelas[old * nsys + nw] - (pairres ? s.pair_resistance[old * nsys + nw] : 0.0f);
                const float t_old = suit[so] + sumelas[old * nsys + old] + s.resistance[old];
                capped.emplace_back(t_new - t_old, c);
            }
            if (static_cast<std::int64_t>(capped.size()) <= s.cap_cells[k]) continue;
            std::stable_sort(capped.begin(), capped.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t q = static_cast<std::size_t>(s.cap_cells[k]); q < capped.size(); ++q)
                out.newco[capped[q].second] = oldco[capped[q].second];
        }
        // comp_demand
        for (int d = 0; d < ndem; ++d) {
            speed[d] += 0.0002f;
            const float inv = 1 / speed[d];
            crt.seed(static_cast<std::uint32_t>(static_cast<std::int64_t>(101 - inv)));
            ran[d] = static_cast<float>(crt.next()) + inv;
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (std::int64_t c = 0; c < ncell; ++c) {
            if (region[c] > -9999) ++counts[out.newco[c]];
        }
        for (int d = 0; d < ndem; ++d) {
            for (int i = 0; i < nsys; ++i) col[i] = lus[i * ndem + d];
            double a;
            if (!exact_from_counts(col, counts, a)) {
                a = 0;
                for (std::int64_t c = 0; c < ncell; ++c) {
                    if (region[c] > -9999) a += col[out.newco[c]];
                }
            }
            area[d] = a;
        }
        for (int d = 0; d < ndem; ++d) {
            eff[d] = area[d];
            if (typed && ((s.kind[d] == 1 && area[d] >= demand[d]) || (s.kind[d] == 2 && area[d] <= demand[d])))
                eff[d] = demand[d];  // a minimum exceeded or a maximum not reached: met
        }
        maxdiff = 0;
        totdiff = 0;
        within = true;
        for (int d = 0; d < ndem; ++d) {
            const double step = ((eff[d] - demand[d]) / demand[d]) / static_cast<double>(speed[d] * ran[d]);
            dem_elas[d] = static_cast<float>(static_cast<double>(dem_elas[d]) - step);
            if (std::fabs(dem_elas[d]) > 2) dem_elas[d] = 0;
        }
        for (int d = 0; d < ndem; ++d) {
            double r;
            if (s.itermode == 0) {
                r = demand[d] > 0 ? std::fabs((eff[d] - demand[d]) / demand[d] * 100)
                                  : std::fabs((eff[d] - demand[d]) / static_cast<double>(ncell) * 100);
            } else {
                r = std::fabs(eff[d] - demand[d]);
            }
            if (r > static_cast<double>(maxdiff)) maxdiff = static_cast<float>(r);
            totdiff = static_cast<float>(static_cast<double>(totdiff) + r);
            if (per_demand && r > static_cast<double>(s.dev_limit[d])) within = false;
        }
    } while (loop < s.max_loops && (per_demand ? !within
                                                : (totdiff / static_cast<float>(ndem) > s.mean_dev ||
                                                   maxdiff > s.max_dev)));
    out.loops = loop;
    out.converged = loop < s.max_loops;
    out.max_diff = maxdiff;
    out.mean_diff = totdiff / static_cast<float>(ndem);
    out.supply = area;
    out.cells = counts;
    out.age.assign(age, age + ncell);
    if (!out.converged) return out;
    // autonomous_change: succession of the cells that kept their system
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (region[c] <= -9999) continue;
        const int old = oldco[c];
        int count = 0;
        for (int i = 0; i < nsys; ++i) {
            const int code = s.allowed[old * nsys + i];
            if (code == 1) continue;
            int tempallow = 1;
            if (code > 1 && code < 100) tempallow = static_cast<int>(allow_maps[static_cast<std::int64_t>(code - 2) * ncell + c]);
            if (code > 1000) tempallow = code;
            if (tempallow > 1000 && ++count > 1)
                throw std::runtime_error("a land system cannot change autonomously into two others");
            if (out.newco[c] == old && tempallow > 1000 && age[c] >= tempallow - 1000) out.newco[c] = i;
        }
    }
    // calc_age (restricted cells age too: see the top of the file)
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (region[c] > -9999) {
            out.age[c] = out.newco[c] == oldco[c] ? age[c] + 1 : 1;
        } else {
            out.age[c] = kNoData;
        }
    }
    // the cells after the autonomous changes
    std::fill(out.cells.begin(), out.cells.end(), 0);
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (region[c] > -9999) ++out.cells[out.newco[c]];
    }
    return out;
}

std::vector<float> clumondo_logit(std::int64_t ncell, const int* region, const float* x, int nx, double constant,
                                  const float* coef) {
    std::vector<float> out(static_cast<std::size_t>(ncell));
    std::int64_t bad = -1;
    for (std::int64_t c = 0; c < ncell; ++c) {
        if (region[c] < 0) {
            out[c] = -9999;
            continue;
        }
        float t = static_cast<float>(constant);
        for (int m = 0; m < nx; ++m) t += coef[m] * x[static_cast<std::int64_t>(m) * ncell + c];
        if (t > 87 && bad < 0) bad = c;
        const float e = static_cast<float>(std::exp(static_cast<double>(t)));
        out[c] = e / (1 + e);
    }
    if (bad >= 0)
        throw std::runtime_error("regression cannot be calculated: linear predictor above 87 at cell " +
                                 std::to_string(bad));
    return out;
}

std::vector<float> clumondo_fraction(std::int64_t ny, std::int64_t nx, const int* oldco, const int* region, int u,
                                     const float* kernel, int radius) {
    std::vector<float> out(static_cast<std::size_t>(ny * nx));
    const int side = 2 * radius + 1;
    lsm::parallel_for(ny * nx >= (1 << 14), ny, [&](std::int64_t j) {
        for (std::int64_t k = 0; k < nx; ++k) {
            const std::int64_t c = j * nx + k;
            if (region[c] == kNoData) {
                out[c] = -9999;
                continue;
            }
            float novalue = 0, cum = 0;
            for (int m = -radius; m <= radius; ++m) {
                for (int n = -radius; n <= radius; ++n) {
                    const std::int64_t jj = j + m, kk = k + n;
                    if (jj < 0 || jj >= ny || kk < 0 || kk >= nx) continue;
                    const std::int64_t cc = jj * nx + kk;
                    if (region[cc] <= -9999) continue;
                    const float w = kernel[(radius + m) * side + (radius + n)];
                    novalue += w;
                    if (oldco[cc] == u) cum += w;
                }
            }
            out[c] = novalue > 0 ? cum / novalue : 0.0f;
        }
    });
    return out;
}

std::vector<int> clumondo_random_age(std::int64_t ncell, int max_age, std::uint32_t seed) {
    if (max_age < 1) throw std::invalid_argument("the maximum age must be at least 1");
    CrtRand crt(seed);
    std::vector<int> out(static_cast<std::size_t>(ncell));
    for (std::int64_t c = 0; c < ncell; ++c) out[c] = crt.next() % max_age + 1;
    return out;
}

}  // namespace sim
