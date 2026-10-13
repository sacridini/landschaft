// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of ls.simulate (landschaft._core.sim_*).

#include "sim/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "lsm/rmath.hpp"
#include "sim/change.hpp"
#include "sim/clues.hpp"
#include "sim/clumondo.hpp"
#include "sim/demand.hpp"
#include "sim/futures.hpp"
#include "sim/molusce.hpp"
#include "sim/observed.hpp"
#include "sim/ordered.hpp"
#include "sim/patches.hpp"
#include "sim/plus.hpp"
#include "sim/rglm.hpp"
#include "sim/rules.hpp"
#include "sim/rng.hpp"
#include "sim/similarity.hpp"
#include "sim/validation.hpp"
#include "sim/woe.hpp"

namespace py = pybind11;

namespace {

using i64 = std::int64_t;

// A C-contiguous numpy array of a numeric type as a map (no copy).
sim::MapView map_view(const py::array& a, std::optional<double> nodata) {
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument("the map must be C-contiguous");
    sim::MapView m;
    m.data = a.data();
    m.n = static_cast<i64>(a.size());
    m.has_nodata = nodata.has_value();
    m.nodata = nodata.value_or(0.0);
    const auto dt = a.dtype();
    const char kind = dt.kind();
    const auto size = dt.itemsize();
    if (kind == 'u' && size == 1) m.type = sim::DType::u8;
    else if (kind == 'i' && size == 1) m.type = sim::DType::i8;
    else if (kind == 'b' && size == 1) m.type = sim::DType::u8;
    else if (kind == 'u' && size == 2) m.type = sim::DType::u16;
    else if (kind == 'i' && size == 2) m.type = sim::DType::i16;
    else if (kind == 'u' && size == 4) m.type = sim::DType::u32;
    else if (kind == 'i' && size == 4) m.type = sim::DType::i32;
    else if (kind == 'i' && size == 8) m.type = sim::DType::i64;
    else if (kind == 'f' && size == 4) m.type = sim::DType::f32;
    else if (kind == 'f' && size == 8) m.type = sim::DType::f64;
    else throw std::invalid_argument("unsupported map type " + std::string(py::str(dt)));
    return m;
}

py::array_t<double> sim_map_classes(const py::array& map, std::optional<double> nodata) {
    const auto view = map_view(map, nodata);
    std::vector<double> c;
    {
        py::gil_scoped_release release;
        c = sim::map_classes(view);
    }
    py::array_t<double> out(static_cast<py::ssize_t>(c.size()));
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

py::array_t<std::int64_t> sim_crosstab(const py::array& a, const py::array& b,
                                       const py::array_t<double, py::array::c_style | py::array::forcecast>& ca,
                                       const py::array_t<double, py::array::c_style | py::array::forcecast>& cb,
                                       std::optional<double> nodata_a, std::optional<double> nodata_b) {
    const auto va = map_view(a, nodata_a), vb = map_view(b, nodata_b);
    std::vector<double> classes_a(ca.data(), ca.data() + ca.size()), classes_b(cb.data(), cb.data() + cb.size());
    std::vector<std::int64_t> c;
    {
        py::gil_scoped_release release;
        c = sim::crosstab(va, vb, classes_a, classes_b);
    }
    py::array_t<std::int64_t> out({static_cast<py::ssize_t>(classes_a.size()), static_cast<py::ssize_t>(classes_b.size())});
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

py::array_t<std::int32_t> sim_change_count(const std::vector<py::array>& maps, std::optional<double> nodata) {
    std::vector<sim::MapView> views;
    for (const auto& m : maps) views.push_back(map_view(m, nodata));
    std::vector<std::int32_t> c;
    {
        py::gil_scoped_release release;
        c = sim::change_count(views);
    }
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(c.size()));
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

void check_count(i64 n) {
    if (n < 0) throw std::invalid_argument("n must be >= 0");
}

py::array_t<std::int32_t> sim_crt_rand(std::uint32_t seed, i64 n) {
    check_count(n);
    py::array_t<std::int32_t> out(n);
    auto* p = out.mutable_data();
    sim::CrtRand r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = r.next();
    return out;
}

py::array_t<double> sim_drand48(i64 seed, i64 n) {
    check_count(n);
    py::array_t<double> out(n);
    auto* p = out.mutable_data();
    sim::Drand48 r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = r.next_double();
    return out;
}

py::array_t<i64> sim_lrand48(i64 seed, i64 n, bool is_signed) {
    check_count(n);
    py::array_t<i64> out(n);
    auto* p = out.mutable_data();
    sim::Drand48 r(seed);
    for (i64 i = 0; i < n; ++i) p[i] = is_signed ? r.next_signed() : r.next_long();
    return out;
}

using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

py::array_t<std::int32_t> sim_woe_buckets(const py::array& var, std::optional<double> nodata, bool categorical,
                                          const DArray& classes, double start, double inc, i64 n_buckets) {
    const auto view = map_view(var, nodata);
    std::vector<double> c(classes.data(), classes.data() + classes.size());
    std::vector<std::int32_t> b;
    {
        py::gil_scoped_release release;
        b = sim::woe_buckets(view, categorical, c, start, inc, n_buckets);
    }
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(b.size()));
    std::copy(b.begin(), b.end(), out.mutable_data());
    return out;
}

py::array_t<std::int64_t> sim_woe_counts(const py::array& t0, const py::array& t1, const IArray& buckets,
                                         i64 n_buckets, double from, const DArray& final_classes,
                                         std::optional<double> nodata_t0, std::optional<double> nodata_t1) {
    const auto a = map_view(t0, nodata_t0), b = map_view(t1, nodata_t1);
    std::vector<std::int32_t> k(buckets.data(), buckets.data() + buckets.size());
    std::vector<double> fc(final_classes.data(), final_classes.data() + final_classes.size());
    std::vector<std::int64_t> c;
    {
        py::gil_scoped_release release;
        c = sim::woe_counts(a, b, k, n_buckets, from, fc);
    }
    py::array_t<std::int64_t> out({static_cast<py::ssize_t>(n_buckets), static_cast<py::ssize_t>(fc.size())});
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
}

py::array_t<double> sim_woe_probability(const py::array& landscape, std::optional<double> nodata, double from,
                                        const std::vector<IArray>& buckets, const std::vector<DArray>& weights) {
    const auto view = map_view(landscape, nodata);
    std::vector<const std::int32_t*> b;
    std::vector<std::vector<double>> w;
    for (const auto& x : buckets) {
        if (static_cast<i64>(x.size()) != view.n) throw std::invalid_argument("buckets and landscape differ in size");
        b.push_back(x.data());
    }
    for (const auto& x : weights) w.emplace_back(x.data(), x.data() + x.size());
    std::vector<double> p;
    {
        py::gil_scoped_release release;
        p = sim::woe_probability(view, from, b, w, view.n);
    }
    py::array_t<double> out(static_cast<py::ssize_t>(p.size()));
    std::copy(p.begin(), p.end(), out.mutable_data());
    return out;
}

// x: (p, n) C-contiguous, i.e. the n x p model matrix column by column (R's layout)
py::dict sim_glm_logit(const DArray& x, const DArray& y, int maxit, double epsilon) {
    if (x.ndim() != 2) throw std::invalid_argument("x must be (p, n)");
    const int p = static_cast<int>(x.shape(0)), n = static_cast<int>(x.shape(1));
    std::vector<double> xv(x.data(), x.data() + x.size()), yv(y.data(), y.data() + y.size());
    sim::GlmFit fit;
    {
        py::gil_scoped_release release;
        fit = sim::glm_logit(xv, yv, n, p, maxit, epsilon);
    }
    py::dict d;
    d["coefficients"] = py::array_t<double>(fit.coefficients.size(), fit.coefficients.data());
    d["pivot"] = py::array_t<int>(fit.pivot.size(), fit.pivot.data());
    d["rank"] = fit.rank;
    d["iterations"] = fit.iterations;
    d["converged"] = fit.converged;
    d["boundary"] = fit.boundary;
    d["deviance"] = fit.deviance;
    d["null_deviance"] = fit.null_deviance;
    return d;
}

py::array_t<double> sim_glm_predict(const DArray& x, const DArray& coefficients, const py::array_t<int>& pivot, int rank) {
    if (x.ndim() != 2) throw std::invalid_argument("x must be (p, n)");
    const int p = static_cast<int>(x.shape(0));
    const i64 n = static_cast<i64>(x.shape(1));
    sim::GlmFit fit;
    fit.coefficients.assign(coefficients.data(), coefficients.data() + coefficients.size());
    fit.pivot.assign(pivot.data(), pivot.data() + pivot.size());
    fit.rank = rank;
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::glm_logit_predict(x.data(), n, p, fit);
    }
    py::array_t<double> r(static_cast<py::ssize_t>(out.size()));
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

py::dict sim_clues(const DArray& tprob, const DArray& landuse, const DArray& demand, const DArray& codes, double jitter,
                   double scale, double max_iter, double max_diff, double mean_diff, nlm::RRng& rng) {
    const int ncode = static_cast<int>(codes.size());
    const i64 ncell = static_cast<i64>(landuse.size());
    if (tprob.size() != static_cast<py::ssize_t>(ncode * ncell)) throw std::invalid_argument("tprob must be (classes, cells)");
    if (demand.size() != ncode) throw std::invalid_argument("one demand per class");
    sim::CluesResult r;
    {
        py::gil_scoped_release release;
        r = sim::clues_allocate(tprob.data(), landuse.data(), ncell, demand.data(), codes.data(), ncode, jitter, scale,
                                max_iter, max_diff, mean_diff, rng);
    }
    py::dict d;
    py::array_t<double> a(static_cast<py::ssize_t>(r.allocation.size()));
    std::copy(r.allocation.begin(), r.allocation.end(), a.mutable_data());
    d["allocation"] = a;
    d["iterations"] = r.iterations;
    d["converged"] = r.converged;
    d["max_diff"] = r.max_diff;
    d["mean_diff"] = r.mean_diff;
    return d;
}

py::array_t<double> sim_focal_mean(const DArray& x, const DArray& w) {
    if (x.ndim() != 2 || w.ndim() != 2) throw std::invalid_argument("x and w must be 2-D");
    if (w.shape(0) % 2 == 0 || w.shape(1) % 2 == 0) throw std::invalid_argument("the window must have odd sides");
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::focal_mean(x.data(), x.shape(0), x.shape(1), w.data(), static_cast<int>(w.shape(0)),
                              static_cast<int>(w.shape(1)));
    }
    py::array_t<double> r({x.shape(0), x.shape(1)});
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v) {
    py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

py::dict sim_clumondo_year(const IArray& oldco, const IArray& region, const IArray& age, const FArray& suit,
                           const FArray& allow_maps, const IArray& allowed, const IArray& lusconv,
                           const FArray& lusmatrix, const FArray& weight, const FArray& resistance,
                           const DArray& demand, const FArray& dem_elas, int itermode, float mean_dev, float max_dev,
                           float speed, int max_loops, const IArray& kind, const FArray& dev_limit, i64 nx,
                           const IArray& patch_radius, const FArray& patch_threshold, const IArray& patch_min,
                           const FArray& pair_resistance, const py::array_t<std::uint8_t, py::array::c_style |
                           py::array::forcecast>& cap_pairs, const py::array_t<std::int64_t, py::array::c_style |
                           py::array::forcecast>& cap_cells) {
    const i64 ncell = static_cast<i64>(oldco.size());
    const int nsys = static_cast<int>(resistance.size());
    const int ndem = static_cast<int>(weight.size());
    if (region.size() != ncell || age.size() != ncell) throw std::invalid_argument("region and age must match the map");
    if (suit.size() != static_cast<py::ssize_t>(nsys) * ncell) throw std::invalid_argument("suit must be (systems, cells)");
    if (allowed.size() != nsys * nsys) throw std::invalid_argument("allowed must be (systems, systems)");
    if (lusconv.size() != nsys * ndem || lusmatrix.size() != nsys * ndem)
        throw std::invalid_argument("lusconv and lusmatrix must be (systems, demands)");
    if (demand.size() != ndem || dem_elas.size() != ndem) throw std::invalid_argument("one demand per demand type");
    const int nmap = ncell ? static_cast<int>(allow_maps.size() / ncell) : 0;
    sim::ClumondoSettings s;
    s.nsys = nsys;
    s.ndem = ndem;
    s.allowed.assign(allowed.data(), allowed.data() + allowed.size());
    s.lusconv.assign(lusconv.data(), lusconv.data() + lusconv.size());
    s.lusmatrix.assign(lusmatrix.data(), lusmatrix.data() + lusmatrix.size());
    s.weight.assign(weight.data(), weight.data() + ndem);
    s.resistance.assign(resistance.data(), resistance.data() + nsys);
    s.itermode = itermode;
    s.mean_dev = mean_dev;
    s.max_dev = max_dev;
    s.speed = speed;
    s.max_loops = max_loops;
    if (kind.size()) {
        if (kind.size() != ndem) throw std::invalid_argument("one demand type per demand");
        s.kind.assign(kind.data(), kind.data() + ndem);
    }
    if (dev_limit.size()) {
        if (dev_limit.size() != ndem) throw std::invalid_argument("one deviation per demand");
        s.dev_limit.assign(dev_limit.data(), dev_limit.data() + ndem);
    }
    s.nx = nx;
    if (patch_radius.size()) {
        if (patch_radius.size() != nsys || patch_threshold.size() != nsys || patch_min.size() != nsys)
            throw std::invalid_argument("patch settings: one per land system");
        s.patch_radius.assign(patch_radius.data(), patch_radius.data() + nsys);
        s.patch_threshold.assign(patch_threshold.data(), patch_threshold.data() + nsys);
        s.patch_min.assign(patch_min.data(), patch_min.data() + nsys);
    }
    if (pair_resistance.size()) {
        if (pair_resistance.size() != nsys * nsys) throw std::invalid_argument("pair_resistance must be (systems, systems)");
        s.pair_resistance.assign(pair_resistance.data(), pair_resistance.data() + nsys * nsys);
    }
    const auto ncap = cap_cells.size();
    if (cap_pairs.size() != ncap * nsys * nsys) throw std::invalid_argument("cap_pairs must be (caps, systems, systems)");
    for (py::ssize_t k = 0; k < ncap; ++k) {
        s.cap_pairs.emplace_back(cap_pairs.data() + k * nsys * nsys, cap_pairs.data() + (k + 1) * nsys * nsys);
        s.cap_cells.push_back(cap_cells.data()[k]);
    }
    std::vector<float> elas(dem_elas.data(), dem_elas.data() + ndem);
    sim::ClumondoYear r;
    {
        py::gil_scoped_release release;
        r = sim::clumondo_year(s, ncell, oldco.data(), region.data(), age.data(), suit.data(), allow_maps.data(), nmap,
                               demand.data(), elas.data());
    }
    py::dict d;
    d["newco"] = to_array(r.newco);
    d["age"] = to_array(r.age);
    d["dem_elas"] = to_array(elas);
    d["supply"] = to_array(r.supply);
    d["cells"] = to_array(r.cells);
    d["loops"] = r.loops;
    d["converged"] = r.converged;
    d["max_diff"] = r.max_diff;
    d["mean_diff"] = r.mean_diff;
    return d;
}

py::array_t<float> sim_clumondo_logit(const IArray& region, const FArray& x, double constant, const FArray& coef) {
    const i64 ncell = static_cast<i64>(region.size());
    const int nx = static_cast<int>(coef.size());
    if (x.size() != static_cast<py::ssize_t>(nx) * ncell) throw std::invalid_argument("x must be (drivers, cells)");
    std::vector<float> out;
    {
        py::gil_scoped_release release;
        out = sim::clumondo_logit(ncell, region.data(), x.data(), nx, constant, coef.data());
    }
    return to_array(out);
}

py::array_t<float> sim_clumondo_fraction(const IArray& oldco, const IArray& region, int system, const FArray& kernel) {
    if (oldco.ndim() != 2 || region.ndim() != 2) throw std::invalid_argument("the maps must be 2-D");
    if (kernel.ndim() != 2 || kernel.shape(0) != kernel.shape(1) || kernel.shape(0) % 2 == 0)
        throw std::invalid_argument("the kernel must be square with odd sides");
    std::vector<float> out;
    {
        py::gil_scoped_release release;
        out = sim::clumondo_fraction(oldco.shape(0), oldco.shape(1), oldco.data(), region.data(), system,
                                     kernel.data(), static_cast<int>(kernel.shape(0) / 2));
    }
    py::array_t<float> r({oldco.shape(0), oldco.shape(1)});
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

py::array_t<int> sim_clumondo_random_age(i64 ncell, int max_age, std::uint32_t seed) {
    return to_array(sim::clumondo_random_age(ncell, max_age, seed));
}

py::dict sim_futures_run(const IArray& developed, const IArray& region, const std::optional<IArray>& pot_region,
                         const FArray& devpressure, const FArray& aggregated, const std::optional<FArray>& weight,
                         const std::optional<IArray>& zone, int nregions, int npot, const DArray& intercept,
                         const DArray& devpressure_coef, int neighborhood, int approach, float gamma, float scaling,
                         const std::vector<std::vector<int>>& patch_sizes, bool single_column, float compactness_mean,
                         float compactness_range, int num_neighbors, bool seed_probability, const FArray& incentive,
                         bool user_effects, const IArray& zone_ids, const IArray& zone_regions,
                         const FArray& zone_effects, const FArray& stringency, const FArray& demand, int nsteps,
                         std::int64_t seed, const std::optional<py::dict>& climate) {
    if (developed.ndim() != 2) throw std::invalid_argument("developed must be 2-D");
    const i64 rows = developed.shape(0), cols = developed.shape(1), n = rows * cols;
    auto check = [n](const py::array& a, const char* what) {
        if (a.size() != n) throw std::invalid_argument(std::string(what) + " differs in size from developed");
    };
    check(region, "region");
    check(devpressure, "devpressure");
    check(aggregated, "aggregated");
    if (pot_region) check(*pot_region, "pot_region");
    if (weight) check(*weight, "weight");
    if (zone) check(*zone, "zone");
    sim::FuturesSettings s;
    s.rows = rows;
    s.cols = cols;
    s.nregions = nregions;
    s.npot = npot;
    s.intercept.assign(intercept.data(), intercept.data() + intercept.size());
    s.devpressure_coef.assign(devpressure_coef.data(), devpressure_coef.data() + devpressure_coef.size());
    s.neighborhood = neighborhood;
    s.approach = approach;
    s.gamma = gamma;
    s.scaling = scaling;
    s.patch_sizes = patch_sizes;
    s.single_column = single_column;
    s.max_patch_size = 1;
    for (const auto& v : patch_sizes)
        for (int x : v) s.max_patch_size = std::max(s.max_patch_size, x);
    s.compactness_mean = compactness_mean;
    s.compactness_range = compactness_range;
    s.num_neighbors = num_neighbors;
    s.seed_probability = seed_probability;
    s.incentive.assign(incentive.data(), incentive.data() + incentive.size());
    s.use_zone = zone.has_value();
    s.user_effects = user_effects;
    s.zone_ids.assign(zone_ids.data(), zone_ids.data() + zone_ids.size());
    s.zone_regions.assign(zone_regions.data(), zone_regions.data() + zone_regions.size());
    s.zone_effects.assign(zone_effects.data(), zone_effects.data() + zone_effects.size());
    s.stringency.assign(stringency.data(), stringency.data() + stringency.size());
    s.use_weight = weight.has_value();
    s.demand.assign(demand.data(), demand.data() + demand.size());
    s.nsteps = nsteps;
    s.seed = seed;
    sim::FuturesClimate cl;
    if (climate) {
        const py::dict& c = *climate;
        auto ivec = [&](const char* k) {
            IArray a = py::cast<IArray>(c[k]);
            return std::vector<int>(a.data(), a.data() + a.size());
        };
        auto fvec = [&](const char* k) {
            FArray a = py::cast<FArray>(c[k]);
            return std::vector<float>(a.data(), a.data() + a.size());
        };
        auto fmaps = [&](const py::handle& h) {  // (k, cells) -> k vectors
            FArray a = py::cast<FArray>(h);
            std::vector<std::vector<float>> out;
            const i64 k = a.size() / n;
            for (i64 i = 0; i < k; ++i) out.emplace_back(a.data() + i * n, a.data() + (i + 1) * n);
            return out;
        };
        cl.depth_mode = py::cast<bool>(c["depth_mode"]);
        cl.huc = ivec("huc");
        cl.nhuc = py::cast<int>(c["nhuc"]);
        cl.huc_ids = ivec("huc_ids");
        cl.flood_steps = ivec("flood_steps");
        if (cl.depth_mode) {
            cl.probabilities = fvec("probabilities");
            for (auto item : py::cast<py::list>(c["depths"])) cl.depths.push_back(fmaps(item));
        } else {
            cl.flood_probability = fmaps(c["flood_probability"]);
            cl.hand = fvec("hand");
            cl.percentile = py::cast<float>(c["percentile"]);
        }
        cl.adaptive_capacity = fvec("adaptive_capacity");
        if (c.contains("adaptation")) cl.adaptation = ivec("adaptation");
        DArray lv = py::cast<DArray>(c["levels"]);
        cl.levels.assign(lv.data(), lv.data() + lv.size());
        DArray dm = py::cast<DArray>(c["damage"]);
        const i64 nl = lv.size();
        for (i64 i = 0; i < dm.size() / nl; ++i) cl.damage.emplace_back(dm.data() + i * nl, dm.data() + (i + 1) * nl);
        cl.ddf_region = ivec("ddf_region");
        FArray resp = py::cast<FArray>(c["response"]);
        cl.vulnerability_a = resp.data()[0];
        cl.vulnerability_b = resp.data()[1];
        cl.resilience_a = resp.data()[2];
        cl.resilience_b = resp.data()[3];
        cl.stddev = py::cast<float>(c["stddev"]);
        cl.has_matrix = c.contains("matrix");
        if (cl.has_matrix) {
            cl.from_ids = ivec("from_ids");
            cl.to_ids = ivec("to_ids");
            FArray m = py::cast<FArray>(c["matrix"]);
            const i64 nt = static_cast<i64>(cl.to_ids.size());
            for (std::size_t i = 0; i < cl.from_ids.size(); ++i)
                cl.matrix.emplace_back(m.data() + i * nt, m.data() + (i + 1) * nt);
            cl.population = fvec("population");
        }
        cl.region_ids = ivec("region_ids");
        if (static_cast<i64>(cl.huc.size()) != n || static_cast<i64>(cl.ddf_region.size()) != n ||
            static_cast<i64>(cl.adaptive_capacity.size()) != n)
            throw std::invalid_argument("climate maps differ in size from developed");
        s.climate = &cl;
    }
    sim::FuturesResult r;
    {
        py::gil_scoped_release release;
        r = sim::futures_run(s, developed.data(), region.data(), pot_region ? pot_region->data() : nullptr,
                             devpressure.data(), aggregated.data(), weight ? weight->data() : nullptr,
                             zone ? zone->data() : nullptr);
    }
    py::dict d;
    py::array_t<int> dev({rows, cols});
    std::copy(r.developed.begin(), r.developed.end(), dev.mutable_data());
    py::array_t<int> series({static_cast<i64>(r.series.size()), rows, cols});
    for (std::size_t k = 0; k < r.series.size(); ++k)
        std::copy(r.series[k].begin(), r.series[k].end(), series.mutable_data() + k * n);
    d["developed"] = dev;
    d["series"] = series;
    d["converted"] = to_array(r.converted);
    d["overflow"] = to_array(r.overflow);
    d["warnings"] = to_array(r.warnings);
    if (climate) {
        py::array_t<int> ad({static_cast<i64>(r.adaptation.size()), rows, cols});
        for (std::size_t k = 0; k < r.adaptation.size(); ++k)
            std::copy(r.adaptation[k].begin(), r.adaptation[k].end(), ad.mutable_data() + k * n);
        d["adaptation"] = ad;
        d["flood_steps"] = to_array(r.flood_steps);
        d["flood_hucs"] = to_array(r.flood_hucs);
        d["flood_probabilities"] = to_array(r.flood_probabilities);
        const i64 nm = r.moved.empty() ? 0 : static_cast<i64>(r.moved[0].size());
        py::array_t<float> mv({static_cast<i64>(r.moved.size()), nm});
        for (std::size_t k = 0; k < r.moved.size(); ++k)
            std::copy(r.moved[k].begin(), r.moved[k].end(), mv.mutable_data() + k * nm);
        d["moved"] = mv;
        d["demand"] = to_array(r.demand);
    }
    return d;
}

py::array_t<float> sim_futures_mfilter(const DArray& input, const DArray& matrix) {
    if (input.ndim() != 2 || matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1) || matrix.shape(0) % 2 == 0)
        throw std::invalid_argument("a 2-D input and a square matrix with odd sides");
    std::vector<float> out;
    {
        py::gil_scoped_release release;
        out = sim::futures_mfilter(input.data(), input.shape(0), input.shape(1), matrix.data(),
                                   static_cast<int>(matrix.shape(0) / 2));
    }
    py::array_t<float> r({input.shape(0), input.shape(1)});
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

py::dict sim_patches(const IArray& landscape, const py::list& transitions, double prune_factor, int win_rows,
                     int win_cols, std::uint64_t seed) {
    if (landscape.ndim() != 2) throw std::invalid_argument("the landscape must be 2-D");
    const i64 ny = landscape.shape(0), nx = landscape.shape(1), n = ny * nx;
    std::vector<sim::PatchTransition> ts;
    for (auto item : transitions) {
        py::dict d = py::cast<py::dict>(item);
        sim::PatchTransition t;
        t.from = py::cast<int>(d["from"]);
        t.to = py::cast<int>(d["to"]);
        DArray p = py::cast<DArray>(d["probability"]);
        if (p.size() != n) throw std::invalid_argument("a probability map differs in size from the landscape");
        t.probability.assign(p.data(), p.data() + n);
        t.expand = py::cast<std::int64_t>(d["expand"]);
        t.patch = py::cast<std::int64_t>(d["patch"]);
        auto e = py::cast<std::vector<double>>(d["expander"]);
        auto q = py::cast<std::vector<double>>(d["patcher"]);
        t.e_mean = e.at(0);
        t.e_var = e.at(1);
        t.e_iso = e.at(2);
        t.p_mean = q.at(0);
        t.p_var = q.at(1);
        t.p_iso = q.at(2);
        ts.push_back(std::move(t));
    }
    sim::PatchResult r;
    {
        py::gil_scoped_release release;
        r = sim::allocate_patches(ny, nx, landscape.data(), ts, prune_factor, win_rows, win_cols, seed);
    }
    py::dict out;
    py::array_t<int> land({ny, nx});
    std::copy(r.landscape.begin(), r.landscape.end(), land.mutable_data());
    out["landscape"] = land;
    out["expanded"] = to_array(r.expanded);
    out["patched"] = to_array(r.patched);
    out["patches"] = to_array(r.patches);
    return out;
}

py::array_t<double> sim_tmc_table(const DArray& x, const DArray& x1, const DArray& y1, const DArray& w) {
    if (x.ndim() != 2 || x1.ndim() != 2 || y1.ndim() != 2) throw std::invalid_argument("(classes, blocks) counts");
    const int n = static_cast<int>(x.shape(0));
    const i64 blocks = x.shape(1);
    if (n < 2 || n > 64) throw std::invalid_argument("the comparison takes 2 to 64 classes");
    if (x1.shape(0) != n || y1.shape(0) != n || x1.shape(1) != blocks || y1.shape(1) != blocks || w.size() != blocks)
        throw std::invalid_argument("the counts and weights differ in size");
    std::vector<double> t;
    {
        py::gil_scoped_release release;
        t = sim::tmc_table(n, blocks, x.data(), x1.data(), y1.data(), w.data());
    }
    const i64 N = n + 1;
    py::array_t<double> out({N * N, N});
    std::copy(t.begin(), t.end(), out.mutable_data());
    return out;
}

py::dict sim_similarity(const IArray& a, const IArray& b, int window, bool exponential) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("two maps of the same shape");
    const i64 ny = a.shape(0), nx = a.shape(1);
    sim::SimilarityResult r;
    {
        py::gil_scoped_release release;
        r = sim::reciprocal_similarity(ny, nx, a.data(), b.data(), window, exponential);
    }
    py::array_t<float> f({ny, nx}), s({ny, nx});
    std::copy(r.first.begin(), r.first.end(), f.mutable_data());
    std::copy(r.second.begin(), r.second.end(), s.mutable_data());
    py::dict out;
    out["first"] = f;
    out["second"] = s;
    out["first_mean"] = r.first_mean;
    out["second_mean"] = r.second_mean;
    out["first_cells"] = r.first_cells;
    out["second_cells"] = r.second_cells;
    return out;
}

// R's rowSums(x, na.rm): each row added up in R's 80-bit LDOUBLE, column by column.
py::array_t<double> sim_r_rowsums(const DArray& x, bool na_rm) {
    if (x.ndim() != 2) throw std::invalid_argument("rowSums needs a matrix");
    const i64 rows = x.shape(0), cols = x.shape(1);
    const double* v = x.data();
    py::array_t<double> out(rows);
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        for (i64 r = 0; r < rows; ++r)
            o[r] = lsm::sum_of(0, cols, [&](i64 c) { return v[r * cols + c]; }, na_rm);
    }
    return out;
}

py::dict sim_cars(const IArray& landscape, const py::object& fixed, const py::object& region, const DArray& probability,
                  const DArray& weights,
                  const DArray& seeds, const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& allowed,
                  const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& demand, int window,
                  double decay, double step, i64 max_iter, std::uint64_t seed) {
    if (landscape.ndim() != 2) throw std::invalid_argument("the landscape must be 2-D");
    const i64 ny = landscape.shape(0), nx = landscape.shape(1), n = ny * nx;
    if (demand.ndim() != 2) throw std::invalid_argument("the demand must be (regions, classes)");
    const auto nreg = static_cast<int>(demand.shape(0));
    const auto k = static_cast<int>(demand.shape(1));
    if (probability.ndim() != 2 || probability.shape(0) != k || probability.shape(1) != n)
        throw std::invalid_argument("the probabilities must be (classes, cells)");
    if (weights.size() != k || seeds.size() != k || allowed.size() != static_cast<i64>(k) * k)
        throw std::invalid_argument("one weight per class and a square transition matrix");
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> fx;
    const std::uint8_t* fptr = nullptr;
    if (!fixed.is_none()) {
        fx = py::cast<py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>>(fixed);
        if (fx.size() != n) throw std::invalid_argument("the fixed cells differ in size from the landscape");
        fptr = fx.data();
    }
    IArray rg;
    const int* rptr = nullptr;
    if (!region.is_none()) {
        rg = py::cast<IArray>(region);
        if (rg.size() != n) throw std::invalid_argument("the regions differ in size from the landscape");
        rptr = rg.data();
    }
    sim::CarsSettings s;
    s.window = window;
    s.decay = decay;
    s.step = step;
    s.max_iter = max_iter;
    s.seed = seed;
    sim::CarsResult r;
    {
        py::gil_scoped_release release;
        r = sim::cars_run(ny, nx, k, landscape.data(), fptr, rptr, nreg, probability.data(), weights.data(), seeds.data(),
                          allowed.data(),
                          demand.data(), s);
    }
    py::dict out;
    py::array_t<int> land({ny, nx});
    std::copy(r.landscape.begin(), r.landscape.end(), land.mutable_data());
    out["landscape"] = land;
    out["iterations"] = r.iterations;
    out["converged"] = r.converged;
    out["decays"] = r.decays;
    const i64 m = static_cast<i64>(nreg) * k;
    const auto rows = static_cast<i64>(r.trace.size()) / std::max<i64>(m, 1);
    py::array_t<std::int64_t> trace({rows, m});
    std::copy(r.trace.begin(), r.trace.end(), trace.mutable_data());
    out["trace"] = trace;
    out["trace_decays"] = to_array(r.trace_decays);
    py::array_t<double> td({rows - 1, m});
    std::copy(r.trace_d.begin(), r.trace_d.end(), td.mutable_data());
    out["trace_d"] = td;
    return out;
}


// -- decision rules, ordered allocation, demands, Markov chains, WoE ranges ---

using LArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

py::array_t<double> cells_by_classes(const std::vector<double>& v, i64 ncell, int ncode) {
    py::array_t<double> out({static_cast<py::ssize_t>(ncell), static_cast<py::ssize_t>(ncode)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<double> sim_allow_matrix(const DArray& x, const LArray& hist, const DArray& classes, const DArray& cd,
                                     const LArray& rules) {
    const i64 ncell = static_cast<i64>(x.size());
    const int ncode = static_cast<int>(classes.size());
    if (hist.size() != x.size()) throw std::invalid_argument("one history value per cell");
    if (cd.size() != ncode) throw std::invalid_argument("one demand change per class");
    if (rules.size() != static_cast<py::ssize_t>(ncode) * ncode) throw std::invalid_argument("the rules must be (classes, classes)");
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::allow_matrix(x.data(), hist.data(), ncell, classes.data(), ncode, cd.data(), rules.data());
    }
    return cells_by_classes(out, ncell, ncode);
}

py::array_t<double> sim_auto_convert(const DArray& x, const DArray& prob, const DArray& classes, const LArray& mask) {
    const i64 ncell = static_cast<i64>(x.size());
    const int ncode = static_cast<int>(classes.size());
    if (prob.size() != static_cast<py::ssize_t>(ncell) * ncode) throw std::invalid_argument("prob must be (cells, classes)");
    if (mask.size() != x.size()) throw std::invalid_argument("one mask value per cell");
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::auto_convert(x.data(), prob.data(), ncell, classes.data(), ncode, mask.data());
    }
    return to_array(out);
}

py::array_t<double> sim_neighbourhood_allow(const DArray& landscape, const DArray& classes,
                                            const std::vector<std::tuple<double, double, DArray>>& rules) {
    if (landscape.ndim() != 2) throw std::invalid_argument("the landscape must be 2-D");
    std::vector<sim::NeighbourhoodRule> r;
    for (const auto& [cls, threshold, w] : rules) {
        if (w.ndim() != 2) throw std::invalid_argument("a neighbourhood window must be 2-D");
        sim::NeighbourhoodRule x;
        x.cls = cls;
        x.threshold = threshold;
        x.wy = static_cast<int>(w.shape(0));
        x.wx = static_cast<int>(w.shape(1));
        x.weights.assign(w.data(), w.data() + w.size());
        r.push_back(std::move(x));
    }
    const int ncode = static_cast<int>(classes.size());
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::neighbourhood_allow(landscape.data(), landscape.shape(0), landscape.shape(1), classes.data(), ncode, r);
    }
    return cells_by_classes(out, landscape.shape(0) * landscape.shape(1), ncode);
}

py::dict sim_ordered(const DArray& tprob, const DArray& x, const DArray& demand, const DArray& classes,
                     const DArray& order, bool stochastic, nlm::RRng& rng) {
    const i64 ncell = static_cast<i64>(x.size());
    const int ncode = static_cast<int>(classes.size());
    if (tprob.size() != static_cast<py::ssize_t>(ncell) * ncode) throw std::invalid_argument("tprob must be (cells, classes)");
    if (demand.size() != ncode || order.size() != ncode) throw std::invalid_argument("one demand and order per class");
    sim::OrderedResult r;
    {
        py::gil_scoped_release release;
        r = sim::ordered_allocate(tprob.data(), x.data(), ncell, demand.data(), classes.data(), ncode, order.data(),
                                  stochastic, rng);
    }
    py::dict d;
    d["allocation"] = to_array(r.allocation);
    d["iterations"] = r.iterations;
    d["converged"] = r.converged;
    d["short"] = r.short_cells;
    return d;
}

py::array_t<double> sim_approx_extrap(const DArray& x, const DArray& y, const DArray& xout) {
    return to_array(sim::approx_extrap({x.data(), x.data() + x.size()}, {y.data(), y.data() + y.size()},
                                       {xout.data(), xout.data() + xout.size()}));
}

py::array_t<double> sim_round_sum(const DArray& x, i64 ncell) {
    if (x.ndim() != 2) throw std::invalid_argument("x must be 2-D");
    const auto out = sim::round_sum({x.data(), x.data() + x.size()}, x.shape(0), x.shape(1), ncell);
    py::array_t<double> r({x.shape(0), x.shape(1)});
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

py::array_t<double> square(const std::vector<double>& v, int k) {
    py::array_t<double> r({static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(k)});
    std::copy(v.begin(), v.end(), r.mutable_data());
    return r;
}

int square_side(const DArray& m) {
    if (m.ndim() != 2 || m.shape(0) != m.shape(1)) throw std::invalid_argument("the transition matrix must be square");
    return static_cast<int>(m.shape(0));
}

py::tuple sim_markov_from_counts(const DArray& counts) {
    const int k = square_side(counts);
    std::vector<int> kept;
    const auto p = sim::markov_from_counts({counts.data(), counts.data() + counts.size()}, k, kept);
    return py::make_tuple(square(p, static_cast<int>(kept.size())), to_array(kept));
}

py::array_t<double> sim_markov_root(const DArray& p, int n) {
    const int k = square_side(p);
    return square(sim::markov_root({p.data(), p.data() + p.size()}, k, n), k);
}

py::array_t<double> sim_markov_project(const DArray& areas, const DArray& p, int steps) {
    const int k = square_side(p);
    if (steps < 0) throw std::invalid_argument("steps must be >= 0");
    const auto out = sim::markov_project({areas.data(), areas.data() + areas.size()}, {p.data(), p.data() + p.size()},
                                         k, steps);
    py::array_t<double> r({static_cast<py::ssize_t>(steps + 1), static_cast<py::ssize_t>(k)});
    std::copy(out.begin(), out.end(), r.mutable_data());
    return r;
}

py::tuple sim_woe_weights(double nb, double b, double n, double a, bool fix) {
    const auto w = sim::woe_weights(nb, b, n, a, fix);
    return py::make_tuple(w.plus, w.contrast, w.significant);
}

std::vector<int> sim_woe_generalise(const LArray& possible, const LArray& executed, double min_delta,
                                    double max_delta, double tolerance, bool fix) {
    return sim::woe_generalise({possible.data(), possible.data() + possible.size()},
                               {executed.data(), executed.data() + executed.size()}, min_delta, max_delta, tolerance,
                               fix);
}

py::dict sim_observed_patches(const DArray& a, const DArray& b, double from, double to) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("the maps must be 2-D and of the same shape");
    sim::ObservedPatches r;
    {
        py::gil_scoped_release release;
        r = sim::observed_patches(a.data(), b.data(), a.shape(0), a.shape(1), from, to);
    }
    py::dict d;
    d["cells"] = r.cells;
    d["expansion"] = r.expansion;
    d["expansion_patches"] = r.expansion_patches;
    d["expansion_mean"] = r.expansion_mean;
    d["expansion_variance"] = r.expansion_variance;
    d["new_patches"] = r.new_patches;
    d["new_mean"] = r.new_mean;
    d["new_variance"] = r.new_variance;
    return d;
}

py::array_t<double> sim_woe_normalise(const DArray& sums) {
    if (sums.ndim() != 2) throw std::invalid_argument("sums must be (transitions, cells)");
    py::array_t<double> out({sums.shape(0), sums.shape(1)});
    std::copy(sums.data(), sums.data() + sums.size(), out.mutable_data());
    {
        py::gil_scoped_release release;
        sim::woe_normalise(out.mutable_data(), static_cast<int>(sums.shape(0)), sums.shape(1));
    }
    return out;
}

using FArray2 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using NbPart = std::tuple<int, float, FArray2, std::vector<bool>, float, std::vector<std::pair<int, float>>>;

py::array_t<float> sim_clumondo_add_neighbourhood(const FArray2& suit, const IArray& oldco, const IArray& region,
                                                  const std::vector<NbPart>& parts) {
    if (oldco.ndim() != 2 || region.ndim() != 2) throw std::invalid_argument("oldco and region must be 2-D");
    const i64 ny = oldco.shape(0), nx = oldco.shape(1);
    if (region.shape(0) != ny || region.shape(1) != nx) throw std::invalid_argument("region differs from oldco");
    if (suit.size() % (ny * nx)) throw std::invalid_argument("suit must be (systems, cells)");
    const int nsys = static_cast<int>(suit.size() / (ny * nx));
    std::vector<sim::ClumondoNeighbourhood> nb;
    for (const auto& [system, weight, kernel, from, constant, coefs] : parts) {
        if (kernel.ndim() != 2 || kernel.shape(0) != kernel.shape(1) || kernel.shape(0) % 2 == 0)
            throw std::invalid_argument("a kernel must be square with an odd side");
        sim::ClumondoNeighbourhood x;
        x.system = system;
        x.weight = weight;
        x.kernel.assign(kernel.data(), kernel.data() + kernel.size());
        x.radius = static_cast<int>((kernel.shape(0) - 1) / 2);
        x.from.assign(from.begin(), from.end());
        x.constant = constant;
        x.coefficients = coefs;
        nb.push_back(std::move(x));
    }
    py::array_t<float> out(suit.size());
    std::copy(suit.data(), suit.data() + suit.size(), out.mutable_data());
    {
        py::gil_scoped_release release;
        sim::clumondo_add_neighbourhood(out.mutable_data(), nsys, ny, nx, oldco.data(), region.data(), nb);
    }
    out.resize({static_cast<py::ssize_t>(nsys), static_cast<py::ssize_t>(ny * nx)});
    return out;
}

py::array_t<float> sim_clumondo_add_preference(const FArray2& row, const IArray& region, const FArray2& values,
                                               float weight) {
    if (region.size() != row.size() || values.size() != row.size())
        throw std::invalid_argument("the preference differs in size from the suitability");
    py::array_t<float> out(row.size());
    std::copy(row.data(), row.data() + row.size(), out.mutable_data());
    sim::clumondo_add_preference(out.mutable_data(), static_cast<i64>(row.size()), region.data(), values.data(), weight);
    return out;
}

py::tuple forecast_tuple(const sim::MolusceForecast& f) {
    return py::make_tuple(to_array(f.prediction), to_array(f.confidence));
}

py::array_t<double> sim_molusce_categories(const DArray& values, const std::vector<double>& bounds) {
    if (!std::is_sorted(bounds.begin(), bounds.end())) throw std::invalid_argument("the bounds must be sorted");
    return to_array(sim::molusce_categories(values.data(), static_cast<i64>(values.size()), bounds));
}

py::array_t<double> sim_molusce_total(const DArray& factors,
                                      const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& weights) {
    if (factors.ndim() != 2) throw std::invalid_argument("the factors must be (factors, cells)");
    const int nf = static_cast<int>(factors.shape(0));
    const i64 n = factors.shape(1);
    std::vector<std::vector<sim::MolusceFactorWeights>> w(weights.size());
    for (std::size_t c = 0; c < weights.size(); ++c) {
        for (const auto& [cats, ws] : weights[c]) {
            if (cats.size() != ws.size()) throw std::invalid_argument("one weight per category");
            w[c].push_back({cats, ws});
        }
    }
    std::vector<double> out;
    {
        py::gil_scoped_release release;
        out = sim::molusce_total(factors.data(), nf, n, w);
    }
    auto a = to_array(out);
    a.resize({static_cast<py::ssize_t>(weights.size()), static_cast<py::ssize_t>(n)});
    return a;
}

py::tuple sim_molusce_prediction_woe(const DArray& state, const std::vector<double>& classes,
                                     const std::vector<i64>& codes, const DArray& totals) {
    const i64 n = static_cast<i64>(state.size());
    if (totals.size() != static_cast<py::ssize_t>(codes.size()) * n)
        throw std::invalid_argument("the totals must be (codes, cells)");
    sim::MolusceForecast f;
    {
        py::gil_scoped_release release;
        f = sim::molusce_prediction_woe(state.data(), n, classes, codes, totals.data());
    }
    return forecast_tuple(f);
}

py::tuple sim_molusce_prediction_classes(const DArray& state, const DArray& layers, const std::vector<double>& classes) {
    const i64 n = static_cast<i64>(state.size());
    const int m = static_cast<int>(classes.size());
    if (layers.size() != static_cast<py::ssize_t>(m) * n) throw std::invalid_argument("the layers must be (classes, cells)");
    sim::MolusceForecast f;
    {
        py::gil_scoped_release release;
        f = sim::molusce_prediction_classes(state.data(), n, m, layers.data(), classes);
    }
    return forecast_tuple(f);
}

py::tuple sim_molusce_ca_step(const DArray& state, const DArray& prediction, const DArray& confidence,
                              const std::vector<std::tuple<double, double, i64>>& counts, sim::NumpyRandomState& rng) {
    const i64 n = static_cast<i64>(state.size());
    if (prediction.size() != state.size() || confidence.size() != state.size())
        throw std::invalid_argument("the prediction and confidence differ in size from the state");
    std::vector<sim::MolusceTransition> c;
    for (const auto& [f, t, k] : counts) c.push_back({f, t, k});
    sim::MolusceStep r;
    {
        py::gil_scoped_release release;
        r = sim::molusce_ca_step(state.data(), n, prediction.data(), confidence.data(), c, rng);
    }
    std::vector<std::tuple<double, double, i64>> changed;
    for (const auto& x : r.changed) changed.emplace_back(x.from, x.to, x.cells);
    return py::make_tuple(to_array(r.state), changed);
}

}  // namespace

namespace sim {

void register_python(py::module_& m) {
    // numpy's legacy RandomState(seed) and its random_sample, as MOLUSCE draws
    py::class_<sim::NumpyRandomState>(m, "NumpyRandomState")
        .def(py::init<std::uint32_t>(), py::arg("seed"))
        .def("random_sample", [](sim::NumpyRandomState& r, i64 n) {
            std::vector<double> v(static_cast<std::size_t>(n));
            for (auto& x : v) x = r.random_sample();
            return to_array(v);
        }, py::arg("n"));
    m.def("sim_clumondo_add_neighbourhood", &sim_clumondo_add_neighbourhood, py::arg("suit"), py::arg("oldco"),
          py::arg("region"), py::arg("parts"),
          "CLUMondo's neighbourhood functions [(system, weight, kernel, from, constant, [(k, b)])] added to the "
          "suitability (systems, cells), in floats.");
    m.def("sim_clumondo_add_preference", &sim_clumondo_add_preference, py::arg("row"), py::arg("region"),
          py::arg("values"), py::arg("weight"),
          "A location-specific preference (values x weight) added to one system's suitability, in floats.");
    m.def("sim_molusce_categories", &sim_molusce_categories, py::arg("values"), py::arg("bounds"),
          "MOLUSCE's categories of a factor: searchsorted(bounds, v, 'right') + 1, or the values without bounds.");
    m.def("sim_molusce_total", &sim_molusce_total, py::arg("factors"), py::arg("weights"),
          "MOLUSCE's total weight of each transition code: (codes, cells), NaN where a factor has none.");
    m.def("sim_molusce_prediction_woe", &sim_molusce_prediction_woe, py::arg("state"), py::arg("classes"),
          py::arg("codes"), py::arg("totals"), "MolusceWoE.prediction: (predicted class, confidence).");
    m.def("sim_molusce_prediction_classes", &sim_molusce_prediction_classes, py::arg("state"), py::arg("layers"),
          py::arg("classes"), "MolusceCA's prediction from a potential per class: (predicted class, confidence).");
    m.def("sim_molusce_ca_step", &sim_molusce_ca_step, py::arg("state"), py::arg("prediction"),
          py::arg("confidence"), py::arg("counts"), py::arg("rng"),
          "One iteration of MOLUSCE's Simulator: (new state, [(from, to, cells changed)]).");
    m.def("sim_crt_rand", &sim_crt_rand, py::arg("seed"), py::arg("n"),
          "n values of the Microsoft C runtime's rand() after srand(seed) (CLUMondo's generator).");
    m.def("sim_drand48", &sim_drand48, py::arg("seed"), py::arg("n"),
          "n values of G_drand48() after G_srand48(seed) (GRASS GIS, r.futures).");
    m.def("sim_lrand48", &sim_lrand48, py::arg("seed"), py::arg("n"), py::arg("signed") = false,
          "n values of G_lrand48() (or G_mrand48() with signed=True) after G_srand48(seed).");
    m.def("sim_map_classes", &sim_map_classes, py::arg("map"), py::arg("nodata") = py::none(),
          "Sorted distinct values of a map's cells that are not missing (NaN or nodata).");
    m.def("sim_crosstab", &sim_crosstab, py::arg("a"), py::arg("b"), py::arg("classes_a"), py::arg("classes_b"),
          py::arg("nodata_a") = py::none(), py::arg("nodata_b") = py::none(),
          "Cells of each class of a (rows) holding each class of b (columns).");
    m.def("sim_change_count", &sim_change_count, py::arg("maps"), py::arg("nodata") = py::none(),
          "Per cell, the number of consecutive maps that differ (-1 where any map is missing).");
    m.def("sim_woe_buckets", &sim_woe_buckets, py::arg("var"), py::arg("nodata"), py::arg("categorical"),
          py::arg("classes"), py::arg("start"), py::arg("inc"), py::arg("n_buckets"),
          "Bucket (increment or category index) of each cell, -1 where missing.");
    m.def("sim_woe_counts", &sim_woe_counts, py::arg("t0"), py::arg("t1"), py::arg("buckets"), py::arg("n_buckets"),
          py::arg("from_class"), py::arg("final_classes"), py::arg("nodata_t0") = py::none(),
          py::arg("nodata_t1") = py::none(), "Cells of initial class from_class by bucket and final class.");
    m.def("sim_woe_probability", &sim_woe_probability, py::arg("landscape"), py::arg("nodata"), py::arg("from_class"),
          py::arg("buckets"), py::arg("weights"), "Sum of the weights of each cell (NaN outside the class).");
    m.def("sim_glm_logit", &sim_glm_logit, py::arg("x"), py::arg("y"), py::arg("maxit") = 25, py::arg("epsilon") = 1e-8,
          "R's glm(family = binomial) fit (glm.fit), x the model matrix as (p, n).");
    m.def("sim_glm_predict", &sim_glm_predict, py::arg("x"), py::arg("coefficients"), py::arg("pivot"),
          py::arg("rank"), "predict(glm, type = 'response') on the model matrix (p, n).");
    m.def("sim_clues", &sim_clues, py::arg("tprob"), py::arg("landuse"), py::arg("demand"), py::arg("codes"),
          py::arg("jitter"), py::arg("scale"), py::arg("max_iter"), py::arg("max_diff"), py::arg("mean_diff"),
          py::arg("rng"), "CLUE-S allocation (lulcc's allocateclues) with R's generator.");
    m.def("sim_focal_mean", &sim_focal_mean, py::arg("x"), py::arg("w"),
          "raster::focal(x, w, fun = mean, pad = TRUE, na.rm = TRUE).");
    m.def("sim_clumondo_year", &sim_clumondo_year, py::arg("oldco"), py::arg("region"), py::arg("age"),
          py::arg("suit"), py::arg("allow_maps"), py::arg("allowed"), py::arg("lusconv"), py::arg("lusmatrix"),
          py::arg("weight"), py::arg("resistance"), py::arg("demand"), py::arg("dem_elas"), py::arg("itermode"),
          py::arg("mean_dev"), py::arg("max_dev"), py::arg("speed"), py::arg("max_loops") = 20000,
          py::arg("kind") = IArray(0), py::arg("dev_limit") = FArray(0), py::arg("nx") = 0,
          py::arg("patch_radius") = IArray(0), py::arg("patch_threshold") = FArray(0),
          py::arg("patch_min") = IArray(0), py::arg("pair_resistance") = FArray(0),
          py::arg("cap_pairs") = py::array_t<std::uint8_t>(0), py::arg("cap_cells") = py::array_t<std::int64_t>(0),
          "One year of CLUMondo's allocation (iteration, autonomous change, ages).");
    m.def("sim_clumondo_logit", &sim_clumondo_logit, py::arg("region"), py::arg("x"), py::arg("constant"),
          py::arg("coef"), "CLUMondo's logistic suitability in single precision (calc_reg).");
    m.def("sim_clumondo_fraction", &sim_clumondo_fraction, py::arg("oldco"), py::arg("region"), py::arg("system"),
          py::arg("kernel"), "CLUMondo's neighbourhood share of a land system (calc_neigh).");
    m.def("sim_clumondo_random_age", &sim_clumondo_random_age, py::arg("ncell"), py::arg("max_age"),
          py::arg("seed") = 1, "CLUMondo's random initial ages: rand() % max_age + 1 per cell.");
    m.def("sim_futures_run", &sim_futures_run, py::arg("developed"), py::arg("region"), py::arg("pot_region"),
          py::arg("devpressure"), py::arg("aggregated"), py::arg("weight"), py::arg("zone"), py::arg("nregions"),
          py::arg("npot"), py::arg("intercept"), py::arg("devpressure_coef"), py::arg("neighborhood"),
          py::arg("approach"), py::arg("gamma"), py::arg("scaling"), py::arg("patch_sizes"), py::arg("single_column"),
          py::arg("compactness_mean"), py::arg("compactness_range"), py::arg("num_neighbors"),
          py::arg("seed_probability"), py::arg("incentive"), py::arg("user_effects"), py::arg("zone_ids"),
          py::arg("zone_regions"), py::arg("zone_effects"), py::arg("stringency"), py::arg("demand"),
          py::arg("nsteps"), py::arg("seed"), py::arg("climate") = py::none(),
          "FUTURES (r.futures.simulation) patch-growing simulation, with its flood submodel.");
    m.def("sim_patches", &sim_patches, py::arg("landscape"), py::arg("transitions"), py::arg("prune_factor") = 10.0,
          py::arg("win_rows") = 3, py::arg("win_cols") = 3, py::arg("seed") = 1,
          "Expander and Patcher (Soares-Filho et al. 2002): allocation of the changes of each transition.");
    m.def("sim_cars", &sim_cars, py::arg("landscape"), py::arg("fixed"), py::arg("region"), py::arg("probability"),
          py::arg("weights"),
          py::arg("seeds"), py::arg("allowed"), py::arg("demand"), py::arg("window") = 3, py::arg("decay") = 0.9,
          py::arg("step") = 500.0, py::arg("max_iter") = 5000, py::arg("seed") = 1,
          "CARS, PLUS's cellular automaton of multi-type random patch seeds (Liang et al. 2021).");
    m.def("sim_tmc_table", &sim_tmc_table, py::arg("x"), py::arg("x1"), py::arg("y1"), py::arg("w"),
          "lulcc's ThreeMapComparison table (Rules 1-4) of one resolution from the class counts per block.");
    m.def("sim_similarity", &sim_similarity, py::arg("a"), py::arg("b"), py::arg("window"),
          py::arg("exponential") = true,
          "Dinamica EGO's CalcReciprocalSimilarityMap: fuzzy similarity of two categorical maps (-1: no class).");
    m.def("sim_r_rowsums", &sim_r_rowsums, py::arg("x"), py::arg("na_rm") = false,
          "R's rowSums (80-bit accumulation; na_rm skips NaN).");
    m.def("sim_futures_mfilter", &sim_futures_mfilter, py::arg("input"), py::arg("matrix"),
          "r.mfilter as r.futures.devpressure runs it (edges keep their values, nulls give null).");
    m.def("sim_allow_matrix", &sim_allow_matrix, py::arg("x"), py::arg("hist"), py::arg("classes"), py::arg("cd"),
          py::arg("rules"), "lulcc's allow(): 1 where a conversion is allowed, NaN where not; (cells, classes).");
    m.def("sim_auto_convert", &sim_auto_convert, py::arg("x"), py::arg("prob"), py::arg("classes"), py::arg("mask"),
          "lulcc's autoconvert: the class each cell is forced to take (NaN: none).");
    m.def("sim_neighbourhood_allow", &sim_neighbourhood_allow, py::arg("landscape"), py::arg("classes"),
          py::arg("rules"), "lulcc's neighbourhood rules [(class, threshold, window)]: 1 or NaN; (cells, classes).");
    m.def("sim_ordered", &sim_ordered, py::arg("tprob"), py::arg("x"), py::arg("demand"), py::arg("classes"),
          py::arg("order"), py::arg("stochastic"), py::arg("rng"),
          "The ordered allocation (lulcc's OrderedModel) with R's generator.");
    m.def("sim_approx_extrap", &sim_approx_extrap, py::arg("x"), py::arg("y"), py::arg("xout"),
          "Hmisc::approxExtrap: linear interpolation, the end segments extrapolated.");
    m.def("sim_round_sum", &sim_round_sum, py::arg("x"), py::arg("ncell"),
          "lulcc's roundSum: each row to integers that add up to ncell.");
    m.def("sim_markov_from_counts", &sim_markov_from_counts, py::arg("counts"),
          "Transition probabilities from a matrix of cell counts: (matrix, indices of the classes kept).");
    m.def("sim_markov_root", &sim_markov_root, py::arg("p"), py::arg("n"),
          "The principal n-th root of a transition matrix.");
    m.def("sim_markov_project", &sim_markov_project, py::arg("areas"), py::arg("p"), py::arg("steps"),
          "Areas after each step: (steps + 1, classes).");
    m.def("sim_woe_weights", &sim_woe_weights, py::arg("nb"), py::arg("b"), py::arg("n"), py::arg("a"),
          py::arg("fix") = false, "W+, contrast and significance of a range of weights of evidence.");
    m.def("sim_woe_generalise", &sim_woe_generalise, py::arg("possible"), py::arg("executed"), py::arg("min_delta"),
          py::arg("max_delta"), py::arg("tolerance"), py::arg("fix") = false,
          "The bucket indices ending each range of a continuous variable (Dinamica EGO's line generalisation).");
    m.def("sim_observed_patches", &sim_observed_patches, py::arg("a"), py::arg("b"), py::arg("from_class"),
          py::arg("to_class"), "The expansion and new patches of an observed transition (8 neighbours).");
    m.def("sim_woe_normalise", &sim_woe_normalise, py::arg("sums"),
          "Probabilities exp(S_j) / (1 + sum exp(S_i)) of transitions from one class, (transitions, cells).");
}

}  // namespace sim
