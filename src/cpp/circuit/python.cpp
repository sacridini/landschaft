// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the circuit module. They go into landschaft._core and,
// on x86-64, also into landschaft._circuit_avx2: the same code compiled for
// AVX2 and FMA in a module of its own (so that the two builds of Eigen's
// templates never meet in one binary), loaded only on CPUs that have them.

#include "circuit/python.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/circuitscape.hpp"
#include "circuit/grid.hpp"
#include "circuit/omniscape.hpp"

namespace py = pybind11;

namespace {

using circuit::i64;

template <class T>
using CArray = py::array_t<T, py::array::c_style>;

// Hand a std::vector over to numpy without copying it.
py::array_t<double> to_array(std::vector<double>&& v, i64 nx, i64 ny) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({ny, nx}, holder->data(), owner);
}

// Runs `f` with the GIL released and, if n_threads > 0, with that many
// OpenMP threads for parallel regions started by this thread.
template <class F>
auto run_parallel(int n_threads, F&& f) {
    py::gil_scoped_release release;
#if defined(_OPENMP)
    const int previous = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(n_threads);
    struct Restore {
        int n;
        ~Restore() { omp_set_num_threads(n); }
    } restore{previous};
#else
    (void)n_threads;
#endif
    return f();
}

py::dict circuit_advanced(const CArray<double>& conductance, const CArray<double>& source,
                          const CArray<double>& ground, bool four_neighbors, int n_threads) {
    if (conductance.ndim() != 2 || source.ndim() != 2 || ground.ndim() != 2 ||
        source.shape(0) != conductance.shape(0) || source.shape(1) != conductance.shape(1) ||
        ground.shape(0) != conductance.shape(0) || ground.shape(1) != conductance.shape(1))
        throw py::value_error("expected conductance, source and ground rasters of the same shape");
    const i64 ny = conductance.shape(0), nx = conductance.shape(1);
    circuit::AdvancedResult r = run_parallel(n_threads, [&] {
        return circuit::advanced(conductance.data(), source.data(), ground.data(), nx, ny, four_neighbors, n_threads);
    });
    py::dict out;
    out["current"] = to_array(std::move(r.current), nx, ny);
    out["voltage"] = to_array(std::move(r.voltage), nx, ny);
    return out;
}

// A condition raster (None: no condition) of the resistance's shape.
const double* condition_data(const py::object& a, i64 nx, i64 ny, const char* name) {
    if (a.is_none()) return nullptr;
    const auto arr = a.cast<CArray<double>>();
    if (arr.ndim() != 2 || arr.shape(0) != ny || arr.shape(1) != nx)
        throw py::value_error(std::string(name) + " must have the resistance's shape");
    return arr.data();
}

py::dict circuit_omniscape(const CArray<double>& resistance, const CArray<double>& source, int radius, int buffer,
                           int block_size, bool four_neighbors, bool resistance_is_conductance,
                           bool correct_artifacts, bool flow_potential, int n_threads, const py::dict& conditions,
                           double memory_limit) {
    if (resistance.ndim() != 2 || source.ndim() != 2 || source.shape(0) != resistance.shape(0) ||
        source.shape(1) != resistance.shape(1))
        throw py::value_error("expected resistance and source rasters of the same shape");
    const i64 ny = resistance.shape(0), nx = resistance.shape(1);
    // Keep the condition arrays alive (c-contiguous float64 copies if needed).
    std::vector<CArray<double>> keep;
    auto condition = [&](int k) {
        circuit::OmniscapeCondition c;
        const std::string key = "condition" + std::to_string(k);
        if (!conditions.contains(key.c_str()) || conditions[key.c_str()].is_none()) return c;
        keep.push_back(conditions[key.c_str()].cast<CArray<double>>());
        c.present = condition_data(keep.back(), nx, ny, key.c_str());
        const std::string fut = key + "_future";
        if (conditions.contains(fut.c_str()) && !conditions[fut.c_str()].is_none()) {
            keep.push_back(conditions[fut.c_str()].cast<CArray<double>>());
            c.future = condition_data(keep.back(), nx, ny, fut.c_str());
        }
        const std::string cmp = conditions[("comparison" + std::to_string(k)).c_str()].cast<std::string>();
        if (cmp != "within" && cmp != "equal") throw py::value_error("comparisons must be 'within' or 'equal'");
        c.equal = cmp == "equal";
        c.lower = conditions[(key + "_lower").c_str()].cast<double>();
        c.upper = conditions[(key + "_upper").c_str()].cast<double>();
        return c;
    };
    circuit::OmniscapeParams p;
    p.radius = radius;
    p.buffer = buffer;
    p.block_size = block_size;
    p.four_neighbors = four_neighbors;
    p.resistance_is_conductance = resistance_is_conductance;
    p.correct_artifacts = correct_artifacts;
    p.flow_potential = flow_potential;
    p.condition1 = condition(1);
    p.condition2 = condition(2);
    p.memory_limit = memory_limit;
    circuit::OmniscapeResult r =
        run_parallel(n_threads, [&] { return circuit::omniscape(resistance.data(), source.data(), nx, ny, p, n_threads); });
    py::dict out;
    out["current"] = to_array(std::move(r.current), nx, ny);
    if (flow_potential) out["flow_potential"] = to_array(std::move(r.flow_potential), nx, ny);
    out["n_targets"] = r.n_targets;
    out["threads"] = r.threads;
    return out;
}

py::array_t<double> circuit_omniscape_correction(int radius, int buffer, int block_size, bool four_neighbors,
                                                 i64 nx, i64 ny) {
    circuit::OmniscapeParams p;
    p.radius = radius;
    p.buffer = buffer;
    p.block_size = block_size;
    p.four_neighbors = four_neighbors;
    const i64 w = 2 * (static_cast<i64>(radius) + buffer) + 1;
    std::vector<double> c;
    {
        py::gil_scoped_release release;
        c = circuit::omniscape_correction(p, nx, ny);
    }
    return to_array(std::move(c), w, w);
}

circuit::ConflictPolicy conflict_policy(const std::string& s) {
    if (s == "keepall") return circuit::ConflictPolicy::KeepAll;
    if (s == "rmvsrc") return circuit::ConflictPolicy::RemoveSource;
    if (s == "rmvgnd") return circuit::ConflictPolicy::RemoveGround;
    if (s == "rmvall") return circuit::ConflictPolicy::RemoveAll;
    throw py::value_error("remove_src_or_gnd must be keepall, rmvsrc, rmvgnd or rmvall");
}

template <class T>
py::array_t<T> vector_to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* q) { delete static_cast<std::vector<T>*>(q); });
    return py::array_t<T>(static_cast<py::ssize_t>(holder->size()), holder->data(), owner);
}

py::dict circuitscape_advanced(const CArray<double>& gmap, const py::object& polymap, const CArray<double>& source,
                               const CArray<double>& ground, bool avg_resistances, bool four_neighbors,
                               const std::string& policy, int n_threads) {
    if (gmap.ndim() != 2 || source.ndim() != 2 || ground.ndim() != 2 || source.shape(0) != gmap.shape(0) ||
        source.shape(1) != gmap.shape(1) || ground.shape(0) != gmap.shape(0) || ground.shape(1) != gmap.shape(1))
        throw py::value_error("expected habitat, source and ground rasters of the same shape");
    const i64 ny = gmap.shape(0), nx = gmap.shape(1);
    CArray<std::int64_t> poly;
    const std::int64_t* pm = nullptr;
    if (!polymap.is_none()) {
        poly = polymap.cast<CArray<std::int64_t>>();
        if (poly.ndim() != 2 || poly.shape(0) != ny || poly.shape(1) != nx)
            throw py::value_error("the polygon raster must have the habitat's shape");
        pm = poly.data();
    }
    const circuit::ConflictPolicy pol = conflict_policy(policy);
    circuit::RasterGraph g;
    circuit::NodeSolution sol;
    run_parallel(n_threads, [&] {
        g = circuit::raster_graph(gmap.data(), pm, nx, ny, avg_resistances, four_neighbors);
        sol = circuit::raster_advanced(g, circuit::node_sums(g, source.data()), circuit::node_sums(g, ground.data()),
                                       pol, n_threads);
        return 0;
    });
    py::dict out;
    std::vector<std::int32_t> nodemap = g.nodemap;
    auto* holder = new std::vector<std::int32_t>(std::move(nodemap));
    py::capsule owner(holder, [](void* q) { delete static_cast<std::vector<std::int32_t>*>(q); });
    out["nodemap"] = py::array_t<std::int32_t>({ny, nx}, holder->data(), owner);
    out["voltage"] = vector_to_array(std::move(sol.voltage));
    out["current"] = vector_to_array(std::move(sol.current));
    out["solved"] = sol.solved;
    out["n_components"] = g.n_comp;
    return out;
}

const std::int64_t* polymap_data(const py::object& polymap, CArray<std::int64_t>& keep, i64 nx, i64 ny) {
    if (polymap.is_none()) return nullptr;
    keep = polymap.cast<CArray<std::int64_t>>();
    if (keep.ndim() != 2 || keep.shape(0) != ny || keep.shape(1) != nx)
        throw py::value_error("the polygon raster must have the habitat's shape");
    return keep.data();
}

py::dict circuitscape_pairwise(const CArray<double>& gmap, const py::object& polymap,
                               const CArray<std::int64_t>& point_cells, const CArray<std::uint8_t>& excluded,
                               bool avg_resistances, bool four_neighbors, const py::dict& options,
                               const py::object& emit, int n_threads) {
    if (gmap.ndim() != 2) throw py::value_error("expected a 2-D habitat raster");
    const i64 ny = gmap.shape(0), nx = gmap.shape(1);
    const std::int64_t np = point_cells.size();
    CArray<std::int64_t> keep;
    const std::int64_t* pm = polymap_data(polymap, keep, nx, ny);
    circuit::PairwiseOptions o;
    auto flag = [&](const char* k) { return options.contains(k) && options[k].cast<bool>(); };
    o.volt_maps = flag("volt_maps");
    o.cur_maps = flag("cur_maps");
    o.cum = flag("cum");
    o.max = flag("max");
    o.focal_zero = flag("focal_zero");
    o.null_currents = flag("null_currents");
    o.null_voltages = flag("null_voltages");
    o.log_transform = flag("log_transform");
    std::vector<std::uint8_t> ex(excluded.data(), excluded.data() + excluded.size());
    circuit::RasterGraph g;
    circuit::PairwiseResult r;
    std::vector<std::int32_t> nodes(static_cast<std::size_t>(np));
    // Maps go to Python one at a time, with the GIL.
    std::function<void(circuit::PairMaps&&)> callback = [&](circuit::PairMaps&& m) {
        py::gil_scoped_acquire gil;
        if (emit.is_none()) return;
        py::object v = py::none(), c = py::none();
        if (!m.voltage.empty()) v = to_array(std::move(m.voltage), nx, ny);
        if (!m.current.empty()) c = to_array(std::move(m.current), nx, ny);
        emit(m.ci, m.cj, v, c);
    };
    const bool regions = options.contains("region_ids") && !options["region_ids"].is_none();
    std::vector<std::int64_t> region_ids;
    std::vector<circuit::i64> region_cells;
    if (regions) {
        const auto a = options["region_ids"].cast<CArray<std::int64_t>>();
        region_ids.assign(a.data(), a.data() + a.size());
        region_cells.assign(point_cells.data(), point_cells.data() + point_cells.size());
    }
    run_parallel(n_threads, [&] {
        if (regions) {
            r = circuit::raster_pairwise_regions(gmap.data(), pm, nx, ny, avg_resistances, four_neighbors, region_cells,
                                                 region_ids, ex, o, callback, n_threads);
            return 0;
        }
        g = circuit::raster_graph(gmap.data(), pm, nx, ny, avg_resistances, four_neighbors);
        for (std::int64_t i = 0; i < np; ++i) {
            const std::int64_t c = point_cells.data()[i];
            nodes[static_cast<std::size_t>(i)] = c >= 0 && c < nx * ny ? g.nodemap[static_cast<std::size_t>(c)] : -1;
        }
        r = circuit::raster_pairwise(g, gmap.data(), nodes, ex, o, callback, n_threads);
        return 0;
    });
    const std::int64_t nr = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(r.resistances.size()))));
    py::dict out;
    auto* holder = new std::vector<double>(std::move(r.resistances));
    py::capsule owner(holder, [](void* q) { delete static_cast<std::vector<double>*>(q); });
    out["resistances"] = py::array_t<double>({nr, nr}, holder->data(), owner);
    if (o.cum) out["cum"] = to_array(std::move(r.cum), nx, ny);
    if (o.max) out["max"] = to_array(std::move(r.max), nx, ny);
    out["nodes"] = vector_to_array(std::move(nodes));
    return out;
}

circuit::RasterGraph network_from(const CArray<std::int32_t>& a, const CArray<std::int32_t>& b,
                                  const CArray<double>& w, std::int32_t m) {
    if (a.size() != b.size() || a.size() != w.size()) throw py::value_error("edge arrays of different lengths");
    return circuit::network_graph(std::vector<std::int32_t>(a.data(), a.data() + a.size()),
                                  std::vector<std::int32_t>(b.data(), b.data() + b.size()),
                                  std::vector<double>(w.data(), w.data() + w.size()), m);
}

// Node of the adjacency list that holds each entry.
std::vector<std::int32_t> entry_owner(const circuit::RasterGraph& g) {
    std::vector<std::int32_t> owner(g.adj.size());
    for (std::int32_t c = 0; c < g.n_nodes; ++c)
        for (std::int64_t e = g.ptr[c]; e < g.ptr[c + 1]; ++e) owner[static_cast<std::size_t>(e)] = c;
    return owner;
}

// Branch currents as three arrays: node r, node c (r < c), current.
py::tuple branch_arrays(const circuit::RasterGraph& g, const std::vector<std::int32_t>& owner,
                        const std::vector<std::pair<std::int64_t, double>>& branch) {
    const auto n = static_cast<py::ssize_t>(branch.size());
    py::array_t<std::int32_t> r(n), c(n);
    py::array_t<double> v(n);
    for (py::ssize_t k = 0; k < n; ++k) {
        const std::int64_t e = branch[static_cast<std::size_t>(k)].first;
        r.mutable_data()[k] = g.adj[static_cast<std::size_t>(e)];
        c.mutable_data()[k] = owner[static_cast<std::size_t>(e)];
        v.mutable_data()[k] = branch[static_cast<std::size_t>(k)].second;
    }
    return py::make_tuple(r, c, v);
}

// Circuitscape's advanced mode on a network: sources and grounds by node
// (grounds as conductances, +Inf direct). Voltages by node, and the node and
// branch currents of the whole graph (as write_cur_maps computes them).
py::dict network_advanced(const CArray<std::int32_t>& a, const CArray<std::int32_t>& b, const CArray<double>& w,
                          std::int32_t m, const CArray<double>& source, const CArray<double>& ground,
                          const std::string& policy, int n_threads) {
    if (source.size() != m || ground.size() != m) throw py::value_error("sources and grounds must have one value per node");
    const circuit::ConflictPolicy pol = conflict_policy(policy);
    circuit::RasterGraph g;
    circuit::NodeSolution sol;
    std::vector<double> current;
    std::vector<std::pair<std::int64_t, double>> branch;
    run_parallel(n_threads, [&] {
        g = network_from(a, b, w, m);
        std::vector<double> src(source.data(), source.data() + m), gnd(ground.data(), ground.data() + m);
        std::vector<double> finite(static_cast<std::size_t>(m), 0.0);
        bool any_finite = false;
        for (std::int32_t v = 0; v < m; ++v) {
            finite[v] = gnd[v] < std::numeric_limits<double>::infinity() ? gnd[v] : 0.0;
            any_finite = any_finite || finite[v] != 0;
        }
        sol = circuit::raster_advanced(g, std::move(src), std::move(gnd), pol, n_threads);
        std::vector<std::int32_t> all(static_cast<std::size_t>(m));
        for (std::int32_t v = 0; v < m; ++v) all[v] = v;
        current.assign(static_cast<std::size_t>(m), 0.0);
        circuit::network_currents(g, all.data(), m, sol.voltage.data(), any_finite ? finite.data() : nullptr,
                                  current.data(), branch);
        return 0;
    });
    py::dict out;
    out["voltage"] = vector_to_array(std::move(sol.voltage));
    out["current"] = vector_to_array(std::move(current));
    out["branch"] = branch_arrays(g, entry_owner(g), branch);
    out["solved"] = sol.solved;
    out["n_components"] = g.n_comp;
    return out;
}

// Circuitscape's pairwise mode on a network: `points` are nodes (-1: not in
// the graph). With map options every pair calls emit(ci, cj, nodes, voltage
// or None, current, (branch r, c, current)) and the summed node currents and
// branch currents (per edge r < c: arrays r, c, value) are returned.
py::dict network_pairwise(const CArray<std::int32_t>& a, const CArray<std::int32_t>& b, const CArray<double>& w,
                          std::int32_t m, const CArray<std::int32_t>& points, const CArray<std::uint8_t>& excluded,
                          const py::dict& options, const py::object& emit, int n_threads) {
    circuit::PairwiseOptions o;
    auto flag = [&](const char* k) { return options.contains(k) && options[k].cast<bool>(); };
    o.volt_maps = flag("volt_maps");
    o.cur_maps = flag("cur_maps");
    o.cum = flag("cum");
    o.max = flag("max");
    o.network = true;
    std::vector<std::uint8_t> ex(excluded.data(), excluded.data() + excluded.size());
    std::vector<std::int32_t> pts(points.data(), points.data() + points.size());
    for (std::int32_t& v : pts)
        if (v < 0 || v >= m) v = -1;
    circuit::RasterGraph g;
    std::vector<std::int32_t> owner;
    circuit::PairwiseResult r;
    std::function<void(circuit::PairMaps&&)> callback = [&](circuit::PairMaps&& pm) {
        py::gil_scoped_acquire gil;
        if (emit.is_none()) return;
        py::object v = py::none();
        if (!pm.voltage.empty()) v = vector_to_array(std::move(pm.voltage));
        emit(pm.ci, pm.cj, vector_to_array(std::move(pm.nodes)), v, vector_to_array(std::move(pm.current)),
             branch_arrays(g, owner, pm.branch));
    };
    run_parallel(n_threads, [&] {
        g = network_from(a, b, w, m);
        owner = entry_owner(g);
        r = circuit::raster_pairwise(g, nullptr, pts, ex, o, callback, n_threads);
        return 0;
    });
    const std::int64_t nr = static_cast<std::int64_t>(pts.size());
    py::dict out;
    auto* holder = new std::vector<double>(std::move(r.resistances));
    py::capsule own(holder, [](void* q) { delete static_cast<std::vector<double>*>(q); });
    out["resistances"] = py::array_t<double>({nr, nr}, holder->data(), own);
    if (!r.cum.empty()) {
        out["cum"] = vector_to_array(std::move(r.cum));
        std::vector<std::pair<std::int64_t, double>> cb;
        for (std::int32_t c = 0; c < g.n_nodes; ++c)
            for (std::int64_t e = g.ptr[c]; e < g.ptr[c + 1]; ++e)
                if (g.adj[static_cast<std::size_t>(e)] < c) cb.emplace_back(e, r.cum_branch[static_cast<std::size_t>(e)]);
        out["cum_branch"] = branch_arrays(g, owner, cb);
    }
    out["n_components"] = g.n_comp;
    return out;
}

// A batch of advanced solves on one graph (the one-to-all driver): every
// problem gives its source and ground cells and values (summed per node in
// column-major cell order) and the cell of its check node (-1: every
// component). Problems run in parallel, one thread each.
py::dict circuitscape_multi_advanced(const CArray<double>& gmap, const py::object& polymap, bool avg_resistances,
                                     bool four_neighbors, const py::list& problems, const std::string& policy,
                                     int n_threads) {
    if (gmap.ndim() != 2) throw py::value_error("expected a 2-D habitat raster");
    const i64 ny = gmap.shape(0), nx = gmap.shape(1);
    CArray<std::int64_t> keep;
    const std::int64_t* pm = polymap_data(polymap, keep, nx, ny);
    const circuit::ConflictPolicy pol = conflict_policy(policy);
    struct Problem {
        std::vector<std::int64_t> src_cells, gnd_cells;
        std::vector<double> src_vals, gnd_vals;
        std::int64_t check = -1;
    };
    std::vector<Problem> probs;
    for (const py::handle& h : problems) {
        const py::dict d = h.cast<py::dict>();
        Problem q;
        const auto sc = d["src_cells"].cast<CArray<std::int64_t>>();
        const auto sv = d["src_vals"].cast<CArray<double>>();
        const auto gc = d["gnd_cells"].cast<CArray<std::int64_t>>();
        const auto gv = d["gnd_vals"].cast<CArray<double>>();
        q.src_cells.assign(sc.data(), sc.data() + sc.size());
        q.src_vals.assign(sv.data(), sv.data() + sv.size());
        q.gnd_cells.assign(gc.data(), gc.data() + gc.size());
        q.gnd_vals.assign(gv.data(), gv.data() + gv.size());
        q.check = d["check_cell"].cast<std::int64_t>();
        probs.push_back(std::move(q));
    }
    circuit::RasterGraph g;
    std::vector<circuit::NodeSolution> sols(probs.size());
    run_parallel(n_threads, [&] {
        g = circuit::raster_graph(gmap.data(), pm, nx, ny, avg_resistances, four_neighbors);
        // Per node sums in column-major cell order.
        auto sums = [&](const std::vector<std::int64_t>& cells, const std::vector<double>& vals) {
            std::vector<std::size_t> order(cells.size());
            for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                const std::int64_t ca = cells[a], cb = cells[b];
                return std::make_pair(ca % nx, ca / nx) < std::make_pair(cb % nx, cb / nx);
            });
            std::vector<double> s(static_cast<std::size_t>(g.n_nodes), 0.0);
            for (std::size_t k : order) {
                const std::int32_t v = g.nodemap[static_cast<std::size_t>(cells[k])];
                if (v >= 0 && vals[k] != 0) s[static_cast<std::size_t>(v)] += vals[k];
            }
            return s;
        };
        std::vector<std::vector<double>> srcs, gnds;
        std::vector<std::int32_t> only;
        for (const Problem& q : probs) {
            srcs.push_back(sums(q.src_cells, q.src_vals));
            gnds.push_back(sums(q.gnd_cells, q.gnd_vals));
            std::int32_t o = -1;
            if (q.check >= 0) {
                o = g.nodemap[static_cast<std::size_t>(q.check)];
                if (o < 0) o = -2;
            }
            only.push_back(o);
        }
        sols = circuit::raster_advanced_many(g, srcs, gnds, only, pol, n_threads);
        return 0;
    });
    py::dict out;
    std::vector<std::int32_t> nodemap = g.nodemap;
    auto* holder = new std::vector<std::int32_t>(std::move(nodemap));
    py::capsule owner(holder, [](void* q) { delete static_cast<std::vector<std::int32_t>*>(q); });
    out["nodemap"] = py::array_t<std::int32_t>({ny, nx}, holder->data(), owner);
    py::list results;
    for (auto& sol : sols) {
        py::dict r;
        r["voltage"] = vector_to_array(std::move(sol.voltage));
        r["current"] = vector_to_array(std::move(sol.current));
        r["active"] = vector_to_array(std::move(sol.active));
        r["solved"] = sol.solved;
        results.append(r);
    }
    out["results"] = results;
    return out;
}

// The body of an ESRI ASCII grid: every value in its shortest form that
// reads back to the same double (NaN written as -9999), rows in parallel.
py::bytes ascii_grid_rows(const CArray<double>& a) {
    if (a.ndim() != 2) throw py::value_error("expected a 2-D array");
    const i64 ny = a.shape(0), nx = a.shape(1);
    const double* d = a.data();
    std::vector<std::string> rows(static_cast<std::size_t>(ny));
    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long long y = 0; y < static_cast<long long>(ny); ++y) {
            std::string& out = rows[static_cast<std::size_t>(y)];
            out.reserve(static_cast<std::size_t>(nx) * 12);
            char buf[64];
            for (i64 x = 0; x < nx; ++x) {
                double v = d[y * nx + x];
                if (std::isnan(v)) v = -9999.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
                const auto r = std::to_chars(buf, buf + sizeof buf, v);
                const std::size_t len = static_cast<std::size_t>(r.ptr - buf);
#else
                const int len = std::snprintf(buf, sizeof buf, "%.17g", v);
#endif
                if (x) out.push_back(' ');
                out.append(buf, static_cast<std::size_t>(len));
            }
            out.push_back('\n');
        }
    }
    std::size_t total = 0;
    for (const auto& r : rows) total += r.size();
    std::string all;
    all.reserve(total);
    for (const auto& r : rows) all += r;
    return py::bytes(all);
}

}  // namespace

namespace circuit {

void register_python(py::module_& m) {
    m.def("circuit_advanced", &circuit_advanced, py::arg("conductance"), py::arg("source"), py::arg("ground"),
          py::arg("four_neighbors") = false, py::arg("n_threads") = 0,
          "Circuitscape's advanced mode on a raster: node currents and voltages.");
    m.def("circuit_omniscape", &circuit_omniscape, py::arg("resistance"), py::arg("source"), py::arg("radius"),
          py::arg("buffer"), py::arg("block_size"), py::arg("four_neighbors"), py::arg("resistance_is_conductance"),
          py::arg("correct_artifacts"), py::arg("flow_potential"), py::arg("n_threads") = 0,
          py::arg("conditions") = py::dict(), py::arg("memory_limit") = 0.0,
          "Omniscape's cumulative current (and flow potential) over moving windows. `conditions`: "
          "condition1/2 (arrays), condition1/2_future, comparison1/2, condition1/2_lower/upper.");
    m.def("circuit_omniscape_correction", &circuit_omniscape_correction, py::arg("radius"), py::arg("buffer"),
          py::arg("block_size"), py::arg("four_neighbors"), py::arg("nx"), py::arg("ny"),
          "Omniscape's block-artifact correction factors.");
    m.def("circuitscape_advanced", &circuitscape_advanced, py::arg("gmap"), py::arg("polymap"), py::arg("source"),
          py::arg("ground"), py::arg("avg_resistances"), py::arg("four_neighbors"), py::arg("policy"),
          py::arg("n_threads") = 0,
          "Circuitscape's advanced mode on a raster: node map, and voltages and currents per node.");
    m.def("circuitscape_pairwise", &circuitscape_pairwise, py::arg("gmap"), py::arg("polymap"),
          py::arg("point_cells"), py::arg("excluded"), py::arg("avg_resistances"), py::arg("four_neighbors"),
          py::arg("options"), py::arg("emit") = py::none(), py::arg("n_threads") = 0,
          "Circuitscape's pairwise mode on a raster: resistances and current maps.");
    m.def("network_advanced", &network_advanced, py::arg("a"), py::arg("b"), py::arg("w"), py::arg("n_nodes"),
          py::arg("source"), py::arg("ground"), py::arg("policy"), py::arg("n_threads") = 0,
          "Circuitscape's advanced mode on a network (0-based nodes): voltages, node and branch currents.");
    m.def("network_pairwise", &network_pairwise, py::arg("a"), py::arg("b"), py::arg("w"), py::arg("n_nodes"),
          py::arg("points"), py::arg("excluded"), py::arg("options"), py::arg("emit") = py::none(),
          py::arg("n_threads") = 0, "Circuitscape's pairwise mode on a network: resistances and currents.");
    m.def("ascii_grid_rows", &ascii_grid_rows, py::arg("values"),
          "Rows of an ASCII grid: shortest round-trip values, NaN as -9999.");
    m.def("circuitscape_multi_advanced", &circuitscape_multi_advanced, py::arg("gmap"), py::arg("polymap"),
          py::arg("avg_resistances"), py::arg("four_neighbors"), py::arg("problems"), py::arg("policy"),
          py::arg("n_threads") = 0, "Advanced-mode solves of several source/ground sets on one raster graph.");
}

}  // namespace circuit

#if defined(LANDSCHAFT_CIRCUIT_AVX2_MODULE)
PYBIND11_MODULE(_circuit_avx2, m) {
    m.doc() = "landschaft's circuit module built for AVX2 and FMA";
    circuit::register_python(m);
}
#endif
