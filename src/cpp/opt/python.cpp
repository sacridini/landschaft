// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the MILP prioritization (landschaft._core.opt_*).

#include "opt/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "opt/actions.hpp"
#include "opt/cover.hpp"
#include "opt/cut.hpp"
#include "opt/features.hpp"
#include "opt/marxan.hpp"
#include "opt/maxflow.hpp"
#include "opt/milp.hpp"
#include "opt/wide.hpp"

namespace py = pybind11;

namespace {

using opt::i64;
namespace actions = opt::actions;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Bytes = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

template <class T, class A>
std::vector<T> vec(const A& a) {
    if (a.ndim() != 1) throw std::invalid_argument("expected a 1-d array");
    return std::vector<T>(a.data(), a.data() + a.size());
}

template <class T>
py::array_t<T> to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

opt::Model model_of(const Ints& start, const Ints& index, const Doubles& value, const Doubles& obj,
                    const Doubles& lb, const Doubles& ub, const Doubles& row_lo, const Doubles& row_hi,
                    const Bytes& integer, bool maximize, double offset) {
    opt::Model m;
    m.start = vec<i64>(start);
    m.index = vec<i64>(index);
    m.value = vec<double>(value);
    m.obj = vec<double>(obj);
    m.lb = vec<double>(lb);
    m.ub = vec<double>(ub);
    m.row_lo = vec<double>(row_lo);
    m.row_hi = vec<double>(row_hi);
    m.integer = vec<std::uint8_t>(integer);
    m.nrow = static_cast<i64>(m.row_lo.size());
    m.ncol = static_cast<i64>(m.obj.size());
    m.maximize = maximize;
    m.offset = offset;
    m.check();
    return m;
}

opt::Presolve presolve_of(const std::string& s) {
    if (s == "choose") return opt::Presolve::choose;
    if (s == "off") return opt::Presolve::off;
    if (s == "on") return opt::Presolve::on;
    throw std::invalid_argument("presolve must be 'choose', 'off' or 'on'");
}

py::dict result_dict(opt::SolveResult&& r) {
    py::dict d;
    d["status"] = r.status;
    d["has_solution"] = r.has_solution;
    d["x"] = r.has_solution ? py::object(to_array(std::move(r.x))) : py::object(py::none());
    d["objective"] = r.objective;
    d["bound"] = r.bound;
    d["gap"] = r.gap;
    d["nodes"] = r.nodes;
    d["seconds"] = r.seconds;
    return d;
}

py::dict py_solve_highs(const Ints& start, const Ints& index, const Doubles& value, const Doubles& obj,
                        const Doubles& lb, const Doubles& ub, const Doubles& row_lo, const Doubles& row_hi,
                        const Bytes& integer, bool maximize, double offset, double gap, double time_limit,
                        int threads, const std::string& presolve, bool verbose, bool relaxation,
                        const py::object& initial, const std::string& method, std::int64_t n_units) {
    const opt::Model m = model_of(start, index, value, obj, lb, ub, row_lo, row_hi, integer, maximize, offset);
    opt::SolveOptions o;
    o.gap = gap;
    o.time_limit = time_limit;
    o.threads = threads;
    o.presolve = presolve_of(presolve);
    o.verbose = verbose;
    o.relaxation = relaxation;
    if (!initial.is_none()) o.start = vec<double>(initial.cast<Doubles>());
    if (method != "highs" && method != "covering" && method != "wide")
        throw std::invalid_argument("method must be 'highs', 'covering' or 'wide'");
    if (method == "covering" && !opt::is_covering(m))
        throw std::invalid_argument("not a covering model (a minimisation with only >= rows and finite bounds)");
    if (method == "wide" && !opt::is_wide(m, n_units))
        throw std::invalid_argument("not a wide model (n_units units with finite bounds, few other columns)");
    opt::SolveResult r;
    {
        py::gil_scoped_release release;
        r = method == "covering" ? opt::solve_covering(m, o)
            : method == "wide"   ? opt::solve_wide(m, n_units, o)
                                 : opt::solve_highs(m, o);
    }
    return result_dict(std::move(r));
}

// The inputs of a multi-action problem from a dict of arrays (the tables' columns, internal ids).
actions::Input actions_input(const py::dict& d) {
    actions::Input in;
    auto dbl = [&](const char* k) { return vec<double>(d[k].cast<Doubles>()); };
    auto ints = [&](const char* k) { return vec<i64>(d[k].cast<Ints>()); };
    in.n_units = d["n_units"].cast<int>();
    in.n_features = d["n_features"].cast<int>();
    in.n_threats = d["n_threats"].cast<int>();
    in.monitoring_cost = dbl("monitoring_cost");
    in.unit_status = dbl("unit_status");
    in.target_recovery = dbl("target_recovery");
    in.target_conservation = dbl("target_conservation");
    in.df_unit = ints("df_unit");
    in.df_feature = ints("df_feature");
    in.df_amount = dbl("df_amount");
    in.blm_actions = dbl("blm_actions");
    in.act_unit = ints("act_unit");
    in.act_threat = ints("act_threat");
    in.act_amount = dbl("act_amount");
    in.act_cost = dbl("act_cost");
    in.act_status = dbl("act_status");
    in.sens_feature = ints("sens_feature");
    in.sens_threat = ints("sens_threat");
    in.sens_delta = dbl("sens_delta");
    in.has_boundary = d["has_boundary"].cast<bool>();
    if (in.has_boundary) {
        in.bnd_id1 = ints("bnd_id1");
        in.bnd_id2 = ints("bnd_id2");
        in.bnd_value = dbl("bnd_value");
    }
    in.check();
    return in;
}

actions::Settings actions_settings(const py::dict& d) {
    actions::Settings s;
    s.maximize = d["maximize"].cast<bool>();
    s.blm = d["blm"].cast<double>();
    s.curve = d["curve"].cast<int>();
    s.segments = d["segments"].cast<int>();
    s.step = d["step"].cast<double>();
    s.budget = d["budget"].cast<double>();
    if (d.contains("dep_from")) {
        s.dep_from = vec<i64>(d["dep_from"].cast<Ints>());
        s.dep_to = vec<i64>(d["dep_to"].cast<Ints>());
        s.dep_weight = vec<double>(d["dep_weight"].cast<Doubles>());
        s.dep_hard = vec<std::uint8_t>(d["dep_hard"].cast<Bytes>());
    }
    return s;
}

py::dict compiled_dict(actions::Compiled&& c) {
    py::dict out;
    out["start"] = to_array(std::move(c.model.start));
    out["index"] = to_array(std::move(c.model.index));
    out["value"] = to_array(std::move(c.model.value));
    out["obj"] = to_array(std::move(c.model.obj));
    out["lb"] = to_array(std::move(c.model.lb));
    out["ub"] = to_array(std::move(c.model.ub));
    out["integer"] = to_array(std::move(c.model.integer));
    out["rhs"] = to_array(std::move(c.rhs));
    out["sense"] = to_array(std::move(c.sense));
    out["n_pairs"] = c.n_pairs;
    out["first_pow"] = c.first_pow;
    return out;
}

}  // namespace

namespace opt {

void register_python(py::module_& m) {
    m.def(
        "opt_marxan",
        [](const Doubles& cost, const py::array_t<int, py::array::c_style | py::array::forcecast>& status,
           const Ints& offset, const py::array_t<int, py::array::c_style | py::array::forcecast>& spindex,
           const Doubles& amount, const Doubles& target, const Doubles& prop, const Doubles& spf,
           const py::array_t<int, py::array::c_style | py::array::forcecast>& targetocc, const Doubles& fixedcost,
           const Ints& nb_start, const py::array_t<int, py::array::c_style | py::array::forcecast>& nb,
           const Doubles& nb_cost, const py::dict& opts) {
            marxan::Data d;
            d.npu = static_cast<int>(cost.size());
            d.nsp = static_cast<int>(target.size());
            d.cost = vec<double>(cost);
            d.status = std::vector<int>(status.data(), status.data() + status.size());
            d.offset = vec<std::int64_t>(offset);
            d.spindex = std::vector<int>(spindex.data(), spindex.data() + spindex.size());
            d.amount = vec<double>(amount);
            d.target = vec<double>(target);
            d.prop = vec<double>(prop);
            d.spf = vec<double>(spf);
            d.targetocc = std::vector<int>(targetocc.data(), targetocc.data() + targetocc.size());
            d.fixedcost = vec<double>(fixedcost);
            d.nb_start = vec<std::int64_t>(nb_start);
            d.nb = std::vector<int>(nb.data(), nb.data() + nb.size());
            d.nb_cost = vec<double>(nb_cost);
            if (static_cast<int>(d.status.size()) != d.npu || static_cast<int>(d.offset.size()) != d.npu + 1 ||
                static_cast<int>(d.fixedcost.size()) != d.npu || static_cast<int>(d.nb_start.size()) != d.npu + 1)
                throw std::invalid_argument("inconsistent planning-unit arrays");
            marxan::Options o;
            auto get = [&](const char* k, auto& v) {
                if (opts.contains(k)) v = opts[k].cast<std::decay_t<decltype(v)>>();
            };
            get("blm", o.blm);
            get("prop", o.prop);
            get("iterations", o.iterations);
            get("tinit", o.tinit);
            get("tcool", o.tcool);
            get("titns", o.titns);
            get("repeats", o.repeats);
            get("costthresh", o.costthresh);
            get("tpf1", o.tpf1);
            get("tpf2", o.tpf2);
            get("misslevel", o.misslevel);
            get("runmode", o.runmode);
            get("heurtype", o.heurtype);
            get("itimptype", o.itimptype);
            get("connectivity_in", o.connectivity_in);
            get("seed", o.seed);
            get("threads", o.threads);
            get("time_limit", o.time_limit);
            get("fast_penalties", o.fast_penalties);
            get("repair", o.repair);
            get("time_schedule", o.time_schedule);
            get("spf_scale", o.spf_scale);
            marxan::Result r;
            {
                py::gil_scoped_release release;
                r = marxan::solve(d, o);
            }
            const auto nr = static_cast<py::ssize_t>(r.runs.size());
            py::array_t<int> sol({nr, static_cast<py::ssize_t>(d.npu)});
            std::vector<double> total, cst, conn, pen, shortfall, connectivity, mpm, itotal, ipen, iconn, icost;
            std::vector<std::int64_t> pus, missing;
            for (py::ssize_t k = 0; k < nr; ++k) {
                const auto& run = r.runs[static_cast<std::size_t>(k)];
                std::copy(run.R.begin(), run.R.end(), sol.mutable_data() + k * d.npu);
                total.push_back(run.score.total);
                itotal.push_back(run.incremental.total);
                ipen.push_back(run.incremental.penalty);
                iconn.push_back(run.incremental.connection);
                icost.push_back(run.incremental.cost);
                cst.push_back(run.score.cost);
                conn.push_back(run.score.connection);
                pen.push_back(run.score.penalty);
                shortfall.push_back(run.shortfall);
                connectivity.push_back(run.connectivity);
                mpm.push_back(run.min_prop_met);
                pus.push_back(run.score.pus);
                missing.push_back(run.missing);
            }
            py::dict out;
            out["penalty"] = to_array(std::move(r.penalty));
            out["target"] = to_array(std::move(r.target));
            out["unmet"] = r.unmet;
            out["timed_out"] = r.timed_out;
            out["best"] = r.best;
            out["solutions"] = sol;
            out["score"] = to_array(std::move(total));
            out["cost"] = to_array(std::move(cst));
            out["connection"] = to_array(std::move(conn));
            out["penalty_total"] = to_array(std::move(pen));
            out["shortfall"] = to_array(std::move(shortfall));
            out["connectivity"] = to_array(std::move(connectivity));
            out["mpm"] = to_array(std::move(mpm));
            out["pus"] = to_array(std::move(pus));
            out["missing"] = to_array(std::move(missing));
            out["incremental_score"] = to_array(std::move(itotal));
            out["incremental_penalty"] = to_array(std::move(ipen));
            out["incremental_connection"] = to_array(std::move(iconn));
            out["incremental_cost"] = to_array(std::move(icost));
            return out;
        },
        py::arg("cost"), py::arg("status"), py::arg("offset"), py::arg("spindex"), py::arg("amount"),
        py::arg("target"), py::arg("prop"), py::arg("spf"), py::arg("targetocc"), py::arg("fixedcost"),
        py::arg("nb_start"), py::arg("nb"), py::arg("nb_cost"), py::arg("options"),
        "Marxan 4.0.6's runs (opt/marxan.hpp).");
    py::class_<FeatureMatrix>(m, "OptFeatureMatrix",
                              "Features x planning units of raster units, built block by block (opt/features.hpp).")
        .def(py::init<std::int64_t>(), py::arg("layers"))
        .def(
            "add",
            [](FeatureMatrix& self, const Doubles& values, const Ints& unit, int threads) {
                if (values.ndim() != 2 || values.shape(0) != self.layers())
                    throw std::invalid_argument("values must be (layers, cells)");
                if (unit.ndim() != 1 || unit.shape(0) != values.shape(1))
                    throw std::invalid_argument("unit must have one entry per cell");
                py::gil_scoped_release release;
                self.add(values.data(), values.shape(1), unit.data(), threads);
            },
            py::arg("values"), py::arg("unit"), py::arg("threads") = 1,
            "Add the next cells: values (layers, cells), unit: each cell's planning unit (-1: none).")
        .def(
            "finish",
            [](const FeatureMatrix& self) {
                std::vector<std::int64_t> start, index;
                std::vector<double> value;
                self.csr(start, index, value);
                std::vector<double> totals = self.totals();
                return py::make_tuple(to_array(std::move(totals)), to_array(std::move(start)),
                                      to_array(std::move(index)), to_array(std::move(value)));
            },
            "(totals, start, index, value): every layer's total and its amounts in the units (compressed rows).");
    using IntArr = py::array_t<int, py::array::c_style | py::array::forcecast>;
    m.def(
        "opt_maxflow",
        [](int n, const IntArr& i, const IntArr& j, const Doubles& cap_ij, const Doubles& cap_ji,
           const Doubles& source, const Doubles& sink) {
            MaxFlow g(n, std::vector<int>(i.data(), i.data() + i.size()),
                      std::vector<int>(j.data(), j.data() + j.size()), vec<double>(cap_ij), vec<double>(cap_ji));
            double f;
            py::array_t<std::uint8_t> side(n);
            {
                py::gil_scoped_release release;
                f = g.solve(vec<double>(source), vec<double>(sink));
            }
            for (int v = 0; v < n; ++v) side.mutable_data()[v] = g.source_side(v);
            return py::make_tuple(f, side);
        },
        "(flow, source side) of the minimum s-t cut (Boykov-Kolmogorov).");
    m.def(
        "opt_solve_cut",
        [](const Doubles& cost, const IntArr& status, const Ints& offset, const IntArr& spindex, const Doubles& amount,
           const Doubles& target, const Doubles& fixedcost, const Ints& nb_start, const IntArr& nb,
           const Doubles& nb_cost, double gap, double time_limit, int max_evaluations, bool verbose,
           const IntArr& initial) {
            marxan::Data d;
            d.npu = static_cast<int>(cost.size());
            d.nsp = static_cast<int>(target.size());
            d.cost = vec<double>(cost);
            d.status = std::vector<int>(status.data(), status.data() + status.size());
            d.offset = vec<std::int64_t>(offset);
            d.spindex = std::vector<int>(spindex.data(), spindex.data() + spindex.size());
            d.amount = vec<double>(amount);
            d.target = vec<double>(target);
            d.prop.assign(static_cast<std::size_t>(d.nsp), 0.0);
            d.spf.assign(static_cast<std::size_t>(d.nsp), 1.0);
            d.targetocc.assign(static_cast<std::size_t>(d.nsp), 0);
            d.fixedcost = vec<double>(fixedcost);
            d.nb_start = vec<std::int64_t>(nb_start);
            d.nb = std::vector<int>(nb.data(), nb.data() + nb.size());
            d.nb_cost = vec<double>(nb_cost);
            if (static_cast<int>(d.status.size()) != d.npu || static_cast<int>(d.offset.size()) != d.npu + 1 ||
                static_cast<int>(d.fixedcost.size()) != d.npu || static_cast<int>(d.nb_start.size()) != d.npu + 1)
                throw std::invalid_argument("inconsistent planning-unit arrays");
            cut::Options o;
            o.gap = gap;
            o.time_limit = time_limit;
            o.max_evaluations = max_evaluations;
            o.verbose = verbose;
            std::vector<int> init(initial.data(), initial.data() + initial.size());
            cut::Result r;
            {
                py::gil_scoped_release release;
                r = cut::solve(d, o, init);
            }
            py::dict out;
            out["status"] = r.status;
            out["x"] = to_array(std::vector<double>(r.x.begin(), r.x.end()));
            out["objective"] = r.objective;
            out["bound"] = r.bound;
            out["gap"] = r.gap;
            out["evaluations"] = r.evaluations;
            out["seconds"] = r.seconds;
            return out;
        },
        "The cut engine (Lagrangian bound by minimum cuts, repaired selections) on Marxan's problem at BLM 1.");
    m.def(
        "opt_actions_pairs",
        [](const py::dict& data) {
            const actions::Input in = actions_input(data);
            actions::Pairs p = actions::pairs(in);
            py::dict out;
            out["feature"] = to_array(std::move(p.feature));
            out["unit"] = to_array(std::move(p.unit));
            out["amount"] = to_array(std::move(p.amount));
            out["threatened"] = to_array(std::move(p.threatened));
            out["sum_alpha"] = to_array(std::move(p.sum_alpha));
            out["start"] = to_array(std::move(p.start));
            out["action"] = to_array(std::move(p.action));
            out["coef"] = to_array(std::move(p.coef));
            out["contribution"] = to_array(std::move(p.contribution));
            out["action_status"] = to_array(std::move(p.action_status));
            return out;
        },
        py::arg("data"), "Every (feature, unit) pair of a multi-action problem and its actions' coefficients.");
    m.def(
        "opt_actions_benefit",
        [](const py::dict& data, const py::object& solution) {
            const actions::Input in = actions_input(data);
            const actions::Pairs p = actions::pairs(in);
            std::vector<double> sol;
            if (!solution.is_none()) sol = vec<double>(solution.cast<Doubles>());
            actions::Benefit b = actions::benefit(in, p, sol);
            py::dict out;
            out["dist"] = to_array(std::move(b.dist));
            out["dist_threatened"] = to_array(std::move(b.dist_threatened));
            out["conservation"] = to_array(std::move(b.conservation));
            out["recovery"] = to_array(std::move(b.recovery));
            return out;
        },
        py::arg("data"), py::arg("solution"), "The most benefit of every feature, or a solution's (units, actions).");
    m.def(
        "opt_actions_compile",
        [](const py::dict& data, const py::dict& settings, bool disaggregate, bool curves) {
            const actions::Input in = actions_input(data);
            const actions::Pairs p = actions::pairs(in);
            return compiled_dict(actions::compile(in, p, actions_settings(settings), disaggregate, curves));
        },
        py::arg("data"), py::arg("settings"), py::arg("disaggregate") = false, py::arg("curves") = false,
        "prioriactions's model (rows in CSR, sense 0 <=, 1 >=, 2 =).");
    m.def(
        "opt_actions_equivalent",
        [](const py::dict& data, const py::dict& settings) -> py::object {
            const actions::Input in = actions_input(data);
            const actions::Pairs p = actions::pairs(in);
            actions::Compiled c = actions::equivalent(in, p, actions_settings(settings));
            if (c.model.nrow == 0 && c.model.ncol == 0) return py::none();
            return compiled_dict(std::move(c));
        },
        py::arg("data"), py::arg("settings"), "The equivalent model for solving (curve 1), or None.");
    m.def(
        "opt_actions_solve",
        [](const py::dict& data, const py::dict& settings, double gap, double time_limit, int threads, bool verbose,
           bool engine, const py::object& initial) {
            const actions::Input in = actions_input(data);
            const actions::Pairs p = actions::pairs(in);
            const actions::Settings st = actions_settings(settings);
            opt::SolveOptions o;
            o.gap = gap;
            o.time_limit = time_limit;
            o.threads = threads;
            o.verbose = verbose;
            if (!initial.is_none()) o.start = vec<double>(initial.cast<Doubles>());
            opt::SolveResult r;
            {
                py::gil_scoped_release release;
                r = actions::solve(in, p, st, o, engine);
            }
            return result_dict(std::move(r));
        },
        py::arg("data"), py::arg("settings"), py::arg("gap"), py::arg("time_limit"), py::arg("threads"),
        py::arg("verbose"), py::arg("engine") = true, py::arg("initial") = py::none(),
        "Solve a multi-action problem: the closure engine, then HiGHS (x: units, actions, pair benefits).");
    m.def(
        "opt_actions_local_benefit",
        [](const py::dict& data, const Doubles& x) {
            const actions::Input in = actions_input(data);
            const actions::Pairs p = actions::pairs(in);
            std::vector<double> xv = vec<double>(x);
            if (static_cast<i64>(xv.size()) < in.n_units + in.n_actions())
                throw std::invalid_argument("x needs the units and the actions");
            return to_array(actions::local_benefit(p, in.n_units, xv));
        },
        py::arg("data"), py::arg("x"), "Every pair's benefit for a solution (units, then actions).");
    m.def(
        "opt_actions_costs",
        [](const py::dict& data, const Doubles& units, const Doubles& acts) {
            return to_array(actions::costs(actions_input(data), vec<double>(units), vec<double>(acts)));
        },
        py::arg("data"), py::arg("units"), py::arg("actions"), "(monitoring, every threat's actions' cost).");
    m.def(
        "opt_actions_connectivity",
        [](const py::dict& data, const Doubles& units, const Doubles& acts) {
            return to_array(actions::connectivity(actions_input(data), vec<double>(units), vec<double>(acts)));
        },
        py::arg("data"), py::arg("units"), py::arg("actions"), "(units, every threat's actions' boundary).");
    m.def("opt_solve_highs", &py_solve_highs, py::arg("start"), py::arg("index"), py::arg("value"), py::arg("obj"),
          py::arg("lb"), py::arg("ub"), py::arg("row_lo"), py::arg("row_hi"), py::arg("integer"),
          py::arg("maximize"), py::arg("offset"), py::arg("gap"), py::arg("time_limit"), py::arg("threads"),
          py::arg("presolve"), py::arg("verbose"), py::arg("relaxation"), py::arg("initial"),
          py::arg("method") = "highs", py::arg("n_units") = -1,
          "Solve a model (rows in CSR) on HiGHS ('highs'), with the covering engine ('covering': sifted LP, "
          "rounding, then HiGHS) or the wide engine ('wide': the first n_units columns are the planning units): "
          "status, has_solution, x, objective, bound, gap, nodes, seconds.");
}

}  // namespace opt
