// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the MILP prioritization (landschaft._core.opt_*).

#include "opt/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "opt/milp.hpp"

namespace py = pybind11;

namespace {

using opt::i64;
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
                        const py::object& initial) {
    const opt::Model m = model_of(start, index, value, obj, lb, ub, row_lo, row_hi, integer, maximize, offset);
    opt::SolveOptions o;
    o.gap = gap;
    o.time_limit = time_limit;
    o.threads = threads;
    o.presolve = presolve_of(presolve);
    o.verbose = verbose;
    o.relaxation = relaxation;
    if (!initial.is_none()) o.start = vec<double>(initial.cast<Doubles>());
    opt::SolveResult r;
    {
        py::gil_scoped_release release;
        r = opt::solve_highs(m, o);
    }
    return result_dict(std::move(r));
}

}  // namespace

namespace opt {

void register_python(py::module_& m) {
    m.def("opt_solve_highs", &py_solve_highs, py::arg("start"), py::arg("index"), py::arg("value"), py::arg("obj"),
          py::arg("lb"), py::arg("ub"), py::arg("row_lo"), py::arg("row_hi"), py::arg("integer"),
          py::arg("maximize"), py::arg("offset"), py::arg("gap"), py::arg("time_limit"), py::arg("threads"),
          py::arg("presolve"), py::arg("verbose"), py::arg("relaxation"), py::arg("initial"),
          "Solve a model (rows in CSR) on HiGHS: status, has_solution, x, objective, bound, gap, nodes, seconds.");
}

}  // namespace opt
