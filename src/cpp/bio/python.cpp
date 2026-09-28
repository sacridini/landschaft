// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the biodiversity module (landschaft._core.bio_*).

#include "bio/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "bio/conditions.hpp"

namespace py = pybind11;

namespace {

using bio::i32;
using bio::i64;

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> vector_to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

template <class T>
std::vector<T> to_vector(const py::handle& h) {
    const CArray<T> a = py::cast<CArray<T>>(h);
    return std::vector<T>(a.data(), a.data() + a.size());
}

bio::Groups groups_from(const py::dict& d) {
    bio::Groups g;
    const CArray<double> coords = d["coords"].cast<CArray<double>>();
    if (coords.ndim() != 2) throw std::invalid_argument("coords must be 2-D");
    g.n = coords.shape(0);
    g.ndim = static_cast<i32>(coords.shape(1));
    g.coords.assign(coords.data(), coords.data() + coords.size());
    g.numeric = to_vector<std::uint8_t>(d["numeric"]);
    g.axis_str = to_vector<i32>(d["axis_str"]);
    g.name_str = to_vector<i32>(d["name_str"]);
    g.cell_sizes = to_vector<double>(d["cell_sizes"]);
    g.strings = d["strings"].cast<std::vector<std::string>>();
    return g;
}

bio::Csr csr_from(const py::handle& h) {
    const py::tuple t = h.cast<py::tuple>();
    bio::Csr c;
    c.indptr = to_vector<i64>(t[0]);
    c.indices = to_vector<i32>(t[1]);
    return c;
}

bio::Program program_from(const py::dict& d) {
    bio::Program p;
    const CArray<i32> ops = d["ops"].cast<CArray<i32>>();
    const CArray<i32> abc = d["abc"].cast<CArray<i32>>();
    const CArray<double> v = d["v"].cast<CArray<double>>();
    const auto n = ops.size();
    p.nodes.resize(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        bio::Node& node = p.nodes[static_cast<std::size_t>(i)];
        node.op = static_cast<bio::Op>(ops.data()[i]);
        node.a = abc.data()[3 * i];
        node.b = abc.data()[3 * i + 1];
        node.c = abc.data()[3 * i + 2];
        node.v = v.data()[i];
    }
    p.seq = to_vector<i32>(d["seq"]);
    p.root = d["root"].cast<i32>();
    p.n_locals = d["n_locals"].cast<i32>();
    for (const py::handle& h : d["calls"].cast<py::list>()) {
        const py::dict c = h.cast<py::dict>();
        bio::Call call;
        call.kind = static_cast<bio::CallKind>(c["kind"].cast<i32>());
        call.p = c["p"].cast<std::vector<double>>();
        call.axes = c["axes"].cast<std::vector<i32>>();
        call.sizes = c["sizes"].cast<std::vector<double>>();
        call.origin = c["origin"].cast<std::vector<double>>();
        call.which = c["which"].cast<i32>();
        call.index = c["index"].cast<i32>();
        call.text = c["text"].cast<i32>();
        call.axis = c["axis"].cast<i32>();
        p.calls.push_back(std::move(call));
    }
    for (const py::handle& h : d["masks"].cast<py::list>()) p.masks.push_back(to_vector<std::uint8_t>(h));
    for (const py::handle& h : d["sets"].cast<py::list>()) p.sets.push_back(csr_from(h));
    for (const py::handle& h : d["rotations"].cast<py::list>()) p.rotations.push_back(to_vector<double>(h));
    p.use_euc = d["use_euc"].cast<bool>();
    p.use_cell = d["use_cell"].cast<bool>();
    p.cell_lists = d["cell_lists"].cast<bool>();
    p.iter_axes = d["iter_axes"].cast<std::vector<i32>>();
    p.bounds = d["bounds"].cast<std::vector<double>>();
    p.self_only = d["self_only"].cast<bool>();
    p.always_true = d["always_true"].cast<bool>();
    p.always_false = d["always_false"].cast<bool>();
    return p;
}

}  // namespace

namespace bio {

void register_python(py::module_& m) {
    m.def(
        "bio_find_neighbours",
        [](const py::dict& groups, const py::dict& program, const CArray<i32>& procs, const py::object& exclude,
           bool parallel) {
            const Groups g = groups_from(groups);
            const Program p = program_from(program);
            std::vector<i32> pr(procs.data(), procs.data() + procs.size());
            Csr ex;
            const Csr* exp = nullptr;
            if (!exclude.is_none()) {
                ex = csr_from(exclude);
                exp = &ex;
            }
            Csr out;
            {
                py::gil_scoped_release release;
                out = find_neighbours(g, p, pr, exp, parallel);
            }
            return py::make_tuple(vector_to_array(std::move(out.indptr)), vector_to_array(std::move(out.indices)));
        },
        py::arg("groups"), py::arg("program"), py::arg("procs"), py::arg("exclude"), py::arg("parallel"),
        "Neighbour lists (CSR) of the processing groups under a compiled spatial condition.");
    m.def(
        "bio_definition_query",
        [](const py::dict& groups, const py::dict& program, bool parallel) {
            const Groups g = groups_from(groups);
            const Program p = program_from(program);
            std::vector<std::uint8_t> out;
            {
                py::gil_scoped_release release;
                out = definition_query(g, p, parallel);
            }
            return vector_to_array(std::move(out));
        },
        py::arg("groups"), py::arg("program"), py::arg("parallel"), "Groups passing a compiled definition query.");
    m.def(
        "bio_side_rotation",
        [](const CArray<double>& coords, i32 a0, i32 a1, double angle) {
            Groups g;
            g.n = coords.shape(0);
            g.ndim = static_cast<i32>(coords.shape(1));
            g.coords.assign(coords.data(), coords.data() + coords.size());
            return vector_to_array(side_rotation(g, a0, a1, angle));
        },
        py::arg("coords"), py::arg("axis0"), py::arg("axis1"), py::arg("angle"),
        "Rotated, scaled x coordinates of the groups (Biodiverse's sidedness conditions).");
}

}  // namespace bio
