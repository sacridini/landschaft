// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the biodiversity module (landschaft._core.bio_*).

#include "bio/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "bio/conditions.hpp"
#include "bio/indices.hpp"
#include "bio/phylocom.hpp"

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

bio::BaseDataView basedata_from(const py::dict& d) {
    bio::BaseDataView b;
    b.n_groups = d["n_groups"].cast<i64>();
    b.n_labels = d["n_labels"].cast<i64>();
    b.indptr = to_vector<i64>(d["indptr"]);
    b.indices = to_vector<i32>(d["indices"]);
    b.counts = to_vector<double>(d["counts"]);
    b.label_range = to_vector<double>(d["label_range"]);
    b.label_abundance = to_vector<double>(d["label_abundance"]);
    b.label_sample_count = to_vector<double>(d["label_sample_count"]);
    b.abc_any_variant = d["abc_any_variant"].cast<i32>();
    b.n_label_levels = d["n_label_levels"].cast<i32>();
    b.label_prefix = to_vector<i32>(d["label_prefix"]);
    b.prefix_names = d["prefix_names"].cast<std::vector<std::string>>();
    return b;
}

bio::TreeView tree_from(const py::dict& d) {
    bio::TreeView t;
    t.n = d["n"].cast<i32>();
    t.parent = to_vector<i32>(d["parent"]);
    t.length = to_vector<double>(d["length"]);
    t.child_ptr = to_vector<i64>(d["child_ptr"]);
    t.child_idx = to_vector<i32>(d["child_idx"]);
    t.depth = to_vector<i32>(d["depth"]);
    t.terminal = to_vector<std::uint8_t>(d["terminal"]);
    t.in_trimmed = to_vector<std::uint8_t>(d["in_trimmed"]);
    t.trimmed_child_count = to_vector<i32>(d["trimmed_child_count"]);
    t.total_length = d["total_length"].cast<double>();
    t.trimmed_total_length = d["trimmed_total_length"].cast<double>();
    t.root = d["root"].cast<i32>();
    t.label_node = to_vector<i32>(d["label_node"]);
    t.node_label = to_vector<i32>(d["node_label"]);
    t.n_terminals = d["n_terminals"].cast<i32>();
    t.n_trimmed_terminals = d["n_trimmed_terminals"].cast<i32>();
    t.zero_length_count = d["zero_length_count"].cast<i32>();
    t.eq_full_total = d["eq_full_total"].cast<double>();
    t.eq_trimmed_total = d["eq_trimmed_total"].cast<double>();
    t.eq_trimmed_length = d["eq_trimmed_length"].cast<double>();
    t.default_eq_len = d["default_eq_len"].cast<double>();
    if (d.contains("pc_steps")) {
        t.pc_steps = d["pc_steps"].cast<std::vector<i32>>();
        t.pc_vpd = d["pc_vpd"].cast<bool>();
        t.pc_seed = d["pc_seed"].cast<std::uint64_t>();
    }
    return t;
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
        "bio_calculations",
        []() {
            py::list out;
            for (const CalcInfo& c : calculation_registry()) {
                py::list lists;
                for (const auto& l : c.lists) lists.append(py::make_tuple(l.first, static_cast<int>(l.second)));
                out.append(py::make_tuple(c.name, c.scalars, lists, c.uses_nbr_lists, c.uses_processing_element,
                                          c.uses_tree));
            }
            return out;
        },
        "The calculations of the index engine: (name, scalar indices, list indices, neighbour sets, "
        "uses the processing group).");
    m.def(
        "bio_mt_shuffle",
        [](std::uint64_t seed, i32 n, i32 times) {
            MT64 mt({seed});
            std::vector<std::vector<i32>> out;
            for (i32 k = 0; k < times; ++k) {
                std::vector<i32> v(static_cast<std::size_t>(n));
                for (i32 i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i;
                mt.shuffle(v);
                out.push_back(v);
            }
            return out;
        },
        "Shuffles of 0..n-1 by Math::Random::MT::Auto seeded with `seed` (tests).");
    m.def(
        "bio_ultrametric",
        [](const py::dict& tree) {
            const TreeView tv = tree_from(tree);
            PhyloCom pc(tv, false, 0);
            return py::make_tuple(pc.ultrametric, pc.multifurcating);
        },
        "Biodiverse's is_ultrametric, and whether a node has more than two children.");
    m.def(
        "bio_run_indices",
        [](const py::dict& basedata, const CArray<i32>& procs, const py::tuple& set1, const py::object& set2,
           const std::vector<std::string>& calcs, bool parallel, const py::object& tree) {
            const BaseDataView bd = basedata_from(basedata);
            TreeView tv;
            const TreeView* tvp = nullptr;
            if (!tree.is_none()) {
                tv = tree_from(tree.cast<py::dict>());
                tvp = &tv;
            }
            std::vector<i32> pr(procs.data(), procs.data() + procs.size());
            const Csr s1 = csr_from(set1);
            Csr s2;
            const Csr* s2p = nullptr;
            if (!set2.is_none()) {
                s2 = csr_from(set2);
                s2p = &s2;
            }
            IndexResult r;
            {
                py::gil_scoped_release release;
                r = run_indices(bd, pr, s1, s2p, calcs, parallel, tvp);
            }
            py::dict out;
            const auto np = static_cast<py::ssize_t>(pr.size());
            const auto ns = static_cast<py::ssize_t>(r.scalar_names.size());
            py::array_t<double> scal({np, ns});
            std::copy(r.scalars.begin(), r.scalars.end(), scal.mutable_data());
            out["scalar_names"] = r.scalar_names;
            out["scalars"] = scal;
            py::list lists;
            for (std::size_t k = 0; k < r.lists.size(); ++k) {
                ListOut& lo = r.lists[k];
                lists.append(py::make_tuple(r.list_names[k], static_cast<int>(r.list_kinds[k]),
                                            vector_to_array(std::move(lo.indptr)), vector_to_array(std::move(lo.keys)),
                                            vector_to_array(std::move(lo.values))));
            }
            out["lists"] = lists;
            out["texts"] = r.texts;
            return out;
        },
        py::arg("basedata"), py::arg("procs"), py::arg("set1"), py::arg("set2"), py::arg("calculations"),
        py::arg("parallel"), py::arg("tree") = py::none(),
        "Biodiverse indices of the processing groups over their neighbour sets.");
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
