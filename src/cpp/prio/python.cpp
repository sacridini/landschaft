// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the spatial prioritization (landschaft._core.prio_*).

#include "prio/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "prio/prio.hpp"

namespace py = pybind11;

namespace {

using prio::i64;
using Floats = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Bytes = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> to_array(std::vector<T>&& v) {
    auto* holder = new std::vector<T>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

template <class T>
py::array_t<T> copy_array(const std::vector<T>& v) {
    return to_array(std::vector<T>(v));
}

prio::ReadRules rules_of(const std::vector<float>& nodata, const std::vector<float>& extract,
                         const std::vector<float>& thresholds) {
    prio::ReadRules r;
    r.layers = static_cast<int>(nodata.size());
    r.nodata = nodata;
    r.extract = extract;
    r.thresholds = thresholds;
    return r;
}

// (layers, h, w) float32 block and its (h, w) analysis-area mask
void check_block(int layers, const Floats& values, const py::object& area) {
    if (values.ndim() != 3 || values.shape(0) != layers)
        throw std::invalid_argument("expected a (layers, rows, cols) block");
    if (!area.is_none()) {
        const Bytes a = area.cast<Bytes>();
        if (a.ndim() != 2 || a.shape(0) != values.shape(1) || a.shape(1) != values.shape(2))
            throw std::invalid_argument("the area mask must have the block's rows and columns");
    }
}

prio::Rule rule_of(const std::string& s) {
    if (s == "ABF") return prio::Rule::ABF;
    if (s == "CAZ1") return prio::Rule::CAZ1;
    if (s == "CAZ2") return prio::Rule::CAZ2;
    if (s == "CAZMAX") return prio::Rule::CAZMAX;
    if (s == "CAZP") return prio::Rule::CAZP;
    if (s == "CAZPW") return prio::Rule::CAZPW;
    if (s == "RAND") return prio::Rule::RAND;
    if (s == "LOAD") return prio::Rule::LOAD;
    throw std::invalid_argument("unknown marginal loss rule: " + s);
}

template <class T>
std::vector<T> vec_of(const py::object& o) {
    if (o.is_none()) return {};
    const auto a = o.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
    return std::vector<T>(a.data(), a.data() + a.size());
}

py::dict py_rank(const prio::Data& data, const std::string& rule, double param, const std::vector<float>& weights,
                 const std::vector<float>& abf_z, const py::object& cost, bool cost_in_ranking,
                 const py::object& hmask, const py::object& solution, int max_iterations, float ranking_error,
                 bool stop_on_plateau, float smoothing, float curves_interval, std::uint64_t seed) {
    prio::RankSettings s;
    s.rule = rule_of(rule);
    s.param = param;
    s.weights = weights;
    s.abf_z = abf_z;
    s.cost = vec_of<float>(cost);
    s.cost_in_ranking = cost_in_ranking;
    s.hmask = vec_of<std::uint8_t>(hmask);
    s.solution = vec_of<float>(solution);
    s.max_iterations = max_iterations;
    s.ranking_error = ranking_error;
    s.stop_on_plateau = stop_on_plateau;
    s.smoothing = smoothing;
    s.curves_interval = curves_interval;
    s.seed = seed;
    prio::RankResult r;
    {
        py::gil_scoped_release release;
        r = prio::rank(data, s);
    }
    const auto samples = static_cast<py::ssize_t>(r.curve_rank.size());
    py::dict d;
    d["rank"] = to_array(std::move(r.rank));
    d["order"] = to_array(std::move(r.order));
    d["iterations"] = r.iterations;
    d["ranking_error"] = r.ranking_error;
    d["ranking_errors"] = r.ranking_errors;
    d["rank_change"] = r.rank_change;
    d["history_error"] = to_array(std::move(r.history_error));
    d["history_errors"] = to_array(std::move(r.history_errors));
    d["history_change"] = to_array(std::move(r.history_change));
    d["curve_rank"] = to_array(std::move(r.curve_rank));
    py::array_t<float> levels = to_array(std::move(r.curve_levels));
    d["curve_levels"] = levels.reshape({samples, static_cast<py::ssize_t>(data.rules.layers)});
    d["curve_cost"] = to_array(std::move(r.curve_cost));
    d["auc"] = r.auc;
    d["total_cost"] = r.total_cost;
    return d;
}

// Zonation's float sums of the rows of a (samples, layers) matrix: its columns of
// the curve matrix (starting row * layers floats after an aligned start), or
// fresh arrays of a subset of the layers.
py::array_t<float> py_eigen_sums(const Floats& m, const py::object& subset) {
    if (m.ndim() != 2) throw std::invalid_argument("expected a 2-D matrix");
    const i64 S = m.shape(0), L = m.shape(1);
    std::vector<float> out(static_cast<size_t>(S));
    if (subset.is_none()) {
        for (i64 s = 0; s < S; ++s) out[s] = prio::eigen_avx_sum(m.data() + s * L, L, (s * L) % 8);
    } else {
        const auto idx = vec_of<std::int64_t>(subset);
        std::vector<float> v(idx.size());
        for (i64 s = 0; s < S; ++s) {
            for (size_t k = 0; k < idx.size(); ++k) v[k] = m.data()[s * L + idx[k]];
            out[s] = prio::eigen_avx_sum(v.data(), static_cast<i64>(v.size()), 0);
        }
    }
    return to_array(std::move(out));
}

}  // namespace

namespace prio {

void register_python(py::module_& m) {
    py::class_<Stats>(m, "PrioStats")
        .def(py::init([](const std::vector<float>& nodata, const std::vector<float>& extract,
                         const std::vector<float>& thresholds) { return Stats(rules_of(nodata, extract, thresholds)); }),
             py::arg("nodata"), py::arg("extract"), py::arg("thresholds"))
        .def(
            "add_block",
            [](Stats& s, const Floats& values, const py::object& area) {
                check_block(s.rules.layers, values, area);
                Bytes a = area.is_none() ? Bytes() : area.cast<Bytes>();
                const std::uint8_t* ap = area.is_none() ? nullptr : a.data();
                py::gil_scoped_release release;
                s.add_block(values.data(), values.shape(1), values.shape(2), ap);
            },
            py::arg("values"), py::arg("area") = py::none())
        .def_readonly("pixels", &Stats::pixels)
        .def_readonly("data_cells", &Stats::data_cells)
        .def_property_readonly("zeros", [](const Stats& s) { return copy_array(s.zeros); })
        .def_property_readonly("negative", [](const Stats& s) { return copy_array(s.negative); })
        .def_property_readonly("nodata", [](const Stats& s) { return copy_array(s.nodata); })
        .def_property_readonly("min_values", [](const Stats& s) { return copy_array(s.min_values); })
        .def_property_readonly("max_values", [](const Stats& s) { return copy_array(s.max_values); })
        .def_property_readonly("orig_max_values", [](const Stats& s) { return copy_array(s.orig_max_values); });

    py::class_<Data>(m, "PrioData")
        .def(py::init([](const std::vector<float>& nodata, const std::vector<float>& extract,
                         const std::vector<float>& thresholds, std::vector<float> max_values,
                         std::vector<std::uint8_t> keep_zeros, int bits) {
                 return Data(rules_of(nodata, extract, thresholds), std::move(max_values), std::move(keep_zeros),
                             bits);
             }),
             py::arg("nodata"), py::arg("extract"), py::arg("thresholds"), py::arg("max_values"),
             py::arg("keep_zeros"), py::arg("bits"))
        .def(
            "add_block",
            [](Data& d, const Floats& values, const py::object& area) {
                check_block(d.rules.layers, values, area);
                Bytes a = area.is_none() ? Bytes() : area.cast<Bytes>();
                const std::uint8_t* ap = area.is_none() ? nullptr : a.data();
                py::gil_scoped_release release;
                d.add_block(values.data(), values.shape(1), values.shape(2), ap);
            },
            py::arg("values"), py::arg("area") = py::none())
        .def("flush", &Data::flush)
        .def("finish", &Data::finish)
        .def_readonly("bits", &Data::bits)
        .def_readonly("pixels", &Data::pixels)
        .def_property_readonly("layers", [](const Data& d) { return d.rules.layers; })
        .def_property_readonly("cells", &Data::cells)
        .def_property_readonly("entries", [](const Data& d) { return static_cast<i64>(d.entry_layer.size()); })
        .def_property_readonly("cell_index", [](const Data& d) { return copy_array(d.cell_index); })
        .def_property_readonly("max_norm_sums", [](const Data& d) { return copy_array(d.max_norm_sums); })
        .def_property_readonly("orig_abs_sums", [](const Data& d) { return copy_array(d.orig_abs_sums); })
        .def_property_readonly("layer_scale", [](const Data& d) { return copy_array(d.layer_scale); })
        .def_property_readonly("initial_levels", [](const Data& d) { return copy_array(d.initial_levels); })
        .def_property_readonly("quantization_errors", [](const Data& d) { return copy_array(d.quantization_errors); })
        .def_property_readonly("quantized_zeros", [](const Data& d) { return copy_array(d.quantized_zeros); })
        .def(
            "layer",
            [](const Data& d, int layer, bool scaled, float missing) {
                // one layer's values per data cell (sum-normalized or quantized), `missing` where it has none
                if (layer < 0 || layer >= d.rules.layers) throw std::out_of_range("layer");
                std::vector<double> out(static_cast<size_t>(d.cells()), static_cast<double>(missing));
                for (i64 c = 0; c < d.cells(); ++c)
                    for (i64 e = d.cell_ptr[c]; e < d.cell_ptr[c + 1]; ++e)
                        if (d.entry_layer[e] == layer) {
                            const double q = d.entry_value[e];
                            out[c] = scaled ? q * d.layer_scale[layer] : q;
                            break;
                        }
                return to_array(std::move(out));
            },
            py::arg("layer"), py::arg("scaled") = true, py::arg("missing") = 0.0f);

    m.def("prio_rank", &py_rank, py::arg("data"), py::arg("rule"), py::arg("param"), py::arg("weights"),
          py::arg("abf_z"), py::arg("cost"), py::arg("cost_in_ranking"), py::arg("hmask"), py::arg("solution"),
          py::arg("max_iterations"), py::arg("ranking_error"), py::arg("stop_on_plateau"), py::arg("smoothing"),
          py::arg("curves_interval"), py::arg("seed"),
          "Zonation 5's priority ranking of the cells of a PrioData.");
    m.def("prio_eigen_sums", &py_eigen_sums, py::arg("matrix"), py::arg("subset") = py::none(),
          "Zonation's (Eigen AVX) float sums of the rows of a (samples, layers) matrix.");
}

}  // namespace prio
