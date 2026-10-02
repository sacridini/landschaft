// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of MaxNet (landschaft._core.sdm_maxnet_*): the fit returns
// the model as plain arrays (kept by ls.sdm.MaxNet, which pickles), the
// prediction takes them back.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "sdm/maxnet.hpp"

namespace py = pybind11;

namespace sdm {

namespace {

using i64 = std::int64_t;
using Cont = py::array_t<double, py::array::f_style | py::array::forcecast>;
using Cat = py::array_t<std::int32_t, py::array::f_style | py::array::forcecast>;
using Vec = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IVec = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> to_array(const std::vector<T>& v) {
    py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

maxnet::Table table(const Cont& cont, const Cat& cat, const std::vector<std::int32_t>& n_levels) {
    if (cont.ndim() != 2 || cat.ndim() != 2) throw std::invalid_argument("cont and cat must be 2-D (rows x variables)");
    if (cont.shape(0) != cat.shape(0)) throw std::invalid_argument("cont and cat differ in rows");
    maxnet::Table t;
    t.n = static_cast<i64>(cont.shape(0));
    t.n_cont = static_cast<std::int32_t>(cont.shape(1));
    t.n_cat = static_cast<std::int32_t>(cat.shape(1));
    t.cont = cont.data();
    t.cat = cat.data();
    t.n_levels = n_levels;
    if (static_cast<std::int32_t>(n_levels.size()) != t.n_cat) throw std::invalid_argument("one n_levels per factor");
    return t;
}

py::dict py_fit(const Cont& cont, const Cat& cat, const std::vector<std::int32_t>& n_levels, const Vec& p,
                const std::string& classes, double regmult, bool add_samples_to_background, i64 maxit) {
    const maxnet::Table t = table(cont, cat, n_levels);
    if (p.size() != t.n) throw std::invalid_argument("p must have one value per row");
    maxnet::Settings s;
    s.classes = classes;
    s.regmult = regmult;
    s.add_samples_to_background = add_samples_to_background;
    s.maxit = maxit;
    maxnet::Model m;
    {
        py::gil_scoped_release release;
        m = maxnet::fit(t, p.data(), s);
    }
    const std::size_t nf = m.features.size();
    std::vector<std::int32_t> kind(nf), var(nf), var2(nf);
    std::vector<double> k1(nf), k2(nf), pk1(nf), pk2(nf);
    std::vector<std::string> s1(nf), s2(nf);
    for (std::size_t j = 0; j < nf; ++j) {
        const maxnet::Feature& f = m.features[j];
        kind[j] = static_cast<std::int32_t>(f.kind);
        var[j] = f.var;
        var2[j] = f.var2;
        k1[j] = f.k1;
        k2[j] = f.k2;
        pk1[j] = f.pk1;
        pk2[j] = f.pk2;
        s1[j] = f.s1;
        s2[j] = f.s2;
    }
    py::dict out;
    out["classes"] = m.classes;
    out["kind"] = to_array(kind);
    out["var"] = to_array(var);
    out["var2"] = to_array(var2);
    out["k1"] = to_array(k1);
    out["k2"] = to_array(k2);
    out["pk1"] = to_array(pk1);
    out["pk2"] = to_array(pk2);
    out["s1"] = s1;
    out["s2"] = s2;
    out["penalty"] = to_array(m.penalty);
    out["lambda"] = to_array(m.lambda);
    out["feature_min"] = to_array(m.feature_min);
    out["feature_max"] = to_array(m.feature_max);
    out["var_min"] = to_array(m.var_min);
    out["var_max"] = to_array(m.var_max);
    out["sample_means"] = to_array(m.sample_means);
    out["majority"] = to_array(m.majority);
    out["beta_index"] = to_array(m.beta_index);
    out["betas"] = to_array(m.betas);
    out["alpha"] = m.alpha;
    out["entropy"] = m.entropy;
    out["n_passes"] = m.n_passes;
    out["n_added"] = m.n_added;
    out["n_rows"] = m.n_rows;
    return out;
}

py::array_t<double> py_predict(const IVec& kind, const IVec& var, const IVec& var2, const Vec& pk1, const Vec& pk2,
                               const Vec& betas, const Vec& feature_min, const Vec& feature_max, const Vec& var_min,
                               const Vec& var_max, double alpha, double entropy, const Cont& cont, const Cat& cat,
                               const std::vector<std::int32_t>& n_levels, bool clamp, const std::string& type) {
    const py::ssize_t nb = betas.size();
    if (kind.size() != nb || var.size() != nb || var2.size() != nb || pk1.size() != nb || pk2.size() != nb ||
        feature_min.size() != nb || feature_max.size() != nb)
        throw std::invalid_argument("one value per coefficient expected");
    const maxnet::Table t = table(cont, cat, n_levels);
    maxnet::Predictor m;
    for (py::ssize_t j = 0; j < nb; ++j) {
        maxnet::Feature f;
        f.kind = static_cast<maxnet::Kind>(kind.data()[j]);
        f.var = var.data()[j];
        f.var2 = var2.data()[j];
        f.pk1 = pk1.data()[j];
        f.pk2 = pk2.data()[j];
        const bool categorical = f.kind == maxnet::Kind::Categorical;
        if (f.var < 0 || f.var >= (categorical ? t.n_cat : t.n_cont) ||
            (f.kind == maxnet::Kind::Product && (f.var2 < 0 || f.var2 >= t.n_cont)))
            throw std::invalid_argument("a feature refers to a variable that is not in the data");
        m.features.push_back(f);
    }
    m.betas.assign(betas.data(), betas.data() + nb);
    m.feature_min.assign(feature_min.data(), feature_min.data() + nb);
    m.feature_max.assign(feature_max.data(), feature_max.data() + nb);
    m.var_min.assign(var_min.data(), var_min.data() + var_min.size());
    m.var_max.assign(var_max.data(), var_max.data() + var_max.size());
    m.alpha = alpha;
    m.entropy = entropy;
    maxnet::Output o;
    if (type == "link") {
        o = maxnet::Output::Link;
    } else if (type == "exponential") {
        o = maxnet::Output::Exponential;
    } else if (type == "cloglog") {
        o = maxnet::Output::Cloglog;
    } else if (type == "logistic") {
        o = maxnet::Output::Logistic;
    } else {
        throw std::invalid_argument("type must be 'link', 'exponential', 'cloglog' or 'logistic'");
    }
    py::array_t<double> out(static_cast<py::ssize_t>(t.n));
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        maxnet::predict(m, t, clamp, o, po);
    }
    return out;
}

}  // namespace

void register_maxnet_python(py::module_& m) {
    m.def("sdm_maxnet_fit", &py_fit, py::arg("cont"), py::arg("cat"), py::arg("n_levels"), py::arg("p"),
          py::arg("classes") = "default", py::arg("regmult") = 1.0, py::arg("add_samples_to_background") = true,
          py::arg("maxit") = 100000,
          "maxnet::maxnet(): features, regularisation, glmnet's path, alpha and entropy (a dict of arrays)");
    m.def("sdm_maxnet_predict", &py_predict, py::arg("kind"), py::arg("var"), py::arg("var2"), py::arg("pk1"),
          py::arg("pk2"), py::arg("betas"), py::arg("feature_min"), py::arg("feature_max"), py::arg("var_min"),
          py::arg("var_max"), py::arg("alpha"), py::arg("entropy"), py::arg("cont"), py::arg("cat"),
          py::arg("n_levels"), py::arg("clamp"), py::arg("type"), "maxnet's predict(type, clamp)");
    m.def(
        "sdm_r_as_character", [](double x) { return maxnet::r_as_character(x); }, py::arg("x"),
        "R's as.character() of a double");
}

}  // namespace sdm
