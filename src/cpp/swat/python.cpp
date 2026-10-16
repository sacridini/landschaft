// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the SWAT+ port (landschaft._core.swat_*).

#include "swat/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "swat/fio.hpp"
#include "swat/swat.hpp"

namespace py = pybind11;

namespace swat {

namespace {

// The daily records of one tag: (yrc, day, id) per record and the values.
py::dict dumps_to_python(const std::vector<DumpRecord>& dumps) {
    std::map<int, std::vector<const DumpRecord*>> by_tag;
    for (const DumpRecord& r : dumps) by_tag[r.tag].push_back(&r);
    py::dict out;
    for (const auto& [tag, recs] : by_tag) {
        std::size_t width = 0;
        for (const DumpRecord* r : recs) width = std::max(width, r->values.size());
        const auto n = static_cast<py::ssize_t>(recs.size());
        py::array_t<std::int32_t> keys({n, static_cast<py::ssize_t>(3)});
        py::array_t<float> vals({n, static_cast<py::ssize_t>(width)});
        auto k = keys.mutable_unchecked<2>();
        auto v = vals.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const DumpRecord& r = *recs[static_cast<std::size_t>(i)];
            k(i, 0) = r.yrc;
            k(i, 1) = r.day;
            k(i, 2) = r.id;
            for (std::size_t j = 0; j < width; ++j)
                v(i, static_cast<py::ssize_t>(j)) = j < r.values.size() ? r.values[j] : 0.0f;
        }
        out[py::int_(tag)] = py::make_tuple(keys, vals);
    }
    return out;
}

}  // namespace

void register_python(py::module_& m) {
    m.def(
        "swat_run",
        [](const std::map<std::string, py::bytes>& files, py::object dump, py::object inject, bool c_factor_plant) {
            Sim s;
            for (const auto& [name, text] : files) s.files.add(name, std::string(text));
            if (!inject.is_none()) {
                // {tag: (keys, vals)} from the reference: 40 (hd of HRU objects), 10 (hwb_d), 6 (ru_elem dr)
                for (auto item : py::cast<py::dict>(inject)) {
                    const int tag = py::cast<int>(item.first);
                    auto kv = py::cast<py::tuple>(item.second);
                    auto keys = py::cast<py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(kv[0]);
                    auto vals = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(kv[1]);
                    auto k = keys.unchecked<2>();
                    auto v = vals.unchecked<2>();
                    for (py::ssize_t r = 0; r < k.shape(0); ++r) {
                        std::vector<float> row(static_cast<std::size_t>(v.shape(1)));
                        for (py::ssize_t c = 0; c < v.shape(1); ++c) row[static_cast<std::size_t>(c)] = v(r, c);
                        const InjectKey key{k(r, 0), k(r, 1), k(r, 2)};
                        if (tag == 40) s.inject_hd[key] = std::move(row);
                        else if (tag == 10) s.inject_hwb[key] = std::move(row);
                        else if (tag == 6) s.inject_dr[k(r, 2)] = std::move(row);
                        else throw py::value_error("inject: tags 6, 10 and 40");
                    }
                }
            }
            RunOptions opt;
            opt.c_factor_plant = c_factor_plant;
            if (py::isinstance<py::bool_>(dump)) {
                opt.dump = py::cast<bool>(dump);
            } else {
                opt.dump = true;
                for (auto t : dump) s.dump_tags.insert(py::cast<int>(t));
            }
            {
                py::gil_scoped_release release;
                run(s, opt);
            }
            py::dict out;
            out["dump"] = dumps_to_python(s.dumps);
            out["prints"] = dumps_to_python(s.prints);
            out["warnings"] = s.warnings;
            std::vector<std::string> stations;
            for (int i = 1; i <= s.db_mx.wst; ++i) stations.push_back(s.wst(i).name);
            out["stations"] = stations;
            out["start"] = py::make_tuple(s.time_init.yrc_start, s.time_init.day_start);
            return out;
        },
        py::arg("files"), py::arg("dump") = true, py::arg("inject") = py::none(), py::arg("c_factor_plant") = false,
        "Run a SWAT+ project given as {file name: contents}; returns the daily records (dump: True, False or the "
        "tags to keep) and the printed ones. inject: reference "
        "records replacing hru_control (tests of the routing).");

    m.def(
        "swat_list_read",
        [](const std::string& text, const std::vector<std::string>& statements) {
            // each statement a string of item kinds: i integer, f real, d double, s character, l logical
            fio::FileSet fs;
            fs.add("t", text);
            fio::File f = fs.open("t");
            py::list out;
            for (const std::string& kinds : statements) {
                std::vector<int> iv(kinds.size(), -999);
                std::vector<float> fv(kinds.size(), -999.0f);
                std::vector<double> dv(kinds.size(), -999.0);
                std::vector<std::string> sv(kinds.size(), "<unset>");
                std::vector<bool> lv_(kinds.size(), false);
                int iostat = 0;
                {
                    fio::File::Reader r = f.reader();
                    for (std::size_t k = 0; k < kinds.size(); ++k) {
                        switch (kinds[k]) {
                            case 'i': r.item(iv[k]); break;
                            case 'f': r.item(fv[k]); break;
                            case 'd': r.item(dv[k]); break;
                            case 's': r.item(sv[k]); break;
                            default: throw py::value_error("item kinds are i, f, d, s");
                        }
                    }
                    iostat = r.finish();
                }
                py::list vals;
                for (std::size_t k = 0; k < kinds.size(); ++k) {
                    switch (kinds[k]) {
                        case 'i': vals.append(iv[k]); break;
                        case 'f': vals.append(fv[k]); break;
                        case 'd': vals.append(dv[k]); break;
                        default: vals.append(sv[k]); break;
                    }
                }
                out.append(py::make_tuple(iostat, vals));
            }
            return out;
        },
        py::arg("text"), py::arg("statements"),
        "Fortran list-directed READ statements over a text (for the tests of the reader).");

    m.def(
        "swat_parse_float",
        [](const std::string& text) {
            float v = 0.0f;
            if (!fio::to_float(text, v)) throw py::value_error("not a Fortran real: " + text);
            return v;
        },
        "Text to float as the Fortran runtime reads it (correctly rounded).");
}

}  // namespace swat
