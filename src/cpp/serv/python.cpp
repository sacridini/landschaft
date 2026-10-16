// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the ecosystem services (landschaft._core.serv_*).

#include "serv/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "serv/services.hpp"

namespace py = pybind11;

namespace {

using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

void same_size(const py::array& a, const py::array& b, const char* what) {
    if (a.size() != b.size()) throw std::invalid_argument(what);
}

}  // namespace

namespace serv {

void register_python(py::module_& m) {
    m.def("serv_classify", [](const I64& lulc, const U8& valid, const I64& codes, int threads) {
        same_size(lulc, valid, "the mask differs in size from the map");
        I32 out(lulc.request().shape);
        i64 first;
        {
            py::gil_scoped_release release;
            first = classify(lulc.data(), valid.data(), static_cast<i64>(lulc.size()), codes.data(),
                             static_cast<int>(codes.size()), out.mutable_data(), threads);
        }
        return py::make_tuple(out, first);
    }, py::arg("lulc"), py::arg("valid"), py::arg("codes"), py::arg("threads"),
       "Class index of every cell (-1 missing, -2 unknown code) and the first cell of unknown code (-1: none).");

    m.def("serv_carbon_pools", [](const I32& cls, const F32& pools, int threads) {
        if (pools.ndim() != 2 || pools.shape(1) != 4) throw std::invalid_argument("pools must be k x 4");
        std::vector<F32> out;
        float* ptr[5];
        for (int p = 0; p < 5; ++p) {
            out.emplace_back(cls.request().shape);
            ptr[p] = out.back().mutable_data();
        }
        {
            py::gil_scoped_release release;
            carbon_pools(cls.data(), static_cast<i64>(cls.size()), pools.data(), ptr, threads);
        }
        return py::make_tuple(out[0], out[1], out[2], out[3], out[4]);
    }, py::arg("cls"), py::arg("pools"), py::arg("threads"),
       "The four carbon pools and their float32 total, by cell.");

    m.def("serv_total", [](const F32& x, int threads) {
        py::gil_scoped_release release;
        return total(x.data(), static_cast<i64>(x.size()), threads);
    }, py::arg("x"), py::arg("threads"), "Sum of the non-NaN values in float64, in a fixed order.");

    m.def("serv_distance_transform", [](const U8& region, int threads) {
        if (region.ndim() != 2) throw std::invalid_argument("expected a 2-D raster");
        const int ny = static_cast<int>(region.shape(0)), nx = static_cast<int>(region.shape(1));
        F32 out({ny, nx});
        {
            py::gil_scoped_release release;
            distance_transform(region.data(), ny, nx, out.mutable_data(), threads);
        }
        return out;
    }, py::arg("region"), py::arg("threads"),
       "pygeoprocessing's Euclidean distance transform (float32, pixels) to the non-zero cells.");

    m.def("serv_threat_degradation", [](const F32& dist, std::optional<F32> given, const I32& cls, const F32& sens,
                                        std::optional<F32> access, double max_px, bool exponential, double weight,
                                        F32& total, int threads) {
        same_size(dist, cls, "the distance differs in size from the map");
        same_size(dist, total, "the total differs in size from the map");
        if (access) same_size(dist, *access, "the access differs in size from the map");
        if (given) same_size(dist, *given, "the decay differs in size from the map");
        F32 deg(cls.request().shape);
        {
            py::gil_scoped_release release;
            threat_degradation(dist.data(), given ? given->data() : nullptr, cls.data(), sens.data(), access ? access->data() : nullptr, max_px,
                               exponential, weight, static_cast<i64>(cls.size()), deg.mutable_data(),
                               total.mutable_data(), threads);
        }
        return deg;
    }, py::arg("dist"), py::arg("given"), py::arg("cls"), py::arg("sens"), py::arg("access"), py::arg("max_px"),
       py::arg("exponential"), py::arg("weight"), py::arg("total"), py::arg("threads"),
       "One threat's degradation by cell (added to total, in place).");

    m.def("serv_quality", [](const F32& total, const I32& cls, const F32& habitat, double ksq, int threads) {
        same_size(total, cls, "the degradation differs in size from the map");
        F32 out(cls.request().shape);
        {
            py::gil_scoped_release release;
            quality(total.data(), cls.data(), habitat.data(), ksq, static_cast<i64>(cls.size()), out.mutable_data(),
                    threads);
        }
        return out;
    }, py::arg("total"), py::arg("cls"), py::arg("habitat"), py::arg("ksq"), py::arg("threads"),
       "Habitat quality h (1 - d^2.5 / (d^2.5 + k^2.5)) by cell.");
}

}  // namespace serv
