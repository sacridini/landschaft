// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the ecosystem services (landschaft._core.serv_*).

#include "serv/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "serv/sdr.hpp"
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

void register_water(py::module_& m);
void register_sdr(py::module_& m);

void register_python(py::module_& m) {
    register_water(m);
    register_sdr(m);
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

namespace {

template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array water_fraction_t(const I32& cls, const F32& kc, const F32& root, const F32& veg, const py::array& eto_,
                           const py::array& precip_, const py::array& soil_, const py::array& pawc_, double z,
                           int threads) {
    const auto eto = Arr<T>::ensure(eto_), precip = Arr<T>::ensure(precip_), soil = Arr<T>::ensure(soil_),
               pawc = Arr<T>::ensure(pawc_);
    for (const py::array* a : {static_cast<const py::array*>(&eto), static_cast<const py::array*>(&precip),
                               static_cast<const py::array*>(&soil), static_cast<const py::array*>(&pawc)})
        same_size(cls, *a, "the rasters differ in size from the land use map");
    F32 out(cls.request().shape);
    {
        py::gil_scoped_release release;
        water_fraction<T>(cls.data(), kc.data(), root.data(), veg.data(), eto.data(), precip.data(), soil.data(),
                          pawc.data(), z, static_cast<i64>(cls.size()), out.mutable_data(), threads);
    }
    return out;
}

template <typename U>
py::tuple water_yield_t(const F32& fractp, const py::array& precip_, int threads) {
    const auto precip = Arr<U>::ensure(precip_);
    same_size(fractp, precip, "the precipitation differs in size from the map");
    Arr<U> wy(fractp.request().shape), aet(fractp.request().shape);
    {
        py::gil_scoped_release release;
        water_yield<U>(fractp.data(), precip.data(), static_cast<i64>(fractp.size()), wy.mutable_data(),
                       aet.mutable_data(), threads);
    }
    return py::make_tuple(wy, aet);
}

template <typename V>
py::array potential_et_t(const I32& cls, const F32& kc, const py::array& eto_, int threads) {
    const auto eto = Arr<V>::ensure(eto_);
    same_size(cls, eto, "the reference evapotranspiration differs in size from the map");
    Arr<V> out(cls.request().shape);
    {
        py::gil_scoped_release release;
        potential_et<V>(cls.data(), kc.data(), eto.data(), static_cast<i64>(cls.size()), out.mutable_data(), threads);
    }
    return out;
}

}  // namespace

void register_sdr(py::module_& m) {
    using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
    m.def("serv_pgp_slope", [](const F64& z, double xcell, double ycell, int threads) {
        if (z.ndim() != 2) throw std::invalid_argument("expected a 2-D raster");
        const int ny = static_cast<int>(z.shape(0)), nx = static_cast<int>(z.shape(1));
        F32 out({ny, nx});
        {
            py::gil_scoped_release release;
            pgp_slope(z.data(), ny, nx, xcell, ycell, out.mutable_data(), threads);
        }
        return out;
    }, py::arg("z"), py::arg("xcell"), py::arg("ycell"), py::arg("threads"),
       "pygeoprocessing's percent slope (float32, NaN missing).");
    m.def("serv_sediment_deposition", [](const py::array& dir, const F32& e_prime, const F32& sdr, int block) {
        if (dir.ndim() != 2) throw std::invalid_argument("expected 2-D flow directions");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        same_size(dir, e_prime, "E' differs in size from the directions");
        same_size(dir, sdr, "the SDR differs in size from the directions");
        F32 dep({ny, nx}), flux({ny, nx});
        if (dir.dtype().is(py::dtype::of<std::uint8_t>())) {
            const auto d = U8::ensure(dir);
            py::gil_scoped_release release;
            sediment_deposition_d8(d.data(), e_prime.data(), sdr.data(), ny, nx, block, dep.mutable_data(),
                                   flux.mutable_data());
        } else {
            const auto d = I32::ensure(dir);
            py::gil_scoped_release release;
            sediment_deposition_mfd(d.data(), e_prime.data(), sdr.data(), ny, nx, block, dep.mutable_data(),
                                    flux.mutable_data());
        }
        return py::make_tuple(dep, flux);
    }, py::arg("dir"), py::arg("e_prime"), py::arg("sdr"), py::arg("block"),
       "InVEST's sediment deposition and flux (D8 for uint8 directions, else packed MFD).");
}

void register_water(py::module_& m) {
    m.def("serv_water_fraction", [](const I32& cls, const F32& kc, const F32& root, const F32& veg,
                                    const py::array& eto, const py::array& precip, const py::array& soil,
                                    const py::array& pawc, double z, bool wide, int threads) {
        if (wide)
            return water_fraction_t<double>(cls, kc, root, veg, eto, precip, soil, pawc, z, threads);
        return water_fraction_t<float>(cls, kc, root, veg, eto, precip, soil, pawc, z, threads);
    }, py::arg("cls"), py::arg("kc"), py::arg("root"), py::arg("veg"), py::arg("eto"), py::arg("precip"),
       py::arg("soil"), py::arg("pawc"), py::arg("z"), py::arg("wide"), py::arg("threads"),
       "AET / P by cell (InVEST's fractp, float32), computed in float64 if wide else float32.");
    m.def("serv_water_yield", [](const F32& fractp, const py::array& precip, bool wide, int threads) {
        return wide ? water_yield_t<double>(fractp, precip, threads) : water_yield_t<float>(fractp, precip, threads);
    }, py::arg("fractp"), py::arg("precip"), py::arg("wide"), py::arg("threads"),
       "Water yield and actual evapotranspiration by cell.");
    m.def("serv_potential_et", [](const I32& cls, const F32& kc, const py::array& eto, bool wide, int threads) {
        return wide ? potential_et_t<double>(cls, kc, eto, threads) : potential_et_t<float>(cls, kc, eto, threads);
    }, py::arg("cls"), py::arg("kc"), py::arg("eto"), py::arg("wide"), py::arg("threads"),
       "Potential evapotranspiration eto * kc by cell.");
}

}  // namespace serv
