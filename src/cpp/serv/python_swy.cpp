// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the seasonal water yield routing (landschaft._core.serv_swy_*).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "serv/swy.hpp"

namespace py = pybind11;

namespace {

using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_months(const py::array& a, int ny, int nx, const char* what) {
    if (a.ndim() != 3 || a.shape(0) != 12 || a.shape(1) != ny || a.shape(2) != nx)
        throw std::invalid_argument(std::string(what) + " must be 12 months on the grid of the flow directions");
}

void check_grid(const py::array& a, int ny, int nx, const char* what) {
    if (a.ndim() != 2 || a.shape(0) != ny || a.shape(1) != nx)
        throw std::invalid_argument(std::string(what) + " differs in shape from the flow directions");
}

template <class P, class E>
py::tuple recharge(const py::array& dir, const py::array& precip, const py::array& et0, const F32& qf, const F32& kc,
                   const F32& alpha, float beta, float gamma, int block) {
    const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
    const auto p = py::array_t<P, py::array::c_style | py::array::forcecast>::ensure(precip);
    const auto e = py::array_t<E, py::array::c_style | py::array::forcecast>::ensure(et0);
    if (alpha.size() != 12) throw std::invalid_argument("alpha must hold 12 months");
    F64 l({ny, nx}), la({ny, nx}), lsa({ny, nx}), aet({ny, nx}), pa({ny, nx});
    serv::SwyMonths<P, E> in{p.data(), e.data(), qf.data(), kc.data(), alpha.data()};
    if (dir.dtype().is(py::dtype::of<std::uint8_t>())) {
        const auto d = U8::ensure(dir);
        py::gil_scoped_release release;
        serv::swy_local_recharge_d8(d.data(), ny, nx, block, in, beta, gamma, l.mutable_data(), la.mutable_data(),
                                    lsa.mutable_data(), aet.mutable_data(), pa.mutable_data());
    } else {
        const auto d = I32::ensure(dir);
        py::gil_scoped_release release;
        serv::swy_local_recharge_mfd(d.data(), ny, nx, block, in, beta, gamma, l.mutable_data(), la.mutable_data(),
                                     lsa.mutable_data(), aet.mutable_data(), pa.mutable_data());
    }
    return py::make_tuple(l, la, lsa, aet, pa);
}

}  // namespace

namespace serv {

void register_swy(py::module_& m) {
    m.def("serv_swy_local_recharge", [](const py::array& dir, const py::array& precip, const py::array& et0,
                                        const F32& qf, const F32& kc, const F32& alpha, float beta, float gamma,
                                        int block) {
        if (dir.ndim() != 2) throw std::invalid_argument("expected 2-D flow directions");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        check_months(precip, ny, nx, "the precipitation");
        check_months(et0, ny, nx, "the ET0");
        check_months(qf, ny, nx, "the quickflow");
        check_months(kc, ny, nx, "the Kc");
        const bool p64 = precip.dtype().is(py::dtype::of<double>()), e64 = et0.dtype().is(py::dtype::of<double>());
        if (p64 && e64) return recharge<double, double>(dir, precip, et0, qf, kc, alpha, beta, gamma, block);
        if (p64) return recharge<double, float>(dir, precip, et0, qf, kc, alpha, beta, gamma, block);
        if (e64) return recharge<float, double>(dir, precip, et0, qf, kc, alpha, beta, gamma, block);
        return recharge<float, float>(dir, precip, et0, qf, kc, alpha, beta, gamma, block);
    }, py::arg("dir"), py::arg("precip"), py::arg("et0"), py::arg("quickflow"), py::arg("kc"), py::arg("alpha"),
       py::arg("beta"), py::arg("gamma"), py::arg("block"),
       "InVEST's local recharge: (L, L_avail, L_sum_avail, AET, P) in float64, NaN where not reached "
       "(D8 for uint8 directions, else packed MFD; precipitation and ET0 float64 or else float32).");
    m.def("serv_swy_baseflow", [](const py::array& dir, const F64& l, const F64& l_avail, const F64& l_sum,
                                  const U8& stream, int block) {
        if (dir.ndim() != 2) throw std::invalid_argument("expected 2-D flow directions");
        const int ny = static_cast<int>(dir.shape(0)), nx = static_cast<int>(dir.shape(1));
        check_grid(l, ny, nx, "L");
        check_grid(l_avail, ny, nx, "L_avail");
        check_grid(l_sum, ny, nx, "L_sum");
        check_grid(stream, ny, nx, "the stream map");
        F64 b({ny, nx}), b_sum({ny, nx});
        if (dir.dtype().is(py::dtype::of<std::uint8_t>())) {
            const auto d = U8::ensure(dir);
            py::gil_scoped_release release;
            swy_baseflow_d8(d.data(), ny, nx, block, l.data(), l_avail.data(), l_sum.data(), stream.data(),
                            b.mutable_data(), b_sum.mutable_data());
        } else {
            const auto d = I32::ensure(dir);
            py::gil_scoped_release release;
            swy_baseflow_mfd(d.data(), ny, nx, block, l.data(), l_avail.data(), l_sum.data(), stream.data(),
                             b.mutable_data(), b_sum.mutable_data());
        }
        return py::make_tuple(b, b_sum);
    }, py::arg("dir"), py::arg("l"), py::arg("l_avail"), py::arg("l_sum"), py::arg("stream"), py::arg("block"),
       "InVEST's baseflow: (B, B_sum) in float64, NaN where not reached.");
}

}  // namespace serv
