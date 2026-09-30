// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the surface metrics (landschaft._core.surf_*).

#include "surf/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "surf/surface.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Extent = std::tuple<double, double, double, double>;

surf::Grid grid_of(const Array& z, const Extent& extent) {
    if (z.ndim() != 2) throw std::invalid_argument("expected a 2-D raster");
    surf::Grid g{};
    g.z = z.data();
    g.ny = static_cast<int>(z.shape(0));
    g.nx = static_cast<int>(z.shape(1));
    std::tie(g.xmin, g.xmax, g.ymin, g.ymax) = extent;
    return g;
}

py::array_t<double> to_raster(std::vector<double>&& v, int ny, int nx) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)}, holder->data(), owner);
}

py::dict py_amplitude(const Array& z, unsigned parts) {
    surf::Amplitude a{};
    {
        py::gil_scoped_release release;
        a = surf::amplitude(z.data(), static_cast<surf::i64>(z.size()), parts);
    }
    py::dict d;
    d["sa"] = a.sa;
    d["sq"] = a.sq;
    d["ssk"] = a.ssk;
    d["ssk_unadj"] = a.ssk_unadj;
    d["sku"] = a.sku;
    d["sku_raw"] = a.sku_raw;
    d["sv"] = a.sv;
    d["sph"] = a.sph;
    d["smean"] = a.smean;
    return d;
}

std::tuple<py::array_t<double>, int> py_fit_plane(const Array& z, const Extent& extent, int order) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<double> out;
    int ifail = 0;
    {
        py::gil_scoped_release release;
        out = surf::fit_plane(g, order, &ifail);
    }
    return {to_raster(std::move(out), g.ny, g.nx), ifail};
}

std::tuple<py::array_t<double>, int> py_best_fit_plane(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<double> out;
    int order = 0;
    {
        py::gil_scoped_release release;
        out = surf::best_fit_plane(g, &order);
    }
    return {to_raster(std::move(out), g.ny, g.nx), order};
}

// A metric of the grid, with the GIL released.
template <class F>
double grid_metric(const Array& z, const Extent& extent, F&& f) {
    const surf::Grid g = grid_of(z, extent);
    py::gil_scoped_release release;
    return f(g);
}

py::dict py_hybrid(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    double sdq, sdq6, area, flat;
    {
        py::gil_scoped_release release;
        sdq = surf::sdq(g);
        sdq6 = surf::sdq6(g);
        area = surf::surface_area(g);
        flat = surf::flat_area(g);
    }
    py::dict d;
    d["sdq"] = sdq;
    d["sdq6"] = sdq6;
    d["surface_area"] = area;
    d["flat_area"] = flat;
    d["sdr"] = ((area - flat) / flat) * 100;
    return d;
}

py::array_t<std::int64_t> cells_array(std::vector<surf::i64>&& v) {
    auto* holder = new std::vector<surf::i64>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<surf::i64>*>(p); });
    return py::array_t<std::int64_t>({static_cast<py::ssize_t>(holder->size())}, holder->data(), owner);
}

py::array_t<std::int64_t> py_find_peaks(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<surf::i64> cells;
    {
        py::gil_scoped_release release;
        cells = surf::find_peaks(g);
    }
    return cells_array(std::move(cells));
}

py::array_t<std::int64_t> py_find_valleys(const Array& z, const Extent& extent) {
    const surf::Grid g = grid_of(z, extent);
    std::vector<surf::i64> cells;
    {
        py::gil_scoped_release release;
        cells = surf::find_valleys(g);
    }
    return cells_array(std::move(cells));
}

py::dict py_bearing(const Array& z, py::object probs, double sq, bool parameters) {
    std::vector<double> ps = probs.is_none() ? std::vector<double>() : probs.cast<std::vector<double>>();
    surf::Bearing b;
    std::vector<double> heights(ps.size());
    surf::BearingParameters p{};
    {
        py::gil_scoped_release release;
        b = surf::bearing_curve(z.data(), static_cast<surf::i64>(z.size()));
        for (std::size_t k = 0; k < ps.size(); ++k) heights[k] = b.height(ps[k]);
        if (parameters) p = surf::bearing_parameters(b, sq);
    }
    py::dict d;
    d["bearing"] = py::array_t<double>(static_cast<py::ssize_t>(b.bearing.size()), b.bearing.data());
    d["height"] = py::array_t<double>(static_cast<py::ssize_t>(b.heights.size()), b.heights.data());
    d["heights"] = py::array_t<double>(static_cast<py::ssize_t>(heights.size()), heights.data());
    if (parameters) {
        d["sbi"] = p.sbi;
        d["svi"] = p.svi;
        d["sci"] = p.sci;
        d["sk"] = p.sk;
        d["spk"] = p.spk;
        d["svk"] = p.svk;
    }
    return d;
}

py::tuple py_aacf(const Array& z, const Extent& extent, bool lonlat) {
    const surf::Grid g = grid_of(z, extent);
    surf::Raster r;
    {
        py::gil_scoped_release release;
        r = surf::aacf_image(g, lonlat);
    }
    return py::make_tuple(to_raster(std::move(r.v), r.ny, r.nx), py::make_tuple(r.xmin, r.xmax, r.ymin, r.ymax));
}

template <class F>
std::vector<double> grid_vector(const Array& z, const Extent& extent, F&& f) {
    const surf::Grid g = grid_of(z, extent);
    py::gil_scoped_release release;
    return f(g);
}

}  // namespace

namespace surf {

void register_python(py::module_& m) {
    m.def("surf_hybrid", &py_hybrid, py::arg("z"), py::arg("extent"),
          "sdq, sdq6, surface_area, flat_area and sdr.");
    m.def("surf_sdq", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::sdq); },
          py::arg("z"), py::arg("extent"), "geodiv's sdq.");
    m.def("surf_sdq6", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::sdq6); },
          py::arg("z"), py::arg("extent"), "sdq6.");
    m.def("surf_find_peaks", &py_find_peaks, py::arg("z"), py::arg("extent"),
          "geodiv's findpeaks: the peak cells (row-order indices).");
    m.def("surf_find_valleys", &py_find_valleys, py::arg("z"), py::arg("extent"),
          "The valley cells (row-order indices).");
    m.def("surf_sds", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::sds); },
          py::arg("z"), py::arg("extent"), "geodiv's sds.");
    m.def("surf_ssc", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::ssc); },
          py::arg("z"), py::arg("extent"), "geodiv's ssc.");
    m.def("surf_bearing", &py_bearing, py::arg("z"), py::arg("probs") = py::none(), py::arg("sq") = 0.0,
          py::arg("parameters") = false,
          "geodiv's bearing area curve (bearing, height), height_ba at probs and, with parameters, "
          "sbi (from sq), svi, sci, sk, spk, svk.");
    m.def("surf_aacf", &py_aacf, py::arg("z"), py::arg("extent"), py::arg("lonlat"),
          "geodiv's aacf: the image and its extent (xmin, xmax, ymin, ymax).");
    m.def("surf_scl", [](const Array& z, const Extent& e, bool lonlat, const std::vector<double>& t) {
        return grid_vector(z, e, [&](const surf::Grid& g) { return surf::scl(g, lonlat, t); });
    }, py::arg("z"), py::arg("extent"), py::arg("lonlat"), py::arg("thresholds"), "geodiv's scl.");
    m.def("surf_std", [](const Array& z, const Extent& e, bool lonlat) {
        return grid_vector(z, e, [&](const surf::Grid& g) { return surf::texture_direction(g, lonlat); });
    }, py::arg("z"), py::arg("extent"), py::arg("lonlat"), "geodiv's std: (std, stdi).");
    m.def("surf_srw", [](const Array& z, const Extent& e, bool lonlat) {
        return grid_vector(z, e, [&](const surf::Grid& g) { return surf::radial_wavelength(g, lonlat); });
    }, py::arg("z"), py::arg("extent"), py::arg("lonlat"), "geodiv's srw: (srw, srwi, shw).");
    m.def("surf_sfd", [](const Array& z, const Extent& e) {
        const surf::Grid g = grid_of(z, e);
        py::gil_scoped_release release;
        return surf::sfd(g);
    }, py::arg("z"), py::arg("extent"), "The fractal dimension (triangular prisms).");
    m.def("surf_window", [](const Array& z, const Extent& e, bool lonlat, int size, bool circle,
                            const std::vector<int>& ids) {
        const surf::Grid g = grid_of(z, e);
        std::vector<std::vector<double>> res;
        {
            py::gil_scoped_release release;
            res = surf::window_metrics(g, lonlat, size, circle, ids);
        }
        py::list out;
        for (auto& r : res) out.append(to_raster(std::move(r), g.ny, g.nx));
        return out;
    }, py::arg("z"), py::arg("extent"), py::arg("lonlat"), py::arg("size"), py::arg("circle"),
       py::arg("ids"), "Surface metrics in a moving window (one raster per metric id).");
    using Mask = std::vector<int>;
    auto chars = [](const Mask& m) { return std::vector<char>(m.begin(), m.end()); };
    auto grid2d = [](const Array& z, const Extent& e) { return grid_of(z, e); };
    m.def("surf_terrain", [grid2d](const Array& z, const Extent& e, bool lonlat, int neighbors, bool degrees,
                                   bool aspect) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            v = surf::terrain(g, lonlat, neighbors, degrees, aspect);
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("lonlat"), py::arg("neighbors"), py::arg("degrees"),
       py::arg("aspect"), "terra's terrain: slope or aspect.");
    m.def("surf_focal", [grid2d, chars](const Array& z, const Extent& e, int wr, int wc, const Mask& mask, int stat) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            v = surf::focal(g, wr, wc, chars(mask), static_cast<surf::FocalStat>(stat));
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("wr"), py::arg("wc"), py::arg("mask"), py::arg("stat"),
       "A focal statistic (0 mean, 1 sum, 2 min, 3 max, 4 sd) over a window mask.");
    m.def("surf_tri", [grid2d](const Array& z, const Extent& e, int wr, int wc) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            v = surf::tri(g, wr, wc);
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("wr"), py::arg("wc"), "spatialEco's tri.");
    m.def("surf_curvature", [grid2d](const Array& z, const Extent& e, int type) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            v = surf::curvature(g, type);
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("type"), "Curvature (0 planform, 1 profile, 2 total, 3 mcnab, 4 bolstad).");
    m.def("surf_tpi", [grid2d, chars](const Array& z, const Extent& e, int wr, int wc, const Mask& mask, bool normalize) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            v = surf::tpi(g, wr, wc, chars(mask), normalize);
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("wr"), py::arg("wc"), py::arg("mask"), py::arg("normalize"),
       "Topographic position index.");
    m.def("surf_hsp", [grid2d, chars](const Array& z, const Extent& e, const std::vector<int>& rows,
                                      const std::vector<int>& cols, const std::vector<Mask>& masks, bool normalize) {
        const surf::Grid g = grid2d(z, e);
        std::vector<double> v;
        {
            py::gil_scoped_release release;
            std::vector<std::vector<char>> cm;
            for (const auto& mk : masks) cm.push_back(chars(mk));
            v = surf::hsp(g, rows, cols, cm, normalize);
        }
        return to_raster(std::move(v), g.ny, g.nx);
    }, py::arg("z"), py::arg("extent"), py::arg("rows"), py::arg("cols"), py::arg("masks"), py::arg("normalize"),
       "Hierarchical slope position.");
    m.def("surf_s10z", [](const Array& z, const Extent& e) { return grid_metric(z, e, surf::s10z); },
          py::arg("z"), py::arg("extent"), "s10z.");
    m.def("surf_amplitude", &py_amplitude, py::arg("z"), py::arg("parts") = 63u,
          "geodiv's sa, sq, ssk, sku, sv, sph and smean of the values (NaN = NA); parts: "
          "1 sa, 2 sq, 4 ssk, 8 sku, 16 sv and sph, 32 smean.");
    m.def("surf_fit_plane", &py_fit_plane, py::arg("z"), py::arg("extent"), py::arg("order"),
          "geodiv's fitplane (spatial::surf.ls): the surface and spatial's ifail.");
    m.def("surf_best_fit_plane", &py_best_fit_plane, py::arg("z"), py::arg("extent"),
          "geodiv's bestfitplane: the surface and its order.");
}

}  // namespace surf
