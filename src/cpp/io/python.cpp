// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the zonal statistics and the extraction at points
// (landschaft._core.io_*).

#include "io/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "io/io.hpp"
#include "io/warp.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using lsio::i64;

// Calls f with the typed data of a C-contiguous array of one of the raster types.
template <class F>
auto with_typed(const py::array& a, F&& f) {
    if (!(a.flags() & py::array::c_style)) throw py::value_error("expected a C-contiguous array");
    const py::dtype dt = a.dtype();
    if (dt.is(py::dtype::of<std::uint8_t>())) return f(static_cast<const std::uint8_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int8_t>())) return f(static_cast<const std::int8_t*>(a.data()));
    if (dt.is(py::dtype::of<std::uint16_t>())) return f(static_cast<const std::uint16_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int16_t>())) return f(static_cast<const std::int16_t*>(a.data()));
    if (dt.is(py::dtype::of<std::uint32_t>())) return f(static_cast<const std::uint32_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int32_t>())) return f(static_cast<const std::int32_t*>(a.data()));
    if (dt.is(py::dtype::of<std::int64_t>())) return f(static_cast<const std::int64_t*>(a.data()));
    if (dt.is(py::dtype::of<float>())) return f(static_cast<const float*>(a.data()));
    if (dt.is(py::dtype::of<double>())) return f(static_cast<const double*>(a.data()));
    throw py::type_error("unsupported raster type: use (u)int8/16/32, int64, float32 or float64");
}

py::array_t<double> to_array(std::vector<double>&& v, std::vector<py::ssize_t> shape) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, holder->data(), owner);
}

// (bands, ny, nx) of a 2-D or 3-D raster
std::array<int, 3> raster_shape(const py::array& a) {
    if (a.ndim() == 2) return {1, static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1))};
    if (a.ndim() == 3) return {static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)), static_cast<int>(a.shape(2))};
    throw py::value_error("expected a 2-D (y, x) or 3-D (band, y, x) raster");
}


using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int8Array = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
using I64Array = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using PointerArray = py::array_t<std::uintptr_t, py::array::c_style | py::array::forcecast>;

// The source coordinates of the warp kernel's samples (checked): samples rows
// r0 .. last_row - 1, columns c0 .. last_col - 1 must be covered.
lsio::WarpCoords make_coords(int kind, double offset, const std::array<double, 6>& gt,
                             const std::array<double, 6>& inv, const Int8Array& level, const DoubleArray& grid0,
                             const I64Array& node_off, const DoubleArray& nodes, const std::array<i64, 3>& cells,
                             const DoubleArray& dense, i64 dense_row0, i64 dense_col0, std::uintptr_t transformer,
                             const PointerArray& transformer_args, i64 r0, i64 c0, i64 last_row, i64 last_col) {
    using lsio::kWarpCell;
    using lsio::WarpCoords;
    WarpCoords c;
    c.kind = kind;
    c.offset = offset;
    std::copy(gt.begin(), gt.end(), c.gt);
    std::copy(inv.begin(), inv.end(), c.inv);
    if (kind == WarpCoords::LATTICE) {
        const i64 cells_x = cells[0], cell_x0 = cells[1], cell_y0 = cells[2];
        if (cells_x < 1 || cell_x0 < 0 || cell_y0 < 0 || level.size() % cells_x != 0) {
            throw py::value_error("bad lattice");
        }
        const i64 cells_y = static_cast<i64>(level.size()) / cells_x;
        if (cell_x0 * kWarpCell > c0 || cell_y0 * kWarpCell > r0 || (cell_x0 + cells_x) * kWarpCell < last_col ||
            (cell_y0 + cells_y) * kWarpCell < last_row) {
            throw py::value_error("the lattice does not cover the destination region");
        }
        if (node_off.size() != level.size()) throw py::value_error("one node offset per lattice cell");
        if (grid0.size() != (cells_x + 1) * (cells_y + 1) * 2) {
            throw py::value_error("grid0 must hold the corners of every lattice cell");
        }
        const i64 n_nodes = static_cast<i64>(nodes.size()) / 2;
        for (i64 k = 0; k < static_cast<i64>(level.size()); ++k) {
            const int lev = level.data()[k];
            if (lev < 0 || lev > 4) throw py::value_error("lattice levels go from 0 to 4");
            if (lev == 0) continue;
            const i64 side = kWarpCell / (kWarpCell >> lev) + 1;
            if (node_off.data()[k] < 0 || node_off.data()[k] + side * side > n_nodes) {
                throw py::value_error("lattice nodes out of range");
            }
        }
        c.cells_x = cells_x;
        c.cells_y = cells_y;
        c.cell_x0 = cell_x0;
        c.cell_y0 = cell_y0;
        c.level = level.data();
        c.grid0 = grid0.data();
        c.node_off = node_off.data();
        c.nodes = nodes.data();
    } else if (kind == WarpCoords::DENSE) {
        if (dense.ndim() != 3 || dense.shape(2) != 2) {
            throw py::value_error("dense coordinates must be (rows, columns, 2)");
        }
        if (dense_row0 > r0 || dense_col0 > c0 || dense_row0 + dense.shape(0) < last_row ||
            dense_col0 + dense.shape(1) < last_col) {
            throw py::value_error("the dense coordinates do not cover the destination region");
        }
        c.dense = dense.data();
        c.dense_row0 = dense_row0;
        c.dense_col0 = dense_col0;
        c.dense_stride = dense.shape(1);
    } else if (kind != WarpCoords::AFFINE && kind != WarpCoords::EXACT) {
        throw py::value_error("unknown coordinate kind");
    }
    if (transformer != 0) {
        if (transformer_args.size() < 1) throw py::value_error("one transformer argument per thread");
        c.transformer = reinterpret_cast<lsio::TransformerFunc>(transformer);
        c.transformer_args = reinterpret_cast<void* const*>(transformer_args.data());
        c.n_transformer_args = static_cast<int>(transformer_args.size());
    } else if (kind == WarpCoords::EXACT) {
        throw py::value_error("exact coordinates need the transformer");
    }
    return c;
}

}  // namespace

namespace lsio {

void register_python(py::module_& m) {
    py::class_<Runs>(m, "IoRuns", "The cells of zones as runs of rows (internal).")
        .def_property_readonly("zones", &Runs::zones)
        .def_property_readonly("runs", [](const Runs& r) { return static_cast<i64>(r.y.size()); })
        .def(
            "cells",
            [](const Runs& r) {
                std::vector<double> c(static_cast<std::size_t>(r.zones()), 0.0);
                for (i64 z = 0; z < r.zones(); ++z) {
                    for (i64 k = r.offsets[z]; k < r.offsets[z + 1]; ++k) c[z] += r.x1[k] - r.x0[k];
                }
                return to_array(std::move(c), {r.zones()});
            },
            "Cells of every zone.");

    m.def(
        "io_polygon_runs",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& xy,
           const py::array_t<i64, py::array::c_style | py::array::forcecast>& ring_offsets,
           const py::array_t<i64, py::array::c_style | py::array::forcecast>& geom_rings, const std::array<double, 6>& gt,
           int nx, int ny, bool all_touched) {
            if (xy.ndim() != 2 || xy.shape(1) != 2) throw py::value_error("xy must be (n, 2)");
            if (geom_rings.size() < 1) throw py::value_error("geom_rings must have ngeoms + 1 entries");
            const i64 ngeoms = static_cast<i64>(geom_rings.size()) - 1;
            py::gil_scoped_release release;
            return polygon_runs(xy.data(), ring_offsets.data(), geom_rings.data(), ngeoms, gt.data(), nx, ny,
                                all_touched);
        },
        py::arg("xy"), py::arg("ring_offsets"), py::arg("geom_rings"), py::arg("geotransform"), py::arg("nx"),
        py::arg("ny"), py::arg("all_touched"), "Cells of polygons as GDAL's rasterizer burns them.");

    m.def(
        "io_label_runs",
        [](const py::array& labels, bool has_nodata, double nodata) {
            if (labels.ndim() != 2) throw py::value_error("zones must be a 2-D raster");
            const int ny = static_cast<int>(labels.shape(0)), nx = static_cast<int>(labels.shape(1));
            std::vector<double> labs;
            Runs runs = with_typed(labels, [&](const auto* p) {
                py::gil_scoped_release release;
                return label_runs(p, ny, nx, has_nodata, nodata, &labs);
            });
            const auto n = static_cast<py::ssize_t>(labs.size());
            return py::make_tuple(std::move(runs), to_array(std::move(labs), {n}));
        },
        py::arg("labels"), py::arg("has_nodata"), py::arg("nodata"), "Cells of every label of a raster of zones.");

    m.def(
        "io_zonal_stats",
        [](const py::array& values, bool has_nodata, double nodata, const Runs& runs, const std::vector<int>& stats,
           const std::vector<double>& percentiles) {
            const auto [bands, ny, nx] = raster_shape(values);
            if (percentiles.size() != stats.size()) throw py::value_error("one percentile per statistic");
            for (i64 r = 0; r < static_cast<i64>(runs.y.size()); ++r) {
                if (runs.y[r] >= ny || runs.x1[r] > nx) throw py::value_error("zones larger than the raster");
            }
            std::vector<double> out = with_typed(values, [&](const auto* p) {
                py::gil_scoped_release release;
                return zonal_stats(p, bands, ny, nx, has_nodata, nodata, runs, stats, percentiles);
            });
            return to_array(std::move(out), {runs.zones(), bands, static_cast<py::ssize_t>(stats.size())});
        },
        py::arg("values"), py::arg("has_nodata"), py::arg("nodata"), py::arg("runs"), py::arg("stats"),
        py::arg("percentiles"), "Statistics of every zone and band: (zones, bands, stats).");

    m.def(
        "io_burn_runs",
        [](py::array& out, const Runs& runs, const py::array_t<double, py::array::c_style | py::array::forcecast>& value,
           bool add) {
            if (out.ndim() != 2) throw py::value_error("out must be 2-D");
            if (!(out.flags() & py::array::c_style) || !out.writeable()) {
                throw py::value_error("out must be C-contiguous and writeable");
            }
            if (value.size() != runs.zones()) throw py::value_error("one value per zone");
            const int ny = static_cast<int>(out.shape(0)), nx = static_cast<int>(out.shape(1));
            const py::dtype dt = out.dtype();
            auto run = [&](auto* p) {
                py::gil_scoped_release release;
                burn_runs(p, ny, nx, runs, value.data(), add);
            };
            void* d = out.mutable_data();
            if (dt.is(py::dtype::of<std::uint8_t>())) run(static_cast<std::uint8_t*>(d));
            else if (dt.is(py::dtype::of<std::int8_t>())) run(static_cast<std::int8_t*>(d));
            else if (dt.is(py::dtype::of<std::uint16_t>())) run(static_cast<std::uint16_t*>(d));
            else if (dt.is(py::dtype::of<std::int16_t>())) run(static_cast<std::int16_t*>(d));
            else if (dt.is(py::dtype::of<std::uint32_t>())) run(static_cast<std::uint32_t*>(d));
            else if (dt.is(py::dtype::of<std::int32_t>())) run(static_cast<std::int32_t*>(d));
            else if (dt.is(py::dtype::of<std::int64_t>())) run(static_cast<std::int64_t*>(d));
            else if (dt.is(py::dtype::of<float>())) run(static_cast<float*>(d));
            else if (dt.is(py::dtype::of<double>())) run(static_cast<double*>(d));
            else throw py::type_error("unsupported raster type");
        },
        py::arg("out"), py::arg("runs"), py::arg("value"), py::arg("add"),
        "Burns one value per zone into out, zone after zone (or adds them up).");

    m.def(
        "io_block_mean",
        [](const py::array& values, int factor, bool has_nodata, double nodata) {
            if (values.ndim() != 2) throw py::value_error("expected a 2-D raster");
            if (factor < 1) throw py::value_error("factor must be at least 1");
            const int ny = static_cast<int>(values.shape(0)), nx = static_cast<int>(values.shape(1));
            std::vector<float> out = with_typed(values, [&](const auto* p) {
                py::gil_scoped_release release;
                return block_mean(p, ny, nx, factor, has_nodata, nodata);
            });
            auto* holder = new std::vector<float>(std::move(out));
            py::capsule owner(holder, [](void* q) { delete static_cast<std::vector<float>*>(q); });
            return py::array_t<float>({static_cast<py::ssize_t>(ny / factor), static_cast<py::ssize_t>(nx / factor)},
                                      holder->data(), owner);
        },
        py::arg("values"), py::arg("factor"), py::arg("has_nodata"), py::arg("nodata"),
        "Mean of every factor x factor block (cells with data), float32.");

    m.def(
        "io_warp",
        [](const py::array& src, int method, const std::array<int, 2>& src_shape, const std::array<int, 2>& window,
           const std::array<double, 2>& scales, bool has_src_nodata, double src_nodata, double dst_nodata,
           const std::array<int, 4>& region, int kind, double offset, const std::array<double, 6>& gt,
           const std::array<double, 6>& inv,
           const Int8Array& level, const DoubleArray& grid0, const I64Array& node_off, const DoubleArray& nodes,
           const std::array<i64, 3>& cells, const DoubleArray& dense, i64 dense_row0, i64 dense_col0,
           std::uintptr_t transformer, const PointerArray& transformer_args) {
            if (src.ndim() != 3) throw py::value_error("src must be (bands, rows, columns)");
            if (!(src.flags() & py::array::c_style)) throw py::value_error("src must be C-contiguous");
            WarpSpec sp;
            sp.method = method;
            sp.bands = static_cast<int>(src.shape(0));
            sp.src_h = src_shape[0];
            sp.src_w = src_shape[1];
            sp.win_r0 = window[0];
            sp.win_c0 = window[1];
            sp.win_h = static_cast<int>(src.shape(1));
            sp.win_w = static_cast<int>(src.shape(2));
            sp.xscale = scales[0];
            sp.yscale = scales[1];
            sp.has_src_nodata = has_src_nodata;
            sp.src_nodata = src_nodata;
            sp.dst_nodata = dst_nodata;
            sp.dst_r0 = region[0];
            sp.dst_c0 = region[1];
            sp.dst_h = region[2];
            sp.dst_w = region[3];
            const bool valid_method = (method >= WARP_NEAREST && method <= WARP_MODE) ||
                                      (method >= WARP_MAX && method <= WARP_Q3) || method == WARP_RMS;
            if (!valid_method) throw py::value_error("unknown warp method");
            if (sp.bands < 1 || sp.dst_h < 0 || sp.dst_w < 0 || sp.dst_r0 < 0 || sp.dst_c0 < 0) {
                throw py::value_error("bad destination region");
            }
            if (sp.win_r0 < 0 || sp.win_c0 < 0 || sp.win_r0 + sp.win_h > sp.src_h || sp.win_c0 + sp.win_w > sp.src_w) {
                throw py::value_error("the source window must lie in the source");
            }
            if (!(scales[0] > 0) || !(scales[1] > 0)) throw py::value_error("scales must be positive");
            // Samples needed: rows r0 .. r0 + h (+ 1 for the corners of the area methods).
            const i64 extra = warp_by_area(method) ? 1 : 0;
            const i64 last_row = static_cast<i64>(sp.dst_r0) + sp.dst_h + extra;  // exclusive
            const i64 last_col = static_cast<i64>(sp.dst_c0) + sp.dst_w + extra;
            const WarpCoords c = make_coords(kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense,
                                             dense_row0, dense_col0, transformer, transformer_args, sp.dst_r0,
                                             sp.dst_c0, last_row, last_col);
#ifdef _OPENMP
            if (c.transformer != nullptr && c.n_transformer_args < omp_get_max_threads()) {
                throw py::value_error("one transformer argument per thread");
            }
#endif
            py::array out(src.dtype(), std::vector<py::ssize_t>{sp.bands, sp.dst_h, sp.dst_w});
            void* o = out.mutable_data();
            auto run = [&](const auto* p) {
                using T = std::remove_const_t<std::remove_pointer_t<decltype(p)>>;
                py::gil_scoped_release release;
                warp<T>(p, static_cast<T*>(o), c, sp);
                return 0;
            };
            if (src.dtype().is(py::dtype::of<std::uint64_t>())) run(static_cast<const std::uint64_t*>(src.data()));
            else with_typed(src, run);
            return out;
        },
        py::arg("src"), py::arg("method"), py::arg("src_shape"), py::arg("window"), py::arg("scales"),
        py::arg("has_src_nodata"), py::arg("src_nodata"), py::arg("dst_nodata"), py::arg("region"), py::arg("kind"),
        py::arg("offset"), py::arg("gt"), py::arg("inv"), py::arg("level"), py::arg("grid0"), py::arg("node_off"),
        py::arg("nodes"), py::arg("cells"), py::arg("dense"), py::arg("dense_row0"), py::arg("dense_col0"),
        py::arg("transformer"), py::arg("transformer_args"),
        "GDAL's warp kernel on a region of the destination: (bands, rows, columns).");

    m.def(
        "io_warp_coords",
        [](int kind, double offset, const std::array<double, 6>& gt, const std::array<double, 6>& inv,
           const Int8Array& level, const DoubleArray& grid0, const I64Array& node_off, const DoubleArray& nodes,
           const std::array<i64, 3>& cells, const DoubleArray& dense, i64 dense_row0, i64 dense_col0,
           std::uintptr_t transformer, const PointerArray& transformer_args, const std::array<i64, 4>& region) {
            const i64 r0 = region[0], c0 = region[1], nr = region[2], nc = region[3];
            if (r0 < 0 || c0 < 0 || nr < 0 || nc < 0 || nc > INT32_MAX) throw py::value_error("bad region");
            const WarpCoords c = make_coords(kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense,
                                             dense_row0, dense_col0, transformer, transformer_args, r0, c0, r0 + nr,
                                             c0 + nc);
            py::array_t<double> out({nr, nc, static_cast<i64>(2)});
            double* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                std::vector<double> x(static_cast<std::size_t>(nc)), y(static_cast<std::size_t>(nc));
                for (i64 j = 0; j < nr; ++j) {
                    c.row(r0 + j, c0, static_cast<int>(nc), x.data(), y.data());
                    for (i64 i = 0; i < nc; ++i) {
                        o[2 * (j * nc + i)] = x[static_cast<std::size_t>(i)];
                        o[2 * (j * nc + i) + 1] = y[static_cast<std::size_t>(i)];
                    }
                }
            }
            return out;
        },
        py::arg("kind"), py::arg("offset"), py::arg("gt"), py::arg("inv"), py::arg("level"), py::arg("grid0"),
        py::arg("node_off"), py::arg("nodes"), py::arg("cells"), py::arg("dense"), py::arg("dense_row0"),
        py::arg("dense_col0"), py::arg("transformer"), py::arg("transformer_args"), py::arg("region"),
        "The source coordinates the warp kernel takes for the samples of a region: (rows, columns, 2).");

    m.def(
        "io_lattice_checks",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& g, int src_h, int src_w) {
            if (g.ndim() != 4 || g.shape(3) != 2 || g.shape(1) % 2 != 1 || g.shape(2) % 2 != 1 || g.shape(1) < 3 ||
                g.shape(2) < 3) {
                throw py::value_error("g must be (n, 2 my + 1, 2 mx + 1, 2)");
            }
            const i64 n = g.shape(0);
            const int my = static_cast<int>((g.shape(1) - 1) / 2), mx = static_cast<int>((g.shape(2) - 1) / 2);
            py::array_t<double> err({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<bool> bad({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<bool> outside({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<double> bbox({n, static_cast<i64>(my), static_cast<i64>(mx), static_cast<i64>(4)});
            {
                py::gil_scoped_release release;
                lattice_checks(g.data(), n, my, mx, src_h, src_w, err.mutable_data(),
                               reinterpret_cast<std::uint8_t*>(bad.mutable_data()),
                               reinterpret_cast<std::uint8_t*>(outside.mutable_data()), bbox.mutable_data());
            }
            return py::make_tuple(err, bad, outside, bbox);
        },
        py::arg("g"), py::arg("src_h"), py::arg("src_w"),
        "Checks of lattice cells: (err, bad, outside, bbox), one per cell of every grid.");

    m.def(
        "io_extract",
        [](const py::array& values, bool has_nodata, double nodata, const std::array<double, 6>& gt,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& x,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& y, bool bilinear, int row0,
           int ny_total) {
            const auto [bands, ny, nx] = raster_shape(values);
            if (ny_total < 0) ny_total = ny;
            if (x.size() != y.size()) throw py::value_error("x and y must have the same length");
            const auto n = static_cast<i64>(x.size());
            std::vector<double> out = with_typed(values, [&](const auto* p) {
                py::gil_scoped_release release;
                return extract(p, bands, ny, nx, has_nodata, nodata, gt.data(), x.data(), y.data(), n, bilinear, row0,
                               ny_total);
            });
            return to_array(std::move(out), {static_cast<py::ssize_t>(n), bands});
        },
        py::arg("values"), py::arg("has_nodata"), py::arg("nodata"), py::arg("geotransform"), py::arg("x"),
        py::arg("y"), py::arg("bilinear"), py::arg("row0") = 0, py::arg("ny_total") = -1,
        "Values of every band at points: (n, bands); values may be the rows [row0, row0 + ny) of the grid.");
}

}  // namespace lsio
