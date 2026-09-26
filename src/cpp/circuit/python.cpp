// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the circuit module. They go into landschaft._core and,
// on x86-64, also into landschaft._circuit_avx2: the same code compiled for
// AVX2 and FMA in a module of its own (so that the two builds of Eigen's
// templates never meet in one binary), loaded only on CPUs that have them.

#include "circuit/python.hpp"

#include <pybind11/numpy.h>

#include <cstring>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/grid.hpp"
#include "circuit/omniscape.hpp"

namespace py = pybind11;

namespace {

using circuit::i64;

template <class T>
using CArray = py::array_t<T, py::array::c_style>;

// Hand a std::vector over to numpy without copying it.
py::array_t<double> to_array(std::vector<double>&& v, i64 nx, i64 ny) {
    auto* holder = new std::vector<double>(std::move(v));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({ny, nx}, holder->data(), owner);
}

// Runs `f` with the GIL released and, if n_threads > 0, with that many
// OpenMP threads for parallel regions started by this thread.
template <class F>
auto run_parallel(int n_threads, F&& f) {
    py::gil_scoped_release release;
#if defined(_OPENMP)
    const int previous = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(n_threads);
    struct Restore {
        int n;
        ~Restore() { omp_set_num_threads(n); }
    } restore{previous};
#else
    (void)n_threads;
#endif
    return f();
}

py::dict circuit_advanced(const CArray<double>& conductance, const CArray<double>& source,
                          const CArray<double>& ground, bool four_neighbors, int n_threads) {
    if (conductance.ndim() != 2 || source.ndim() != 2 || ground.ndim() != 2 ||
        source.shape(0) != conductance.shape(0) || source.shape(1) != conductance.shape(1) ||
        ground.shape(0) != conductance.shape(0) || ground.shape(1) != conductance.shape(1))
        throw py::value_error("expected conductance, source and ground rasters of the same shape");
    const i64 ny = conductance.shape(0), nx = conductance.shape(1);
    circuit::AdvancedResult r = run_parallel(n_threads, [&] {
        return circuit::advanced(conductance.data(), source.data(), ground.data(), nx, ny, four_neighbors, n_threads);
    });
    py::dict out;
    out["current"] = to_array(std::move(r.current), nx, ny);
    out["voltage"] = to_array(std::move(r.voltage), nx, ny);
    return out;
}

py::dict circuit_omniscape(const CArray<double>& resistance, const CArray<double>& source, int radius, int buffer,
                           int block_size, bool four_neighbors, bool resistance_is_conductance,
                           bool correct_artifacts, bool flow_potential, int n_threads) {
    if (resistance.ndim() != 2 || source.ndim() != 2 || source.shape(0) != resistance.shape(0) ||
        source.shape(1) != resistance.shape(1))
        throw py::value_error("expected resistance and source rasters of the same shape");
    const i64 ny = resistance.shape(0), nx = resistance.shape(1);
    circuit::OmniscapeParams p;
    p.radius = radius;
    p.buffer = buffer;
    p.block_size = block_size;
    p.four_neighbors = four_neighbors;
    p.resistance_is_conductance = resistance_is_conductance;
    p.correct_artifacts = correct_artifacts;
    p.flow_potential = flow_potential;
    circuit::OmniscapeResult r =
        run_parallel(n_threads, [&] { return circuit::omniscape(resistance.data(), source.data(), nx, ny, p, n_threads); });
    py::dict out;
    out["current"] = to_array(std::move(r.current), nx, ny);
    if (flow_potential) out["flow_potential"] = to_array(std::move(r.flow_potential), nx, ny);
    out["n_targets"] = r.n_targets;
    return out;
}

py::array_t<double> circuit_omniscape_correction(int radius, int buffer, int block_size, bool four_neighbors,
                                                 i64 nx, i64 ny) {
    circuit::OmniscapeParams p;
    p.radius = radius;
    p.buffer = buffer;
    p.block_size = block_size;
    p.four_neighbors = four_neighbors;
    const i64 w = 2 * (static_cast<i64>(radius) + buffer) + 1;
    std::vector<double> c;
    {
        py::gil_scoped_release release;
        c = circuit::omniscape_correction(p, nx, ny);
    }
    return to_array(std::move(c), w, w);
}

}  // namespace

namespace circuit {

void register_python(py::module_& m) {
    m.def("circuit_advanced", &circuit_advanced, py::arg("conductance"), py::arg("source"), py::arg("ground"),
          py::arg("four_neighbors") = false, py::arg("n_threads") = 0,
          "Circuitscape's advanced mode on a raster: node currents and voltages.");
    m.def("circuit_omniscape", &circuit_omniscape, py::arg("resistance"), py::arg("source"), py::arg("radius"),
          py::arg("buffer"), py::arg("block_size"), py::arg("four_neighbors"), py::arg("resistance_is_conductance"),
          py::arg("correct_artifacts"), py::arg("flow_potential"), py::arg("n_threads") = 0,
          "Omniscape's cumulative current (and flow potential) over moving windows.");
    m.def("circuit_omniscape_correction", &circuit_omniscape_correction, py::arg("radius"), py::arg("buffer"),
          py::arg("block_size"), py::arg("four_neighbors"), py::arg("nx"), py::arg("ny"),
          "Omniscape's block-artifact correction factors.");
}

}  // namespace circuit

#if defined(LANDSCHAFT_CIRCUIT_AVX2_MODULE)
PYBIND11_MODULE(_circuit_avx2, m) {
    m.doc() = "landschaft's circuit module built for AVX2 and FMA";
    circuit::register_python(m);
}
#endif
