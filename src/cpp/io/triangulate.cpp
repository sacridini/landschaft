// SPDX-License-Identifier: GPL-3.0-or-later
//
// Polygons cut into triangles for drawing them with WebGL (ls.plot's large vector layers):
// mapbox's earcut (earcut/earcut.hpp, ISC), one polygon per thread.

#include <array>
#include <cstdint>
#include <vector>

#include "earcut/earcut.hpp"
#include "io/io.hpp"

namespace lsio {

std::vector<std::uint32_t> triangulate(const double* xy, const i64* ring_offsets, const i64* polygon_rings,
                                       i64 npolygons, std::vector<i64>* counts) {
    using Point = std::array<double, 2>;
    std::vector<std::vector<std::uint32_t>> parts(static_cast<std::size_t>(npolygons));
#pragma omp parallel for schedule(dynamic, 256)
    for (i64 p = 0; p < npolygons; ++p) {
        std::vector<std::vector<Point>> rings;
        for (i64 r = polygon_rings[p]; r < polygon_rings[p + 1]; ++r) {
            std::vector<Point> ring;
            ring.reserve(static_cast<std::size_t>(ring_offsets[r + 1] - ring_offsets[r]));
            for (i64 i = ring_offsets[r]; i < ring_offsets[r + 1]; ++i) ring.push_back({xy[2 * i], xy[2 * i + 1]});
            rings.push_back(std::move(ring));
        }
        parts[static_cast<std::size_t>(p)] = mapbox::earcut<std::uint32_t>(rings);
    }
    counts->assign(static_cast<std::size_t>(npolygons), 0);
    std::size_t total = 0;
    for (i64 p = 0; p < npolygons; ++p) {
        (*counts)[static_cast<std::size_t>(p)] = static_cast<i64>(parts[static_cast<std::size_t>(p)].size());
        total += parts[static_cast<std::size_t>(p)].size();
    }
    std::vector<std::uint32_t> out;
    out.reserve(total);
    for (auto& v : parts) out.insert(out.end(), v.begin(), v.end());
    return out;
}

}  // namespace lsio
