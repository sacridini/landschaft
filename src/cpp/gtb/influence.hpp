// SPDX-License-Identifier: GPL-3.0-or-later
//
// GuidosToolbox "Distance -> Influence Zones" and "Distance -> Proximity",
// with the IDL routines they rely on (MORPH_DISTANCE, HIST_EQUAL, WATERSHED).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

// Metric of IDL's MORPH_DISTANCE(..., NEIGHBOR_SAMPLING=3) as used by GTB.
enum class DistanceMetric { Euclidean, QuasiEuclidean };

// IDL MORPH_DISTANCE(mask, /BACKGROUND): distance of every zero pixel to the
// nearest non-zero pixel (non-zero pixels: 0). Without `background`: distance
// of every non-zero pixel to the nearest zero pixel (zero pixels: 0). Pixels
// outside the image are ignored.
std::vector<float> morph_distance(const ImageU8& mask, bool background, DistanceMetric metric);

// IDL HIST_EQUAL of a float image with the default settings (MINV/MAXV = the
// image range, BINSIZE = (MAXV - MINV) / 5000, TOP = 255).
ImageU8 hist_equal(const std::vector<float>& im, i64 nx, i64 ny);

// IDL WATERSHED(image, CONNECTIVITY=graph, /LONG): Vincent & Soille (1991)
// immersion. Watershed lines are 0, catchment basins 1..nregions, numbered in
// the order their minima are reached (grey level, then raster order).
Image<std::int32_t> watershed(const ImageU8& im, int graph, std::int32_t& nregions);

// GTB viewport colours of the Influence Zones / Proximity maps.
constexpr std::array<u8, 12> kInfluenceObjectColors = {59, 5, 9, 17, 33, 65, 18, 56, 55, 57, 58, 101};
constexpr u8 kProximityObject = 12;
constexpr u8 kInfluenceLine = 4;
constexpr u8 kInfluenceHole = 0;
constexpr u8 kInfluenceNeglected = 176;
constexpr u8 kInfluenceMissing = 129;
constexpr u8 kInfluenceBufferEdge = 80;
constexpr u8 kInfluenceBuffer = 160;
constexpr u8 kProximityNear = 165;
constexpr u8 kProximityMin = 193;

struct ProximityLine {
    i64 x = 0, y = 0;             // GTB's 1-based column and row (row counted from the top)
    i64 nbr = 0;                  // number of adjacent zones
    i64 clength = 0;              // connector length (proximity distance)
    i64 cag = 0;                  // connected area gain
    std::vector<std::int32_t> ids;  // adjacent zone ids
    std::vector<i64> areas;       // their object areas
};

struct InfluenceResult {
    ImageU8 classes;                 // GTB viewport codes
    Image<std::int32_t> ids;         // object id; 0 zone, -1 hole, -2 neglected FG, -3 line, -4 missing, -5/-6 buffer
    Image<std::int32_t> zones;       // watershed zone labels (0 = line)
    Image<std::int32_t> distance;    // proximity only: round(2 * distance to the objects), updated on lines
    Image<std::int64_t> cag;         // proximity only: CAG on watershed pixels
    std::vector<i64> object_area;    // object_area[k] = area of object k without its holes (index 0: rest)
    std::int32_t n_objects = 0;
    // proximity only
    std::vector<ProximityLine> lines;
    i64 prox_min = -1;
    i64 prox_min_count = 0;
};

// GTB Influence Zones (proximity = false) or Proximity (proximity = true) of
// a map coded 0 missing, 1 background, 2 foreground. Objects are 8-connected
// foreground areas of at least `min_area` pixels (after removing a
// `core_buffer` pixel wide edge when core_buffer > 0), with their holes filled.
// `buffer` is the background buffer width (influence zones) or the maximum
// proximity to mark (proximity).
InfluenceResult compute_influence(const ImageU8& im, i64 min_area, int core_buffer, int buffer, bool proximity,
                                  DistanceMetric metric);

}  // namespace gtb
