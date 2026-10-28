// SPDX-License-Identifier: GPL-3.0-or-later
//
// Terrain hydrology: depression filling, flow directions (D8, D-infinity, multiple
// flow directions), flow accumulation and the height above the nearest drainage.
//
// Grids are row-major, ny rows by nx columns, the first row on top; NaN marks a
// missing cell. Neighbours are numbered as in WhiteboxTools: 0 NE, 1 E, 2 SE, 3 S,
// 4 SW, 5 W, 6 NW, 7 N (the parity reference), and the first of equal slopes wins.
#pragma once

#include <cstdint>
#include <vector>

namespace hydro {

using i64 = std::int64_t;

inline constexpr int DCOL[8] = {1, 1, 1, 0, -1, -1, -1, 0};
inline constexpr int DROW[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

// D8 codes of a direction raster.
inline constexpr std::int8_t NO_FLOW = -1;   // a pit, a flat or a cell draining off the map
inline constexpr std::int8_t MISSING = -2;

struct Grid {
    const double* z = nullptr;
    int ny = 0, nx = 0;
    double dx = 1.0, dy = 1.0;   // cell width and height
};

// Priority-Flood (Barnes, Lehman & Mulla 2014): every depression raised to its spill
// elevation, the outlets being the cells on the map's edge or next to a missing cell.
// epsilon: Priority-Flood+epsilon, each filled cell raised to the next representable
// value above the cell it was reached from, so that filled areas drain.
std::vector<double> fill(const Grid& g, bool epsilon);

// Least-cost depression breaching (WhiteboxTools' BreachDepressionsLeastCost, after
// Lindsay & Dhun 2015): each pit, from the lowest, raised to just below its lowest
// neighbour, then joined to the nearest cell it can drain to (lower than the pit along
// the channel, missing, or off the map) by the channel of least cost within max_length
// cells, lowered to descend by small_num per cell. min_dist: the cost counts the
// channel's length (cost-distance), otherwise only the lowering. Pits left unsolved
// are filled when fill is set. small_num <= 0: chosen from the elevations as
// WhiteboxTools does.
std::vector<double> breach(const Grid& g, int max_length, double max_cost, bool min_dist, double small_num,
                           bool fill_rest);

// D8 (O'Callaghan & Mark 1984): the neighbour of steepest descent (drop / distance),
// NO_FLOW where none is lower. resolve_flats: flat areas drain by Barnes, Lehman &
// Mulla (2014), towards lower terrain and away from higher terrain.
std::vector<std::int8_t> d8(const Grid& g, bool resolve_flats, int n_threads);

// The directions of the flat cells only (Barnes et al. 2014), NO_FLOW elsewhere.
std::vector<std::int8_t> flat_directions(const Grid& g, const std::vector<std::int8_t>& dir);

// Accumulations: each cell starts with its weight (1 without weights) and passes it
// on downstream. Cells are taken in the order WhiteboxTools takes them (a stack seeded
// with the cells without inflow, row by row), so the sums are the same.
std::vector<double> d8_accumulation(const std::int8_t* dir, const double* weight, int ny, int nx);

// D-infinity (Tarboton 1997, as WhiteboxTools computes it): the azimuth of steepest
// descent in degrees clockwise from north, -1 where none, NaN where missing. Flat cells
// take the direction of `flats` (a D8 direction, NO_FLOW elsewhere) when given.
std::vector<double> dinf(const Grid& g, const std::int8_t* flats, int n_threads);
std::vector<double> dinf_accumulation(const double* angle, const double* weight, int ny, int nx);

// Multiple flow directions: each lower neighbour gets slope^exponent (times its contour
// length with `quinn`: 0.5 cardinal, 0.354 diagonal, Quinn et al. 1991) of the flow.
// Flat cells pass all to their `flats` direction when given.
std::vector<double> mfd_accumulation(const Grid& g, double exponent, bool quinn, const std::int8_t* flats,
                                     const double* weight);

// pygeoprocessing's multiple flow directions (the InVEST models' routing): each cell's
// flow split among its lower neighbours in proportion to the drop (over sqrt 2 on the
// diagonals; the cell size plays no part), plateaus draining to their nearest outlets,
// packed as eight 4-bit weights (0 E, 1 NE, 2 N, ... 7 SE; 0: no flow). block: the
// block size pygeoprocessing iterates the DEM by (the order plateaus are searched in).
std::vector<std::int32_t> mfd_directions(const Grid& g, int block);
std::vector<double> mfd_packed_accumulation(const std::int32_t* dir, const double* weight, int ny, int nx);
// The mean distance (in cells, or the sum of the weights of the cells passed) along the
// flow to a channel cell; NaN where no flow reaches one.
std::vector<double> mfd_distance_to_channel(const std::int32_t* dir, const std::uint8_t* channel,
                                            const double* weight, int ny, int nx);

// pygeoprocessing's stream extraction from the MFD accumulation: the cells of at least
// threshold traced up from the outlets, with tributaries of trace_proportion x threshold
// kept where they join a stream downstream. 1 stream, 0 not, 255 missing.
std::vector<std::uint8_t> mfd_streams(const double* acc, const std::int32_t* dir, int ny, int nx, double threshold,
                                      double trace_proportion, int block);
// pygeoprocessing's D8: 0 E, 1 NE, ... 7 SE, 128 missing; plateaus drain to their
// nearest outlets. Its accumulation (pulled from upstream) and distance to channel
// (the weight of the cell entered upstream, or 1 / sqrt 2).
std::vector<std::uint8_t> pgp_d8_directions(const Grid& g, int block);
std::vector<double> pgp_d8_accumulation(const std::uint8_t* dir, const double* weight, int ny, int nx);
std::vector<double> pgp_d8_distance_to_channel(const std::uint8_t* dir, const std::uint8_t* channel,
                                               const double* weight, int ny, int nx);

// The height of each cell above the stream cell it drains to along `dir` (Rennó et
// al. 2008): 0 on the streams, NaN where the cell reaches no stream.
std::vector<double> hand(const Grid& g, const std::int8_t* dir, const std::uint8_t* streams);

// The links of a stream network along D8 directions: the stretches between a source, a
// junction or a break cell (an outlet the link must end at) and the next. link: the link
// of each cell (1..n, 0 off the streams); start/end: its first and last cell (raster
// order of the starts); down: the link the last cell drains into (0: none).
struct StreamLinks {
    std::vector<std::int32_t> link;
    std::vector<i64> start, end;
    std::vector<std::int32_t> down;
};
StreamLinks stream_links(const std::int8_t* dir, const std::uint8_t* stream, const std::uint8_t* breaks, int ny,
                         int nx);

// Each cell labelled with the first non-zero seed found along its D8 path (seed cells
// keep their own), 0 where the path meets none.
std::vector<std::int32_t> watersheds(const std::int8_t* dir, const std::int32_t* seeds, int ny, int nx);

// The length of the D8 path from each cell to the first target cell (0 on targets, NaN
// where the path meets none).
std::vector<double> flow_distance(const std::int8_t* dir, const std::uint8_t* targets, int ny, int nx, double dx,
                                  double dy);

// Rivers as a network, on `down` (the cell each cell drains to, -1 none) and `step` (the
// length of the step from each cell to it).
//
// The distance along the network between every two points (cells; -1 off the network),
// the pairs (i, j > i) in a row: down from each to the first cell their paths share (NaN
// and connected 0 when none), flow 1 when that cell is one of theirs.
struct RiverDistances {
    std::vector<double> network;
    std::vector<std::uint8_t> connected, flow;
};
RiverDistances river_distances(const i64* down, const double* step, i64 n, const i64* cells, i64 p);

// The stretches of river within `downstream` down the path of each point and within
// `upstream` up the stream cells (`on`) flowing into it: each cell's nearest point along
// the network (1.., 0 none) and its distance (infinity none).
void along_network(const i64* down, const double* step, const std::uint8_t* on, i64 n, const i64* cells, i64 p,
                   double upstream, double downstream, std::int32_t* label, double* dist);

}  // namespace hydro
