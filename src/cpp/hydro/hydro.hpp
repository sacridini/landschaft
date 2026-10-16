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

// The height of each cell above the stream cell it drains to along `dir` (Rennó et
// al. 2008): 0 on the streams, NaN where the cell reaches no stream.
std::vector<double> hand(const Grid& g, const std::int8_t* dir, const std::uint8_t* streams);

}  // namespace hydro
