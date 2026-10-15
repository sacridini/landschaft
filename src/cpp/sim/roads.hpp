// SPDX-License-Identifier: GPL-3.0-or-later
//
// Endogenous road building (Dinamica EGO's CalcCostMap, CalcDestinationMap and
// CalcPathwayMap, the road constructor of Soares-Filho et al. 2004), clean-room
// from their documentation and black-box runs of Dinamica EGO 8.13.
//
//  - friction_cost: the cost of reaching every cell from the nearest source,
//    a step into a cell costing its friction (x sqrt(2) on diagonals when
//    they cost more), 8 neighbours; Dijkstra (Dinamica's push-broom with as
//    many passes as needed gives the same sums, bit for bit). The direction
//    of each cell is its predecessor on a cheapest path (codes 1 NW, 2 N,
//    3 NE, 4 E, 5 SE, 6 S, 7 SW, 8 W; 0 at sources and where unreached),
//    ties to the first in row order of the 3 x 3 window (Dinamica's ties
//    follow its sweeps).
//  - destinations: the map cut into quadrants (columns into `horizontal`,
//    rows into `vertical` parts); in each, up to `per_quadrant` cells drawn
//    in proportion to their attraction (> 0), a draw closer than
//    `min_distance` to an earlier destination redrawn up to `max_attempts`
//    times. Dinamica gives each quadrant n / q cells (the remainder never
//    gets a destination) and rounds its draw to an integer of the summed
//    attraction (the first cell gets half its share, and a draw past the
//    end takes the quadrant's last cell, of no attraction): here the
//    quadrants cover the map and the draw is exact.
//  - pathways: from each source (row order) down the cost surface to the
//    network, each step to the cheapest of the 8 neighbours (ties: the
//    first in row order of the window, or one at random with `lottery`),
//    until a cell of cost <= 0; or following the directions until a cell
//    of the network or of direction 0.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct FrictionCost {
    std::vector<double> cost;       // NaN where friction is missing or unreachable
    std::vector<std::uint8_t> dir;  // predecessor codes (above)
};

// source: non-zero = source; friction: per cell (NaN = impassable); diagonal: the
// diagonal step's factor (sqrt(2) or 1).
FrictionCost friction_cost(std::int64_t ny, std::int64_t nx, const std::uint8_t* source, const double* friction,
                           double diagonal, bool directions);

// The destinations (cell indices, in the order drawn).
std::vector<std::int64_t> destinations(std::int64_t ny, std::int64_t nx, const double* attraction, int horizontal,
                                       int vertical, int per_quadrant, double min_distance, int max_attempts,
                                       double dx, double dy, std::uint64_t seed);

// New path cells (1) of the sources (non-zero, row order) towards the network
// (cost <= 0, or `network` non-zero with directions); `cost` or `dir` (the
// other null; direction 255 = missing). A source of missing cost or direction
// is skipped. Throws on a missing direction along a path, a local minimum of
// the costs or a loop of directions.
std::vector<std::uint8_t> pathways(std::int64_t ny, std::int64_t nx, const std::uint8_t* network,
                                   const std::uint8_t* sources, const double* cost, const std::uint8_t* dir,
                                   bool lottery, std::uint64_t seed);

}  // namespace sim
