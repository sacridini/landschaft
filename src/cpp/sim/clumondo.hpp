// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLUMondo (Van Asselen & Verburg 2013; github.com/VUEG/CLUMondo, GPL-3) as
// its Win32 build runs it: the land system allocation of one year
// (calc_change_ch, comp_demand, autonomous_change, calc_age), the logistic
// suitability (calc_reg) and the neighbourhood fractions (calc_neigh), in
// single precision where CLUMondo uses floats. exp() of a float is the
// double exp rounded to float (what the Win32 runtime's expf gives), and the
// iteration's random factor comes from the Microsoft runtime's rand().
//
// Cells carry CLUMondo's region codes: >= 0 inside the study area, -9998
// restricted (the system stays), -9999 no data.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

constexpr int kNoData = -9999;
constexpr int kRestricted = -9998;

struct ClumondoSettings {
    int nsys = 0, ndem = 0;
    std::vector<int> allowed;      // nsys x nsys (from, to): CLUMondo's codes; 2..99 = allow map (code - 2)
    std::vector<int> lusconv;      // nsys x ndem: rank of each system for each demand (-1: not part of it)
    std::vector<float> lusmatrix;  // nsys x ndem: supply of one cell
    std::vector<float> weight;     // ndem: weight of each demand in the iteration variables
    std::vector<float> resistance; // nsys: conversion resistance (ltstat)
    int itermode = 0;              // 0: deviations in % of the demand; 1: absolute
    float mean_dev = 0, max_dev = 0;
    float speed = 0.05f;           // starting step of the iteration (main.1, line 24)
    int max_loops = 20000;
    // Extensions of the 2017 executable (not in the source; from the manual, see clumondo.cpp)
    std::vector<int> kind;         // ndem: 0 exact, 1 minimum (overshoot accepted), 2 maximum (undershoot)
    std::vector<float> dev_limit;  // ndem: deviation (%) allowed per demand (iteration mode 2), or empty
    std::int64_t nx = 0;           // columns of the map (patches)
    std::vector<int> patch_radius; // nsys: patch radius (0: no patches), or empty
    std::vector<float> patch_threshold;
    std::vector<int> patch_min;
    // Extensions of ls.simulate (land system roles)
    std::vector<float> pair_resistance;  // nsys x nsys (from, to): subtracted when a cell changes, or empty
    std::vector<std::vector<unsigned char>> cap_pairs;  // per cap: nsys x nsys transitions it limits
    std::vector<std::int64_t> cap_cells;                 // per cap: most cells per year
};

struct ClumondoYear {
    std::vector<int> newco;        // the system of each cell (-9999 no data)
    std::vector<int> age;          // years each cell has had its system
    std::vector<double> supply;    // ndem: supplied at the end of the iteration
    std::vector<std::int64_t> cells;  // nsys
    int loops = 0;
    bool converged = false;
    float max_diff = 0, mean_diff = 0;
};

// One year: oldco and age at the start, suit (nsys x ncell, the suitability
// with neighbourhood and location additions), allow_maps (nmap x ncell),
// demand (ndem) for the year, dem_elas (ndem, the iteration variables, kept
// from year to year). Throws when a cell has no allowed system.
ClumondoYear clumondo_year(const ClumondoSettings& s, std::int64_t ncell, const int* oldco, const int* region,
                           const int* age, const float* suit, const float* allow_maps, int nmap,
                           const double* demand, float* dem_elas);

// calc_reg: p = e / (1 + e), e = exp(c + sum(b_m x_m)) in floats (the
// constant rounded to float, the terms added in order); -9999 where region
// < 0. Throws when the linear predictor exceeds 87 (CLUMondo stops there).
std::vector<float> clumondo_logit(std::int64_t ncell, const int* region, const float* x, int nx, double constant,
                                  const float* coef);

// calc_neigh: the weighted share of the kernel (side 2r + 1, weights over
// cells inside the map and not no data) in system u; -9999 at no data; 0
// where the weights add up to 0.
std::vector<float> clumondo_fraction(std::int64_t ny, std::int64_t nx, const int* oldco, const int* region, int u,
                                     const float* kernel, int radius);

// The ages at the start: rand() % max_age + 1 for every cell in row order,
// from the runtime generator seeded with seed (CLUMondo's default: 1).
std::vector<int> clumondo_random_age(std::int64_t ncell, int max_age, std::uint32_t seed);

}  // namespace sim
