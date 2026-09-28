// SPDX-License-Identifier: GPL-3.0-or-later
//
// The models NLMR computes in C++ with std::mt19937 (rcpp_mpd.cpp,
// rcpp_neigh.cpp, rcpp_randomrectangularcluster.cpp), with the draws of
// libstdc++ (rng.hpp) in the original order.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"

namespace nlm {

namespace {

inline i64 wrap(i64 v, i64 n) {
    i64 r = v % n;
    return r < 0 ? r + n : r;
}

// `n` deviates of fresh std::normal_distribution(0, sd) objects (each call
// keeps the first value of a polar pair). The generator and the acceptance
// test run in sequence; the log and sqrt of every pair in parallel.
void fresh_normals(StdMt19937& mt, double sd, std::vector<double>& out, std::size_t n) {
    out.resize(n);
    std::vector<double> r2s(n);
    for (std::size_t k = 0; k < n; ++k) {
        double x, y, r2;
        do {
            x = 2.0 * std_canonical(mt) - 1.0;
            y = 2.0 * std_canonical(mt) - 1.0;
            r2 = x * x + y * y;
        } while (r2 > 1.0 || r2 == 0.0);
        out[k] = y;
        r2s[k] = r2;
    }
    lsm::parallel_for(n >= 4096, static_cast<i64>(n), [&](i64 k) {
        const double r2 = r2s[static_cast<std::size_t>(k)];
        const double mult = std::sqrt(-2 * lsm::r_log(r2) / r2);
        out[static_cast<std::size_t>(k)] = out[static_cast<std::size_t>(k)] * mult * sd + 0.0;
    });
}

}  // namespace

// -- midpoint displacement ------------------------------------------------------------

std::vector<double> mpd_square(i64 size, const std::vector<double>& rand_dev, std::uint32_t seed, bool torus) {
    // mpd() of rcpp_mpd.cpp on a size x size map (map[a][b] stored at
    // a * size + b). Within a diamond or square step every cell reads only
    // cells of earlier steps (and itself), so the normal deviates are drawn
    // first, in the original order, and the cells computed in parallel. On a
    // torus the last square step can read cells of its own step: those steps
    // run in order.
    std::vector<double> map(static_cast<std::size_t>(size * size), 0.0);
    auto at = [size](i64 a, i64 b) { return static_cast<std::size_t>(a * size + b); };
    StdMt19937 mt(seed);
    const i64 n_steps = static_cast<i64>(std::ceil(std::log2(static_cast<double>(size - 1))));
    if (static_cast<i64>(rand_dev.size()) != n_steps) throw std::invalid_argument("mpd: wrong number of deviations");
    map[at(0, 0)] = std_uniform_real(mt, 0, 1);
    map[at(0, size - 1)] = std_uniform_real(mt, 0, 1);
    map[at(size - 1, 0)] = std_uniform_real(mt, 0, 1);
    map[at(size - 1, size - 1)] = std_uniform_real(mt, 0, 1);

    std::vector<double> noise;
    std::vector<std::pair<i64, i64>> cells;
    for (i64 step = 0; step < n_steps; ++step) {
        const i64 side = i64{1} << (n_steps - step);
        const i64 half = side / 2;
        const double sd = rand_dev[static_cast<std::size_t>(step)];
        // diamond step
        cells.clear();
        for (i64 col = half; col < size; col += side)
            for (i64 row = half; row < size; row += side) cells.emplace_back(col, row);
        fresh_normals(mt, sd, noise, cells.size());
        lsm::parallel_for(cells.size() >= 4096, static_cast<i64>(cells.size()), [&](i64 k) {
            const i64 col = cells[static_cast<std::size_t>(k)].first, row = cells[static_cast<std::size_t>(k)].second;
            const double self = map[at(col, row)];
            double a = self, b = self, c = self, d = self;
            if (col - half >= 0 && row - half >= 0) a = map[at(col - half, row - half)];
            if (col - half >= 0 && row + half < size) b = map[at(col - half, row + half)];
            if (col + half < size && row - half >= 0) c = map[at(col + half, row - half)];
            if (col + half < size && row + half < size) d = map[at(col + half, row + half)];
            map[at(col, row)] = (0.25 * (a + b + c + d)) + noise[static_cast<std::size_t>(k)];
        });
        // square step
        cells.clear();
        i64 parity = 0;
        for (i64 col = 0; col < size; col += half) {
            const i64 row_start = (parity % 2 == 0) ? half : 0;
            ++parity;
            for (i64 row = row_start; row < size; row += side) cells.emplace_back(col, row);
        }
        fresh_normals(mt, sd, noise, cells.size());
        auto square = [&](i64 k) {
            const i64 col = cells[static_cast<std::size_t>(k)].first, row = cells[static_cast<std::size_t>(k)].second;
            const double self = map[at(col, row)];
            double a = self, b = self, c = self, d = self;
            if (torus) {
                a = map[at(wrap(col - half, size), row)];
                b = map[at(col, wrap(row + half, size))];
                c = map[at(wrap(col + half, size), row)];
                d = map[at(col, wrap(row - half, size))];
            } else {
                if (col - half >= 0) a = map[at(col - half, row)];
                if (row + half < size) b = map[at(col, row + half)];
                if (col + half < size) c = map[at(col + half, row)];
                if (row - half >= 0) d = map[at(col, row - half)];
            }
            map[at(col, row)] = (0.25 * (a + b + c + d)) + noise[static_cast<std::size_t>(k)];
        };
        if (torus) {
            for (i64 k = 0; k < static_cast<i64>(cells.size()); ++k) square(k);
        } else {
            lsm::parallel_for(cells.size() >= 4096, static_cast<i64>(cells.size()), square);
        }
    }
    return map;
}

// -- neighbourhood clusters -----------------------------------------------------------------

std::vector<double> neigh(i64 nrow, i64 ncol, int n_categories, const std::vector<std::int64_t>& cells_per_cat,
                          int neighbourhood, float p_neigh, float p_empty, std::uint32_t seed) {
    // rcpp_neigh(): cells in a shuffled order; the last one takes category
    // `cat` with probability p_neigh when a neighbour (on a torus) holds a
    // non-zero category, p_empty otherwise, else it is swapped with a random
    // earlier one. Categories from n_categories down to 0.
    StdMt19937 mt(seed);
    std::vector<int> mat(static_cast<std::size_t>(nrow * ncol), 0);
    auto m = [&](i64 r, i64 c) -> int& { return mat[static_cast<std::size_t>(r * ncol + c)]; };
    std::vector<std::pair<int, int>> cell_index(static_cast<std::size_t>(nrow * ncol));
    std::size_t cntr = 0;
    for (i64 col = 0; col < ncol; ++col)
        for (i64 row = 0; row < nrow; ++row) cell_index[cntr++] = {static_cast<int>(col), static_cast<int>(row)};
    std_shuffle(cell_index.data(), static_cast<i64>(cell_index.size()), mt);
    const double pn = p_neigh, pe = p_empty;
    for (int cat = n_categories; cat >= 0; --cat) {
        std::int64_t n_cells = cells_per_cat[static_cast<std::size_t>(cat)];
        while (n_cells > 0) {
            if (cell_index.empty()) throw std::runtime_error("neigh: more cells requested than the raster has");
            const i64 col = cell_index.back().first;
            const i64 row = cell_index.back().second;
            const i64 up = wrap(row - 1, nrow), left = wrap(col - 1, ncol);
            const i64 right = wrap(col + 1, ncol), down = wrap(row + 1, nrow);
            unsigned adjacent = static_cast<unsigned>(m(up, col)) + static_cast<unsigned>(m(row, left)) +
                                static_cast<unsigned>(m(row, right)) + static_cast<unsigned>(m(down, col));
            if (adjacent == 0 && neighbourhood == 8) {
                adjacent += static_cast<unsigned>(m(up, left)) + static_cast<unsigned>(m(up, right)) +
                            static_cast<unsigned>(m(down, left)) + static_cast<unsigned>(m(down, right));
            }
            const double prob = adjacent > 0 ? pn : pe;
            const double rnd = std_uniform_real(mt, 0, 1);
            if (rnd < prob) {
                m(row, col) = cat;
                --n_cells;
                cell_index.pop_back();
            } else {
                const std::size_t back = cell_index.size() - 1;
                const std::size_t idx = static_cast<std::size_t>(std::floor(std_uniform_real(mt, 0, 1) * static_cast<double>(back)));
                std::swap(cell_index[idx], cell_index[back]);
            }
        }
    }
    return std::vector<double>(mat.begin(), mat.end());
}

// -- random rectangular clusters ----------------------------------------------------------

std::vector<double> random_rectangular_cluster(i64 nrow, i64 ncol, int minl, int maxl, std::uint32_t seed) {
    // rcpp_randomrectangularcluster(): rectangles of random size and value,
    // wrapped around the edges, until no cell is left empty. The original
    // scans the whole matrix after every rectangle; a count of the empty
    // cells gives the same stopping point.
    StdMt19937 mt(seed);
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol), -1.0);
    i64 empty = nrow * ncol;
    while (empty > 0) {
        const i64 row_pos = std_uniform_int(mt, 0, nrow - 1);
        const i64 col_pos = std_uniform_int(mt, 0, ncol - 1);
        const i64 width = std_uniform_int(mt, minl, maxl) + row_pos;
        const i64 height = std_uniform_int(mt, minl, maxl) + col_pos;
        const double land_use = std_uniform_real(mt, 0, 1);
        for (i64 row = row_pos; row < width; ++row) {
            const i64 r = wrap(row, nrow);
            for (i64 col = col_pos; col < height; ++col) {
                double& v = out[static_cast<std::size_t>(r * ncol + wrap(col, ncol))];
                if (v == -1.0) --empty;
                v = land_use;
            }
        }
    }
    return out;
}

}  // namespace nlm
