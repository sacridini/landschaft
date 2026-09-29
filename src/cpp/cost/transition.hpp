// SPDX-License-Identifier: GPL-3.0-or-later
//
// gdistance's transition matrices on a raster grid. A transition holds one
// value per cell and direction, values[d * ncell + cell]: the conductance (or
// resistance) of the connection from the cell to its neighbour in direction d;
// 0 = no connection. Symmetric transitions keep only the directions after
// (0, 0) in (row, column) order, the value of the pair {a, b}; asymmetric ones
// all directions, T[cell, neighbour]. Directions: 4 (rook), 8 (queen) or 16
// (queen plus knight moves), as raster::adjacent; on a global lon/lat grid
// the columns wrap around the 180th meridian, as there.
#pragma once

#include <cstdint>
#include <vector>

namespace cost {

using i64 = std::int64_t;

struct Offset {
    int dr, dc;
};

struct Grid {
    i64 ny = 0, nx = 0;
    double xmin = 0, xmax = 1, ymin = 0, ymax = 1;
    bool wrap = false;  // columns wrap around (global lon/lat)

    i64 ncell() const { return ny * nx; }
    double xres() const { return (xmax - xmin) / static_cast<double>(nx); }
    double yres() const { return (ymax - ymin) / static_cast<double>(ny); }
    // Cell centres as raster's xyFromCell computes them.
    double x(i64 col) const { return (static_cast<double>(col) + 0.5) * xres() + xmin; }
    double y(i64 row) const { return ymax - (static_cast<double>(row) + 0.5) * yres(); }
    // The neighbour of cell (r, c) at offset o, or -1.
    i64 neighbour(i64 r, i64 c, Offset o) const {
        const i64 r2 = r + o.dr;
        if (r2 < 0 || r2 >= ny) return -1;
        i64 c2 = c + o.dc;
        if (c2 < 0 || c2 >= nx) {
            if (!wrap) return -1;
            c2 = ((c2 % nx) + nx) % nx;
        }
        return r2 * nx + c2;
    }
};

// The directions stored for (directions, symmetric), in (row, column) order.
std::vector<Offset> offsets(int directions, bool symmetric);

enum class Builtin { Mean, Min, Max, Sum, From, To, Diff, Harmonic, Geometric, Product };

// values (n_directions x ncell) of f(x[a], x[b]) for neighbouring cells with
// finite values: symmetric: a the lower cell index; asymmetric: a the cell,
// b its neighbour. R's arithmetic (mean(), sum() in long double...).
void builtin_values(const double* x, const Grid& g, int directions, bool symmetric, Builtin f, double* out);

// The neighbour cell of every (direction, cell) of the layout, -1 where there
// is none (out n_directions x ncell).
void neighbours(const Grid& g, int directions, bool symmetric, i64* out);

// gdistance's geoCorrection: values divided (conductance) or multiplied
// (resistance) by the distance between the cell centres (great-circle on
// lon/lat grids), over `scale`; with type 'r' on lon/lat grids the north-south
// component is scaled by the cosine of the mean latitude too. factors: write
// the correction factors (at the connections) instead of the corrected values.
void geo_correction(const Grid& g, int directions, bool symmetric, bool lonlat, char type, bool resistance,
                    bool scale, bool factors, double* values);

// R's scaleValue of geoCorrection(scl = TRUE): the distance from the centre of
// the extent to a point one cell to the east.
double scale_value(const Grid& g, bool lonlat);

// Row (margin 0) or column (margin 1) sums of the transition matrix as the
// Matrix package adds them (double, in increasing index order); with `count`
// the number of connections instead.
void margin_sums(const Grid& g, int directions, bool symmetric, const double* values, int margin, bool count,
                 double* out);

// Diagonal-free sparse matrix of the transition in compressed columns (both
// triangles), rows sorted: p (ncell + 1), i, x.
struct Csc {
    std::vector<i64> p, i;
    std::vector<double> x;
};
Csc to_csc(const Grid& g, int directions, bool symmetric, const double* values);

}  // namespace cost
