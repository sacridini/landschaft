// SPDX-License-Identifier: GPL-3.0-or-later
//
// terra::distance() of a raster, as nlm_distancegradient() calls it: on a
// planar raster terra runs GDALComputeProximity (GDAL 3.9, alg/
// gdalproximity.cpp), on a lon/lat raster its own haversine search
// (terra 1.9-50, distRaster.cpp, distance_crds_lonlat_fast). Both are ported
// operation by operation; the exact Euclidean distance is the documented
// behaviour.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"

namespace nlm {

namespace {

double square_distance(double x1, double x2, double y1, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return dx * dx + dy * dy;
}

// ProcessProximityLine() with every non-zero source pixel as a target and
// no nodata.
void process_proximity_line(const std::uint8_t* src, int* near_x, int* near_y, bool forward, int line, int nx,
                            double max_dist, float* prox) {
    const int start = forward ? 0 : nx - 1;
    const int end = forward ? nx : -1;
    const int step = forward ? 1 : -1;
    for (int ip = start; ip != end; ip += step) {
        if (src[ip] != 0) {
            prox[ip] = 0.0f;
            near_x[ip] = ip;
            near_y[ip] = line;
            continue;
        }
        const double m = std::max(max_dist, static_cast<double>(nx));
        double near_sq = m * m * 2.0;
        if (near_x[ip] != -1) {
            const double d = square_distance(near_x[ip], ip, near_y[ip], line);
            if (d < near_sq) {
                near_sq = d;
            } else {
                near_x[ip] = -1;
                near_y[ip] = -1;
            }
        }
        const int last = ip - step;
        if (ip != start && near_x[last] != -1) {
            const double d = square_distance(near_x[last], ip, near_y[last], line);
            if (d < near_sq) {
                near_sq = d;
                near_x[ip] = near_x[last];
                near_y[ip] = near_y[last];
            }
        }
        const int tr = ip + step;
        if (tr != end && near_x[tr] != -1) {
            const double d = square_distance(near_x[tr], ip, near_y[tr], line);
            if (d < near_sq) {
                near_sq = d;
                near_x[ip] = near_x[tr];
                near_y[ip] = near_y[tr];
            }
        }
        if (near_x[ip] != -1 && near_sq <= max_dist * max_dist &&
            (prox[ip] < 0 || near_sq < static_cast<double>(prox[ip]) * prox[ip]))
            prox[ip] = static_cast<float>(std::sqrt(near_sq));
    }
}

}  // namespace

std::vector<double> distance_gdal(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res) {
    // GDALComputeProximity(DISTUNITS=GEO): two scans with the nearest
    // target carried along, float32 distances, times |pixel width|.
    const int nx = static_cast<int>(ncol), ny = static_cast<int>(nrow);
    const double max_dist = static_cast<double>(nx) + static_cast<double>(ny);
    const float nodata = 65535.0f;
    std::vector<float> work(static_cast<std::size_t>(nrow * ncol));
    std::vector<int> near_x(static_cast<std::size_t>(nx), -1), near_y(static_cast<std::size_t>(nx), -1);
    for (int line = 0; line < ny; ++line) {
        float* prox = work.data() + static_cast<std::size_t>(line) * nx;
        const std::uint8_t* src = origin.data() + static_cast<std::size_t>(line) * nx;
        for (int i = 0; i < nx; ++i) prox[i] = -1.0f;
        process_proximity_line(src, near_x.data(), near_y.data(), true, line, nx, max_dist, prox);
        process_proximity_line(src, near_x.data(), near_y.data(), false, line, nx, max_dist, prox);
    }
    std::fill(near_x.begin(), near_x.end(), -1);
    std::fill(near_y.begin(), near_y.end(), -1);
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol));
    for (int line = ny - 1; line >= 0; --line) {
        float* prox = work.data() + static_cast<std::size_t>(line) * nx;
        const std::uint8_t* src = origin.data() + static_cast<std::size_t>(line) * nx;
        process_proximity_line(src, near_x.data(), near_y.data(), false, line, nx, max_dist, prox);
        process_proximity_line(src, near_x.data(), near_y.data(), true, line, nx, max_dist, prox);
        for (int i = 0; i < nx; ++i) {
            if (prox[i] < 0.0f)
                prox[i] = nodata;
            else if (prox[i] > 0.0f)
                prox[i] = static_cast<float>(prox[i] * res);
            out[static_cast<std::size_t>(line) * nx + i] = prox[i];
        }
    }
    return out;
}

std::vector<double> distance_haversine(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double xmin,
                                       double ymax, double xres, double yres) {
    // edges(inner, 8): origin cells with a non-origin cell among their 8
    // neighbours inside the raster (terra pads by repeating the border).
    std::vector<double> px, py;
    auto is_origin = [&](i64 r, i64 c) { return origin[static_cast<std::size_t>(r * ncol + c)] != 0; };
    for (i64 r = 0; r < nrow; ++r) {
        for (i64 c = 0; c < ncol; ++c) {
            if (!is_origin(r, c)) continue;
            bool edge = false;
            for (i64 dr = -1; dr <= 1 && !edge; ++dr)
                for (i64 dc = -1; dc <= 1 && !edge; ++dc) {
                    const i64 rr = r + dr, cc = c + dc;
                    if ((dr || dc) && rr >= 0 && rr < nrow && cc >= 0 && cc < ncol && !is_origin(rr, cc)) edge = true;
                }
            if (edge) {
                px.push_back(xmin + ((static_cast<double>(c) + 0.5) * xres));
                py.push_back(ymax - ((static_cast<double>(r) + 0.5) * yres));
            }
        }
    }
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol), 0.0);
    if (px.empty()) return out;

    const double D2R = 0.017453292519943295;
    const double EARTH_R = 6378137.0;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> pm(py.size());
    std::iota(pm.begin(), pm.end(), 0);
    std::sort(pm.begin(), pm.end(), [&](std::size_t i, std::size_t j) { return py[i] < py[j]; });
    const std::size_t np = px.size();
    std::vector<double> qx(np), qy(np), sin_qy(np), cos_qy(np);
    for (std::size_t j = 0; j < np; ++j) {
        qx[j] = px[pm[j]] * D2R;
        qy[j] = py[pm[j]] * D2R;
        sin_qy[j] = lsm::r_sin(qy[j]);
        cos_qy[j] = lsm::r_cos(qy[j]);
    }
    std::vector<double> tox(static_cast<std::size_t>(ncol));
    for (i64 c = 0; c < ncol; ++c) tox[static_cast<std::size_t>(c)] = (xmin + ((static_cast<double>(c) + 0.5) * xres)) * D2R;

    auto haversine = [&](double dlat, double lon_diff, double cpy, std::size_t j) {
        const double sLat = lsm::r_sin(dlat * 0.5);
        const double sLon = lsm::r_sin(lon_diff * 0.5);
        double a = sLat * sLat + cpy * cos_qy[j] * sLon * sLon;
        if (a < 0.0)
            a = 0.0;
        else if (a > 1.0)
            a = 1.0;
        return 2.0 * lsm::r_atan2(std::sqrt(a), std::sqrt(1.0 - a));
    };

    lsm::parallel_for_dynamic(nrow * ncol >= 4096, nrow, [&](i64 r) {
        const double pyr = (ymax - ((static_cast<double>(r) + 0.5) * yres)) * D2R;
        const double cpy = lsm::r_cos(pyr);
        const std::size_t mid = static_cast<std::size_t>(std::lower_bound(qy.begin(), qy.end(), pyr) - qy.begin());
        for (i64 c = 0; c < ncol; ++c) {
            double& d = out[static_cast<std::size_t>(r * ncol + c)];
            if (is_origin(r, c)) {
                d = 0.0;
                continue;
            }
            const double pxc = tox[static_cast<std::size_t>(c)];
            double cur = inf;
            for (std::size_t j = mid; j < np; ++j) {
                const double dlat = qy[j] - pyr;
                if (dlat >= cur) break;
                const double dd = haversine(dlat, qx[j] - pxc, cpy, j);
                if (dd < cur) cur = dd;
            }
            for (std::size_t jj = mid; jj > 0; --jj) {
                const std::size_t j = jj - 1;
                const double dlat = pyr - qy[j];
                if (dlat >= cur) break;
                const double dd = haversine(dlat, qx[j] - pxc, cpy, j);
                if (dd < cur) cur = dd;
            }
            d = cur * EARTH_R;
        }
    });
    return out;
}

std::vector<double> distance_exact(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res) {
    // Felzenszwalb-Huttenlocher squared distance transform: columns, then rows.
    const double big = std::numeric_limits<double>::infinity();
    std::vector<double> g(static_cast<std::size_t>(nrow * ncol));
    lsm::parallel_for(nrow * ncol >= 65536, ncol, [&](i64 c) {
        // 1-D: distance along the column to the nearest origin cell.
        double last = -big;
        for (i64 r = 0; r < nrow; ++r) {
            if (origin[static_cast<std::size_t>(r * ncol + c)]) last = static_cast<double>(r);
            g[static_cast<std::size_t>(r * ncol + c)] = static_cast<double>(r) - last;
        }
        last = big;
        for (i64 r = nrow - 1; r >= 0; --r) {
            if (origin[static_cast<std::size_t>(r * ncol + c)]) last = static_cast<double>(r);
            double& v = g[static_cast<std::size_t>(r * ncol + c)];
            v = std::min(v, last - static_cast<double>(r));
        }
    });
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol));
    lsm::parallel_for(nrow * ncol >= 65536, nrow, [&](i64 r) {
        std::vector<double> f(static_cast<std::size_t>(ncol)), z(static_cast<std::size_t>(ncol + 1));
        std::vector<i64> v(static_cast<std::size_t>(ncol));
        for (i64 c = 0; c < ncol; ++c) {
            const double gv = g[static_cast<std::size_t>(r * ncol + c)];
            f[static_cast<std::size_t>(c)] = std::isinf(gv) ? 1e300 : gv * gv;
        }
        // lower envelope of the parabolas f[p] + (q - p)^2
        i64 k = 0;
        v[0] = 0;
        z[0] = -big;
        z[1] = big;
        auto intersection = [&](i64 q, i64 p) {
            return ((f[static_cast<std::size_t>(q)] + static_cast<double>(q * q)) -
                    (f[static_cast<std::size_t>(p)] + static_cast<double>(p * p))) /
                   (2.0 * static_cast<double>(q - p));
        };
        for (i64 q = 1; q < ncol; ++q) {
            double s = intersection(q, v[static_cast<std::size_t>(k)]);
            while (s <= z[static_cast<std::size_t>(k)]) {
                --k;
                s = intersection(q, v[static_cast<std::size_t>(k)]);
            }
            ++k;
            v[static_cast<std::size_t>(k)] = q;
            z[static_cast<std::size_t>(k)] = s;
            z[static_cast<std::size_t>(k + 1)] = big;
        }
        k = 0;
        for (i64 q = 0; q < ncol; ++q) {
            while (z[static_cast<std::size_t>(k + 1)] < static_cast<double>(q)) ++k;
            const i64 p = v[static_cast<std::size_t>(k)];
            const double dq = static_cast<double>(q - p);
            const double d2 = dq * dq + f[static_cast<std::size_t>(p)];
            out[static_cast<std::size_t>(r * ncol + q)] = d2 >= 1e300 ? std::numeric_limits<double>::quiet_NaN()
                                                                      : std::sqrt(d2) * res;
        }
    });
    return out;
}

}  // namespace nlm
