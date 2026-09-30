// SPDX-License-Identifier: GPL-3.0-or-later
//
// The spatial parameters (geodiv's aacf.R, fourier.R) and fractal dimension
// (its sfd.cpp), with what they call:
// - R's fft() (nlm::r_fft2), e1071's hanning.window (R's cos: the x87 fcos
//   of mingw-w64), geodiv's fftshift;
// - terra 1.9-50's crop() (align with snap "near", origin, setExtent keeping
//   the resolution, colFromX / rowFromY) for the square without NA around the
//   centre that geodiv looks for.
// Corrected where geodiv errs: the images' upper half includes the row of zero
// lag / frequency (row floor(N/2), column floor(M/2), where fftshift puts it;
// geodiv's leaves it out and std's angles start one column off), distances
// are exact in map units (geodiv: GDAL's proximity in single precision, with
// the cell width in both directions; on longitude/latitude rasters terra's
// haversine, kept), std leaves out the zero frequency and keeps to the largest
// circle within the half image, and sfd has its typo, extremes and indices
// fixed.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "nlm/rfft.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double Inf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793238462643383279502884;
using Index = std::size_t;

// terra's is_equal (math_utils.cpp).
bool is_equal(double a, double b, double tolerance = 10.0) {
    const double tol = std::max(tolerance, std::fabs(std::min(a, b))) * std::numeric_limits<double>::epsilon();
    return (a == b) || (std::fabs(a - b) < tol);
}

// terra's SpatRaster::origin().
void origin(const Raster& r, double& ox, double& oy) {
    const double rx = r.xres(), ry = r.yres();
    ox = r.xmin - rx * (std::round(r.xmin / rx));
    oy = r.ymax - ry * (std::round(r.ymax / ry));
    if (is_equal(rx + ox, std::fabs(ox))) ox = std::fabs(ox);
    if (is_equal(ry + oy, std::fabs(oy))) oy = std::fabs(oy);
}

// terra's colFromX / rowFromY (-1 outside).
i64 col_from_x(const Raster& r, double x) {
    if (x >= r.xmin && x < r.xmax) return static_cast<i64>(std::trunc((x - r.xmin) / r.xres()));
    if (x == r.xmax) return r.nx - 1;
    return -1;
}
i64 row_from_y(const Raster& r, double y) {
    if (y > r.ymin && y <= r.ymax) return static_cast<i64>(std::trunc((r.ymax - y) / r.yres()));
    if (y == r.ymin) return r.ny - 1;
    return -1;
}

// Where terra::crop(r, ext(xmin, xmax, ymin, ymax)) with snap = "near" puts
// the result: its extent and size, and the first row and column of r it reads.
struct Window {
    double xmin, xmax, ymin, ymax;
    int nr, nc;
    i64 row1, col1;
    bool same;  // the whole raster: terra returns a copy (with r's extent)
};

Window crop_window(const Raster& r, double xmin, double xmax, double ymin, double ymax) {
    // e.intersect(ein)
    const double ex0 = std::max(xmin, r.xmin), ex1 = std::min(xmax, r.xmax);
    const double ey0 = std::max(ymin, r.ymin), ey1 = std::min(ymax, r.ymax);
    if (!(ex1 > ex0 && ey1 > ey0)) throw std::invalid_argument("extents do not overlap");
    // setExtent(e, keepRes = true, expand = false, "near"): align
    const double rx = r.xres(), ry = r.yres();
    double ox, oy;
    origin(r, ox, oy);
    double xmn = std::round((ex0 - ox) / rx) * rx + ox;
    double xmx = std::round((ex1 - ox) / rx) * rx + ox;
    double ymn = std::round((ey0 - oy) / ry) * ry + oy;
    double ymx = std::round((ey1 - oy) / ry) * ry + oy;
    if (xmn == xmx) {
        if (xmn < ex0) xmx = xmx + rx;
        else xmn = xmn - rx;
    }
    if (ymn == ymx) {
        if (ymn < ey0) ymx = ymx + ry;
        else ymn = ymn - ry;
    }
    xmn = std::max(xmn, r.xmin);
    xmx = std::min(xmx, r.xmax);
    ymn = std::max(ymn, r.ymin);
    ymx = std::min(ymx, r.ymax);
    Window w{};
    w.nc = static_cast<int>(std::max(1.0, std::round((xmx - xmn) / rx)));
    w.nr = static_cast<int>(std::max(1.0, std::round((ymx - ymn) / ry)));
    w.xmin = xmn;
    w.ymin = ymn;
    w.xmax = xmn + w.nc * rx;
    w.ymax = ymn + w.nr * ry;
    const double hxr = rx / 2, hyr = ry / 2;
    w.col1 = col_from_x(r, w.xmin + hxr);
    w.row1 = row_from_y(r, w.ymax - hyr);
    const i64 col2 = col_from_x(r, w.xmax - hxr), row2 = row_from_y(r, w.ymin + hyr);
    w.same = w.row1 == 0 && row2 == r.ny - 1 && w.col1 == 0 && col2 == r.nx - 1;
    return w;
}

// terra::crop(r, ext(xmin, xmax, ymin, ymax)) with snap = "near".
Raster crop(const Raster& r, double xmin, double xmax, double ymin, double ymax) {
    const Window w = crop_window(r, xmin, xmax, ymin, ymax);
    if (w.same) return r;
    Raster out;
    out.ny = w.nr;
    out.nx = w.nc;
    out.xmin = w.xmin;
    out.xmax = w.xmax;
    out.ymin = w.ymin;
    out.ymax = w.ymax;
    out.lonlat = r.lonlat;
    out.v.assign(static_cast<Index>(w.nr) * static_cast<Index>(w.nc), NaN);
    for (int i = 0; i < w.nr; ++i)
        for (int j = 0; j < w.nc; ++j) {
            const i64 rr = w.row1 + i, cc = w.col1 + j;
            if (rr >= 0 && rr < r.ny && cc >= 0 && cc < r.nx)
                out.v[static_cast<Index>(i) * w.nc + j] = r.v[static_cast<Index>(rr * r.nx + cc)];
        }
    return out;
}

// geodiv's search for the largest square without NA around the centre
// ("potentials"): the raster cropped to it, or unchanged (without NA, or
// when there is none). The NA in each candidate crop are counted from a
// summed-area table instead of copying it.
Raster na_crop(const Raster& x) {
    bool any = false;
    for (double v : x.v) any |= std::isnan(v);
    if (!any) return x;
    // origin: the mean of the cell centres' coordinates (R's mean)
    const i64 n = static_cast<i64>(x.ny) * x.nx;
    std::vector<double> cx(static_cast<Index>(n)), cy(static_cast<Index>(n));
    for (int r = 0; r < x.ny; ++r)
        for (int c = 0; c < x.nx; ++c) {
            cx[static_cast<Index>(r) * x.nx + c] = x.x(c);
            cy[static_cast<Index>(r) * x.nx + c] = x.y(r);
        }
    const double ox = lsm::r_mean(cx.data(), n), oy = lsm::r_mean(cy.data(), n);
    const double rx = x.xres(), ry = x.yres();
    // summed-area table of the NA cells
    const Index W = static_cast<Index>(x.nx) + 1;
    std::vector<i64> sat(W * (static_cast<Index>(x.ny) + 1), 0);
    for (int r = 0; r < x.ny; ++r)
        for (int c = 0; c < x.nx; ++c)
            sat[(r + 1) * W + c + 1] = sat[r * W + c + 1] + sat[(r + 1) * W + c] - sat[r * W + c] +
                                       (std::isnan(x.v[static_cast<Index>(r) * x.nx + c]) ? 1 : 0);
    auto na_in = [&](const Window& w) -> i64 {
        const i64 r0 = w.row1, c0 = w.col1, r1 = w.row1 + w.nr, c1 = w.col1 + w.nc;  // [r0, r1) x [c0, c1)
        const i64 cr0 = std::max<i64>(r0, 0), cc0 = std::max<i64>(c0, 0);
        const i64 cr1 = std::min<i64>(r1, x.ny), cc1 = std::min<i64>(c1, x.nx);
        const i64 inside = cr1 > cr0 && cc1 > cc0 ? (cr1 - cr0) * (cc1 - cc0) : 0;
        const i64 outside = static_cast<i64>(w.nr) * w.nc - inside;  // read as NA
        if (inside == 0) return outside;
        const i64 na = sat[cr1 * W + cc1] - sat[cr0 * W + cc1] - sat[cr1 * W + cc0] + sat[cr0 * W + cc0];
        return na + outside;
    };
    int best = -1;
    for (int k = 1; k <= x.ny / 2; ++k) {
        const Window w = crop_window(x, ox - rx * k, ox + rx * k, oy - ry * k, oy + ry * k);
        if ((w.same ? sat.back() : na_in(w)) == 0) best = k;
    }
    if (best < 0) return x;
    return crop(x, ox - rx * best, ox + rx * best, oy - ry * best, oy + ry * best);
}

// geodiv's fftshift of an ny x nx matrix (row order): the rows rotated up by
// ceiling(ny / 2), the columns left by ceiling(nx / 2).
std::vector<double> fftshift(const std::vector<double>& m, int ny, int nx) {
    const int rh = (ny + 1) / 2, ch = (nx + 1) / 2;
    std::vector<double> out(m.size());
    for (int r = 0; r < ny; ++r) {
        const int sr = (r + rh) % ny;
        for (int c = 0; c < nx; ++c) out[static_cast<Index>(r) * nx + c] = m[static_cast<Index>(sr) * nx + (c + ch) % nx];
    }
    return out;
}

// R's fft() of an ny x nx matrix given in row order (real, or interleaved
// complex with `complex_in`): interleaved complex, column-major.
std::vector<double> r_fft(const std::vector<double>& m, int ny, int nx, bool complex_in, bool inverse) {
    std::vector<double> z(static_cast<Index>(ny) * nx * 2);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) {
            const Index k = (static_cast<Index>(c) * ny + r) * 2;
            if (complex_in) {
                const Index s = (static_cast<Index>(r) * nx + c) * 2;
                z[k] = m[s];
                z[k + 1] = m[s + 1];
            } else {
                z[k] = m[static_cast<Index>(r) * nx + c];
                z[k + 1] = 0.0;
            }
        }
#if defined(_OPENMP)
    const bool parallel = !omp_in_parallel();  // not inside the moving windows' threads
#else
    const bool parallel = false;
#endif
    nlm::r_fft2(z, ny, nx, inverse, parallel);
    return z;
}

// The value (row order) of column-major complex element (r, c).
inline double re(const std::vector<double>& z, int ny, int r, int c) {
    return z[(static_cast<Index>(c) * ny + r) * 2];
}
inline double im(const std::vector<double>& z, int ny, int r, int c) {
    return z[(static_cast<Index>(c) * ny + r) * 2 + 1];
}

// e1071's hanning.window(n): 0.5 - 0.5 cos(2 pi k / (n - 1)).
std::vector<double> hanning(int n) {
    if (n == 1) return {1.0};
    const int m = n - 1;
    std::vector<double> w(static_cast<Index>(n));
    for (int k = 0; k <= m; ++k) w[static_cast<Index>(k)] = 0.5 - 0.5 * lsm::r_cos(2 * kPi * k / m);
    return w;
}

// The upper half of an image with the zero row: rows 0 .. floor(ny / 2).
Raster upper_half(const Raster& r) {
    const int nr = r.ny / 2 + 1;
    const double yr = (r.ymax - r.ymin) / r.ny;
    Raster out = r;
    out.ny = nr;
    out.ymin = r.ymax - nr * yr;
    out.v.assign(r.v.begin(), r.v.begin() + static_cast<std::ptrdiff_t>(nr) * r.nx);
    return out;
}

// The distance of every cell to cell (tr, tc): in map units on projected
// rasters, terra's haversine (metres) on longitude/latitude ones.
std::vector<double> distance_to(const Raster& g, int tr, int tc) {
    std::vector<double> d(static_cast<Index>(g.ny) * g.nx);
    if (!g.lonlat) {
        const double xr = g.xres(), yr = g.yres();
        const bool scalar = d.size() == 1;  // R's sqrt of one value
        for (int r = 0; r < g.ny; ++r)
            for (int c = 0; c < g.nx; ++c) {
                const double dx = static_cast<double>(c - tc) * xr, dy = static_cast<double>(r - tr) * yr;
                d[static_cast<Index>(r) * g.nx + c] = lsm::r_sqrt((dx * dx) + (dy * dy), scalar);
            }
        return d;
    }
    // distance_crds_lonlat_fast, haversine
    const double D2R = 0.017453292519943295, EARTH_R = 6378137.0;
    const double qx = g.x(tc) * D2R, qy = g.y(tr) * D2R;
    const double cos_qy = lsm::r_cos(qy);
    for (int r = 0; r < g.ny; ++r) {
        const double py = g.y(r) * D2R;
        const double cpy = lsm::r_cos(py);
        for (int c = 0; c < g.nx; ++c) {
            if (r == tr && c == tc) {
                d[static_cast<Index>(r) * g.nx + c] = 0.0;
                continue;
            }
            const double px = g.x(c) * D2R;
            const double dlat = qy >= py ? qy - py : py - qy;
            const double sLat = lsm::r_sin(dlat * 0.5);
            const double sLon = lsm::r_sin((qx - px) * 0.5);
            double a = sLat * sLat + cpy * cos_qy * sLon * sLon;
            if (a < 0.0) a = 0.0;
            else if (a > 1.0) a = 1.0;
            const double dd = 2.0 * lsm::r_atan2(std::sqrt(a), std::sqrt(1.0 - a));
            d[static_cast<Index>(r) * g.nx + c] = std::min(Inf, dd) * EARTH_R;
        }
    }
    return d;
}

// The raster of a grid.
Raster raster_of(const Grid& g, bool lonlat) {
    Raster r;
    r.v.assign(g.z, g.z + g.size());
    r.ny = g.ny;
    r.nx = g.nx;
    r.xmin = g.xmin;
    r.xmax = g.xmax;
    r.ymin = g.ymin;
    r.ymax = g.ymax;
    r.lonlat = lonlat;
    return r;
}

// The fft amplitude image of std / srw: fftshift(|fft(z)|) on the raster
// (without its NA, see na_crop), and its upper half.
Raster amplitude_half(const Raster& x) {
    const int ny = x.ny, nx = x.nx;
    const std::vector<double> ft = r_fft(x.v, ny, nx, false, false);
    std::vector<double> reim(static_cast<Index>(ny) * nx * 2);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) {
            reim[(static_cast<Index>(r) * nx + c) * 2] = re(ft, ny, r, c);
            reim[(static_cast<Index>(r) * nx + c) * 2 + 1] = im(ft, ny, r, c);
        }
    // fftshift of the complex values, then sqrt(Re^2 + Im^2) (R's vector sqrt)
    std::vector<double> amp(static_cast<Index>(ny) * nx);
    const int rh = (ny + 1) / 2, ch = (nx + 1) / 2;
    const bool scalar = amp.size() == 1;
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) {
            const Index s = (static_cast<Index>((r + rh) % ny) * nx + (c + ch) % nx) * 2;
            const double a = reim[s], b = reim[s + 1];
            amp[static_cast<Index>(r) * nx + c] = lsm::r_sqrt((a * a) + (b * b), scalar);
        }
    Raster img = x;
    img.v = std::move(amp);
    return upper_half(img);
}

// R's sum() of the values at the cells, in cell order, into accumulators.
struct Sums {
    std::vector<lsm::Extended> s;
    explicit Sums(Index n) : s(n, lsm::Extended(0.0)) {}
    void add(Index k, double v) {
        if (!std::isnan(v)) s[k] += v;
    }
    double operator[](Index k) const { return static_cast<double>(s[k]); }
};

}  // namespace

Raster aacf_image(const Grid& g, bool lonlat) {
    Raster x = na_crop(raster_of(g, lonlat));
    const int ny = x.ny, nx = x.nx;
    bool all_na = true;
    for (double v : x.v) all_na &= std::isnan(v);
    if (all_na) {
        std::fill(x.v.begin(), x.v.end(), NaN);
        return x;
    }
    const std::vector<double> wc = hanning(nx), wr = hanning(ny);
    std::vector<double> zw(x.v.size());
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c)
            zw[static_cast<Index>(r) * nx + c] = x.v[static_cast<Index>(r) * nx + c] * (wc[static_cast<Index>(c)] * wr[static_cast<Index>(r)]);
    const std::vector<double> ft = r_fft(zw, ny, nx, false, false);
    const double mn = static_cast<double>(nx) * ny;
    // ps = abs(ft)^2 / (M N) (R's abs of a complex: hypot)
    std::vector<double> ps(static_cast<Index>(ny) * nx);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) {
            const double a = std::hypot(re(ft, ny, r, c), im(ft, ny, r, c));
            ps[static_cast<Index>(r) * nx + c] = (a * a) / mn;
        }
    // Re(fft(ps, inverse = TRUE) / (M N))
    const std::vector<double> back = r_fft(ps, ny, nx, false, true);
    std::vector<double> af(static_cast<Index>(ny) * nx);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) af[static_cast<Index>(r) * nx + c] = re(back, ny, r, c) / mn;
    std::vector<double> sh = fftshift(af, ny, nx);
    double hi = -Inf;
    for (double v : sh)
        if (!std::isnan(v)) hi = std::max(hi, v);
    for (double& v : sh) v = v / hi;
    x.v = std::move(sh);
    return x;
}

std::vector<double> scl(const Grid& g, bool lonlat, const std::vector<double>& thresholds) {
    const Raster img = aacf_image(g, lonlat);
    std::vector<double> out(thresholds.size() * 2, NaN);
    bool all_na = true;
    for (double v : img.v) all_na &= std::isnan(v);
    if (all_na) return out;
    const Raster half = upper_half(img);
    // distances from zero lag: the last row of the half, column floor(M / 2)
    const std::vector<double> d = distance_to(half, half.ny - 1, half.nx / 2);
    const Index k = thresholds.size();
    for (Index t = 0; t < k; ++t) {
        double lo = Inf, hi = -Inf;
        for (Index i = 0; i < half.v.size(); ++i)
            if (half.v[i] <= thresholds[t] && !std::isnan(d[i])) {
                lo = std::min(lo, d[i]);
                hi = std::max(hi, d[i]);
            }
        out[t] = lo;
        out[k + t] = hi;
    }
    return out;
}

std::vector<double> texture_direction(const Grid& g, bool lonlat) {
    const Raster x = na_crop(raster_of(g, lonlat));
    bool all_na = true;
    for (double v : x.v) all_na &= std::isnan(v);
    if (all_na) return {NaN, NaN};
    const Raster amp = amplitude_half(x);
    const int r0 = amp.ny - 1, c0 = amp.nx / 2;
    const std::vector<double> d = distance_to(amp, r0, c0);
    // the largest circle around zero frequency within the half image
    auto at = [&](int r, int c) { return d[static_cast<Index>(r) * amp.nx + c]; };
    const double min_dist = std::min(std::min(at(r0, 0), at(r0, amp.nx - 1)), at(0, c0));
    // angles (degrees) from zero frequency, summed in bins of 0.5 degree
    const double xc = amp.x(c0), yc = amp.y(r0);
    // orientations 0 to 179.5 degrees: in the zero row only the positive side
    // (the negative side repeats it at 180 degrees)
    const int na = 360;  // seq(0, 179.5, 0.5)
    Sums a(static_cast<Index>(na));
    for (int r = 0; r < amp.ny; ++r)
        for (int c = 0; c < amp.nx; ++c) {
            const Index i = static_cast<Index>(r) * amp.nx + c;
            if (!(d[i] <= min_dist) || (r == r0 && c <= c0)) continue;
            const double angle = lsm::r_atan2(amp.y(r) - yc, amp.x(c) - xc) / (kPi / 180);
            const int k0 = static_cast<int>(std::floor(angle * 2)) - 1;
            for (int k = std::max(0, k0); k <= std::min(na - 1, k0 + 3); ++k)
                if (std::fabs(angle - k * 0.5) < 0.25) a.add(static_cast<Index>(k), amp.v[i]);
        }
    std::vector<double> aa(static_cast<Index>(na));
    double hi = -Inf;
    for (int k = 0; k < na; ++k) {
        aa[static_cast<Index>(k)] = a[static_cast<Index>(k)];
        if (!std::isnan(aa[static_cast<Index>(k)])) hi = std::max(hi, aa[static_cast<Index>(k)]);
    }
    double std_angle = Inf;
    for (int k = 0; k < na; ++k)
        if (aa[static_cast<Index>(k)] == hi) std_angle = std::min(std_angle, k * 0.5);
    const double stdi = lsm::r_mean(aa.data(), na) / hi;
    return {std_angle, stdi};
}

std::vector<double> radial_wavelength(const Grid& g, bool lonlat) {
    const Raster x = na_crop(raster_of(g, lonlat));
    bool all_na = true;
    for (double v : x.v) all_na &= std::isnan(v);
    if (all_na) return {NaN, NaN, NaN};
    const Raster amp = amplitude_half(x);
    const int ncircles = std::floor(0.5 * amp.nx) <= 100 ? static_cast<int>(std::floor(0.5 * amp.nx)) : 100;
    const std::vector<double> d = distance_to(amp, amp.ny - 1, amp.nx / 2);
    double half = -Inf;
    for (int c = 0; c < amp.nx; ++c) half = std::max(half, d[static_cast<Index>(amp.ny - 1) * amp.nx + c]);
    // radius <- seq(0, half_dist, length.out = ncircles)
    std::vector<double> radius;
    if (ncircles > 2) {
        if (half == 0) {
            radius.assign(static_cast<Index>(ncircles), 0.0);
        } else {
            const double by = (half - 0) / (static_cast<double>(ncircles) - 1);
            radius.push_back(0);
            for (int k = 1; k < ncircles - 1; ++k) radius.push_back(0 + k * by);
            radius.push_back(half);
        }
    } else if (ncircles == 2) {
        radius = {0.0, half};
    } else if (ncircles == 1) {
        radius = {0.0};
    }
    const Index nr = radius.size();
    if (nr < 3) return {NaN, NaN, NaN};  // geodiv: radius[3] is NA
    const double tol = (radius[2] - radius[1]) / 2;
    Sums br(nr);
    const double step = radius[1];
    for (Index i = 0; i < amp.v.size(); ++i) {
        const double v = d[i];
        if (std::isnan(v)) continue;
        const i64 k0 = step > 0 ? static_cast<i64>(std::floor(v / step)) - 1 : 0;
        for (i64 k = std::max<i64>(0, k0); k <= std::min<i64>(static_cast<i64>(nr) - 1, k0 + 3); ++k)
            if (std::fabs(v - radius[static_cast<Index>(k)]) < tol) br.add(static_cast<Index>(k), amp.v[i]);
        // the last radius is half_dist, not (n - 1) * by: check it too
        if (k0 + 3 < static_cast<i64>(nr) - 1 && std::fabs(v - radius[nr - 1]) < tol) br.add(nr - 1, amp.v[i]);
    }
    std::vector<double> b(nr);
    for (Index k = 0; k < nr; ++k) b[k] = br[k];
    b[0] = 0;
    double hi = -Inf;
    for (double v : b)
        if (!std::isnan(v)) hi = std::max(hi, v);
    std::vector<Index> at;
    for (Index k = 0; k < nr; ++k)
        if (b[k] == hi) at.push_back(k);
    const double srw = at.size() == 1 ? radius[at[0]] : NaN;
    const double srwi = lsm::r_mean(b.data(), static_cast<i64>(nr)) / hi;
    const double half_val = 0.5 * lsm::r_sum(b.data(), static_cast<i64>(nr));
    // sum(Br[1:i]) - half_val, the radius where it is closest to 0
    std::vector<double> dev(nr);
    lsm::Extended acc(0.0);
    for (Index k = 0; k < nr; ++k) {
        acc += b[k];
        dev[k] = std::fabs(static_cast<double>(acc) - half_val);
    }
    double lo = Inf;
    for (double v : dev) lo = std::min(lo, v);
    at.clear();
    for (Index k = 0; k < nr; ++k)
        if (dev[k] == lo) at.push_back(k);
    const double shw = at.size() == 1 ? radius[at[0]] : NaN;
    return {srw, srwi, shw};
}

double sfd(const Grid& g) {
    // geodiv's sfd_ (Clarke's triangular prisms), corrected: the extremes of
    // the data (not 99999 / -99999), (b - c)^2 for its typo (b - e)(b - c),
    // the central square from 0-based indices (geodiv reads past the matrix);
    // prism areas summed per step as R sums a vector
    const int rowmax = g.ny, colmax = g.nx;
    const i64 n_all = g.size();
    double lo = Inf, hi = -Inf;
    for (i64 i = 0; i < n_all; ++i) {
        if (std::isnan(g.z[i])) return NaN;
        lo = std::min(lo, g.z[i]);
        hi = std::max(hi, g.z[i]);
    }
    std::vector<double> nm(static_cast<Index>(n_all));
    for (i64 i = 0; i < n_all; ++i) nm[static_cast<Index>(i)] = (g.z[i] - lo) / (hi - lo);
    auto at = [&](int r, int c) { return nm[static_cast<Index>(r) * colmax + c]; };
    const int size = std::min(rowmax, colmax);
    int n = 1, steps = 1;
    do {
        steps++;
        n *= 2;
    } while (n < size);
    n /= 2;
    steps--;
    const int br = (rowmax - n) / 2, bc = (colmax - n) / 2;  // 0-based first row / column
    std::vector<double> la(static_cast<Index>(steps)), lr(static_cast<Index>(steps));
    std::vector<double> sums;
    int step = 1;
    for (int t = 0; t < steps; ++t) {
        const double side = step;
        const double diag = step * std::sqrt(2.0) / 2;
        const int nb = (n + step - 1) / step;  // rows (and columns) of blocks
        const bool scalar = nb == 1;
        auto S = [scalar](double v) { return lsm::r_sqrt(v, scalar); };
        sums.assign(static_cast<Index>(nb) * nb, 0.0);
#pragma omp parallel for schedule(static) if (nb > 64)
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j) {
                const int row = br + i * step, col = bc + j * step;
                const double a = at(row, col), b = at(row, col + step);
                const double c = at(row + step, col + step), d = at(row + step, col);
                const double e = 0.25 * (a + b + c + d);
                const double w = S((a - b) * (a - b) + side * side);
                const double x = S((b - c) * (b - c) + side * side);
                const double y = S((c - d) * (c - d) + side * side);
                const double z = S((a - d) * (a - d) + side * side);
                const double o = S((a - e) * (a - e) + diag * diag);
                const double p = S((b - e) * (b - e) + diag * diag);
                const double q = S((c - e) * (c - e) + diag * diag);
                const double s = S((d - e) * (d - e) + diag * diag);
                const double sa = 0.5 * (w + p + o), sb = 0.5 * (x + p + q);
                const double sc = 0.5 * (y + q + s), sd = 0.5 * (z + o + s);
                const double aa = S(std::fabs(sa * (sa - w) * (sa - p) * (sa - o)));
                const double ab = S(std::fabs(sb * (sb - x) * (sb - p) * (sb - q)));
                const double ac = S(std::fabs(sc * (sc - y) * (sc - q) * (sc - s)));
                const double ad = S(std::fabs(sd * (sd - z) * (sd - o) * (sd - s)));
                sums[static_cast<Index>(i) * nb + j] = aa + ab + ac + ad;
            }
        const double area = lsm::r_sum(sums.data(), static_cast<i64>(sums.size()));
        la[static_cast<Index>(t)] = area;
        lr[static_cast<Index>(t)] = static_cast<double>(step) * step;
        step *= 2;
    }
    // log (R's vector log), the regression's slope
    const bool one = steps == 1;
    (void)one;
    for (int t = 0; t < steps; ++t) {
        lr[static_cast<Index>(t)] = lsm::r_log(lr[static_cast<Index>(t)]);
        la[static_cast<Index>(t)] = lsm::r_log(la[static_cast<Index>(t)]);
    }
    const double mr = lsm::r_sum(lr.data(), steps) / steps, ma = lsm::r_sum(la.data(), steps) / steps;
    std::vector<double> cr(static_cast<Index>(steps)), rr(static_cast<Index>(steps)), aa2(static_cast<Index>(steps));
    for (int t = 0; t < steps; ++t) {
        const double dr = lr[static_cast<Index>(t)] - mr, da = la[static_cast<Index>(t)] - ma;
        cr[static_cast<Index>(t)] = dr * da;
        rr[static_cast<Index>(t)] = dr * dr;
        aa2[static_cast<Index>(t)] = da * da;
    }
    double cross = lsm::r_sum(cr.data(), steps), sumres = lsm::r_sum(rr.data(), steps),
           sumarea = lsm::r_sum(aa2.data(), steps);
    if (sumres == 0) sumres = 1;
    if (sumarea == 0) sumarea = 1;
    const double crossr = cross / std::sqrt(sumres * sumarea);
    return 2 - ((crossr * std::sqrt(sumarea)) / std::sqrt(sumres));
}

}  // namespace surf
