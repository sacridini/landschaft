// SPDX-License-Identifier: GPL-3.0-or-later
//
// Surface metrics of a continuous raster (ISO 25178 as the R package geodiv
// computes them): the same bits as geodiv on R for Windows (80-bit sums, the
// x87 maths of mingw-w64, the C code of the packages it calls).
#pragma once

#include <cstdint>
#include <vector>

namespace surf {

using i64 = std::int64_t;

// A raster: ny x nx values in row order (NaN = NA) on an extent, as a terra
// SpatRaster.
struct Grid {
    const double* z;
    int ny, nx;
    double xmin, xmax, ymin, ymax;

    i64 size() const { return static_cast<i64>(ny) * nx; }
    // terra's xFromCol / yFromRow: the cell centres.
    double x(int col) const { return xmin + ((col + 0.5) * ((xmax - xmin) / nx)); }
    double y(int row) const { return ymax - ((row + 0.5) * ((ymax - ymin) / ny)); }
};

// -- amplitude (geodiv's basics.R) --------------------------------------------

struct Amplitude {
    double sa, sq;
    double ssk, ssk_unadj;  // adjusted Fisher-Pearson skewness, and not
    double sku, sku_raw;    // excess kurtosis, and not
    double sv, sph, smean;
};

// What amplitude() computes (the others are left at 0).
enum AmplitudeParts : unsigned {
    kSa = 1u, kSq = 2u, kSsk = 4u, kSku = 8u, kExtremes = 16u, kSmean = 32u, kAllAmplitude = 63u
};

// The amplitude parameters of the values (NaN = NA) of z.
Amplitude amplitude(const double* z, i64 n, unsigned parts = kAllAmplitude);

// -- trend surfaces (geodiv's fitplane: spatial::surf.ls) ---------------------

// The least-squares polynomial surface of order `order` (0-6) of the grid's
// values in the cell centres, evaluated in every cell (row order). `ifail` is
// spatial's failure code (0: fitted; then the surface is 0).
std::vector<double> fit_plane(const Grid& g, int order, int* ifail = nullptr);

// geodiv's bestfitplane: the surface of order 0-3 whose residuals have the
// smallest absolute mean (the lowest order on ties). `order` gets it.
std::vector<double> best_fit_plane(const Grid& g, int* order = nullptr);

// -- hybrid parameters (geodiv's sdq.R, surfacearea.R) ------------------------
// `iso`: the corrections of convention="iso" (sdq6's stencil; surface areas on
// the heights and cell sizes as they are, not rescaled).

double sdq(const Grid& g);
double sdq6(const Grid& g, bool iso);
double surface_area(const Grid& g, bool iso);
double flat_area(const Grid& g, bool iso);
double sdr(const Grid& g, bool iso);

// -- local features (geodiv's localsurface.R) ---------------------------------

// The peaks / valleys (cell indices, row order). `transposed`: geodiv's
// findvalleys, which reads the neighbours from the values filled by columns.
std::vector<i64> find_peaks(const Grid& g);
std::vector<i64> find_valleys(const Grid& g, bool transposed);
double sds(const Grid& g);
double ssc(const Grid& g);
double s10z(const Grid& g, bool transposed_valleys);

// -- bearing area curve (geodiv's bearingarea.R, bestfit.R, simpsons.R) ------

// The Abbott-Firestone curve of the values: R's ecdf(1 - z).
struct Bearing {
    std::vector<double> s;          // 1 - z sorted (no NA)
    std::vector<double> vals, cum;  // the distinct values and the ecdf there
    double quantile(double p) const;  // R's quantile(type = 7)
    double ecdf(double v) const;
    double height(double p) const { return 1 - quantile(p); }  // geodiv's height_ba
};
Bearing bearing_curve(const double* z, i64 n);

// geodiv's area_above(f, a, b, n) (Simpson's rule).
double area_above(const Bearing& b, double a, double bb, int n);

// geodiv's find_flat: the core line's heights at 0 and 1 and the bearing
// ratios Smr1, Smr2 where it crosses the curve (NaN where geodiv gives NA).
struct CoreLine {
    double high, low, smr1, smr2;
};
CoreLine find_flat(const Bearing& b, double perc);

struct BearingParameters {
    double sbi, svi, sci, sk, spk, svk;
};
// sbi needs sq (the standard deviation of the values).
BearingParameters bearing_parameters(const Bearing& b, double sq);

}  // namespace surf
