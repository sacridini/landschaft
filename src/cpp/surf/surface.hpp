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

double sdq(const Grid& g);
double sdq6(const Grid& g);
double surface_area(const Grid& g);
double flat_area(const Grid& g);
double sdr(const Grid& g);

// -- local features (geodiv's localsurface.R) ---------------------------------

// The peaks / valleys (cell indices, row order).
std::vector<i64> find_peaks(const Grid& g);
std::vector<i64> find_valleys(const Grid& g);
double sds(const Grid& g);
double ssc(const Grid& g);
double s10z(const Grid& g);

// -- bearing area curve (geodiv's bearingarea.R, bestfit.R, simpsons.R) ------

// The Abbott-Firestone curve of the values.
struct Bearing {
    std::vector<double> s;                 // z sorted (no NA)
    std::vector<double> heights, bearing;  // distinct heights from the top, share at or above each
    double quantile(double q) const;       // R's quantile(z, q, type = 7)
    double share_above(double v) const;
    double height(double p) const { return quantile(1 - p); }  // the height at bearing p
};
Bearing bearing_curve(const double* z, i64 n);

// The area between the curve and its top over bearings [a, b] (Simpson, n
// intervals).
double area_above(const Bearing& b, double a, double bb, int n);

// geodiv's find_flat: the core line's heights at 0 and 1 and the bearing
// ratios Smr1, Smr2 where it crosses the curve (NaN where there is none).
struct CoreLine {
    double high, low, smr1, smr2;
};
CoreLine find_flat(const Bearing& b);

struct BearingParameters {
    double sbi, svi, sci, sk, spk, svk;
};
// sbi needs sq (the standard deviation of the values).
BearingParameters bearing_parameters(const Bearing& b, double sq);

// -- spatial parameters and fractal dimension (geodiv's aacf.R, fourier.R,
// sfd.cpp) --------------------------------------------------------------------

// A raster with its values, as terra's SpatRaster (row order).
struct Raster {
    std::vector<double> v;
    int ny = 0, nx = 0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    bool lonlat = false;
    double xres() const { return (xmax - xmin) / nx; }
    double yres() const { return (ymax - ymin) / ny; }
    double x(int col) const { return xmin + ((col + 0.5) * xres()); }
    double y(int row) const { return ymax - ((row + 0.5) * yres()); }
};

// geodiv's aacf: the normalised areal autocorrelation image (on the largest
// square without NA around the centre when there are NA).
Raster aacf_image(const Grid& g, bool lonlat);
// geodiv's scl: the fastest decay lengths to each threshold, then the slowest.
std::vector<double> scl(const Grid& g, bool lonlat, const std::vector<double>& thresholds);
// geodiv's std: the texture direction (degrees) and index.
std::vector<double> texture_direction(const Grid& g, bool lonlat);
// geodiv's srw: the dominant radial wavelength, its index, the half wavelength.
std::vector<double> radial_wavelength(const Grid& g, bool lonlat);
// The fractal dimension by triangular prisms (geodiv's sfd, corrected).
double sfd(const Grid& g);

// -- moving windows ------------------------------------------------------------

// The metrics window_metrics() computes (the order of ls.surface.window_metrics).
enum WindowMetric : int {
    kWSa, kWSq, kWSsk, kWSku, kWSv, kWSph, kWSmean, kWSdq, kWSdq6, kWSdr, kWSds, kWSsc, kWS10z,
    kWSbi, kWSci, kWSvi, kWSk, kWSpk, kWSvk, kWSal, kWStr, kWStd, kWStdi, kWSrw, kWSrwi, kWShw, kWSfd
};

// Every metric of `ids` in the size x size window around each cell (row
// order; NA beyond the edges and, with `circle`, beyond size / 2 cells).
std::vector<std::vector<double>> window_metrics(const Grid& g, bool lonlat, int size, bool circle,
                                                const std::vector<int>& ids);

// -- terrain (spatialEco's tpi, tri, curvature, hsp; terra's terrain) --------

// Slope or aspect (radians, or degrees) as terra::terrain: Horn's 8 or 4
// neighbours, NA on the edge cells.
std::vector<double> terrain(const Grid& g, bool lonlat, int neighbors, bool degrees, bool aspect);

enum FocalStat : int { kFocalMean, kFocalSum, kFocalMin, kFocalMax, kFocalSd };
// A statistic of the cells of a wr x wc window mask around every cell (NA
// where any of them is NA or beyond the edge).
std::vector<double> focal(const Grid& g, int wr, int wc, const std::vector<char>& mask, FocalStat stat);

std::vector<double> tri(const Grid& g, int wr, int wc);

enum Curvature : int { kCurvPlanform, kCurvProfile, kCurvTotal, kCurvMcnab, kCurvBolstad };
std::vector<double> curvature(const Grid& g, int type);

std::vector<double> tpi(const Grid& g, int wr, int wc, const std::vector<char>& mask, bool normalize);
// masks[k]: the rows[k] x cols[k] window mask of scale k.
std::vector<double> hsp(const Grid& g, const std::vector<int>& rows, const std::vector<int>& cols,
                        const std::vector<std::vector<char>>& masks, bool normalize);

}  // namespace surf
