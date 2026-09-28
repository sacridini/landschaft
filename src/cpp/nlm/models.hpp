// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR's neutral landscape models. Rasters are row-major (nrow x ncol)
// arrays of doubles, NaN for missing values; R fills matrices by column,
// which the models reproduce when they draw random numbers.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlm/rng.hpp"

namespace nlm {

// -- utilities -------------------------------------------------------------------

// util_rescale(): (x - min) / (max - min) over the non-NaN cells; 0 when
// min and max are all.equal() (NaN stays NaN).
void rescale(double* x, i64 n);

// -- simple models ----------------------------------------------------------------

// nlm_random(): runif() by column.
std::vector<double> random_field(RRng& rng, i64 nrow, i64 ncol);
// nlm_percolation(): runif(1) < prob by column (1 or 0).
std::vector<double> percolation(RRng& rng, i64 nrow, i64 ncol, double prob);
// nlm_planargradient() before rescaling: southness * row + eastness * col.
std::vector<double> planar_gradient(i64 nrow, i64 ncol, double direction);
// The edge gradient of a rescaled planar gradient: (|0.5 - g| * -2) + 1.
void edge_from_planar(double* g, i64 n);

// -- distances ---------------------------------------------------------------------

// Distance of every cell to the nearest cell with origin[i] != 0.
//  - "gdal": terra::distance() on a planar raster (GDALComputeProximity,
//    cell units times `res`, float32 values);
//  - "haversine": terra::distance() on a lon/lat raster (terra's fast
//    haversine path on a 6378137 m sphere; cell centres from the extent
//    xmin, ymax and the cell sizes);
//  - "exact": exact Euclidean distance between cell centres times `res`.
std::vector<double> distance_gdal(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res);
std::vector<double> distance_haversine(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double xmin,
                                       double ymax, double xres, double yres);
std::vector<double> distance_exact(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res);

// -- NLMR's C++ models (std::mt19937) --------------------------------------------------

// rcpp_mpd()'s diamond-square on a size x size map (map[a][b] at
// a * size + b), with rand_dev[step] the deviation of each step.
std::vector<double> mpd_square(i64 size, const std::vector<double>& rand_dev, std::uint32_t seed, bool torus);
// rcpp_neigh(): categories 0..n_categories (row-major nrow x ncol).
std::vector<double> neigh(i64 nrow, i64 ncol, int n_categories, const std::vector<std::int64_t>& cells_per_cat,
                          int neighbourhood, float p_neigh, float p_empty, std::uint32_t seed);
// rcpp_randomrectangularcluster().
std::vector<double> random_rectangular_cluster(i64 nrow, i64 ncol, int minl, int maxl, std::uint32_t seed);

// -- NLMR's R-level cluster models ----------------------------------------------------------

// nlm_randomcluster(): types 1..ai.size() (row-major nrow x ncol).
std::vector<double> random_cluster(RRng& rng, i64 nrow, i64 ncol, double p, const std::vector<double>& ai,
                                   int neighbourhood);
// nlm_curds(): 1 for curds, 0 for the matrix, on a size x size grid (size =
// the product of the steps).
std::vector<double> curds(RRng& rng, const std::vector<double>& curds_p, const std::vector<std::int64_t>& steps,
                          const std::vector<double>* wheyes, i64& size);

// -- spectral models (NLMR 1.4's R code) ---------------------------------------------------

// simulate_gaussian_field() (row-major nrow x ncol).
std::vector<double> gaussian_field(RRng& rng, i64 nrow, i64 ncol, double autocorr_range, double mag_var, double nug,
                                   double mean);
// simulate_fbm() (row-major nrow x ncol).
std::vector<double> fbm(RRng& rng, i64 nrow, i64 ncol, double fract_dim);

// -- mosaic fields (spatstat) --------------------------------------------------------------

// One rMosaicField(rpoislinetess(4), rnorm, dimyx = c(nrow, ncol)) image,
// rows from the bottom (y increasing), NaN outside every tile.
std::vector<double> mosaic_tile_field(RRng& rng, i64 nrow, i64 ncol, double mean, double sd);
// log(exp(Z)) of rLGCP("exp", mu, var, scale) on the unit square, rows from
// the bottom.
std::vector<double> lgcp_log_field(RRng& rng, i64 nrow, i64 ncol, double mu, double var, double scale);

struct MosaicFieldResult {
    std::vector<double> sum;                 // rows from the top; empty when n == 0
    std::vector<std::vector<double>> steps;  // `collect`
    std::vector<double> inf;                 // `infinit`
};
MosaicFieldResult mosaic_field(RRng& rng, i64 nrow, i64 ncol, int n, bool nlmr_compat, double mean, double sd,
                               bool collect, bool infinit);

// -- Voronoi mosaics -----------------------------------------------------------------------

struct GermDraws {
    std::vector<double> x, y, v;  // germs and their values (in draw order)
};
// nlm_mosaictess(): sf::st_sample()'s germs, then runif(germs).
GermDraws tess_germs(RRng& rng, i64 germs, double width, double height);
// nlm_mosaicgibbs(): spatstat's rSSI() germs, then the shuffled classes.
GermDraws gibbs_germs(RRng& rng, i64 germs, double r, int patch_classes, double width, double height, int giveup);
// The value of the nearest germ at every cell centre (row-major, first row
// on top, cells of size 1 from (0, 0)).
std::vector<double> nearest_germ_raster(const std::vector<double>& gx, const std::vector<double>& gy,
                                        const std::vector<double>& value, i64 nrow, i64 ncol);

// -- classification -----------------------------------------------------------------

// NLMR's rcpp_get_jenksbreaks(sorted d, k): k + 1 break values.
std::vector<double> jenks_breaks(const std::vector<double>& d, int k);
// base::cut(x, breaks, include.lowest = TRUE, right = TRUE, labels = FALSE)
// (breaks sorted and unique); NaN outside the breaks or for NaN.
void cut_include_lowest(const double* x, i64 n, const std::vector<double>& breaks, double* out);

}  // namespace nlm
