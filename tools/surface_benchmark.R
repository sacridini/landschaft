# SPDX-License-Identifier: GPL-3.0-or-later
#
# Timings of geodiv 1.1.1 and spatialEco 2.0-5 for tools/benchmark_surface.py:
# each task on geodiv's example rasters, the best of `reps` runs, as
# "task<TAB>seconds" lines.
#
#   Rscript tools/surface_benchmark.R data_dir reps

suppressMessages({
  library(terra)
  library(geodiv)
  library(spatialEco)
})

args <- commandArgs(TRUE)
data <- args[1]
reps <- as.integer(args[2])

nf <- rast(file.path(data, "normforest.tif"))
ndvi <- rast(file.path(data, "orforest.tif"))
dem <- rast(file.path(data, "orelevation.tif"))
small <- nf[101:200, 101:200, drop = FALSE]  # 100 x 100 cells
tiny <- nf[101:130, 101:130, drop = FALSE]  # 30 x 30 cells

# the best of n runs; a fast task is repeated until a run takes 0.5 s (the
# timer's resolution on Windows is 10 ms) and the time divided
once <- function(f, k) system.time(for (i in seq_len(k)) suppressWarnings(capture.output(f())))[["elapsed"]] / k
best <- function(f, n = reps) {
  t1 <- once(f, 1)
  k <- if (t1 < 0.5) ceiling(0.5 / max(t1, 0.001)) else 1
  min(c(if (k == 1) t1, sapply(seq_len(n), function(i) once(f, k))))
}

tasks <- list(
  "amplitude (sa, sq, ssk, sku)" = function() { geodiv::sa(nf); geodiv::sq(nf); geodiv::ssk(nf); geodiv::sku(nf) },
  "best trend surface (bestfitplane)" = function() geodiv::bestfitplane(ndvi),
  "slopes and area (sdq, sdq6, sdr)" = function() { geodiv::sdq(nf); geodiv::sdq6(nf); geodiv::sdr(nf) },
  "peaks, valleys (sds, ssc, s10z)" = function() { geodiv::sds(nf); geodiv::ssc(nf); geodiv::s10z(nf) },
  "bearing area (sbi, sci, svi, sk, spk, svk)" = function() {
    geodiv::sbi(nf); geodiv::sci(nf); geodiv::svi(nf); geodiv::sk(nf); geodiv::spk(nf); geodiv::svk(nf) },
  "autocorrelation lengths (scl, stxr)" = function() { geodiv::scl(nf); geodiv::stxr(nf) },
  "direction, wavelength (std, srw)" = function() { geodiv::std(nf); geodiv::srw(nf) },
  "fractal dimension (sfd)" = function() geodiv::sfd(nf, silent = TRUE),
  "texture image, sa in 5 x 5 windows, 100 x 100" = function()
    geodiv::focal_metrics(small, window = matrix(1, 5, 5), metrics = list("sa"), progress = FALSE),
  "texture image, sk in 5 x 5 windows, 30 x 30" = function()
    geodiv::focal_metrics(tiny, window = matrix(1, 5, 5), metrics = list("sk"), progress = FALSE),
  "tpi, 5 x 5" = function() spatialEco::tpi(dem, 5),
  "tri" = function() spatialEco::tri(dem),
  "vrm" = function() spatialEco::vrm(dem),
  "curvature (total)" = function() spatialEco::curvature(dem, "total"),
  "hsp, scales 3-27" = function() spatialEco::hsp(dem),
  "spherical sd" = function() spatialEco::spherical.sd(dem, 3)
)

for (name in names(tasks)) {
  n <- if (grepl("texture image|curvature|tri$|bearing", name)) 1 else reps
  cat(name, "\t", best(tasks[[name]], n), "\n", sep = "")
}
