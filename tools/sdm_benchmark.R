# SPDX-License-Identifier: GPL-3.0-or-later
#
# Timings of the R packages behind ls.sdm for tools/benchmark_sdm.py: each
# task on the inputs the Python script wrote to `work`, the best of `reps`
# runs (a fast task repeated until a run takes 0.5 s), as "task<TAB>seconds".
#
#   Rscript tools/sdm_benchmark.R work_dir bioclim.tif reps

suppressMessages({
  library(terra)
  library(sf)
  library(CoordinateCleaner)
  library(spThin)
  library(usdm)
  library(dismo)
  library(blockCV)
  library(predicts)
  library(ecospat)
  library(maxnet)
})
sf_use_s2(FALSE)

args <- commandArgs(TRUE)
work <- args[1]
env <- rast(args[2])
reps <- as.integer(args[3])

once <- function(f, k) system.time(for (i in seq_len(k)) suppressWarnings(suppressMessages(capture.output(f()))))[["elapsed"]] / k
best <- function(f, n = reps) {
  t1 <- once(f, 1)
  if (t1 > 20) return(t1)  # a slow task runs once
  k <- if (t1 < 0.5) ceiling(0.5 / max(t1, 0.001)) else 1
  min(c(if (k == 1) t1, sapply(seq_len(n), function(i) once(f, k))))
}

records <- read.csv(file.path(work, "records.csv"))
thinpts <- read.csv(file.path(work, "thin.csv"))
pts <- read.csv(file.path(work, "points.csv"))
ptsf <- st_as_sf(pts, coords = c("lon", "lat"), crs = 4326)
ev <- read.csv(file.path(work, "evaluation.csv"))
p <- ev$pred[ev$y == 1]; a <- ev$pred[ev$y == 0]
mx <- read.csv(file.path(work, "maxnet.csv"))
mask <- ifel(any(is.na(env)), NA, 1)
pres <- as.matrix(pts[pts$occ == 1, c("lon", "lat")])

tasks <- list(
  "clean records" = function() clean_coordinates(records, tests = c("capitals", "centroids", "equal", "gbif",
    "institutions", "outliers", "zeros"), value = "flagged", verbose = FALSE),
  "thin" = function() thin.algorithm(data.frame(thinpts$lon, thinpts$lat), 10, 10),
  "vif selection" = function() { set.seed(1); vifstep(env, th = 10) },
  "background" = function() { set.seed(1); randomPoints(raster::raster(mask), 10000, p = pres) },
  "spatial blocks" = function() cv_spatial(ptsf, column = "occ", size = 500000, k = 5, iteration = 100, seed = 1,
    progress = FALSE, report = FALSE, plot = FALSE, biomod2 = FALSE),
  "spatial clusters" = function() { set.seed(1); cv_cluster(ptsf, column = "occ", k = 5, report = FALSE, biomod2 = FALSE) },
  "autocorrelation range" = function() { set.seed(1); cv_spatial_autocor(env, num_sample = 5000, plot = FALSE, progress = FALSE) },
  "evaluate" = function() pa_evaluate(p, a),
  "boyce" = function() ecospat.boyce(c(a, p), p, PEplot = FALSE),
  "mess" = function() mess(env, as.data.frame(mx[, names(env)])),
  "maxnet" = function() maxnet(mx$y, mx[, names(env)])
)

for (name in names(tasks)) {
  t <- best(tasks[[name]])
  cat(name, "\t", t, "\n", sep = "")
}
