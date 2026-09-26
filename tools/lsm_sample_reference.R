# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run landscapemetrics::sample_lsm on one raster and write every value.
#
# Usage:
#   Rscript lsm_sample_reference.R input.tif plots.csv output.csv [key=value ...]
#
# plots.csv has columns x, y (points, buffered with shape/size) or wkt
# (polygons, in the raster's CRS), and optionally plot_id. Keys: shape, size,
# all_classes, vector (points as a SpatVector in the raster's CRS), level
# (comma separated) and the calculate_lsm options directions,
# count_boundary, consider_boundary, edge_depth, cell_center, classes_max,
# neighbourhood, ordered, base. Values are written with 17 significant
# digits.
#
# Set LANDSCHAFT_R_LIBS to a library that holds the landscapemetrics version
# to compare against (it is put first on .libPaths()).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: lsm_sample_reference.R input.tif plots.csv output.csv [key=value ...]")

lib <- Sys.getenv("LANDSCHAFT_R_LIBS")
if (nzchar(lib)) .libPaths(c(strsplit(lib, .Platform$path.sep)[[1]], .libPaths()))

suppressPackageStartupMessages({
    library(terra)
    library(landscapemetrics)
})

opts <- list(shape = "square", size = NULL, all_classes = FALSE, vector = FALSE,
             level = c("patch", "class", "landscape"),
             directions = 8, count_boundary = FALSE, consider_boundary = FALSE,
             edge_depth = 1, cell_center = FALSE, classes_max = NULL,
             neighbourhood = 4, ordered = TRUE, base = "log2")
for (kv in args[-(1:3)]) {
    key <- sub("=.*$", "", kv)
    value <- sub("^[^=]*=", "", kv)
    if (!key %in% names(opts)) stop("unknown option: ", key)
    opts[[key]] <- switch(key,
        directions = , edge_depth = , neighbourhood = as.integer(value),
        size = , classes_max = if (value == "NULL") NULL else as.numeric(value),
        all_classes = , vector = , count_boundary = , consider_boundary = , cell_center = , ordered = as.logical(value),
        shape = , base = value,
        level = strsplit(value, ",")[[1]])
}

r <- terra::rast(args[[1]])
plots <- read.csv(args[[2]], stringsAsFactors = FALSE)
if ("wkt" %in% names(plots)) {
    y <- terra::vect(plots$wkt, crs = terra::crs(r))
} else if (opts$vector) {
    y <- terra::vect(as.matrix(plots[, c("x", "y")]), crs = terra::crs(r))
} else {
    y <- as.matrix(plots[, c("x", "y")])
}
plot_id <- if ("plot_id" %in% names(plots)) plots$plot_id else NULL

what <- list_lsm(level = opts$level, simplify = TRUE)
if (!isTRUE(all.equal(terra::res(r)[[1]], terra::res(r)[[2]]))) what <- what[!grepl("circle", what)]
res <- suppressWarnings(sample_lsm(
    r, y, plot_id = plot_id, shape = opts$shape, size = opts$size,
    all_classes = opts$all_classes, what = what, verbose = FALSE,
    directions = opts$directions, count_boundary = opts$count_boundary,
    consider_boundary = opts$consider_boundary, edge_depth = opts$edge_depth,
    cell_center = opts$cell_center, classes_max = opts$classes_max,
    neighbourhood = opts$neighbourhood, ordered = opts$ordered, base = opts$base))

out <- data.frame(level = res$level, class = res$class, id = res$id, metric = res$metric,
                  value = sprintf("%.17g", res$value), plot_id = res$plot_id,
                  percentage_inside = sprintf("%.17g", res$percentage_inside))
write.csv(out, args[[3]], row.names = FALSE, quote = FALSE, na = "NA")
