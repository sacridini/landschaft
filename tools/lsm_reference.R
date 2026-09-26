# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run landscapemetrics::calculate_lsm on one raster and write every value.
#
# Usage:
#   Rscript lsm_reference.R input.tif output.csv [key=value ...]
#
# Keys are calculate_lsm arguments: directions, count_boundary,
# consider_boundary, edge_depth, cell_center, classes_max, neighbourhood,
# ordered, base, level (comma separated). Values are written with 17
# significant digits so the comparison is limited by the metric, not by
# the text format.
#
# Set LANDSCHAFT_R_LIBS to a library that holds the landscapemetrics version
# to compare against (it is put first on .libPaths()).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: lsm_reference.R input.tif output.csv [key=value ...]")

lib <- Sys.getenv("LANDSCHAFT_R_LIBS")
if (nzchar(lib)) .libPaths(c(strsplit(lib, .Platform$path.sep)[[1]], .libPaths()))

suppressPackageStartupMessages({
    library(terra)
    library(landscapemetrics)
})

opts <- list(directions = 8, count_boundary = FALSE, consider_boundary = FALSE,
             edge_depth = 1, cell_center = FALSE, classes_max = NULL,
             neighbourhood = 4, ordered = TRUE, base = "log2",
             level = c("patch", "class", "landscape"))
for (kv in args[-(1:2)]) {
    key <- sub("=.*$", "", kv)
    value <- sub("^[^=]*=", "", kv)
    if (!key %in% names(opts)) stop("unknown option: ", key)
    opts[[key]] <- switch(key,
        directions = , edge_depth = , neighbourhood = as.integer(value),
        classes_max = if (value == "NULL") NULL else as.numeric(value),
        count_boundary = , consider_boundary = , cell_center = , ordered = as.logical(value),
        base = value,
        level = strsplit(value, ",")[[1]])
}

r <- terra::rast(args[[1]])
what <- list_lsm(level = opts$level, simplify = TRUE)
# circle metrics only exist for square cells in landscapemetrics (same test)
if (!isTRUE(all.equal(terra::res(r)[[1]], terra::res(r)[[2]]))) what <- what[!grepl("circle", what)]
res <- suppressWarnings(calculate_lsm(
    r, what = what, directions = opts$directions,
    count_boundary = opts$count_boundary, consider_boundary = opts$consider_boundary,
    edge_depth = opts$edge_depth, cell_center = opts$cell_center,
    classes_max = opts$classes_max, neighbourhood = opts$neighbourhood,
    ordered = opts$ordered, base = opts$base, verbose = FALSE))

out <- data.frame(level = res$level, class = res$class, id = res$id,
                  metric = res$metric, value = sprintf("%.17g", res$value))
write.csv(out, args[[2]], row.names = FALSE, quote = FALSE, na = "NA")
cat("landscapemetrics", as.character(packageVersion("landscapemetrics")), "\n")
