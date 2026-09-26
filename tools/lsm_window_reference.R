# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run landscapemetrics::window_lsm on one raster and write every value.
#
# Usage:
#   Rscript lsm_window_reference.R input.tif window.csv output.csv [key=value ...]
#
# window.csv holds the window matrix (no header; 1 = in the window, NA =
# out). Keys: what (comma separated lsm_l_* names) and the options
# directions, count_boundary, consider_boundary, edge_depth, classes_max,
# neighbourhood, ordered, base. The output has one row per metric and cell
# (cells numbered from 1, row by row) with 17 significant digits.
#
# Set LANDSCHAFT_R_LIBS to a library that holds the landscapemetrics version
# to compare against (it is put first on .libPaths()).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: lsm_window_reference.R input.tif window.csv output.csv [key=value ...]")

lib <- Sys.getenv("LANDSCHAFT_R_LIBS")
if (nzchar(lib)) .libPaths(c(strsplit(lib, .Platform$path.sep)[[1]], .libPaths()))

suppressPackageStartupMessages({
    library(terra)
    library(landscapemetrics)
})

opts <- list(what = NULL, directions = 8, count_boundary = FALSE, consider_boundary = FALSE,
             edge_depth = 1, classes_max = NULL, neighbourhood = 4, ordered = TRUE, base = "log2")
for (kv in args[-(1:3)]) {
    key <- sub("=.*$", "", kv)
    value <- sub("^[^=]*=", "", kv)
    if (!key %in% names(opts)) stop("unknown option: ", key)
    opts[[key]] <- switch(key,
        directions = , edge_depth = , neighbourhood = as.integer(value),
        classes_max = if (value == "NULL") NULL else as.numeric(value),
        count_boundary = , consider_boundary = , ordered = as.logical(value),
        base = value,
        what = strsplit(value, ",")[[1]])
}

r <- terra::rast(args[[1]])
window <- as.matrix(read.csv(args[[2]], header = FALSE))
dimnames(window) <- NULL
res <- suppressWarnings(window_lsm(
    r, window = window, what = opts$what,
    directions = opts$directions, count_boundary = opts$count_boundary,
    consider_boundary = opts$consider_boundary, edge_depth = opts$edge_depth,
    classes_max = opts$classes_max, neighbourhood = opts$neighbourhood,
    ordered = opts$ordered, base = opts$base))[[1]]

rows <- lapply(names(res), function(m) {
    v <- terra::values(res[[m]], mat = FALSE)
    data.frame(metric = sub("^lsm_l_", "", m), cell = seq_along(v), value = sprintf("%.17g", v))
})
write.csv(do.call(rbind, rows), args[[3]], row.names = FALSE, quote = FALSE, na = "NA")
