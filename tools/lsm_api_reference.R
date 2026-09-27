# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run landscapemetrics' public helpers and the full_name / return_raster
# options on one raster and write their results (one CSV each) into a folder.
#
# Usage:
#   Rscript lsm_api_reference.R input.tif points.csv size outdir
#
# points.csv has columns x, y (inside the raster): sample_lsm plots and
# construct_buffer centres, buffered by `size`. Values are written with 17
# significant digits.
#
# Set LANDSCHAFT_R_LIBS to a library that holds the landscapemetrics version
# to compare against (it is put first on .libPaths()).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: lsm_api_reference.R input.tif points.csv size outdir")

lib <- Sys.getenv("LANDSCHAFT_R_LIBS")
if (nzchar(lib)) .libPaths(c(strsplit(lib, .Platform$path.sep)[[1]], .libPaths()))

suppressPackageStartupMessages({
    library(terra)
    library(landscapemetrics)
})

r <- terra::rast(args[[1]])
pts <- as.matrix(read.csv(args[[2]])[, c("x", "y")])
size <- as.numeric(args[[3]])
out <- args[[4]]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
fmt <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))
save <- function(df, name) {
    for (col in names(df)) if (is.double(df[[col]])) df[[col]] <- fmt(df[[col]])
    write.csv(df, file.path(out, paste0(name, ".csv")), row.names = FALSE, quote = FALSE, na = "NA")
}
save_raster <- function(x, name) {
    m <- if (inherits(x, "SpatRaster")) terra::as.matrix(x, wide = TRUE) else x
    df <- data.frame(row = rep(seq_len(nrow(m)), times = ncol(m)), col = rep(seq_len(ncol(m)), each = nrow(m)),
                     value = as.vector(m))
    save(df, name)
    if (inherits(x, "SpatRaster")) {
        e <- as.vector(terra::ext(x))
        save(data.frame(xmin = e[[1]], xmax = e[[2]], ymin = e[[3]], ymax = e[[4]], nrow = nrow(x), ncol = ncol(x)),
             paste0(name, "_extent"))
    }
}
square <- isTRUE(all.equal(terra::res(r)[[1]], terra::res(r)[[2]]))

# calculate_lsm(full_name = TRUE): the metric descriptions and the row order
what <- list_lsm(level = c("class", "landscape"), simplify = TRUE)
if (!square) what <- what[!grepl("circle", what)]
full <- suppressWarnings(calculate_lsm(r, what = c("lsm_p_area", "lsm_p_shape", what), full_name = TRUE,
                                       verbose = FALSE))
save(data.frame(layer = full$layer, level = full$level, class = full$class, id = full$id, metric = full$metric,
                value = full$value, name = full$name, type = full$type, function_name = full$function_name),
     "full_name")
writeLines(names(full), file.path(out, "full_name_columns.txt"))

# pad_raster / unpad_raster (with cells that are not square, terra recycles
# ext + res(x) * n as xmin - rx, xmax + ry, ymin - rx, ymax + ry and the
# padded matrix does not fit: landscapemetrics fails)
if (square) {
    for (g in c(FALSE, TRUE)) {
        p <- pad_raster(r, pad_raster_value = -999, pad_raster_cells = 2, global = g)[[1]]
        save_raster(p, paste0("pad_", tolower(g)))
    }
    p <- pad_raster(r, pad_raster_value = 7, pad_raster_cells = 3)[[1]]
    save_raster(unpad_raster(p, unpad_raster_cells = 3)[[1]], "unpad")
}

# raster_to_points
for (na in c(TRUE, FALSE)) {
    xyz <- raster_to_points(r, return_NA = na)
    save(data.frame(layer = xyz[, "layer"], x = xyz[, "x"], y = xyz[, "y"], z = xyz[, "z"]),
         paste0("points_", tolower(na)))
}

# get_unique_values
save(data.frame(value = get_unique_values(r, simplify = TRUE)), "unique")

# construct_buffer
for (shape in c("square", "circle")) {
    m <- construct_buffer(pts, shape = shape, size = size, return_vec = FALSE)
    save(data.frame(x = m[, 1], y = m[, 2], id = m[, 3]), paste0("buffer_", shape))
    v <- terra::geom(construct_buffer(pts, shape = shape, size = size, return_vec = TRUE))
    save(data.frame(geom = v[, "geom"], x = v[, "x"], y = v[, "y"]), paste0("buffer_", shape, "_vector"))
}

# sample_lsm(return_raster = TRUE): each plot's raster
for (shape in c("square", "circle")) {
    s <- suppressWarnings(sample_lsm(r, pts, shape = shape, size = size, what = "lsm_l_ta", return_raster = TRUE,
                                     verbose = FALSE))
    for (k in seq_len(nrow(s))) {
        save_raster(s$raster_sample_plots[[k]], paste0("sample_", shape, "_", s$plot_id[[k]]))
    }
    writeLines(names(s), file.path(out, paste0("sample_", shape, "_columns.txt")))
}
s <- suppressWarnings(sample_lsm(r, pts, shape = "square", size = size, what = "lsm_l_ta", full_name = TRUE,
                                 verbose = FALSE))
writeLines(names(s), file.path(out, "sample_full_name_columns.txt"))
cat("landscapemetrics", as.character(packageVersion("landscapemetrics")), "\n")
