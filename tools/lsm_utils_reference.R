# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run landscapemetrics' utility functions on one raster and write their
# results (one CSV each) into a folder.
#
# Usage:
#   Rscript lsm_utils_reference.R input.tif outdir [key=value ...]
#
# Keys: directions, consider_boundary, edge_depth, points (CSV with x, y),
# neighbourhood (CSV of a neighbourhood matrix, 1 = neighbour, 0 = centre),
# stack (more rasters, comma separated, for landscape-level correlations).
# Values are written with 17 significant digits.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: lsm_utils_reference.R input.tif outdir [key=value ...]")

lib <- Sys.getenv("LANDSCHAFT_R_LIBS")
if (nzchar(lib)) .libPaths(c(strsplit(lib, .Platform$path.sep)[[1]], .libPaths()))

suppressPackageStartupMessages({
    library(terra)
    library(landscapemetrics)
})

opts <- list(directions = 8, consider_boundary = FALSE, edge_depth = 1, points = NULL, neighbourhood = NULL,
             stack = NULL)
for (kv in args[-(1:2)]) {
    key <- sub("=.*$", "", kv)
    value <- sub("^[^=]*=", "", kv)
    if (!key %in% names(opts)) stop("unknown option: ", key)
    opts[[key]] <- switch(key,
        directions = , edge_depth = as.integer(value),
        consider_boundary = as.logical(value),
        points = , neighbourhood = , stack = value)
}

r <- terra::rast(args[[1]])
out <- args[[2]]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
fmt <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))
save <- function(df, name) {
    for (col in names(df)) if (is.double(df[[col]])) df[[col]] <- fmt(df[[col]])
    write.csv(df, file.path(out, paste0(name, ".csv")), row.names = FALSE, quote = FALSE, na = "NA")
}
save_matrix <- function(m, name) {
    df <- data.frame(row = rep(seq_len(nrow(m)), times = ncol(m)), col = rep(seq_len(ncol(m)), each = nrow(m)),
                     value = as.vector(m))
    save(df, name)
}
square <- isTRUE(all.equal(terra::res(r)[[1]], terra::res(r)[[2]]))
d <- opts$directions

# get_centroids
for (cc in c(FALSE, TRUE)) {
    cen <- suppressWarnings(get_centroids(r, directions = d, cell_center = cc, verbose = FALSE))
    save(data.frame(class = cen$class, id = cen$id, x = cen$x, y = cen$y), paste0("centroids_", tolower(cc)))
}

# get_nearestneighbour on the patches of each class (return_id)
patches <- get_patches(r, directions = d)[[1]]
nn <- do.call(rbind, lapply(names(patches), function(nm) {
    p <- patches[[nm]]
    if (length(unique(terra::values(p, mat = FALSE)[!is.na(terra::values(p, mat = FALSE))])) < 2) return(NULL)
    res <- get_nearestneighbour(p, return_id = TRUE)
    data.frame(class = as.integer(sub("class_", "", nm)), id = res$id, dist = res$dist, id_neighbour = res$id_neighbour)
}))
if (!is.null(nn)) save(nn, "nearestneighbour")

# get_circumscribingcircle
if (square) {
    for (lv in c("patch", "class")) {
        cir <- get_circumscribingcircle(r, directions = d, level = lv)
        save(data.frame(class = cir$class, id = cir$id, value = cir$value, center_x = cir$center_x,
                        center_y = cir$center_y), paste0("circle_", lv))
    }
}

# get_adjacencies
for (nb in c(4, 8)) {
    for (what in c("full", "like", "unlike", "triangle")) {
        for (upper in c(FALSE, TRUE)) {
            a <- get_adjacencies(r, neighbourhood = nb, what = what, upper = upper)[[1]]
            save_matrix(a, paste0("adjacencies_", nb, "_", what, "_", tolower(upper)))
        }
    }
}
if (!is.null(opts$neighbourhood)) {
    m <- as.matrix(read.csv(opts$neighbourhood, header = FALSE))
    dimnames(m) <- NULL
    save_matrix(get_adjacencies(r, neighbourhood = m)[[1]], "adjacencies_custom")
}

# get_boundaries: of the raster values, and of the patches of each class
for (pid in c(FALSE, TRUE)) {
    b <- get_boundaries(r, consider_boundary = opts$consider_boundary, edge_depth = opts$edge_depth,
                        patch_id = pid)[[1]]
    save_matrix(terra::as.matrix(b, wide = TRUE), paste0("boundaries_values_", tolower(pid)))
}
bp <- NULL
for (nm in names(patches)) {
    b <- terra::as.matrix(get_boundaries(patches[[nm]], consider_boundary = opts$consider_boundary,
                                         edge_depth = opts$edge_depth, patch_id = TRUE)[[1]], wide = TRUE)
    if (is.null(bp)) bp <- b else bp[!is.na(b)] <- b[!is.na(b)]
}
save_matrix(bp, "boundaries_patches")

# extract_lsm
if (!is.null(opts$points)) {
    pts <- as.matrix(read.csv(opts$points)[, c("x", "y")])
    # circle refuses non-square cells
    patch_what <- list_lsm(level = "patch", simplify = TRUE)
    if (!square) patch_what <- setdiff(patch_what, "lsm_p_circle")
    ex <- suppressWarnings(extract_lsm(r, y = pts, what = patch_what, directions = d, verbose = FALSE))
    save(data.frame(level = ex$level, class = ex$class, id = ex$id, metric = ex$metric, value = ex$value,
                    extract_id = ex$extract_id), "extract")
}

# check_landscape
ch <- suppressWarnings(check_landscape(r, verbose = FALSE))
save(data.frame(crs = ch$crs, units = ch$units, class = ch$class, n_classes = ch$n_classes,
                OK = sprintf("U+%04X", utf8ToInt(ch$OK))), "check")

# calculate_correlation (patch and class level of this raster)
mets <- calculate_lsm(r, what = c("lsm_p_area", "lsm_p_perim", "lsm_p_shape", "lsm_p_frac", "lsm_p_contig",
                                  "lsm_p_enn", "lsm_c_ca", "lsm_c_np", "lsm_c_pland", "lsm_c_te", "lsm_c_lpi",
                                  "lsm_c_enn_mn"),
                      directions = d, verbose = FALSE)
if (!is.null(opts$stack)) {
    layers <- c(list(r), lapply(strsplit(opts$stack, ",")[[1]], terra::rast))
    land <- calculate_lsm(layers, what = c("lsm_l_ta", "lsm_l_np", "lsm_l_shdi", "lsm_l_contag", "lsm_l_ed",
                                          "lsm_l_area_mn"), directions = d, verbose = FALSE)
    mets <- rbind(mets, land)
}
for (method in c("pearson", "spearman", "kendall")) {
    co <- suppressWarnings(calculate_correlation(mets, method = method))
    for (lv in names(co)) {
        save(data.frame(metric_1 = co[[lv]]$metric_1, metric_2 = co[[lv]]$metric_2, value = co[[lv]]$value),
             paste0("correlation_", method, "_", lv))
    }
}
cat("landscapemetrics", as.character(packageVersion("landscapemetrics")), "\n")
