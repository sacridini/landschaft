# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs the geodiv / spatialEco cases of tests/surface_cases.py (driven by
# tools/make_reference_surface.py): reads the job (JSON), rebuilds each raster
# from its values (float64, row-major, NaN = NA) with terra, evaluates each
# case's R code (a named list) and writes every result as raw float64 files
# plus a manifest.
#
#   Rscript tools/surface_reference.R job.json out_dir tools/surface_fixed.R

suppressMessages({
  library(terra)
  library(geodiv)
  library(spatialEco)
  library(jsonlite)
})

args <- commandArgs(TRUE)
job <- fromJSON(args[1], simplifyVector = FALSE)
out <- args[2]
source(args[3])  # geodiv with its errors corrected: fx$...

load_raster <- function(meta) {
  v <- readBin(meta$file, "double", n = meta$nrows * meta$ncols, endian = "little")
  v[is.nan(v)] <- NA
  crs <- if (is.null(meta$crs)) "" else meta$crs
  r <- rast(nrows = meta$nrows, ncols = meta$ncols, xmin = meta$xmin, xmax = meta$xmax,
            ymin = meta$ymin, ymax = meta$ymax, crs = crs)
  values(r) <- v
  r
}

manifest <- list()
counter <- 0L

write_array <- function(case, key, values, kind, extra = list()) {
  counter <<- counter + 1L
  file <- file.path(out, sprintf("a%06d.bin", counter))
  writeBin(as.double(values), file, endian = "little")
  manifest[[length(manifest) + 1L]] <<- c(list(case = case, key = key, kind = kind, file = basename(file),
                                              n = length(values)), extra)
}

dump <- function(case, key, x) {
  if (inherits(x, "SpatRaster")) {
    write_array(case, key, values(x)[, 1], "raster", list(dim = c(nrow(x), ncol(x))))
  } else if (is.data.frame(x)) {
    for (name in names(x)) dump(case, paste0(key, if (nchar(key)) "/" else "", name), x[[name]])
  } else if (is.matrix(x)) {
    write_array(case, key, t(x), "matrix", list(dim = dim(x)))
  } else if (is.list(x)) {
    for (name in names(x)) dump(case, paste0(key, if (nchar(key)) "/" else "", name), x[[name]])
  } else if (is.logical(x) || is.numeric(x)) {
    write_array(case, key, as.double(unname(x)), "vector")
  } else {
    stop("cannot dump ", class(x)[1], " (", case, "/", key, ")")
  }
}

rasters <- lapply(job$rasters, load_raster)

for (case in job$cases) {
  env <- new.env()
  for (name in names(rasters)) assign(name, rasters[[name]], envir = env)
  if (!is.null(case$raster)) assign("r", rasters[[case$raster]], envir = env)
  result <- tryCatch(withCallingHandlers(
    {
      # geodiv prints progress (bestfitplane's order, ...) with print()
      capture.output(value <- eval(parse(text = case$code), envir = env))
      value
    },
    message = function(m) invokeRestart("muffleMessage"),
    warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) stop("case ", case$name, ": ", conditionMessage(e), call. = FALSE))
  dump(case$name, "", result)
}

versions <- sapply(c("geodiv", "spatialEco", "terra"), function(p) as.character(packageVersion(p)))
writeLines(toJSON(list(manifest = manifest, versions = as.list(versions), r = R.version.string),
                  auto_unbox = TRUE, digits = NA), file.path(out, "manifest.json"))
