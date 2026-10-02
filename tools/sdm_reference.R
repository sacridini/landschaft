# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs the cases of tests/sdm_cases.py (driven by tools/make_reference_sdm.py):
# reads the job (JSON), rebuilds the inputs (tables: numeric columns from raw
# float64 files, NaN = NA, text columns from JSON; rasters: GeoTIFF read with
# terra; vectors: GeoPackage read with sf), evaluates each case's R code (a
# named list) and writes every result as raw float64 files (numbers, logicals)
# or JSON (text) plus a manifest.
#
#   Rscript tools/sdm_reference.R job.json out_dir

suppressMessages({
  library(jsonlite)
  library(terra)
  library(sf)
})

args <- commandArgs(TRUE)
job <- fromJSON(args[1], simplifyVector = FALSE)
out <- args[2]

read_table <- function(meta) {
  cols <- list()
  for (col in meta$columns) {
    if (col$type == "numeric") {
      v <- readBin(col$file, "double", n = meta$nrow, endian = "little")
      v[is.nan(v)] <- NA
      cols[[col$name]] <- v
    } else if (col$type == "logical") {
      v <- readBin(col$file, "double", n = meta$nrow, endian = "little")
      cols[[col$name]] <- ifelse(is.nan(v), NA, v != 0)
    } else {
      v <- unlist(lapply(col$values, function(s) if (is.null(s)) NA_character_ else s))
      cols[[col$name]] <- if (length(v)) v else character(0)
    }
  }
  as.data.frame(cols, stringsAsFactors = FALSE, check.names = FALSE)
}

load_input <- function(meta) {
  switch(meta$kind,
         table = read_table(meta),
         raster = rast(meta$file),
         vector = st_read(meta$file, quiet = TRUE),
         stop("unknown input kind ", meta$kind))
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

write_text <- function(case, key, values) {
  manifest[[length(manifest) + 1L]] <<- list(case = case, key = key, kind = "text",
                                             values = as.list(as.character(values)))
}

join <- function(key, name) paste0(key, if (nchar(key)) "/" else "", name)

dump <- function(case, key, x) {
  if (inherits(x, "SpatRaster")) {
    for (k in seq_len(nlyr(x))) {
      v <- values(x[[k]], mat = FALSE)
      write_array(case, join(key, names(x)[k]), v, "raster", list(dim = c(nrow(x), ncol(x))))
    }
  } else if (inherits(x, "sf") || inherits(x, "SpatVector")) {
    if (inherits(x, "SpatVector")) x <- st_as_sf(x)
    geom <- st_geometry(x)
    write_text(case, join(key, "geometry_wkt"), st_as_text(geom, digits = 17))
    dump(case, key, st_drop_geometry(x))
  } else if (is.data.frame(x)) {
    for (name in names(x)) dump(case, join(key, name), x[[name]])
  } else if (inherits(x, "dist")) {
    m <- as.matrix(x)
    write_array(case, key, t(m), "matrix", list(dim = dim(m)))
  } else if (is.matrix(x) && (is.numeric(x) || is.logical(x))) {
    write_array(case, key, t(x), "matrix", list(dim = dim(x)))
  } else if (is.list(x)) {
    nm <- names(x)
    if (is.null(nm)) nm <- as.character(seq_along(x))
    for (i in seq_along(x)) dump(case, join(key, nm[i]), x[[i]])
  } else if (is.factor(x) || is.character(x)) {
    write_text(case, key, as.character(x))
  } else if (is.logical(x) || is.numeric(x)) {
    write_array(case, key, as.double(x), "vector")
  } else if (is.null(x)) {
    write_array(case, key, numeric(0), "vector")
  } else {
    stop("cannot dump ", class(x)[1], " (", case, "/", key, ")")
  }
}

inputs <- lapply(job$inputs, load_input)

for (case in job$cases) {
  env <- new.env()
  for (name in names(inputs)) assign(name, inputs[[name]], envir = env)
  result <- tryCatch(withCallingHandlers(
    eval(parse(text = case$code), envir = env),
    message = function(m) invokeRestart("muffleMessage"),
    warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) stop("case ", case$name, ": ", conditionMessage(e), call. = FALSE))
  dump(case$name, "", result)
}

pkgs <- c("CoordinateCleaner", "spThin", "usdm", "blockCV", "ENMeval", "ecospat", "predicts", "dismo", "maxnet",
          "glmnet", "biomod2", "mop", "terra", "sf")
versions <- lapply(pkgs, function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA))
names(versions) <- pkgs
writeLines(toJSON(list(manifest = manifest, versions = versions, r = R.version.string),
                  auto_unbox = TRUE, digits = NA, null = "null"),
           file.path(out, "manifest.json"))
