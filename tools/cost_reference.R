# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs the gdistance cases of tests/cost_cases.py (driven by
# tools/make_reference_cost.py): reads the job (JSON), rebuilds each raster
# from its values (float64, row-major, NaN = NA), evaluates each case's R code
# (a named list) and writes every result as raw float64 files plus a manifest.
#
#   Rscript tools/cost_reference.R job.json out_dir

suppressMessages({
  library(gdistance)
  library(jsonlite)
})

args <- commandArgs(TRUE)
job <- fromJSON(args[1], simplifyVector = FALSE)
out <- args[2]

read_values <- function(meta) {
  v <- readBin(meta$file, "double", n = meta$nrows * meta$ncols * meta$nlayers, endian = "little")
  v[is.nan(v)] <- NA
  v
}

load_raster <- function(meta) {
  crs <- if (is.null(meta$crs)) NA else meta$crs
  make <- function(values) {
    r <- raster(nrows = meta$nrows, ncols = meta$ncols, xmn = meta$xmin, xmx = meta$xmax,
                ymn = meta$ymin, ymx = meta$ymax, crs = crs)
    r[] <- values
    r
  }
  v <- read_values(meta)
  if (meta$nlayers == 1) return(make(v))
  n <- meta$nrows * meta$ncols
  brick(lapply(seq_len(meta$nlayers), function(k) make(v[((k - 1) * n + 1):(k * n)])))
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
  if (is(x, "TransitionLayer")) {
    tm <- transitionMatrix(x)
    g <- as(as(tm, "generalMatrix"), "TsparseMatrix")
    extra <- list(symmetric = is(tm, "symmetricMatrix"), matrix_values = matrixValues(x),
                  ncell = ncell(x))
    write_array(case, paste0(key, "/i"), g@i, "transition", extra)
    write_array(case, paste0(key, "/j"), g@j, "transition", extra)
    write_array(case, paste0(key, "/x"), g@x, "transition", extra)
  } else if (is(x, "TransitionStack")) {
    for (k in seq_len(nlayers(x))) dump(case, paste0(key, "/", k), x[[k]])
  } else if (is(x, "RasterLayer")) {
    write_array(case, key, getValues(x), "raster", list(dim = c(nrow(x), ncol(x))))
  } else if (inherits(x, "dist")) {
    m <- as.matrix(x)
    write_array(case, key, t(m), "matrix", list(dim = dim(m)))
  } else if (is.matrix(x) || is(x, "Matrix")) {
    m <- as.matrix(x)
    write_array(case, key, t(m), "matrix", list(dim = dim(m)))
  } else if (is.list(x)) {
    for (name in names(x)) dump(case, paste0(key, if (nchar(key)) "/" else "", name), x[[name]])
  } else if (is.logical(x) || is.numeric(x)) {
    write_array(case, key, as.double(x), "vector")
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
    eval(parse(text = case$code), envir = env),
    message = function(m) invokeRestart("muffleMessage"),
    warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) stop("case ", case$name, ": ", conditionMessage(e), call. = FALSE))
  dump(case$name, "", result)
}

writeLines(toJSON(manifest, auto_unbox = TRUE, digits = NA), file.path(out, "manifest.json"))
