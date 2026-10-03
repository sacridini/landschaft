# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs the cases of tests/optimize_cases.py (driven by
# tools/make_reference_optimize.py): reads the job (JSON), rebuilds the inputs
# (tables: numeric columns from raw float64 files, NaN = NA, text columns from
# JSON; rasters: GeoTIFF read with terra; vectors: GeoPackage read with sf),
# evaluates each case's R code (a named list) and writes every result as raw
# float64 files (numbers, logicals) or JSON (text) plus a manifest. Compiled
# models (prioritizr and prioriactions) are written as their constraint matrix
# in 0-based triplets and their vectors.
#
#   Rscript tools/optimize_reference.R job.json out_dir

suppressMessages({
  library(jsonlite)
  library(terra)
  library(sf)
  library(Matrix)
  library(prioritizr)  # prioriactions is used through prioriactions:: (it masks problem() and solve())
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
         vector = read_sf(meta$file),
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

# the fields of a compiled model, the same for both packages
model_fields <- function(x) {
  if (inherits(x, "R6")) {  # prioritizr
    l <- as.list(x)
    return(list(A_i = l$A_i, A_j = l$A_j, A_x = l$A_x, obj = l$obj, lb = l$lb, ub = l$ub, rhs = l$rhs,
                sense = l$sense, vtype = l$vtype, modelsense = l$modelsense, row_ids = l$row_ids,
                col_ids = l$col_ids))
  }
  l <- x$getDataList()  # prioriactions
  A <- as(l$A, "TsparseMatrix")
  n <- ncol(A)
  lb <- rep(0, n)
  ub <- rep(Inf, n)
  if (!is.null(l$bounds$lower)) lb[l$bounds$lower$ind] <- l$bounds$lower$val
  if (!is.null(l$bounds$upper)) ub[l$bounds$upper$ind] <- l$bounds$upper$val
  out <- list(A_i = A@i, A_j = A@j, A_x = A@x, obj = l$obj, lb = lb, ub = ub, rhs = l$rhs, sense = l$sense,
              vtype = l$vtype, modelsense = l$modelsense, n_row = nrow(A), n_col = n)
  if (length(l$genconpow)) {  # y = x^a (Gurobi general constraints); xvar, yvar 1-based as in Gurobi's R API
    g <- l$genconpow
    out$genconpow <- list(xvar = sapply(g, `[[`, "xvar"), yvar = sapply(g, `[[`, "yvar"), a = sapply(g, `[[`, "a"))
  }
  out
}

dump <- function(case, key, x) {
  if (inherits(x, "OptimizationProblem")) x <- model_fields(x)
  if (inherits(x, "SpatRaster")) {
    for (k in seq_len(nlyr(x))) {
      v <- values(x[[k]], mat = FALSE)
      write_array(case, join(key, names(x)[k]), v, "raster", list(dim = c(nrow(x), ncol(x))))
    }
  } else if (is.data.frame(x)) {
    if (inherits(x, "sf")) x <- st_drop_geometry(x)
    for (name in names(x)) dump(case, join(key, name), x[[name]])
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

# --- helpers the cases use

# the solution in planning-unit order (the cells or rows with a cost, in order)
solution_values <- function(s, p) {
  if (inherits(s, "SpatRaster")) {
    return(values(s[[1]], mat = FALSE)[p$planning_unit_indices()])
  }
  if (inherits(s, "sf")) s <- st_drop_geometry(s)
  s[["solution_1"]][p$planning_unit_indices()]
}

reference <- function(p) {
  model <- compile(p)
  s <- solve(add_gurobi_solver(p, gap = 0, threads = 1, verbose = FALSE))
  list(model = model, objective = unname(attr(s, "objective")), status = unname(attr(s, "status")),
       solution = solution_values(s, p))
}

actions_reference <- function(m) {
  s <- prioriactions::solve(m, solver = "gurobi", gap_limit = 0, cores = 1, output_file = FALSE,
                            verbose = FALSE)
  list(model = m, objective = s$data$objval, status = s$data$status, gap = s$data$gap,
       solution = s$data$sol)
}

pa_data <- function(pu = pa_pu, features = pa_features, dist_features = pa_dist_features, threats = pa_threats,
                    dist_threats = pa_dist_threats, sensitivity = pa_sensitivity, boundary = pa_boundary) {
  prioriactions::inputData(pu = pu, features = features, dist_features = dist_features, threats = threats,
                           dist_threats = dist_threats, sensitivity = sensitivity, boundary = boundary)
}

inputs <- lapply(job$inputs, load_input)

for (case in job$cases) {
  env <- new.env()
  for (name in names(inputs)) assign(name, inputs[[name]], envir = env)
  # the pa_* defaults of pa_data() are looked up where the case runs
  environment(pa_data) <- env
  assign("pa_data", pa_data, envir = env)
  result <- tryCatch(withCallingHandlers(
    eval(parse(text = case$code), envir = env),
    message = function(m) invokeRestart("muffleMessage"),
    warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) stop("case ", case$name, ": ", conditionMessage(e), call. = FALSE))
  dump(case$name, "", result)
}

pkgs <- c("prioritizr", "prioriactions", "gurobi", "terra", "sf", "exactextractr", "Matrix")
versions <- lapply(pkgs, function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA))
names(versions) <- pkgs
writeLines(toJSON(list(manifest = manifest, versions = versions, r = R.version.string),
                  auto_unbox = TRUE, digits = NA, null = "null"),
           file.path(out, "manifest.json"))
