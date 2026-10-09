# SPDX-License-Identifier: GPL-3.0-or-later
#
# Builds one benchmark problem of tools/benchmark_optimize.py with prioritizr or
# prioriactions, times its compilation and writes the compiled model as raw
# files (A_i, A_j: int32 0-based triplets; A_x, obj, lb, ub, rhs: float64;
# sense, vtype: one character per row / column; modelsense: text); the time
# covers problem(), the add_* calls and compile() (prioriactions: inputData()
# and problem()), the rasters already opened. With
# "solve": true it also times the package's own solve on Gurobi (build +
# solve, as a user would run it).
#
#   Rscript tools/optimize_benchmark.R job.json out_dir

suppressMessages({
  library(jsonlite)
  library(terra)
  library(Matrix)
  library(prioritizr)
})

args <- commandArgs(TRUE)
job <- fromJSON(args[1], simplifyVector = TRUE)
out <- args[2]

read_table <- function(meta) {
  cols <- list()
  for (i in seq_along(meta$columns$name)) {
    v <- readBin(meta$columns$file[i], "double", n = meta$nrow, endian = "little")
    cols[[meta$columns$name[i]]] <- v
  }
  as.data.frame(cols, check.names = FALSE)
}

build <- function(job) {
  if (job$family == "actions") {
    t <- lapply(job$tables, read_table)
    t$features$name <- paste0("feature", t$features$id)
    t$threats$name <- paste0("threat", t$threats$id)
    return(function() suppressWarnings({
      d <- prioriactions::inputData(pu = t$pu, features = t$features, dist_features = t$dist_features,
                                    threats = t$threats, dist_threats = t$dist_threats, sensitivity = t$sensitivity)
      prioriactions::problem(d, model_type = "minimizeCosts")
    }))
  }
  cost <- rast(job$cost)
  feats <- rast(job$features)
  if (startsWith(job$family, "zones_")) {
    # three zones: protection (the features), use (half of them), restoration (the gain of them)
    zn <- c("protect", "use", "restore")
    costs <- c(cost, rast(job$zones$cost_use), rast(job$zones$cost_restore))
    gain <- rast(job$zones$gain)
    totals <- global(feats, "sum", na.rm = TRUE)[[1]]
    return(function() {
      z <- zones(feats, feats * 0.5, feats * gain, zone_names = zn, feature_names = names(feats))
      p <- problem(costs, z)
      p <- switch(job$family,
        zones_minset = , zones_boundary = p |> add_min_set_objective() |>
          add_manual_targets(tibble::tibble(feature = names(feats), zone = rep(list(zn), nlyr(feats)),
                                            type = "absolute", target = job$target * totals)),
        zones_maxutil = p |> add_max_utility_objective(budget = job$budget),
        stop("unknown family ", job$family))
      if (job$family == "zones_boundary") p <- p |> add_boundary_penalties(job$penalty, zones = diag(3))
      p |> add_binary_decisions()
    })
  }
  function() {
    p <- problem(cost, feats)
    p <- switch(job$family,
      minset = , boundary = , contiguity = p |> add_min_set_objective() |> add_relative_targets(job$target),
      maxutil = p |> add_max_utility_objective(budget = job$budget),
      stop("unknown family ", job$family))
    if (job$family == "boundary") p <- p |> add_boundary_penalties(job$penalty)
    if (job$family == "contiguity") p <- p |> add_contiguity_constraints()
    p |> add_binary_decisions()
  }
}

write_model <- function(m, actions) {
  if (actions) {
    l <- m$getDataList()
    A <- as(l$A, "TsparseMatrix")
    n <- ncol(A)
    lb <- rep(0, n)
    ub <- rep(Inf, n)
    if (!is.null(l$bounds$lower)) lb[l$bounds$lower$ind] <- l$bounds$lower$val
    if (!is.null(l$bounds$upper)) ub[l$bounds$upper$ind] <- l$bounds$upper$val
    f <- list(A_i = A@i, A_j = A@j, A_x = A@x, obj = l$obj, lb = lb, ub = ub, rhs = l$rhs, sense = l$sense,
              vtype = l$vtype, modelsense = l$modelsense)
  } else {
    f <- as.list(m)
  }
  writeBin(as.integer(f$A_i), file.path(out, "A_i.bin"), size = 4L, endian = "little")
  writeBin(as.integer(f$A_j), file.path(out, "A_j.bin"), size = 4L, endian = "little")
  for (k in c("A_x", "obj", "lb", "ub", "rhs")) writeBin(as.double(f[[k]]), file.path(out, paste0(k, ".bin")),
                                                       endian = "little")
  s <- ifelse(f$sense == "<=", "L", ifelse(f$sense == ">=", "G", "E"))
  writeLines(paste(s, collapse = ""), file.path(out, "sense.txt"))
  writeLines(paste(f$vtype, collapse = ""), file.path(out, "vtype.txt"))
  writeLines(f$modelsense, file.path(out, "modelsense.txt"))
  c(nrow = length(f$rhs), ncol = length(f$obj), nnz = length(f$A_x))
}

actions <- job$family == "actions"
make <- build(job)
t0 <- Sys.time()
p <- make()
m <- if (actions) p else compile(p)
compile_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
dims <- write_model(m, actions)
res <- list(compile_seconds = compile_seconds, nrow = dims[["nrow"]], ncol = dims[["ncol"]], nnz = dims[["nnz"]])

if (isTRUE(job$solve)) {
  t0 <- Sys.time()
  if (actions) {
    s <- prioriactions::solve(make(), solver = "gurobi", gap_limit = job$gap, time_limit = job$time_limit,
                              cores = job$threads, output_file = FALSE, verbose = FALSE)
    res$solve <- list(objective = s$data$objval, gap = s$data$gap, status = s$data$status)
  } else {
    s <- solve(make() |> add_gurobi_solver(gap = job$gap, time_limit = job$time_limit, threads = job$threads,
                                           verbose = FALSE), run_checks = FALSE)
    res$solve <- list(objective = unname(attr(s, "objective")), gap = unname(attr(s, "gap")),
                      status = unname(attr(s, "status")))
  }
  res$solve$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
}

res$versions <- list(prioritizr = as.character(packageVersion("prioritizr")),
                     prioriactions = as.character(packageVersion("prioriactions")),
                     gurobi = as.character(packageVersion("gurobi")))
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null"), file.path(out, "result.json"))
