# SPDX-License-Identifier: GPL-3.0-or-later
#
# Golden outputs of plangea's preparation functions (International Institute for Sustainability;
# GPL >= 3) for the restoration planning inputs: original vegetation (harmonize_oa), area of habitat
# (harmonize_bd), ecological intactness (multi_calc_it) and its uncertainty helpers.
#
# Usage: Rscript plangea_reference.R job.json out_dir   (run by tools/make_reference_plangea.py)
#
# plangea is not installed (most of its dependencies are missing): its R files are sourced from the
# folder in the job. Changes, all of them to plumbing or to the errors landschaft corrects:
# - aux_save / aux_load write and read RDS files (plangea needs the qs package);
# - harmonize_bd runs its foreach loops sequentially (no snow cluster, no devtools::load_all);
# - "fixed" variants: multi_calc_it with the window clipped at the raster's edges (plangea skips the
#   cells near the edges and one more row and column); harmonize_oa's equal ecoregion shares without
#   ecoregion maps; multi_exrisk_aggregate's square root after the loop over species; aux_calc_var_sd's
#   squared cross term and default deviation of the variable.

args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(jsonlite)
  library(raster)
  library(foreach)
  library(purrr)
  library(dplyr)
  library(tools)
})
job <- fromJSON(args[1], simplifyVector = TRUE, simplifyDataFrame = FALSE, simplifyMatrix = FALSE)
out_dir <- args[2]
pl <- job$plangea

read_src <- function(f) paste(readLines(file.path(pl, f), warn = FALSE), collapse = "\n")
patched <- function(f, subs, first_only = FALSE) {
  src <- read_src(f)
  for (s in subs) {
    if (!grepl(s[[1]], src, fixed = TRUE)) stop(paste0(f, ": anchor not found: ", s[[1]]))
    src <- if (first_only) sub(s[[1]], s[[2]], src, fixed = TRUE) else gsub(s[[1]], s[[2]], src, fixed = TRUE)
  }
  src
}
load_src <- function(src) eval(parse(text = src), envir = globalenv())

for (f in c("aux_load_raster.R", "harmonize_oa.R", "harmonize_bd_gen_usphab.R", "harmonize_bd_grep_raster_ext.R",
            "harmonize_bd_spid_constr.R", "multi_calc_bd.R", "multi_extinction_slope.R", "multi_extinction_risk.R",
            "multi_calc_it.R", "aux_plot_vals.R", "aux_calc_var_sd.R", "multi_exrisk_aggregate.R")) {
  load_src(read_src(f))
}
aux_save <- function(object, file) saveRDS(object, paste0(file, ".rds"))
aux_load <- function(file, ...) readRDS(if (file.exists(file)) file else paste0(file, ".rds"))

# harmonize_bd without its cluster
load_src(patched("harmonize_bd.R", list(
  c("cl = snow::makeCluster(threads, outfile = '')", "cl = NULL"),
  c("doSNOW::registerDoSNOW(cl)", "foreach::registerDoSEQ()"),
  c("snow::stopCluster(cl)", "invisible(NULL)"),
  c("devtools::load_all(cfg$dir$plangea_dir, quiet = TRUE)", "NULL"))))

# the corrected forms
harmonize_oa_fixed <- local({
  src <- patched("harmonize_oa.R", list(c(
    "Reduce('+', c_lu_vals[lu_types == \"N\"]) / length(which(lu_types == \"N\"))",
    "0 * x + 1 / length(which(lu_types == \"N\"))")))
  e <- new.env(); eval(parse(text = src), envir = e); f <- e$harmonize_oa; environment(f) <- globalenv(); f
})
multi_calc_it_fixed <- local({
  src <- patched("multi_calc_it.R", list(
    c("for (r in foc_index:(nrow(qm) - foc_index)){",
      paste0("qpad = matrix(NA, nrow(qm) + 2 * foc_dist, ncol(qm) + 2 * foc_dist)\n",
             "  qpad[(foc_dist + 1):(foc_dist + nrow(qm)), (foc_dist + 1):(foc_dist + ncol(qm))] = qm\n",
             "  for (r in 1:nrow(qm)){")),
    c("for (c in foc_index:(ncol(qm) - foc_index)){", "for (c in 1:ncol(qm)){"),
    c("total = (nrow(qm) - foc_index)", "total = nrow(qm)"),
    c("x = as.vector(qm[(r-foc_dist):(r+foc_dist), (c-foc_dist):(c+foc_dist)])",
      "x = as.vector(qpad[r:(r+2*foc_dist), c:(c+2*foc_dist)])")))
  e <- new.env(); eval(parse(text = src), envir = e); f <- e$multi_calc_it; environment(f) <- globalenv(); f
})
multi_exrisk_aggregate_fixed <- local({
  src <- patched("multi_exrisk_aggregate.R", list(
    c("if (quad_unc) {res_unc = sqrt(res_unc)}", ""),
    c("return(list(val = res, unc = res_unc))", "if (quad_unc) {res_unc = sqrt(res_unc)}; return(list(val = res, unc = res_unc))")))
  e <- new.env(); eval(parse(text = src), envir = e); f <- e$multi_exrisk_aggregate; environment(f) <- globalenv(); f
})
aux_calc_var_sd_fixed <- local({
  src <- patched("aux_calc_var_sd.R", list(
    c("sum(x_sd_vals   * var_sd_vals)", "sum(x_sd_vals^2 * var_sd_vals^2)"),
    c("var_sd_vals = 0.1 * sd(x_vals)", "var_sd_vals = 0.1 * sd(var_vals)")), first_only = TRUE)
  e <- new.env(); eval(parse(text = src), envir = e); f <- e$aux_calc_var_sd; environment(f) <- globalenv(); f
})
# post_aggregate_table_row's propagation of pixel uncertainties (two closures)
prop_closures <- function(px_area, x_vals) {
  lines <- readLines(file.path(pl, "post_aggregate_table_row.R"), warn = FALSE)
  e <- new.env()
  e$px_area <- px_area
  e$x_vals <- x_vals
  for (l in lines[grepl("^\\s*prop_(quad|lin) = function", lines)]) eval(parse(text = l), envir = e)
  list(quad = e$prop_quad, lin = e$prop_lin)
}

# ------------------------------------------------------------------ output
manifest <- list()
put <- function(case, key, x) {
  x <- as.numeric(unlist(x, use.names = FALSE))
  file <- sprintf("%04d.bin", length(manifest))
  writeBin(x, file.path(out_dir, file), size = 8, endian = "little")
  manifest[[length(manifest) + 1]] <<- list(case = case, key = key, file = file, n = length(x))
}
put_names <- function(case, key, x) {
  manifest[[length(manifest) + 1]] <<- list(case = case, key = key, kind = "text", text = as.list(as.character(x)))
}

# ------------------------------------------------------------------ cases
lu_raster <- function(path) raster(path)

for (case in job$cases) {
  name <- case$name
  tmp <- tempfile("plangea_"); dir.create(tmp)
  if (case$kind == "oa" || case$kind == "bd") {
    o <- case$oa
    lu_ras <- lapply(o$lu_files, function(f) aux_load_raster(f, na_fill = "zero"))
    names(lu_ras) <- o$class_names
    types <- unlist(o$class_types)
    optim <- types; optim[optim == "A"] <- "R"
    if ("D" %in% o$problem_type) optim[optim == "N"] <- "D"
    interest <- Reduce("+", lu_ras[optim %in% o$problem_type])
    master_index <- which(values(interest) > 0)
    terrestrial_index <- which(values(Reduce("+", lu_ras[types %in% c("N", "A")])) > 0)
    c_lu_vals <- lapply(lu_ras, function(x) x[master_index])
    cfg <- list(
      dir = list(past_lu_dir = paste0(o$past_dir, "/"), er_dir = paste0(o$er_dir, "/"), in_dir = paste0(tmp, "/")),
      landscape_features = list(
        land_use = list(class_names = o$class_names, class_types = types),
        original_areas = list(include_past = isTRUE(o$include_past), past_raster_names = unlist(o$past_files),
                              past_class_names = unlist(o$natural), include_ecoregions = isTRUE(o$include_er),
                              ecoregions_raster_names = unlist(o$er_files), failsafe_classes_to_fill = unlist(o$failsafe))),
      scenarios = list(problem_type = unlist(o$problem_type)))
    fn <- if (isTRUE(o$fixed)) harmonize_oa_fixed else harmonize_oa
    oa <- fn(cfg, file_log = list(), flag_log = list(master = TRUE), c_lu_vals = c_lu_vals, lu_types = types,
             master_index = master_index, verbose = FALSE, force_comp = TRUE)
    put(name, "master_index", master_index)
    for (cl in names(oa$oa_vals)) put(name, paste0("oa/", cl), oa$oa_vals[[cl]])
    for (j in seq_len(ncol(oa$prop_restore))) put(name, paste0("restore/", colnames(oa$prop_restore)[j]), oa$prop_restore[, j])
    if (case$kind == "bd") {
      b <- case$bd
      px_terr <- area(lu_ras[[1]])[terrestrial_index] * 1e-6
      px_area <- area(lu_ras[[1]])[master_index] * 1e-6
      lu_terr <- lapply(lu_ras, function(x) x[terrestrial_index])
      cfg$dir$spp_dir <- paste0(b$spp_dir, "/")
      cfg$dir$var_dir <- paste0(b$var_dir, "/")
      cfg$dir$plangea_dir <- pl
      cfg$variables <- list(calc_bd = list(
        bd_subfolders = unlist(b$subfolders), bd_classes = unlist(b$subfolders), threads = 1,
        include_hab_pot_areas = FALSE, include_hab_now_areas = FALSE,
        spp_table = list(spp_filename = b$table, spp_names_column = "taxonid", lu_names_column = "habitat_class",
                         include_absent_spp = isTRUE(b$include_absent), table_delim = ","),
        calc_extra_constraints = isTRUE(b$elevation),
        constraints_raster = b$elevation_file,
        constraints_tables = list(upperelevation_table = b$upper_table, lowerelevation_table = b$lower_table,
                                  upperelevation_column = "upperelevation", lowerelevation_column = "lowerelevation",
                                  spp_names_column = "taxonid", table_delim = ",")))
      bd <- harmonize_bd(cfg, file_log = list(), flag_log = list(master = TRUE), lu_terr = lu_terr,
                         master_index = master_index, terrestrial_index = terrestrial_index,
                         oa_vals = oa$oa_vals, prop_restore = oa$prop_restore, verbose = FALSE, force_comp = TRUE,
                         lu_vals = c_lu_vals, lu_class_types = types, threads = 1, px_area = px_area, px_terr = px_terr)
      valid <- names(bd$hab_now_areas)
      put_names(name, "species", valid)
      put(name, "hab_now", bd$hab_now_areas[valid])
      put(name, "hab_pot", bd$hab_pot_areas[valid])
      put(name, "bd", bd$bd)
      # the habitat gained per unit of a pixel restored: prop_restore %*% the species' classes, by its area
      for (i in seq_along(bd$usphab_index)) {
        for (s in bd$usphab_index[[i]]) {
          hab <- if (is.matrix(bd$usphab_proc)) oa$prop_restore %*% bd$usphab_proc[i, ] else oa$prop_restore
          inr <- master_index %in% bd$spp_main_range[[s]]
          put(name, paste0("gain_index/", s), master_index[inr])
          put(name, paste0("gain/", s), px_area[inr] * hab[inr, 1])
        }
      }
      if (isTRUE(b$exrisk)) {
        x <- unlist(b$x_vals)
        for (q in c(TRUE, FALSE)) {
          r <- multi_exrisk_aggregate_fixed(x_vals = x * px_area, prop_restore = oa$prop_restore,
                                            usphab_proc = bd$usphab_proc, usphab_index = bd$usphab_index,
                                            hab_now_areas = bd$hab_now_areas, hab_pot_areas = bd$hab_pot_areas,
                                            spp_main_range = bd$spp_main_range, master_index = master_index, quad_unc = q)
          put(name, paste0("exrisk_", if (q) "quad" else "lin"), c(r$val, r$unc))
        }
        put(name, "x_vals", x)
      }
    }
  } else if (case$kind == "it") {
    hfi <- raster(case$hfi)
    base_ras <- 0 * hfi
    base_ras[is.na(hfi)] <- NA
    master_index <- which(!is.na(values(hfi)))
    saveRDS(list(terrestrial_index = master_index), file.path(tmp, "harmonize_lu.rds"))
    cfg <- list(dir = list(in_dir = paste0(tmp, "/")),
                scenarios = list(problem_type = "R"),
                variables = list(variable_names = "it", calc_it = list(n_pixels = case$radius, it_variable_name = "it",
                                                                        hfi_weights = unlist(case$hfi_weights))),
                uncertainties = list(quadratic_propagation = TRUE))
    pc <- NULL
    if (!is.null(case$prop_convert)) pc <- sapply(case$prop_convert, function(f) raster(f)[master_index])
    x <- if (is.null(case$x_vals)) 0 else unlist(case$x_vals)
    run <- function(f, xv) f(cfg = cfg, base_ras = base_ras, master_index = master_index, HFI = hfi, prop_convert = pc,
                              x_vals = xv, gamma = case$gamma, beta = case$beta, z = case$z, verbose = FALSE,
                              aggregate = TRUE)
    put(name, "master_index", master_index)
    put(name, "fixed", run(multi_calc_it_fixed, x))
    put(name, "original", suppressWarnings(run(multi_calc_it, x)))
    if (isTRUE(case$gradient)) {
      h <- case$step
      g <- numeric(length(master_index))
      for (k in seq_along(master_index)) {
        xp <- x; xm <- x
        xp[k] <- xp[k] + h; xm[k] <- xm[k] - h
        g[k] <- (sum(run(multi_calc_it_fixed, xp)) - sum(run(multi_calc_it_fixed, xm))) / (2 * h)
      }
      put(name, "gradient", g)
    }
  } else if (case$kind == "unc") {
    v <- unlist(case$v); a <- unlist(case$px_area); x <- unlist(case$x_vals)
    pc <- prop_closures(a, x)
    put(name, "prop_quad", pc$quad(v))
    put(name, "prop_lin", pc$lin(v))
    xv <- unlist(case$var_x); vv <- unlist(case$var_v); sx <- unlist(case$sd_x); sv <- unlist(case$sd_v)
    put(name, "var_sd_quad", aux_calc_var_sd_fixed(xv, vv, sx, sv, quad = TRUE))
    put(name, "var_sd_lin", aux_calc_var_sd_fixed(xv, vv, sx, sv, quad = FALSE))
    put(name, "var_sd_quad_original", aux_calc_var_sd(xv, vv, sx, sv, quad = TRUE))
  }
}

writeLines(toJSON(list(manifest = manifest, r = R.version.string,
                       hashes = as.list(sapply(list.files(pl, pattern = "\\.R$"), function(f) unname(tools::md5sum(file.path(pl, f)))))),
                  auto_unbox = TRUE, digits = NA), file.path(out_dir, "manifest.json"))
