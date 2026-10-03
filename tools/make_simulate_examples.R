# SPDX-License-Identifier: GPL-3.0-or-later
#
# Writes the example data of lulcc (GPL >= 2, Moulds 2015) shipped as
# ls.simulate.example_data(): Plum Island Ecosystems (Pontius & Parmentier
# 2014) and Sibuyan Island (Verburg et al. 2002), as GeoTIFFs and a CSV.
#
#   Rscript tools/make_simulate_examples.R path/to/lulcc/data out_dir
#
# lulcc's data/*.rda hold raster package RasterLayers; land use is written as
# uint8 (NA = 255), drivers as float32 (NA = NaN), band names kept.

suppressMessages({
  library(raster)
  library(terra)
})

args <- commandArgs(TRUE)
src <- args[1]
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

stack_of <- function(layers, names) {
  r <- rast(lapply(layers, rast))
  names(r) <- names
  r
}

write_lu <- function(r, file) {
  writeRaster(r, file.path(out, file), datatype = "INT1U", NAflag = 255, overwrite = TRUE,
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2"))
}

write_float <- function(r, file) {
  writeRaster(r, file.path(out, file), datatype = "FLT4S", overwrite = TRUE,
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3"))
}

e <- new.env()
suppressWarnings(load(file.path(src, "pie.rda"), envir = e))
pie <- e$pie
write_lu(stack_of(pie[c("lu_pie_1985", "lu_pie_1991", "lu_pie_1999")], c("1985", "1991", "1999")),
         "pie_landuse.tif")
write_float(stack_of(pie[c("ef_001", "ef_002", "ef_003")], c("elevation", "slope", "dist_built_1985")),
            "pie_drivers.tif")

suppressWarnings(load(file.path(src, "sibuyan.rda"), envir = e))
sib <- e$sibuyan
m <- sib$maps
write_lu(stack_of(m[c("lu_sib_1997", "lu_sib_2011")], c("1997", "2011")), "sibuyan_landuse.tif")
ef_names <- c("dist_sea", "pop_density", "diorite", "ultramafic", "sediments", "no_erosion",
              "moderate_erosion", "elevation", "slope", "aspect", "dist_roads", "dist_urban", "dist_streams")
write_float(stack_of(m[sprintf("ef_%03d", 1:13)], ef_names), "sibuyan_drivers.tif")
write_lu(stack_of(m[c("restr1", "restr2")], c("national_park", "proposed_park")), "sibuyan_restrictions.tif")

labels <- c("forest", "coconut", "grassland", "rice", "other")
demand <- do.call(rbind, lapply(names(sib$demand), function(s) {
  d <- as.data.frame(sib$demand[[s]])
  names(d) <- labels
  cbind(scenario = s, time = seq_len(nrow(d)) - 1L, d)
}))
write.csv(demand, file.path(out, "sibuyan_demand.csv"), row.names = FALSE)
