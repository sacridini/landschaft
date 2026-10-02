# SPDX-License-Identifier: GPL-3.0-or-later
#
# CoordinateCleaner 3.0.1 functions with their errors corrected ("correct
# form"): every function is the original code with the minimal change marked
# "FIX". Sourced by the ls.sdm reference cases (tests/sdm_cases.py) after
# library(CoordinateCleaner); the originals stay available as
# CoordinateCleaner::cc_*. The errors are listed in IDEAS.md and in the
# docstring of ls.sdm.clean_occurrences.

suppressMessages({library(terra); library(CoordinateCleaner)})
.wgs84 <- "+proj=longlat +datum=WGS84 +no_defs"

# ---------------------------------------------------------------------------
# geod = TRUE polygons (73 vertices, bearings 0..360 by 5, GeographicLib 2.x)
# FIX (B10): dateline-crossing and pole-containing circles.  For ordinary
# circles the vertices are bit-identical to the original.
.geod_polys <- function(xy, buffer) {
  xy <- matrix(xy, ncol = 2)
  dg <- seq(from = 0, to = 360, by = 5)
  n <- nrow(xy)
  buff_XY <- geosphere::destPoint(p = xy, b = rep(dg, each = n), d = buffer)
  id <- rep(seq_len(n), times = length(dg))
  lst <- lapply(split(data.frame(buff_XY), f = id), as.matrix)
  out <- list()
  for (k in seq_len(n)) {
    m <- lst[[k]]
    lon0 <- xy[k, 1]
    to_pole_n <- geosphere::distGeo(xy[k, ], c(lon0, 90))
    to_pole_s <- geosphere::distGeo(xy[k, ], c(lon0, -90))
    if (to_pole_n < buffer || to_pole_s < buffer) {
      # polar cap: sort vertices by lon and close along the pole edge
      pole <- if (to_pole_n < buffer) 90 else -90
      m <- m[-nrow(m), , drop = FALSE]
      m <- m[order(m[, 1]), , drop = FALSE]
      m <- rbind(c(-180, m[1, 2]), m, c(180, m[nrow(m), 2]), c(180, pole),
                 c(-180, pole), c(-180, m[1, 2]))
      out[[length(out) + 1]] <- vect(m, type = "polygons", crs = .wgs84)
    } else if (diff(range(m[, 1])) > 180) {
      # unwrap around the centre and add the copy shifted by -/+360
      d <- m[, 1] - lon0
      m[d > 180, 1] <- m[d > 180, 1] - 360
      m[d < -180, 1] <- m[d < -180, 1] + 360
      s <- if (lon0 > 0) -360 else 360
      m2 <- m; m2[, 1] <- m2[, 1] + s
      out[[length(out) + 1]] <- vect(list(vect(m, type = "polygons"),
                                          vect(m2, type = "polygons")))
      out[[length(out)]] <- aggregate(out[[length(out)]])
      crs(out[[length(out)]]) <- .wgs84
    } else {
      out[[length(out) + 1]] <- vect(m, type = "polygons", crs = .wgs84)
    }
  }
  vect(out)
}

.flag_by_polys <- function(polys, dat) {
  ext_dat <- terra::extract(polys, dat)
  is.na(ext_dat[!duplicated(ext_dat[, 1]), 2])
}

# ---------------------------------------------------------------------------
# B1: cc_zero buffer in decimal degrees (planar GEOS circle, quadsegs = 10),
# as documented.  terra >= 1.8 guesses a lon/lat CRS for the 0/0 point and
# silently turns the 0.5 into 0.5 metres.
cc_zero_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          buffer = 0.5, value = "clean", verbose = TRUE) {
  match.arg(value, choices = c("clean", "flagged"))
  t1 <- !(x[[lon]] == 0 | x[[lat]] == 0)
  dat <- terra::vect(x[, c(lon, lat)], geom = c(lon, lat), crs = "")      # FIX
  if (buffer == 0) buffer <- 0.00000000000001
  buff <- terra::buffer(terra::vect(data.frame(lon = 0, lat = 0), crs = ""), # FIX
                        width = buffer)
  ext_dat <- suppressWarnings(terra::extract(buff, dat))
  t2 <- is.na(ext_dat[!duplicated(ext_dat[, 1]), 2])
  out <- t1 & t2
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

# ---------------------------------------------------------------------------
# Shared core of cc_cap / cc_cen / cc_gbif / cc_aohi with the B10 fix.
.cc_point_ref <- function(x, lon, lat, ref_xy, buffer, geod, crop = TRUE) {
  dat <- terra::vect(x[, c(lon, lat)], geom = c(lon, lat), crs = .wgs84)
  buffer <- ifelse(buffer == 0, 0.00000000001, buffer)
  ref <- terra::vect(data.frame(lon = ref_xy[, 1], lat = ref_xy[, 2]),
                     geom = c("lon", "lat"), crs = .wgs84)
  if (crop) {
    limits <- terra::ext(terra::buffer(dat, width = buffer))
    ref <- terra::crop(ref, limits)
  }
  if (nrow(ref) == 0) return(rep(TRUE, nrow(x)))
  if (geod) {
    .flag_by_polys(.geod_polys(terra::geom(ref)[, c("x", "y")], buffer), dat)
  } else {
    .flag_by_polys(terra::buffer(ref, width = buffer), dat)
  }
}

cc_cap_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                         species = "species", buffer = 10000, geod = TRUE,
                         ref = NULL, verify = FALSE, value = "clean") {
  if (is.null(ref)) {
    ref <- CoordinateCleaner::countryref
    ref <- ref[!is.na(ref$capital), ]
  }
  out <- .cc_point_ref(x, lon, lat, cbind(ref$capital.lon, ref$capital.lat),
                       buffer, geod)
  if (verify & sum(out) > 0) out <- .verify_counts(x, out, lon, lat, species)
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

# B6: test = "provinces" selected nothing (switch label was "province").
cc_cen_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                         species = "species", buffer = 1000, geod = TRUE,
                         test = "both", ref = NULL, verify = FALSE,
                         value = "clean") {
  match.arg(test, choices = c("both", "country", "provinces"))
  if (is.null(ref)) {
    ref <- CoordinateCleaner::countryref
    switch(test, country = {
      ref <- ref[ref$type == "country", ]
    }, provinces = {                                                       # FIX
      ref <- ref[ref$type == "province", ]
    })
  }
  out <- .cc_point_ref(x, lon, lat, cbind(ref$centroid.lon, ref$centroid.lat),
                       buffer, geod)
  if (verify & sum(out) > 0) out <- .verify_counts(x, out, lon, lat, species)
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

cc_gbif_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          species = "species", buffer = 1000, geod = TRUE,
                          verify = FALSE, value = "clean") {
  out <- .cc_point_ref(x, lon, lat, cbind(12.58, 55.67), buffer, geod,
                       crop = FALSE)
  if (verify & sum(out) > 0) out <- .verify_counts(x, out, lon, lat, species)
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

cc_aohi_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          species = "species",
                          taxa = c("Aves", "Insecta", "Mammalia", "Plantae"),
                          buffer = 10000, geod = TRUE, value = "clean") {
  aohi <- get0("aohi", envir = asNamespace("CoordinateCleaner"))
  aohi <- aohi[aohi$taxa %in% taxa, ]
  out <- .cc_point_ref(x, lon, lat,
                       cbind(aohi$decimalLongitude, aohi$decimalLatitude),
                       buffer, geod, crop = FALSE)
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

# verify = TRUE logic of cc_cap / cc_cen / cc_gbif (unchanged semantics)
.verify_counts <- function(x, out, lon, lat, species) {
  ver <- x[!out, ]
  ver_count <- aggregate(ver[[species]] ~ ver[[lon]] + ver[[lat]], FUN = "length")
  names(ver_count) <- c(lon, lat, "coord.count")
  ver_spec <- aggregate(ver[[lon]] ~ ver[[species]], FUN = "length")
  names(ver_spec) <- c(species, "species.count")
  tester <- data.frame(x, ord = seq_len(nrow(x)))
  tester <- merge(tester, ver_count, by = c(lon, lat), all = TRUE)
  tester <- merge(tester, ver_spec, by = species, all = TRUE)
  tester <- tester[order(tester$ord), ]
  tester[is.na(tester)] <- 0
  tester$coord.count <= tester$species.count | out
}

# ---------------------------------------------------------------------------
# B9: cc_inst verify.  geod = TRUE used the institution polygons instead of
# the per-record verification polygons (error); geod = FALSE appended the
# result as new named elements instead of overwriting the flagged positions,
# and buffered without a CRS (units depended on the column names).
cc_inst_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          species = "species", buffer = 100, geod = FALSE,
                          ref = NULL, verify = FALSE, verify_mltpl = 10,
                          value = "clean") {
  buffer <- ifelse(buffer == 0, 0.00000000001, buffer)
  dat <- terra::vect(x[, c(lon, lat)], geom = c(lon, lat), crs = .wgs84)
  if (is.null(ref)) {
    ref <- CoordinateCleaner::institutions
    ref <- ref[!is.na(ref$decimalLongitude) & !is.na(ref$decimalLatitude), ]
  }
  limits <- terra::ext(dat) + buffer
  ref <- terra::crop(terra::vect(as.data.frame(ref)[, c("decimalLongitude", "decimalLatitude")],
                                 geom = c("decimalLongitude", "decimalLatitude"),
                                 crs = .wgs84), limits)
  if (nrow(ref) == 0) {
    out <- rep(TRUE, nrow(x))
  } else if (geod) {
    out <- .flag_by_polys(.geod_polys(terra::geom(ref)[, c("x", "y")], buffer), dat)
  } else {
    out <- .flag_by_polys(terra::buffer(ref, width = buffer), dat)
  }
  if (verify) {
    idx <- which(!out)                                                     # FIX
    if (length(idx) > 0) {
      ref_in <- x[idx, ]
      xy <- as.matrix(ref_in[, c(lon, lat)])
      if (geod) {
        vpol <- .geod_polys(xy, buffer * verify_mltpl)                     # FIX
      } else {
        vpol <- terra::buffer(terra::vect(xy, crs = .wgs84),               # FIX
                              width = buffer * verify_mltpl)
      }
      res <- logical(length(idx))
      for (i in seq_along(idx)) {
        same <- x[[species]] == ref_in[[species]][i]
        pts <- terra::vect(as.matrix(x[same, c(lon, lat)]), crs = .wgs84)
        hit <- terra::extract(vpol[i], pts)
        n_in <- sum(!is.na(hit[!duplicated(hit[, 1]), 2]))
        n_same <- sum(ref_in[[species]] == ref_in[[species]][i] &
                      ref_in[[lon]] == ref_in[[lon]][i] &
                      ref_in[[lat]] == ref_in[[lat]][i])
        res[i] <- n_in > n_same
      }
      out[idx] <- res                                                      # FIX
    }
  }
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

# ---------------------------------------------------------------------------
# B7: cc_coun without buffer misaligned the output when a point is covered by
# more than one polygon (shared borders).  Fixed: a record passes when its
# code matches ANY covering polygon (same rule as the buffer branch); a
# missing code never passes.
cc_coun_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          iso3 = "countrycode", value = "clean", ref = NULL,
                          ref_col = "iso_a3", buffer = NULL) {
  if (is.null(ref)) {
    ref <- terra::vect(rnaturalearth::ne_countries(scale = "medium", returnclass = "sf"))
  }
  dat <- terra::vect(x[, c(lon, lat)], geom = c(lon, lat), crs = ref)
  if (is.numeric(buffer)) {
    buffer <- ifelse(buffer == 0, 0.00000000001, buffer)
    ref_buff <- terra::buffer(ref, buffer)
    ref <- terra::vect(stats::na.omit(terra::geom(ref_buff)), type = "polygon", crs = ref)
    terra::values(ref) <- terra::values(ref_buff)
  }
  country <- terra::extract(ref, dat)
  count_dat <- as.character(unlist(x[, iso3]))
  out <- logical(length(dat))
  for (i in seq_along(dat)) {                                              # FIX
    codes <- country[country[, 1] == i, ref_col]
    out[i] <- !is.na(count_dat[i]) && count_dat[i] %in% codes[!is.na(codes)]
  }
  switch(value, clean = return(x[out, ]), flagged = return(out))
}

# ---------------------------------------------------------------------------
# B3 + B4: cc_outl.  (a) value = "ids" returned ALL ids when every species
# had fewer than min_occs unique records -> clean_coordinates flagged them all;
# (b) the raster/thinning branch never flagged anything (`pts %in% ...` on a
# data.frame).  The sampling-correction branch is not reproduced here.
cc_outl_fixed <- function(x, lon = "decimalLongitude", lat = "decimalLatitude",
                          species = "species", method = "quantile", mltpl = 5,
                          tdi = 1000, value = "clean", min_occs = 7,
                          thinning = FALSE, thinning_res = 0.5) {
  init_ids <- rownames(x)
  rownames(x) <- NULL
  splist <- split(x, f = as.character(x[[species]]))
  test <- lapply(splist, function(k) duplicated(k[, c(species, lon, lat)]))
  test <- as.vector(unlist(lapply(lapply(test, "!"), "sum")))
  splist <- splist[test >= min_occs]
  if (all(test < min_occs)) {
    switch(value, clean = return(x), flagged = return(rep(TRUE, nrow(x))),
           ids = return(character(0)))                                     # FIX (a)
  }
  record_numbers <- unlist(lapply(splist, nrow))
  if (any(record_numbers >= 10000) | thinning) {
    ras <- CoordinateCleaner:::ras_create(x = x, lat = lat, lon = lon,
                                          thinning_res = thinning_res)
  }
  flags <- lapply(splist, function(k) {
    raster_flag <- nrow(k) >= 10000 | thinning
    if (raster_flag) {
      dist_obj <- CoordinateCleaner:::ras_dist(x = k, lat = lat, lon = lon,
                                                ras = ras, weights = !thinning)
      pts <- dist_obj$pts
      dist <- dist_obj$dist
      wm <- dist_obj$wm
    } else {
      dist <- geosphere::distm(k[, c(lon, lat)], fun = geosphere::distHaversine) / 1000
      dist[dist == 0] <- NA
    }
    if (method == "distance") {
      mins <- apply(dist, 1, min, na.rm = TRUE)
      out <- which(mins > tdi)
    } else if (raster_flag & !thinning) {
      mins <- apply(dist, 1, sum, na.rm = TRUE) / rowSums(wm, na.rm = TRUE)
    } else {
      mins <- apply(dist, 1, mean, na.rm = TRUE)
    }
    if (method == "quantile") {
      quo <- quantile(mins, c(0.25, 0.75), na.rm = TRUE)
      out <- which(mins > (quo[2] + stats::IQR(mins) * mltpl))
    }
    if (method == "mad") {
      quo <- stats::median(mins, na.rm = TRUE)
      tester <- stats::mad(mins, na.rm = TRUE)
      out <- which(mins > (quo + tester * mltpl))
    }
    if (length(out) == 0) return(NA)
    if (raster_flag) {
      rownames(k)[pts$lyr.1 %in% gsub("X", "", names(out))]               # FIX (b)
    } else {
      rownames(k)[out]
    }
  })
  flags <- as.numeric(as.vector(unlist(flags)))
  flags <- flags[!is.na(flags)]
  out <- rep(TRUE, nrow(x))
  out[flags] <- FALSE
  switch(value, clean = return(x[out, ]), flagged = return(out),
         ids = return(init_ids[flags]))
}
