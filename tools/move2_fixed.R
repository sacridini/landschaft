# SPDX-License-Identifier: GPL-3.0-or-later
# move2 0.5.0 with its errors corrected, the references ls.move is tested against.
#
# mt_aeqd_crs: the centroid is transformed to EPSG:4236 (the Hu Tzu Shan 1950
# datum, a typo for 4326), which moves the centre of the projection by about
# a kilometre; here to EPSG:4326 as meant. The rest is move2's code.

mt_aeqd_crs_fixed <- function(x, center = c("centroid", "center"), units = c("m", "km")) {
  units <- rlang::arg_match(units)
  if (is.character(center)) {
    center <- rlang::arg_match(center)
    if (inherits(x, "sf")) x <- sf::st_geometry(x)
    center <- switch(center,
      centroid = sf::st_coordinates(sf::st_transform(
        sf::st_centroid(sf::st_cast(sf::st_geometry(x)[!sf::st_is_empty(x)], "MULTIPOINT", ids = 1L)),
        4326L
      ))[1L, ],
      center = apply(apply(sf::st_coordinates(sf::st_transform(x[!sf::st_is_empty(x)], 4326L)), 2L, range), 2L, mean)
    )
  }
  sf::st_crs(sprintf(
    "+proj=aeqd +lat_0=%.6f +lon_0=%.6f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=%s +no_defs",
    (center)[2L], (center)[1L], units
  ))
}
