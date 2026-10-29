# SPDX-License-Identifier: GPL-3.0-or-later
# amt 0.3.1 with its errors corrected, the references ls.move is tested against,
# and the helpers of the amt cases (tests/move_cases.py).
#
# direction_abs(lonlat = TRUE): geosphere::bearing, clockwise from north, is
# taken as the anticlockwise angle from east; here it is converted, 90 - bearing.
# sinuosity: the turn angles, already in radians, are converted from degrees
# (cos(ta * pi / 180)); here they are taken as radians.
# fit_distr(x, "exp"): fitdistrplus' BFGS fails for rates below 1e-3 (its
# numerical gradient steps the rate below 0); here the maximum likelihood rate,
# 1 / mean(x), from which amt's BFGS starts.
# The rest is amt's code.

local({
  ns <- asNamespace("amt")

  direction_abs_fixed <- function(x, full_circle = FALSE, zero_dir = "E", clockwise = FALSE,
                                  append_last = TRUE, lonlat = FALSE, ...) {
    zero_dir <- toupper(zero_dir)
    if (!zero_dir %in% c("E", "N", "W", "S")) {
      stop("zero_dir should be one of 'E', 'N', 'W', or 'S'.")
    }
    if (zero_dir == "E") {
      zero_dir <- "East"
    }
    x$dx <- diff_x(x)
    x$dy <- diff_y(x)
    x$zero_step <- x$dx == 0 & x$dy == 0
    a <- if (!lonlat) {
      atan2(x$dy, x$dx)
    } else {
      xx <- sf::st_coordinates(as_sf(x))
      c(90 - geosphere::bearing(xx[-nrow(xx), ], xx[-1, ]), NA) * pi / 180
    }
    a[x$zero_step] <- NA
    a <- if (append_last) a else a[-length(a)]
    a <- ifelse(a < 0, 2 * pi + a, a)
    a <- switch(zero_dir,
      "East" = a,
      "S" = (2.5 * pi + a) %% (2 * pi),
      "W" = (3 * pi + a) %% (2 * pi),
      "N" = (3.5 * pi + a) %% (2 * pi)
    )
    a <- if (clockwise) (2 * pi - a) %% (2 * pi) else a
    a <- if (full_circle) a else ifelse(a > pi, (2 * pi - a) * -1, a)
    a
  }

  sinuosity_fixed <- function(x, ...) {
    x <- steps(x)
    p <- mean(x$sl_)
    cv <- sd(x$sl_) / p
    mc <- mean(cos(x$ta_), na.rm = TRUE)
    ms <- mean(sin(x$ta_), na.rm = TRUE)
    2 * (p * ((1 - mc^2 - ms^2) / ((1 - mc)^2 + ms^2) + cv^2))^(-0.5)
  }

  environment(direction_abs_fixed) <- ns
  environment(sinuosity_fixed) <- ns
  # UseMethod finds a method in the generic's namespace before the registered ones
  for (f in list(list("direction_abs.track_xy", direction_abs_fixed), list("sinuosity.track_xy", sinuosity_fixed))) {
    unlockBinding(f[[1]], ns)
    assign(f[[1]], f[[2]], envir = ns)
    lockBinding(f[[1]], ns)
  }
  registerS3method("direction_abs", "track_xy", direction_abs_fixed, envir = ns)
  registerS3method("sinuosity", "track_xy", sinuosity_fixed, envir = ns)
})

amt_fit_distr_fixed <- function(x, dist_name, na.rm = TRUE) {
  if (dist_name == "exp") {
    if (na.rm) x <- x[!is.na(x)]
    return(amt::make_exp_distr(rate = 1 / mean(x)))
  }
  amt::fit_distr(x, dist_name, na.rm = na.rm)
}

# One amt track_xyt per track of the move2 object x, in the order of x.
amt_tracks <- function(x) {
  xy <- sf::st_coordinates(x)
  id <- as.character(move2::mt_track_id(x))
  t <- move2::mt_time(x)
  crs <- sf::st_crs(x)
  lapply(split(seq_len(nrow(xy)), factor(id, levels = unique(id))), function(i) {
    amt::make_track(data.frame(x = xy[i, 1], y = xy[i, 2], t = t[i]), x, y, t,
                    crs = if (is.na(crs)) NA else crs)
  })
}

# The column `col` of each table of a list, one after the other.
amt_cat <- function(tables, col) unlist(lapply(tables, function(d) as.numeric(d[[col]])), use.names = FALSE)
