# SPDX-License-Identifier: GPL-3.0-or-later
#
# geodiv 1.1.1's surface metrics with their errors corrected: the reference
# for ls.surface where geodiv is wrong (sourced by tools/surface_reference.R;
# everything else is compared with geodiv itself). Each function takes a terra
# SpatRaster, as geodiv's, and keeps geodiv's arithmetic wherever it is right.
#
#   sq, ssk, sku  ISO 25178's Sq: the root mean square (over N, not N - 1),
#                 also inside ssk and sku (geodiv mixes sd() with moments over N)
#   sdq           divided by the number of pairs with both values (geodiv
#                 counts the pairs with a missing value too)
#   sdq6          the seven-point stencil +9 z[-2] (geodiv: -9 z[+2] twice) and
#                 the mean over the cells (geodiv divides by length() of a
#                 SpatRaster, 1)
#   findvalleys   the cell's own neighbours (geodiv fills its matrix by
#                 columns); s10z with them
#   ssc           ISO's mean summit curvature, -1/(2n) sum(d2z/dx2 + d2z/dy2)
#                 (geodiv: the opposite sign and a first-difference scale)
#   bearing area  heights from z itself (quantile(z, 1 - p), not
#                 1 - quantile(1 - z, p)); area_above = the area between the
#                 curve and its top (geodiv's depends on a shift and its
#                 Simpson's rule drops the last point); one-sided slopes at the
#                 ends of the curve divided by h (geodiv: 2h)
#   aacf / scl / std / srw
#                 the zero lag / frequency where fftshift puts it (row
#                 floor(N/2) + 1, column floor(M/2) + 1; geodiv's upper half
#                 leaves that row out and std's angles start one column off),
#                 distances in map units exactly (geodiv: GDAL's proximity in
#                 single precision with the cell width in both directions),
#                 std within the largest circle that fits in the half image,
#                 over orientations 0 to 179.5 degrees (without the zero
#                 frequency and, in its row, the negative side, which repeats
#                 the positive one at 180 degrees)
#   sfd           geodiv's C code with its typo ((b - e) (b - c) for
#                 (b - c)^2), extremes from the data (not 99999) and 0-based
#                 indices (geodiv reads past the matrix)
#   bestfitplane  the order with the least BIC (geodiv takes the least
#                 absolute mean residual, which is 0 for every least-squares
#                 fit: rounding decides)

fx <- new.env()

fx$vals <- function(r) {
  z <- values(r)[, 1]
  z[!is.na(z)]
}

# rows of a matrix into one vector (R's cell order of a SpatRaster)
fx$rowwise <- function(m) as.vector(t(m))

fx$zmat <- function(r) matrix(values(r)[, 1], nrow(r), ncol(r), byrow = TRUE)

# -- amplitude -------------------------------------------------------------------

fx$sq <- function(r) {
  z <- fx$vals(r)
  zbar <- mean(z)
  sqrt(sum((z - zbar)^2) / length(z))
}

fx$ssk <- function(r, adj = TRUE) {
  z <- fx$vals(r)
  N <- length(z)
  zbar <- mean(z)
  s <- fx$sq(r)
  g1 <- (sum((z - zbar)^3) / N) / (s^3)
  if (adj) (sqrt((N * (N - 1))) / (N - 2)) * g1 else g1
}

fx$sku <- function(r, excess = TRUE) {
  z <- fx$vals(r)
  N <- length(z)
  zbar <- mean(z)
  s <- fx$sq(r)
  k <- (sum((z - zbar)^4) / N) / (s^4)
  if (excess) k - 3 else k
}

# -- trend surfaces ----------------------------------------------------------------

fx$bestfitplane <- function(r) {
  z <- values(r)[, 1]
  N <- sum(!is.na(z))
  bic <- sapply(0:3, function(k) {
    fit <- as.vector(t(geodiv::fitplane(r, k)))
    rss <- sum((z - fit)^2, na.rm = TRUE)
    N * log(rss / N) + ((k + 1) * (k + 2) / 2) * log(N)
  })
  best <- which(bic == min(bic))[1] - 1
  list(order = best, fit = geodiv::fitplane(r, best))
}

# -- slopes ------------------------------------------------------------------------

fx$sdq <- function(r) {
  z <- fx$zmat(r)
  N <- nrow(z); M <- ncol(z)
  a <- z[1:(N - 1), 1:(M - 1), drop = FALSE]
  d <- fx$rowwise((((a - z[1:(N - 1), 2:M, drop = FALSE]) / 1)^2) + (((a - z[2:N, 1:(M - 1), drop = FALSE]) / 1)^2))
  sqrt(sum(d, na.rm = TRUE) / sum(!is.na(d)))
}

fx$sdq6 <- function(r) {
  z <- fx$zmat(r)
  N <- nrow(z); M <- ncol(z)
  if (N < 7 || M < 7) return(NaN)
  rows <- 4:(N - 3); cols <- 4:(M - 3)
  s <- function(dr, dc) z[rows + dr, cols + dc, drop = FALSE]
  pklx <- (1 / (60 * 1)) * (s(0, 3) - (9 * s(0, 2)) + (45 * s(0, 1)) - (45 * s(0, -1)) + (9 * s(0, -2)) - s(0, -3))
  pkly <- (1 / (60 * 1)) * (s(3, 0) - (9 * s(2, 0)) + (45 * s(1, 0)) - (45 * s(-1, 0)) + (9 * s(-2, 0)) - s(-3, 0))
  d <- fx$rowwise((pklx^2) + (pkly^2))
  sqrt(sum(d, na.rm = TRUE) / sum(!is.na(d)))
}

# -- local features ----------------------------------------------------------------

fx$findvalleys <- function(r) {
  z <- fx$zmat(r)
  N <- nrow(z); M <- ncol(z)
  out <- NULL
  if (N >= 3 && M >= 3) {
    for (i in 2:(N - 1)) for (j in 2:(M - 1)) {
      v <- z[i, j]
      nb <- z[(i - 1):(i + 1), (j - 1):(j + 1)][-5]
      if (!is.na(v) && !anyNA(nb) && v < 0 && all(v < nb)) out <- rbind(out, c(i, j, v))
    }
  }
  if (is.null(out)) return(data.frame(row = integer(0), col = integer(0), val = numeric(0)))
  data.frame(row = out[, 1], col = out[, 2], val = out[, 3])
}

fx$s10z <- function(r) {
  peaks <- geodiv::findpeaks(r)
  valleys <- fx$findvalleys(r)
  top_peaks <- peaks[order(peaks$val, decreasing = TRUE)[1:5], ]
  bottom_valleys <- valleys[order(valleys$val)[1:5], ]
  (sum(top_peaks$val, na.rm = TRUE) + sum(abs(bottom_valleys$val), na.rm = TRUE)) / 5
}

fx$ssc <- function(r) {
  z <- fx$zmat(r)
  zn <- (z - min(z, na.rm = TRUE)) / (max(z, na.rm = TRUE) - min(z, na.rm = TRUE))
  dx <- res(r)[1] / res(r)[1]
  dy <- res(r)[2] / res(r)[1]
  peaks <- geodiv::findpeaks(r)
  n <- nrow(peaks)
  if (n == 0) return(NA_real_)
  rc <- cbind(peaks$row, peaks$col)
  v <- zn[rc]
  xp <- zn[cbind(rc[, 1], rc[, 2] + 1)]; xm <- zn[cbind(rc[, 1], rc[, 2] - 1)]
  yp <- zn[cbind(rc[, 1] + 1, rc[, 2])]; ym <- zn[cbind(rc[, 1] - 1, rc[, 2])]
  -(1 / (2 * n)) * sum((((xp - (2 * v)) + xm) / (dx * dx)) + (((yp - (2 * v)) + ym) / (dy * dy)))
}

# -- bearing area curve ------------------------------------------------------------

fx$height <- function(z, p) unname(stats::quantile(z, probs = 1 - p, type = 7))

fx$curve <- function(r) {
  z <- fx$vals(r)
  v <- sort(unique(z), decreasing = TRUE)
  counts <- tabulate(match(z, v), length(v))
  list(bearing = cumsum(counts) / length(z), height = v)
}

fx$area_above <- function(z, a, b, n) {
  h <- (b - a) / n
  x <- seq(a, b, length.out = n + 1)
  y <- fx$height(z, x)
  s <- (h / 3) * (y[[1]] + sum(4 * y[seq(2, n, by = 2)]) + sum(2 * y[seq(3, n - 1, by = 2)]) + y[[n + 1]])
  (max(y) * (max(x) - min(x))) - s
}

fx$find_flat <- function(z) {
  cv <- list(bearing = NULL)
  v <- sort(unique(z), decreasing = TRUE)
  xval <- cumsum(tabulate(match(z, v), length(v))) / length(z)
  yval <- v
  even_x <- seq(0, 1, length.out = 10000)
  h <- 0.001
  x <- even_x
  space <- 1 / length(x)
  end_length <- h / space
  if (end_length < 1) end_length <- 1
  begin <- ceiling(1 + end_length)
  end <- floor(length(x) - end_length)
  n <- length(x)
  slope <- numeric(n)
  i1 <- 1:(begin - 1); i2 <- begin:end; i3 <- (end + 1):n
  slope[i1] <- (fx$height(z, x[i1] + h) - fx$height(z, x[i1])) / h
  slope[i2] <- (fx$height(z, x[i2] + h) - fx$height(z, x[i2] - h)) / (2 * h)
  slope[i3] <- (fx$height(z, x[i3]) - fx$height(z, x[i3] - h)) / h
  l <- 0.4
  len <- length(x) * l
  end_ind <- length(x) - len
  begin_ind <- 1 + len
  means <- sapply(seq(1, end_ind), function(i) mean(abs(slope[i:(i + len)])))
  w <- which(means == min(means))[1]
  xstart <- x[w]; xend <- x[w + len]
  keep <- xval >= xstart & xval <= xend
  if (!any(keep)) return(c(NA, NA, NA, NA))
  fit <- stats::lm(y ~ x, data = data.frame(x = xval[keep], y = yval[keep]))
  b <- stats::coef(fit)
  if (is.na(b[2])) { high <- b[[1]]; low <- b[[1]] } else {
    high <- (0 + b[[1]] * 1) + b[[2]] * 0
    low <- (0 + b[[1]] * 1) + b[[2]] * 1
  }
  c(high, low, sum(z >= high) / length(z), sum(z >= low) / length(z))
}

fx$bearing <- function(r) {
  z <- fx$vals(r)
  ff <- fx$find_flat(z)
  svi <- fx$area_above(z, 0.8, 1, 500)
  list(sbi = fx$sq(r) / fx$height(z, 0.05), svi = svi, sci = fx$area_above(z, 0.05, 1, 1000) - svi,
       sk = abs(ff[1] - ff[2]),
       spk = abs(fx$height(z, 0) - fx$height(z, ff[3])),
       svk = abs(fx$height(z, 1) - fx$height(z, ff[4])))
}

# -- spatial parameters ------------------------------------------------------------

# geodiv's crop to the largest square without NA around the centre
fx$na_crop <- function(x) {
  if (!anyNA(values(x))) return(x)
  N <- nrow(x)
  origin <- c(mean(crds(x, na.rm = FALSE)[, 1]), mean(crds(x, na.rm = FALSE)[, 2]))
  k <- seq(1, floor(N / 2))
  na <- sapply(k, function(i) {
    sum(is.na(terra::crop(x, ext(origin[1] - res(x)[1] * i, origin[1] + res(x)[1] * i,
                                 origin[2] - res(x)[2] * i, origin[2] + res(x)[2] * i))[]))
  })
  if (!any(na <= 0)) return(x)
  i <- max(which(na <= 0))
  terra::crop(x, ext(origin[1] - res(x)[1] * i, origin[1] + res(x)[1] * i,
                     origin[2] - res(x)[2] * i, origin[2] + res(x)[2] * i))
}

# geodiv's fftshift, also for one row or column (geodiv's indexes x[2:1, ])
fx$fftshift <- function(m) {
  N <- nrow(m); M <- ncol(m)
  m[((seq_len(N) - 1 + ceiling(N / 2)) %% N) + 1, ((seq_len(M) - 1 + ceiling(M / 2)) %% M) + 1, drop = FALSE]
}

# rows 1 .. floor(N/2) + 1 of an image (the upper half with the zero row)
fx$upper <- function(img) {
  r0 <- floor(nrow(img) / 2) + 1
  e <- ext(img)
  yr <- (e$ymax - e$ymin) / nrow(img)
  out <- rast(nrows = r0, ncols = ncol(img), xmin = e$xmin, xmax = e$xmax, ymin = e$ymax - r0 * yr,
              ymax = e$ymax, crs = crs(img))
  values(out) <- values(img)[1:(r0 * ncol(img)), 1]
  out
}

# distances of every cell to cell (last row, column floor(M/2) + 1)
fx$dist0 <- function(img) {
  r0 <- nrow(img); c0 <- floor(ncol(img) / 2) + 1
  if (is.lonlat(img)) {
    d <- img
    values(d) <- NA
    d[r0, c0] <- 1
    return(values(terra::distance(d))[, 1])
  }
  xr <- res(img)[1]; yr <- res(img)[2]
  rows <- rep(1:nrow(img), each = ncol(img)); cols <- rep(1:ncol(img), nrow(img))
  dx <- (cols - c0) * xr; dy <- (rows - r0) * yr
  sqrt((dx^2) + (dy^2))
}

fx$aacf <- function(r) {
  x <- fx$na_crop(r)
  M <- ncol(x); N <- nrow(x)
  zmat <- matrix(x[], ncol = M, nrow = N, byrow = TRUE)
  wc <- e1071::hanning.window(M)
  wr <- e1071::hanning.window(N)
  w <- pracma::meshgrid(wc, wr)
  w <- w[[1]] * w[[2]]
  ft <- stats::fft(zmat * w)
  ps <- (abs(ft)^2) / (M * N)
  af <- Re(stats::fft(ps, inverse = TRUE) / (M * N))
  af <- fx$fftshift(af)
  terra::setValues(x, af / max(as.numeric(af), na.rm = TRUE))
}

fx$scl <- function(r, threshold = c(0.20, 1 / exp(1))) {
  half <- fx$upper(fx$aacf(r))
  d <- fx$dist0(half)
  a <- values(half)[, 1]
  if (all(is.na(a))) return(rep(NA_real_, 2 * length(threshold)))  # no autocorrelation (NA)
  lo <- sapply(threshold, function(t) suppressWarnings(min(d[which(a <= t)], na.rm = TRUE)))
  hi <- sapply(threshold, function(t) suppressWarnings(max(d[which(a <= t)], na.rm = TRUE)))
  c(lo, hi)
}

fx$stxr <- function(r, threshold = c(0.20, 1 / exp(1))) {
  s <- fx$scl(r, threshold)
  k <- length(threshold)
  s[1:k] / s[(k + 1):(2 * k)]
}

fx$amplitude <- function(r) {
  x <- fx$na_crop(r)
  zmat <- matrix(x[], ncol = ncol(x), nrow = nrow(x), byrow = TRUE)
  ft <- fx$fftshift(stats::fft(zmat))
  fx$upper(terra::setValues(x, sqrt((Re(ft)^2) + (Im(ft)^2))))
}

fx$std <- function(r) {
  amp <- fx$amplitude(r)
  d <- fx$dist0(amp)
  nr <- nrow(amp); nc <- ncol(amp); c0 <- floor(nc / 2) + 1
  cell <- function(i, j) (i - 1) * nc + j
  min_dist <- min(d[cell(nr, 1)], d[cell(nr, nc)], d[cell(1, c0)])
  xy <- crds(amp, na.rm = FALSE)
  centre <- cell(nr, c0)
  angles <- atan2(xy[, 2] - xy[centre, 2], xy[, 1] - xy[centre, 1]) / (pi / 180)
  vals <- values(amp)[, 1]
  # orientations in [0, 180): in the zero row, only the positive side
  rows <- rep(1:nr, each = nc); cols <- rep(1:nc, nr)
  good <- which(d <= min_dist & !(rows == nr & cols <= c0))
  alpha <- seq(0, 179.5, 0.5)
  A <- sapply(alpha, function(a) sum(vals[good[abs(angles[good] - a) < 0.25]], na.rm = TRUE))
  c(min(alpha[which(A == max(A, na.rm = TRUE))], na.rm = TRUE), mean(A, na.rm = TRUE) / max(A, na.rm = TRUE))
}

fx$srw <- function(r) {
  amp <- fx$amplitude(r)
  d <- fx$dist0(amp)
  nr <- nrow(amp); nc <- ncol(amp)
  ncircles <- if ((0.5 * nc) <= 100) floor(0.5 * nc) else 100
  half_dist <- max(d[((nr - 1) * nc + 1):(nr * nc)])
  radius <- seq(0, half_dist, length.out = ncircles)
  if (length(radius) < 3) return(c(NA_real_, NA_real_, NA_real_))  # too few circles
  tolerance <- (radius[3] - radius[2]) / 2
  vals <- values(amp)[, 1]
  Br <- sapply(radius, function(i) sum(vals[which(abs(d - i) < tolerance)], na.rm = TRUE))
  Br[1] <- 0
  Srw <- radius[which(Br == max(Br, na.rm = TRUE))]
  Srwi <- mean(Br, na.rm = TRUE) / max(Br, na.rm = TRUE)
  half_val <- 0.5 * sum(Br, na.rm = TRUE)
  cum <- sapply(seq_along(Br), function(i) sum(Br[1:i], na.rm = TRUE))
  Shw <- radius[which(abs(cum - half_val) == min(abs(cum - half_val), na.rm = TRUE))]
  c(if (length(Srw) == 1) Srw else NA, Srwi, if (length(Shw) == 1) Shw else NA)
}

# -- fractal dimension -------------------------------------------------------------

fx$sfd <- function(r) {
  mat <- fx$zmat(r)
  if (anyNA(mat)) return(NaN)
  rowmax <- nrow(mat); colmax <- ncol(mat)
  nm <- (mat - min(mat)) / (max(mat) - min(mat))
  size <- min(rowmax, colmax)
  n <- 1; steps <- 1
  repeat { steps <- steps + 1; n <- n * 2; if (!(n < size)) break }
  n <- n / 2; steps <- steps - 1
  br <- floor((rowmax - n) / 2) + 1; bc <- floor((colmax - n) / 2) + 1   # 1-based first row / column
  area <- numeric(steps); resolution <- numeric(steps)
  step <- 1
  for (time in 1:steps) {
    side <- step
    diag <- step * sqrt(2) / 2
    rows <- seq(br, br + n - 1, by = step); cols <- seq(bc, bc + n - 1, by = step)
    rr <- rep(rows, each = length(cols)); cc <- rep(cols, length(rows))
    a <- nm[cbind(rr, cc)]; b <- nm[cbind(rr, cc + step)]
    c <- nm[cbind(rr + step, cc + step)]; d <- nm[cbind(rr + step, cc)]
    e <- 0.25 * (a + b + c + d)
    w <- sqrt((a - b) * (a - b) + side * side)
    x <- sqrt((b - c) * (b - c) + side * side)
    y <- sqrt((c - d) * (c - d) + side * side)
    z <- sqrt((a - d) * (a - d) + side * side)
    o <- sqrt((a - e) * (a - e) + diag * diag)
    p <- sqrt((b - e) * (b - e) + diag * diag)
    q <- sqrt((c - e) * (c - e) + diag * diag)
    s <- sqrt((d - e) * (d - e) + diag * diag)
    sa <- 0.5 * (w + p + o); sb <- 0.5 * (x + p + q); sc <- 0.5 * (y + q + s); sd <- 0.5 * (z + o + s)
    aa <- sqrt(abs(sa * (sa - w) * (sa - p) * (sa - o)))
    ab <- sqrt(abs(sb * (sb - x) * (sb - p) * (sb - q)))
    ac <- sqrt(abs(sc * (sc - y) * (sc - q) * (sc - s)))
    ad <- sqrt(abs(sd * (sd - z) * (sd - o) * (sd - s)))
    area[time] <- sum(aa + ab + ac + ad)
    resolution[time] <- step * step
    step <- step * 2
  }
  lr <- log(resolution); la <- log(area)
  mr <- sum(lr) / steps; ma <- sum(la) / steps
  cross <- sum((lr - mr) * (la - ma)); sumres <- sum((lr - mr) * (lr - mr)); sumarea <- sum((la - ma) * (la - ma))
  if (sumres == 0) sumres <- 1
  if (sumarea == 0) sumarea <- 1
  crossr <- cross / sqrt(sumres * sumarea)
  2 - crossr * sqrt(sumarea) / sqrt(sumres)
}

# -- moving windows ----------------------------------------------------------------

# f (a SpatRaster -> named numeric vector of k metrics) on the size x size
# window around every cell (row order): NA beyond the edges and, with circle,
# beyond size %/% 2 cells from the centre; each window a raster on its extent.
fx$window <- function(r, size, circle, f, names) {
  h <- size %/% 2; N <- nrow(r); M <- ncol(r)
  z <- fx$zmat(r)
  e <- ext(r)
  xr <- (e$xmax - e$xmin) / M; yr <- (e$ymax - e$ymin) / N
  k <- length(names)
  res <- matrix(NA_real_, N * M, k)
  cell <- 0
  for (i in 1:N) for (j in 1:M) {
    cell <- cell + 1
    w <- matrix(NA_real_, size, size)
    for (a in 1:size) for (b in 1:size) {
      ii <- i - h - 1 + a; jj <- j - h - 1 + b
      if (ii >= 1 && ii <= N && jj >= 1 && jj <= M && (!circle || ((a - h - 1)^2 + (b - h - 1)^2) <= h^2))
        w[a, b] <- z[ii, jj]
    }
    xmin <- e$xmin + ((j - 1) - h) * xr
    ymax <- e$ymax - ((i - 1) - h) * yr
    wr <- rast(nrows = size, ncols = size, xmin = xmin, xmax = xmin + size * xr, ymin = ymax - size * yr,
               ymax = ymax, crs = crs(r))
    values(wr) <- as.vector(t(w))
    v <- tryCatch(suppressWarnings(f(wr)), error = function(err) rep(NA_real_, k))
    res[cell, ] <- as.numeric(v)
  }
  out <- lapply(seq_len(k), function(m) res[, m])
  names(out) <- names
  out
}
