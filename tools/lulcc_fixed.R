# SPDX-License-Identifier: GPL-3.0-or-later
#
# lulcc 1.0.4 with its defects corrected, the reference for the corrected
# forms in ls.simulate (source(file.path(TOOLS, "lulcc_fixed.R")) after
# library(lulcc)).
#
# - The CLUE-S allocation (src/allocateclues.c) adds the signed differences
#   between demand and allocation for its mean criterion: they sum to zero,
#   so the criterion never acts. Here it is the mean of the absolute
#   differences, as CLUE-S defines it; everything else (the jitter drawn per
#   non-NA suitability in column order, the first class of highest
#   suitability, the passes and the iteration variables) is the C code step
#   for step.
# - Updating a NeighbRasterStack keeps only the last class's map
#   (`maps <- eval(cl)` for `maps[[i]] <- eval(cl)`), so neighbourhood rules
#   on more than one class stop with an error.
# - CluesModel and OrderedModel compare the number of neighbourhood rules with
#   length(neighb), which for a RasterStack is cells x layers: neighbourhood
#   rules always stop with an error. Here the length of a NeighbRasterStack is
#   its number of maps.
# - The ordered allocation leaves the cells a shrinking class releases and no
#   class served after it takes as -1 (an invalid class in the output map,
#   and in the next step's input). Here they keep their class.

clues_fixed <- function(tprob, map0.vals, demand, categories, jitter.f, scale.f, max.iter, max.diff, ave.diff) {
  ncell <- nrow(tprob)
  ncode <- length(categories)
  lu.ix <- match(map0.vals, categories)
  if (anyNA(lu.ix)) stop("land use map contains invalid codes")
  ok <- !is.na(tprob)
  nok <- sum(ok)
  iter <- rep(0, ncode)
  count <- 0
  repeat {
    pprob <- tprob
    u <- runif(nok)
    pprob[ok] <- tprob[ok] + (-jitter.f + (jitter.f - (-jitter.f)) * u)
    best <- rep(-999, ncell)
    ix <- lu.ix
    for (j in seq_len(ncode)) {
      v <- pprob[, j] + iter[j]
      upd <- !is.na(v) & best < v
      best[upd] <- v[upd]
      ix[upd] <- j
    }
    total <- tabulate(ix, ncode)
    diff <- demand - total
    maxd <- 0
    aved <- 0
    for (i in seq_len(ncode)) {
      a <- if (diff[i] < 0) diff[i] * -1 else diff[i]
      if (maxd < a) maxd <- a
      aved <- aved + a / ncode
    }
    if (count >= max.iter) break
    iter <- iter + diff * scale.f
    count <- count + 1
    if (!(maxd > max.diff || aved > ave.diff)) break
  }
  as.integer(categories[ix])
}
utils::assignInNamespace(".clues", clues_fixed, "lulcc")

setMethod("NeighbRasterStack", signature(x = "RasterLayer", weights = "ANY", neighb = "NeighbRasterStack"),
          function(x, weights, neighb) {
            categories <- neighb@categories
            calls <- list()
            maps <- list()
            for (i in seq_along(categories)) {
              r <- (x == categories[i])
              cl <- neighb@calls[[i]]
              cl$x <- r
              maps[[i]] <- eval(cl)
              cl$x <- NA
              calls[[i]] <- cl
            }
            maps <- stack(maps)
            new("NeighbRasterStack", maps, calls = calls, categories = categories)
          }, where = globalenv())

setMethod("length", "NeighbRasterStack", function(x) nlayers(x), where = globalenv())

ordered_orig <- utils::getFromNamespace(".ordered", "lulcc")
if (!is.null(attr(ordered_orig, "fixed"))) ordered_orig <- attr(ordered_orig, "fixed")
ordered_fixed <- function(tprob, map0.vals, demand, categories, order, stochastic) {
  out <- ordered_orig(tprob, map0.vals, demand, categories, order, stochastic)
  left <- out == -1
  out[left] <- map0.vals[left]
  out
}
attr(ordered_fixed, "fixed") <- ordered_orig
utils::assignInNamespace(".ordered", ordered_fixed, "lulcc")
