# SPDX-License-Identifier: GPL-3.0-or-later
#
# diffeR 0.0-8 functions with their defects corrected, the reference for the
# corrected forms in ls.simulate (source(file.path(TOOLS, "differ_fixed.R"))).

# sample2pop: the population of the row's category (population[i, 2]), not of
# the row's position (diffeR uses population[j, 2]).
sample2pop_fixed <- function(ctmatrix, population) {
  for (i in 1:dim(population)[1]) {
    j <- which(colnames(ctmatrix) == population[i, 1])
    ctmatrix[j, ] <- population[i, 2] * ctmatrix[j, ] / sum(ctmatrix[j, ])
  }
  ctmatrix
}

# composite: the categories are the classes of the two maps (diffeR counts NA as one, which leaves a row
# and a column of NA, and names the rows in the order the values first appear, not in the layers' order);
# the shares of each class are matched by class.
composite_fixed <- function(comp, ref, factor) {
  cats <- sort(unique(c(terra::values(comp), terra::values(ref))))
  compAg <- terra::aggregate(terra::segregate(comp, classes = cats), fact = factor, fun = mean)
  refAg <- terra::aggregate(terra::segregate(ref, classes = cats), fact = factor, fun = mean)
  n <- length(cats)
  conf.matrix <- matrix(nrow = n, ncol = n)
  weights <- comp - comp + 1
  wghtAg <- terra::aggregate(weights, fact = factor, fun = sum)
  sumDiagPix <- 0
  for (i in 1:n) sumDiagPix <- sumDiagPix + min(compAg[[i]], refAg[[i]])
  for (i in 1:n) {
    minsi <- min(compAg[[i]], refAg[[i]])
    for (j in 1:n) {
      minsj <- min(compAg[[j]], refAg[[j]])
      Aij <- wghtAg * ((compAg[[i]] - minsi) * (refAg[[j]] - minsj)) / (1 - sumDiagPix)
      conf.matrix[i, j] <- terra::global(Aij, fun = sum, na.rm = TRUE)[1, 1] /
        terra::global(wghtAg, fun = sum, na.rm = TRUE)[1, 1]
    }
  }
  for (i in 1:n) {
    min.ii <- min(compAg[[i]], refAg[[i]]) * wghtAg
    conf.matrix[i, i] <- terra::global(min.ii, fun = sum, na.rm = TRUE)[1, 1] /
      terra::global(wghtAg, fun = sum, na.rm = TRUE)[1, 1]
  }
  colnames(conf.matrix) <- paste0("ref.", cats)
  rownames(conf.matrix) <- paste0("comp.", cats)
  conf.matrix
}

# differenceMR with composite_fixed (eval = "multiple").
differenceMR_fixed <- function(comp, ref, percent = TRUE, fact = 2) {
  ctmatrix <- crosstabm(comp, ref, percent = TRUE)
  res1 <- cbind(overallQtyD(ctmatrix), overallExchangeD(ctmatrix), overallShiftD(ctmatrix), overallDiff(ctmatrix))
  mdim <- max(ncol(comp), nrow(comp))
  maxp <- floor(log(mdim, fact))
  if ((log(mdim, fact) - round(log(mdim, fact), 0)) != 0) maxp <- maxp + 1
  factvect <- c(1, fact^(1:maxp))
  if (factvect[length(factvect)] > mdim) factvect[length(factvect)] <- mdim
  resa <- data.frame(matrix(nrow = length(factvect), ncol = 5))
  colnames(resa) <- c("Resolution", "Quantity", "Exchange", "Shift", "Overall")
  resa[, 1] <- factvect * res(comp)[1]
  resa[1, 2:5] <- res1
  for (i in 2:length(factvect)) {
    ctmatrix <- composite_fixed(comp, ref, factvect[i]) * 100
    resa[i, 2:5] <- cbind(overallQtyD(ctmatrix), overallExchangeD(ctmatrix), overallShiftD(ctmatrix),
                          overallDiff(ctmatrix))
  }
  if (percent == FALSE) resa[, 2:5] <- resa[, 2:5] / 100
  cbind(Multiples = factvect, resa)
}
