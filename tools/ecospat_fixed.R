# SPDX-License-Identifier: GPL-3.0-or-later
#
# ecospat.boyce (ecospat 4.1.3) with its errors corrected, the reference of
# ls.sdm.boyce. Every change is marked "FIX":
#   - the "+1 trick" on the last moving window (and the -1 on its centre)
#     made the index depend on the scale of the predictions (the last window
#     vanished for predictions on 0-1, stayed for 0-1000): removed;
#   - rm.duplicate compared the last ratio with TRUE (1), dropping a final
#     ratio of exactly 1: the last ratio is always kept;
#   - user breaks (vector nclass) failed on `nclass == 0` with R >= 4.2 and
#     kept breaks below the minimum: fixed.

ecospat_boyce_fixed <- function(fit, obs, nclass = 0, window.w = "default", res = 100,
                                rm.duplicate = TRUE, method = "spearman") {
  boycei <- function(interval, obs, fit) {
    pi <- sum(as.numeric(obs >= interval[1] & obs <= interval[2])) / length(obs)
    ei <- sum(as.numeric(fit >= interval[1] & fit <= interval[2])) / length(fit)
    return(round(pi / ei, 10))
  }
  mini <- min(fit, obs)
  maxi <- max(fit, obs)
  if (length(nclass) == 1) {
    if (nclass == 0) {
      if (window.w == "default") window.w <- (max(fit) - min(fit)) / 10
      vec.mov <- seq(from = mini, to = maxi - window.w, by = (maxi - mini - window.w) / res)
      # FIX: no "+1 trick" on the last window
      interval <- cbind(vec.mov, vec.mov + window.w)
    } else {
      vec.mov <- seq(from = mini, to = maxi, by = (maxi - mini) / nclass)
      interval <- cbind(vec.mov, c(vec.mov[-1], maxi))
    }
  } else {
    vec.mov <- c(mini, sort(nclass[nclass >= mini & nclass <= maxi]))  # FIX: the breaks within the range
    interval <- cbind(vec.mov, c(vec.mov[-1], maxi))
  }
  f <- apply(interval, 1, boycei, obs, fit)
  to.keep <- which(f != "NaN")
  f <- f[to.keep]
  r <- seq_along(f)
  if (length(f) < 2) {
    b <- NA
  } else {
    if (rm.duplicate) r <- which(c(f[-1] != f[-length(f)], TRUE))  # FIX: the last ratio is always kept
    b <- cor(f[r], vec.mov[to.keep][r], method = method)
  }
  HS <- apply(interval, 1, sum) / 2  # FIX: no correction of the removed trick
  HS <- HS[to.keep]
  list(F.ratio = f, cor = round(b, 3), HS = HS, used = r - 1)
}
