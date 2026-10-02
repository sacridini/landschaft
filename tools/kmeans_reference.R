# SPDX-License-Identifier: GPL-3.0-or-later
#
# R 4.4.2 references of stats::kmeans() for tests/test_sdm_kmeans.py: prints
# the CASES literal pasted there. Run with
#   Rscript tools/kmeans_reference.R > kmeans_cases.txt
# The data sets are rebuilt in the test with landschaft._core.RRandom (R's
# generator), so only the results are stored; doubles as C99 hex ("%a").

Sys.setenv(LANGUAGE = "en")  # R's messages untranslated

mk <- function(name) switch(name,
  coords = { set.seed(11); cbind(runif(60, -60, -40), runif(60, -20, 0)) },
  coords_big = { set.seed(21); cbind(runif(400, -75, -35), runif(400, -33, 5)) },
  env3 = { set.seed(12); matrix(rnorm(150), 50, 3) + rep(c(0, 3, 6, 0, 3), each = 10) },
  dups = { set.seed(13); b <- cbind(runif(25, 0, 10), runif(25, 0, 10)); b[c(1:25, 1:10, 5:15, 3, 3, 3), ] },
  near = { set.seed(14); b <- cbind(runif(30, -50, 50), runif(30, -50, 50))
           rbind(b, b[1:10, ] * (1 + 2^-52), b[11:20, ] + 1e-13) },
  ties = { set.seed(16); d <- floor(runif(20, 1e14, 1e15)); t <- (d + 0.5) / 1e13
           cbind(c(t, t * (1 + 2^-52), t * (1 - 2^-52)), rep(c(1, 2), 30)) },
  oned = { set.seed(15); u <- runif(40); matrix(c(u, u[1:10], u[11:15] * (1 + 2^-52)), ncol = 1) },
  qtran = { v <- "1200100001202112122002221020221120222201022100012210221212220021201221112221201121222221212012222111"
            matrix(1e12 + as.integer(strsplit(v, "")[[1]]) * 0.001, ncol = 1) },
  tiny = rbind(c(0, 0), c(1e-170, 0), c(5, 5), c(6, 6))
)

hex <- function(v) {
  s <- sprintf("%a", v)
  stopifnot(identical(as.numeric(s), as.numeric(v)) || all(is.nan(v)))
  paste0("[", paste0('"', s, '"', collapse = ", "), "]")
}

cases <- list(
  # blockCV::cv_cluster: kmeans(xy, k, iter.max = 500, nstart = 25)
  list("coords", 2, 25, 500, "Hartigan-Wong", 1), list("coords", 3, 25, 500, "Hartigan-Wong", 2),
  list("coords", 4, 25, 500, "Hartigan-Wong", 3), list("coords", 5, 25, 500, "Hartigan-Wong", 4),
  list("coords", 6, 25, 500, "Hartigan-Wong", 5), list("coords", 7, 25, 500, "Hartigan-Wong", 6),
  list("coords", 8, 25, 500, "Hartigan-Wong", 7), list("coords", 5, 1, 500, "Hartigan-Wong", 8),
  list("coords", 3, 1, 10, "Hartigan-Wong", 9), list("coords", 1, 1, 10, "Hartigan-Wong", 10),
  list("coords_big", 5, 25, 500, "Hartigan-Wong", 2), list("coords_big", 8, 25, 500, "Hartigan-Wong", 42),
  list("coords_big", 6, 25, 500, "Lloyd", 3), list("coords_big", 6, 25, 500, "MacQueen", 3),
  list("coords_big", 4, 25, 1, "Hartigan-Wong", 5), list("coords_big", 4, 25, 2, "Lloyd", 5),
  list("coords_big", 4, 3, 1, "MacQueen", 5),
  list("env3", 3, 25, 500, "Hartigan-Wong", 1), list("env3", 6, 25, 500, "Hartigan-Wong", 2),
  list("env3", 4, 1, 500, "Hartigan-Wong", 3), list("env3", 4, 25, 500, "Forgy", 4),
  list("env3", 5, 25, 500, "MacQueen", 5), list("env3", 2, 25, 500, "Hartigan-Wong", 6),
  list("dups", 4, 25, 500, "Hartigan-Wong", 1), list("dups", 2, 25, 500, "Hartigan-Wong", 2),
  list("dups", 6, 1, 500, "Hartigan-Wong", 3), list("dups", 6, 1, 500, "Hartigan-Wong", 4),
  list("dups", 6, 1, 500, "Hartigan-Wong", 5), list("dups", 6, 1, 500, "Lloyd", 6),
  list("near", 3, 25, 500, "Hartigan-Wong", 1), list("near", 7, 25, 500, "Hartigan-Wong", 2),
  list("near", 5, 1, 500, "Hartigan-Wong", 3),
  list("ties", 4, 25, 500, "Hartigan-Wong", 1), list("ties", 8, 25, 500, "Hartigan-Wong", 2),
  list("oned", 3, 25, 500, "Hartigan-Wong", 1), list("oned", 5, 1, 500, "Hartigan-Wong", 2),
  list("oned", 4, 25, 500, "Lloyd", 3),
  list("qtran", 3, 1, 50, "Hartigan-Wong", 1), list("qtran", 3, 4, 50, "Hartigan-Wong", 2),
  list("tiny", 3, 1, 10, "Hartigan-Wong", 1), list("tiny", 3, 1, 10, "Hartigan-Wong", 2),
  list("tiny", 3, 1, 10, "Hartigan-Wong", 3), list("tiny", 3, 1, 10, "Hartigan-Wong", 4),
  list("dups", 30, 25, 10, "Hartigan-Wong", 1),
  # initial centres given as a matrix (seed only for the draw after)
  list("tiny", 3, 1, 10, "Hartigan-Wong", 1, "first3"), list("coords", 3, 1, 10, "Lloyd", 1, "far"),
  list("coords", 3, 1, 10, "MacQueen", 1, "far"), list("coords", 3, 1, 500, "Hartigan-Wong", 1, "first3")
)
centres <- function(x, spec) switch(spec, first3 = x[1:3, , drop = FALSE], far = rbind(x[1, ], x[2, ], c(1000, 1000)))

cat("CASES = [\n")
for (cs in cases) {
  x <- mk(cs[[1]])
  set.seed(cs[[6]])
  w <- character(0)
  res <- tryCatch(withCallingHandlers(
    kmeans(x, if (length(cs) > 6) centres(x, cs[[7]]) else cs[[2]], iter.max = cs[[4]], nstart = cs[[3]], algorithm = cs[[5]]),
    warning = function(e) { w <<- c(w, conditionMessage(e)); invokeRestart("muffleWarning") }),
    error = function(e) conditionMessage(e))
  after <- runif(1)
  head <- sprintf('    dict(data="%s", k=%d, nstart=%d, iter_max=%d, algorithm="%s", seed=%d, %s',
                  cs[[1]], cs[[2]], cs[[3]], cs[[4]], cs[[5]], cs[[6]],
                  if (length(cs) > 6) sprintf('init="%s", ', cs[[7]]) else "")
  if (is.character(res)) {
    cat(head, sprintf('error="%s"),\n', res), sep = "")
    next
  }
  cat(head, "\n", sep = "")
  cat(sprintf('         cluster="%s",\n', paste(res$cluster, collapse = "")))
  cat(sprintf('         centers=%s,\n', hex(as.vector(t(res$centers)))))
  cat(sprintf('         withinss=%s, size=[%s], iter=%d, ifault=%s,\n', hex(res$withinss),
              paste(res$size, collapse = ", "), res$iter,
              if (is.null(res$ifault)) "None" else as.character(res$ifault)))
  cat(sprintf('         totss=%s, tot_withinss=%s, betweenss=%s,\n', hex(res$totss), hex(res$tot.withinss),
              hex(res$betweenss)))
  cat(sprintf('         warnings=[%s], after=%s),\n', if (length(w)) paste0('"', w, '"', collapse = ", ") else "", hex(after)))
}
cat("]\n")
