# SPDX-License-Identifier: GPL-3.0-or-later
#
# Golden outputs of R's maxnet (0.1.4, glmnet 4.1-10) for ls.sdm.MaxNet,
# driven by tools/make_reference_maxnet.py. Each case builds its data here
# (maxnet's Bradypus data, synthetic data from R's generator), fits
# maxnet() and writes the inputs and every result as raw little-endian
# files plus a manifest:
#
#   Rscript tools/maxnet_reference.R out_dir

suppressMessages({
  library(jsonlite)
  library(maxnet)
  library(glmnet)
})

out <- commandArgs(TRUE)[1]
manifest <- list()
counter <- 0L

put <- function(case, key, values, kind = "f8", dim = NULL) {
  counter <<- counter + 1L
  file <- file.path(out, sprintf("a%06d.bin", counter))
  if (kind == "f8") writeBin(as.double(values), file, endian = "little")
  else if (kind == "i4") writeBin(as.integer(values), file, size = 4, endian = "little")
  else if (kind == "i2") writeBin(as.integer(values), file, size = 2, endian = "little")
  item <- list(case = case, key = key, kind = kind, file = basename(file))
  if (!is.null(dim)) item$dim <- as.list(dim)
  manifest[[length(manifest) + 1L]] <<- item
}

put_text <- function(case, key, values) {
  manifest[[length(manifest) + 1L]] <<- list(case = case, key = key, kind = "text",
                                             values = as.list(as.character(values)))
}

# 60 rows beyond the data: the factor values of random rows, continuous
# values drawn up to half the range outside it
extrapolated <- function(data, seed = 7) {
  set.seed(seed)
  cont <- sapply(data, class) != "factor"
  ex <- data[sample(nrow(data), 60), , drop = FALSE]
  for (v in names(data)[cont]) {
    r <- range(data[[v]]); w <- diff(r)
    ex[[v]] <- runif(60, r[1] - 0.5 * w, r[2] + 0.5 * w)
  }
  rownames(ex) <- NULL
  ex
}

# the level codes (0-based) of the factors, one factor after the other
codes <- function(df) as.integer(unlist(lapply(df, function(v) as.integer(v) - 1L)))

# centi: the continuous data are K / 100 with integer K (not stored)
run_case <- function(case, p, data, classes = "default", regmult = 1, add = TRUE, centi = NULL,
                     all_names = FALSE, n_pred = 100) {
  f <- maxnet.formula(p, data, classes = classes)
  t0 <- proc.time()[["elapsed"]]
  m <- maxnet(p, data, f = f, regmult = regmult, addsamplestobackground = add)
  secs <- proc.time()[["elapsed"]] - t0
  cont <- sapply(data, class) != "factor"
  n <- nrow(data)
  # inputs
  if (is.null(centi)) {
    put(case, "cont", as.matrix(data[, cont, drop = FALSE]), "f8", c(sum(cont), n))
  } else {
    # the test makes these data again with R's generator (landschaft's RRandom): only a check here
    put(case, "centi_colsums", colSums(centi))
  }
  put(case, "cat", codes(data[, !cont, drop = FALSE]), "i4", c(sum(!cont), n))
  put_text(case, "cont_names", names(data)[cont])
  put_text(case, "cat_names", names(data)[!cont])
  for (v in names(data)[!cont]) put_text(case, paste0("levels/", v), levels(data[[v]]))
  put(case, "p", p)
  put_text(case, "classes", classes)
  put(case, "regmult", regmult)
  put(case, "add", as.numeric(add))
  put(case, "r_seconds", secs)
  # results
  put(case, "penalty", m$penalty.factor)
  put(case, "lambda", m$lambda)
  put(case, "beta", m$beta[, 200])
  put(case, "betas", m$betas)
  put_text(case, "betas_names", names(m$betas))
  if (all_names) put_text(case, "feature_names", rownames(m$beta))
  put(case, "alpha", m$alpha)
  put(case, "entropy", m$entropy)
  put(case, "npasses", m$npasses)
  put(case, "featuremins", m$featuremins)
  put(case, "featuremaxs", m$featuremaxs)
  put(case, "varmin", m$varmin)
  put(case, "varmax", m$varmax)
  put(case, "samplemeans", m$samplemeans)
  # predictions: some training rows, then the extrapolated ones
  rows <- unique(round(seq(1, n, length.out = min(n_pred, n))))
  ex <- extrapolated(data)
  nd <- rbind(data[rows, , drop = FALSE], ex)
  put(case, "pred_rows", rows - 1L)
  put(case, "new_cont", as.matrix(ex[, cont, drop = FALSE]), "f8", c(sum(cont), 60))
  put(case, "new_cat", codes(ex[, !cont, drop = FALSE]), "i4", c(sum(!cont), 60))
  for (clamp in c(TRUE, FALSE)) for (type in c("link", "exponential", "cloglog", "logistic")) {
    pr <- predict(m, nd, clamp = clamp, type = type)
    stopifnot(length(pr) == nrow(nd))
    put(case, sprintf("pred/%s/%s", type, if (clamp) "clamp" else "noclamp"), pr)
  }
  cat(sprintf("%-16s n=%d np=%d features=%d betas=%d npasses=%d %.2fs\n", case, n, sum(p), nrow(m$beta),
              length(m$betas), m$npasses, secs))
}

# maxnet's Bradypus data: 116 presences, 1000 background points, 13 integer
# variables and the ecoregion (factor, 14 levels)
data(bradypus)
p <- bradypus$presence
data <- bradypus[, -1]
run_case("brad_default", p, data)
run_case("brad_lq", p, data, "lq")
run_case("brad_h_rm2", p, data, "h", 2)
run_case("brad_t", p, data, "t")
run_case("brad_lqpht_rm05", p, data, "lqpht", 0.5)
num <- data[, names(data) != "ecoreg"]
run_case("brad_num_lqp15", p, num, "lqp", 1.5)
# presences that are also background rows: maxnet does not add them again
pres <- which(p == 1)[1:5]
run_case("brad_num_dup", c(p, rep(0, 5)), rbind(num, num[pres, ]), "lq")

# synthetic: three double variables and a factor whose last level never occurs
set.seed(42)
mk <- function(n, np, nlev = 4) {
  d <- data.frame(a = rnorm(n, 10, 3), b = runif(n, -1, 1) * exp(rnorm(n)), c = rgamma(n, 2, 0.01))
  lv <- c("x", "y", "z", "w", "v")[1:nlev]
  d$k <- factor(sample(lv[1:(nlev - 1)], n, TRUE), levels = lv)
  pp <- rep(0, n)
  score <- d$a / 3 - abs(d$b) + (d$k == "y")
  pp[order(-score + rnorm(n) * 0.7)[1:np]] <- 1
  list(p = pp, d = d)
}
s1 <- mk(2040, 40); run_case("syn_np40", s1$p, s1$d)
s2 <- mk(512, 12); run_case("syn_np12", s2$p, s2$d)
s3 <- mk(300, 8, 3); run_case("syn_np8", s3$p, s3$d)
s4 <- mk(1500, 90); run_case("syn_np90_noadd", s4$p, s4$d, "lqpht", 1, add = FALSE, all_names = TRUE)
# products without linear terms (the names of the product columns follow the formula's variables)
run_case("syn_qp", s2$p, s2$d, "qp", all_names = TRUE)

# larger: 20 correlated variables with two decimals, 200 presences, 10000 background points
set.seed(11)
nb <- 10000; np <- 200; nv <- 20
Z <- matrix(rnorm((nb + np) * 5), ncol = 5)
A <- matrix(rnorm(5 * nv), 5)
X <- Z %*% A + matrix(rnorm((nb + np) * nv, sd = 0.5), ncol = nv)
K <- round(X * 100)
storage.mode(K) <- "integer"
X <- K / 100 + 0
d <- as.data.frame(X)
names(d) <- paste0("bio", 1:nv)
score <- X[, 1] - 0.5 * X[, 2]^2 / 5 + X[, 3] * 0.3
pb <- c(rep(1, np), rep(0, nb))
idx <- order(-(score + rnorm(nb + np)))[1:np]
sel <- c(idx, sample(nb + np, nb))
d <- d[sel, ]
rownames(d) <- NULL
run_case("big", pb, d, centi = K[sel, ])

versions <- lapply(c("maxnet", "glmnet", "Matrix"), function(p) as.character(packageVersion(p)))
names(versions) <- c("maxnet", "glmnet", "Matrix")
writeLines(toJSON(list(manifest = manifest, versions = versions, r = R.version.string),
                  auto_unbox = TRUE, digits = NA, null = "null"),
           file.path(out, "manifest.json"))
