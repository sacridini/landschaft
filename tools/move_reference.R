# SPDX-License-Identifier: GPL-3.0-or-later
# Golden outputs of R's move2 for ls.move (run by tools/make_reference_move.py).
#
# Usage: Rscript move_reference.R job.json outdir
# job.json: {"sets": {name: {"dir", "n", "crs", "track", "extra"}}, "cases": [{"name", "data", "code"}]}
# Each case's code runs with `x` (the move2 object of its track set) and returns a
# named list of numeric vectors, written as <case>__<key>.bin (little-endian f64)
# with outdir/manifest.json.

suppressPackageStartupMessages({
  library(jsonlite)
  library(sf)
  library(move2)
})

args <- commandArgs(trailingOnly = TRUE)
job <- fromJSON(args[1], simplifyVector = FALSE)
outdir <- args[2]
tools <- Sys.getenv("TOOLS")
source(file.path(tools, "move2_fixed.R"))

read_f64 <- function(path, n) readBin(path, "double", n = n, size = 8, endian = "little")

make_set <- function(name) {
  if (name == "fishers") return(mt_read(mt_example()))
  s <- job$sets[[name]]
  n <- s$n
  df <- data.frame(
    x = read_f64(file.path(s$dir, "x.bin"), n),
    y = read_f64(file.path(s$dir, "y.bin"), n),
    time = as.POSIXct(read_f64(file.path(s$dir, "t.bin"), n), origin = "1970-01-01", tz = "UTC"),
    track = unlist(s$track)
  )
  for (e in unlist(s$extra)) df[[e]] <- read_f64(file.path(s$dir, paste0(e, ".bin")), n)
  crs <- if (is.null(s$crs)) NA_crs_ else st_crs(s$crs)
  p <- st_as_sf(df, coords = c("x", "y"), crs = crs, na.fail = FALSE)
  mt_as_move2(p, time_column = "time", track_id_column = "track")
}

sets <- list()
manifest <- list()
for (case in job$cases) {
  if (is.null(sets[[case$data]])) sets[[case$data]] <- make_set(case$data)
  x <- sets[[case$data]]
  res <- eval(parse(text = case$code))
  for (key in names(res)) {
    v <- res[[key]]
    if (is.null(v)) next
    v <- as.numeric(v)
    file <- paste0(case$name, "__", key, ".bin")
    writeBin(v, file.path(outdir, file), size = 8, endian = "little")
    manifest[[length(manifest) + 1]] <- list(case = case$name, key = key, file = file, n = length(v))
  }
}
writeLines(toJSON(manifest, auto_unbox = TRUE), file.path(outdir, "manifest.json"))
