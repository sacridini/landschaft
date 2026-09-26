# SPDX-License-Identifier: GPL-3.0-or-later
#
# Reference values of Makurhini's MK_dPCIIC for landschaft's habitat graphs.
#
#   Rscript conn_makurhini_reference.R export <out_dir>
#       writes Makurhini's example `habitat_nodes` (centroid distances, areas
#       in km2) as habitat_nodes.csv / habitat_pairs.csv
#   Rscript conn_makurhini_reference.R run <jobs.csv> <out.csv>
#       jobs.csv: scene, nodes, matrix, metric, distance, probability, pij_min, la
#       (pij_min / la empty = NULL). Writes the public output of MK_dPCIIC
#       (rounded as Makurhini rounds it) and the unrounded values of its
#       internal steps (get_sdist, nodes_dPC / nodes_dIIC).
suppressPackageStartupMessages(library(Makurhini))

args <- commandArgs(trailingOnly = TRUE)
fmt <- function(x) sprintf("%.17g", x)

if (args[1] == "export") {
    data("habitat_nodes", package = "Makurhini")
    d <- distancefile(habitat_nodes, id = "Id", type = "centroid")
    nodes <- data.frame(id = habitat_nodes$Id, attribute = as.numeric(sf::st_area(habitat_nodes)) / 1e6)
    write.csv(data.frame(id = nodes$id, attribute = fmt(nodes$attribute)),
              file.path(args[2], "habitat_nodes.csv"), row.names = FALSE, quote = FALSE)
    write.csv(data.frame(from = d$From, to = d$To, distance = fmt(d$Distance)),
              file.path(args[2], "habitat_pairs.csv"), row.names = FALSE, quote = FALSE)
    quit(save = "no")
}

jobs <- read.csv(args[2], stringsAsFactors = FALSE, colClasses = "character")
rows <- list()
add <- function(job, level, id, name, value) {
    rows[[length(rows) + 1]] <<- data.frame(scene = job$scene, metric = job$metric, distance = job$distance,
                                            probability = job$probability, pij_min = job$pij_min, la = job$la,
                                            level = level, id = id, name = name, value = fmt(value))
}
num_or_null <- function(x) if (is.na(x) || x == "") NULL else as.numeric(x)

for (k in seq_len(nrow(jobs))) {
    job <- jobs[k, ]
    nodes <- read.csv(job$nodes)
    dist <- as.matrix(read.csv(job$matrix, header = FALSE))
    rownames(dist) <- colnames(dist) <- nodes$id
    thr <- as.numeric(job$distance)
    prob <- num_or_null(job$probability)
    pij_min <- num_or_null(job$pij_min)
    la <- num_or_null(job$la)

    res <- MK_dPCIIC(nodes = nodes, distance = dist, metric = job$metric, probability = prob, pij_min = pij_min,
                     distance_thresholds = thr, overall = TRUE, LA = la, intern = FALSE)
    imp <- res[[1]]
    ov <- res[[2]]
    for (col in names(imp)[-(1:2)]) add(job, "node", imp[[1]], col, imp[[col]])
    add(job, "overall", NA, ov$Index, ov$Value)

    # The unrounded steps of MK_dPCIIC (same calls, same arguments).
    attr <- nodes[[2]]
    d1 <- dist
    rownames(d1) <- colnames(d1) <- seq_len(nrow(d1))
    mat1 <- Makurhini:::get_sdist(dist_nodes = d1, attr_nodes = attr, metric = job$metric, probability = prob,
                                   distance_threshold = thr, igraph_Dijkstra = job$metric == "IIC", parallel = NULL,
                                   return_graph = TRUE, min_nodes = 0, loop = TRUE, G1 = 1000, pij_min = pij_min,
                                   return_pij = TRUE, intern = FALSE)
    num <- sum(mat1[[1]], na.rm = TRUE)
    if (job$metric == "PC") {
        delta <- Makurhini:::nodes_dPC(graph_n = mat1$graph, attrib_n = attr, num = num, G1 = 1000, min_nodes = 0,
                                       parallel = NULL, parallel_mode = 0, intern = FALSE)
    } else {
        delta <- Makurhini:::nodes_dIIC(graph_n = mat1$graph, attrib_n = attr, num = num, G1 = 1000, min_nodes = 0,
                                        parallel = NULL, parallel_mode = 0, intern = FALSE)
    }
    rs <- rowSums(mat1[[1]], na.rm = TRUE)
    dintra <- attr^2 / num * 100
    dflux <- 2 * (rs - attr^2) / num * 100
    m <- job$metric
    add(job, "node_raw", nodes$id, paste0("d", m), delta)
    add(job, "node_raw", nodes$id, paste0("d", m, "intra"), dintra)
    add(job, "node_raw", nodes$id, paste0("d", m, "flux"), dflux)
    add(job, "node_raw", nodes$id, paste0("d", m, "connector"), pmax(delta - dintra - dflux, 0))

    matr <- outer(attr, attr)
    intra <- sum(diag(matr))
    diag(matr) <- 0
    if (m == "PC") {
        direct <- sum(matr * mat1$pij, na.rm = TRUE)
        step <- sum(matr * (mat1$`pij*` - mat1$pij), na.rm = TRUE)
    } else {
        matr2 <- mat1$`pij*`
        matr2[matr2 == 16443701] <- 0
        direct <- sum(matr[which(matr2 == 1)] / 2)
        step <- sum(matr[which(matr2 >= 2)] / (1 + mat1$`pij*`[which(matr2 >= 2)]))
    }
    add(job, "overall_raw", NA, c(paste0(m, "num"), paste0(m, "intra(%)"), paste0(m, "direct(%)"),
                                  paste0(m, "step(%)")),
        c(num, intra / num * 100, direct / num * 100, step / num * 100))
}
write.csv(do.call(rbind, rows), args[3], row.names = FALSE)
