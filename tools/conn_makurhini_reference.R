# SPDX-License-Identifier: GPL-3.0-or-later
#
# Reference values of Makurhini's MK_dPCIIC for landschaft's habitat graphs.
#
#   Rscript conn_makurhini_reference.R spatial <out_dir> <polygons.gpkg> <resistance.tif>
#       nodesfile / distancefile (centroid, edge) on Makurhini's example raster
#       `habitat_nodes_raster` (also written to out_dir) and on the polygons,
#       least-cost distances over `resistance_matrix` (written too) and over
#       the given resistance for the polygons, and MK_dPCIIC run end to end
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

if (args[1] == "spatial") {
    # Nodes and distances from Makurhini's example raster and polygons
    # (nodesfile, distancefile) and MK_dPCIIC run on them end to end.
    out <- args[2]
    polygons <- sf::st_read(args[3], quiet = TRUE)
    data("habitat_nodes_raster", package = "Makurhini")
    r <- terra::rast(habitat_nodes_raster)
    terra::writeRaster(r, file.path(out, "habitat_nodes_raster.tif"), overwrite = TRUE, datatype = "INT4S",
                       gdal = c("COMPRESS=DEFLATE"))
    rows <- list()
    put <- function(what, table) {
        table <- as.data.frame(table)
        for (col in names(table)) if (is.numeric(table[[col]])) table[[col]] <- fmt(table[[col]])
        write.csv(table, file.path(out, paste0(what, ".csv")), row.names = FALSE, quote = FALSE)
    }
    put("raster_nodes_m2", nodesfile(r))
    put("raster_nodes_ha", nodesfile(r, area_unit = "ha"))
    put("raster_nodes_weighted", nodesfile(r, attribute = seq_len(398) / 398, weighted = TRUE, area_unit = "km2"))
    put("vector_nodes_km2", nodesfile(polygons, id = "id", area_unit = "km2"))
    put("raster_centroid", distancefile(r, type = "centroid"))
    put("raster_edge", distancefile(r, type = "edge"))
    put("vector_centroid", distancefile(polygons, id = "id", type = "centroid"))
    put("vector_edge", distancefile(polygons, id = "id", type = "edge"))
    put("vector_edge_2km", distancefile(polygons, id = "id", type = "edge", threshold = 2, distance_unit = "km"))
    # Least-cost distances (gdistance), on the rasters as written to disk.
    data("resistance_matrix", package = "Makurhini")
    terra::writeRaster(terra::rast(resistance_matrix), file.path(out, "resistance_matrix.tif"), overwrite = TRUE,
                       datatype = "FLT8S", gdal = c("COMPRESS=DEFLATE"))
    res_r <- raster::raster(terra::rast(file.path(out, "resistance_matrix.tif")))
    r_disk <- terra::rast(file.path(out, "habitat_nodes_raster.tif"))
    put("raster_lcd", distancefile(r_disk, type = "least-cost", resistance = res_r))
    put("raster_lcd_4", distancefile(r_disk, type = "least-cost", resistance = res_r, ngh = 4))
    put("raster_lcd_mean", distancefile(r_disk, type = "least-cost", resistance = res_r,
                                        CostFun = function(x) 1 / mean(x)))
    res_v <- raster::raster(terra::rast(args[4]))
    put("vector_lcd", distancefile(polygons, id = "id", type = "least-cost", resistance = res_v))
    res <- MK_dPCIIC(nodes = r_disk, area_unit = "ha", distance = list(type = "least-cost", resistance = res_r),
                     metric = "PC", probability = 0.5, distance_thresholds = 250000, overall = TRUE, intern = FALSE)
    z <- terra::zonal(res[[1]][[-1]], r_disk, fun = "min")
    names(z)[1] <- "id"
    put("raster_lcd_pc_nodes", z)
    put("raster_lcd_pc_overall", res[[2]])
    # End to end: importance rasters read back per node id.
    res <- MK_dPCIIC(nodes = r, area_unit = "ha", distance = list(type = "edge"), metric = "PC",
                     probability = 0.5, distance_thresholds = 5000, overall = TRUE, intern = FALSE)
    imp <- res[[1]]
    z <- terra::zonal(imp[[-1]], r, fun = "min")
    names(z)[1] <- "id"
    put("raster_edge_pc_nodes", z)
    put("raster_edge_pc_overall", res[[2]])
    res <- MK_dPCIIC(nodes = polygons, area_unit = "km2", distance = list(type = "centroid"),
                     metric = "IIC", distance_thresholds = 3000, overall = TRUE, intern = FALSE)
    put("vector_centroid_iic_nodes", sf::st_drop_geometry(res[[1]]))
    put("vector_centroid_iic_overall", res[[2]])
    quit(save = "no")
}

if (args[1] == "links") {
    # MK_dPCIIC_links (removal and change) and MK_dPCIIC with restoration.
    # jobs.csv: scene, nodes, matrix, change, metric, distance, probability, pij_min, restore
    jobs <- read.csv(args[2], stringsAsFactors = FALSE, colClasses = "character")
    rows <- list()
    for (k in seq_len(nrow(jobs))) {
        job <- jobs[k, ]
        nodes <- read.csv(job$nodes)
        dist <- as.matrix(read.csv(job$matrix, header = FALSE))
        change <- as.matrix(read.csv(job$change, header = FALSE))
        rownames(dist) <- colnames(dist) <- nodes$id
        rownames(change) <- colnames(change) <- seq_len(nrow(change))
        thr <- as.numeric(job$distance)
        prob <- if (job$probability == "") NULL else as.numeric(job$probability)
        pij_min <- if (job$pij_min == "") NULL else as.numeric(job$pij_min)
        res <- MK_dPCIIC_links(nodes = nodes, distance = dist, metric = job$metric, probability = prob,
                               pij_min = pij_min, distance_thresholds = thr, removal = TRUE, change = change,
                               intern = FALSE)
        for (part in res) {
            value <- names(part)[ncol(part)]
            rows[[length(rows) + 1]] <- data.frame(scene = job$scene, metric = job$metric, distance = job$distance,
                                                   probability = job$probability, pij_min = job$pij_min,
                                                   level = value, source = nodes$id[part$Source],
                                                   destination = nodes$id[part$Destination],
                                                   value = fmt(part[[value]]))
        }
        n_res <- as.integer(job$restore)
        if (n_res > 0) {
            nodes$restoration <- c(rep(1, nrow(nodes) - n_res), rep(0, n_res))
            r2 <- MK_dPCIIC(nodes = nodes, distance = dist, metric = job$metric, probability = prob,
                            pij_min = pij_min, distance_thresholds = thr, restoration = "restoration",
                            intern = FALSE)
            col <- paste0("d", job$metric, "res")
            rows[[length(rows) + 1]] <- data.frame(scene = job$scene, metric = job$metric, distance = job$distance,
                                                   probability = job$probability, pij_min = job$pij_min,
                                                   level = col, source = r2[[1]], destination = NA,
                                                   value = fmt(r2[[col]]))
        }
    }
    write.csv(do.call(rbind, rows), args[3], row.names = FALSE)
    quit(save = "no")
}

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
