# SPDX-License-Identifier: GPL-3.0-or-later
#
# Reference values of Makurhini's MK_dPCIIC for landschaft's habitat graphs.
#
#   Rscript conn_makurhini_reference.R spatial <out_dir> <polygons.gpkg> <resistance.tif>
#       nodesfile / distancefile (centroid, edge) on Makurhini's example raster
#       `habitat_nodes_raster` (also written to out_dir) and on the polygons,
#       least-cost distances over `resistance_matrix` (written too) and over
#       the given resistance for the polygons, and MK_dPCIIC run end to end
#   Rscript conn_makurhini_reference.R centrality <polygons.gpkg> <raster.tif> <out.csv>
#       MK_RMCentrality (binary and weighted) on the polygons and the example raster
#   Rscript conn_makurhini_reference.R focal <polygons.gpkg> <protconn_areas.gpkg> <out.csv>
#       MK_Focal_nodes (IIC and PC, one and two distances) on the polygons and the ProtConn areas
#   Rscript conn_makurhini_reference.R commute <out_dir> <polygons.gpkg> <polygons_resistance.tif>
#       commute-time distances (distancefile) on the polygons and the example raster
#   Rscript conn_makurhini_reference.R samc <polygons.gpkg> <resistance.tif> <absorption.tif> <out.csv>
#       MK_dPC_SAMC (and the settlement probabilities) on the polygons
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

if (args[1] == "deca") {
    # MK_dECA on time series: subsets of the polygons and of the patch raster.
    # steps.csv: kind (vector / raster), time, id (the nodes kept at that time)
    polygons <- sf::st_read(args[2], quiet = TRUE)
    r <- terra::rast(args[3])
    steps <- read.csv(args[4])
    out <- list()
    keep <- function(kind, metric, tab) {
        tab <- as.data.frame(tab)
        names(tab) <- c("time", "LA", "area", "distance", "ECA", "ECA_LA", "ECA_area", "dA", "dECA", "rECA",
                         "comparison", "change")
        for (col in c("LA", "area", "distance", "ECA", "ECA_LA", "ECA_area", "dA", "dECA", "rECA")) {
            tab[[col]] <- fmt(as.numeric(tab[[col]]))
        }
        out[[length(out) + 1]] <<- cbind(kind = kind, metric = metric, tab)
    }
    times <- sort(unique(steps$time))
    vec <- lapply(times, function(t) polygons[polygons$id %in% steps$id[steps$kind == "vector" & steps$time == t], ])
    la_v <- as.numeric(args[5])
    for (m in c("PC", "IIC")) {
        res <- MK_dECA(nodes = vec, area_unit = "km2", distance = list(type = "centroid"), metric = m,
                       probability = if (m == "PC") 0.5 else NULL, distance_thresholds = c(2000, 6000),
                       LA = la_v, plot = FALSE, intern = FALSE)
        for (tab in res) keep("vector", m, tab)
    }
    ras <- lapply(times, function(t) {
        ids <- steps$id[steps$kind == "raster" & steps$time == t]
        x <- terra::deepcopy(r)
        v <- terra::values(x, mat = FALSE)
        v[!(v %in% ids)] <- NA
        terra::values(x) <- v
        raster::raster(x)
    })
    res <- MK_dECA(nodes = ras, area_unit = "ha", distance = list(type = "centroid"), metric = "PC",
                   probability = 0.5, distance_thresholds = 10000, LA = as.numeric(args[6]), plot = FALSE,
                   intern = FALSE)
    keep("raster", "PC", res)
    write.csv(do.call(rbind, out), args[7], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "protconn") {
    # MK_ProtConn on synthetic protected areas and regions, with its inner steps
    # (Protconn_nodes, protconn_dist, get_protconn_grid unrounded) and the delta.
    # jobs.csv: case, region, distance (";"-separated), transboundary, type, bound
    pas <- sf::st_read(args[2], quiet = TRUE)
    regions <- sf::st_read(args[3], quiet = TRUE)
    jobs <- read.csv(args[4], stringsAsFactors = FALSE)
    out <- list()
    add <- function(case, level, name, value, extra = NA) {
        out[[length(out) + 1]] <<- data.frame(case = case, level = level, name = name, extra = extra,
                                              value = fmt(as.numeric(value)))
    }
    for (k in seq_len(nrow(jobs))) {
        job <- jobs[k, ]
        region <- regions[regions$id == job$region, ]
        d <- as.numeric(strsplit(as.character(job$distance), ";")[[1]])
        tb <- if (is.na(job$transboundary)) NULL else job$transboundary
        res <- MK_ProtConn(nodes = pas, region = region, area_unit = "ha", distance = list(type = "edge"),
                           distance_thresholds = d, probability = 0.5, transboundary = tb,
                           transboundary_type = job$type, protconn_bound = as.logical(job$bound),
                           delta = TRUE, intern = FALSE)
        if (length(d) == 1) res <- list(res)
        for (i in seq_along(d)) {
            r <- res[[i]]
            tab <- as.data.frame(r[[1]])
            add(job$case, "public", tab[[3]], tab[[4]], d[i])
            dl <- sf::st_drop_geometry(r[[2]])
            if (is.data.frame(dl) && "dProtConn" %in% names(dl)) {
                add(job$case, "dProt", dl$id, dl$dProt, d[i])
                add(job$case, "dProtConn", dl$id, dl$dProtConn, d[i])
            }
        }
        # Inner steps, as MK_ProtConn runs them.
        bp1 <- Makurhini:::input_grid(node = pas, landscape = region, unit = "ha",
                                      bdist = if (is.null(tb)) 0 else tb)
        nodes <- Makurhini:::Protconn_nodes(x = bp1@region, y = bp1@nodes, buff = if (is.null(tb)) -Inf else tb,
                                            method = job$type, metrunit = "ha",
                                            protconn_bound = as.logical(job$bound), delta = FALSE)
        if (is.list(nodes)) {
            la <- as.numeric(sf::st_area(bp1@region)) * 0.0001
            add(job$case, "LA", "LA", la)
            nd <- nodes[[1]]
            add(job$case, "node_attribute", nd$type, nd$attribute, nd$OBJECTID)
            add(job$case, "piece", "piece", nodes[[2]]$attribute)
            dist <- Makurhini:::protconn_dist(nd, id = "OBJECTID", y = list(type = "edge"), r = bp1@region)
            ij <- which(upper.tri(dist), arr.ind = TRUE)
            add(job$case, "distance", paste(ij[, 1], ij[, 2]), dist[ij])
            for (dd in d) {
                raw <- Makurhini:::get_protconn_grid(x = nodes, y = dist, p = 0.5, pij_min = NULL, d = dd,
                                                     th = FALSE, LA = la, bound = as.logical(job$bound))
                add(job$case, "raw", names(raw), unlist(raw[1, ]), dd)
            }
        }
    }
    write.csv(do.call(rbind, out), args[5], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "protconn_raster") {
    # MK_ProtConn_raster on a raster of protected-area ids (ids in scan order). It
    # needs two or more transboundary nodes (it numbers the centroids only then).
    pas <- raster::raster(terra::rast(args[2]))
    regions <- sf::st_read(args[3], quiet = TRUE)
    region <- regions[regions$id == 1, ]
    out <- list()
    for (case in c("nodes", "region", "bound")) {
        res <- MK_ProtConn_raster(nodes = pas, region = region, area_unit = "ha", distance = list(type = "edge"),
                                  distance_thresholds = c(2000, 6000), probability = 0.5,
                                  transboundary = if (case == "bound") 6000 else 4000,
                                  transboundary_type = if (case == "region") "region" else "nodes",
                                  protconn_bound = case == "bound", intern = FALSE)
        for (i in seq_along(res)) {
            tab <- as.data.frame(res[[i]])
            out[[length(out) + 1]] <- data.frame(case = case, distance = c(2000, 6000)[i], name = tab[[3]],
                                                 value = fmt(as.numeric(tab[[4]])))
        }
    }
    write.csv(do.call(rbind, out), args[4], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "deca_grid") {
    # MK_dECA_grid on the three times of the polygons (deca_steps.csv), over their bounding box.
    polygons <- sf::st_read(args[2], quiet = TRUE)
    steps <- read.csv(args[3])
    times <- sort(unique(steps$time))
    vec <- lapply(times, function(t) polygons[polygons$id %in% steps$id[steps$kind == "vector" & steps$time == t], ])
    region <- sf::st_sf(geometry = sf::st_as_sfc(sf::st_bbox(polygons)))
    res <- MK_dECA_grid(nodes = vec, region = region, grid = list(hexagonal = TRUE, cellsize = 6e7),
                        area_unit = "km2", distance = list(type = "centroid"), metric = "PC",
                        distance_threshold = 3000, probability = 0.5, intern = FALSE)
    out <- do.call(rbind, lapply(res, function(x) {
        tab <- sf::st_drop_geometry(x)
        for (col in names(tab)) if (is.numeric(tab[[col]])) tab[[col]] <- fmt(tab[[col]])
        tab
    }))
    write.csv(out, args[4], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "grid") {
    # make_grid and MK_Connect_grid (ProtConn and PC) on the ProtConn polygons.
    pas <- sf::st_read(args[2], quiet = TRUE)
    region <- sf::st_read(args[3], quiet = TRUE)
    region <- region[region$id == 1, ]
    out <- args[4]
    for (hex in c(TRUE, FALSE)) {
        g <- Makurhini::make_grid(region, hexagonal = hex, cell_area = 2.5e7)
        write.csv(data.frame(OBJECTID = g$OBJECTID, wkt = sf::st_as_text(sf::st_geometry(g), digits = 17)),
                  file.path(out, paste0("grid_", if (hex) "hex" else "square", ".csv")), row.names = FALSE)
    }
    for (case in c("protconn", "protconn_pij0", "pc")) {
        res <- MK_Connect_grid(nodes = pas, region = region, area_unit = "ha",
                               grid = list(hexagonal = TRUE, cellsize = 2.5e7), protconn = case != "pc",
                               distance_threshold = 3000, probability = 0.5, transboundary = 2000,
                               pij_min = if (case == "protconn_pij0") 0 else NULL,
                               distance = list(type = "edge"), intern = FALSE)
        tab <- sf::st_drop_geometry(res)
        for (col in names(tab)) if (is.numeric(tab[[col]])) tab[[col]] <- fmt(tab[[col]])
        write.csv(tab, file.path(out, paste0("connect_grid_", case, ".csv")), row.names = FALSE, quote = FALSE)
    }
    quit(save = "no")
}

if (args[1] == "centrality") {
    # MK_RMCentrality (centroid distances) on the polygons and the example raster.
    polygons <- sf::st_read(args[2], quiet = TRUE)
    r <- terra::rast(args[3])
    cases <- list(list("vector", polygons, 2000, NULL, TRUE), list("vector", polygons, 5000, NULL, TRUE),
                  list("vector", polygons, 5000, NULL, FALSE), list("vector", polygons, 5000, 0.5, FALSE),
                  list("raster", r, 20000, NULL, TRUE), list("raster", r, 20000, 0.5, FALSE))
    out <- list()
    for (k in seq_along(cases)) {
        cs <- cases[[k]]
        set.seed(1)
        res <- MK_RMCentrality(nodes = cs[[2]], distance = list(type = "centroid"), distance_thresholds = cs[[3]],
                               probability = cs[[4]], binary = cs[[5]], intern = FALSE)
        if (cs[[1]] == "vector") {
            tab <- sf::st_drop_geometry(res)
            names(tab)[names(tab) == "id"] <- "node"
        } else {
            tab <- unique(terra::as.data.frame(res, na.rm = TRUE))
            names(tab)[1] <- "node"
        }
        tab$memb.louvain <- NULL
        long <- do.call(rbind, lapply(setdiff(names(tab), "node"), function(col)
            data.frame(case = k, kind = cs[[1]], distance = cs[[3]], probability = if (is.null(cs[[4]])) "" else cs[[4]],
                       binary = cs[[5]], node = tab$node, name = col, value = fmt(tab[[col]]))))
        out[[k]] <- long
    }
    write.csv(do.call(rbind, out), args[4], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "focal") {
    # MK_Focal_nodes on the polygons and the ProtConn areas.
    polygons <- sf::st_read(args[2], quiet = TRUE)
    pas <- sf::st_read(args[3], quiet = TRUE)
    cases <- list(list(polygons, "IIC", NULL, 3000, 6000, NULL, 0.01),
                  list(polygons, "PC", 0.5, c(2000, 5000), c(4000, 8000), NULL, 0.01),
                  list(polygons, "IIC", NULL, c(1500, 4000), 2500, NULL, 0.01),
                  list(polygons, "PC", 0.2, 4000, 7000, NULL, 0.001),
                  list(pas, "PC", 0.5, 3000, 5000, NULL, 0.01),
                  list(pas, "IIC", NULL, 2500, 4000, NULL, 0.01))
    out <- list()
    for (k in seq_along(cases)) {
        cs <- cases[[k]]
        d <- cs[[4]]
        sb <- if (length(cs[[5]]) < length(d)) rep(cs[[5]], length(d)) else cs[[5]]
        res <- MK_Focal_nodes(nodes = cs[[1]], id = "id", metric = cs[[2]], probability = cs[[3]],
                              distance_thresholds = d, search_buffer = sb, threshold = cs[[6]], pij_min = cs[[7]],
                              intern = FALSE)
        if (length(d) == 1) res <- list(res)
        for (j in seq_along(d)) {
            tab <- sf::st_drop_geometry(res[[j]])
            long <- do.call(rbind, lapply(setdiff(names(tab), "id"), function(col)
                data.frame(case = k, data = if (k <= 4) "polygons" else "pas", metric = cs[[2]],
                           probability = if (is.null(cs[[3]])) "" else cs[[3]], distance = d[j], search_buffer = sb[j],
                           threshold = if (is.null(cs[[6]])) "" else cs[[6]], pij_min = cs[[7]],
                           id = tab$id, name = col, value = fmt(tab[[col]]))))
            out[[length(out) + 1]] <- long
        }
    }
    write.csv(do.call(rbind, out), args[4], row.names = FALSE)
    quit(save = "no")
}

if (args[1] == "commute") {
    # Commute-time distances (gdistance's commuteDistance) with distancefile.
    out <- args[2]
    r <- terra::rast(file.path(dirname(args[3]), "habitat_nodes_raster.tif"))
    res_r <- raster::raster(terra::rast(file.path(dirname(args[3]), "resistance_matrix.tif")))
    polygons <- sf::st_read(args[3], quiet = TRUE)
    res_v <- raster::raster(terra::rast(args[4]))
    put <- function(what, table) {
        table <- as.data.frame(table)
        for (col in names(table)) if (is.numeric(table[[col]])) table[[col]] <- fmt(table[[col]])
        write.csv(table, file.path(out, paste0(what, ".csv")), row.names = FALSE, quote = FALSE)
    }
    t0 <- Sys.time()
    put("vector_commute", tryCatch(distancefile(polygons, id = "id", type = "commute-time", resistance = res_v),
                                   error = function(e) data.frame(error = conditionMessage(e))))
    put("vector_commute_4", tryCatch(distancefile(polygons, id = "id", type = "commute-time", resistance = res_v,
                                                  ngh = 4, CostFun = function(x) 1 / mean(x)),
                                     error = function(e) data.frame(error = conditionMessage(e))))
    cat("vector", format(Sys.time() - t0), "
")
    t0 <- Sys.time()
    put("raster_commute", distancefile(r, type = "commute-time", resistance = res_r))
    cat("raster", format(Sys.time() - t0), "
")
    quit(save = "no")
}

if (args[1] == "samc") {
    # MK_dPC_SAMC on the polygons over their resistance, with an explicit
    # absorption raster (Makurhini's default rasterizes its own example data).
    polygons <- sf::st_read(args[2], quiet = TRUE)
    resistance <- terra::rast(args[3])
    absorption <- terra::rast(args[4])
    out <- args[5]
    cases <- list(list(8, sum, "sum", 10, NULL), list(4, mean, "mean", 10, NULL), list(8, sum, "sum", NULL, 0.2),
                  list(8, mean, "mean", 3, NULL))
    rows <- list()
    for (k in seq_along(cases)) {
        cs <- cases[[k]]
        res <- MK_dPC_SAMC(nodes = polygons, attribute = NULL, area_unit = "ha", LA = 2e6,
                           disp_kernel = list(resistance = resistance, CostFun = function(x) 1 / mean(x),
                                              absorption = absorption, dir = cs[[1]], SettlFunc = cs[[2]],
                                              nvertices = cs[[4]], pvertices = cs[[5]], systematic = TRUE),
                           dispersal_dist = 5000, overall = TRUE, intern = FALSE)
        tab <- sf::st_drop_geometry(res[[1]])
        for (col in c("dPC", "dPCintra", "dPCflux", "dPCconnector"))
            rows[[length(rows) + 1]] <- data.frame(case = k, dir = cs[[1]], settle = cs[[3]],
                                                   nvertices = if (is.null(cs[[4]])) "" else cs[[4]],
                                                   pvertices = if (is.null(cs[[5]])) "" else cs[[5]],
                                                   level = "node", id = tab$id, name = col, value = fmt(tab[[col]]))
        ov <- res[[2]]
        rows[[length(rows) + 1]] <- data.frame(case = k, dir = cs[[1]], settle = cs[[3]],
                                               nvertices = if (is.null(cs[[4]])) "" else cs[[4]],
                                               pvertices = if (is.null(cs[[5]])) "" else cs[[5]],
                                               level = "overall", id = "", name = ov$Index, value = fmt(ov$Value))
        # the settlement probabilities themselves
        s <- terra::rast(args[3]); s <- c(s, absorption); s[any(is.na(s))] <- NA
        obj <- samc::samc(data = s[[1]], absorption = s[[2]],
                          model = list(fun = function(x) 1 / mean(x), dir = cs[[1]], sym = TRUE))
        nodes <- polygons; nodes$IdTemp <- seq_len(nrow(nodes))
        pij <- Makurhini:::samc_settlement(hn = nodes, samc_class = obj, pvertices = cs[[5]], nvertices = cs[[4]],
                                           SettlFunc = cs[[2]], sampletype = TRUE, simpl = NULL, intern = FALSE)
        ij <- which(upper.tri(pij), arr.ind = TRUE)
        rows[[length(rows) + 1]] <- data.frame(case = k, dir = cs[[1]], settle = cs[[3]],
                                               nvertices = if (is.null(cs[[4]])) "" else cs[[4]],
                                               pvertices = if (is.null(cs[[5]])) "" else cs[[5]],
                                               level = "pij", id = paste(polygons$id[ij[, 1]], polygons$id[ij[, 2]]),
                                               name = "pij", value = fmt(pij[ij]))
    }
    write.csv(do.call(rbind, rows), out, row.names = FALSE)
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
