# Phase 0 of the Omniscape port (IDEAS.md): time one Omniscape moving-window solve
# (Circuitscape.compute_omniscape_current) on a disk of radius R with random conductances.
# Run with the private depot: JULIA_DEPOT_PATH=... julia --project=... -t 1 tools/omniscape_window_benchmark.jl 25 50 100 200
using Circuitscape, Random, SparseArrays, LinearAlgebra
import Circuitscape: compute_omniscape_current

function window(R, rng)
    W = 2R + 1
    g = 1 ./ (1 .+ 99 .* rand(rng, W, W))
    for i in 1:W, j in 1:W
        sqrt((i - R - 1)^2 + (j - R - 1)^2) > R && (g[i, j] = -9999.0)
    end
    s = zeros(W, W)
    for i in 1:W, j in 1:W
        g[i, j] > 0 && (s[i, j] = 1.0)
    end
    s[R + 1, R + 1] = 0.0
    s ./= sum(s)
    gr = zeros(W, W)
    gr[R + 1, R + 1] = Inf
    g, s, gr
end

function cfg(solver)
    d = Dict("ground_file_is_resistances" => "True", "use_direct_grounds" => "False",
             "output_file" => "temp", "write_cum_cur_map_only" => "False", "scenario" => "Advanced",
             "suppress_messages" => "True", "connect_four_neighbors_only" => "false",
             "cholmod_batch_size" => "1000", "solver" => solver)
    c = Circuitscape.init_config()
    Circuitscape.update!(c, d)
    c
end

rng = MersenneTwister(42)
radii = isempty(ARGS) ? [25, 50, 100, 200] : parse.(Int, ARGS)
BLAS.set_num_threads(1)
for R in radii
    g, s, gr = window(R, rng)
    for solver in ("cg+amg", "cholmod")
        c = cfg(solver)
        compute_omniscape_current(g, s, gr, c)
        reps = R <= 50 ? 20 : (R <= 100 ? 5 : 2)
        t = @elapsed for _ in 1:reps
            compute_omniscape_current(g, s, gr, c)
        end
        println("R=$R solver=$solver  $(round(t / reps * 1e3, digits = 3)) ms")
    end
    flush(stdout)
end
