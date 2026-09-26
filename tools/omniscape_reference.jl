# SPDX-License-Identifier: GPL-3.0-or-later
#
# Reference outputs of Circuitscape 6.1 / Omniscape 0.7 for landschaft's
# circuit module; called by tools/make_reference_omniscape.py.
#
#   julia omniscape_reference.jl advanced <inputs.npz> <outputs.npz>
#
# `advanced`: every case `<name>/conductance`, `<name>/source`, `<name>/ground`
# (NaN = no node) and `<name>/four` goes through compute_omniscape_current
# with the cholmod and the cg+amg solvers.
using Circuitscape, NPZ
import Circuitscape: compute_omniscape_current

function cs_config(solver, four)
    d = Dict("ground_file_is_resistances" => "True", "use_direct_grounds" => "False",
             "output_file" => "temp", "write_cum_cur_map_only" => "False", "scenario" => "Advanced",
             "suppress_messages" => "True", "connect_four_neighbors_only" => four ? "True" : "False",
             "cholmod_batch_size" => "1000", "solver" => solver)
    c = Circuitscape.init_config()
    Circuitscape.update!(c, d)
    c
end

nan_to(x, v) = map(a -> isnan(a) ? v : a, x)

function advanced(inputs, outputs)
    data = npzread(inputs)
    names = sort(unique(first.(split.(collect(keys(data)), "/"))))
    out = Dict{String, Any}()
    for name in names
        g = nan_to(Float64.(data["$name/conductance"]), -9999.0)
        s = nan_to(Float64.(data["$name/source"]), 0.0)
        gr = nan_to(Float64.(data["$name/ground"]), 0.0)
        four = only(data["$name/four"]) != 0
        for (label, solver) in (("cholmod", "cholmod"), ("cgamg", "cg+amg"))
            out["$name/current_$label"] = compute_omniscape_current(g, s, gr, cs_config(solver, four))
        end
        println("advanced $name done")
    end
    npzwrite(outputs, out)
end

if ARGS[1] == "advanced"
    advanced(ARGS[2], ARGS[3])
else
    error("unknown task $(ARGS[1])")
end
