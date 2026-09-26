# SPDX-License-Identifier: GPL-3.0-or-later
#
# Reference outputs of Circuitscape 6.1 for landschaft's circuit module;
# called by tools/make_reference_circuitscape.py.
#
#   julia circuitscape_reference.jl <root> <outdir> <ini> [<ini> ...]
#
# From `root` (the copy of Circuitscape's test directory, whose INI files hold
# paths relative to it), every INI runs through Circuitscape.compute with the
# cholmod solver and its outputs written to `outdir` under the INI's name; the
# returned matrix goes to `<outdir>/<name>_result.txt`.
using ArchGDAL, Circuitscape, DelimitedFiles
import Circuitscape: parse_config

root, outdir = ARGS[1], ARGS[2]
cd(root)
for ini in ARGS[3:end]
    name = splitext(basename(ini))[1]
    d = Dict{String,String}(parse_config(ini))
    d["solver"] = "cholmod"
    d["parallelize"] = "false"
    d["output_file"] = joinpath(outdir, name * ".out")
    haskey(d, "log_file") && d["log_file"] != "None" && (d["log_file"] = joinpath(outdir, name * ".log"))
    r = Circuitscape.compute(d)
    writedlm(joinpath(outdir, name * "_result.txt"), r)
    println("circuitscape $name done")
    flush(stdout)
end
