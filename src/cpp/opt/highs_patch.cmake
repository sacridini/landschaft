# Changes to HiGHS 1.15.1's sources, applied by FetchContent's PATCH_COMMAND (run from the source tree).
#
# Both keep the MIP's time limit on wide models (10^5 columns, 10^6 non-zeros), where the MIP root
# otherwise ran 20 s past a 30 s limit:
# 1. the line search of the central rounding (from the root LP's solution to the analytic centre, with
#    propagation and conflict analysis at every step) checks the limits between its steps;
# 2. the analytic centre's interior point method gets the time the MIP has left.

function(patch_once file marker anchor added)
    file(READ "${file}" text)
    string(FIND "${text}" "${marker}" done)
    if(NOT done EQUAL -1)
        return()
    endif()
    string(FIND "${text}" "${anchor}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "highs_patch.cmake: '${anchor}' is not in ${file} (not HiGHS 1.15.1?)")
    endif()
    string(REPLACE "${anchor}" "${anchor}${added}" text "${text}")
    file(WRITE "${file}" "${text}")
endfunction()

patch_once("highs/mip/HighsPrimalHeuristics.cpp" "landschaft: the line search keeps"
    "  while (alpha < 1.0) {\n"
    "    // landschaft: the line search keeps the MIP's limits\n    if (mipsolver.mipdata_->checkLimits()) return false;\n")

patch_once("highs/mip/HighsMipSolverData.cpp" "landschaft: the analytic centre"
    "    ipm.setOptionValue(\"ipm_iteration_limit\", 200);\n"
    "    // landschaft: the analytic centre keeps the MIP's time limit\n    ipm.setOptionValue(\"time_limit\", std::max(0.0, mipsolver.options_mip_->time_limit - mipsolver.timer_.read()));\n")
