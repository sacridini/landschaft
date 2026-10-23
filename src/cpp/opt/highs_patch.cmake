# Changes to HiGHS 1.15.1's sources, applied by FetchContent's PATCH_COMMAND (run from the source tree).
#
# The first two keep the MIP's time limit on wide models (10^5 columns, 10^6 non-zeros), where the MIP
# root otherwise ran 20 s past a 30 s limit:
# 1. the line search of the central rounding (from the root LP's solution to the analytic centre, with
#    propagation and conflict analysis at every step) checks the limits between its steps;
# 2. the analytic centre's interior point method gets the time the MIP has left.
# 3. A separator of the user's cuts (HighsCallback::user_separator, set by Highs::setUserSeparator), run
#    with HiGHS's own separators at the root and at the nodes: told the LP's solution, it gives cuts
#    a x <= upper on the model's columns, which go to the cut pool. HiGHS's callbacks have no such thing
#    (kCallbackMipDefineLazyConstraints is declared, not implemented). It runs only while the MIP works on
#    the user's columns: no presolve reductions (presolve off) and not in a sub-MIP of the heuristics.

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

patch_once("highs/lp_data/HighsCallback.h" "landschaft: a separator"
    "  HighsCallbackInput data_in;\n"
    "  // landschaft: a separator of the user's cuts, told the LP's solution at the root and at the nodes; it\n  // appends cuts a x <= upper on the model's columns in compressed rows (start holds 0 when called)\n  std::function<void(const std::vector<double>& x, std::vector<HighsInt>& start,\n                     std::vector<HighsInt>& index, std::vector<double>& value,\n                     std::vector<double>& upper)>\n      user_separator = nullptr;\n")

patch_once("highs/Highs.h" "landschaft: the separator"
    "  HighsStatus setCallback(HighsCCallbackType c_callback,\n                          void* user_callback_data = nullptr);\n"
    "\n  // landschaft: the separator of the user's cuts in the MIP (HighsCallback::user_separator)\n  void setUserSeparator(decltype(HighsCallback::user_separator) separator) {\n    callback_.user_separator = std::move(separator);\n  }\n")

patch_once("highs/mip/HighsSeparation.cpp" "landschaft: the user's separator,"
    "#include \"mip/HighsTransformedLp.h\"\n"
    "
// landschaft: the user's separator, asked while the MIP works on the user's own columns
namespace {
class HighsUserSeparator : public HighsSeparator {
 public:
  explicit HighsUserSeparator(const HighsMipSolver& mipsolver)
      : HighsSeparator(mipsolver, \"Separation: user\") {}

  void separateLpSolution(HighsLpRelaxation& lpRelaxation, HighsLpAggregator&,
                          HighsTransformedLp&, HighsCutPool& cutpool) override {
    const HighsMipSolver& mipsolver = lpRelaxation.getMipSolver();
    if (mipsolver.submip ||
        mipsolver.model_->num_col_ != mipsolver.orig_model_->num_col_ ||
        mipsolver.mipdata_->postSolveStack.numReductions() != 0)
      return;
    start_.assign(1, 0);
    index_.clear();
    value_.clear();
    upper_.clear();
    mipsolver.callback_->user_separator(lpRelaxation.getSolution().col_value,
                                        start_, index_, value_, upper_);
    for (size_t k = 0; k + 1 < start_.size(); ++k)
      cutpool.addCut(mipsolver, index_.data() + start_[k],
                     value_.data() + start_[k], start_[k + 1] - start_[k],
                     upper_[k]);
  }

 private:
  std::vector<HighsInt> start_, index_;
  std::vector<double> value_, upper_;
};
}  // namespace
")

patch_once("highs/mip/HighsSeparation.cpp" "landschaft: the user's separator last"
    "  separators.emplace_back(new HighsModkSeparator(mipsolver));\n"
    "  // landschaft: the user's separator last\n  if (mipsolver.callback_ != nullptr && mipsolver.callback_->user_separator &&\n      !mipsolver.submip)\n    separators.emplace_back(new HighsUserSeparator(mipsolver));\n")
