// SPDX-License-Identifier: GPL-3.0-or-later
//
// The MILP prioritization's Python functions (opt_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace opt {

void register_python(pybind11::module_& m);

}  // namespace opt
