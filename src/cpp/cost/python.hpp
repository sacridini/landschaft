// SPDX-License-Identifier: GPL-3.0-or-later
//
// The grid distances' Python functions (cost_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace cost {

void register_python(pybind11::module_& m);

}  // namespace cost
