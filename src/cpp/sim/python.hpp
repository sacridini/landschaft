// SPDX-License-Identifier: GPL-3.0-or-later
//
// The land change simulation Python functions (sim_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace sim {

void register_python(pybind11::module_& m);

}  // namespace sim
