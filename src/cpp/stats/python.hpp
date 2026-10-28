// SPDX-License-Identifier: GPL-3.0-or-later
//
// The statistics of maps' Python functions (stats_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace stats {

void register_python(pybind11::module_& m);

}  // namespace stats
