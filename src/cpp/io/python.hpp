// SPDX-License-Identifier: GPL-3.0-or-later
//
// Zonal statistics and extraction at points (io_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace lsio {

void register_python(pybind11::module_& m);

}  // namespace lsio
