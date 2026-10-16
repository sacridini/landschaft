// SPDX-License-Identifier: GPL-3.0-or-later
//
// The terrain hydrology's Python functions (hydro_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace hydro {

void register_python(pybind11::module_& m);

}  // namespace hydro
