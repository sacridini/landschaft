// SPDX-License-Identifier: GPL-3.0-or-later
//
// Movement ecology's Python functions (move_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace move {

void register_python(pybind11::module_& m);

}  // namespace move
