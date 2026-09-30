// SPDX-License-Identifier: GPL-3.0-or-later
//
// The surface metrics' Python functions (surf_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace surf {

void register_python(pybind11::module_& m);

}  // namespace surf
