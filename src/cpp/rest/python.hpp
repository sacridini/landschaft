// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python functions of the restoration planning inputs (landschaft._core.rest_*): original
// vegetation, area of habitat, ecological intactness.
#pragma once

#include <pybind11/pybind11.h>

namespace rest {

void register_python(pybind11::module_& m);

}  // namespace rest
