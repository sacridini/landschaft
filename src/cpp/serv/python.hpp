// SPDX-License-Identifier: GPL-3.0-or-later
//
// The ecosystem services' Python functions (serv_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace serv {

void register_python(pybind11::module_& m);

}  // namespace serv
