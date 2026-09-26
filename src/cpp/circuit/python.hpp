// SPDX-License-Identifier: GPL-3.0-or-later
//
// The circuit module's Python functions (circuit_advanced, circuit_omniscape,
// circuit_omniscape_correction), registered into a pybind11 module.
#pragma once

#include <pybind11/pybind11.h>

namespace circuit {

void register_python(pybind11::module_& m);

}  // namespace circuit
