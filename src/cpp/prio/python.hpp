// SPDX-License-Identifier: GPL-3.0-or-later
//
// The spatial prioritization's Python classes and functions (prio_*),
// registered into landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace prio {

void register_python(pybind11::module_& m);

}  // namespace prio
