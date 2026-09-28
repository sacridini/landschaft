// SPDX-License-Identifier: GPL-3.0-or-later
//
// The neutral landscape models' Python functions (nlm_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace nlm {

void register_python(pybind11::module_& m);

}  // namespace nlm
