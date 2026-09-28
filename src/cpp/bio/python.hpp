// SPDX-License-Identifier: GPL-3.0-or-later
//
// The biodiversity module's Python functions (bio_*), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace bio {

void register_python(pybind11::module_& m);

}  // namespace bio
