// SPDX-License-Identifier: GPL-3.0-or-later
//
// The species distribution modelling Python functions (sdm_*) and R's random
// number generator as a Python object (RRandom), registered into
// landschaft._core.
#pragma once

#include <pybind11/pybind11.h>

namespace sdm {

void register_python(pybind11::module_& m);

}  // namespace sdm
