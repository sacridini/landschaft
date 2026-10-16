// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <pybind11/pybind11.h>

namespace swat {

void register_python(pybind11::module_& m);

}  // namespace swat
