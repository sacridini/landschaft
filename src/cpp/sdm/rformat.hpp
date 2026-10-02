// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's text form of a double, as.character(x) (and so paste(x)): the keys by
// which R groups numbers in as.factor() and merge().
#pragma once

#include <string>

namespace sdm {

// as.character(<double>) of R 4.4.2 with the default options (scipen = 0,
// OutDec = "."): coerceToString() with R_print.digits = DBL_DIG (15), i.e.
// formatReal() of the one value (format.c: scientific() in R's 80-bit long
// double, so not always the correctly rounded 15 digits) and
// EncodeRealDrop0() (printutils.c) printing through trio as R for Windows
// does. Checked against R on 8.2 million values. NA gives "NA", NaN "NaN", +-Inf "Inf" /
// "-Inf", -0 "0".
std::string r_as_character(double x);

}  // namespace sdm
