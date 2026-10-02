// SPDX-License-Identifier: GPL-3.0-or-later
//
// The two GeographicLib versions the R packages use, compiled with R's
// elementary functions:
//  * GeographicLib C++ 2.7 as vendored by geosphere 1.6-8
//    (vendor/geographiclib-2.7; the GeodesicExact path removed), with its
//    namespace renamed to sdm_geosphere_GeographicLib;
//  * GeographicLib C 1.52 as vendored by terra 1.9-50
//    (vendor/geographiclib-c-1.52, unchanged), compiled as C++ inside
//    namespace sdm::terra_geodesic with its public functions renamed
//    (sdm_terra_geod_*), so neither clashes with another copy of
//    GeographicLib in the same program.
// The vendored sources are #included here and must not be compiled on their
// own. sin, cos, atan2, hypot and cbrt are mapped to geocircle_libm.hpp
// (mingw-w64's bits); everything else is the standard library's (exact
// operations or correctly rounded sqrt).
#include "sdm/geocircle/geocircle.hpp"

// Every standard header the vendored code includes, before the macros.
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <float.h>
#include <limits.h>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdm/geocircle/geocircle_libm.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4244 4456 4457 4458 4701 4702 4706 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#define sin(x) ::sdm::geocircle_libm::sin(x)
#define cos(x) ::sdm::geocircle_libm::cos(x)
#define atan2(y, x) ::sdm::geocircle_libm::atan2(y, x)
#define hypot(...) ::sdm::geocircle_libm::hypot(__VA_ARGS__)
#define cbrt(x) ::sdm::geocircle_libm::cbrt(x)

// --- GeographicLib C++ 2.7 (geosphere) -------------------------------------
#define GeographicLib sdm_geosphere_GeographicLib
#include "vendor/geographiclib-2.7/Math.cpp"
#include "vendor/geographiclib-2.7/Geodesic.cpp"
#include "vendor/geographiclib-2.7/GeodesicLine.cpp"
#undef GeographicLib

// --- GeographicLib C 1.52 (terra) ------------------------------------------
#undef GEOGRAPHICLIB_GEODESIC_ORDER  // the C++ headers' definition; the C file sets its own
#define geod_init sdm_terra_geod_init
#define geod_direct sdm_terra_geod_direct
#define geod_gendirect sdm_terra_geod_gendirect
#define geod_inverse sdm_terra_geod_inverse
#define geod_geninverse sdm_terra_geod_geninverse
#define geod_lineinit sdm_terra_geod_lineinit
#define geod_directline sdm_terra_geod_directline
#define geod_gendirectline sdm_terra_geod_gendirectline
#define geod_inverseline sdm_terra_geod_inverseline
#define geod_position sdm_terra_geod_position
#define geod_genposition sdm_terra_geod_genposition
#define geod_setdistance sdm_terra_geod_setdistance
#define geod_gensetdistance sdm_terra_geod_gensetdistance
#define geod_polygon_init sdm_terra_geod_polygon_init
#define geod_polygon_clear sdm_terra_geod_polygon_clear
#define geod_polygon_addpoint sdm_terra_geod_polygon_addpoint
#define geod_polygon_addedge sdm_terra_geod_polygon_addedge
#define geod_polygon_compute sdm_terra_geod_polygon_compute
#define geod_polygon_testpoint sdm_terra_geod_polygon_testpoint
#define geod_polygon_testedge sdm_terra_geod_polygon_testedge
#define geod_polygonarea sdm_terra_geod_polygonarea
namespace sdm {
namespace terra_geodesic {
#include "vendor/geographiclib-c-1.52/geodesic.c"
}  // namespace terra_geodesic
}  // namespace sdm

#undef sin
#undef cos
#undef atan2
#undef hypot
#undef cbrt

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace sdm {

namespace {

// WGS84 as geosphere and terra pass it: a = 6378137, f = 1/298.257223563.
constexpr double kA = 6378137.0;
constexpr double kF = 1 / 298.257223563;

const sdm_geosphere_GeographicLib::Geodesic& geosphere_wgs84() {
    static const sdm_geosphere_GeographicLib::Geodesic g(kA, kF);
    return g;
}

const terra_geodesic::geod_geodesic* terra_wgs84() {
    static const terra_geodesic::geod_geodesic g = [] {
        terra_geodesic::geod_geodesic t;
        terra_geodesic::sdm_terra_geod_init(&t, kA, kF);
        return t;
    }();
    return &g;
}

}  // namespace

void geosphere_direct(double lon1, double lat1, double azi1, double s12, double& lon2, double& lat2) {
    double azi2;
    geosphere_wgs84().Direct(lat1, lon1, azi1, s12, lat2, lon2, azi2);
}

double geosphere_distance(double lon1, double lat1, double lon2, double lat2) {
    double s12, azi1, azi2;
    geosphere_wgs84().Inverse(lat1, lon1, lat2, lon2, s12, azi1, azi2);
    return s12;
}

void terra_direct(double lon1, double lat1, double azi1, double s12, double& lon2, double& lat2) {
    double azi2;
    terra_geodesic::sdm_terra_geod_direct(terra_wgs84(), lat1, lon1, azi1, s12, &lat2, &lon2, &azi2);
}

double terra_distance(double lon1, double lat1, double lon2, double lat2) {
    double s12, azi1, azi2;
    terra_geodesic::sdm_terra_geod_inverse(terra_wgs84(), lat1, lon1, lat2, lon2, &s12, &azi1, &azi2);
    return s12;
}

}  // namespace sdm
