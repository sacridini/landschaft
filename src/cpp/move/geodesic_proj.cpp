// SPDX-License-Identifier: GPL-3.0-or-later
//
// GeographicLib C 2.1 as PROJ 9.4.1 ships it (src/geodesic.c, MIT), the
// geodesics lwgeom 0.2's liblwgeom calls (lwgeom_azumith_spheroid ->
// spheroid_direction -> geod_inverse) when built with PROJ, as R's Windows
// binaries are. Compiled as C++ inside namespace move::proj_geodesic with its
// public functions renamed (move_proj_geod_*), so it does not clash with the
// other GeographicLib copies (sdm/geocircle). sin, cos, atan2, hypot and cbrt
// are mingw-w64's (sdm/geocircle/geocircle_libm.hpp), as in Rtools' PROJ.
// The vendored source is #included here and must not be compiled on its own.
#include "move/geo.hpp"

#include <cfloat>
#include <cmath>
#include <float.h>
#include <math.h>

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

#define geod_init move_proj_geod_init
#define geod_direct move_proj_geod_direct
#define geod_gendirect move_proj_geod_gendirect
#define geod_inverse move_proj_geod_inverse
#define geod_geninverse move_proj_geod_geninverse
#define geod_lineinit move_proj_geod_lineinit
#define geod_directline move_proj_geod_directline
#define geod_gendirectline move_proj_geod_gendirectline
#define geod_inverseline move_proj_geod_inverseline
#define geod_position move_proj_geod_position
#define geod_genposition move_proj_geod_genposition
#define geod_setdistance move_proj_geod_setdistance
#define geod_gensetdistance move_proj_geod_gensetdistance
#define geod_polygon_init move_proj_geod_polygon_init
#define geod_polygon_clear move_proj_geod_polygon_clear
#define geod_polygon_addpoint move_proj_geod_polygon_addpoint
#define geod_polygon_addedge move_proj_geod_polygon_addedge
#define geod_polygon_compute move_proj_geod_polygon_compute
#define geod_polygon_testpoint move_proj_geod_polygon_testpoint
#define geod_polygon_testedge move_proj_geod_polygon_testedge
#define geod_polygonarea move_proj_geod_polygonarea
namespace move {
namespace proj_geodesic {
#include "vendor/geographiclib-c-2.1/geodesic.c"
}  // namespace proj_geodesic
}  // namespace move

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

namespace move {

Spheroid::Spheroid(double a, double inv_flattening) {
    auto* g = new proj_geodesic::geod_geodesic;
    // liblwgeom's spheroid_init(a, b = a * (1 - 1 / rf)) and its f = (a - b) / a
    const double b = a * (1.0 - 1.0 / inv_flattening);
    proj_geodesic::move_proj_geod_init(g, a, (a - b) / a);
    impl_ = g;
}

Spheroid::~Spheroid() { delete static_cast<proj_geodesic::geod_geodesic*>(impl_); }

double Spheroid::azimuth_deg(double lat1, double lon1, double lat2, double lon2) const {
    double azi1 = 0;
    proj_geodesic::move_proj_geod_inverse(static_cast<const proj_geodesic::geod_geodesic*>(impl_), lat1, lon1, lat2,
                                          lon2, nullptr, &azi1, nullptr);
    return azi1;
}

}  // namespace move
