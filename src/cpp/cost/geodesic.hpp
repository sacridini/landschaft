// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geodesic distances on the WGS84 ellipsoid as the R package raster computes
// them (pointDistance(longlat = TRUE), which gdistance's geoCorrection calls):
// GeographicLib 2.0's geod_inverse (C. F. F. Karney, MIT licence, vendored in
// geographiclib/), with the maths of the mingw-w64 C library R's packages are
// built against on Windows (x87 sin, cos and atan2, the UCRT's hypot, Cephes'
// cbrt), so the distances are raster's to the last bit there.
#pragma once

namespace cost {

// A geodesic calculator for one ellipsoid (a: equatorial radius, f: flattening).
class Geodesic {
public:
    Geodesic(double a = 6378137.0, double f = 1.0 / 298.257223563);
    ~Geodesic();
    Geodesic(const Geodesic&) = delete;
    Geodesic& operator=(const Geodesic&) = delete;

    // Distance in metres from (lon1, lat1) to (lon2, lat2), degrees.
    double distance(double lon1, double lat1, double lon2, double lat2) const;

private:
    void* g_;
};

// mingw-w64's cbrt (Cephes), exposed for the tests.
double mingw_cbrt(double x);

}  // namespace cost
