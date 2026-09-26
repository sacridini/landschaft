// SPDX-License-Identifier: GPL-3.0-or-later
//
// Polygon masks as GDAL's rasterizer burns them with ALL_TOUCHED=TRUE (what
// terra::crop(mask = TRUE) and rasterio.features.rasterize(all_touched=True)
// use for sample plots): every ring is walked by GDALdllImageLineAllTouched
// (intersect-only variant, with GDAL's epsilons), then the polygon is filled
// by GDALdllImageFilledPolygon's centre-of-cell scanlines. Coordinates are
// mapped to pixels with GDAL's inverse geotransform, so the cells are the
// same as GDAL's (GDAL 3.x, alg/llrasterize.cpp; MIT licence, adapted).
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lsm/landscape.hpp"

namespace lsm {

namespace {

struct Mask {
    int nx, ny;
    std::uint8_t* data;
    void point(int y, int x) const {
        if (x >= 0 && x < nx && y >= 0 && y < ny) data[static_cast<std::size_t>(y) * nx + x] = 1;
    }
    void scanline(int y, int x0, int x1) const {
        if (y < 0 || y >= ny) return;
        x0 = std::max(x0, 0);
        x1 = std::min(x1, nx - 1);
        for (int x = x0; x <= x1; ++x) data[static_cast<std::size_t>(y) * nx + x] = 1;
    }
};

// GDALdllImageLineAllTouched with bIntersectOnly and without variants or
// avoided points.
void line_all_touched(const Mask& m, const std::vector<int>& part_size, const std::vector<double>& px,
                      const std::vector<double>& py) {
    constexpr double kEpsilonIntersectOnly = 1e-4;
    const int nx = m.nx;
    const int ny = m.ny;
    for (std::size_t i = 0, n = 0; i < part_size.size(); n += part_size[i++]) {
        for (int j = 1; j < part_size[i]; ++j) {
            double dfX = px[n + j - 1];
            double dfY = py[n + j - 1];
            double dfXEnd = px[n + j];
            double dfYEnd = py[n + j];
            if ((dfY < 0.0 && dfYEnd < 0.0) || (dfY > ny && dfYEnd > ny) || (dfX < 0.0 && dfXEnd < 0.0) ||
                (dfX > nx && dfXEnd > nx)) {
                continue;
            }
            if (!(dfX >= INT_MIN && dfX <= INT_MAX && dfY >= INT_MIN && dfY <= INT_MAX && dfXEnd >= INT_MIN &&
                  dfXEnd <= INT_MAX && dfYEnd >= INT_MIN && dfYEnd <= INT_MAX)) {
                continue;
            }
            if (dfX > dfXEnd) {
                std::swap(dfX, dfXEnd);
                std::swap(dfY, dfYEnd);
            }
            // vertical lines
            if (std::fabs(dfX - dfXEnd) < .01) {
                if (std::abs(dfX - std::round(dfX)) < kEpsilonIntersectOnly &&
                    std::abs(dfXEnd - std::round(dfXEnd)) < kEpsilonIntersectOnly) {
                    continue;
                }
                if (dfYEnd < dfY) std::swap(dfY, dfYEnd);
                const int iX = static_cast<int>(std::floor(dfXEnd));
                int iY = static_cast<int>(std::floor(dfY));
                int iYEnd = static_cast<int>(std::floor(dfYEnd - kEpsilonIntersectOnly));
                if (iX < 0 || iX >= nx) continue;
                if (iY < 0) iY = 0;
                if (iYEnd >= ny) iYEnd = ny - 1;
                for (; iY <= iYEnd; iY++) m.point(iY, iX);
                continue;
            }
            // horizontal lines
            if (std::fabs(dfY - dfYEnd) < .01) {
                if (std::abs(dfY - std::round(dfY)) < kEpsilonIntersectOnly &&
                    std::abs(dfYEnd - std::round(dfYEnd)) < kEpsilonIntersectOnly) {
                    continue;
                }
                if (dfXEnd < dfX) std::swap(dfX, dfXEnd);
                int iX = static_cast<int>(std::floor(dfX));
                const int iY = static_cast<int>(std::floor(dfY));
                int iXEnd = static_cast<int>(std::floor(dfXEnd - kEpsilonIntersectOnly));
                if (iY < 0 || iY >= ny) continue;
                if (iX < 0) iX = 0;
                if (iXEnd >= nx) iXEnd = nx - 1;
                for (; iX <= iXEnd; iX++) m.point(iY, iX);
                continue;
            }
            // general case, left to right
            const double dfSlope = (dfYEnd - dfY) / (dfXEnd - dfX);
            if (dfXEnd > nx) {
                dfYEnd -= (dfXEnd - nx) * dfSlope;
                dfXEnd = nx;
            }
            if (dfX < 0.0) {
                dfY += (0.0 - dfX) * dfSlope;
                dfX = 0.0;
            }
            if (dfYEnd > dfY) {
                if (dfY < 0.0) {
                    const double dfDiffX = (0.0 - dfY) / dfSlope;
                    dfX += dfDiffX;
                    dfY = 0.0;
                }
                if (dfYEnd >= ny) {
                    dfXEnd += (dfYEnd - ny) / dfSlope;
                    if (dfXEnd > nx) dfXEnd = nx;
                }
            } else {
                if (dfY >= ny) {
                    const double dfDiffX = (ny - dfY) / dfSlope;
                    dfX += dfDiffX;
                    dfY = ny;
                }
                if (dfYEnd < 0.0) dfXEnd -= (dfYEnd - 0) / dfSlope;
            }
            while (dfX >= 0.0 && dfX < dfXEnd) {
                const int iX = static_cast<int>(std::floor(dfX));
                const int iY = static_cast<int>(std::floor(dfY));
                if (iY >= 0 && iY < ny) m.point(iY, iX);
                double dfStepX = std::floor(dfX + 1.0) - dfX;
                double dfStepY = dfStepX * dfSlope;
                if (static_cast<int>(std::floor(dfY + dfStepY)) == iY) {
                    dfX += dfStepX;
                    dfY += dfStepY;
                } else if (dfSlope < 0) {
                    dfStepY = iY - dfY;
                    if (dfStepY > -0.000000001) dfStepY = -0.000000001;
                    dfStepX = dfStepY / dfSlope;
                    dfX += dfStepX;
                    dfY += dfStepY;
                } else {
                    dfStepY = (iY + 1) - dfY;
                    if (dfStepY < 0.000000001) dfStepY = 0.000000001;
                    dfStepX = dfStepY / dfSlope;
                    dfX += dfStepX;
                    dfY += dfStepY;
                }
            }
        }
    }
}

// GDALdllImageFilledPolygon without avoided points: cells whose centre is
// inside (even-odd over all rings), plus the bottom horizontal segments.
void filled_polygon(const Mask& m, const std::vector<int>& part_size, const std::vector<double>& padfX,
                    const std::vector<double>& padfY) {
    const int n = static_cast<int>(padfX.size());
    if (part_size.empty() || n == 0) return;
    std::vector<int> polyInts(static_cast<std::size_t>(n));
    double dminy = padfY[0];
    double dmaxy = padfY[0];
    for (int i = 1; i < n; i++) {
        if (padfY[i] < dminy) {
            dminy = padfY[i];
        } else if (padfY[i] > dmaxy) {
            dmaxy = padfY[i];
        }
    }
    const int miny = static_cast<int>(std::max(0.0, dminy));
    const int maxy = static_cast<int>(std::min<double>(dmaxy, m.ny - 1));
    constexpr int minx = 0;
    const int maxx = m.nx - 1;
    for (int y = miny; y <= maxy; y++) {
        int partoffset = 0;
        const double dy = y + 0.5;
        std::size_t part = 0;
        int ints = 0;
        for (int i = 0; i < n; i++) {
            if (i == partoffset + part_size[part]) {
                partoffset += part_size[part];
                part++;
            }
            int ind1, ind2;
            if (i == partoffset) {
                ind1 = partoffset + part_size[part] - 1;
                ind2 = partoffset;
            } else {
                ind1 = i - 1;
                ind2 = i;
            }
            double dy1 = padfY[ind1];
            double dy2 = padfY[ind2];
            if ((dy1 < dy && dy2 < dy) || (dy1 > dy && dy2 > dy)) continue;
            double dx1 = 0.0;
            double dx2 = 0.0;
            if (dy1 < dy2) {
                dx1 = padfX[ind1];
                dx2 = padfX[ind2];
            } else if (dy1 > dy2) {
                std::swap(dy1, dy2);
                dx2 = padfX[ind1];
                dx1 = padfX[ind2];
            } else {
                // bottom horizontal segments are filled separately
                if (padfX[ind1] > padfX[ind2]) {
                    const double dhorizontal_x1 = std::floor(padfX[ind2] + 0.5);
                    const double dhorizontal_x2 = std::floor(padfX[ind1] + 0.5);
                    if ((dhorizontal_x1 > static_cast<double>(maxx)) || (dhorizontal_x2 <= 0)) continue;
                    const int horizontal_x1 = static_cast<int>(std::max(dhorizontal_x1, 0.0));
                    const int horizontal_x2 = static_cast<int>(std::min<double>(dhorizontal_x2, m.nx));
                    m.scanline(y, horizontal_x1, horizontal_x2 - 1);
                }
                continue;
            }
            if (dy < dy2 && dy >= dy1) {
                const double intersect =
                    std::clamp<double>((dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1, INT_MIN, INT_MAX);
                polyInts[ints++] = static_cast<int>(std::floor(intersect + 0.5));
            }
        }
        std::sort(polyInts.begin(), polyInts.begin() + ints);
        for (int i = 0; i + 1 < ints; i += 2) {
            if (polyInts[i] <= maxx && polyInts[i + 1] > minx) m.scanline(y, polyInts[i], polyInts[i + 1] - 1);
        }
    }
}

}  // namespace

std::vector<std::uint8_t> rasterize_all_touched(const std::vector<std::vector<std::pair<double, double>>>& rings,
                                                const double* gt, int nx, int ny) {
    if (gt[2] != 0.0 || gt[4] != 0.0 || gt[1] == 0.0 || gt[5] == 0.0) {
        throw std::invalid_argument("rasterize: rotated or degenerate geotransform");
    }
    // GDALInvGeoTransform without rotation, then GDALApplyGeoTransform
    const double inv[6] = {-gt[0] / gt[1], 1.0 / gt[1], 0.0, -gt[3] / gt[5], 0.0, 1.0 / gt[5]};
    std::vector<int> part_size;
    std::vector<double> px, py;
    for (const auto& ring : rings) {
        part_size.push_back(static_cast<int>(ring.size()));
        for (const auto& p : ring) {
            px.push_back(inv[0] + p.first * inv[1] + p.second * inv[2]);
            py.push_back(inv[3] + p.first * inv[4] + p.second * inv[5]);
        }
    }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(nx) * ny, 0);
    const Mask m{nx, ny, out.data()};
    line_all_touched(m, part_size, px, py);
    filled_polygon(m, part_size, px, py);
    return out;
}

}  // namespace lsm
