// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of parts of GEOS 3.12.2 (LGPL-2.1-or-later, Copyright (C) 2001-2023
// the GEOS contributors and Vivid Solutions), see geos_lite.hpp. Function and
// variable names follow GEOS so the code can be checked against it; each
// block names the GEOS source it comes from.
#include "sdm/geocircle/geos_lite.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace sdm {
namespace geos_lite {

namespace {

using std::size_t;

bool equals2D(const Coord& a, const Coord& b) { return a.x == b.x && a.y == b.y; }

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("geos_lite: unsupported overlay configuration: " + what);
}

// ---------------------------------------------------------------------------
// math/DD.cpp: double-double arithmetic (the operations CGAlgorithmsDD uses)

struct DD {
    double hi;
    double lo;
    explicit DD(double x) : hi(x), lo(0.0) {}
    DD(double h, double l) : hi(h), lo(l) {}
};

constexpr double SPLIT = 134217729.0;  // 2^27 + 1

void selfAdd(DD& a, double yhi, double ylo) {
    double H, h, T, t, S, s, e, f;
    S = a.hi + yhi;
    T = a.lo + ylo;
    e = S - a.hi;
    f = T - a.lo;
    s = S - e;
    t = T - f;
    s = (yhi - e) + (a.hi - s);
    t = (ylo - f) + (a.lo - t);
    e = s + T;
    H = S + e;
    h = e + (S - H);
    e = t + h;
    const double zhi = H + e;
    const double zlo = e + (H - zhi);
    a.hi = zhi;
    a.lo = zlo;
}

DD operator+(const DD& lhs, const DD& rhs) {
    DD rv(lhs.hi, lhs.lo);
    selfAdd(rv, rhs.hi, rhs.lo);
    return rv;
}

DD operator-(const DD& lhs, const DD& rhs) {
    DD rv(lhs.hi, lhs.lo);
    selfAdd(rv, -1 * rhs.hi, -1 * rhs.lo);  // DD::selfSubtract
    return rv;
}

DD operator*(const DD& lhs, const DD& rhs) {
    const double yhi = rhs.hi, ylo = rhs.lo;
    double hx, tx, hy, ty, C, c;
    C = SPLIT * lhs.hi;
    hx = C - lhs.hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = lhs.hi - hx;
    hy = c - yhi;
    C = lhs.hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (lhs.hi * ylo + lhs.lo * yhi);
    const double zhi = C + c;
    hx = C - zhi;
    const double zlo = c + hx;
    return DD(zhi, zlo);
}

DD operator/(const DD& lhs, const DD& rhs) {
    const double yhi = rhs.hi, ylo = rhs.lo;
    double hc, tc, hy, ty, C, c, U, u;
    C = lhs.hi / yhi;
    c = SPLIT * C;
    hc = c - C;
    u = SPLIT * yhi;
    hc = c - hc;
    tc = C - hc;
    hy = u - yhi;
    U = C * yhi;
    hy = u - hy;
    ty = yhi - hy;
    u = (((hc * hy - U) + hc * ty) + tc * hy) + tc * ty;
    c = ((((lhs.hi - U) - u) + lhs.lo) - C * ylo) / yhi;
    u = C + c;
    return DD(u, (C - u) + c);
}

// ---------------------------------------------------------------------------
// algorithm/CGAlgorithmsDD.cpp, Orientation.cpp

constexpr int CLOCKWISE = -1;
constexpr int COLLINEAR = 0;
constexpr int COUNTERCLOCKWISE = 1;
constexpr int FAILURE = 2;

int orientation(double x) {
    if (x < 0) return CLOCKWISE;
    if (x > 0) return COUNTERCLOCKWISE;
    return COLLINEAR;
}

int orientationIndexFilter(double pax, double pay, double pbx, double pby, double pcx, double pcy) {
    constexpr double DP_SAFE_EPSILON = 1e-15;
    double detsum;
    const double detleft = (pax - pcx) * (pby - pcy);
    const double detright = (pay - pcy) * (pbx - pcx);
    const double det = detleft - detright;
    if (detleft > 0.0) {
        if (detright <= 0.0) return orientation(det);
        detsum = detleft + detright;
    } else if (detleft < 0.0) {
        if (detright >= 0.0) return orientation(det);
        detsum = -detleft - detright;
    } else {
        return orientation(det);
    }
    const double errbound = DP_SAFE_EPSILON * detsum;
    if ((det >= errbound) || (-det >= errbound)) return orientation(det);
    return FAILURE;
}

// Orientation::index(p1, p2, q): CGAlgorithmsDD::orientationIndex.
int orientationIndex(const Coord& p1, const Coord& p2, const Coord& q) {
    if (!std::isfinite(q.x) || !std::isfinite(q.y)) unsupported("NaN/Inf coordinate");
    const int index = orientationIndexFilter(p1.x, p1.y, p2.x, p2.y, q.x, q.y);
    if (index <= 1) return index;
    const DD dx1 = DD(p2.x) + DD(-p1.x);
    const DD dy1 = DD(p2.y) + DD(-p1.y);
    const DD dx2 = DD(q.x) + DD(-p2.x);
    const DD dy2 = DD(q.y) + DD(-p2.y);
    const DD mx1y2(dx1 * dy2);
    const DD my1x2(dy1 * dx2);
    const DD d = mx1y2 - my1x2;
    if (d.hi < 0.0 || (d.hi == 0.0 && d.lo < 0.0)) return CLOCKWISE;
    if (d.hi > 0.0 || (d.hi == 0.0 && d.lo > 0.0)) return COUNTERCLOCKWISE;
    return COLLINEAR;
}

// CGAlgorithmsDD::intersection; returns false where GEOS returns a null
// coordinate (parallel lines).
bool intersectionDD(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2, Coord& out) {
    const DD q1x(q1.x), q1y(q1.y), q2x(q2.x), q2y(q2.y);
    const DD p1x(p1.x), p1y(p1.y), p2x(p2.x), p2y(p2.y);
    const DD px = p1y - p2y;
    const DD py = p2x - p1x;
    const DD pw = (p1x * p2y) - (p2x * p1y);
    const DD qx = q1y - q2y;
    const DD qy = q2x - q1x;
    const DD qw = (q1x * q2y) - (q2x * q1y);
    const DD x = (py * qw) - (qy * pw);
    const DD y = (qx * pw) - (px * qw);
    const DD w = (px * qy) - (qx * py);
    const DD xd = x / w, yd = y / w;
    const double xInt = xd.hi + xd.lo;  // DD::doubleValue
    const double yInt = yd.hi + yd.lo;
    if (!std::isfinite(xInt) || !std::isfinite(yInt)) return false;
    out = Coord{xInt, yInt};
    return true;
}

// ---------------------------------------------------------------------------
// geom/Envelope

struct Envelope {
    bool null = true;
    double minx = 0, maxx = -1, miny = 0, maxy = -1;

    void expandToInclude(const Coord& p) {
        if (null) {
            minx = maxx = p.x;
            miny = maxy = p.y;
            null = false;
            return;
        }
        if (p.x < minx) minx = p.x;
        if (p.x > maxx) maxx = p.x;
        if (p.y < miny) miny = p.y;
        if (p.y > maxy) maxy = p.y;
    }
    double width() const { return null ? 0.0 : maxx - minx; }
    double height() const { return null ? 0.0 : maxy - miny; }
    void expandBy(double d) {
        if (null) return;
        minx -= d;
        maxx += d;
        miny -= d;
        maxy += d;
        if (minx > maxx || miny > maxy) null = true;
    }
    bool intersects(const Envelope& o) const {
        if (null || o.null) return false;
        return !(o.minx > maxx || o.maxx < minx || o.miny > maxy || o.maxy < miny);
    }
    // Envelope::intersects(a, b): the envelope of segment a-b.
    bool intersects(const Coord& a, const Coord& b) const {
        if (null) return false;
        const double envminx = (a.x < b.x) ? a.x : b.x;
        if (!(maxx >= envminx)) return false;
        const double envmaxx = (a.x > b.x) ? a.x : b.x;
        if (envmaxx < minx) return false;
        const double envminy = (a.y < b.y) ? a.y : b.y;
        if (envminy > maxy) return false;
        const double envmaxy = (a.y > b.y) ? a.y : b.y;
        if (envmaxy < miny) return false;
        return true;
    }
    bool covers(const Envelope& o) const {
        if (null || o.null) return false;
        return o.minx >= minx && o.maxx <= maxx && o.miny >= miny && o.maxy <= maxy;
    }
    bool disjoint(const Envelope& o) const { return !intersects(o); }
    // Envelope::intersection(other, result)
    bool intersection(const Envelope& o, Envelope& result) const {
        if (null || o.null || !intersects(o)) return false;
        result.null = false;
        result.minx = minx > o.minx ? minx : o.minx;
        result.miny = miny > o.miny ? miny : o.miny;
        result.maxx = maxx < o.maxx ? maxx : o.maxx;
        result.maxy = maxy < o.maxy ? maxy : o.maxy;
        return true;
    }
};

Envelope envelopeOf(const Ring& r) {
    Envelope e;
    for (const Coord& c : r) e.expandToInclude(c);
    return e;
}

// Envelope::intersects(p1, p2, q1, q2): the envelopes of two segments.
bool segmentEnvelopesIntersect(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2) {
    double minq = std::min(q1.x, q2.x);
    double maxq = std::max(q1.x, q2.x);
    double minp = std::min(p1.x, p2.x);
    double maxp = std::max(p1.x, p2.x);
    if (minp > maxq) return false;
    if (maxp < minq) return false;
    minq = std::min(q1.y, q2.y);
    maxq = std::max(q1.y, q2.y);
    minp = std::min(p1.y, p2.y);
    maxp = std::max(p1.y, p2.y);
    if (minp > maxq) return false;
    if (maxp < minq) return false;
    return true;
}

// Envelope::intersects(p1, p2, q): q in the envelope of p1-p2.
bool envelopeContains(const Coord& p1, const Coord& p2, const Coord& q) {
    return ((q.x >= (p1.x < p2.x ? p1.x : p2.x)) && (q.x <= (p1.x > p2.x ? p1.x : p2.x))) &&
           ((q.y >= (p1.y < p2.y ? p1.y : p2.y)) && (q.y <= (p1.y > p2.y ? p1.y : p2.y)));
}

// ---------------------------------------------------------------------------
// algorithm/Distance.cpp, LineIntersector

double distance(const Coord& a, const Coord& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double pointToSegment(const Coord& p, const Coord& A, const Coord& B) {
    if (equals2D(A, B)) return distance(p, A);
    const double len2 = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
    const double r = ((p.x - A.x) * (B.x - A.x) + (p.y - A.y) * (B.y - A.y)) / len2;
    if (r <= 0.0) return distance(p, A);
    if (r >= 1.0) return distance(p, B);
    const double s = ((A.y - p.y) * (B.x - A.x) - (A.x - p.x) * (B.y - A.y)) / len2;
    return std::fabs(s) * std::sqrt(len2);
}

const Coord& nearestEndpoint(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2) {
    const Coord* nearestPt = &p1;
    double minDist = pointToSegment(p1, q1, q2);
    double dist = pointToSegment(p2, q1, q2);
    if (dist < minDist) {
        minDist = dist;
        nearestPt = &p2;
    }
    dist = pointToSegment(q1, p1, p2);
    if (dist < minDist) {
        minDist = dist;
        nearestPt = &q1;
    }
    dist = pointToSegment(q2, p1, p2);
    if (dist < minDist) nearestPt = &q2;
    return *nearestPt;
}

constexpr int NO_INTERSECTION = 0;
constexpr int POINT_INTERSECTION = 1;
constexpr int COLLINEAR_INTERSECTION = 2;

// LineIntersector::computeIntersect (segments p1-p2 and q1-q2); the
// intersection points go to intPt. Floating precision model (no rounding).
int computeIntersect(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2, Coord intPt[2]) {
    if (!segmentEnvelopesIntersect(p1, p2, q1, q2)) return NO_INTERSECTION;
    const int Pq1 = orientationIndex(p1, p2, q1);
    const int Pq2 = orientationIndex(p1, p2, q2);
    if ((Pq1 > 0 && Pq2 > 0) || (Pq1 < 0 && Pq2 < 0)) return NO_INTERSECTION;
    const int Qp1 = orientationIndex(q1, q2, p1);
    const int Qp2 = orientationIndex(q1, q2, p2);
    if ((Qp1 > 0 && Qp2 > 0) || (Qp1 < 0 && Qp2 < 0)) return NO_INTERSECTION;

    if (Pq1 == 0 && Pq2 == 0 && Qp1 == 0 && Qp2 == 0) {
        // computeCollinearIntersection
        const bool q1inP = envelopeContains(p1, p2, q1);
        const bool q2inP = envelopeContains(p1, p2, q2);
        const bool p1inQ = envelopeContains(q1, q2, p1);
        const bool p2inQ = envelopeContains(q1, q2, p2);
        if (q1inP && q2inP) {
            intPt[0] = q1;
            intPt[1] = q2;
            return COLLINEAR_INTERSECTION;
        }
        if (p1inQ && p2inQ) {
            intPt[0] = p1;
            intPt[1] = p2;
            return COLLINEAR_INTERSECTION;
        }
        if (q1inP && p1inQ) {
            intPt[0] = q1;
            intPt[1] = p1;
            return equals2D(q1, p1) && !q2inP && !p2inQ ? POINT_INTERSECTION : COLLINEAR_INTERSECTION;
        }
        if (q1inP && p2inQ) {
            intPt[0] = q1;
            intPt[1] = p2;
            return equals2D(q1, p2) && !q2inP && !p1inQ ? POINT_INTERSECTION : COLLINEAR_INTERSECTION;
        }
        if (q2inP && p1inQ) {
            intPt[0] = q2;
            intPt[1] = p1;
            return equals2D(q2, p1) && !q1inP && !p2inQ ? POINT_INTERSECTION : COLLINEAR_INTERSECTION;
        }
        if (q2inP && p2inQ) {
            intPt[0] = q2;
            intPt[1] = p2;
            return equals2D(q2, p2) && !q1inP && !p1inQ ? POINT_INTERSECTION : COLLINEAR_INTERSECTION;
        }
        return NO_INTERSECTION;
    }

    Coord p;
    if (Pq1 == 0 || Pq2 == 0 || Qp1 == 0 || Qp2 == 0) {
        // the intersection is an endpoint: copy it (exact)
        if (equals2D(p1, q1)) p = p1;
        else if (equals2D(p1, q2)) p = p1;
        else if (equals2D(p2, q1)) p = p2;
        else if (equals2D(p2, q2)) p = p2;
        else if (Pq1 == 0) p = q1;
        else if (Pq2 == 0) p = q2;
        else if (Qp1 == 0) p = p1;
        else p = p2;  // Qp2 == 0
    } else {
        // LineIntersector::intersection -> intersectionSafe
        Coord ptInt;
        if (!intersectionDD(p1, p2, q1, q2, ptInt)) ptInt = nearestEndpoint(p1, p2, q1, q2);
        // isInSegmentEnvelopes
        const bool inEnv = envelopeContains(p1, p2, ptInt) && envelopeContains(q1, q2, ptInt);
        if (!inEnv) ptInt = nearestEndpoint(p1, p2, q1, q2);
        p = ptInt;
    }
    intPt[0] = p;
    return POINT_INTERSECTION;
}

// ---------------------------------------------------------------------------
// noding: Octant, SegmentPointComparator, SegmentNode, SegmentNodeList,
// NodedSegmentString, IntersectionAdder

int octant(double dx, double dy) {
    if (dx == 0.0 && dy == 0.0) unsupported("octant of a zero-length segment");
    const double adx = std::fabs(dx);
    const double ady = std::fabs(dy);
    if (dx >= 0) {
        if (dy >= 0) return adx >= ady ? 0 : 1;
        return adx >= ady ? 7 : 6;
    }
    if (dy >= 0) return adx >= ady ? 3 : 2;
    return adx >= ady ? 4 : 5;
}

int relativeSign(double x0, double x1) {
    if (x0 < x1) return -1;
    if (x0 > x1) return 1;
    return 0;
}

int compareValue(int s0, int s1) {
    if (s0 < 0) return -1;
    if (s0 > 0) return 1;
    if (s1 < 0) return -1;
    if (s1 > 0) return 1;
    return 0;
}

int segmentPointCompare(int oct, const Coord& p0, const Coord& p1) {
    if (equals2D(p0, p1)) return 0;
    const int xSign = relativeSign(p0.x, p1.x);
    const int ySign = relativeSign(p0.y, p1.y);
    switch (oct) {
        case 0: return compareValue(xSign, ySign);
        case 1: return compareValue(ySign, xSign);
        case 2: return compareValue(ySign, -xSign);
        case 3: return compareValue(-xSign, ySign);
        case 4: return compareValue(-xSign, -ySign);
        case 5: return compareValue(-ySign, -xSign);
        case 6: return compareValue(-ySign, xSign);
        case 7: return compareValue(xSign, -ySign);
        default: return 0;
    }
}

struct SegmentNode {
    Coord coord;
    size_t segmentIndex;
    int segmentOctant;
    bool isInterior;

    int compareTo(const SegmentNode& other) const {
        if (segmentIndex < other.segmentIndex) return -1;
        if (segmentIndex > other.segmentIndex) return 1;
        if (equals2D(coord, other.coord)) return 0;
        if (!isInterior) return -1;
        if (!other.isInterior) return 1;
        return segmentPointCompare(segmentOctant, coord, other.coord);
    }
};

struct SegmentString {
    std::vector<Coord> pts;
    int geomIndex;  // 0 or 1: the input polygon
    std::vector<SegmentNode> nodes;
    bool ready = false;

    bool isClosed() const { return equals2D(pts.front(), pts.back()); }

    // SegmentString::getSegmentOctant (safeOctant)
    int segmentOctant(size_t index) const {
        if (index >= pts.size() - 1) return -1;
        const Coord& p0 = pts[index];
        const Coord& p1 = pts[index + 1];
        if (equals2D(p0, p1)) return 0;
        return octant(p1.x - p0.x, p1.y - p0.y);
    }

    void addNode(const Coord& c, size_t segmentIndex) {
        nodes.push_back(SegmentNode{c, segmentIndex, segmentOctant(segmentIndex),
                                    !equals2D(c, pts[segmentIndex])});
        ready = false;
    }

    void prepare() {
        if (ready) return;
        std::sort(nodes.begin(), nodes.end(),
                  [](const SegmentNode& a, const SegmentNode& b) { return a.compareTo(b) < 0; });
        nodes.erase(std::unique(nodes.begin(), nodes.end(),
                                [](const SegmentNode& a, const SegmentNode& b) { return a.compareTo(b) == 0; }),
                    nodes.end());
        ready = true;
    }

    // NodedSegmentString::addIntersection: a point on a vertex is normalised
    // to the higher segment index.
    void addIntersection(const Coord& intPt, size_t segmentIndex) {
        if (segmentIndex > pts.size() - 2) unsupported("segment index out of range");
        size_t normalizedSegmentIndex = segmentIndex;
        const size_t nextSegIndex = normalizedSegmentIndex + 1;
        if (nextSegIndex < pts.size() && equals2D(intPt, pts[nextSegIndex])) normalizedSegmentIndex = nextSegIndex;
        addNode(intPt, normalizedSegmentIndex);
    }
};

// SegmentNodeList::findCollapseIndex
bool findCollapseIndex(const SegmentNode& ei0, const SegmentNode& ei1, size_t& collapsedVertexIndex) {
    if (!equals2D(ei0.coord, ei1.coord)) return false;
    size_t numVerticesBetween = ei1.segmentIndex - ei0.segmentIndex;
    if (!ei1.isInterior) numVerticesBetween--;
    if (numVerticesBetween == 1) {
        collapsedVertexIndex = ei0.segmentIndex + 1;
        return true;
    }
    return false;
}

// SegmentNodeList::addSplitEdges (with addEndpoints, addCollapsedNodes and
// createSplitEdgePts): the noded substrings of `ss`, in order.
void addSplitEdges(SegmentString& ss, std::vector<SegmentString>& out) {
    const size_t maxSegIndex = ss.pts.size() - 1;
    ss.addNode(ss.pts[0], 0);
    ss.addNode(ss.pts[maxSegIndex], maxSegIndex);

    std::vector<size_t> collapsed;
    ss.prepare();
    for (size_t i = 1; i < ss.nodes.size(); ++i) {
        size_t idx;
        if (findCollapseIndex(ss.nodes[i - 1], ss.nodes[i], idx)) collapsed.push_back(idx);
    }
    if (ss.pts.size() >= 2) {
        for (size_t i = 0, n = ss.pts.size() - 2; i < n; ++i)
            if (equals2D(ss.pts[i], ss.pts[i + 2])) collapsed.push_back(i + 1);
    }
    for (size_t v : collapsed) ss.addNode(ss.pts[v], v);
    ss.prepare();

    const SegmentNode* eiPrev = &ss.nodes[0];
    for (size_t k = 1; k < ss.nodes.size(); ++k) {
        const SegmentNode* ei = &ss.nodes[k];
        if (!ei->compareTo(*eiPrev)) continue;
        SegmentString e;
        e.geomIndex = ss.geomIndex;
        if (ei->segmentIndex == eiPrev->segmentIndex) {
            e.pts = {eiPrev->coord, ei->coord};
        } else {
            const Coord& lastSegStartPt = ss.pts[ei->segmentIndex];
            const bool useIntPt1 = ei->isInterior || !equals2D(ei->coord, lastSegStartPt);
            e.pts.push_back(eiPrev->coord);
            for (size_t i = eiPrev->segmentIndex + 1; i <= ei->segmentIndex; ++i) e.pts.push_back(ss.pts[i]);
            if (useIntPt1) e.pts.push_back(ei->coord);
        }
        out.push_back(std::move(e));
        eiPrev = ei;
    }
}

// IntersectionAdder::processIntersections for segment i0 of e0 and i1 of e1.
void processIntersections(SegmentString& e0, size_t i0, SegmentString& e1, size_t i1) {
    if (&e0 == &e1 && i0 == i1) return;
    Coord intPt[2];
    const int n = computeIntersect(e0.pts[i0], e0.pts[i0 + 1], e1.pts[i1], e1.pts[i1 + 1], intPt);
    if (n == NO_INTERSECTION) return;
    // isTrivialIntersection (GEOS compares with size() - 1, so the closing
    // pair of segments of a ring is not trivial; harmless, kept as is)
    bool trivial = false;
    if (&e0 == &e1 && n == 1) {
        const size_t d = i0 > i1 ? i0 - i1 : i1 - i0;
        if (d == 1) trivial = true;
        else if (e0.isClosed()) {
            const size_t maxSegIndex = e0.pts.size() - 1;
            if ((i0 == 0 && i1 == maxSegIndex) || (i1 == 0 && i0 == maxSegIndex)) trivial = true;
        }
    }
    if (trivial) return;
    if (n == COLLINEAR_INTERSECTION) unsupported("collinear (overlapping) segments");
    e0.addIntersection(intPt[0], i0);
    e1.addIntersection(intPt[0], i1);
}

// ---------------------------------------------------------------------------
// operation/overlayng: RingClipper, OverlayUtil::clippingEnvelope,
// RobustClipEnvelopeComputer

constexpr int BOX_BOTTOM = 0, BOX_RIGHT = 1, BOX_TOP = 2, BOX_LEFT = 3;

struct RingClipper {
    Envelope clipEnv;

    bool isInsideEdge(const Coord& p, int edgeIndex) const {
        if (clipEnv.null) return false;
        switch (edgeIndex) {
            case BOX_BOTTOM: return p.y > clipEnv.miny;
            case BOX_RIGHT: return p.x < clipEnv.maxx;
            case BOX_TOP: return p.y < clipEnv.maxy;
            default: return p.x > clipEnv.minx;
        }
    }
    static double intersectionLineY(const Coord& a, const Coord& b, double y) {
        const double m = (b.x - a.x) / (b.y - a.y);
        const double intercept = (y - a.y) * m;
        return a.x + intercept;
    }
    static double intersectionLineX(const Coord& a, const Coord& b, double x) {
        const double m = (b.y - a.y) / (b.x - a.x);
        const double intercept = (x - a.x) * m;
        return a.y + intercept;
    }
    Coord intersection(const Coord& a, const Coord& b, int edgeIndex) const {
        switch (edgeIndex) {
            case BOX_BOTTOM: return Coord{intersectionLineY(a, b, clipEnv.miny), clipEnv.miny};
            case BOX_RIGHT: return Coord{clipEnv.maxx, intersectionLineX(a, b, clipEnv.maxx)};
            case BOX_TOP: return Coord{intersectionLineY(a, b, clipEnv.maxy), clipEnv.maxy};
            default: return Coord{clipEnv.minx, intersectionLineX(a, b, clipEnv.minx)};
        }
    }
    static void addNoRepeat(std::vector<Coord>& v, const Coord& c) {
        if (!v.empty() && equals2D(v.back(), c)) return;
        v.push_back(c);
    }
    std::vector<Coord> clipToBoxEdge(const std::vector<Coord>& pts, int edgeIndex, bool closeRing) const {
        std::vector<Coord> ptsClip;
        Coord p0 = pts.back();
        for (const Coord& p1 : pts) {
            if (isInsideEdge(p1, edgeIndex)) {
                if (!isInsideEdge(p0, edgeIndex)) addNoRepeat(ptsClip, intersection(p0, p1, edgeIndex));
                addNoRepeat(ptsClip, p1);
            } else if (isInsideEdge(p0, edgeIndex)) {
                addNoRepeat(ptsClip, intersection(p0, p1, edgeIndex));
            }
            p0 = p1;
        }
        if (closeRing && !ptsClip.empty() && !equals2D(ptsClip.front(), ptsClip.back()))
            ptsClip.push_back(ptsClip.front());
        return ptsClip;
    }
    std::vector<Coord> clip(const std::vector<Coord>& cs) const {
        std::vector<Coord> pts = cs;
        for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex) {
            pts = clipToBoxEdge(pts, edgeIndex, edgeIndex == 3);
            if (pts.empty()) return pts;
        }
        return pts;
    }
};

// OverlayUtil::safeEnv for a floating precision model.
Envelope safeEnv(const Envelope& env) {
    double minSize = std::min(env.height(), env.width());
    if (minSize <= 0.0) minSize = std::max(env.height(), env.width());
    Envelope r = env;
    r.expandBy(0.1 * minSize);  // SAFE_ENV_BUFFER_FACTOR
    return r;
}

// OverlayUtil::clippingEnvelope for INTERSECTION (UNION does not clip).
bool clippingEnvelope(const Ring& a, const Ring& b, Envelope& clipEnv) {
    Envelope target;
    if (!safeEnv(envelopeOf(a)).intersection(safeEnv(envelopeOf(b)), target)) target = Envelope();
    // RobustClipEnvelopeComputer: expand by the segments meeting the target
    Envelope rc = target;
    for (const Ring* r : {&a, &b}) {
        for (size_t i = 1; i < r->size(); ++i) {
            if (target.intersects((*r)[i - 1], (*r)[i])) {
                rc.expandToInclude((*r)[i - 1]);
                rc.expandToInclude((*r)[i]);
            }
        }
    }
    clipEnv = safeEnv(rc);
    return true;
}

// ---------------------------------------------------------------------------
// point location (algorithm/RayCrossingCounter), for the edge labels

enum Location { INTERIOR, BOUNDARY, EXTERIOR };

Location locatePointInRing(const Coord& p, const Ring& ring) {
    int crossingCount = 0;
    for (size_t i = 1; i < ring.size(); ++i) {
        const Coord& p1 = ring[i - 1];
        const Coord& p2 = ring[i];
        if (p1.x < p.x && p2.x < p.x) continue;
        if (p.x == p2.x && p.y == p2.y) return BOUNDARY;
        if (p1.y == p.y && p2.y == p.y) {
            double minx = p1.x, maxx = p2.x;
            if (minx > maxx) std::swap(minx, maxx);
            if (p.x >= minx && p.x <= maxx) return BOUNDARY;
            continue;
        }
        if (((p1.y > p.y) && (p2.y <= p.y)) || ((p2.y > p.y) && (p1.y <= p.y))) {
            int orient = orientationIndex(p1, p2, p);
            if (orient == COLLINEAR) return BOUNDARY;
            if (p2.y < p1.y) orient = -orient;
            if (orient == COUNTERCLOCKWISE) crossingCount++;
        }
    }
    return (crossingCount % 2) == 1 ? INTERIOR : EXTERIOR;
}

// ---------------------------------------------------------------------------
// OverlayNG graph and ring building

struct HalfEdge {
    const std::vector<Coord>* pts;
    bool forward;
    bool inResult = false;
    bool visited = false;
    size_t next = 0;
    const Coord& orig() const { return forward ? pts->front() : pts->back(); }
    const Coord& dest() const { return forward ? pts->back() : pts->front(); }
};

// OverlayEdge::addCoordinates. GEOS names the flag isFirstEdge but sets it
// when the list is NOT empty, so the first edge of a ring skips its origin
// and the ring starts at that edge's second point.
void addCoordinates(const HalfEdge& e, std::vector<Coord>& coords) {
    const bool isFirstEdge = !coords.empty();
    const std::vector<Coord>& pts = *e.pts;
    auto addNoRepeat = [&coords](const Coord& c) {
        if (!coords.empty() && equals2D(coords.back(), c)) return;
        coords.push_back(c);
    };
    if (e.forward) {
        const size_t startIndex = isFirstEdge ? 0 : 1;
        for (size_t i = startIndex; i < pts.size(); ++i) addNoRepeat(pts[i]);
    } else {
        const long long startIndex = static_cast<long long>(pts.size()) - (isFirstEdge ? 1 : 2);
        for (long long i = startIndex; i >= 0; --i) addNoRepeat(pts[static_cast<size_t>(i)]);
    }
}

bool isResultOfOp(OverlayOp op, Location loc0, Location loc1) {
    if (op == OverlayOp::Intersection) return loc0 == INTERIOR && loc1 == INTERIOR;
    return loc0 == INTERIOR || loc1 == INTERIOR;
}

}  // namespace

bool is_ccw(const Ring& ring) {
    const int inPts = static_cast<int>(ring.size()) - 1;
    if (inPts < 3) return false;
    const unsigned nPts = static_cast<unsigned>(inPts);
    static const Coord nullCoord{std::nan(""), std::nan("")};
    const Coord* upHiPt = &ring[0];
    const Coord* upLowPt = &nullCoord;
    double prevY = upHiPt->y;
    unsigned iUpHi = 0;
    for (unsigned i = 1; i <= nPts; ++i) {
        const double py = ring[i].y;
        if (py > prevY && py >= upHiPt->y) {
            iUpHi = i;
            upHiPt = &ring[i];
            upLowPt = &ring[i - 1];
        }
        prevY = py;
    }
    if (iUpHi == 0) return false;
    unsigned iDownLow = iUpHi;
    do {
        iDownLow = (iDownLow + 1) % nPts;
    } while (iDownLow != iUpHi && ring[iDownLow].y == upHiPt->y);
    const Coord& downLowPt = ring[iDownLow];
    const unsigned iDownHi = iDownLow > 0 ? iDownLow - 1 : nPts - 1;
    const Coord& downHiPt = ring[iDownHi];
    if (equals2D(*upHiPt, downHiPt)) {
        if (equals2D(*upLowPt, *upHiPt) || equals2D(downLowPt, *upHiPt) || equals2D(*upLowPt, downLowPt))
            return false;
        return orientationIndex(*upLowPt, *upHiPt, downLowPt) == COUNTERCLOCKWISE;
    }
    return downHiPt.x - upHiPt->x < 0;
}

std::vector<Ring> overlay(const Ring& a, const Ring& b, OverlayOp op) {
    const Ring* geoms[2] = {&a, &b};
    for (const Ring* g : geoms)
        if (g->size() < 4 || !equals2D(g->front(), g->back())) unsupported("input ring not closed");

    // OverlayUtil::isEmptyResult (intersection of disjoint envelopes)
    if (op == OverlayOp::Intersection && envelopeOf(a).disjoint(envelopeOf(b))) return {};

    // EdgeNodingBuilder: clip (intersection only) or remove repeated points
    Envelope clipEnv;
    const bool clipping = op == OverlayOp::Intersection && clippingEnvelope(a, b, clipEnv);
    RingClipper clipper{clipEnv};
    std::vector<SegmentString> input;
    for (int gi = 0; gi < 2; ++gi) {
        const Ring& ring = *geoms[gi];
        const Envelope env = envelopeOf(ring);
        if (clipping && clipEnv.disjoint(env)) continue;  // isClippedCompletely
        SegmentString ss;
        ss.geomIndex = gi;
        if (!clipping || clipEnv.covers(env)) {
            for (const Coord& c : ring)  // RepeatedPointRemover
                if (ss.pts.empty() || !equals2D(ss.pts.back(), c)) ss.pts.push_back(c);
        } else {
            ss.pts = clipper.clip(ring);
        }
        if (ss.pts.size() < 2) continue;
        input.push_back(std::move(ss));
    }

    // MCIndexNoder + IntersectionAdder: every pair of segments (the monotone
    // chain index only skips pairs that cannot intersect)
    for (size_t s0 = 0; s0 < input.size(); ++s0) {
        for (size_t s1 = s0; s1 < input.size(); ++s1) {
            SegmentString& e0 = input[s0];
            SegmentString& e1 = input[s1];
            for (size_t i0 = 0; i0 + 1 < e0.pts.size(); ++i0) {
                for (size_t i1 = (s0 == s1 ? i0 + 1 : 0); i1 + 1 < e1.pts.size(); ++i1)
                    processIntersections(e0, i0, e1, i1);
            }
        }
    }

    // noded substrings -> edges (collapsed ones dropped)
    std::vector<SegmentString> split;
    for (SegmentString& ss : input) addSplitEdges(ss, split);
    std::vector<SegmentString> edges;
    for (SegmentString& e : split) {
        const std::vector<Coord>& p = e.pts;
        const size_t sz = p.size();
        if (sz < 2 || equals2D(p[0], p[1]) || (sz > 2 && equals2D(p[sz - 1], p[sz - 2]))) continue;
        edges.push_back(std::move(e));
    }
    // EdgeMerger: identical edges would be merged; not supported here
    for (size_t i = 0; i < edges.size(); ++i) {
        for (size_t j = i + 1; j < edges.size(); ++j) {
            const auto& p = edges[i].pts;
            const auto& q = edges[j].pts;
            if (p.size() != q.size()) continue;
            bool same = true, rev = true;
            for (size_t k = 0; k < p.size(); ++k) {
                same = same && equals2D(p[k], q[k]);
                rev = rev && equals2D(p[k], q[p.size() - 1 - k]);
            }
            if (same || rev) unsupported("coincident edges");
        }
    }

    // labels: each input shell has its interior on the right of its forward
    // edges if it is clockwise (EdgeNodingBuilder::computeDepthDelta); an
    // edge lies inside or outside the other polygon as a whole.
    const bool cw[2] = {!is_ccw(a), !is_ccw(b)};
    std::vector<HalfEdge> he;
    he.reserve(2 * edges.size());
    for (const SegmentString& e : edges) {
        const int own = e.geomIndex, other = 1 - own;
        const Coord mid{(e.pts[0].x + e.pts[1].x) / 2, (e.pts[0].y + e.pts[1].y) / 2};
        const Location locOther = locatePointInRing(mid, *geoms[other]);
        if (locOther == BOUNDARY) unsupported("edge on the other polygon's boundary");
        for (int dir = 0; dir < 2; ++dir) {
            HalfEdge h;
            h.pts = &e.pts;
            h.forward = dir == 0;
            Location loc[2];
            loc[own] = (h.forward == cw[own]) ? INTERIOR : EXTERIOR;  // right side
            loc[other] = locOther;
            h.inResult = isResultOfOp(op, loc[0], loc[1]);
            he.push_back(h);
        }
    }
    // OverlayLabeller::unmarkDuplicateEdgesFromResultArea
    for (size_t i = 0; i < he.size(); i += 2) {
        if (he[i].inResult && he[i + 1].inResult) he[i].inResult = he[i + 1].inResult = false;
    }

    // MaximalEdgeRing::linkResultAreaMaxRingAtNode: with one result edge in
    // and one out at every node the link is unique
    for (size_t i = 0; i < he.size(); ++i) {
        if (!he[i].inResult) continue;
        size_t found = he.size();
        for (size_t j = 0; j < he.size(); ++j) {
            if (!he[j].inResult || !equals2D(he[j].orig(), he[i].dest())) continue;
            if (found != he.size()) unsupported("node with several result edges");
            found = j;
        }
        if (found == he.size()) unsupported("no outgoing result edge");
        he[i].next = found;
    }

    // PolygonBuilder: rings in the order of their first result edge
    std::vector<Ring> shells;
    for (size_t i = 0; i < he.size(); ++i) {
        if (!he[i].inResult || he[i].visited) continue;
        Ring pts;
        size_t e = i;
        do {
            if (he[e].visited) unsupported("edge visited twice");
            addCoordinates(he[e], pts);
            he[e].visited = true;
            e = he[e].next;
        } while (e != i);
        pts.push_back(pts.front());  // OverlayEdgeRing::closeRing
        if (is_ccw(pts)) unsupported("result hole");
        shells.push_back(std::move(pts));
    }
    return shells;
}

std::vector<Ring> unary_union(const std::vector<Ring>& polys) {
    if (polys.empty()) return {};
    if (polys.size() == 1) return polys;  // unionSafe(g0, nullptr) returns a copy
    if (polys.size() > 2) unsupported("unary union of more than two polygons");
    // TemplateSTRtree (node capacity 10): one slice; nodes sorted by
    // minX + maxX, then by minY + maxY (std::sort of two elements keeps the
    // order of equal keys)
    size_t order[2] = {0, 1};
    const Envelope e0 = envelopeOf(polys[0]), e1 = envelopeOf(polys[1]);
    const double x0 = e0.minx + e0.maxx, x1 = e1.minx + e1.maxx;
    const double y0 = e0.miny + e0.maxy, y1 = e1.miny + e1.maxy;
    if (x1 < x0) std::swap(order[0], order[1]);
    const double ya = order[0] == 0 ? y0 : y1, yb = order[0] == 0 ? y1 : y0;
    if (yb < ya) std::swap(order[0], order[1]);
    return overlay(polys[order[0]], polys[order[1]], OverlayOp::Union);
}

}  // namespace geos_lite
}  // namespace sdm
