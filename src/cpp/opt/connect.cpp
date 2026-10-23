// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/connect.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "opt/maxflow.hpp"

namespace opt {

namespace {

constexpr double kBig = 1.0e9;  // an unbounded capacity: every flow here is at most a few units

void push_cut(Cuts& out, const std::vector<int>& index, const std::vector<double>& value, double upper) {
    out.index.insert(out.index.end(), index.begin(), index.end());
    out.value.insert(out.value.end(), value.begin(), value.end());
    out.upper.push_back(upper);
    out.start.push_back(static_cast<int>(out.index.size()));
}

double at(const std::vector<double>& x, i64 j) { return std::clamp(x[static_cast<std::size_t>(j)], 0.0, 1.0); }

}  // namespace

void separate_connected(const Connected& c, const std::vector<double>& x, const SeparateOptions& o, Cuts& out) {
    const int n = static_cast<int>(c.nodes.size());
    const int na = static_cast<int>(c.tail.size());
    if (n == 0 || na % 2 != 0) {
        if (na % 2 != 0) throw std::invalid_argument("the arcs of a patch must come in pairs");
        return;
    }
    const int ne = na / 2;
    std::vector<double> xv(n), rv(n), yv(na);
    for (int v = 0; v < n; ++v) {
        xv[v] = at(x, c.nodes[v]);
        rv[v] = at(x, c.root + v);
    }
    for (int a = 0; a < na; ++a) yv[a] = at(x, c.arc + a);

    std::vector<int> order;
    for (int v = 0; v < n; ++v)
        if (xv[v] >= o.min_value) order.push_back(v);
    if (order.empty()) return;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return xv[a] > xv[b]; });
    if (static_cast<int>(order.size()) > o.max_candidates) order.resize(o.max_candidates);

    // edge e: arc e (tail -> head) one way, arc e + ne the other
    std::vector<int> ei(ne), ej(ne);
    std::vector<double> cij(ne), cji(ne);
    for (int e = 0; e < ne; ++e) {
        if (c.tail[e + ne] != c.head[e] || c.head[e + ne] != c.tail[e])
            throw std::invalid_argument("the second half of a patch's arcs must reverse the first");
        ei[e] = c.tail[e];
        ej[e] = c.head[e];
        cij[e] = yv[e];
        cji[e] = yv[e + ne];
    }
    MaxFlow g(n, ei, ej, cij, cji);
    std::vector<double> sink(n, 0.0);
    std::vector<char> inside(n), covered(n, 0);
    std::vector<int> idx;
    std::vector<double> val;
    for (const int k : order) {
        if (covered[k]) continue;  // inside the W of a cut found already: most likely the same cut
        sink[k] = kBig;
        const double flow = g.solve(rv, sink);
        sink[k] = 0.0;
        if (flow >= xv[k] - o.min_violation) continue;
        // W: the sink's side of the minimum cut nearest k (the nodes that reach k in the residual graph): the
        // sparsest cut
        idx.clear();
        val.clear();
        for (int v = 0; v < n; ++v) {
            inside[v] = g.sink_side(v) || v == k;
            if (inside[v]) {
                covered[v] = 1;
                idx.push_back(static_cast<int>(c.root + v));
                val.push_back(-1.0);
            }
        }
        for (int a = 0; a < na; ++a)
            if (inside[c.head[a]] && !inside[c.tail[a]]) {
                idx.push_back(static_cast<int>(c.arc + a));
                val.push_back(-1.0);
            }
        idx.push_back(static_cast<int>(c.nodes[k]));
        val.push_back(1.0);
        push_cut(out, idx, val, 0.0);
    }
}

void separate_corridor(const Corridor& c, const std::vector<double>& x, const SeparateOptions& o, Cuts& out) {
    const int n = c.n;
    if (n == 0 || c.sinks.empty()) return;
    // node v: in = v, out = n + v; through a unit at its selection, the terminals unbounded; the edges both ways
    // from out to in, unbounded
    std::vector<char> terminal(n, 0);
    terminal[c.source] = 1;
    for (const int t : c.sinks) terminal[t] = 1;
    std::vector<int> ei, ej;
    std::vector<double> cij, cji;
    const std::size_t ne = c.tail.size();
    ei.reserve(n + 2 * ne);
    ej.reserve(n + 2 * ne);
    cij.reserve(n + 2 * ne);
    cji.reserve(n + 2 * ne);
    for (int v = 0; v < n; ++v) {
        ei.push_back(v);
        ej.push_back(n + v);
        cij.push_back(terminal[v] ? kBig : at(x, v));
        cji.push_back(0.0);
    }
    for (std::size_t e = 0; e < ne; ++e) {
        const int u = c.tail[e], v = c.head[e];
        ei.push_back(n + u);
        ej.push_back(v);
        cij.push_back(kBig);
        cji.push_back(0.0);
        ei.push_back(n + v);
        ej.push_back(u);
        cij.push_back(kBig);
        cji.push_back(0.0);
    }
    MaxFlow g(2 * n, ei, ej, cij, cji);
    std::vector<double> source(2 * n, 0.0), sink(2 * n, 0.0);
    source[n + c.source] = kBig;
    std::vector<int> idx;
    std::vector<double> val;
    for (const int t : c.sinks) {
        sink[t] = kBig;
        const double flow = g.solve(source, sink);
        sink[t] = 0.0;
        if (flow >= 1.0 - o.min_violation) continue;
        idx.clear();
        val.clear();
        for (int v = 0; v < n; ++v)
            if (g.source_side(v) && !g.source_side(n + v)) {
                idx.push_back(v);
                val.push_back(-1.0);
            }
        if (!idx.empty()) push_cut(out, idx, val, -1.0);
    }
}

Separator connect_separator(std::vector<Connected> connected, std::vector<Corridor> corridors, SeparateOptions options) {
    return [connected = std::move(connected), corridors = std::move(corridors), options](
               const std::vector<double>& x, std::vector<int>& start, std::vector<int>& index,
               std::vector<double>& value, std::vector<double>& upper) {
        Cuts cuts;
        for (const auto& c : connected) separate_connected(c, x, options, cuts);
        for (const auto& c : corridors) separate_corridor(c, x, options, cuts);
        const int base = start.empty() ? 0 : start.back();
        if (start.empty()) start.push_back(0);
        for (std::size_t k = 1; k < cuts.start.size(); ++k) start.push_back(base + cuts.start[k]);
        index.insert(index.end(), cuts.index.begin(), cuts.index.end());
        value.insert(value.end(), cuts.value.begin(), cuts.value.end());
        upper.insert(upper.end(), cuts.upper.begin(), cuts.upper.end());
    };
}

}  // namespace opt
