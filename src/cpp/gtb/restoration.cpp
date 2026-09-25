// SPDX-License-Identifier: GPL-3.0-or-later
#include "restoration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

namespace gtb {

namespace {

// Fast Disjoint Set Union (DSU) with path compression
struct DisjointSet {
    std::vector<i64> parent;

    explicit DisjointSet(std::size_t n) : parent(n) {
        for (std::size_t i = 0; i < n; ++i) {
            parent[i] = static_cast<i64>(i);
        }
    }

    i64 find(i64 i) {
        while (parent[static_cast<std::size_t>(i)] != i) {
            parent[static_cast<std::size_t>(i)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(i)])];
            i = parent[static_cast<std::size_t>(i)];
        }
        return i;
    }

    void unite(i64 a, i64 b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            if (a < b) {
                parent[static_cast<std::size_t>(b)] = a;
            } else {
                parent[static_cast<std::size_t>(a)] = b;
            }
        }
    }
};

}  // namespace

RestorationStatusResult compute_restoration_status(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 total_pixels = im.size();

    i64 missing_pixels = 0;
    i64 fg_pixels = 0;

    for (i64 i = 0; i < total_pixels; ++i) {
        const u8 v = im.px[static_cast<std::size_t>(i)];
        if (v == 0) ++missing_pixels;
        else if (v == 2) ++fg_pixels;
    }

    RestorationStatusResult result;
    result.data_area = total_pixels - missing_pixels;

    if (fg_pixels == 0) {
        result.rest_pot = 100.0;
        return result;
    }

    // Two-pass Connected Component Labeling (8-connectivity) for foreground (v == 2)
    // Map pixel (x, y) to a temporary label
    std::vector<i64> labels(static_cast<std::size_t>(total_pixels), 0);
    // Overestimate number of labels to size DSU
    DisjointSet dsu(static_cast<std::size_t>(fg_pixels + 1));
    i64 next_label = 1;

    for (i64 y = 0; y < ny; ++y) {
        const i64 row_off = y * nx;
        const i64 prev_row_off = (y > 0) ? (y - 1) * nx : 0;

        for (i64 x = 0; x < nx; ++x) {
            const i64 idx = row_off + x;
            if (im.px[static_cast<std::size_t>(idx)] != 2) continue;

            i64 min_root = 0;

            // Check 8-connected already-visited neighbors:
            // (x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y)
            if (y > 0) {
                if (x > 0) {
                    const i64 n_idx = prev_row_off + (x - 1);
                    if (labels[static_cast<std::size_t>(n_idx)] > 0) {
                        const i64 r = dsu.find(labels[static_cast<std::size_t>(n_idx)]);
                        min_root = (min_root == 0) ? r : std::min(min_root, r);
                    }
                }
                {
                    const i64 n_idx = prev_row_off + x;
                    if (labels[static_cast<std::size_t>(n_idx)] > 0) {
                        const i64 r = dsu.find(labels[static_cast<std::size_t>(n_idx)]);
                        min_root = (min_root == 0) ? r : std::min(min_root, r);
                    }
                }
                if (x + 1 < nx) {
                    const i64 n_idx = prev_row_off + (x + 1);
                    if (labels[static_cast<std::size_t>(n_idx)] > 0) {
                        const i64 r = dsu.find(labels[static_cast<std::size_t>(n_idx)]);
                        min_root = (min_root == 0) ? r : std::min(min_root, r);
                    }
                }
            }
            if (x > 0) {
                const i64 n_idx = row_off + (x - 1);
                if (labels[static_cast<std::size_t>(n_idx)] > 0) {
                    const i64 r = dsu.find(labels[static_cast<std::size_t>(n_idx)]);
                    min_root = (min_root == 0) ? r : std::min(min_root, r);
                }
            }

            if (min_root == 0) {
                const i64 lbl = next_label++;
                labels[static_cast<std::size_t>(idx)] = lbl;
            } else {
                labels[static_cast<std::size_t>(idx)] = min_root;
                // Unite all neighbors to min_root
                if (y > 0) {
                    if (x > 0 && labels[static_cast<std::size_t>(prev_row_off + (x - 1))] > 0) {
                        dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + (x - 1))]);
                    }
                    if (labels[static_cast<std::size_t>(prev_row_off + x)] > 0) {
                        dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + x)]);
                    }
                    if (x + 1 < nx && labels[static_cast<std::size_t>(prev_row_off + (x + 1))] > 0) {
                        dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + (x + 1))]);
                    }
                }
                if (x > 0 && labels[static_cast<std::size_t>(row_off + (x - 1))] > 0) {
                    dsu.unite(min_root, labels[static_cast<std::size_t>(row_off + (x - 1))]);
                }
            }
        }
    }

    // Pass 2: accumulate areas per root
    std::vector<i64> component_areas(static_cast<std::size_t>(next_label), 0);
    for (i64 i = 0; i < total_pixels; ++i) {
        if (labels[static_cast<std::size_t>(i)] > 0) {
            const i64 root = dsu.find(labels[static_cast<std::size_t>(i)]);
            component_areas[static_cast<std::size_t>(root)]++;
        }
    }

    // Gather statistics
    i64 nr_comp = 0;
    i64 eca_max = 0;
    i64 larg_obj = 0;
    double pcnum = 0.0;

    for (std::size_t i = 1; i < component_areas.size(); ++i) {
        const i64 area = component_areas[i];
        if (area > 0) {
            ++nr_comp;
            eca_max += area;
            if (area > larg_obj) larg_obj = area;
            const double a = static_cast<double>(area);
            pcnum += a * a;
        }
    }

    result.nr_comp = nr_comp;
    result.eca_max = eca_max;
    result.larg_obj = larg_obj;

    const double eca = std::sqrt(pcnum);
    result.eca = eca;

    if (eca_max > 0) {
        result.doc = (eca / static_cast<double>(eca_max)) * 100.0;
        result.rest_pot = 100.0 - result.doc;
        result.aps = static_cast<double>(eca_max) / static_cast<double>(nr_comp);
    } else {
        result.doc = 0.0;
        result.rest_pot = 100.0;
        result.aps = 0.0;
    }

    if (result.data_area > 0) {
        result.rac = (static_cast<double>(eca_max) / static_cast<double>(result.data_area)) * 100.0;
    } else {
        result.rac = 0.0;
    }

    // CNOA = floor((2.0 * b * c^2)/(b^2 - c^2) + 1.0)
    const double b = static_cast<double>(eca_max);
    const double c = eca;
    const double denom = (b * b) - (c * c);
    if (denom > 1e-9) {
        result.cnoa = std::floor(((2.0 * b * c * c) / denom) + 1.0);
    } else {
        result.cnoa = 0.0;
    }

    return result;
}

Image<float> compute_cost_distance(
    const ImageU8& resistance,
    const ImageU8& source_mask
) {
    if (resistance.nx != source_mask.nx || resistance.ny != source_mask.ny) {
        throw std::invalid_argument("resistance and source_mask dimensions mismatch");
    }

    const i64 nx = resistance.nx;
    const i64 ny = resistance.ny;
    const i64 npix = nx * ny;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    Image<float> dist(nx, ny, kInf);

    // Min-priority queue: (cost, index)
    using Element = std::pair<float, i64>;
    std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;

    for (i64 i = 0; i < npix; ++i) {
        if (source_mask.px[static_cast<std::size_t>(i)] > 0) {
            dist.px[static_cast<std::size_t>(i)] = 0.0f;
            pq.push({0.0f, i});
        }
    }

    // 8-neighbor offsets and distances
    const i64 dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const i64 dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float step_w[8] = {
        1.41421356f, 1.0f, 1.41421356f,
        1.0f,              1.0f,
        1.41421356f, 1.0f, 1.41421356f
    };

    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();

        if (d > dist.px[static_cast<std::size_t>(u)]) continue;

        const i64 ux = u % nx;
        const i64 uy = u / nx;
        const float res_u = static_cast<float>(resistance.px[static_cast<std::size_t>(u)]);

        for (int k = 0; k < 8; ++k) {
            const i64 vx = ux + dx[k];
            const i64 vy = uy + dy[k];

            if (vx < 0 || vx >= nx || vy < 0 || vy >= ny) continue;

            const i64 v = vy * nx + vx;
            const u8 res_v_val = resistance.px[static_cast<std::size_t>(v)];
            if (res_v_val == 0) continue;  // Impassable barrier / missing

            const float res_v = static_cast<float>(res_v_val);
            const float edge_cost = step_w[k] * 0.5f * (res_u + res_v);
            const float new_dist = d + edge_cost;

            if (new_dist < dist.px[static_cast<std::size_t>(v)]) {
                dist.px[static_cast<std::size_t>(v)] = new_dist;
                pq.push({new_dist, v});
            }
        }
    }

    return dist;
}

LeastCostPathResult compute_least_cost_path(
    const ImageU8& resistance,
    const ImageU8& marker_a,
    const ImageU8& marker_b,
    const Image<float>* dist_map
) {
    const i64 nx = resistance.nx;
    const i64 ny = resistance.ny;
    const i64 npix = nx * ny;

    // Compute cost distance from marker A and marker B
    const Image<float> dist_a = compute_cost_distance(resistance, marker_a);
    const Image<float> dist_b = compute_cost_distance(resistance, marker_b);

    constexpr float kInf = std::numeric_limits<float>::infinity();
    LeastCostPathResult result;
    result.combined_cost = Image<float>(nx, ny, kInf);
    result.lcp_mask = ImageU8(nx, ny, 0);

    float min_cost = kInf;
    i64 min_idx = -1;

    for (i64 i = 0; i < npix; ++i) {
        const float da = dist_a.px[static_cast<std::size_t>(i)];
        const float db = dist_b.px[static_cast<std::size_t>(i)];
        if (std::isfinite(da) && std::isfinite(db)) {
            const float c = da + db;
            result.combined_cost.px[static_cast<std::size_t>(i)] = c;
            if (c < min_cost) {
                min_cost = c;
                min_idx = i;
            }
        }
    }

    if (min_idx < 0) {
        // No path exists between markers
        return result;
    }

    // Trace path from min_idx back to marker A and marker B using steepest descent
    const i64 dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const i64 dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float step_w[8] = {
        1.41421356f, 1.0f, 1.41421356f,
        1.0f,              1.0f,
        1.41421356f, 1.0f, 1.41421356f
    };

    auto trace_to_source = [&](i64 start_idx, const Image<float>& d_map, const ImageU8& target_marker) {
        i64 curr = start_idx;
        while (target_marker.px[static_cast<std::size_t>(curr)] == 0) {
            result.lcp_mask.px[static_cast<std::size_t>(curr)] = 1;
            const i64 cx = curr % nx;
            const i64 cy = curr / nx;
            const float res_c = static_cast<float>(resistance.px[static_cast<std::size_t>(curr)]);

            float best_d = d_map.px[static_cast<std::size_t>(curr)];
            i64 next_idx = -1;

            for (int k = 0; k < 8; ++k) {
                const i64 nx_pos = cx + dx[k];
                const i64 ny_pos = cy + dy[k];
                if (nx_pos < 0 || nx_pos >= nx || ny_pos < 0 || ny_pos >= ny) continue;

                const i64 n_idx = ny_pos * nx + nx_pos;
                const float dn = d_map.px[static_cast<std::size_t>(n_idx)];
                if (dn < best_d) {
                    best_d = dn;
                    next_idx = n_idx;
                }
            }

            if (next_idx < 0 || next_idx == curr) break;
            curr = next_idx;
        }
        result.lcp_mask.px[static_cast<std::size_t>(curr)] = 1;
    };

    trace_to_source(min_idx, dist_a, marker_a);
    trace_to_source(min_idx, dist_b, marker_b);

    // Compute metrics along LCP
    i64 path_len = 0;
    i64 rest_pixels = 0;
    double expense = 0.0;
    double dist_sum = 0.0;

    for (i64 i = 0; i < npix; ++i) {
        if (result.lcp_mask.px[static_cast<std::size_t>(i)] > 0) {
            ++path_len;
            // Restore pixels are pixels along path outside the two markers
            if (marker_a.px[static_cast<std::size_t>(i)] == 0 && marker_b.px[static_cast<std::size_t>(i)] == 0) {
                ++rest_pixels;
                expense += static_cast<double>(resistance.px[static_cast<std::size_t>(i)]);
                if (dist_map) {
                    dist_sum += static_cast<double>(dist_map->px[static_cast<std::size_t>(i)]);
                }
            }
        }
    }

    result.path_length = path_len;
    result.rest_pixels = rest_pixels;
    result.expense = expense;
    result.avg_distance = (rest_pixels > 0) ? (dist_sum / static_cast<double>(rest_pixels)) : 0.0;

    return result;
}

}  // namespace gtb
