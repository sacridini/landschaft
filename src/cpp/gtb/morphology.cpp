// SPDX-License-Identifier: GPL-3.0-or-later
#include "morphology.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace gtb {

namespace {

struct Point {
    i64 x = 0;
    i64 y = 0;

    bool operator==(const Point& o) const {
        return x == o.x && y == o.y;
    }
};

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

std::vector<i64> collapse_consecutive(const std::vector<i64>& v) {
    std::vector<i64> res;
    res.reserve(v.size());
    for (const i64 x : v) {
        if (res.empty() || res.back() != x) {
            res.push_back(x);
        }
    }
    return res;
}

}  // namespace

ParcellationResult compute_parcellation(
    const ImageU8& im,
    int connectivity,
    bool return_labels
) {
    if (connectivity != 4 && connectivity != 8) {
        throw std::invalid_argument("connectivity must be 4 or 8");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 total_pixels = im.size();

    // Find valid data area (pixels != 0) and unique classes
    i64 data_area = 0;
    std::set<u8> unique_classes;

    for (i64 i = 0; i < total_pixels; ++i) {
        const u8 v = im.px[static_cast<std::size_t>(i)];
        if (v != 0) {
            ++data_area;
            unique_classes.insert(v);
        }
    }

    ParcellationResult result;
    result.global_stats.data_area = data_area;

    if (data_area == 0 || unique_classes.empty()) {
        return result;
    }

    if (return_labels) {
        result.parcel_labels = Image<std::uint32_t>(nx, ny, 0);
    }

    const double div_up = -std::log(1.0e-6);
    const double div_scale = 100.0 / div_up;

    std::vector<i64> all_patch_sizes;
    std::uint32_t current_global_parcel_id = 1;

    for (const u8 cls : unique_classes) {
        // Label components for this specific class
        std::vector<i64> labels(static_cast<std::size_t>(total_pixels), 0);
        DisjointSet dsu(static_cast<std::size_t>(data_area + 1));
        i64 next_label = 1;

        for (i64 y = 0; y < ny; ++y) {
            const i64 row_off = y * nx;
            const i64 prev_row_off = (y > 0) ? (y - 1) * nx : 0;

            for (i64 x = 0; x < nx; ++x) {
                const i64 idx = row_off + x;
                if (im.px[static_cast<std::size_t>(idx)] != cls) continue;

                i64 min_root = 0;

                if (y > 0) {
                    if (connectivity == 8 && x > 0) {
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
                    if (connectivity == 8 && x + 1 < nx) {
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
                    if (y > 0) {
                        if (connectivity == 8 && x > 0 && labels[static_cast<std::size_t>(prev_row_off + (x - 1))] > 0) {
                            dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + (x - 1))]);
                        }
                        if (labels[static_cast<std::size_t>(prev_row_off + x)] > 0) {
                            dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + x)]);
                        }
                        if (connectivity == 8 && x + 1 < nx && labels[static_cast<std::size_t>(prev_row_off + (x + 1))] > 0) {
                            dsu.unite(min_root, labels[static_cast<std::size_t>(prev_row_off + (x + 1))]);
                        }
                    }
                    if (x > 0 && labels[static_cast<std::size_t>(row_off + (x - 1))] > 0) {
                        dsu.unite(min_root, labels[static_cast<std::size_t>(row_off + (x - 1))]);
                    }
                }
            }
        }

        // Count patch areas per root
        std::map<i64, i64> root_areas;
        std::map<i64, std::uint32_t> root_to_parcel_id;

        for (i64 i = 0; i < total_pixels; ++i) {
            if (labels[static_cast<std::size_t>(i)] > 0) {
                const i64 root = dsu.find(labels[static_cast<std::size_t>(i)]);
                root_areas[root]++;
            }
        }

        std::vector<i64> class_patch_sizes;
        class_patch_sizes.reserve(root_areas.size());
        for (const auto& [root, area] : root_areas) {
            class_patch_sizes.push_back(area);
            all_patch_sizes.push_back(area);
            if (return_labels) {
                root_to_parcel_id[root] = current_global_parcel_id++;
            }
        }

        if (return_labels) {
            for (i64 i = 0; i < total_pixels; ++i) {
                if (labels[static_cast<std::size_t>(i)] > 0) {
                    const i64 root = dsu.find(labels[static_cast<std::size_t>(i)]);
                    result.parcel_labels.px[static_cast<std::size_t>(i)] = root_to_parcel_id[root];
                }
            }
        }

        ClassParcellationStats cs;
        cs.class_id = cls;
        cs.count = static_cast<i64>(class_patch_sizes.size());

        i64 class_area = 0;
        double sum_sq = 0.0;
        for (const i64 a : class_patch_sizes) {
            class_area += a;
            sum_sq += static_cast<double>(a) * static_cast<double>(a);
        }
        cs.area = class_area;

        if (cs.count > 0 && class_area > 0) {
            const double ca = static_cast<double>(class_area);
            cs.aps = ca / static_cast<double>(cs.count);
            cs.awaps_rel = sum_sq / ca;
            cs.awaps_data = sum_sq / static_cast<double>(data_area);

            double s_rel = 0.0;
            for (const i64 a : class_patch_sizes) {
                const double p = static_cast<double>(a) / ca;
                s_rel += p * p;
            }
            cs.division = 1.0 - s_rel;

            double div = -std::log(s_rel);
            if (div > div_up) div = div_up;
            cs.parc = div_scale * std::abs(div);
        }

        result.class_stats.push_back(cs);
    }

    // Global stats across all classes
    result.global_stats.total_parcels = static_cast<i64>(all_patch_sizes.size());
    if (result.global_stats.total_parcels > 0 && data_area > 0) {
        const double da = static_cast<double>(data_area);
        result.global_stats.aps = da / static_cast<double>(result.global_stats.total_parcels);

        double total_sum_sq = 0.0;
        double total_s = 0.0;
        for (const i64 a : all_patch_sizes) {
            const double da_val = static_cast<double>(a);
            total_sum_sq += da_val * da_val;
            const double p = da_val / da;
            total_s += p * p;
        }

        result.global_stats.awaps = total_sum_sq / da;
        result.global_stats.division = 1.0 - total_s;

        double div = -std::log(total_s);
        if (div > div_up) div = div_up;
        result.global_stats.parc = div_scale * std::abs(div);
    }

    return result;
}

Image<std::uint32_t> compute_contortion(
    const ImageU8& im,
    i64 min_area
) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 total_pixels = im.size();

    // Output raster:
    // 0: nodata (im == 0)
    // 1: background (im == 1)
    // 2: neglected objects (im == 2 but area < min_area or no core)
    // >= 3: contortion value (scaled as contortion + 1)
    Image<std::uint32_t> out(nx, ny, 1);
    for (i64 i = 0; i < total_pixels; ++i) {
        const u8 v = im.px[static_cast<std::size_t>(i)];
        if (v == 0) out.px[static_cast<std::size_t>(i)] = 0;
        else if (v == 1) out.px[static_cast<std::size_t>(i)] = 1;
        else if (v == 2) out.px[static_cast<std::size_t>(i)] = 2;
    }

    // Label 8-connected components on foreground (im == 2)
    std::vector<i64> labels(static_cast<std::size_t>(total_pixels), 0);
    DisjointSet dsu(static_cast<std::size_t>(total_pixels + 1));
    i64 next_label = 1;

    for (i64 y = 0; y < ny; ++y) {
        const i64 row_off = y * nx;
        const i64 prev_row_off = (y > 0) ? (y - 1) * nx : 0;

        for (i64 x = 0; x < nx; ++x) {
            const i64 idx = row_off + x;
            if (im.px[static_cast<std::size_t>(idx)] != 2) continue;

            i64 min_root = 0;

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

    // Remap roots to contiguous component IDs 1..num_comp
    std::map<i64, i64> root_to_comp;
    i64 num_comp = 0;
    std::vector<std::vector<Point>> comp_pixels;
    comp_pixels.push_back({}); // dummy 0

    // Also check for 3x3 core pixels: pixel whose 8 neighbors are all foreground (v == 2)
    std::vector<bool> has_core;
    has_core.push_back(false);

    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 idx = y * nx + x;
            if (labels[static_cast<std::size_t>(idx)] > 0) {
                const i64 r = dsu.find(labels[static_cast<std::size_t>(idx)]);
                auto it = root_to_comp.find(r);
                i64 cid = 0;
                if (it == root_to_comp.end()) {
                    cid = ++num_comp;
                    root_to_comp[r] = cid;
                    comp_pixels.push_back({});
                    has_core.push_back(false);
                } else {
                    cid = it->second;
                }
                comp_pixels[static_cast<std::size_t>(cid)].push_back({x, y});

                // Check core status (all 8 neighbors == 2)
                if (!has_core[static_cast<std::size_t>(cid)]) {
                    if (x > 0 && x + 1 < nx && y > 0 && y + 1 < ny) {
                        bool all_fg = true;
                        for (i64 dy = -1; dy <= 1 && all_fg; ++dy) {
                            for (i64 dx = -1; dx <= 1 && all_fg; ++dx) {
                                if (im.px[static_cast<std::size_t>((y + dy) * nx + (x + dx))] != 2) {
                                    all_fg = false;
                                }
                            }
                        }
                        if (all_fg) {
                            has_core[static_cast<std::size_t>(cid)] = true;
                        }
                    }
                }
            }
        }
    }

    // Map labels raster to canonical comp IDs
    Image<std::uint32_t> comp_im(nx, ny, 0);
    for (i64 i = 0; i < total_pixels; ++i) {
        if (labels[static_cast<std::size_t>(i)] > 0) {
            const i64 r = dsu.find(labels[static_cast<std::size_t>(i)]);
            comp_im.px[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(root_to_comp[r]);
        }
    }

    // Moore-neighbor direction offsets (matching GTB darray)
    // 0: (1,0), 1: (1,1), 2: (0,1), 3: (-1,1), 4: (-1,0), 5: (-1,-1), 6: (0,-1), 7: (1,-1)
    const i64 d_x[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const i64 d_y[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (i64 cid = 1; cid <= num_comp; ++cid) {
        const auto& pts = comp_pixels[static_cast<std::size_t>(cid)];
        const i64 area = static_cast<i64>(pts.size());

        if (area < min_area || !has_core[static_cast<std::size_t>(cid)]) {
            // Neglected object (stays 2)
            continue;
        }

        // Trace boundary using Moore-neighbor tracing
        const Point first_pt = pts[0]; // first pixel in raster scan order
        std::vector<Point> b_pts;
        b_pts.reserve(pts.size());
        b_pts.push_back(first_pt);

        Point sec_pt{-1, -1};
        Point curr = first_pt;
        i64 from_dir = 4; // came from left (West)

        constexpr i64 kMaxSteps = 1000000;
        i64 steps = 0;

        while (++steps < kMaxSteps) {
            bool found_next = false;
            for (i64 j = 1; j <= 7; ++j) {
                const i64 to_dir = (from_dir + j) % 8;
                const i64 nx_pos = curr.x + d_x[to_dir];
                const i64 ny_pos = curr.y + d_y[to_dir];

                if (nx_pos >= 0 && nx_pos < nx && ny_pos >= 0 && ny_pos < ny) {
                    if (comp_im.px[static_cast<std::size_t>(ny_pos * nx + nx_pos)] == static_cast<std::uint32_t>(cid)) {
                        const Point next_pt{nx_pos, ny_pos};
                        b_pts.push_back(next_pt);
                        from_dir = (to_dir + 4) % 8;
                        curr = next_pt;
                        found_next = true;
                        break;
                    }
                }
            }

            if (!found_next) {
                // Solitary or isolated point
                break;
            }

            if (b_pts.size() == 2) {
                sec_pt = b_pts[1];
            } else if (b_pts.size() > 2) {
                // Termination condition: arrived back at first_pt and sec_pt
                if (b_pts.back() == sec_pt && b_pts[b_pts.size() - 2] == first_pt) {
                    // Pop the repeated first_pt and sec_pt
                    b_pts.pop_back();
                    b_pts.pop_back();
                    break;
                }
            }
        }

        i64 contortion = 2;
        if (b_pts.size() >= 2) {
            // GTB formulation:
            // xb = [b_pts.x, b_pts[0].x, b_pts[1].x]
            // yb = [b_pts.y, b_pts[0].y, b_pts[1].y]
            std::vector<i64> xb;
            std::vector<i64> yb;
            xb.reserve(b_pts.size() + 2);
            yb.reserve(b_pts.size() + 2);

            for (const auto& p : b_pts) {
                xb.push_back(p.x);
                yb.push_back(p.y);
            }
            xb.push_back(b_pts[0].x);
            xb.push_back(b_pts[1].x);
            yb.push_back(b_pts[0].y);
            yb.push_back(b_pts[1].y);

            const std::size_t q = xb.size() - 1;
            std::vector<i64> dxb;
            std::vector<i64> dyb;
            dxb.reserve(q + 2);
            dyb.reserve(q + 2);

            dxb.push_back(5);
            for (std::size_t i = 0; i < q; ++i) {
                const i64 diff = xb[i + 1] - xb[i];
                if (diff != 0) dxb.push_back(diff);
            }
            dxb.push_back(6);

            dyb.push_back(5);
            for (std::size_t i = 0; i < q; ++i) {
                const i64 diff = yb[i + 1] - yb[i];
                if (diff != 0) dyb.push_back(diff);
            }
            dyb.push_back(6);

            const auto u_dxb = collapse_consecutive(dxb);
            const auto u_dyb = collapse_consecutive(dyb);

            contortion = std::max(i64{2}, static_cast<i64>(u_dxb.size() + u_dyb.size()) - 7);
        }

        // Assign contortion to all pixels of component (contortion + 1 for GTB palette scaling)
        const std::uint32_t val = static_cast<std::uint32_t>(contortion + 1);
        for (const auto& p : pts) {
            out.px[static_cast<std::size_t>(p.y * nx + p.x)] = val;
        }
    }

    return out;
}

}  // namespace gtb
