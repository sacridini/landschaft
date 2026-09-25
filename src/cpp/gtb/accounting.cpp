// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of the 'accounting' / 'batch_accounting' events of guidostoolbox.pro
// (GuidosToolbox, (C) European Union (JRC), Peter Vogt, GPL-3.0-or-later).
#include "accounting.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "primitives.hpp"

namespace gtb {

std::vector<i64> accounting_auto_thresholds(const std::vector<i64>& areas8) {
    if (areas8.size() < 2) throw std::invalid_argument("no foreground objects");
    const auto [lo, hi] = std::minmax_element(areas8.begin() + 1, areas8.end());
    // IDL: ulong64(oba_min + [0.001, 0.01, 0.1, 0.5, 0.9] * oba_range), all in float
    const float fractions[5] = {0.001f, 0.01f, 0.1f, 0.5f, 0.9f};
    const float range = static_cast<float>(*hi - *lo);
    std::vector<i64> out;
    for (const float f : fractions) {
        const float v = static_cast<float>(*lo) + f * range;
        const i64 t = static_cast<i64>(v);
        if (out.empty() || out.back() != t) out.push_back(t);  // cat(uniq(cat))
    }
    return out;
}

AccountingResult compute_accounting(const ImageU8& im, int graph, const std::vector<i64>& thresholds_in,
                                    bool highlight_largest) {
    AccountingResult res;
    const i64 n = label_regions(im, 2, graph, /*bottom_up=*/true, res.ids, res.object_area);
    const auto& area = res.object_area;
    if (n == 0) throw std::invalid_argument("no foreground objects");

    if (thresholds_in.empty()) {
        // GTB derives the proposed classes from the 8-connected objects.
        if (graph == 8) {
            res.thresholds = accounting_auto_thresholds(area);
        } else {
            Image<std::int32_t> ids8;
            std::vector<i64> areas8;
            label_regions(im, 2, 8, /*bottom_up=*/true, ids8, areas8);
            res.thresholds = accounting_auto_thresholds(areas8);
        }
    } else {
        res.thresholds = thresholds_in;
    }
    const auto& thresholds = res.thresholds;
    if (thresholds.size() > 5) throw std::invalid_argument("at most 5 size thresholds");
    for (std::size_t k = 0; k < thresholds.size(); ++k) {
        if (thresholds[k] < 1 || (k > 0 && thresholds[k] <= thresholds[k - 1]))
            throw std::invalid_argument("size thresholds must be >= 1 and strictly increasing");
    }

    // Colour of every object: the first class whose threshold it does not exceed.
    std::vector<u8> code(static_cast<std::size_t>(n) + 1, 0);
    for (i64 k = 1; k <= n; ++k) {
        const auto cls = std::lower_bound(thresholds.begin(), thresholds.end(), area[k]) - thresholds.begin();
        code[k] = kAccountingClassCodes[static_cast<std::size_t>(cls)];
    }

    // Three largest areas; each rank takes every object of that area (GTB
    // repeatedly selects `obj_area eq max` among the objects left).
    std::vector<i64> sorted(area.begin() + 1, area.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (std::size_t rank = 0; rank < 3 && rank < sorted.size(); ++rank) {
        for (i64 k = 1; k <= n; ++k) {
            if (area[k] == sorted[rank]) res.largest[rank].push_back(static_cast<std::int32_t>(k));
        }
    }
    if (highlight_largest && n >= 3) {
        for (const auto& ids : res.largest) {
            for (const std::int32_t k : ids) code[k] = kAccountingLargest;
        }
    }

    const i64 npx = im.size();
    res.classes = ImageU8(im.nx, im.ny, 0);
    res.area = Image<std::int32_t>(im.nx, im.ny, 0);
    const u8* in = im.data();
    u8* cls = res.classes.data();
    std::int32_t* ids = res.ids.data();
    std::int32_t* ar = res.area.data();
    const u8* code_p = code.data();
    const i64* area_p = area.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (npx >= kParallelThreshold)
#endif
    for (i64 i = 0; i < npx; ++i) {
        switch (in[i]) {
            case 2: {
                const std::int32_t k = ids[i];
                cls[i] = code_p[k];
                ar[i] = static_cast<std::int32_t>(area_p[k]);
                break;
            }
            case 0:
                cls[i] = kAccountingMissing;
                ids[i] = ar[i] = -1;
                break;
            case 3:
                cls[i] = kAccountingSpecial1;
                ids[i] = ar[i] = -3;
                break;
            case 4:
                cls[i] = kAccountingSpecial2;
                ids[i] = ar[i] = -4;
                break;
            default:
                break;  // background: 0 everywhere
        }
    }
    return res;
}

}  // namespace gtb
