// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of uc_sqtgsym (miallib ggeo.c, (C) European Union (JRC), Pierre Soille,
// GPL-3.0-or-later), as run by GuidosToolbox's `ggeo` program.
#include "isochrone.hpp"

#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gtb {

Image<std::uint32_t> geodesic_time(const ImageU8& m, const ImageU8& ref_in) {
    if (m.nx != ref_in.nx || m.ny != ref_in.ny) throw std::invalid_argument("mask and reference differ in size");
    constexpr u8 kMsk = 3, kRetrieved = 2;
    const i64 nx = m.nx, ny = m.ny, n = m.size();
    ImageU8 ref = ref_in;
    set_frame<u8>(ref, 1, kMsk);  // generic_framebox(im_r, box, MSK)
    ImageU8 mk = m;
    // generic_framebox(im_m, box, PIX_MSB): frame pixels of the mask become 128
    set_frame<u8>(mk, 1, 128);
    Image<std::uint32_t> out(nx, ny, std::numeric_limits<std::uint32_t>::max());
    set_frame<std::uint32_t>(out, 1, 0);

    // set_seq_shift for graph 8: the 8 neighbours
    const i64 shift[8] = {-nx - 1, -nx, -nx + 1, -1, 1, nx - 1, nx, nx + 1};
    using Item = std::pair<std::uint32_t, i64>;  // (priority, offset); ties: any order gives the same times
    std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;

    const u8* pm = mk.data();
    u8* pr = ref.data();
    std::uint32_t* po = out.data();
    for (i64 i = 0; i < n; ++i) {
        if (pr[i] != 1) continue;
        po[i] = 0;
        for (const i64 s : shift) {
            const i64 k = i + s;
            if (pr[k] != 0) continue;
            const std::uint32_t w = static_cast<std::uint32_t>(pm[i]) + pm[k];
            if (w < po[k]) {
                po[k] = w;
                pq.emplace(w, k);
            }
        }
    }
    while (!pq.empty()) {
        const auto [prio, ofs] = pq.top();
        pq.pop();
        if (pr[ofs] == kRetrieved) continue;
        pr[ofs] = kRetrieved;
        for (const i64 s : shift) {
            const i64 k = ofs + s;
            if (pr[k] == kMsk || pr[k] == kRetrieved) continue;
            const std::uint32_t c = prio + pm[ofs] + pm[k];
            if (c < po[k]) {
                po[k] = c;
                pq.emplace(c, k);
            }
        }
    }
    return out;
}

}  // namespace gtb
