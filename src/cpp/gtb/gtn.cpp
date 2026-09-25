// SPDX-License-Identifier: GPL-3.0-or-later
#include "gtn.hpp"

#include <algorithm>
#include <set>

namespace gtb {

std::vector<std::pair<std::int64_t, std::int64_t>> extract_node_link_contacts(
    const ImageI64& nodes,
    const ImageI64& links
) {
    if (nodes.nx != links.nx || nodes.ny != links.ny) {
        throw std::invalid_argument("nodes and links dimensions mismatch");
    }

    const i64 nx = nodes.nx;
    const i64 ny = nodes.ny;

    std::set<std::pair<std::int64_t, std::int64_t>> contacts;

    // 8-neighbor offsets
    const i64 dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const i64 dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const std::int64_t lid = links.at(x, y);
            if (lid <= 0) continue;

            for (int k = 0; k < 8; ++k) {
                const i64 nx_pos = x + dx[k];
                const i64 ny_pos = y + dy[k];
                if (nx_pos >= 0 && nx_pos < nx && ny_pos >= 0 && ny_pos < ny) {
                    const std::int64_t nid = nodes.at(nx_pos, ny_pos);
                    if (nid > 0) {
                        contacts.insert({lid, nid});
                    }
                }
            }
        }
    }

    return std::vector<std::pair<std::int64_t, std::int64_t>>(contacts.begin(), contacts.end());
}

}  // namespace gtb
