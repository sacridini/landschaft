// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "image.hpp"

namespace gtb {

using ImageI64 = Image<std::int64_t>;

// Scans 8-neighborhood of each bridge pixel to extract unique (link_id, node_id)
// contact pairs. Returns sorted pairs of (link_id, node_id).
std::vector<std::pair<std::int64_t, std::int64_t>> extract_node_link_contacts(
    const ImageI64& nodes,
    const ImageI64& links
);

}  // namespace gtb
