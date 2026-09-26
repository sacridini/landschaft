// SPDX-License-Identifier: GPL-3.0-or-later
//
// Loops that run in parallel only when asked. A loop with an OpenMP `if`
// clause still sets up a (serialised) parallel region when the condition is
// false, which costs more than the work itself on small maps (moving
// windows, sample plots: thousands of small landscapes per second); these
// helpers run a plain loop instead.
#pragma once

#include <cstdint>

namespace lsm {

// body(i) for i in [0, n), statically scheduled when `parallel`.
template <class F>
inline void parallel_for(bool parallel, std::int64_t n, F&& body) {
    if (parallel) {
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) body(i);
    } else {
        for (std::int64_t i = 0; i < n; ++i) body(i);
    }
}

// The same with dynamic scheduling (uneven iterations).
template <class F>
inline void parallel_for_dynamic(bool parallel, std::int64_t n, F&& body) {
    if (parallel) {
#pragma omp parallel for schedule(dynamic)
        for (std::int64_t i = 0; i < n; ++i) body(i);
    } else {
        for (std::int64_t i = 0; i < n; ++i) body(i);
    }
}

}  // namespace lsm
