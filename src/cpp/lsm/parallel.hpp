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

// body(i, scratch) for i in [0, n), dynamically scheduled in chunks of
// `chunk` when `parallel`, with one Scratch (buffers reused by the
// iterations, which reset what they use) per thread. The serial loop keeps
// its Scratch for the thread's next call (small maps come one after the
// other), so the body must not run another loop with the same Scratch type.
template <class Scratch, class F>
inline void parallel_for_scratch(bool parallel, std::int64_t n, std::int64_t chunk, F&& body) {
    if (parallel) {
#pragma omp parallel
        {
            Scratch scratch{};
#pragma omp for schedule(dynamic, chunk)
            for (std::int64_t i = 0; i < n; ++i) body(i, scratch);
        }
    } else {
        thread_local Scratch scratch{};
        for (std::int64_t i = 0; i < n; ++i) body(i, scratch);
    }
}

}  // namespace lsm
