// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reuse of large image buffers inside a tool call.
//
// A tool such as MSPA allocates and frees dozens of full-size images. Large
// blocks come straight from the operating system, which hands out fresh
// zeroed pages on every allocation: on a 2000 x 2000 map that page faulting
// costs about a fifth of MSPA's run time. While a PoolScope is alive, freed
// blocks of at least kPoolMinBytes are kept (up to kPoolMaxCached bytes) and
// handed out again for requests of the same size; when the last scope ends,
// the kept blocks are released. Outside a scope the allocator is plain
// operator new / delete, so buffers may cross scope boundaries freely.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace gtb {

namespace pool_detail {

constexpr std::size_t kPoolMinBytes = std::size_t(1) << 18;    // 256 KiB
constexpr std::size_t kPoolMaxCached = std::size_t(1) << 30;   // 1 GiB

struct Pool {
    std::mutex mutex;
    int scopes = 0;
    std::size_t cached = 0;
    std::unordered_map<std::size_t, std::vector<void*>> blocks;  // by size in bytes

    void release_all() {  // with the mutex held
        for (auto& [bytes, list] : blocks)
            for (void* p : list) ::operator delete(p);
        blocks.clear();
        cached = 0;
    }
};

inline Pool& pool() {
    static Pool p;
    return p;
}

inline void* allocate(std::size_t bytes) {
    if (bytes >= kPoolMinBytes) {
        Pool& p = pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        if (p.scopes > 0) {
            auto it = p.blocks.find(bytes);
            if (it != p.blocks.end() && !it->second.empty()) {
                void* block = it->second.back();
                it->second.pop_back();
                p.cached -= bytes;
                return block;
            }
        }
    }
    return ::operator new(bytes);
}

inline void deallocate(void* block, std::size_t bytes) {
    if (bytes >= kPoolMinBytes) {
        Pool& p = pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        if (p.scopes > 0 && p.cached + bytes <= kPoolMaxCached) {
            p.blocks[bytes].push_back(block);
            p.cached += bytes;
            return;
        }
    }
    ::operator delete(block);
}

}  // namespace pool_detail

// Keeps freed large buffers for reuse while alive (scopes may nest and run
// on several threads; the buffers are released when the last one ends).
class PoolScope {
public:
    PoolScope() {
        auto& p = pool_detail::pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        ++p.scopes;
    }
    ~PoolScope() {
        auto& p = pool_detail::pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        if (--p.scopes == 0) p.release_all();
    }
    PoolScope(const PoolScope&) = delete;
    PoolScope& operator=(const PoolScope&) = delete;
};

template <class T>
struct PoolAllocator {
    using value_type = T;
    PoolAllocator() noexcept = default;
    template <class U>
    PoolAllocator(const PoolAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) { return static_cast<T*>(pool_detail::allocate(n * sizeof(T))); }
    void deallocate(T* p, std::size_t n) noexcept { pool_detail::deallocate(p, n * sizeof(T)); }
    template <class U>
    bool operator==(const PoolAllocator<U>&) const noexcept { return true; }
    template <class U>
    bool operator!=(const PoolAllocator<U>&) const noexcept { return false; }
};

// Pixel storage of images and per-pixel work arrays.
template <class T>
using Buffer = std::vector<T, PoolAllocator<T>>;

}  // namespace gtb
