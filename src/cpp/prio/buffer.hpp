// SPDX-License-Identifier: GPL-3.0-or-later
//
// Arrays for the large data of the prioritization (the feature entries, the
// rows of the ranking and its work arrays): in memory, or in temporary files
// mapped into memory when a storage directory is given, so that the operating
// system pages them to disk when they do not fit (out-of-core runs). The access
// patterns of the ranking are sequential or nearly so.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace prio {

// A temporary file mapped into memory, deleted when closed.
class MappedFile {
  public:
    MappedFile(const std::string& dir, std::size_t bytes);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    // make it `bytes` long, keeping the contents (the address may change)
    void grow(std::size_t bytes);
    void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void map();
    void unmap();
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// A vector of trivially copyable values, in memory or in a mapped temporary file
// (new values are zero in both).
template <class T>
class Buffer {
    static_assert(std::is_trivially_copyable_v<T>, "Buffer holds plain values");

  public:
    using value_type = T;

    Buffer() = default;
    explicit Buffer(std::size_t n) { resize(n); }
    Buffer(std::initializer_list<T> v) {
        resize(v.size());
        std::copy(v.begin(), v.end(), data());
    }
    Buffer(const Buffer& o) : dir_(o.dir_) {
        resize(o.size());
        if (size_) std::memcpy(data(), o.data(), size_ * sizeof(T));
    }
    Buffer& operator=(const Buffer& o) {
        if (this != &o) {
            Buffer t(o);
            swap(t);
        }
        return *this;
    }
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept {
        swap(o);
        return *this;
    }
    // from a vector (in memory)
    Buffer& operator=(const std::vector<T>& v) {
        resize(v.size());
        if (size_) std::memcpy(data(), v.data(), size_ * sizeof(T));
        return *this;
    }

    // keep the values in temporary files in `dir` (empty: in memory); only while empty
    void set_storage(const std::string& dir) {
        release();
        dir_ = dir;
    }
    const std::string& storage() const { return dir_; }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* data() { return ptr(); }
    const T* data() const { return ptr(); }
    T* begin() { return ptr(); }
    T* end() { return ptr() + size_; }
    const T* begin() const { return ptr(); }
    const T* end() const { return ptr() + size_; }
    T& operator[](std::size_t i) { return ptr()[i]; }
    const T& operator[](std::size_t i) const { return ptr()[i]; }
    T& back() { return ptr()[size_ - 1]; }

    void reserve(std::size_t n) {
        if (n > cap_) set_capacity(n);
    }
    // new values are zero
    void resize(std::size_t n) {
        if (n > cap_) set_capacity(std::max(n, cap_ + cap_ / 2));
        if (n > size_) std::memset(static_cast<void*>(ptr() + size_), 0, (n - size_) * sizeof(T));
        size_ = n;
    }
    void assign(std::size_t n, T v) {
        resize(n);
        std::fill(ptr(), ptr() + n, v);
    }
    void push_back(T v) {
        if (size_ == cap_) set_capacity(std::max<std::size_t>(16, cap_ * 2));
        ptr()[size_++] = v;
    }
    // append n values (amortized growth)
    void append(const T* v, std::size_t n) {
        const std::size_t s = size_;
        if (s + n > cap_) set_capacity(std::max(s + n, cap_ + cap_ / 2));
        if (n) std::memcpy(static_cast<void*>(ptr() + s), v, n * sizeof(T));
        size_ = s + n;
    }
    void clear() { size_ = 0; }
    void shrink_to_fit() {
        if (dir_.empty() && cap_ > size_) set_capacity(size_);
    }
    // free the storage
    void release() {
        mem_.reset();
        file_.reset();
        p_ = nullptr;
        size_ = cap_ = 0;
    }
    void swap(Buffer& o) noexcept {
        std::swap(dir_, o.dir_);
        std::swap(mem_, o.mem_);
        std::swap(file_, o.file_);
        std::swap(p_, o.p_);
        std::swap(size_, o.size_);
        std::swap(cap_, o.cap_);
    }

  private:
    T* ptr() const { return p_; }
    void set_capacity(std::size_t n) {
        if (!dir_.empty()) {
            const std::size_t bytes = std::max<std::size_t>(n * sizeof(T), 1);
            if (file_)
                file_->grow(bytes);
            else
                file_ = std::make_unique<MappedFile>(dir_, bytes);
            p_ = static_cast<T*>(file_->data());
        } else {
            std::unique_ptr<T[]> m(n ? new T[n] : nullptr);
            if (size_ && n) std::memcpy(static_cast<void*>(m.get()), mem_.get(), std::min(size_, n) * sizeof(T));
            mem_ = std::move(m);
            p_ = mem_.get();
        }
        cap_ = n;
        size_ = std::min(size_, n);
    }

    std::string dir_;
    std::unique_ptr<T[]> mem_;
    std::unique_ptr<MappedFile> file_;
    T* p_ = nullptr;
    std::size_t size_ = 0, cap_ = 0;
};

}  // namespace prio
