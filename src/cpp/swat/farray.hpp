// SPDX-License-Identifier: GPL-3.0-or-later
//
// Arrays with Fortran bounds (default lower bound 1, column-major for 2-D/3-D), so the
// SWAT+ code ports index for index. Out-of-bounds access throws: the Fortran reference
// would read or write past its arrays there, which no port can reproduce.

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "swat/fio.hpp"

namespace swat {

// A fixed array with Fortran bounds lo:hi (dimension (12) -> Fix<T, 1, 12>).
template <class T, int Lo, int Hi>
struct Fix {
    static_assert(Hi >= Lo, "empty fixed array");
    T v[Hi - Lo + 1]{};
    static Fix filled(const T& x) {
        Fix f;
        for (auto& e : f.v) e = x;
        return f;
    }
    T& operator()(int i) { return v[idx(i)]; }
    const T& operator()(int i) const { return v[idx(i)]; }
    static constexpr int size() { return Hi - Lo + 1; }
    void fill(const T& x) {
        for (auto& e : v) e = x;
    }
    fio::Items<T> items() { return fio::items(v, size()); }

private:
    static int idx(int i) {
        if (i < Lo || i > Hi)
            throw std::out_of_range("SWAT+ array index " + std::to_string(i) + " outside " + std::to_string(Lo) + ":" +
                                    std::to_string(Hi));
        return i - Lo;
    }
};

// dimension (lo1:hi1, lo2:hi2), column-major
template <class T, int Lo1, int Hi1, int Lo2, int Hi2>
struct Fix2 {
    static constexpr int n1 = Hi1 - Lo1 + 1, n2 = Hi2 - Lo2 + 1;
    T v[n1 * n2]{};
    static Fix2 filled(const T& x) {
        Fix2 f;
        for (auto& e : f.v) e = x;
        return f;
    }
    T& operator()(int i, int j) { return v[idx(i, j)]; }
    const T& operator()(int i, int j) const { return v[idx(i, j)]; }
    fio::Items<T> items() { return fio::items(v, n1 * n2); }

private:
    static int idx(int i, int j) {
        if (i < Lo1 || i > Hi1 || j < Lo2 || j > Hi2) throw std::out_of_range("SWAT+ 2-D array index");
        return (i - Lo1) + n1 * (j - Lo2);
    }
};

template <class T>
class Vec {
public:
    Vec() = default;
    explicit Vec(int n, const T& v = T()) { alloc(1, n, v); }
    Vec(int lo, int hi, const T& v) { alloc(lo, hi, v); }
    void alloc(int lo, int hi, const T& v = T()) {
        lo_ = lo;
        d_.assign(hi >= lo ? static_cast<std::size_t>(hi - lo + 1) : 0, v);
    }
    void alloc(int n) { alloc(1, n, T()); }
    bool allocated() const { return allocated_flag(); }
    int lbound() const { return lo_; }
    int ubound() const { return lo_ + static_cast<int>(d_.size()) - 1; }
    int size() const { return static_cast<int>(d_.size()); }
    T& operator()(int i) { return d_[idx(i)]; }
    const T& operator()(int i) const { return d_[idx(i)]; }
    void fill(const T& v) { std::fill(d_.begin(), d_.end(), v); }
    T* data() { return d_.data(); }
    const T* data() const { return d_.data(); }
    auto begin() { return d_.begin(); }
    auto end() { return d_.end(); }
    auto begin() const { return d_.begin(); }
    auto end() const { return d_.end(); }

private:
    bool allocated_flag() const { return !d_.empty(); }
    std::size_t idx(int i) const {
        const long long k = static_cast<long long>(i) - lo_;
        if (k < 0 || k >= static_cast<long long>(d_.size()))
            throw std::out_of_range("SWAT+ array index " + std::to_string(i) + " outside " + std::to_string(lo_) + ":" +
                                    std::to_string(ubound()));
        return static_cast<std::size_t>(k);
    }
    int lo_ = 1;
    std::vector<T> d_;
};

template <class T>
class Mat {
public:
    Mat() = default;
    void alloc(int lo1, int hi1, int lo2, int hi2, const T& v = T()) {
        lo1_ = lo1;
        lo2_ = lo2;
        n1_ = hi1 >= lo1 ? hi1 - lo1 + 1 : 0;
        n2_ = hi2 >= lo2 ? hi2 - lo2 + 1 : 0;
        d_.assign(static_cast<std::size_t>(n1_) * static_cast<std::size_t>(n2_), v);
    }
    void alloc(int n1, int n2, const T& v = T()) { alloc(1, n1, 1, n2, v); }
    bool allocated() const { return !d_.empty() || n1_ > 0 || n2_ > 0; }
    int ubound1() const { return lo1_ + n1_ - 1; }
    int ubound2() const { return lo2_ + n2_ - 1; }
    T& operator()(int i, int j) { return d_[idx(i, j)]; }
    const T& operator()(int i, int j) const { return d_[idx(i, j)]; }
    void fill(const T& v) { std::fill(d_.begin(), d_.end(), v); }

private:
    std::size_t idx(int i, int j) const {
        const int a = i - lo1_, b = j - lo2_;
        if (a < 0 || a >= n1_ || b < 0 || b >= n2_)
            throw std::out_of_range("SWAT+ array index (" + std::to_string(i) + "," + std::to_string(j) + ") outside (" +
                                    std::to_string(lo1_) + ":" + std::to_string(ubound1()) + "," + std::to_string(lo2_) +
                                    ":" + std::to_string(ubound2()) + ")");
        return static_cast<std::size_t>(a) + static_cast<std::size_t>(n1_) * static_cast<std::size_t>(b);
    }
    int lo1_ = 1, lo2_ = 1, n1_ = 0, n2_ = 0;
    std::vector<T> d_;
};

template <class T>
class Cube {
public:
    void alloc(int n1, int n2, int n3, const T& v = T()) {
        n1_ = n1;
        n2_ = n2;
        n3_ = n3;
        d_.assign(static_cast<std::size_t>(n1) * n2 * n3, v);
    }
    T& operator()(int i, int j, int k) { return d_[idx(i, j, k)]; }
    const T& operator()(int i, int j, int k) const { return d_[idx(i, j, k)]; }

private:
    std::size_t idx(int i, int j, int k) const {
        if (i < 1 || i > n1_ || j < 1 || j > n2_ || k < 1 || k > n3_) throw std::out_of_range("SWAT+ 3-D array index");
        return static_cast<std::size_t>(i - 1) + static_cast<std::size_t>(n1_) * (static_cast<std::size_t>(j - 1) +
                                                                                  static_cast<std::size_t>(n2_) * (k - 1));
    }
    int n1_ = 0, n2_ = 0, n3_ = 0;
    std::vector<T> d_;
};

}  // namespace swat
