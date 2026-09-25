// SPDX-License-Identifier: GPL-3.0-or-later
//
// Minimal 2-D raster container used by the landschaft C++ core.
//
// Pixels are stored row-major (x varies fastest), exactly like miallib's
// IMAGE buffers and like a C-contiguous numpy array of shape (ny, nx).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gtb {

using u8 = std::uint8_t;
using i64 = std::int64_t;

constexpr i64 kParallelThreshold = i64(1) << 24;

template <class T>
struct Image {
    i64 nx = 0;
    i64 ny = 0;
    std::vector<T> px;

    Image() = default;
    Image(i64 nx_, i64 ny_, T value = T{})
        : nx(nx_), ny(ny_), px(static_cast<std::size_t>(nx_ * ny_), value) {}

    i64 size() const { return nx * ny; }
    T* data() { return px.data(); }
    const T* data() const { return px.data(); }
    T& operator[](i64 i) { return px[static_cast<std::size_t>(i)]; }
    const T& operator[](i64 i) const { return px[static_cast<std::size_t>(i)]; }
    T& at(i64 x, i64 y) { return px[static_cast<std::size_t>(y * nx + x)]; }
    const T& at(i64 x, i64 y) const { return px[static_cast<std::size_t>(y * nx + x)]; }
};

using ImageU8 = Image<u8>;

// Equivalent of miallib's generic_addframebox with a symmetric 2-D box.
template <class T>
Image<T> add_frame(const Image<T>& im, i64 b, T value) {
    Image<T> out(im.nx + 2 * b, im.ny + 2 * b, value);
    for (i64 y = 0; y < im.ny; ++y) {
        std::copy_n(im.data() + y * im.nx, im.nx, out.data() + (y + b) * out.nx + b);
    }
    return out;
}

// Equivalent of miallib's generic_subframebox with a symmetric 2-D box.
template <class T>
Image<T> sub_frame(const Image<T>& im, i64 b) {
    if (im.nx <= 2 * b || im.ny <= 2 * b) throw std::invalid_argument("sub_frame: frame larger than image");
    Image<T> out(im.nx - 2 * b, im.ny - 2 * b);
    for (i64 y = 0; y < out.ny; ++y) {
        std::copy_n(im.data() + (y + b) * im.nx + b, out.nx, out.data() + y * out.nx);
    }
    return out;
}

// Equivalent of miallib's generic_framebox: overwrite a border of width b.
template <class T>
void set_frame(Image<T>& im, i64 b, T value) {
    const i64 bx = std::min(b, im.nx);
    const i64 by = std::min(b, im.ny);
    for (i64 y = 0; y < im.ny; ++y) {
        T* row = im.data() + y * im.nx;
        if (y < by || y >= im.ny - by) {
            std::fill_n(row, im.nx, value);
        } else {
            std::fill_n(row, bx, value);
            std::fill_n(row + im.nx - bx, bx, value);
        }
    }
}

}  // namespace gtb
