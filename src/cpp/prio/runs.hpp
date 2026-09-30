// SPDX-License-Identifier: GPL-3.0-or-later
//
// The quantized features of a cell (Zonation's run-length sparse vector, per
// cell): runs over the layers in order of NoData, of zeros and of values. A
// header byte holds the kind (2 bits) and the length - 1 (6 bits; 63: 64 + a
// LEB128 number follows); a run of values is followed by its values (1 byte, or
// 2 little-endian with more than 8 bits). The runs of the cell end at its last
// layer with data. Zeros cost nothing per layer, so maps of many sparse
// features (species ranges, 0 outside) take about their occurrences.
#pragma once

#include <cstdint>
#include <vector>

#include "prio/buffer.hpp"

namespace prio::runs {

enum Kind : unsigned { NODATA = 0, ZERO = 1, VALUE = 2 };

inline unsigned read_value(const std::uint8_t* p, int vb) {
    return vb == 1 ? p[0] : static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8);
}

// the run at p: kind and length; returns the byte after its header
inline const std::uint8_t* read_run(const std::uint8_t* p, unsigned& kind, std::int64_t& len) {
    const unsigned h = *p++;
    kind = h >> 6;
    const unsigned code = h & 63u;
    if (code < 63) {
        len = code + 1;
        return p;
    }
    std::int64_t v = 0;
    int shift = 0;
    for (;;) {
        const unsigned b = *p++;
        v |= static_cast<std::int64_t>(b & 0x7fu) << shift;
        if (!(b & 0x80u)) break;
        shift += 7;
    }
    len = 64 + v;
    return p;
}

inline int header_bytes(std::int64_t len) {
    if (len <= 63) return 1;
    int n = 2;
    for (std::int64_t v = (len - 64) >> 7; v; v >>= 7) ++n;
    return n;
}

inline std::uint8_t* write_run(std::uint8_t* p, unsigned kind, std::int64_t len) {
    if (len <= 63) {
        *p++ = static_cast<std::uint8_t>(kind << 6 | static_cast<unsigned>(len - 1));
        return p;
    }
    *p++ = static_cast<std::uint8_t>(kind << 6 | 63u);
    std::int64_t v = len - 64;
    do {
        std::uint8_t b = static_cast<std::uint8_t>(v & 0x7f);
        v >>= 7;
        if (v) b |= 0x80;
        *p++ = b;
    } while (v);
    return p;
}

inline std::uint8_t* write_value(std::uint8_t* p, unsigned q, int vb) {
    *p++ = static_cast<std::uint8_t>(q & 0xffu);
    if (vb == 2) *p++ = static_cast<std::uint8_t>(q >> 8);
    return p;
}

// f(layer, value) for the layers with data in the row [p, end); with zeros =
// false the runs of zeros are skipped (runs of values may still hold zeros)
template <bool Zeros, class F>
void each(const std::uint8_t* p, const std::uint8_t* end, int vb, F&& f) {
    int l = 0;
    while (p < end) {
        unsigned kind;
        std::int64_t len;
        p = read_run(p, kind, len);
        if (kind == VALUE) {
            for (std::int64_t k = 0; k < len; ++k, p += vb) f(l + static_cast<int>(k), read_value(p, vb));
        } else if (Zeros && kind == ZERO) {
            for (std::int64_t k = 0; k < len; ++k) f(l + static_cast<int>(k), 0u);
        }
        l += static_cast<int>(len);
    }
}

// Builds the runs of a cell layer by layer (in order).
class Writer {
  public:
    explicit Writer(int vb) : vb_(vb) {}
    // the layer's kind (and value for VALUE); layers skipped are NoData
    void add(int layer, unsigned kind, unsigned q) {
        if (layer > next_) {
            flush();
            push(NODATA, layer - next_, nullptr);
        }
        if (kind != run_kind_ || run_len_ == 0) {
            flush();
            run_kind_ = kind;
        }
        ++run_len_;
        if (kind == VALUE) values_.push_back(q);
        next_ = layer + 1;
    }
    // append the cell's runs to out; returns the bytes written
    template <class Out>
    std::int64_t finish(Out& out) {
        flush();
        const std::int64_t n = static_cast<std::int64_t>(bytes_.size());
        for (std::uint8_t b : bytes_) out.push_back(b);
        bytes_.clear();
        next_ = 0;
        return n;
    }

  private:
    void flush() {
        if (run_len_ > 0) push(run_kind_, run_len_, values_.data());
        run_len_ = 0;
        values_.clear();
    }
    void push(unsigned kind, std::int64_t len, const unsigned* values) {
        std::uint8_t tmp[16];
        std::uint8_t* e = write_run(tmp, kind, len);
        bytes_.insert(bytes_.end(), tmp, e);
        if (kind == VALUE)
            for (std::int64_t k = 0; k < len; ++k) {
                std::uint8_t* v = write_value(tmp, values[k], vb_);
                bytes_.insert(bytes_.end(), tmp, v);
            }
    }

    int vb_;
    int next_ = 0;
    unsigned run_kind_ = NODATA;
    std::int64_t run_len_ = 0;
    std::vector<unsigned> values_;
    std::vector<std::uint8_t> bytes_;
};

}  // namespace prio::runs
