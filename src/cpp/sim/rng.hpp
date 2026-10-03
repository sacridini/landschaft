// SPDX-License-Identifier: GPL-3.0-or-later
//
// The random number generators of the land change models ls.simulate
// reproduces (R's own generator, used by lulcc, is nlm::RRandom):
//
//  - CrtRand: the C runtime's rand() of Microsoft's compilers (UCRT and the
//    older CRTs), which CLUMondo calls: a 32-bit linear congruential
//    generator, 15-bit outputs, seed 1 until srand();
//  - Drand48: the 48-bit linear congruential generator of POSIX drand48 /
//    lrand48 / mrand48, as GRASS GIS implements it portably (G_srand48,
//    G_drand48, lib/gis/lrand48.c), which r.futures calls.
#pragma once

#include <cstdint>

namespace sim {

class CrtRand {
public:
    explicit CrtRand(std::uint32_t seed = 1) : state_(seed) {}
    void seed(std::uint32_t s) { state_ = s; }
    // rand(): 0 .. 32767
    int next() {
        state_ = state_ * 214013u + 2531011u;
        return static_cast<int>((state_ >> 16) & 0x7fffu);
    }
    std::uint32_t state() const { return state_; }

private:
    std::uint32_t state_;
};

class Drand48 {
public:
    // G_srand48(seedval): the low 32 bits of the seed, then 0x330E
    explicit Drand48(std::int64_t seed = 0) { this->seed(seed); }
    void seed(std::int64_t s) {
        state_ = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s)) << 16) | 0x330Eu;
    }
    // G_drand48(): [0, 1), the 48-bit state over 2^48 (exact in a double)
    double next_double() { return static_cast<double>(step()) / 281474976710656.0; }
    // G_lrand48(): 0 .. 2^31 - 1
    std::int64_t next_long() { return static_cast<std::int64_t>(step() >> 17); }
    // G_mrand48(): -2^31 .. 2^31 - 1
    std::int64_t next_signed() {
        return static_cast<std::int64_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(step() >> 16)));
    }
    std::uint64_t state() const { return state_; }

private:
    std::uint64_t step() {
        state_ = (0x5DEECE66Dull * state_ + 0xBull) & 0xFFFFFFFFFFFFull;
        return state_;
    }
    std::uint64_t state_ = 0;
};

}  // namespace sim
