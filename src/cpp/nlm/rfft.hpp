// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's fft() (Singleton's mixed-radix FFT, R 4.4's stats/src/fft.c).
#pragma once

#include <vector>

#include "nlm/rng.hpp"

namespace nlm {

// fft_factor() of one length, then fft_work() on interleaved complex data.
struct RFftPlan {
    explicit RFftPlan(int n);
    // The transforms of `nseg` segments of `n` values `nspn` apart (R's
    // fft_work); unnormalised, e^{+i} when `inverse`.
    void run(double* z, int nseg, int nspn, bool inverse) const;

    int n;
    int nfac[20] = {};
    int m_fac = 0, kt = 0, maxf = 0, maxp = 0;
};

// R's fft(z, inverse) of an nrow x ncol complex matrix stored column-major
// with interleaved real and imaginary parts. `parallel` runs every column's
// and every row's transform on its own (the same bits as R's single call).
void r_fft2(std::vector<double>& z, int nrow, int ncol, bool inverse, bool parallel);

}  // namespace nlm
