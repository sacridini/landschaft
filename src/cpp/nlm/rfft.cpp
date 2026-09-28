// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's fft() (src/library/stats/src/fft.c of R 4.4, Singleton's mixed-radix
// algorithm as translated by Ross Ihaka), ported statement by statement so
// that the transforms have R's bits; sin/cos are mingw-w64's (x87), as in R's
// Windows build. The 2-D transform runs the 1-D transforms of the columns
// and then of the rows in parallel: each sequence goes through the same
// operations as in R's call over all of them at once.

#include "nlm/rfft.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace nlm {

namespace {

inline int imax2(int a, int b) { return a > b ? a : b; }
inline int imin2(int a, int b) { return a < b ? a : b; }

// fftmx() of R's fft.c; `nfac` is overwritten (hence passed by copy).
void fftmx(double* a, double* b, int ntot, int n, int nspan, int isn, int m, int kt, double* at, double* ck,
           double* bt, double* sk, int* np, int* nfac) {
    double aa, aj, ajm, ajp, ak, akm, akp;
    double bb, bj, bjm, bjp, bk, bkm, bkp;
    double c1, c2 = 0, c3 = 0, c72, cd;
    double dr, rad;
    double s1, s120, s2 = 0, s3 = 0, s72, sd;
    int i, inc, j, jc, jf, jj;
    int k, k1, k2, k3 = 0, k4, kk, klim, ks, kspan, kspnn;
    int lim, maxf, mm, nn, nt;

    a--;
    b--;
    at--;
    ck--;
    bt--;
    sk--;
    np--;

    inc = std::abs(isn);
    nt = inc * ntot;
    ks = inc * nspan;
    rad = 0.785398163397448309615660845820; /* M_PI_4 */
    s72 = rad / 0.625;
    c72 = lsm::r_cos(s72);
    s72 = lsm::r_sin(s72);
    s120 = 0.5 * 1.732050807568877293527446341506; /* M_SQRT_3 */
    if (isn <= 0) {
        s72 = -s72;
        s120 = -s120;
        rad = -rad;
    }

    kspan = ks;
    nn = nt - inc;
    jc = ks / n;

    lim = 32;
    klim = lim * jc;
    i = 0;
    jf = 0;
    maxf = nfac[m - kt - 1];
    if (kt > 0) maxf = imax2(nfac[kt - 1], maxf);

L_start:
    dr = (8.0 * jc) / kspan;
    cd = lsm::r_sin(0.5 * dr * rad);
    cd = 2.0 * cd * cd;
    sd = lsm::r_sin(dr * rad);
    kk = 1;
    i++;
    if (nfac[i - 1] != 2) goto L110;

    kspan /= 2;
    k1 = kspan + 2;
    do {
        do {
            k2 = kk + kspan;
            ak = a[k2];
            bk = b[k2];
            a[k2] = a[kk] - ak;
            b[k2] = b[kk] - bk;
            a[kk] += ak;
            b[kk] += bk;
            kk = k2 + kspan;
        } while (kk <= nn);
        kk -= nn;
    } while (kk <= jc);

    if (kk > kspan) goto L_fin;
L60:
    c1 = 1.0 - cd;
    s1 = sd;
    mm = imin2(k1 / 2, klim);
    goto L80;

L70:
    ak = c1 - (cd * c1 + sd * s1);
    s1 = (sd * c1 - cd * s1) + s1;
    c1 = ak;

L80:
    do {
        k2 = kk + kspan;
        ak = a[kk] - a[k2];
        bk = b[kk] - b[k2];
        a[kk] += a[k2];
        b[kk] += b[k2];
        a[k2] = c1 * ak - s1 * bk;
        b[k2] = s1 * ak + c1 * bk;
        kk = k2 + kspan;
    } while (kk < nt);
    k2 = kk - nt;
    c1 = -c1;
    kk = k1 - k2;
    if (kk > k2) goto L80;
    kk += jc;
    if (kk <= mm) goto L70;
    if (kk >= k2) {
        k1 = k1 + inc + inc;
        kk = (k1 - kspan) / 2 + jc;
        if (kk <= jc + jc) goto L60;
        goto L_start;
    }

    s1 = ((kk - 1) / jc) * dr * rad;
    c1 = lsm::r_cos(s1);
    s1 = lsm::r_sin(s1);
    mm = imin2(k1 / 2, mm + klim);
    goto L80;

L100:
    k1 = kk + kspan;
    k2 = k1 + kspan;
    ak = a[kk];
    bk = b[kk];
    aj = a[k1] + a[k2];
    bj = b[k1] + b[k2];
    a[kk] = ak + aj;
    b[kk] = bk + bj;
    ak = -0.5 * aj + ak;
    bk = -0.5 * bj + bk;
    aj = (a[k1] - a[k2]) * s120;
    bj = (b[k1] - b[k2]) * s120;
    a[k1] = ak - bj;
    b[k1] = bk + aj;
    a[k2] = ak + bj;
    b[k2] = bk - aj;
    kk = k2 + kspan;
    if (kk < nn) goto L100;
    kk = kk - nn;
    if (kk <= kspan) goto L100;
    goto L290;

L110:
    if (nfac[i - 1] != 4) goto L_f_odd;
    kspnn = kspan;
    kspan /= 4;
L120:
    c1 = 1.0;
    s1 = 0;
    mm = imin2(kspan, klim);
    goto L150;
L130:
    c2 = c1 - (cd * c1 + sd * s1);
    s1 = (sd * c1 - cd * s1) + s1;
    c1 = c2;

L140:
    c2 = c1 * c1 - s1 * s1;
    s2 = c1 * s1 * 2.0;
    c3 = c2 * c1 - s2 * s1;
    s3 = c2 * s1 + s2 * c1;

L150:
    k1 = kk + kspan;
    k2 = k1 + kspan;
    k3 = k2 + kspan;
    akp = a[kk] + a[k2];
    akm = a[kk] - a[k2];
    ajp = a[k1] + a[k3];
    ajm = a[k1] - a[k3];
    a[kk] = akp + ajp;
    ajp = akp - ajp;
    bkp = b[kk] + b[k2];
    bkm = b[kk] - b[k2];
    bjp = b[k1] + b[k3];
    bjm = b[k1] - b[k3];
    b[kk] = bkp + bjp;
    bjp = bkp - bjp;
    if (isn < 0) goto L180;
    akp = akm - bjm;
    akm = akm + bjm;
    bkp = bkm + ajm;
    bkm = bkm - ajm;
    if (s1 == 0.0) goto L190;
L160:
    a[k1] = akp * c1 - bkp * s1;
    b[k1] = akp * s1 + bkp * c1;
    a[k2] = ajp * c2 - bjp * s2;
    b[k2] = ajp * s2 + bjp * c2;
    a[k3] = akm * c3 - bkm * s3;
    b[k3] = akm * s3 + bkm * c3;
    kk = k3 + kspan;
    if (kk <= nt) goto L150;
L170:
    kk = kk - nt + jc;
    if (kk <= mm) goto L130;
    if (kk < kspan) goto L200;
    kk = kk - kspan + inc;
    if (kk <= jc) goto L120;
    if (kspan == jc) goto L_fin;
    goto L_start;
L180:
    akp = akm + bjm;
    akm = akm - bjm;
    bkp = bkm - ajm;
    bkm = bkm + ajm;
    if (s1 != 0.0) goto L160;
L190:
    a[k1] = akp;
    b[k1] = bkp;
    a[k2] = ajp;
    b[k2] = bjp;
    a[k3] = akm;
    b[k3] = bkm;
    kk = k3 + kspan;
    if (kk <= nt) goto L150;
    goto L170;
L200:
    s1 = ((kk - 1) / jc) * dr * rad;
    c1 = lsm::r_cos(s1);
    s1 = lsm::r_sin(s1);
    mm = imin2(kspan, mm + klim);
    goto L140;

L_f5:
    c2 = c72 * c72 - s72 * s72;
    s2 = 2.0 * c72 * s72;
L220:
    k1 = kk + kspan;
    k2 = k1 + kspan;
    k3 = k2 + kspan;
    k4 = k3 + kspan;
    akp = a[k1] + a[k4];
    akm = a[k1] - a[k4];
    bkp = b[k1] + b[k4];
    bkm = b[k1] - b[k4];
    ajp = a[k2] + a[k3];
    ajm = a[k2] - a[k3];
    bjp = b[k2] + b[k3];
    bjm = b[k2] - b[k3];
    aa = a[kk];
    bb = b[kk];
    a[kk] = aa + akp + ajp;
    b[kk] = bb + bkp + bjp;
    ak = akp * c72 + ajp * c2 + aa;
    bk = bkp * c72 + bjp * c2 + bb;
    aj = akm * s72 + ajm * s2;
    bj = bkm * s72 + bjm * s2;
    a[k1] = ak - bj;
    a[k4] = ak + bj;
    b[k1] = bk + aj;
    b[k4] = bk - aj;
    ak = akp * c2 + ajp * c72 + aa;
    bk = bkp * c2 + bjp * c72 + bb;
    aj = akm * s2 - ajm * s72;
    bj = bkm * s2 - bjm * s72;
    a[k2] = ak - bj;
    a[k3] = ak + bj;
    b[k2] = bk + aj;
    b[k3] = bk - aj;
    kk = k4 + kspan;
    if (kk < nn) goto L220;
    kk = kk - nn;
    if (kk <= kspan) goto L220;
    goto L290;

L_f_odd:
    k = nfac[i - 1];
    kspnn = kspan;
    kspan /= k;
    if (k == 3) goto L100;
    if (k == 5) goto L_f5;
    if (k == jf) goto L250;
    jf = k;
    s1 = rad / (k / 8.0);
    c1 = lsm::r_cos(s1);
    s1 = lsm::r_sin(s1);
    ck[jf] = 1.0;
    sk[jf] = 0.0;

    for (j = 1; j < k; j++) {
        ck[j] = ck[k] * c1 + sk[k] * s1;
        sk[j] = ck[k] * s1 - sk[k] * c1;
        k--;
        ck[k] = ck[j];
        sk[k] = -sk[j];
    }

L250:
    k1 = kk;
    k2 = kk + kspnn;
    aa = a[kk];
    bb = b[kk];
    ak = aa;
    bk = bb;
    j = 1;
    k1 = k1 + kspan;
L260:
    k2 = k2 - kspan;
    j++;
    at[j] = a[k1] + a[k2];
    ak = at[j] + ak;
    bt[j] = b[k1] + b[k2];
    bk = bt[j] + bk;
    j++;
    at[j] = a[k1] - a[k2];
    bt[j] = b[k1] - b[k2];
    k1 = k1 + kspan;
    if (k1 < k2) goto L260;
    a[kk] = ak;
    b[kk] = bk;
    k1 = kk;
    k2 = kk + kspnn;
    j = 1;
L270:
    k1 += kspan;
    k2 -= kspan;
    jj = j;
    ak = aa;
    bk = bb;
    aj = 0.0;
    bj = 0.0;
    for (k = 2; k < jf; k++) {
        ak += at[k] * ck[jj];
        bk += bt[k] * ck[jj];
        k++;
        aj += at[k] * sk[jj];
        bj += bt[k] * sk[jj];
        jj += j;
        if (jj > jf) jj -= jf;
    }
    k = jf - j;
    a[k1] = ak - bj;
    b[k1] = bk + aj;
    a[k2] = ak + bj;
    b[k2] = bk - aj;
    j++;
    if (j < k) goto L270;
    kk = kk + kspnn;
    if (kk <= nn) goto L250;
    kk = kk - nn;
    if (kk <= kspan) goto L250;

L290:
    if (i == m) goto L_fin;
    kk = jc + 1;
L300:
    c2 = 1.0 - cd;
    s1 = sd;
    mm = imin2(kspan, klim);

    do {
        c1 = c2;
        s2 = s1;
        kk += kspan;
        do {
            do {
                ak = a[kk];
                a[kk] = c2 * ak - s2 * b[kk];
                b[kk] = s2 * ak + c2 * b[kk];
                kk += kspnn;
            } while (kk <= nt);
            ak = s1 * s2;
            s2 = s1 * c2 + c1 * s2;
            c2 = c1 * c2 - ak;
            kk += -nt + kspan;
        } while (kk <= kspnn);
        kk += -kspnn + jc;
        if (kk <= mm) {
            c2 = c1 - (cd * c1 + sd * s1);
            s1 = s1 + (sd * c1 - cd * s1);
            continue;
        }
        if (kk >= kspan) {
            kk = kk - kspan + jc + inc;
            if (kk <= jc + jc) goto L300;
            goto L_start;
        }
        s1 = ((kk - 1) / jc) * dr * rad;
        c2 = lsm::r_cos(s1);
        s1 = lsm::r_sin(s1);
        mm = imin2(kspan, mm + klim);
    } while (1);

L_fin:
    np[1] = ks;
    if (kt == 0) goto L440;
    k = kt + kt + 1;
    if (m < k) k--;
    np[k + 1] = jc;
    for (j = 1; j < k; j++, k--) {
        np[j + 1] = np[j] / nfac[j - 1];
        np[k] = np[k + 1] * nfac[j - 1];
    }
    k3 = np[k + 1];
    kspan = np[2];
    kk = jc + 1;
    k2 = kspan + 1;
    j = 1;

    if (n == ntot) {
    L370:
        do {
            ak = a[kk];
            a[kk] = a[k2];
            a[k2] = ak;
            bk = b[kk];
            b[kk] = b[k2];
            b[k2] = bk;
            kk += inc;
            k2 += kspan;
        } while (k2 < ks);
    L380:
        do {
            k2 -= np[j];
            j++;
            k2 += np[j + 1];
        } while (k2 > np[j]);
        j = 1;
        do {
            if (kk < k2) goto L370;
            kk += inc;
            k2 += kspan;
        } while (k2 < ks);
        if (kk < ks) goto L380;
        jc = k3;
    } else {
    L400:
        k = kk + jc;
        do {
            ak = a[kk];
            a[kk] = a[k2];
            a[k2] = ak;
            bk = b[kk];
            b[kk] = b[k2];
            b[k2] = bk;
            kk += inc;
            k2 += inc;
        } while (kk < k);
        kk += ks - jc;
        k2 += ks - jc;
        if (kk < nt) goto L400;
        k2 += -nt + kspan;
        kk += -nt + jc;
        if (k2 < ks) goto L400;

        do {
            do {
                k2 -= np[j];
                j++;
                k2 += np[j + 1];
            } while (k2 > np[j]);
            j = 1;
            do {
                if (kk < k2) goto L400;
                kk += jc;
                k2 += kspan;
            } while (k2 < ks);
        } while (kk < ks);
        jc = k3;
    }

L440:
    if (2 * kt + 1 >= m) return;
    kspnn = np[kt + 1];

    nn = m - kt;
    nfac[nn] = 1;
    for (j = nn; j > kt; j--) nfac[j - 1] *= nfac[j];
    kt++;
    nn = nfac[kt - 1] - 1;
    jj = 0;
    j = 0;
    goto L480;
L460:
    jj -= k2;
    k2 = kk;
    k++;
    kk = nfac[k - 1];
L470:
    jj += kk;
    if (jj >= k2) goto L460;
    np[j] = jj;
L480:
    k2 = nfac[kt - 1];
    k = kt + 1;
    kk = nfac[k - 1];
    j++;
    if (j <= nn) goto L470;

    j = 0;
    goto L500;

    do {
        do {
            k = kk;
            kk = np[k];
            np[k] = -kk;
        } while (kk != j);
        k3 = kk;
    L500:
        do {
            j++;
            kk = np[j];
        } while (kk < 0);
    } while (kk != j);
    np[j] = -j;
    if (j != nn) goto L500;
    maxf *= inc;
    goto L570;

L_ord:
    do j--;
    while (np[j] < 0);
    jj = jc;

L520:
    kspan = imin2(jj, maxf);
    jj -= kspan;
    k = np[j];
    kk = jc * k + i + jj;

    for (k1 = kk + kspan, k2 = 1; k1 != kk; k1 -= inc, k2++) {
        at[k2] = a[k1];
        bt[k2] = b[k1];
    }

    do {
        k1 = kk + kspan;
        k2 = k1 - jc * (k + np[k]);
        k = -np[k];
        do {
            a[k1] = a[k2];
            b[k1] = b[k2];
            k1 -= inc;
            k2 -= inc;
        } while (k1 != kk);
        kk = k2;
    } while (k != j);

    for (k1 = kk + kspan, k2 = 1; k1 > kk; k1 -= inc, k2++) {
        a[k1] = at[k2];
        b[k1] = bt[k2];
    }

    if (jj != 0) goto L520;
    if (j != 1) goto L_ord;

L570:
    j = k3 + 1;
    nt = nt - kspnn;
    i = nt - inc + 1;
    if (nt >= 0) goto L_ord;
}

}  // namespace

RFftPlan::RFftPlan(int n_) : n(n_) {
    // fft_factor()
    int j, jj, k, sqrtk, kchanged;
    if (n <= 0) throw std::invalid_argument("fft: invalid length");
    m_fac = 0;
    kt = 0;
    maxf = 0;
    maxp = 0;
    k = n;
    if (k == 1) return;
    while (k % 16 == 0) {
        nfac[m_fac++] = 4;
        k /= 16;
    }
    kchanged = 0;
    sqrtk = static_cast<int>(std::sqrt(static_cast<double>(k)));
    for (j = 3; j <= sqrtk; j += 2) {
        jj = j * j;
        while (k % jj == 0) {
            nfac[m_fac++] = j;
            k /= jj;
            kchanged = 1;
        }
        if (kchanged) {
            kchanged = 0;
            sqrtk = static_cast<int>(std::sqrt(static_cast<double>(k)));
        }
    }
    if (k <= 4) {
        kt = m_fac;
        nfac[m_fac] = k;
        if (k != 1) m_fac++;
    } else {
        if (k % 4 == 0) {
            nfac[m_fac++] = 2;
            k /= 4;
        }
        kt = m_fac;
        maxp = imax2(kt + kt + 2, k - 1);
        j = 2;
        do {
            if (k % j == 0) {
                nfac[m_fac++] = j;
                k /= j;
            }
            if (j > INT_MAX - 2) break;
            j = ((j + 1) / 2) * 2 + 1;
        } while (j <= k);
    }
    if (m_fac <= kt + 1) maxp = m_fac + kt + 1;
    if (m_fac + kt > 20) throw std::invalid_argument("fft factorization error");
    if (kt != 0) {
        j = kt;
        while (j != 0) nfac[m_fac++] = nfac[--j];
    }
    maxf = nfac[m_fac - kt - 1];
    if (kt > 0) maxf = imax2(nfac[kt - 1], maxf);
    if (kt > 1) maxf = imax2(nfac[kt - 2], maxf);
    if (kt > 2) maxf = imax2(nfac[kt - 3], maxf);
}

void RFftPlan::run(double* z, int nseg, int nspn, bool inverse) const {
    // fft_work(&z[0].r, &z[0].i, nseg, n, nspn, inverse ? 2 : -2, ...)
    if (n == 1) return;
    std::vector<double> work(static_cast<std::size_t>(4 * std::max(maxf, 1)));
    std::vector<int> iwork(static_cast<std::size_t>(std::max(maxp, 1)));
    int fac[20];
    std::copy(nfac, nfac + 20, fac);
    const std::size_t mf = static_cast<std::size_t>(maxf);
    const int nspan = n * nspn, ntot = nspan * nseg;
    fftmx(z, z + 1, ntot, n, nspan, inverse ? 2 : -2, m_fac, kt, work.data(), work.data() + mf,
          work.data() + 2 * mf, work.data() + 3 * mf, iwork.data(), fac);
}

void r_fft2(std::vector<double>& z, int nrow, int ncol, bool inverse, bool parallel) {
    // fft(z) of an nrow x ncol complex matrix (interleaved re/im, column-major).
    if (static_cast<i64>(z.size()) != 2 * static_cast<i64>(nrow) * ncol) throw std::invalid_argument("fft: bad size");
    if (!parallel) {
        int nseg = nrow * ncol, n = 1, nspn = 1;
        for (int d : {nrow, ncol}) {
            if (d > 1) {
                nspn *= n;
                n = d;
                nseg /= n;
                RFftPlan(n).run(z.data(), nseg, nspn, inverse);
            }
        }
        return;
    }
    if (nrow > 1) {
        const RFftPlan plan(nrow);
        lsm::parallel_for_dynamic(true, ncol, [&](i64 c) {
            plan.run(z.data() + 2 * static_cast<std::size_t>(c) * nrow, 1, 1, inverse);
        });
    }
    if (ncol > 1) {
        const RFftPlan plan(ncol);
        lsm::parallel_for_dynamic(true, nrow, [&](i64 r) {
            std::vector<double> buf(2 * static_cast<std::size_t>(ncol));
            for (int c = 0; c < ncol; ++c) {
                const std::size_t at = 2 * (static_cast<std::size_t>(c) * nrow + static_cast<std::size_t>(r));
                buf[2 * static_cast<std::size_t>(c)] = z[at];
                buf[2 * static_cast<std::size_t>(c) + 1] = z[at + 1];
            }
            plan.run(buf.data(), 1, 1, inverse);
            for (int c = 0; c < ncol; ++c) {
                const std::size_t at = 2 * (static_cast<std::size_t>(c) * nrow + static_cast<std::size_t>(r));
                z[at] = buf[2 * static_cast<std::size_t>(c)];
                z[at + 1] = buf[2 * static_cast<std::size_t>(c) + 1];
            }
        });
    }
}

}  // namespace nlm
