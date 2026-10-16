// SPDX-License-Identifier: GPL-3.0-or-later
//
// Climate: measured series (cli_pmeas, cli_tmeas, cli_smeas, cli_hmeas, cli_wmeas,
// cli_petmeas), the weather generator (cli_wgnread, gcycl, cli_initwgn), stations
// (cli_staread, search) and the daily weather (climate_control, cli_precip_control,
// cli_pgen, cli_weatgn, cli_tgen, cli_clgen, cli_slrgen, cli_rhgen, cli_wndgen,
// cli_bounds_check, atri, cli_dstn1, ee).

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

// The weather path prefix (TRIM(ADJUSTL(path)) // name), or the bare name.
std::string climate_path(const std::string& path, const std::string& name) {
    if (path == "null") return name;
    std::size_t a = path.find_first_not_of(' ');
    if (a == std::string::npos) return name;
    std::size_t b = path.find_last_not_of(' ');
    return path.substr(a, b - a + 1) + name;
}

// Count the records after the title and header (the "imax" loop of the readers).
int count_records(fio::File& f) {
    std::string titldum;
    int imax = 0;
    int eof = 0;
    while (eof == 0) {
        eof = f.read(chars(titldum, 80));
        if (eof < 0) break;
        ++imax;
    }
    return imax;
}

// cli_pmeas.f90
void cli_pmeas(Sim& s) {
    const std::string& cli = s.in_cli.pcp_cli;
    if (!s.files.exists(cli) || cli == "null") {
        s.pcp.alloc(0, 0, climate_measured_data());
        s.pcp_n.alloc(1, 0, "");
        s.db_mx.pcpfiles = 0;
        return;
    }
    std::string titldum, header;
    int imax = 0;
    for (;;) {
        fio::File f = s.files.open(cli);
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        imax = count_records(f);
        s.pcp.alloc(0, imax, climate_measured_data());
        s.pcp_n.alloc(1, imax, "");
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i)
            if (f.read(chars(s.pcp_n(i), 50)) < 0) break;
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i) {
            climate_measured_data& m = s.pcp(i);
            if (f.read(chars(m.filename, 50)) < 0) break;
            fio::File g = s.files.open(climate_path(s.in_path_pcp.pcp, m.filename));
            if (!g.open()) throw std::runtime_error("cannot open precipitation file " + m.filename);
            if (g.read(chars(titldum, 80)) < 0) break;
            if (g.read(chars(header, 80)) < 0) break;
            if (g.read(m.nbyr, m.tstep, m.lat, m.long_, m.elev) < 0) break;
            if (m.tstep > 0) throw std::runtime_error("SWAT+ sub-daily precipitation is not supported");
            m.ts.alloc(366, m.nbyr, 0.0f);
            int iyr = 0, istep = 0;
            if (g.read(iyr, istep) < 0) break;
            m.start_day = istep;
            m.start_yr = iyr;
            g.backspace();
            m.yrs_start = iyr > s.time.yrc ? iyr - s.time.yrc : 0;
            for (;;) {
                if (g.read(iyr, istep) < 0) break;
                if (iyr >= s.time.yrc && istep >= s.time.day_start) break;
            }
            g.backspace();
            int iyr_prev = iyr;
            int iyrs = 1;
            for (;;) {
                float v = 0.0f;
                const int eof = g.read(iyr, istep, v);
                if (eof < 0) break;
                m.ts(istep, iyrs) = v;
                if (istep == 365 || istep == 366) {
                    if (g.read(iyr, istep) < 0) break;
                    g.backspace();
                    if (iyr != iyr_prev) {
                        iyr_prev = iyr;
                        ++iyrs;
                    }
                }
            }
            m.end_day = istep;
            m.end_yr = iyr;
        }
        break;
    }
    s.db_mx.pcpfiles = imax;
}

// cli_tmeas.f90: maximum and minimum temperatures, with the monthly means
void cli_tmeas(Sim& s) {
    const std::string& cli = s.in_cli.tmp_cli;
    if (!s.files.exists(cli) || cli == "null") {
        s.tmp.alloc(0, 0, climate_measured_data());
        s.tmp_n.alloc(1, 0, "");
        s.db_mx.tmpfiles = 0;
        return;
    }
    std::string titldum, header;
    int imax = 0;
    for (;;) {
        fio::File f = s.files.open(cli);
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        imax = count_records(f);
        s.tmp.alloc(0, imax, climate_measured_data());
        s.tmp_n.alloc(1, imax, "");
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i)
            if (f.read(chars(s.tmp_n(i), 50)) < 0) break;
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i) {
            climate_measured_data& m = s.tmp(i);
            if (f.read(chars(m.filename, 50)) < 0) break;
            Fix<int, 1, 12> num_tot;
            fio::File g = s.files.open(climate_path(s.in_path_tmp.tmp, m.filename));
            if (!g.open()) throw std::runtime_error("cannot open temperature file " + m.filename);
            if (g.read(chars(titldum, 80)) < 0) break;
            if (g.read(chars(header, 80)) < 0) break;
            if (g.read(m.nbyr, m.tstep, m.lat, m.long_, m.elev) < 0) break;
            m.ts.alloc(366, m.nbyr, 0.0f);
            m.ts2.alloc(366, m.nbyr, 0.0f);
            int iyr = 0, istep = 0;
            if (g.read(iyr, istep) < 0) break;
            m.start_day = istep;
            m.start_yr = iyr;
            g.backspace();
            m.yrs_start = iyr > s.time.yrc ? iyr - s.time.yrc : 0;
            for (;;) {
                float tempx = 0.0f, tempn = 0.0f;
                if (g.read(iyr, istep, tempx, tempn) < 0) break;
                if (iyr >= s.time.yrc && istep >= s.time.day_start) break;
            }
            g.backspace();
            int iyr_prev = iyr;
            int iyrs = 1;
            // the reference sets the module's month table to this year's (xmon below and
            // cli_initwgn's month lengths then follow the last year read)
            s.ndays = (iyr % 4 == 0) ? s.ndays_leap : s.ndays_noleap;
            int mo = 0, day_mo = 0;
            for (;;) {
                float vx = 0.0f, vn = 0.0f;
                const int eof = g.read(iyr, istep, vx, vn);
                if (eof < 0) break;
                m.ts(istep, iyrs) = vx;
                m.ts2(istep, iyrs) = vn;
                xmon(s, istep, mo, day_mo);
                num_tot(mo) += 1;
                m.max_mon(mo) = m.max_mon(mo) + m.ts(istep, iyrs);
                m.min_mon(mo) = m.min_mon(mo) + m.ts2(istep, iyrs);
                if (istep == 365 || istep == 366) {
                    if (g.read(iyr, istep) < 0) break;
                    g.backspace();
                    if (iyr != iyr_prev) {
                        iyr_prev = iyr;
                        ++iyrs;
                        s.ndays = (iyr % 4 == 0) ? s.ndays_leap : s.ndays_noleap;
                    }
                }
            }
            m.end_day = istep;
            m.end_yr = iyr;
            // the reference divides every month by the count of the last one read
            for (int k = 1; k <= 12; ++k) {
                m.max_mon(k) =
                    m.max_mon(k) / static_cast<float>(num_tot(mo));
                m.min_mon(k) =
                    m.min_mon(k) / static_cast<float>(num_tot(mo));
            }
        }
        break;
    }
    s.db_mx.tmpfiles = imax;
}

// cli_smeas.f90, cli_hmeas.f90, cli_wmeas.f90, cli_petmeas.f90: one value a day
void cli_one_meas(Sim& s, const std::string& cli, const std::string& path, Vec<climate_measured_data>& arr,
                  Vec<std::string>& names, int& nfiles) {
    if (!s.files.exists(cli) || cli == "null") {
        arr.alloc(0, 0, climate_measured_data());
        names.alloc(1, 0, "");
        nfiles = 0;
        return;
    }
    std::string titldum, header;
    int imax = 0;
    for (;;) {
        fio::File f = s.files.open(cli);
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        imax = count_records(f);
        arr.alloc(0, imax, climate_measured_data());
        names.alloc(1, imax, "");
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i)
            if (f.read(chars(names(i), 50)) < 0) break;
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i) {
            climate_measured_data& m = arr(i);
            if (f.read(chars(m.filename, 50)) < 0) break;
            fio::File g = s.files.open(climate_path(path, m.filename));
            if (!g.open()) throw std::runtime_error("cannot open climate file " + m.filename);
            if (g.read(chars(titldum, 80)) < 0) break;
            if (g.read(chars(header, 80)) < 0) break;
            if (g.read(m.nbyr, m.tstep, m.lat, m.long_, m.elev) < 0) break;
            m.ts.alloc(366, m.nbyr, 0.0f);
            int iyr = 0, istep = 0;
            if (g.read(iyr, istep) < 0) break;
            m.start_day = istep;
            m.start_yr = iyr;
            g.backspace();
            m.yrs_start = iyr > s.time.yrc ? iyr - s.time.yrc : 0;
            for (;;) {
                if (g.read(iyr, istep) < 0) break;
                if (iyr >= s.time.yrc && istep >= s.time.day_start) break;
            }
            g.backspace();
            int iyr_prev = iyr;
            int iyrs = 1;
            for (;;) {
                float v = 0.0f;
                if (g.read(iyr, istep, v) < 0) break;
                m.ts(istep, iyrs) = v;
                if (istep == 365 || istep == 366) {
                    if (g.read(iyr, istep) < 0) break;
                    g.backspace();
                    if (iyr != iyr_prev) {
                        iyr_prev = iyr;
                        ++iyrs;
                    }
                }
            }
            m.end_day = istep;
            m.end_yr = iyr;
        }
        break;
    }
    nfiles = imax;
}

float cli_dstn1(float rn1, float rn2) { return std::sqrt(-2.0f * std::log(rn1)) * std::cos(6.283185f * rn2); }

float ee(float tk) {
    float e = 0.0f;
    if (tk + 237.3f != 0.0f) {
        e = (16.78f * tk - 116.9f) / (tk + 237.3f);
        e = std::exp(e);
    }
    return e;
}

float atri(float at1, float at2, float at3, int& at4i) {
    const float u3 = at2 - at1;
    const float rn = aunif(at4i);
    const float y = 2.0f / (at3 - at1);
    const float b2 = at3 - at2;
    const float b1 = rn / y;
    const float x1 = y * u3 / 2.0f;
    float out = 0.0f;
    if (rn <= x1) {
        const float xx = 2.0f * b1 * u3;
        const float yy = xx <= 0.0f ? 0.0f : std::sqrt(xx);
        out = yy + at1;
    } else {
        const float xx = b2 * b2 - 2.0f * b2 * (b1 - 0.5f * u3);
        const float yy = xx <= 0.0f ? 0.0f : std::sqrt(xx);
        out = at3 - yy;
    }
    const float amn = (at3 + at2 + at1) / 3.0f;
    out = out * at2 / amn;
    if (out >= 1.0f) out = 0.99f;
    if (out <= 0.0f) out = 0.001f;
    return out;
}

// gcycl.f90
void gcycl(Sim& s) {
    for (int k = 1; k <= 9; ++k) s.idg(k) = k;
    for (int iwgn = 1; iwgn <= s.db_mx.wgnsta; ++iwgn) {
        s.rndseed(1, iwgn) = 748932582;
        s.rndseed(2, iwgn) = 1985072130;
        s.rndseed(3, iwgn) = 1631331038;
        s.rndseed(4, iwgn) = 67377721;
        s.rndseed(5, iwgn) = 366304404;
        s.rndseed(6, iwgn) = 1094585182;
        s.rndseed(7, iwgn) = 1767585417;
        s.rndseed(8, iwgn) = 1980520317;
        s.rndseed(9, iwgn) = 392682216;
    }
    int rndseed10 = 64298628;
    if (s.bsn_prm.igen != 0) {
        for (int j = 1; j <= 9; ++j) {
            const float rn = aunif(rndseed10);
            const int ii = static_cast<int>(static_cast<float>(100 * s.bsn_prm.igen) * rn);
            for (int k = 1; k <= ii; ++k) (void)aunif(rndseed10);
            s.rndseed(j, 1) = rndseed10;
        }
        s.rndseed_cond = rndseed10;
        for (int j = 9; j >= 1; --j) {
            const int ii = s.idg(j);
            const float rn = aunif(rndseed10);
            const int k = static_cast<int>(static_cast<float>(j) * rn + 1.0f);
            s.idg(j) = s.idg(k);
            s.idg(k) = ii;
        }
    }
    for (int iwgn = 1; iwgn <= s.db_mx.wgnsta; ++iwgn) s.rndseed(10, iwgn) = s.rndseed(s.idg(6), iwgn);
    for (int iwgn = 1; iwgn <= s.db_mx.wgnsta; ++iwgn) {
        s.rnd2(iwgn) = aunif(s.rndseed(s.idg(2), iwgn));
        s.rnd3(iwgn) = aunif(s.rndseed(s.idg(3), iwgn));
        s.rnd8(iwgn) = aunif(s.rndseed(s.idg(8), iwgn));
        s.rnd9(iwgn) = aunif(s.rndseed(s.idg(9), iwgn));
    }
}

// cli_initwgn.f90
void cli_initwgn(Sim& s, int iwgn) {
    weather_generator_db& g = s.wgn(iwgn);
    wgn_parms& p = s.wgn_pms(iwgn);
    auto nd = [&](int k) { return s.ndays(k); };
    float x1 = 0.0f, x2 = 0.0f;
    const float xx = g.lat / 57.296f;
    p.latsin = std::sin(xx);
    p.latcos = std::cos(xx);
    const float lattan = std::tan(xx);
    x1 = 0.4348f * std::abs(lattan);
    if (x1 < 1.0f) x2 = std::acos(x1);
    p.daylmn = 7.6394f * x2;
    float dl = 0.0f;
    if (s.bsn_prm.dorm_hr < 1.e-6f) {
        dl = 0.0f;
        if (std::abs(g.lat) > 40.0f) {
            dl = 1.0f;
        } else if (std::abs(g.lat) < 20.0f) {
            dl = -1.0f;
        } else {
            dl = (std::abs(g.lat) - 20.0f) / 20.0f;
        }
    } else {
        dl = s.bsn_prm.dorm_hr;
    }
    p.daylth = dl;
    Fix<float, 1, 12> rain_hhsm;
    rain_hhsm(1) = (g.rainhmx(12) + g.rainhmx(1) + g.rainhmx(2)) / 3.0f;
    for (int mon = 2; mon <= 11; ++mon)
        rain_hhsm(mon) = (g.rainhmx(mon - 1) + g.rainhmx(mon) + g.rainhmx(mon + 1)) / 3.0f;
    rain_hhsm(12) = (g.rainhmx(11) + g.rainhmx(12) + g.rainhmx(1)) / 3.0f;
    float summx_t = 0.0f, summn_t = 0.0f, summm_p = 0.0f, summm_pet = 0.0f;
    float tmin = 100.0f, tmax = 0.0f;
    for (int mon = 1; mon <= 12; ++mon) {
        const int mdays = nd(mon + 1) - nd(mon);
        float tav = (g.tmpmx(mon) + g.tmpmn(mon)) / 2.0f;
        if (tav > tmax) tmax = tav;
        if (tav < tmin) tmin = tav;
        summx_t = summx_t + g.tmpmx(mon);
        summn_t = summn_t + g.tmpmn(mon);
        if (tav > 0.0f) p.phutot = p.phutot + tav * static_cast<float>(mdays);
        if (g.pr_ww(mon) <= g.pr_wd(mon) || g.pr_wd(mon) <= 0.0f) {
            if (g.pcpd(mon) < 0.1f) g.pcpd(mon) = 0.1f;
            g.pr_wd(mon) = 0.75f * g.pcpd(mon) / static_cast<float>(mdays);
            g.pr_ww(mon) = 0.25f + g.pr_wd(mon);
        } else {
            g.pcpd(mon) = static_cast<float>(mdays) * g.pr_wd(mon) / (1.0f - g.pr_ww(mon) + g.pr_wd(mon));
        }
        if (g.pcpd(mon) <= 0.0f) g.pcpd(mon) = 0.001f;
        p.pr_wdays(mon) = g.pcpd(mon) / static_cast<float>(mdays);
        p.pcpmean(mon) = g.pcpmm(mon) / g.pcpd(mon);
        if (g.pcpskw(mon) < 0.2f) g.pcpskw(mon) = 0.2f;
        summm_p = summm_p + g.pcpmm(mon);
        p.pcpdays = p.pcpdays + g.pcpd(mon);
        tav = (g.tmpmx(mon) + g.tmpmn(mon)) / 2.0f;
        const float tk = tav + 273.0f;
        const float alb = 0.15f;
        const float d = std::exp(21.255f - 5304.0f / tk) * 5304.0f / fpowi(tk, 2);
        const float gma = d / (d + 0.68f);
        const float ho = 23.9f * g.solarav(mon) * (1.0f - alb) / 58.3f;
        const float aph = 1.28f;
        p.pet(mon) = aph * ho * gma * static_cast<float>(nd(mon + 1) - nd(mon));
        summm_pet = summm_pet + p.pet(mon);
    }
    p.idewpt = 1;
    for (int mon = 1; mon <= 12; ++mon) {
        if (g.dewpt(mon) > 1.0f || g.dewpt(mon) < 0.0f) {
            p.idewpt = 0;
            break;
        }
    }
    s.ppet_ndays = 30;
    p.mne_ppet.alloc(1, s.ppet_ndays, 0);
    p.precip_mce.alloc(1, s.ppet_ndays, 0.0f);
    p.pet_mce.alloc(1, s.ppet_ndays, 0.0f);
    for (int inext = 1; inext <= s.ppet_ndays; ++inext) p.mne_ppet(inext) = inext;
    const int mo_ppet = s.time.mo == 1 ? 12 : s.time.mo - 1;
    p.precip_sum = 0.0f;
    p.pet_sum = 0.0f;
    for (int inext = 1; inext <= s.ppet_ndays; ++inext) {
        p.precip_mce(inext) = g.pcpmm(mo_ppet) / static_cast<float>(nd(mo_ppet + 1) - nd(mo_ppet));
        p.pet_mce(inext) = p.pet(mo_ppet) / static_cast<float>(nd(mo_ppet + 1) - nd(mo_ppet));
        p.precip_sum = p.precip_sum + p.precip_mce(inext);
        p.pet_sum = p.pet_sum + p.pet_mce(inext);
    }
    p.pcp_an = summm_p;
    p.ppet_an = summm_pet > 1.e-3f ? summm_p / summm_pet : summm_p / 1.e-3f;
    p.tmp_an = (summx_t + summn_t) / 24.0f;
    int xrnd = s.rndseed(s.idg(3), iwgn);
    float rndm1 = aunif(xrnd);
    for (int mon = 1; mon <= 12; ++mon) {
        const float r6 = g.pcpskw(mon) / 6.0f;
        float sum = 0.0f;
        for (int j = 1; j <= 1000; ++j) {
            const float rnm2 = aunif(xrnd);
            float xlv = (cli_dstn1(rndm1, rnm2) - r6) * r6 + 1.0f;
            rndm1 = rnm2;
            xlv = (fpowi(xlv, 3) - 1.0f) * 2.0f / g.pcpskw(mon);
            float pcp_gen = xlv * g.pcpstd(mon) + p.pcpmean(mon);
            if (pcp_gen < 0.01f) pcp_gen = 0.01f;
            sum = sum + pcp_gen;
        }
        p.pcf(mon) = sum > 0.0f ? 1000.0f * p.pcpmean(mon) / sum : 1.0f;
        if (g.rain_yrs < 1.0f) g.rain_yrs = 10.0f;
        x1 = 0.5f / g.rain_yrs;
        x2 = x1 / g.pcpd(mon);
        const float x3 = rain_hhsm(mon) / std::log(x2);
        if (p.pcpmean(mon) > 1.e-4f) {
            p.amp_r(mon) = s.bsn_prm.adj_pkr * (1.0f - std::exp(x3 / p.pcpmean(mon)));
        } else {
            p.amp_r(mon) = 0.95f;
        }
        if (p.amp_r(mon) < 0.1f) p.amp_r(mon) = 0.1f;
        if (p.amp_r(mon) > 0.95f) p.amp_r(mon) = 0.95f;
    }
    if (summm_p < 508.0f) {
        p.ireg = 1;
    } else if (summm_p >= 508.0f && summm_p < 1016.0f) {
        p.ireg = 2;
    } else if (summm_p >= 1016.0f) {
        p.ireg = 3;
    }
}

// cli_wgnread.f90
void cli_wgnread(Sim& s) {
    const std::string& name = s.in_cli.weat_wgn;
    auto alloc_rng = [&](int lo, int n) {
        s.wgncur.alloc(1, 3, lo, n, 0.0f);
        s.wgnold.alloc(1, 3, lo, n, 0.0f);
        s.rnd2.alloc(lo, n, 0.0f);
        s.rnd3.alloc(lo, n, 0.0f);
        s.rnd8.alloc(lo, n, 0.0f);
        s.rnd9.alloc(lo, n, 0.0f);
        s.rndseed.alloc(1, 10, lo, n, 0);
        s.idg.alloc(1, 9, 0);
    };
    if (!s.files.exists(name) || name == "null") {
        s.wgn.alloc(0, 1, weather_generator_db());
        s.wgn_n.alloc(1, 1, "");
        s.wgn_pms.alloc(0, 1, wgn_parms());
        alloc_rng(0, 1);
        s.frad.alloc(0, 1, 1, 1, 0.0f);
        gcycl(s);
        return;
    }
    std::string titldum, header;
    fio::File f = s.files.open(name);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        int imax = 0;
        int eof = 0;
        while (eof == 0) {
            eof = f.read(chars(titldum, 80));
            if (eof < 0) break;
            eof = f.read(chars(header, 80));
            if (eof < 0) break;
            for (int mo = 1; mo <= 12; ++mo) {
                eof = f.read(chars(titldum, 80));
                if (eof < 0) break;
            }
            ++imax;
        }
        s.db_mx.wgnsta = imax;
        s.wgn.alloc(1, imax, weather_generator_db());
        s.wgn_n.alloc(1, imax, "");
        s.wgn_pms.alloc(1, imax, wgn_parms());
        alloc_rng(1, imax);
        s.frad.alloc(1, imax, 1, s.time.step, 0.0f);
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        gcycl(s);
        for (int iwgn = 1; iwgn <= s.db_mx.wgnsta; ++iwgn) {
            weather_generator_db& g = s.wgn(iwgn);
            eof = f.read(chars(s.wgn_n(iwgn), 50), g.lat, g.long_, g.elev, g.rain_yrs);
            if (eof < 0) break;
            eof = f.read(chars(header, 80));
            if (eof < 0) break;
            for (int mo = 1; mo <= 12; ++mo) {
                const int m = mo;
                eof = f.read(g.tmpmx(m), g.tmpmn(m), g.tmpstdmx(m), g.tmpstdmn(m), g.pcpmm(m), g.pcpstd(m),
                             g.pcpskw(m), g.pr_wd(m), g.pr_ww(m), g.pcpd(m), g.rainhmx(m), g.solarav(m), g.dewpt(m),
                             g.windav(m));
            }
            cli_initwgn(s, iwgn);
            if (eof < 0) break;
        }
        break;
    }
}

}  // namespace

// Fortran comparison of blank-padded strings (the shorter padded with blanks).
int fstrcmp(const std::string& a, const std::string& b) {
    const std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char ca = i < a.size() ? static_cast<unsigned char>(a[i]) : ' ';
        const unsigned char cb = i < b.size() ? static_cast<unsigned char>(b[i]) : ' ';
        if (ca != cb) return ca < cb ? -1 : 1;
    }
    return 0;
}

// search.f90, corrected: an exact search of the name list. The reference's binary
// search misses names of unsorted lists (a tmp.cli not in order: the gage then falls
// back to the weather generator); iseq is left as it was when the name is absent.
void search(const Vec<std::string>& sch, int max, const std::string& cfind, int& iseq) {
    const std::string c = cfind.size() > 50 ? cfind.substr(0, 50) : cfind;
    for (int i = 1; i <= max; ++i) {
        const std::string& v = sch(i);
        if (fstrcmp(v.size() > 50 ? v.substr(0, 50) : v, c) == 0) {
            iseq = i;
            return;
        }
    }
}

void proc_date_time(Sim& s) {
    cli_one_meas(s, s.in_cli.pet_cli, s.in_path_pet.peti, s.petm, s.petm_n, s.db_mx.petfiles);
    cli_pmeas(s);
    cli_tmeas(s);
    cli_one_meas(s, s.in_cli.slr_cli, s.in_path_slr.slr, s.slr, s.slr_n, s.db_mx.slrfiles);
    cli_one_meas(s, s.in_cli.hmd_cli, s.in_path_hmd.hmd, s.hmd, s.hmd_n, s.db_mx.rhfiles);
    cli_one_meas(s, s.in_cli.wnd_cli, s.in_path_wnd.wnd, s.wnd, s.wnd_n, s.db_mx.wndfiles);
    cli_wgnread(s);
}

// cli_staread.f90
void cli_staread(Sim& s) {
    const std::string& name = s.in_cli.weat_sta;
    if (!s.files.exists(name) || name == "null") {
        s.wst.alloc(0, 1, weather_station());
        s.wst_n.alloc(0, 0, "");
        return;
    }
    std::string titldum, header;
    fio::File f = s.files.open(name);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        const int imax = count_records(f);
        s.db_mx.wst = imax;
        s.wst.alloc(1, imax, weather_station());
        s.wst_n.alloc(1, imax, "");
        for (int iwst = 1; iwst <= s.db_mx.wst; ++iwst) {
            weather_station& w = s.wst(iwst);
            w.weat.ts.alloc(1, s.time.step, 0.0f);
            w.weat.ts_next.alloc(1, s.time.step, 0.0f);
            w.weat.precip_prior_day = "dry";
            w.tlag.alloc(1, 6, 0.0f);
            const int iwgn = w.wco.wgn;  // still the default (1): the station's generator is read below
            w.tlag.fill((s.wgn(iwgn).tmpmn(1) + s.wgn(iwgn).tmpstdmx(1)) / 2.0f);
        }
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= s.db_mx.wst; ++i) {
            weather_station& w = s.wst(i);
            int eof = f.read(chars(titldum, 80));
            if (eof < 0) break;
            f.backspace();
            eof = f.read(chars(w.name, 50), chars(w.wco_c.wgn, 50), chars(w.wco_c.pgage, 50), chars(w.wco_c.tgage, 50),
                         chars(w.wco_c.sgage, 50), chars(w.wco_c.hgage, 50), chars(w.wco_c.wgage, 50), chars(w.wco_c.petgage, 50),
                         chars(w.wco_c.atmodep, 50));
            if (eof < 0) break;
            s.wst_n(i) = w.name;
            if (s.db_mx.wgnsta > 0) search(s.wgn_n, s.db_mx.wgnsta, w.wco_c.wgn, w.wco.wgn);
            if (s.db_mx.pcpfiles > 0) search(s.pcp_n, s.db_mx.pcpfiles, w.wco_c.pgage, w.wco.pgage);
            w.pcp_ts = s.pcp(w.wco.pgage).tstep;
            if (s.db_mx.tmpfiles > 0) search(s.tmp_n, s.db_mx.tmpfiles, w.wco_c.tgage, w.wco.tgage);
            if (s.db_mx.slrfiles > 0) search(s.slr_n, s.db_mx.slrfiles, w.wco_c.sgage, w.wco.sgage);
            if (s.db_mx.rhfiles > 0) search(s.hmd_n, s.db_mx.rhfiles, w.wco_c.hgage, w.wco.hgage);
            if (s.db_mx.wndfiles > 0) search(s.wnd_n, s.db_mx.wndfiles, w.wco_c.wgage, w.wco.wgage);
            if (s.db_mx.petfiles > 0) search(s.petm_n, s.db_mx.petfiles, w.wco_c.petgage, w.wco.petgage);
            if (w.wco_c.atmodep != "null" && !w.wco_c.atmodep.empty())
                throw std::runtime_error("SWAT+ atmospheric deposition is not supported");
            // as the reference (diagnostics.out): a gage not found falls back to the generator
            if (w.wco.wgn == 0 && w.wco_c.wgn != "sim") s.warn(w.wco_c.wgn + " file not found (wgn)");
            if (w.wco.pgage == 0 && w.wco_c.pgage != "sim") s.warn(w.wco_c.pgage + " file not found (pgage)");
            if (w.wco.tgage == 0 && w.wco_c.tgage != "sim") s.warn(w.wco_c.tgage + " file not found (tgage)");
            if (w.wco.sgage == 0 && w.wco_c.sgage != "sim") s.warn(w.wco_c.sgage + " file not found (sgage)");
            if (w.wco.hgage == 0 && w.wco_c.hgage != "sim") s.warn(w.wco_c.hgage + " file not found (hgage)");
            if (w.wco.wgage == 0 && w.wco_c.wgage != "sim") s.warn(w.wco_c.wgage + " file not found (wgage)");
            if (w.wco.petgage == 0 && w.wco_c.petgage != "null") s.warn(w.wco_c.petgage + " file not found (petgage)");
        }
        break;
    }
}

// ---- daily weather ----------------------------------------------------------------

namespace {

void cli_bounds_check(const Sim& s, int st_day, int st_yr, int end_day, int end_yr, bool& out_bounds) {
    if (st_yr > s.time.yrc) {
        out_bounds = true;
    } else if (st_yr == s.time.yrc && st_day > s.time.day) {
        out_bounds = true;
    }
    if (end_yr < s.time.yrc) {
        out_bounds = true;
    } else if (end_yr == s.time.yrc && end_day < s.time.day) {
        out_bounds = true;
    }
}

void cli_pgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const weather_generator_db& g = s.wgn(iwgn);
    const int mo = s.time.mo;
    float pcpgen = 0.0f;
    const float vv = aunif(s.rndseed(s.idg(1), iwgn));
    const float xx = w.weat.precip_prior_day == "dry" ? g.pr_wd(mo) : g.pr_ww(mo);
    if (vv > xx) {
        pcpgen = 0.0f;
    } else {
        const float v8 = aunif(s.rndseed(s.idg(3), iwgn));
        const float r6 = g.pcpskw(mo) / 6.0f;
        float xlv = (cli_dstn1(s.rnd3(iwgn), v8) - r6) * r6 + 1.0f;
        xlv = (fpowi(xlv, 3) - 1.0f) * 2.0f / g.pcpskw(mo);
        s.rnd3(iwgn) = v8;
        pcpgen = xlv * g.pcpstd(mo) + s.wgn_pms(iwgn).pcpmean(mo);
        pcpgen = pcpgen * s.wgn_pms(iwgn).pcf(mo);
        if (pcpgen < 0.1f) pcpgen = 0.1f;
    }
    w.weat.precip_next = pcpgen;
}

void cli_weatgn(Sim& s, int iwgn) {
    // a(n,l), b(n,l) column-major from the Fortran RESHAPE
    static const float a[3][3] = {{0.567f, 0.086f, -0.002f}, {0.253f, 0.504f, -0.050f}, {-0.006f, -0.039f, 0.244f}};
    static const float b[3][3] = {{0.781f, 0.0f, 0.0f}, {0.328f, 0.637f, 0.0f}, {0.238f, -0.341f, 0.873f}};
    float xx[3] = {0.0f, 0.0f, 0.0f};
    float e[3] = {0.0f, 0.0f, 0.0f};
    float v2 = aunif(s.rndseed(s.idg(8), iwgn));
    e[0] = cli_dstn1(s.rnd8(iwgn), v2);
    s.rnd8(iwgn) = v2;
    v2 = aunif(s.rndseed(s.idg(9), iwgn));
    e[1] = cli_dstn1(s.rnd9(iwgn), v2);
    s.rnd9(iwgn) = v2;
    v2 = aunif(s.rndseed(s.idg(2), iwgn));
    e[2] = cli_dstn1(s.rnd2(iwgn), v2);
    s.rnd2(iwgn) = v2;
    for (int n = 1; n <= 3; ++n) {
        s.wgncur(n, iwgn) = 0.0f;
        for (int l = 1; l <= 3; ++l) {
            s.wgncur(n, iwgn) = s.wgncur(n, iwgn) + b[n - 1][l - 1] * e[l - 1];
            xx[n - 1] = xx[n - 1] + a[n - 1][l - 1] * s.wgnold(l, iwgn);
        }
    }
    for (int n = 1; n <= 3; ++n) {
        s.wgncur(n, iwgn) = s.wgncur(n, iwgn) + xx[n - 1];
        if (s.wgncur(n, iwgn) > 1.0f) s.wgncur(n, iwgn) = 1.0f;
        if (s.wgncur(n, iwgn) < -1.0f) s.wgncur(n, iwgn) = -1.0f;
        s.wgnold(n, iwgn) = s.wgncur(n, iwgn);
    }
}

void cli_tgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const weather_generator_db& g = s.wgn(iwgn);
    const int mo = s.time.mo;
    const float tamp = 0.5f * (g.tmpmx(mo) - g.tmpmn(mo));
    float txxm = g.tmpmx(mo) + tamp * s.wgn_pms(iwgn).pr_wdays(mo);
    if (w.weat.precip > 0.0f) txxm = txxm - tamp;
    const float tmxg = txxm + g.tmpstdmx(mo) * s.wgncur(1, iwgn);
    float tmng = g.tmpmn(mo) + g.tmpstdmn(mo) * s.wgncur(2, iwgn);
    if (tmng > tmxg) tmng = tmxg - 0.2f * std::abs(tmxg);
    w.weat.tmax = tmxg;
    w.weat.tmin = tmng;
}

void cli_clgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const wgn_parms& p = s.wgn_pms(iwgn);
    w.weat.precip_prior_day = w.weat.precip >= 0.1f ? "wet" : "dry";
    const float day = static_cast<float>(s.time.day);
    const float sd = std::asin(0.4f * std::sin((day - 82.0f) / 58.09f));
    const float dd = 1.0f + 0.033f * std::cos(day / 58.09f);
    const float sdlat = -p.latsin * std::tan(sd) / p.latcos;
    float h = 0.0f;
    if (sdlat > 1.0f) {
        h = 0.0f;
    } else if (sdlat >= -1.0f) {
        h = std::acos(sdlat);
    } else {
        h = 3.1416f;
    }
    w.weat.daylength = 7.6394f * h;
    const float ys = p.latsin * std::sin(sd);
    const float yc = p.latcos * std::cos(sd);
    w.weat.solradmx = 30.0f * dd * (h * ys + yc * std::sin(h));
    float totrho = 0.0f;
    std::vector<float> cosrho(static_cast<std::size_t>(s.time.step), 0.0f);
    for (int ii = 1; ii <= s.time.step; ++ii) {
        const float hr_angle = (12.5f - static_cast<float>(ii)) * 0.2618f * s.time.dtm / 60.0f;
        float c = ys + yc * std::cos(hr_angle);
        if (c <= 0.0f) c = 0.0f;
        cosrho[static_cast<std::size_t>(ii - 1)] = c;
        totrho = totrho + c;
    }
    if (totrho > 0.001f)
        for (int ii = 1; ii <= s.time.step; ++ii) s.frad(iwgn, ii) = cosrho[static_cast<std::size_t>(ii - 1)] / totrho;
}

void cli_slrgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const weather_generator_db& g = s.wgn(iwgn);
    const int mo = s.time.mo;
    float rav = g.solarav(mo) / (1.0f - 0.5f * s.wgn_pms(iwgn).pr_wdays(mo));
    if (w.weat.precip > 0.0f) rav = 0.5f * rav;
    const float rx = w.weat.solradmx - rav;
    w.weat.solrad = rav + s.wgncur(3, iwgn) * rx / 4.0f;
    if (w.weat.solrad <= 0.0f) w.weat.solrad = 0.05f * w.weat.solradmx;
    // the reference then replaces the generated value by the monthly mean
    w.weat.solrad = g.solarav(mo);
}

void cli_rhgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const weather_generator_db& g = s.wgn(iwgn);
    const wgn_parms& p = s.wgn_pms(iwgn);
    const int mo = s.time.mo;
    float rhmo = 0.0f;
    if (p.idewpt == 0) {
        const float tmpmean = (g.tmpmx(mo) + g.tmpmn(mo)) / 2.0f;
        rhmo = ee(g.dewpt(mo)) / ee(tmpmean);
    } else {
        rhmo = g.dewpt(mo);
    }
    const float yy = 0.9f * p.pr_wdays(mo);
    float rhm = (rhmo - yy) / (1.0f - yy);
    if (rhm < 0.05f) rhm = 0.5f * rhmo;
    if (w.weat.precip > 0.0f) rhm = rhm * 0.1f + 0.9f;
    const float vv = rhm - 1.0f;
    const float uplm = rhm - vv * std::exp(vv);
    const float blm = rhm * (1.0f - std::exp(-rhm));
    w.weat.rhum = atri(blm, rhm, uplm, s.rndseed(s.idg(7), iwgn));
}

void cli_wndgen(Sim& s, int iwgn) {
    weather_station& w = s.wst(s.iwst);
    const float v6 = aunif(s.rndseed(s.idg(5), iwgn));
    w.weat.windsp = s.wgn(iwgn).windav(s.time.mo) * fpow(-std::log(v6), 0.3f);
}

}  // namespace

// cli_precip_control.f90 (daily step: no sub-daily precipitation)
void cli_precip_control(Sim& s, int istart) {
    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        w.weat.precip = w.weat.precip_next;
        w.weat.precip_next = 0.0f;
        w.weat.ts = w.weat.ts_next;
        const int iwgn = w.wco.wgn;
        const int ipg = w.wco.pgage;
        if (w.wco_c.pgage == "sim") {
            cli_pgen(s, iwgn);
        } else {
            bool out_bounds = false;
            climate_measured_data& m = s.pcp(ipg);
            int cur_day = s.time.day + istart;
            int yrs_to_start = s.time.yrs - m.yrs_start;
            if (cur_day > s.time.day_end_yr) {
                cur_day = 1;
                yrs_to_start = yrs_to_start + 1;
            }
            cli_bounds_check(s, m.start_day, m.start_yr, m.end_day, m.end_yr, out_bounds);
            if (yrs_to_start > m.end_yr - m.start_yr + 1) out_bounds = true;
            if (out_bounds) {
                w.weat.precip_next = -98.0f;
            } else {
                w.weat.precip_next = m.ts(cur_day, yrs_to_start);
            }
            if (w.weat.precip_next <= -97.0f) {
                cli_pgen(s, iwgn);
                m.days_gen = m.days_gen + 1;
            }
        }
        if (istart > 0) {
            w.precip_aa = w.precip_aa + w.weat.precip;
            w.pet_aa = w.pet_aa + w.weat.pet;
        }
    }
}

// climate_control.f90
void climate_control(Sim& s) {
    cli_precip_control(s, 1);
    const int mo = s.time.mo;

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int iwgn = w.wco.wgn;
        cli_weatgn(s, iwgn);
        if (w.wco_c.tgage == "sim") {
            cli_tgen(s, iwgn);
        } else {
            climate_measured_data& m = s.tmp(w.wco.tgage);
            bool out_bounds = false;
            cli_bounds_check(s, m.start_day, m.start_yr, m.end_day, m.end_yr, out_bounds);
            if (out_bounds) {
                w.weat.tmax = -98.0f;
                w.weat.tmin = -98.0f;
            } else {
                const int yrs_to_start = s.time.yrs - m.yrs_start;
                w.weat.tmax = m.ts(s.time.day, yrs_to_start);
                w.weat.tmin = m.ts2(s.time.day, yrs_to_start);
            }
            if (w.weat.tmax <= -97.0f || w.weat.tmin <= -97.0f) {
                cli_weatgn(s, iwgn);
                cli_tgen(s, iwgn);
                m.days_gen = m.days_gen + 1;
            }
        }
        w.weat.tave = (w.weat.tmax + w.weat.tmin) / 2.0f;
    }

    auto measured = [&](Vec<climate_measured_data>& arr, int ig, float& value, auto&& generate) {
        climate_measured_data& m = arr(ig);
        bool out_bounds = false;
        cli_bounds_check(s, m.start_day, m.start_yr, m.end_day, m.end_yr, out_bounds);
        if (out_bounds) {
            value = -98.0f;
        } else {
            const int yrs_to_start = s.time.yrs - m.yrs_start;
            value = m.ts(s.time.day, yrs_to_start);
        }
        if (value <= -97.0f) {
            generate();
            m.days_gen = m.days_gen + 1;
        }
    };

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int iwgn = w.wco.wgn;
        cli_clgen(s, iwgn);
        if (w.wco_c.sgage == "sim") {
            cli_slrgen(s, iwgn);
        } else {
            measured(s.slr, w.wco.sgage, w.weat.solrad, [&] { cli_slrgen(s, iwgn); });
        }
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int iwgn = w.wco.wgn;
        if (w.wco_c.hgage == "sim") {
            cli_rhgen(s, iwgn);
        } else {
            measured(s.hmd, w.wco.hgage, w.weat.rhum, [&] { cli_rhgen(s, iwgn); });
        }
        w.weat.dewpt = w.weat.tave - (1.0f - w.weat.rhum) / 5.0f;
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int iwgn = w.wco.wgn;
        if (w.wco_c.wgage == "sim") {
            cli_wndgen(s, iwgn);
        } else {
            measured(s.wnd, w.wco.wgage, w.weat.windsp, [&] { cli_wndgen(s, iwgn); });
        }
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int ig = w.wco.petgage;
        if (ig > 0) {
            measured(s.petm, ig, w.weat.pet, [&] {
                const float ramm = w.weat.solradmx * 37.59f / 30.0f;
                if (w.weat.tmax > w.weat.tmin) {
                    const float xl = 2.501f - 2.361e-3f * w.weat.tave;
                    w.weat.pet = 0.0023f * (ramm / xl) * (w.weat.tave + 17.8f) * fpow(w.weat.tmax - w.weat.tmin, 0.5f);
                    w.weat.pet = fmax2(0.0f, w.weat.pet);
                } else {
                    w.weat.pet = 0.0f;
                }
            });
        }
    }

    s.ppet_mce = s.ppet_mce + 1;
    if (s.ppet_mce > s.ppet_ndays) s.ppet_mce = 1;
    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const float ramm = w.weat.solradmx * 37.59f / 30.0f;
        if (w.weat.tmax > w.weat.tmin) {
            const float xl = 2.501f - 2.361e-3f * w.weat.tave;
            w.weat.pet = 0.0023f * (ramm / xl) * (w.weat.tave + 17.8f) * fpow(w.weat.tmax - w.weat.tmin, 0.5f);
            w.weat.pet = fmax2(0.01f, w.weat.pet);
        } else {
            w.weat.pet = 0.01f;
        }
        if (w.weat.pet > 0.1f) w.weat.ppet = w.weat.ppet + w.weat.precip / w.weat.pet;
        wgn_parms& p = s.wgn_pms(w.wco.wgn);
        p.precip_sum = p.precip_sum + w.weat.precip - p.precip_mce(s.ppet_mce);
        p.pet_sum = p.pet_sum + w.weat.pet - p.pet_mce(s.ppet_mce);
        if (p.pet_sum > 1.e-6f) {
            p.p_pet_rto = p.precip_sum / p.pet_sum;
        } else {
            p.p_pet_rto = p.precip_sum / 1.e-6f;
        }
        p.precip_mce(s.ppet_mce) = w.weat.precip;
        p.pet_mce(s.ppet_mce) = w.weat.pet;
        w.tlag(w.tlag_mne) = w.weat.tave;
        w.tlag_mne = w.tlag_mne + 1;
        if (w.tlag_mne > 6) w.tlag_mne = 1;
        w.airlag_temp = w.tlag(w.tlag_mne);
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const int iwgn = w.wco.wgn;
        const float half_hr_mn = 0.02083f;
        const float xx = -125.0f / (w.weat.precip + 5.0f);
        const float half_hr_mx = 1.0f - std::exp(xx);
        w.weat.precip_half_hr = atri(half_hr_mn, s.wgn_pms(iwgn).amp_r(mo), half_hr_mx, s.rndseed(10, iwgn));
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        const wgn_parms& p = s.wgn_pms(w.wco.wgn);
        if (p.phutot > 0.0f) {
            if (w.weat.tave > 0.0f) w.weat.phubase0 = w.weat.phubase0 + w.weat.tave / p.phutot;
        } else {
            w.weat.phubase0 = 0.0f;
        }
        if (s.time.end_yr == 1) w.weat.phubase0 = 0.0f;
    }

    for (s.iwst = 1; s.iwst <= s.db_mx.wst; ++s.iwst) {
        weather_station& w = s.wst(s.iwst);
        w.weat.precip = w.weat.precip * (1.0f + w.rfinc(mo) / 100.0f);
        if (w.weat.precip < 0.0f) w.weat.precip = 0.0f;
        w.weat.tmax = w.weat.tmax + w.tmpinc(mo);
        w.weat.tmin = w.weat.tmin + w.tmpinc(mo);
        w.weat.solrad = w.weat.solrad + w.radinc(mo);
        w.weat.solrad = fmax2(0.0f, w.weat.solrad);
        w.weat.rhum = w.weat.rhum + w.huminc(mo);
        w.weat.rhum = fmax2(0.01f, w.weat.rhum);
        w.weat.rhum = fmin2(0.99f, w.weat.rhum);
    }
}

}  // namespace swat
