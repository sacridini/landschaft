// SPDX-License-Identifier: GPL-3.0-or-later
//
// What SWAT+ prints: lsu_read_elements (the basin fractions of the HRUs), hru_output,
// aquifer_output, sd_channel_output and basin_output, called from command after the
// day's objects. The values each WRITE of the reference prints are kept as print
// records (tag = the Fortran unit: 2000-2043 HRU water, nutrient, losses and plant
// tables, 2050-2083 the basin's, 2520-2523 aquifers, 2500-2503 SWAT-DEG channels;
// the last digit is the step: day, month, year, average annual).

#include <algorithm>
#include <string>
#include <vector>

#include "swat/fmath.hpp"
#include "swat/hyd_ops.hpp"
#include "swat/om_ops.hpp"
#include "swat/out_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

std::vector<float> cat(std::initializer_list<std::vector<float>> parts) {
    std::vector<float> v;
    for (const auto& p : parts) v.insert(v.end(), p.begin(), p.end());
    return v;
}

bool day_print(const Sim& s) { return s.pco.day_print == "y" && s.pco.int_day_cur == s.pco.int_day; }

float month_days(const Sim& s) { return static_cast<float>(s.ndays(s.time.mo + 1) - s.ndays(s.time.mo)); }

// hru_output.f90 (no carbon, pesticide or salt tables)
void hru_output(Sim& s, int j) {
    const basin_print_codes& p = s.pco;
    output_waterbal& wbd = s.hwb_d(j);
    output_plantweather& pwd = s.hpw_d(j);
    s.hwb_m(j) = s.hwb_m(j) + wbd;
    s.hnb_m(j) = s.hnb_m(j) + s.hnb_d(j);
    s.hls_m(j) = s.hls_m(j) + s.hls_d(j);
    float bm_max_m = s.hpw_m(j).bm_max;
    float bm_max_y = s.hpw_y(j).bm_max;
    const float bm_max_a = s.hpw_a(j).bm_max;
    s.hpw_m(j) = s.hpw_m(j) + pwd;
    s.hpw_m(j).bm_max = bm_max_m;
    pwd.bm_max = pwd.bioms;
    s.hpw_m(j).bm_max = fmax2(pwd.bioms, s.hpw_m(j).bm_max);
    s.hpw_y(j).bm_max = fmax2(pwd.bioms, s.hpw_y(j).bm_max);
    s.hpw_a(j).bm_max = fmax2(pwd.bioms, s.hpw_a(j).bm_max);
    wbd.sw_final = s.soil(j).sw;
    wbd.sw = (wbd.sw_init + wbd.sw_final) / 2.0f;
    wbd.sno_final = s.hru(j).sno_mm;
    wbd.snopack = (wbd.sno_init + wbd.sno_final) / 2.0f;
    if (day_print(s)) {
        if (p.wb_hru.d == "y") s.print_record(2000, j, values(wbd));
        wbd.sw_init = wbd.sw_final;
        wbd.sno_init = wbd.sno_final;
        if (p.nb_hru.d == "y") s.print_record(2020, j, values(s.hnb_d(j)));
        if (p.ls_hru.d == "y") s.print_record(2030, j, cat({values(s.hls_d(j)), {pwd.percn}}));
        if (p.pw_hru.d == "y") {
            pwd.bm_max = pwd.bioms;
            s.print_record(2040, j, values(pwd));
        }
    }
    if (s.time.end_mo == 1) {
        bm_max_m = s.hpw_m(j).bm_max;
        s.hwb_y(j) = s.hwb_y(j) + s.hwb_m(j);
        s.hnb_y(j) = s.hnb_y(j) + s.hnb_m(j);
        s.hls_y(j) = s.hls_y(j) + s.hls_m(j);
        bm_max_y = s.hpw_y(j).bm_max;
        s.hpw_y(j) = s.hpw_y(j) + s.hpw_m(j);
        s.hpw_y(j).bm_max = bm_max_y;
        const float c = month_days(s);
        s.hpw_m(j) = ave(s.hpw_m(j), c);
        s.hwb_m(j) = ave(s.hwb_m(j), c);
        s.hpw_m(j).bm_max = bm_max_m;
        s.hwb_m(j).sw_final = wbd.sw_final;
        s.hwb_m(j).sno_final = wbd.sno_final;
        if (p.wb_hru.m == "y") s.print_record(2001, j, values(s.hwb_m(j)));
        if (p.nb_hru.m == "y") s.print_record(2021, j, values(s.hnb_m(j)));
        if (p.ls_hru.m == "y") s.print_record(2031, j, cat({values(s.hls_m(j)), {s.hpw_m(j).percn}}));
        if (p.pw_hru.m == "y") {
            s.hpw_m(j).nplnt = s.pl_mass(j).tot_com.n;
            s.hpw_m(j).pplnt = s.pl_mass(j).tot_com.p;
            s.print_record(2041, j, values(s.hpw_m(j)));
        }
        s.hpw_m(j).bm_max = 0.0f;
        const float sw_init = s.hwb_m(j).sw_final;
        const float sno_init = s.hwb_m(j).sno_final;
        s.hwb_m(j) = s.hwbz;
        s.hwb_m(j).sw_init = sw_init;
        s.hwb_m(j).sno_init = sno_init;
        s.hnb_m(j) = s.hnbz;
        s.hpw_m(j) = s.hpwz;
        s.hls_m(j) = s.hlsz;
    }
    if (s.time.end_yr == 1) {
        bm_max_y = s.hpw_y(j).bm_max;
        s.hwb_a(j) = s.hwb_a(j) + s.hwb_y(j);
        s.hnb_a(j) = s.hnb_a(j) + s.hnb_y(j);
        s.hls_a(j) = s.hls_a(j) + s.hls_y(j);
        s.hpw_a(j) = s.hpw_a(j) + s.hpw_y(j);
        const float c = static_cast<float>(s.time.day_end_yr);
        s.hwb_y(j) = ave(s.hwb_y(j), c);
        s.hpw_y(j) = ave(s.hpw_y(j), c);
        s.hpw_y(j).bm_max = bm_max_y;
        s.hpw_a(j).bm_max = bm_max_a;
        s.hwb_y(j).sw_final = wbd.sw_final;
        s.hwb_y(j).sno_final = wbd.sno_final;
        if (s.hwb_a(j).irr > 10.0f) s.hru(j).irr = 1;
        if (p.wb_hru.y == "y") s.print_record(2002, j, values(s.hwb_y(j)));
        if (p.nb_hru.y == "y") s.print_record(2022, j, values(s.hnb_y(j)));
        if (p.ls_hru.y == "y") s.print_record(2032, j, cat({values(s.hls_y(j)), {s.hpw_y(j).percn}}));
        if (p.pw_hru.y == "y") {
            s.hpw_y(j).nplnt = s.pl_mass(j).tot_com.n;
            s.hpw_y(j).pplnt = s.pl_mass(j).tot_com.p;
            s.print_record(2042, j, values(s.hpw_y(j)));
        }
        s.hpw_y(j).bm_max = 0.0f;
    }
    if (s.time.end_sim == 1) {
        float sw_init = s.hwb_a(j).sw_init;
        float sno_init = s.hwb_a(j).sno_init;
        s.hwb_a(j) = s.hwb_a(j) / s.time.yrs_prt;
        s.hwb_a(j) = ave(s.hwb_a(j), s.time.days_prt);
        s.hwb_a(j).sw_init = sw_init;
        s.hwb_a(j).sw_final = wbd.sw_final;
        s.hwb_a(j).sno_init = sno_init;
        s.hwb_a(j).sno_final = wbd.sno_final;
        if (p.wb_hru.a == "y") s.print_record(2003, j, values(s.hwb_a(j)));
        sw_init = wbd.sw_final;
        sno_init = wbd.sno_final;
        hydrologic_response_unit& h = s.hru(j);
        h.precip_aa = s.hwb_a(j).precip;
        h.flow(1) = s.hwb_a(j).wateryld;
        h.flow(2) = s.hwb_a(j).perc;
        h.flow(3) = s.hwb_a(j).surq_gen;
        h.flow(4) = s.hwb_a(j).latq;
        h.flow(5) = s.hwb_a(j).qtile;
        s.hwb_a(j) = s.hwbz;
        s.hwb_a(j).sw_init = sw_init;
        s.hwb_a(j).sno_init = sno_init;
    }
    if (s.time.end_sim == 1 && p.nb_hru.a == "y") {
        s.hnb_a(j) = s.hnb_a(j) / s.time.yrs_prt;
        s.print_record(2023, j, values(s.hnb_a(j)));
        s.hnb_a(j) = s.hnbz;
    }
    if (s.time.end_sim == 1 && p.ls_hru.a == "y") {
        s.hls_a(j) = s.hls_a(j) / s.time.yrs_prt;
        const float percn_aa = s.hpw_a(j).percn / s.time.yrs_prt;
        s.print_record(2033, j, cat({values(s.hls_a(j)), {percn_aa}}));
        s.hls_a(j) = s.hlsz;
    }
    if (s.time.end_sim == 1 && p.pw_hru.a == "y") {
        s.hpw_a(j) = s.hpw_a(j) / s.time.yrs_prt;
        s.hpw_a(j) = ave(s.hpw_a(j), s.time.days_prt);
        s.hpw_a(j).nplnt = s.pl_mass(j).tot_com.n;
        s.hpw_a(j).pplnt = s.pl_mass(j).tot_com.p;
        s.hpw_a(j).bm_max = bm_max_a;
        s.print_record(2043, j, values(s.hpw_a(j)));
        s.hru(j).strsa = s.hpw_a(j).strsa;
        s.hpw_a(j) = s.hpwz;
    }
    if (s.time.end_sim == 1 && (s.pco.crop_yld == "a" || s.pco.crop_yld == "b")) {
        plant_community& pc = s.pcom(j);
        for (int ipl = 1; ipl <= pc.npl; ++ipl)
            if (pc.plcur(ipl).harv_num > 0)
                s.pl_mass(j).yield_tot(ipl) =
                    s.pl_mass(j).yield_tot(ipl) / static_cast<float>(pc.plcur(ipl).harv_num);
    }
}

// aquifer_output.f90
void aquifer_output(Sim& s, int iaq) {
    const basin_print_codes& p = s.pco;
    aquifer_dynamic& m = s.aqu_m(iaq);
    m = m + s.aqu_d(iaq);
    if (day_print(s) && p.aqu.d == "y") s.print_record(2520, iaq, values(s.aqu_d(iaq)));
    if (s.time.end_mo == 1) {
        const float c = month_days(s);
        m.stor = m.stor / c;
        m.dep_wt = m.dep_wt / c;
        m.no3_st = m.no3_st / c;
        s.aqu_y(iaq) = s.aqu_y(iaq) + m;
        if (p.aqu.m == "y") s.print_record(2521, iaq, values(m));
        m = s.aquz;
    }
    if (s.time.end_yr == 1) {
        aquifer_dynamic& y = s.aqu_y(iaq);
        y.stor = y.stor / 12.0f;
        y.dep_wt = y.dep_wt / 12.0f;
        y.no3_st = y.no3_st / 12.0f;
        s.aqu_a(iaq) = s.aqu_a(iaq) + y;
        if (p.aqu.y == "y") s.print_record(2522, iaq, values(y));
        y = s.aquz;
    }
    if (s.time.end_sim == 1 && p.aqu.a == "y") {
        s.aqu_a(iaq) = s.aqu_a(iaq) / s.time.yrs_prt;
        s.print_record(2523, iaq, values(s.aqu_a(iaq)));
    }
}

std::vector<float> channel_record(const Sim& s, const water_body& w, const hyd_output& in, const hyd_output& out,
                                  int ichan) {
    return cat({{w.area_ha, w.precip, w.evap, w.seep}, values(s.ch_stor(ichan)), values(in), values(out), {s.wtemp}});
}

// sd_channel_output.f90 (daily step)
void sd_channel_output(Sim& s, int ichan) {
    const basin_print_codes& p = s.pco;
    s.ch_in_m(ichan) = s.ch_in_m(ichan) + s.ch_in_d(ichan);
    s.ch_out_m(ichan) = s.ch_out_m(ichan) + s.ch_out_d(ichan);
    s.ch_wat_m(ichan) = s.ch_wat_m(ichan) + s.ch_wat_d(ichan);
    if (day_print(s) && p.sd_chan.d == "y")
        s.print_record(2500, ichan, channel_record(s, s.ch_wat_d(ichan), s.ch_in_d(ichan), s.ch_out_d(ichan), ichan));
    if (s.time.end_mo == 1) {
        const float c = month_days(s);
        s.ch_in_y(ichan) = s.ch_in_y(ichan) + s.ch_in_m(ichan);
        s.ch_out_y(ichan) = s.ch_out_y(ichan) + s.ch_out_m(ichan);
        s.ch_wat_y(ichan) = s.ch_wat_y(ichan) + s.ch_wat_m(ichan);
        s.ch_in_m(ichan).flo = s.ch_in_m(ichan).flo / c;
        s.ch_out_m(ichan).flo = s.ch_out_m(ichan).flo / c;
        s.ch_wat_m(ichan) = ave(s.ch_wat_m(ichan), c);
        if (p.sd_chan.m == "y")
            s.print_record(2501, ichan,
                           channel_record(s, s.ch_wat_m(ichan), s.ch_in_m(ichan), s.ch_out_m(ichan), ichan));
        s.ch_in_m(ichan) = s.chaz;
        s.ch_out_m(ichan) = s.chaz;
        s.ch_wat_m(ichan) = s.wbodz;
    }
    if (s.time.end_yr == 1) {
        const float c = static_cast<float>(s.time.day_end_yr);
        s.ch_in_y(ichan).flo = s.ch_in_y(ichan).flo / c;
        s.ch_out_y(ichan).flo = s.ch_out_y(ichan).flo / c;
        s.ch_in_a(ichan) = s.ch_in_a(ichan) + s.ch_in_y(ichan);
        s.ch_out_a(ichan) = s.ch_out_a(ichan) + s.ch_out_y(ichan);
        s.ch_wat_a(ichan) = s.ch_wat_a(ichan) + s.ch_wat_y(ichan);
        s.ch_wat_y(ichan) = ave(s.ch_wat_y(ichan), c);
        if (p.sd_chan.y == "y")
            s.print_record(2502, ichan,
                           channel_record(s, s.ch_wat_y(ichan), s.ch_in_y(ichan), s.ch_out_y(ichan), ichan));
    }
    if (s.time.end_sim == 1) {
        s.ch_in_a(ichan) = s.ch_in_a(ichan) / s.time.yrs_prt;
        s.ch_out_a(ichan) = s.ch_out_a(ichan) / s.time.yrs_prt;
        s.ch_wat_a(ichan) = s.ch_wat_a(ichan) / s.time.yrs_prt;
        s.ch_wat_a(ichan) = ave(s.ch_wat_a(ichan), s.time.yrs_prt);
        if (p.sd_chan.a == "y")
            s.print_record(2503, ichan,
                           channel_record(s, s.ch_wat_a(ichan), s.ch_in_a(ichan), s.ch_out_a(ichan), ichan));
    }
}

}  // namespace

// lsu_read_elements.f90 (the elements of ls_unit.ele: what basin_output weighs)
void lsu_read_elements(Sim& s) {
    const std::string& file = s.in_regs.ele_lsu;
    if (!s.files.exists(file) || file == "null") return;
    fio::File f = s.files.open(file);
    std::string titldum, header, name;
    if (f.read(chars(titldum, 80)) < 0) return;
    if (f.read(chars(header, 80)) < 0) return;
    int imax = 0, eof = 0;
    while (eof == 0) {
        int i = 0;
        eof = f.read(i);
        if (eof < 0) break;
        imax = std::max(i, imax);
    }
    s.lsu_elem.alloc(1, imax, landscape_elements());
    f.rewind();
    if (f.read(chars(titldum, 80)) < 0) return;
    if (f.read(chars(header, 80)) < 0) return;
    s.db_mx.lsu_elem = imax;
    for (int isp = 1; isp <= imax; ++isp) {
        int i = 0, k = 0;
        if (f.read(i) < 0) break;
        f.backspace();
        landscape_elements& e = s.lsu_elem(i);
        if (f.read(k, chars(e.name, e.name_len), chars(e.obtyp, e.obtyp_len), e.obtypno, e.bsn_frac, e.ru_frac) < 0)
            break;
    }
}

// basin_sw_init.f90 (no landscape unit outputs)
void basin_sw_init(Sim& s) {
    for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
        for (Vec<output_waterbal>* v : {&s.hwb_d, &s.hwb_m, &s.hwb_y, &s.hwb_a}) {
            (*v)(ihru).sw_init = s.soil(ihru).sw;
            (*v)(ihru).sno_init = s.hru(ihru).sno_mm;
        }
    }
    s.bwb_d.sw_init = 0.0f;
    s.bwb_d.sno_init = 0.0f;
    // without ls_unit.ele the reference reads past an unallocated lsu_elem; nothing it
    // prints depends on it then (basin_output needs the elements)
    if (s.db_mx.lsu_elem > 0) {
        for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
            const int iihru = s.lsu_elem(ihru).obtypno;
            const landscape_elements& e = s.lsu_elem(iihru);
            if (e.bsn_frac > 1.e-12f && e.obtyp == "hru") {
                s.bwb_d.sw_init = s.bwb_d.sw_init + s.hwb_d(iihru).sw_init * e.bsn_frac;
                s.bwb_d.sno_init = s.bwb_d.sno_init + s.hwb_d(iihru).sno_init * e.bsn_frac;
            }
        }
    }
    for (output_waterbal* b : {&s.bwb_m, &s.bwb_y, &s.bwb_a}) {
        b->sw_init = s.bwb_d.sw_init;
        b->sno_init = s.bwb_d.sno_init;
    }
}

// the output allocations of aqu_initial and sd_channel_read
void output_allo(Sim& s) {
    const int naq = s.sp_ob.aqu;
    s.aqu_m.alloc(1, naq, aquifer_dynamic());
    s.aqu_y.alloc(1, naq, aquifer_dynamic());
    s.aqu_a.alloc(1, naq, aquifer_dynamic());
    const int nch = s.sp_ob.chandeg;
    for (Vec<hyd_output>* v : {&s.ch_in_m, &s.ch_in_y, &s.ch_in_a, &s.ch_out_m, &s.ch_out_y, &s.ch_out_a})
        v->alloc(0, nch, hyd_output());
    for (Vec<water_body>* v : {&s.ch_wat_m, &s.ch_wat_y, &s.ch_wat_a}) v->alloc(0, nch, water_body());
    if (s.chsd_y.size() == 0) s.chsd_y.alloc(0, nch, sd_ch_output());
    if (s.ch_stor_y.size() == 0) s.ch_stor_y.alloc(0, nch, hyd_output());
}

// basin_output.f90 (HRUs only)
void basin_output(Sim& s) {
    const basin_print_codes& p = s.pco;
    float sw_init = s.bwb_d.sw_init;
    float sno_init = s.bwb_d.sno_init;
    s.bwb_d = s.hwbz;
    s.bwb_d.sw_init = sw_init;
    s.bwb_d.sno_init = sno_init;
    s.bnb_d = s.hnbz;
    s.bls_d = s.hlsz;
    s.bpw_d = s.hpwz;
    for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
        // the element of HRU ihru, then that element's number as an element: as the reference
        const int iihru = s.lsu_elem(ihru).obtypno;
        const landscape_elements& e = s.lsu_elem(iihru);
        if (e.bsn_frac > 1.e-12f && e.obtyp == "hru") {
            const float c = e.bsn_frac;
            s.bwb_d = s.bwb_d + s.hwb_d(iihru) * c;
            s.bwb_d.sw_final = s.bwb_d.sw_final + s.hwb_d(iihru).sw_final * c;
            s.bwb_d.sno_final = s.bwb_d.sno_final + s.hwb_d(iihru).sno_final * c;
            s.bnb_d = s.bnb_d + s.hnb_d(iihru) * c;
            s.bls_d = s.bls_d + s.hls_d(iihru) * c;
            s.bpw_d = s.bpw_d + s.hpw_d(iihru) * c;
        }
    }
    s.bwb_m = s.bwb_m + s.bwb_d;
    s.bnb_m = s.bnb_m + s.bnb_d;
    s.bls_m = s.bls_m + s.bls_d;
    s.bpw_m = s.bpw_m + s.bpw_d;
    if (day_print(s)) {
        if (p.wb_bsn.d == "y") {
            s.bwb_d.sw = (s.bwb_d.sw_init + s.bwb_d.sw_final) / 2.0f;
            s.bwb_d.snopack = (s.bwb_d.sno_init + s.bwb_d.sno_final) / 2.0f;
            s.print_record(2050, 1, values(s.bwb_d));
            s.bwb_d.sw_init = s.bwb_d.sw_final;
            s.bwb_d.sno_init = s.bwb_d.sno_final;
        }
        if (p.nb_bsn.d == "y") s.print_record(2060, 1, values(s.bnb_d));
        if (p.ls_bsn.d == "y") s.print_record(2070, 1, values(s.bls_d));
        if (p.pw_bsn.d == "y") s.print_record(2080, 1, values(s.bpw_d));
    }
    if (s.time.end_mo == 1) {
        s.bwb_y = s.bwb_y + s.bwb_m;
        s.bnb_y = s.bnb_y + s.bnb_m;
        s.bls_y = s.bls_y + s.bls_m;
        s.bpw_y = s.bpw_y + s.bpw_m;
        const float c = month_days(s);
        s.bwb_m = ave(s.bwb_m, c);
        s.bpw_m = ave(s.bpw_m, c);
        if (p.wb_bsn.m == "y") {
            s.bwb_m.sw_final = s.bwb_d.sw_final;
            s.bwb_m.sno_final = s.bwb_d.sno_final;
            s.print_record(2051, 1, values(s.bwb_m));
            s.bwb_m.sw_init = s.bwb_m.sw_final;
            s.bwb_m.sno_init = s.bwb_m.sno_final;
        }
        if (p.nb_bsn.m == "y") s.print_record(2061, 1, values(s.bnb_m));
        if (p.ls_bsn.m == "y") s.print_record(2071, 1, values(s.bls_m));
        if (p.pw_bsn.m == "y") s.print_record(2081, 1, values(s.bpw_m));
        sw_init = s.bwb_m.sw_final;
        sno_init = s.bwb_m.sno_final;
        s.bwb_m = s.hwbz;
        s.bwb_m.sw_init = sw_init;
        s.bwb_m.sno_init = sno_init;
        s.bnb_m = s.hnbz;
        s.bls_m = s.hlsz;
        s.bpw_m = s.hpwz;
    }
    if (s.time.end_yr == 1) {
        s.bwb_a = s.bwb_a + s.bwb_y;
        s.bnb_a = s.bnb_a + s.bnb_y;
        s.bls_a = s.bls_a + s.bls_y;
        s.bpw_a = s.bpw_a + s.bpw_y;
        const float c = static_cast<float>(s.time.day_end_yr);
        s.bwb_y = ave(s.bwb_y, c);
        s.bpw_y = ave(s.bpw_y, c);
        if (p.wb_bsn.y == "y") {
            s.bwb_y.sw_final = s.bwb_d.sw_final;
            s.bwb_y.sno_final = s.bwb_d.sno_final;
            s.print_record(2052, 1, values(s.bwb_y));
            s.bwb_y.sw_init = s.bwb_y.sw_final;
            s.bwb_y.sno_init = s.bwb_y.sno_final;
        }
        if (p.nb_bsn.y == "y") s.print_record(2062, 1, values(s.bnb_y));
        if (p.ls_bsn.y == "y") s.print_record(2072, 1, values(s.bls_y));
        if (p.pw_bsn.y == "y") s.print_record(2082, 1, values(s.bpw_y));
        sw_init = s.bwb_y.sw_final;
        sno_init = s.bwb_y.sno_final;
        s.bwb_y = s.hwbz;
        s.bwb_y.sw_init = sw_init;
        s.bwb_y.sno_init = sno_init;
        s.bnb_y = s.hnbz;
        s.bls_y = s.hlsz;
        s.bpw_y = s.hpwz;
    }
    if (s.time.end_sim == 1) {
        sw_init = s.bwb_a.sw_init;
        sno_init = s.bwb_a.sno_init;
        s.bwb_a = s.bwb_a / s.time.yrs_prt;
        s.bwb_a = ave(s.bwb_a, s.time.days_prt);
        s.bwb_a.sw_init = sw_init;
        s.bwb_a.sw_final = s.bwb_d.sw_final;
        s.bwb_a.sno_init = sno_init;
        s.bwb_a.sno_final = s.bwb_d.sno_final;
        if (p.wb_bsn.a == "y") s.print_record(2053, 1, values(s.bwb_a));
        s.ban_precip_aa = s.bwb_a.precip;
        s.bwb_a = s.hwbz;
        s.bnb_a = s.bnb_a / s.time.yrs_prt;
        if (p.nb_bsn.a == "y") s.print_record(2063, 1, values(s.bnb_a));
        s.bls_a = s.bls_a / s.time.yrs_prt;
        if (p.ls_bsn.a == "y") s.print_record(2073, 1, values(s.bls_a));
        s.bpw_a = s.bpw_a / s.time.yrs_prt;
        s.bpw_a = ave(s.bpw_a, s.time.days_prt);
        if (p.pw_bsn.a == "y") s.print_record(2083, 1, values(s.bpw_a));
    }
}

// command.f90's outputs after the day's objects (the tables of the port)
void output_control(Sim& s) {
    if (s.time.yrs <= s.pco.nyskip) return;
    for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) hru_output(s, ihru);
    for (int iaq = 1; iaq <= s.sp_ob.aqu; ++iaq) aquifer_output(s, iaq);
    for (int jrch = 1; jrch <= s.sp_ob.chandeg; ++jrch) sd_channel_output(s, jrch);
    if (s.db_mx.lsu_elem > 0) basin_output(s);
}

}  // namespace swat
