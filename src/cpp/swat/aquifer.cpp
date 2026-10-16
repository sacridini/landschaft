// SPDX-License-Identifier: GPL-3.0-or-later
//
// 1-D aquifers: proc_aqu.f90 (aqu_read.f90, aqu_initial.f90, aqu_read_init.f90) and
// aqu_1d_control.f90 (return flow, seepage, revap, nitrate and phosphorus). Pesticides,
// salts and constituents, water allocation and the geomorphic baseflow of aqu_cha.lin
// are outside the port.

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/hyd_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

// aqu_read.f90
void aqu_read(Sim& s) {
    const std::string& file = s.in_aqu.aqu;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.aqudb.alloc(0, 0, aquifer_database());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 500)) < 0) break;
        int imax = 0, msh_aqp = 0, eof = 0;
        while (eof == 0) {
            int i = 0;
            eof = f.read(i);
            if (eof < 0) break;
            imax = std::max(imax, i);
            ++msh_aqp;
        }
        s.db_mx.aqudb = msh_aqp;
        s.aqudb.alloc(0, imax, aquifer_database());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 500)) < 0) break;
        for (int ish = 1; ish <= msh_aqp; ++ish) {
            int i = 0, k = 0;
            if (f.read(i) < 0) break;
            f.backspace();
            if (f.read(k, items(s.aqudb(i))) < 0) break;
        }
        break;
    }
}

// aqu_initial.f90
void aqu_initial(Sim& s) {
    const int n = s.sp_ob.aqu;
    s.aqu_om_init.alloc(1, n, aquifer_dynamic());
    s.aqu_d.alloc(1, n, aquifer_dynamic());
    s.aqu_dat.alloc(1, n, aquifer_database());
    s.aqu_prm.alloc(1, n, aquifer_data_parameters());
    for (int iaq = 1; iaq <= n; ++iaq) {
        const int iob = s.sp_ob1.aqu + iaq - 1;
        const int iaqdb = s.ob(iob).props;
        s.aqu_dat(iaq) = s.aqudb(iaqdb);
        const aquifer_database& a = s.aqu_dat(iaq);
        aquifer_data_parameters& p = s.aqu_prm(iaq);
        aquifer_dynamic& d = s.aqu_d(iaq);
        p.area_ha = s.ob(iob).area_ha;
        p.alpha_e = std::exp(-a.alpha);
        p.nloss = std::exp(-0.693f / (a.hlife_n + 0.1f));
        d.flo = a.flo;
        d.dep_wt = a.dep_wt;
        // the reference takes the water table of aqu_d(iaqdb), the aquifer numbered as the database entry
        d.stor = 1000.0f * (a.dep_bot - s.aqu_d(iaqdb).dep_wt) * a.spyld;
        d.no3_st = (10.0f * d.flo * p.area_ha) * a.no3 / 1000.0f;
        d.minp = (10.0f * d.flo * p.area_ha) * a.minp / 1000.0f;
        d.cbn = a.cbn;
        d.rchrg = 0.0f;
        d.seep = 0.0f;
        d.revap = 0.0f;
        d.no3_rchg = 0.0f;
        d.no3_loss = 0.0f;
        d.no3_lat = 0.0f;
        d.no3_seep = 0.0f;
        d.flo_cha = 0.0f;
        d.flo_res = 0.0f;
        d.flo_ls = 0.0f;
    }
}

// aqu_read_init.f90 (initial.aqu: read, its constituents unused)
void aqu_read_init(Sim& s) {
    const std::string& file = s.in_aqu.init;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.aqu_init.alloc(0, 0, aquifer_init_data());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        int imax = 0, eof = 0;
        while (eof == 0) {
            eof = f.read(chars(titldum, 80));
            if (eof < 0) break;
            ++imax;
        }
        s.aqu_init.alloc(0, imax, aquifer_init_data());
        s.aqu_init_dat_c.alloc(0, imax, aquifer_init_data_char());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int i = 1; i <= imax; ++i) {
            if (read_items(f, s.aqu_init_dat_c(i)) < 0) break;
            const aquifer_init_data_char& c = s.aqu_init_dat_c(i);
            if ((c.pest != "null" && !c.pest.empty()) || (c.path != "null" && !c.path.empty()) ||
                (c.salt != "null" && !c.salt.empty()))
                throw std::runtime_error("SWAT+ aquifer constituents (initial.aqu) are not supported");
        }
        break;
    }
}

}  // namespace

void proc_aqu(Sim& s) {
    aqu_read(s);
    aqu_initial(s);
    aqu_read_init(s);
}

// aqu_1d_control.f90
void aqu_1d_control(Sim& s) {
    object_connectivity& o = s.ob(s.icmd);
    const int iaq = o.num;
    s.iwst = o.wst;
    aquifer_dynamic& d = s.aqu_d(iaq);
    const aquifer_database& a = s.aqu_dat(iaq);
    aquifer_data_parameters& p = s.aqu_prm(iaq);
    o.hd(1) = s.hz;
    o.hd(2) = s.hz;
    d.rchrg = o.hin.flo / (10.0f * o.area_ha);
    p.rchrg_prev = d.rchrg;
    d.stor = d.stor + d.rchrg;
    if (p.wallo_pod > 0) throw std::runtime_error("SWAT+ water allocation is not supported");
    d.dep_wt = a.dep_bot - (d.stor / (1000.0f * a.spyld));
    d.dep_wt = fmax2(0.0f, d.dep_wt);
    if (d.dep_wt <= a.flo_min) {
        d.flo = d.flo * p.alpha_e + d.rchrg * (1.0f - p.alpha_e);
        d.flo = fmax2(0.0f, d.flo);
        d.flo = fmin2(d.stor, d.flo);
        d.stor = d.stor - d.flo;
    } else {
        d.flo = 0.0f;
    }
    o.hd(1).flo = 10.0f * d.flo * o.area_ha;
    d.seep = d.rchrg * a.seep;
    d.seep = fmin2(d.seep, d.stor);
    o.hd(2).flo = 10.0f * d.seep * o.area_ha;
    d.stor = d.stor - d.seep;
    if (d.dep_wt < a.revap_min) {
        d.revap = s.wst(s.iwst).weat.pet * a.revap_co;
        d.revap = fmin2(d.revap, d.stor);
        d.stor = d.stor - d.revap;
    } else {
        d.revap = 0.0f;
    }
    d.no3_rchg = o.hin.no3 / o.area_ha;
    d.no3_st = d.no3_st + d.no3_rchg;
    p.rchrgn_prev = d.no3_rchg;
    float conc_no3 = 0.0f;
    if (d.stor > 1.e-6f) {
        conc_no3 = d.no3_st / d.stor;
    } else {
        conc_no3 = 0.0f;
    }
    o.hd(1).no3 = (conc_no3 * d.flo) * o.area_ha;
    o.hd(1).no3 = fmin2(o.hd(1).no3, (d.no3_st * o.area_ha));
    d.no3_lat = o.hd(1).no3 / o.area_ha;
    d.no3_st = d.no3_st - d.no3_lat;
    const float no3_init = d.no3_st;
    d.no3_st = d.no3_st * p.nloss;
    d.no3_loss = no3_init - d.no3_st;
    d.no3_seep = conc_no3 * d.seep;
    d.no3_seep = fmin2(d.no3_seep, d.no3_st);
    d.no3_st = d.no3_st - d.no3_seep;
    o.hd(2).no3 = d.no3_seep * o.area_ha;
    d.minp = o.hin.flo * a.minp / 1000.0f;
    o.hd(1).solp = d.minp;
    d.minp = o.hin.flo * a.minp / 1000.0f;
    o.hd(1).solp = d.minp;
    d.flo_cha = 0.0f;
    d.flo_res = 0.0f;
    d.flo_ls = 0.0f;
    for (int iout = 1; iout <= o.src_tot; ++iout) {
        if (o.htyp_out(iout) == "tot") {
            const std::string& t = o.obtyp_out(iout);
            if (t == "cha" || t == "sdc") {
                d.flo_cha = d.flo_cha + d.flo * o.frac_out(iout);
            } else if (t == "res") {
                d.flo_res = d.flo_res + d.flo * o.frac_out(iout);
            } else if (t == "aqu") {
                d.flo_ls = d.flo_ls + d.flo * o.frac_out(iout);
            }
        }
    }
    o.hin_tot = o.hin_tot + o.hin;
    o.hout_tot = o.hout_tot + o.hd(1) + o.hd(2);
    for (int ii = 1; ii <= s.time.step; ++ii) {
        const float step = static_cast<float>(s.time.step);
        o.ts(1, ii) = o.hd(1) / step;
    }
}

}  // namespace swat
