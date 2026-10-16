// SPDX-License-Identifier: GPL-3.0-or-later
//
// SWAT-DEG channels: proc_cha.f90 with its readers (ch_read_init.f90, ch_read_nut.f90,
// sd_hydsed_read.f90, sd_channel_read.f90, sd_hydsed_init.f90, sd_rating_curve.f90,
// overbank_read.f90, sd_channel_surf_link.f90), the rating curve interpolation
// (rcurv_interp_dep.f90, rcurv_interp_flo.f90, qman.f90) and the daily channel
// (sd_channel_control3.f90, sd_channel_sediment3.f90, ch_rtmusk.f90, ch_watqual4.f90,
// ch_temp.f90, theta.f90, wq_semianalyt.f90, wq_k2m.f90). Pesticides, pathogens,
// salts and constituents, flood plain wetlands, gwflow, water allocation and the
// hydrograph separation outputs (hyd_sep) are outside the port.

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/hyd_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

int count_after_header(fio::File& f) {
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

// ch_read_init.f90
void ch_read_init(Sim& s) {
    const std::string& file = s.in_cha.init;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.ch_init.alloc(0, 0, channel_init_datafiles());
        s.sd_init.alloc(0, 0, swatdeg_init_datafiles());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        const int imax = count_after_header(f);
        s.db_mx.ch_init = imax;
        s.ch_init.alloc(0, imax, channel_init_datafiles());
        s.sd_init.alloc(0, imax, swatdeg_init_datafiles());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int ich = 1; ich <= s.db_mx.ch_init; ++ich)
            if (read_items(f, s.ch_init(ich)) < 0) break;
        break;
    }
}

// ch_read_nut.f90
void ch_read_nut(Sim& s) {
    const std::string& file = s.in_cha.nut;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.ch_nut.alloc(0, 0, channel_nut_data());
        return;
    }
    fio::File f = s.files.open(file);
    const float step = static_cast<float>(s.time.step);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        const int imax = count_after_header(f);
        s.db_mx.ch_nut = imax;
        s.ch_nut.alloc(0, imax, channel_nut_data());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int ich = 1; ich <= s.db_mx.ch_nut; ++ich) {
            if (f.read(chars(titldum, 80)) < 0) break;
            f.backspace();
            if (read_items(f, s.ch_nut(ich)) < 0) break;
            channel_nut_data& n = s.ch_nut(ich);
            if (n.lao <= 0) n.lao = 2;
            if (n.igropt <= 0) n.igropt = 2;
            if (n.ai0 <= 0.0f) n.ai0 = 50.0f;
            if (n.ai1 <= 0.0f) n.ai1 = 0.08f;
            if (n.ai2 <= 0.0f) n.ai2 = 0.015f;
            if (n.ai3 <= 0.0f) n.ai3 = 1.60f;
            if (n.ai4 <= 0.0f) n.ai4 = 2.0f;
            if (n.ai5 <= 0.0f) n.ai5 = 3.5f;
            if (n.ai6 <= 0.0f) n.ai6 = 1.07f;
            if (n.mumax <= 0.0f) n.mumax = 2.0f;
            if (n.rhoq <= 0.0f) n.rhoq = 2.5f;
            if (n.tfact <= 0.0f) n.tfact = 0.3f;
            if (n.k_l <= 0.0f) n.k_l = 0.75f;
            if (n.k_n <= 0.0f) n.k_n = 0.02f;
            if (n.k_p <= 0.0f) n.k_p = 0.025f;
            if (n.lambda0 <= 0.0f) n.lambda0 = 1.0f;
            if (n.lambda1 <= 0.0f) n.lambda1 = 0.03f;
            if (n.lambda2 <= 0.0f) n.lambda2 = 0.054f;
            if (n.p_n <= 0.0f) n.p_n = 0.5f;
            n.k_l = n.k_l * 1.e-3f * 60.0f;
            n.mumax = n.mumax / step;
            n.rhoq = n.rhoq / step;
            if (n.rs1 <= 0.0f) n.rs1 = 1.0f;
            if (n.rs2 <= 0.0f) n.rs2 = 0.05f;
            if (n.rs3 <= 0.0f) n.rs3 = 0.5f;
            if (n.rs4 <= 0.0f) n.rs4 = 0.05f;
            if (n.rs5 <= 0.0f) n.rs5 = 0.05f;
            if (n.rs6 <= 0.0f) n.rs6 = 2.5f;
            if (n.rs7 <= 0.0f) n.rs7 = 2.5f;
            if (n.rk1 <= 0.0f) n.rk1 = 1.71f;
            if (n.rk2 <= 0.0f) n.rk2 = 1.0f;
            if (n.rk4 <= 0.0f) n.rk4 = 2.0f;
            if (n.rk5 <= 0.0f) n.rk5 = 2.0f;
            if (n.rk6 <= 0.0f) n.rk6 = 1.71f;
            if (n.bc1 <= 0.0f) n.bc1 = 0.55f;
            if (n.bc2 <= 0.0f) n.bc2 = 1.1f;
            if (n.bc3 <= 0.0f) n.bc3 = 0.21f;
            if (n.bc4 <= 0.0f) n.bc4 = 0.35f;
            n.rs1 = n.rs1 / step;
            n.rs2 = n.rs2 / step;
            n.rs3 = n.rs3 / step;
            n.rs4 = n.rs4 / step;
            n.rs5 = n.rs5 / step;
            n.rk1 = n.rk1 / step;
            n.rk2 = n.rk2 / step;
            n.rk3 = n.rk3 / step;
            n.rk4 = n.rk4 / step;
            n.bc1 = n.bc1 / step;
            n.bc2 = n.bc2 / step;
            n.bc3 = n.bc3 / step;
            n.bc4 = n.bc4 / step;
        }
        break;
    }
}

// sd_hydsed_read.f90
void sd_hydsed_read(Sim& s) {
    std::string titldum, header;
    s.maxint = 10;
    const int ts_sed = std::max(10, s.time.step);
    s.timeint.alloc(1, ts_sed, 0.0f);
    s.hyd_rad.alloc(1, ts_sed, 0.0f);
    s.trav_time.alloc(1, ts_sed, 0.0f);
    s.flo_dep.alloc(1, ts_sed, 0.0f);
    int imax = 0;
    const std::string& file = s.in_cha.hyd_sed;
    if (!s.files.exists(file) || file == "null") {
        s.sd_chd.alloc(0, 0, swatdeg_hydsed_data());
    } else {
        fio::File f = s.files.open(file);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            imax = count_after_header(f);
            s.db_mx.ch_lte = imax;
            s.sd_chd.alloc(0, imax, swatdeg_hydsed_data());
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int idb = 1; idb <= s.db_mx.ch_lte; ++idb)
                if (read_items(f, s.sd_chd(idb)) < 0) break;
            break;
        }
    }
    if (!s.files.exists("sed_nut.cha")) {
        s.sd_chd1.alloc(1, 1, swatdeg_sednut_data());
    } else {
        // the reference keeps counting from the hydrology file's imax
        fio::File f = s.files.open("sed_nut.cha");
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            imax += count_after_header(f);
            s.db_mx.ch_sednut = imax;
            s.sd_chd1.alloc(0, imax, swatdeg_sednut_data());
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int idb = 1; idb <= s.db_mx.ch_sednut; ++idb)
                if (read_items(f, s.sd_chd1(idb)) < 0) break;
            break;
        }
    }
}

// sd_channel_read.f90 (the allocations of the channel arrays and channel-lte.cha)
void sd_channel_read(Sim& s) {
    const int n = s.sp_ob.chandeg;
    s.gully.alloc(0, 0, gully_data());
    s.sd_ch.alloc(0, n, swatdeg_channel_dynamic());
    s.ch_morph.alloc(0, n, channel_morphology_output());
    s.ch_sed_bud.alloc(0, n, channel_sediment_budget_output());
    s.ch_rcurv.alloc(0, n, channel_rating_curve());
    s.sd_ch_vel.alloc(0, n, channel_velocity_parameters());
    s.chsd_d.alloc(0, n, sd_ch_output());
    s.ch_stor.alloc(0, n, hyd_output());
    s.fp_stor.alloc(0, n, hyd_output());
    s.tot_stor.alloc(0, n, hyd_output());
    s.wet_stor.alloc(0, n, hyd_output());
    s.ch_fp_wb.alloc(0, n, channel_floodplain_water_balance());
    s.ch_wat_d.alloc(0, n, water_body());
    s.ch_in_d.alloc(0, n, hyd_output());
    s.ch_out_d.alloc(0, n, hyd_output());
    s.ch_om_water_init.alloc(0, n, hyd_output());
    s.fp_om_water_init.alloc(0, n, hyd_output());
    std::string titldum, header;
    const std::string& file = s.in_cha.chan_ez;
    if (!s.files.exists(file) || file == "null") {
        s.sd_dat.alloc(0, 0, swatdeg_datafiles());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        int imax = 0, eof = 0;
        while (eof == 0) {
            int i = 0;
            eof = f.read(i);
            if (eof < 0) break;
            imax = std::max(imax, i);
        }
        s.db_mx.sdc_dat = imax;
        s.sd_dat.alloc(0, imax, swatdeg_datafiles());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int ichi = 1; ichi <= s.db_mx.sdc_dat; ++ichi) {
            int i = 0, k = 0;
            if (f.read(i) < 0) break;
            f.backspace();
            swatdeg_datafiles& d = s.sd_dat(i);
            if (f.read(k, chars(d.name, d.name_len), chars(d.initc, d.initc_len), chars(d.hydc, d.hydc_len),
                       chars(d.sedc, d.sedc_len), chars(d.nutc, d.nutc_len)) < 0)
                break;
            // the reference looks the names up with ichi, not i
            swatdeg_datafiles& c = s.sd_dat(ichi);
            for (int isp_ini = 1; isp_ini <= s.db_mx.ch_init; ++isp_ini) {
                if (fstrcmp(c.initc, s.ch_init(isp_ini).name) == 0) {
                    c.init = isp_ini;
                    for (int ics = 1; ics <= s.db_mx.om_water_init; ++ics) {
                        if (fstrcmp(s.ch_init(isp_ini).org_min, s.om_init_name(ics)) == 0) {
                            s.sd_init(isp_ini).org_min = ics;
                            break;
                        }
                    }
                    if (s.ch_init(isp_ini).pest != "null" || s.ch_init(isp_ini).path != "null" ||
                        s.ch_init(isp_ini).salt != "null")
                        throw std::runtime_error("SWAT+ channel constituents (initial.cha) are not supported");
                }
            }
            for (int ihydsed = 1; ihydsed <= s.db_mx.ch_lte; ++ihydsed) {
                if (fstrcmp(s.sd_chd(ihydsed).name, c.hydc) == 0) {
                    c.hyd = ihydsed;
                    break;
                }
            }
            for (int ihydsed = 1; ihydsed <= s.db_mx.ch_sednut; ++ihydsed) {
                if (fstrcmp(s.sd_chd1(ihydsed).name, c.hydc) == 0) {
                    c.sednut = ihydsed;
                    break;
                }
            }
            for (int inut = 1; inut <= s.db_mx.ch_nut; ++inut) {
                if (fstrcmp(s.ch_nut(inut).name, c.nutc) == 0) {
                    c.nut = inut;
                    break;
                }
            }
        }
        break;
    }
}

float qman(float x1, float x2, float x3, float x4) { return x1 * fpow(x2, 0.6666f) * std::sqrt(x4) / (x3 + 0.001f); }

// sd_rating_curve.f90
void sd_rating_curve(Sim& s, int i) {
    swatdeg_channel_dynamic& c = s.sd_ch(i);
    channel_rating_curve& rc = s.ch_rcurv(i);
    float b = c.chw - 2.0f * c.chd * c.chss;
    if (b <= 0.0f) {
        b = 0.5f * c.chw;
        b = fmax2(0.0f, b);
        c.chss = (c.chw - b) / (2.0f * c.chd);
    }
    float p_bf = 0.0f, a_bf = 0.0f, vol_bf = 0.0f;
    for (int i_dep = 1; i_dep <= 2; ++i_dep) {
        float dep = 0.0f;
        if (i_dep == 1) dep = 0.1f * c.chd;
        if (i_dep == 2) dep = c.chd;
        const float p = b + 2.0f * std::sqrt(fpowi(dep, 2) * (1.0f + 1.0f / fpowi(c.chss, 2)));
        const float a = b * dep + dep / c.chss;
        const float rh = a / p;
        channel_rating_curve_parameters& e = rc.elev(i_dep);
        e.dep = dep;
        e.wet_perim = p;
        e.xsec_area = a;
        e.top_wid = b + 2.0f * dep * c.chss;
        e.surf_area = e.top_wid * c.chl;
        e.vol = a * c.chl * 1000.0f;
        e.vol_ch = e.vol;
        e.vol_fp = 0.0f;
        e.flo_rate = qman(a, rh, c.chn, c.chs);
        e.vel = qman(1.0f, rh, c.chn, c.chs);
        e.ttime = c.chl / (3.6f * e.vel);
        if (i_dep == 2) {
            p_bf = p;
            a_bf = a;
            vol_bf = e.vol_ch;
        }
    }
    for (int i_dep = 1; i_dep <= 2; ++i_dep) {
        float frac_abov = 0.0f;
        if (i_dep == 1) frac_abov = 0.2f;
        if (i_dep == 2) frac_abov = 1.0f;
        const float dep = frac_abov * c.chd;
        const float p = p_bf + 2.0f * std::sqrt(fpowi(dep, 2) * (1.0f + 1.0f / fpowi(c.fps, 2)));
        const float a = a_bf + b * dep + fpowi(dep, 2) / c.fps;
        const float rh = a / p;
        channel_rating_curve_parameters& e = rc.elev(i_dep + 2);
        e.dep = (1.0f + frac_abov) * c.chd;
        e.wet_perim = p;
        e.xsec_area = a;
        e.top_wid = c.chw + 2.0f * dep / c.fps;
        e.surf_area = e.top_wid * c.chl;
        e.vol_ch = vol_bf;
        e.vol_fp = e.top_wid * dep * c.chl * 1000.0f;
        e.vol = vol_bf + e.vol_fp;
        e.flo_rate = qman(a, rh, c.fpn, c.chs);
        e.vel = qman(1.0f, rh, c.fpn, c.chs);
        e.ttime = c.chl / (3.6f * e.vel);
    }
}

}  // namespace

// rcurv_interp_dep.f90
void rcurv_interp_dep(Sim& s, int icha, float flow_dep) {
    channel_rating_curve& rc = s.ch_rcurv(icha);
    for (int ielev = 1; ielev <= rc.npts; ++ielev) {
        if (flow_dep < rc.elev(ielev).dep) {
            if (ielev == 1) {
                const float rto = flow_dep / rc.elev(ielev).dep;
                s.rcurv = rc.elev(ielev) * rto;
                s.rcurv.ttime = rc.elev(ielev).ttime;
                break;
            }
            if (ielev > 1) {
                const float rto = (flow_dep - rc.elev(ielev - 1).dep) / (rc.elev(ielev).dep - rc.elev(ielev - 1).dep);
                chrc_interp(rc.elev(ielev - 1), rc.elev(ielev), rto, s.rcurv);
                break;
            }
        }
        if (ielev == 4) {
            const float rto = 1.0f + (flow_dep - rc.elev(ielev).dep) / rc.elev(ielev).dep;
            s.rcurv = rc.elev(ielev) * rto;
            s.rcurv.ttime = s.rcurv.ttime / rto;
            break;
        }
    }
}

// rcurv_interp_flo.f90
void rcurv_interp_flo(Sim& s, int icha, float flo_rate) {
    channel_rating_curve& rc = s.ch_rcurv(icha);
    for (int ielev = 1; ielev <= rc.npts; ++ielev) {
        if (flo_rate < rc.elev(ielev).flo_rate) {
            if (ielev == 1) {
                const float rto = flo_rate / rc.elev(ielev).flo_rate;
                s.rcurv = rc.elev(ielev) * rto;
                s.rcurv.ttime = rc.elev(ielev).ttime;
                break;
            }
            if (ielev > 1) {
                const float rto = (flo_rate - rc.elev(ielev - 1).flo_rate) /
                                  (rc.elev(ielev).flo_rate - rc.elev(ielev - 1).flo_rate);
                chrc_interp(rc.elev(ielev - 1), rc.elev(ielev), rto, s.rcurv);
                break;
            }
        }
        if (ielev == 4) {
            const float rto = 1.0f + (flo_rate - rc.elev(ielev).flo_rate) / rc.elev(ielev).flo_rate;
            s.rcurv = rc.elev(ielev) * rto;
            s.rcurv.ttime = s.rcurv.ttime / rto;
            break;
        }
    }
}

namespace {

// sd_hydsed_init.f90
void sd_hydsed_init(Sim& s) {
    for (int i = 1; i <= s.sp_ob.chandeg; ++i) {
        s.icmd = s.sp_ob1.chandeg + i - 1;
        const int idat = s.ob(s.icmd).props;
        const int idb = s.sd_dat(idat).hyd;
        const int idb1 = s.sd_dat(idat).sednut;
        swatdeg_channel_dynamic& c = s.sd_ch(i);
        const swatdeg_hydsed_data& d = s.sd_chd(idb);
        const swatdeg_sednut_data& d1 = s.sd_chd1(idb1);
        c.name = d.name;
        c.obj_no = s.icmd;
        c.order = d.order;
        c.chw = d.chw;
        c.chd = d.chd;
        c.chs = d.chs;
        if (c.chs < 1.e-9f) c.chs = 0.000001f;
        c.chl = d.chl;
        c.chn = d.chn;
        if (c.chn < 0.01f) c.chn = 0.01f;
        c.chk = d.chk;
        c.bank_exp = d.bank_exp;
        c.cov = d.cov;
        c.sinu = d.sinu;
        if (c.sinu < 1.05f) c.sinu = 1.05f;
        c.vcr_coef = d.vcr_coef;
        c.d50 = d.d50;
        c.ch_clay = d.ch_clay;
        c.carbon = d.carbon;
        c.ch_bd = d.ch_bd;
        c.chss = d.chss;
        c.n_conc = d.n_conc;
        c.p_conc = d.p_conc;
        c.p_bio = d.p_bio;
        c.bankfull_flo = d.bankfull_flo;
        if (c.bankfull_flo <= 1.e-6f) c.bankfull_flo = 0.0f;
        c.fps = d.fps;
        if (c.fps > c.chs) c.fps = c.chs;
        if (c.fps <= 1.e-6f) c.fps = 0.00001f;
        c.fpn = d.fpn;
        c.hc_kh = s.gully(0).hc_kh;
        c.hc_hgt = s.gully(0).hc_hgt;
        c.hc_ini = s.gully(0).hc_ini;
        c.pk_rto = d1.pk_rto;
        c.fp_inun_days = d1.fp_inun_days;
        c.n_setl = d1.n_setl;
        c.p_setl = d1.p_setl;
        c.n_sol_part = d1.n_sol_part;
        c.p_sol_part = d1.p_sol_part;
        c.n_dep_enr = d1.n_dep_enr;
        c.p_dep_enr = d1.p_dep_enr;
        c.arc_len_fr = d1.arc_len_fr;
        c.bed_exp = d1.bed_exp;
        c.wash_bed_fr = d1.wash_bed_fr;
        const float kh = c.hc_kh;
        if (kh > 1.e-6f) {
            c.hc_co = 0.37f * (17.83f + 16.56f * kh - 15.0f * c.cov);
            c.hc_co = fmax2(0.0f, c.hc_co);
        } else {
            c.hc_co = 0.0f;
        }
        float chside = c.chss;
        float b = c.chw - 2.0f * c.chd * chside;
        if (b <= 0.0f) {
            b = 0.5f * c.chw;
            b = fmax2(0.0f, b);
            chside = (c.chw - b) / (2.0f * c.chd);
        }
        s.sd_ch_vel(i).wid_btm = b;
        s.sd_ch_vel(i).dep_bf = c.chd;
        sd_rating_curve(s, i);
        for (int i_dep = 1; i_dep <= 2; ++i_dep) {
            float dep = 0.0f;
            if (i_dep == 1) dep = 0.1f * c.chd;
            if (i_dep == 2) dep = c.chd;
            const float p = b + 2.0f * std::sqrt(fpowi(dep, 2) * (1.0f + 1.0f / fpowi(c.chss, 2)));
            const float a = b * dep + dep / c.chss;
            const float rh = a / p;
            const float vel = qman(1.0f, rh, c.chn, c.chs);
            const float celerity = vel * 5.0f / 3.0f;
            if (i_dep == 1) {
                c.stor_dis_01bf = c.chl / (3.6f * celerity);
            } else {
                c.stor_dis_bf = c.chl / (3.6f * celerity);
            }
        }
        const basin_parms& bp = s.bsn_prm;
        const float msk1 = bp.msk_co1 / (bp.msk_co1 + bp.msk_co2);
        const float msk2 = bp.msk_co2 / (bp.msk_co1 + bp.msk_co2);
        const float xkm = c.stor_dis_bf * msk1 + c.stor_dis_01bf * msk2;
        const float detmax = 2.0f * xkm * (1.0f - bp.msk_x);
        float det = s.time.dtm / 60.0f;
        c.msk.substeps = 1;
        if (det > detmax) c.msk.substeps = static_cast<int>(det / detmax) + 1;
        if (s.bsn_cc.rte == 0 && s.time.step <= 1) c.msk.substeps = 1;
        c.msk.nsteps = s.time.step * c.msk.substeps;
        if (c.msk.nsteps > 0) {
            c.in1_vol = s.rcurv.flo_rate / (86400.0f / static_cast<float>(c.msk.nsteps));
            c.out1_vol = s.rcurv.flo_rate / (86400.0f / static_cast<float>(c.msk.nsteps));
        }
        det = det / static_cast<float>(c.msk.substeps);
        const float denom = 2.0f * xkm * (1.0f - bp.msk_x) + det;
        c.msk.c1 = (det - 2.0f * xkm * bp.msk_x) / denom;
        c.msk.c1 = fmax2(0.0f, c.msk.c1);
        c.msk.c2 = (det + 2.0f * xkm * bp.msk_x) / denom;
        c.msk.c3 = (2.0f * xkm * (1.0f - bp.msk_x) - det) / denom;
        const float sumc = c.msk.c1 + c.msk.c2 + c.msk.c3;
        c.msk.c1 = c.msk.c1 / sumc;
        c.msk.c2 = c.msk.c2 / sumc;
        c.msk.c3 = c.msk.c3 / sumc;
    }
    for (int ich = 1; ich <= s.sp_ob.chandeg; ++ich) {
        if (s.sd_ch(ich).chl > 1.e-3f) {
            const int iob = s.sp_ob1.chandeg + ich - 1;
            const int ichdat = s.ob(iob).props;
            const int ich_ini = s.sd_dat(ichdat).init;
            const int iom_ini = s.sd_init(ich_ini).org_min;
            s.tot_stor(ich) = s.om_init_water(iom_ini);
            const float flow_dep = s.om_init_water(iom_ini).flo * s.sd_ch(ich).chd;
            rcurv_interp_dep(s, ich, flow_dep);
            s.ch_rcurv(ich).in1 = s.rcurv;
            s.ch_rcurv(ich).out1 = s.rcurv;
            s.tot_stor(ich).flo = s.rcurv.vol;
            hyd_convert_conc_to_mass(s.tot_stor(ich));
            if (s.om_init_water(iom_ini).flo <= 1.0f) {
                s.ch_stor(ich) = s.tot_stor(ich);
                s.fp_stor(ich) = s.hz;
            } else {
                const float rto = s.rcurv.vol_ch / s.rcurv.vol;
                s.ch_stor(ich) = rto * s.tot_stor(ich);
                const float rto1 = 1.0f - rto;
                s.fp_stor(ich) = rto1 * s.tot_stor(ich);
            }
        } else {
            s.ch_stor(ich) = s.hz;
            s.fp_stor(ich) = s.hz;
        }
        s.ch_om_water_init(ich) = s.ch_stor(ich);
        s.fp_om_water_init(ich) = s.fp_stor(ich);
    }
}

}  // namespace

// proc_cha.f90 (swat-deg channels only; the old channels, sp_ob%chan, are refused by hyd_connect)
void proc_cha(Sim& s) {
    ch_read_init(s);
    sd_hydsed_read(s);
    ch_read_nut(s);
    sd_channel_read(s);
    sd_hydsed_init(s);
    // aqu2d_init: aqu_cha.lin is refused by hyd_connect (db_mx%aqu2d = 0)
    // overbank_read / sd_channel_surf_link: chan-surf.lin links flood plains to HRUs
    if (s.files.exists(s.in_link.chan_surf) && s.in_link.chan_surf != "null")
        throw std::runtime_error("SWAT+ channel flood plain links (chan-surf.lin) are not supported");
    // ch_temp reads only w_temp(0): the defaults of water_temperature_data
    s.w_temp.alloc(0, 0, water_temperature_data());
    time_conc_init(s);
}

// om_water_init.f90 (initial water quality of the water bodies)
void om_water_init(Sim& s) {
    const std::string& file = s.in_init.om_water;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.om_init_water.alloc(0, 0, hyd_output());
        s.om_init_name.alloc(0, 0, "");
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        const int imax = count_after_header(f);
        s.db_mx.om_water_init = imax;
        s.om_init_water.alloc(0, imax, hyd_output());
        s.om_init_name.alloc(0, imax, "");
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int ichi = 1; ichi <= s.db_mx.om_water_init; ++ichi) {
            if (f.read(chars(titldum, 80)) < 0) break;
            f.backspace();
            if (f.read(chars(s.om_init_name(ichi), 16), items(s.om_init_water(ichi))) < 0) break;
        }
        break;
    }
}

// cli_lapse.f90
void cli_lapse(Sim& s) {
    for (int iob = 1; iob <= s.sp_ob.objs; ++iob) {
        object_connectivity& o = s.ob(iob);
        const int iwst = o.wst;
        const weather_station& w = s.wst(iwst);
        if (w.wco_c.pgage == "sim") {
            const int iwgn = w.wco.wgn;
            o.plaps = s.bsn_prm.plaps * (o.elev - s.wgn(iwgn).elev) / 1000.0f;
        } else {
            const int igage = w.wco.pgage;
            o.plaps = s.bsn_prm.plaps * (o.elev - s.pcp(igage).elev) / 1000.0f;
        }
        if (w.wco_c.tgage == "sim") {
            const int iwgn = w.wco.wgn;
            o.tlaps = s.bsn_prm.tlaps * (s.wgn(iwgn).elev - o.elev) / 1000.0f;
        } else {
            const int igage = w.wco.tgage;
            o.tlaps = s.bsn_prm.tlaps * (s.tmp(igage).elev - o.elev) / 1000.0f;
        }
    }
}

// ---- the daily channel ----------------------------------------------------------------

namespace {

float theta(float r20, float thk, float tmp) { return r20 * fpow(thk, tmp - 20.0f); }

float wq_semianalyt(float tres, float tdel, float term_m, float prock, float cprev, float cint) {
    const float help1 = 1.0f / tres - prock;
    const float help2 = exp_w(-tdel * help1);
    const float help3 = cint / tres + term_m;
    const float help4 = help3 / help1;
    const float term1 = cprev * help2;
    const float term2 = help4 * (1.0f - help2);
    return term1 + term2;
}

float wq_k2m(float t1, float t2, float tk, float c1, float c2) {
    (void)wq_semianalyt(t1, t2, 0.0f, 0.0f, c1, c2);
    const float h2 = wq_semianalyt(t1, t2, 0.0f, tk, c1, c2);
    const float help = exp_w(-t2 / t1);
    const float tm = (h2 - c1 * help) / (t1 * (1.0f - help)) - c2 / t1;
    return tm;
}

// sd_channel_sediment3.f90
void sd_channel_sediment3(Sim& s) {
    s.ich = s.isdch;
    const int ich = s.ich;
    const int iob = s.sp_ob1.chandeg + s.jrch - 1;
    (void)iob;
    swatdeg_channel_dynamic& c = s.sd_ch(ich);
    float ebtm_m = 0.0f, ebank_m = 0.0f, ebtm_t = 0.0f, ebank_t = 0.0f;
    s.fp_dep = s.hz;
    s.ch_dep = s.hz;
    s.bank_ero = s.hz;
    s.bed_ero = s.hz;
    s.ch_trans = s.hz;
    s.ch_wat_d(ich).precip = 0.0f;
    c.bankfull_flo = 1.5f;
    c.fp_inun_days = 1.0f;
    c.d50 = 1.0f;
    c.bed_exp = 1.0f;
    object_connectivity& o = s.ob(s.icmd);
    hyd_output& ht1 = s.ht1;
    if (ht1.flo > 1.e-6f) {
        const float pk_rto = c.pk_rto * (1.0f + 2.66f * fpow(o.area_ha / 100.0f, -0.3f));
        s.peakrate = pk_rto * ht1.flo / 86400.0f;
        rcurv_interp_flo(s, ich, s.peakrate);
        const float vel = s.rcurv.vel;
        s.rttime = c.chl / (3.6f * vel);
        s.sd_ch_vel(ich).vel = vel;
        s.sd_ch_vel(ich).rttime = s.rttime;
        s.ch_wat_d(ich).area_ha = c.chl * c.chw / 10.0f;
        const float precip = 10.0f * s.wst(s.iwst).weat.precip * s.ch_wat_d(ich).area_ha;
        s.ch_wat_d(ich).precip = precip;
        float rto = precip / ht1.flo;
        for (int k = 1; k <= o.tsin.size(); ++k) o.tsin(k) = (1.0f + rto) * o.tsin(k);
        ht1.flo = ht1.flo + precip;
        const float bf_flow = c.bankfull_flo * s.ch_rcurv(ich).elev(2).flo_rate;
        if (s.peakrate > bf_flow) {
            float trap_eff = 0.05f * std::log(c.fp_inun_days) + 0.1f;
            const float fp_m2 = 5.0f * c.chw * c.chl * 1000.0f;
            const float exp_co = 0.0001f * fp_m2 / s.rcurv.flo_rate;
            const float ave_rate = ht1.flo / 86400.0f;
            trap_eff = c.fp_inun_days * (s.rcurv.flo_rate / ave_rate) * (1.0f - std::exp(-exp_co));
            trap_eff = fmin2(1.0f, trap_eff);
            s.fp_dep.sed = trap_eff * ht1.sed;
            s.fp_dep.orgn = trap_eff * c.n_dep_enr * ht1.orgn;
            s.fp_dep.sedp = trap_eff * c.p_dep_enr * ht1.sedp;
            s.fp_dep.no3 = 0.0f;
            s.fp_dep.solp = 0.0f;
            ht1 = ht1 - s.fp_dep;
            // flood plain wetlands (fp%hru_tot) are outside the port: none
        }
        float cohesion = (-87.1f + (42.82f * c.ch_clay) - (0.261f * fpow(c.ch_clay, 2.0f)) +
                          (0.029f * fpow(c.ch_clay, 3.0f)));
        cohesion = fmax2(1000.0f, cohesion);
        const float veg = std::exp(-5.0f * c.chd) * 4500.0f * c.cov;
        const float bd_fac = fmax2(0.001f, 0.03924f * c.ch_bd * 1000.0f - 1000.0f);
        const float cohes_fac = 0.021f * cohesion + veg;
        float vel_cr = std::log10(2200.0f * c.chd) * fpow(0.0004f * (bd_fac + cohes_fac), 0.5f);
        vel_cr = c.vcr_coef * vel_cr;
        const float rad_curv = ((12.0f * c.chw) * fpow(c.sinu, 1.5f)) / (13.0f * fpow(c.sinu - 1.0f, 0.5f));
        const float vel_bend = vel * (1.0f / rad_curv + 1.0f);
        const float vel_rch = 0.33f * vel_bend + 0.66f * vel;
        float b_exp = 12.3f / std::sqrt(c.ch_clay + 1.0f);
        b_exp = fmin2(3.5f, b_exp);
        if (vel_rch > vel_cr) {
            const float dur_scale = 0.0001f * fpow(o.area_ha / 100.0f, -0.0858f);
            const float v_vc = dur_scale * c.chw * (1.0f / (1.0f + exp_w(-4.0f * (vel_rch / vel_cr - 1.0f))));
            const float m_exhaust = 0.0002f * c.chw;
            ebank_m = 1.0f / (1.0f / v_vc + 1.0f / m_exhaust);
        } else {
            ebank_m = 0.0f;
        }
        const float arc_len = 0.25f * c.chl;
        ebank_t = 1000.0f * ebank_m * c.chd * arc_len * c.ch_bd;
        s.bank_ero.sed = ebank_t;
        s.bank_ero.orgn = s.bank_ero.sed * c.n_conc / 1000.0f;
        s.bank_ero.sedp = (1.0f - c.p_bio) * s.bank_ero.sed * c.p_conc / 1000.0f;
        s.bank_ero.no3 = 0.0f;
        s.bank_ero.solp = c.p_bio * s.bank_ero.sed * c.p_conc / 1000.0f;
        s.bank_ero.no2 = 0.0f;
        rto = s.bank_ero.flo / ht1.flo;
        for (int k = 1; k <= o.tsin.size(); ++k) o.tsin(k) = (1.0f - rto) * o.tsin(k);
        ht1 = ht1 + s.bank_ero;
        s.ch_dep.sed = c.wash_bed_fr * s.bank_ero.sed;
        s.ch_dep.orgn = c.n_dep_enr * c.wash_bed_fr * s.bank_ero.orgn;
        s.ch_dep.sedp = c.p_dep_enr * c.wash_bed_fr * s.bank_ero.sedp;
        ht1 = ht1 - s.ch_dep;
        if (c.chs > 0.000001f) {
            vel_cr = 0.293f * fpow(c.d50, 0.5f);
            if (vel > vel_cr) ebtm_m = 0.0001f * fpow(vel / vel_cr, c.bed_exp);
            ebtm_t = 1000.0f * ebtm_m * c.chw * c.chl * c.ch_bd;
        }
        s.bed_ero.sed = c.wash_bed_fr * ebtm_t;
        s.bed_ero.orgn = s.bed_ero.sed * c.n_conc;
        s.bed_ero.sedp = (1.0f - c.p_bio) * s.bed_ero.sed * c.p_conc;
        s.bed_ero.no3 = 0.0f;
        s.bed_ero.solp = c.p_bio * s.bed_ero.sed * c.p_conc;
        s.bed_ero.no2 = 0.0f;
        channel_morphology_output& m = s.ch_morph(ich);
        m.num = 1;
        m.wid = c.chw;
        m.dep = c.chd;
        m.fp_km2 = 5.0f * c.chw * c.chl / 1000.0f;
        m.fp_t = s.fp_dep.sed;
        m.fp_mm = m.fp_t / (5.0f * c.chw * c.chl);
        m.ebank_m = ebank_m;
        m.ebank_t = ebank_t;
        m.ebtm_m = ebtm_m;
        m.ebtm_t = ebtm_t;
    }
}

// ch_rtmusk.f90 (no flood plain wetlands)
void ch_rtmusk(Sim& s) {
    s.jrch = s.isdch;
    const int jrch = s.jrch;
    s.jhyd = s.sd_dat(jrch).hyd;
    swatdeg_channel_dynamic& c = s.sd_ch(jrch);
    object_connectivity& o = s.ob(s.icmd);
    hyd_output& ht1 = s.ht1;
    hyd_output& ht2 = s.ht2;
    ht2 = s.hz;
    o.hyd_flo.fill(0.0f);
    s.hyd_rad.fill(0.0f);
    s.trav_time.fill(0.0f);
    s.flo_dep.fill(0.0f);
    float trans_loss = 0.0f;
    float evap = 0.0f;
    s.ch_wat_d(jrch).evap = 0.0f;
    s.ch_wat_d(jrch).seep = 0.0f;
    float sum_inflo = 0.0f;
    for (int k = 1; k <= o.tsin.size(); ++k) sum_inflo = sum_inflo + o.tsin(k);
    s.wet_stor(jrch) = s.hz;
    const float wet_stor_init = s.wet_stor(jrch).flo;
    const float ch_stor_init = s.ch_stor(jrch).flo;
    const float fp_stor_init = s.fp_stor(jrch).flo;
    const float tot_stor_init = ch_stor_init + fp_stor_init;
    if (s.time.step == 1) {
        c.msk.nsteps = 1;
        c.msk.substeps = 1;
    }
    int irtstep = 1;
    int isubstep = 0;
    const float dts = s.time.dtm / static_cast<float>(c.msk.substeps) * 60.0f;
    const float dthr = dts / 3600.0f;
    float& inflo = s.saved.ch_rtmusk_inflo;  // a SAVEd local: kept when the inflow is zero
    for (int ii = 1; ii <= c.msk.nsteps; ++ii) {
        ++isubstep;
        if (isubstep > c.msk.substeps) {
            ++irtstep;
            isubstep = 1;
        }
        if (ht1.flo > 1.e-6f) {
            if (c.msk.nsteps > 1) {
                inflo = o.tsin(irtstep) / static_cast<float>(c.msk.substeps);
            } else {
                inflo = ht1.flo;
            }
            inflo = o.tsin(irtstep) / static_cast<float>(c.msk.substeps);
            float rto = inflo / ht1.flo;
            rto = fmax2(0.0f, rto);
            rto = fmin2(1.0f, rto);
            s.tot_stor(jrch) = s.tot_stor(jrch) + rto * ht1;
        }
        const float inflo_rate = inflo / 86400.0f;
        rcurv_interp_flo(s, jrch, inflo_rate);
        s.ch_rcurv(jrch).in2 = s.rcurv;
        if (s.tot_stor(jrch).flo < 1.e-6f) {
            s.ch_rcurv(jrch).in1 = s.rcz;
            s.ch_rcurv(jrch).out1 = s.rcz;
            c.in1_vol = 0.0f;
            c.out1_vol = 0.0f;
        } else {
            float outflo = 0.0f;
            if (s.bsn_cc.rte == 1) {
                outflo = c.msk.c1 * inflo + c.msk.c2 * c.in1_vol + c.msk.c3 * c.out1_vol;
                outflo = fmin2(outflo, s.tot_stor(jrch).flo);
                outflo = fmax2(outflo, 0.0f);
                c.in1_vol = inflo;
                c.out1_vol = outflo;
            } else {
                float scoef = dthr / (s.ch_rcurv(jrch).in2.ttime + s.ch_rcurv(jrch).out1.ttime + dthr);
                scoef = s.bsn_prm.scoef * 2.0f * dthr / (2.0f * s.ch_rcurv(jrch).out1.ttime + dthr);
                scoef = fmin2(scoef, 1.0f);
                outflo = scoef * s.tot_stor(jrch).flo;
            }
            const float outflo_rate = outflo / dts;
            rcurv_interp_flo(s, jrch, outflo_rate);
            s.ch_rcurv(jrch).out2 = s.rcurv;
            float rto = outflo / s.tot_stor(jrch).flo;
            rto = fmin2(1.0f, rto);
            ht2 = ht2 + rto * s.tot_stor(jrch);
            o.hyd_flo(1, irtstep) = o.hyd_flo(1, irtstep) + outflo;
            s.tot_stor(jrch) = (1.0f - rto) * s.tot_stor(jrch);
            s.ch_rcurv(jrch).in1 = s.ch_rcurv(jrch).in2;
            s.ch_rcurv(jrch).out1 = s.ch_rcurv(jrch).out2;
            if (s.tot_stor(jrch).flo > s.ch_rcurv(jrch).elev(2).vol_ch) {
                rto = (s.tot_stor(jrch).flo - s.ch_rcurv(jrch).elev(2).vol_ch) / s.tot_stor(jrch).flo;
                s.fp_stor(jrch) = rto * s.tot_stor(jrch);
                s.ch_stor(jrch) = (1.0f - rto) * s.tot_stor(jrch);
            } else {
                s.ch_stor(jrch) = s.tot_stor(jrch);
                s.fp_stor(jrch) = s.hz;
            }
            s.tot_stor(jrch) = s.ch_stor(jrch) + s.fp_stor(jrch);
        }
    }
    if (s.ch_stor(jrch).flo > 1.e-6f) {
        trans_loss = c.chk * c.chl * s.rcurv.wet_perim * 24.0f;
        trans_loss = c.chk * c.chl * c.chw * 24.0f;
        trans_loss = fmin2(trans_loss, s.ch_stor(jrch).flo);
        const float rto = trans_loss / s.ch_stor(jrch).flo;
        s.ch_stor(jrch) = (1.0f - rto) * s.ch_stor(jrch);
    }
    s.ch_wat_d(s.ich).seep = trans_loss;
    if (s.ch_stor(jrch).flo > 1.e-6f) {
        const float topw = 1000.0f * c.chl * c.chw;
        s.iwst = o.wst;
        evap = s.bsn_prm.evrch * s.wst(s.iwst).weat.pet * topw / 1000.0f;
        evap = fmin2(evap, s.ch_stor(jrch).flo);
        const float rto = evap / s.ch_stor(jrch).flo;
        s.ch_stor(jrch).flo = (1.0f - rto) * s.ch_stor(jrch).flo;
    }
    s.ch_wat_d(s.ich).evap = evap;
    s.tot_stor(jrch) = s.ch_stor(jrch) + s.fp_stor(jrch);
    const float sum_outflo = ht2.flo;
    s.wet_stor(jrch) = s.hz;
    channel_floodplain_water_balance& wb = s.ch_fp_wb(jrch);
    wb.inflo = sum_inflo;
    wb.outflo = sum_outflo;
    wb.tl = trans_loss;
    wb.ev = evap;
    wb.ch_stor_init = ch_stor_init;
    wb.ch_stor = s.ch_stor(jrch).flo;
    wb.fp_stor_init = fp_stor_init;
    wb.fp_stor = s.fp_stor(jrch).flo;
    wb.tot_stor_init = tot_stor_init;
    wb.tot_stor = s.tot_stor(jrch).flo;
    wb.wet_stor_init = wet_stor_init;
    wb.wet_stor = s.wet_stor(jrch).flo;
}

// ch_watqual4.f90
void ch_watqual4(Sim& s) {
    s.jrch = s.isdch;
    const int jrch = s.jrch;
    s.icmd = s.sp_ob1.chandeg + jrch - 1;
    const channel_nut_data& n = s.ch_nut(s.jnut);
    hyd_output& ht1 = s.ht1;
    hyd_output& ht2 = s.ht2;
    hyd_output& ht3 = s.ht3;
    const float thgra = 1.047f, thrho = 1.047f, thrs1 = 1.024f, thrs2 = 1.074f, thrs3 = 1.074f, thrs4 = 1.024f,
                thrs5 = 1.024f, thbc1 = 1.083f, thbc2 = 1.047f, thbc3 = 1.047f, thbc4 = 1.047f, thrk1 = 1.047f,
                thrk2 = 1.024f, thrk3 = 1.024f, thrk4 = 1.060f;
    float tday = s.rcurv.ttime / 24.0f;
    tday = fmin2(1.0f, tday);
    s.rt_delt = 1.0f;
    ht3.orgn = 1000.0f * ht1.orgn / ht1.flo;
    ht3.sedp = 1000.0f * ht1.sedp / ht1.flo;
    ht3.no3 = 1000.0f * ht1.no3 / ht1.flo;
    ht3.solp = 1000.0f * ht1.solp / ht1.flo;
    ht3.chla = 1000.0f * ht1.chla / ht1.flo;
    ht3.nh3 = 1000.0f * ht1.nh3 / ht1.flo;
    ht3.no2 = 1000.0f * ht1.no2 / ht1.flo;
    ht3.cbod = 1000.0f * ht1.cbod / ht1.flo;
    ht3.dox = 1000.0f * ht1.dox / ht1.flo;
    float wtmp = 5.0f + 0.75f * s.wst(s.iwst).weat.tave;
    if (wtmp <= 0.0f) wtmp = 0.1f;
    ht2.temp = wtmp;
    const float rs2_s = theta(n.rs2, thrs2, wtmp) * s.ben_area;
    const float rs3_s = theta(n.rs3, thrs3, wtmp) * s.ben_area;
    const float rk4_s = theta(n.rk4, thrk4, wtmp) * s.ben_area;
    const float flo_rate = ht1.flo / 86400.0f;
    rcurv_interp_flo(s, s.ich, flo_rate);
    s.rchdep = s.rcurv.dep;
    const float rchdep = s.rchdep;
    const float rt_delt = s.rt_delt;
    if (ht1.flo > 0.0f && rchdep > 0.0f) {
        float disoxin = ht3.dox - rk4_s / (ht1.flo * 1000.0f);
        disoxin = fmax2(0.0f, disoxin);
        const float ammoin = ht3.nh3 + rs3_s / (ht1.flo * 1000.0f);
        const float cinn = s.ch_stor(jrch).nh3 + s.ch_stor(jrch).no3;
        float ww = -139.34410f + (1.575701e05f / (wtmp + 273.15f));
        float xx = 6.642308e07f / fpowi(wtmp + 273.15f, 2);
        float yy = 1.243800e10f / fpowi(wtmp + 273.15f, 3);
        float zz = 8.621949e11f / fpowi(wtmp + 273.15f, 4);
        float soxy = std::exp(ww - xx + yy - zz);
        if (soxy < 1.e-6f) soxy = 0.0f;
        if (ht3.dox < 0.001f) ht3.dox = 0.001f;
        if (ht3.dox > 30.0f) ht3.dox = 30.0f;
        float algcon = 1000.0f * ht3.chla / n.ai0;
        const float algin = 1000.0f * ht3.chla / n.ai0;
        float lambda = 0.0f;
        if (n.ai0 * algcon > 1.e-6f) {
            lambda = n.lambda0 + (n.lambda1 * n.ai0 * algcon) + n.lambda2 * fpow(n.ai0 * algcon, 0.66667f);
        } else {
            lambda = n.lambda0;
        }
        if (lambda > n.lambda0) lambda = n.lambda0;
        const float fnn = cinn / (cinn + n.k_n);
        const float fpp = ht3.solp / (ht3.solp + n.k_p);
        const int iwgn = s.wst(s.iwst).wco.wgn;
        float algi = 0.0f;
        if (s.wgn_pms(iwgn).daylth > 0.0f) {
            algi = s.wst(s.iwst).weat.solrad * n.tfact / s.wgn_pms(iwgn).daylth;
        } else {
            algi = 0.00001f;
        }
        const float fl_1 =
            (1.0f / (lambda * rchdep)) * std::log((n.k_l + algi) / (n.k_l + algi * (std::exp(-lambda * rchdep))));
        const float fll = 0.92f * (s.wgn_pms(iwgn).daylth / 24.0f) * fl_1;
        float& gra = s.saved.ch_watqual4_gra;  // a SAVEd local: kept when algcon >= 5000
        if (algcon < 5000.0f) {
            switch (n.igropt) {
                case 1: gra = n.mumax * fll * fnn * fpp; break;
                case 2: gra = n.mumax * fll * fmin2(fnn, fpp); break;
                case 3:
                    if (fnn > 1.e-6f && fpp > 1.e-6f) {
                        gra = n.mumax * fll * 2.0f / ((1.0f / fnn) + (1.0f / fpp));
                    } else {
                        gra = 0.0f;
                    }
                    break;
                default: break;
            }
        }
        float factk = theta(gra, thgra, wtmp) - theta(n.rhoq, thrho, wtmp);
        algcon = 1000.0f * ht3.chla / n.ai0;
        const float alg_m1 = wq_semianalyt(tday, rt_delt, 0.0f, factk, algcon, algin);
        const float alg_m = wq_semianalyt(tday, rt_delt, 0.0f, factk, algcon, algin);
        const float alg_m2 = alg_m - alg_m1;
        float alg_set = 0.0f;
        if (rchdep > 0.001f) alg_set = theta(n.rs1, thrs1, wtmp) / rchdep;
        float algcon_out = wq_semianalyt(tday, rt_delt, alg_m, -alg_set, algcon, algin);
        if (algcon_out < 1.e-6f) algcon_out = 0.0f;
        if (algcon_out > 5000.0f) algcon_out = 5000.0f;
        ht3.chla = algcon_out * n.ai0 / 1000.0f;
        const float rk1_k = -theta(n.rk1, thrk1, wtmp);
        const float rk1_m = wq_k2m(tday, rt_delt, rk1_k, ht3.cbod, ht3.cbod);
        float rk3_k = 0.0f;
        if (rchdep > 0.001f) rk3_k = -theta(n.rk3, thrk3, wtmp) / rchdep;
        float factm = rk1_m;
        factk = rk3_k;
        ht3.cbod = wq_semianalyt(tday, rt_delt, factm, factk, ht3.cbod, ht3.cbod);
        float bc1_k = theta(n.bc1, thbc1, wtmp);
        float bc3_k = theta(n.bc3, thbc3, wtmp);
        bc1_k = bc1_k * 2.0f;
        bc3_k = bc3_k * 2.0f;
        float rs4_k = 0.0f;
        if (rchdep > 0.001f) rs4_k = theta(n.rs4, thrs4, wtmp) / rchdep;
        const float bc3_m = wq_k2m(tday, rt_delt, -bc3_k, ht3.orgn, ht3.orgn);
        factk = -rs4_k;
        factm = bc3_m;
        ht3.orgn = wq_semianalyt(tday, rt_delt, factm, factk, ht3.orgn, ht3.orgn);
        if (ht3.orgn < 0.0f) ht3.orgn = 0.0f;
        const float rk2_m = theta(n.rk2, thrk2, wtmp) * soxy;
        const float rk2_k = theta(n.rk2, thrk2, wtmp);
        (void)(n.ai4 * alg_m2 + n.ai3 * alg_m1);
        factk = -rk2_k;
        const float bc2_k = -theta(n.bc2, thbc2, wtmp);
        const float bc1_m = wq_k2m(tday, rt_delt, factk, ht3.nh3, ammoin);
        const float bc2_m = wq_k2m(tday, rt_delt, bc2_k, ht3.no2, ht3.no2);
        factm = rk1_m + rk2_m - rs4_k + bc1_m * n.ai5 + bc2_m * n.ai6;
        ht3.dox = wq_semianalyt(tday, rt_delt, factm, factk, ht3.dox, ht3.dox);
        if (ht3.dox < 0.0f) ht3.dox = 0.0f;
        factk = -bc1_k;
        factm = bc1_m - bc3_m;
        ht3.nh3 = wq_semianalyt(tday, rt_delt, factm, 0.0f, ht3.nh3, ammoin);
        if (ht3.nh3 < 1.e-6f) ht3.nh3 = 0.0f;
        factm = -bc1_m + bc2_m;
        ht3.no2 = wq_semianalyt(tday, rt_delt, factm, 0.0f, ht3.no2, ht3.no2);
        if (ht3.no2 < 1.e-6f) ht3.no2 = 0.0f;
        factk = 0.0f;
        factm = -bc2_m;
        ht3.no3 = wq_semianalyt(tday, rt_delt, factm, 0.0f, ht3.no3, ht3.no3);
        if (ht3.no3 < 1.e-6f) ht3.no3 = 0.0f;
        const float bc4_k = theta(n.bc4, thbc4, wtmp);
        const float bc4_m = wq_k2m(tday, rt_delt, -bc4_k, ht3.sedp, ht3.sedp);
        float rs5_k = 0.0f;
        if (rchdep > 0.001f) rs5_k = theta(n.rs5, thrs5, wtmp) / rchdep;
        factk = -rs5_k;
        factm = bc4_m;
        ht3.sedp = wq_semianalyt(tday, rt_delt, factm, factk, ht3.sedp, ht3.sedp);
        if (ht3.sedp < 1.e-6f) ht3.sedp = 0.0f;
        xx = theta(n.bc4, thbc4, wtmp) * ht3.sedp;
        yy = theta(n.rs2, thrs2, wtmp) / (s.sd_chd(jrch).chd);  // the reference indexes the database by channel
        zz = n.ai2 * theta(gra, thgra, wtmp) * algin;
        ht3.solp = ht3.solp + (xx + yy - zz) * tday;
        if (ht3.solp < 1.e-6f) ht3.solp = 0.0f;
        ht2.orgn = ht3.orgn * ht1.flo / 1000.0f;
        ht2.sedp = ht3.sedp * ht1.flo / 1000.0f;
        ht2.no3 = ht3.no3 * ht1.flo / 1000.0f;
        ht2.solp = ht3.solp * ht1.flo / 1000.0f;
        ht2.chla = ht3.chla * ht1.flo / 1000.0f;
        ht2.nh3 = ht3.nh3 * ht1.flo / 1000.0f;
        ht2.no2 = ht3.no2 * ht1.flo / 1000.0f;
        ht2.cbod = ht3.cbod * ht1.flo / 1000.0f;
        ht2.dox = ht3.dox * ht1.flo / 1000.0f;
        (void)disoxin;
        (void)rs2_s;  // dispin in the reference, unused
    } else {
        ht2 = s.hz;
        s.ch_stor(jrch) = s.hz;
    }
}

// Fortran SUM of tmp(ig)%ts(lo:hi, iyr)
float sum_ts(const Mat<float>& ts, int lo, int hi, int iyr) {
    float t = 0.0f;
    for (int k = lo; k <= hi; ++k) t = t + ts(k, iyr);
    return t;
}

// ch_temp.f90 (water temperature; w_temp(0) holds the defaults)
void ch_temp(Sim& s) {
    const int ich = s.ich;
    const int jday = s.time.day;
    const int iob = s.sp_ob1.chandeg + ich - 1;
    object_connectivity& o = s.ob(iob);
    s.iwst = o.wst;
    const int ig = s.wst(s.iwst).wco.tgage;
    s.w = s.wst(s.iwst).weat;
    const weather_daily& w = s.w;
    const float tw_def = 5.0f + 0.75f * w.tave;
    int ru_count = 0;
    for (int in = 1; in <= o.obtyp_in.size(); ++in)
        if (o.obtyp_in(in) == "ru") ru_count = ru_count + 1;
    std::vector<int> ruid_array(static_cast<std::size_t>(ru_count), 0);
    int ru_index = 1;
    hyd_output& ht1 = s.ht1;
    ht1 = o.hd(1);
    for (int in = 1; in <= o.rcv_tot; ++in) {
        if (o.obtyp_in(in) == "chandeg") {
            const float trib1_temp = o.hin_d(in - 1).temp;
            const float trib2_temp = o.hin_d(in).temp;
            const float trib1_flo = o.hin_d(in - 1).flo / 86400.0f;
            const float trib2_flo = o.hin_d(in).flo / 86400.0f;
            const float trib_flo = trib1_flo + trib2_flo;
            if (trib_flo < 1e-6f) {
                ht1.temp = 5.0f + 0.75f * w.tave;
            } else {
                ht1.temp = (trib1_temp * trib1_flo + trib2_temp * trib2_flo) / trib_flo;
            }
        } else {
            ht1.temp = 5.0f + 0.75f * w.tave;
        }
        if (o.obtyp_in(in) == "ru") {
            ruid_array[static_cast<std::size_t>(ru_index - 1)] = o.obtypno_in(in);
            ru_index = ru_index + 1;
        }
    }
    const water_temperature_data& wt = s.w_temp(0);
    int sno_lag = static_cast<int>(wt.sno_lag);
    int gw_lag = static_cast<int>(wt.gw_lag);
    const float t_air = w.tave;
    int lat_lag = static_cast<int>(wt.lat_lag);
    int surf_lag = static_cast<int>(wt.surf_lag);
    const float surf_lag_coef = wt.surf_lag_coef;
    const float gw_lag_coef = wt.gw_lag_coef;
    const float lat_lag_coef = wt.lat_lag_coef;
    const float sno_coef = wt.sno_mlt;
    const float gw_coef = wt.gw;
    const float sur_lat_coef = wt.sur_lat;
    float q_ru_sno = 0.0f, q_ru_surf = 0.0f, q_ru_lat = 0.0f, q_ru_wyld = 0.0f;
    for (ru_index = 1; ru_index <= ru_count; ++ru_index) {
        const int iru = ruid_array[static_cast<std::size_t>(ru_index - 1)];
        for (int ielem = 1; ielem <= s.ru_def(iru).num_tot; ++ielem) {
            // the reference tests element ielem of the element list, not of the unit
            if (s.ru_elem(ielem).obtyp == "hru") {
                const int ise = s.ru_def(iru).num(ielem);
                const int ihru = s.ru_elem(ise).obtypno;
                const float cst = s.ru_elem(ise).frac;
                q_ru_sno = q_ru_sno + s.hwb_d(ihru).snomlt * cst;
                q_ru_surf = q_ru_surf + s.hwb_d(ihru).surq_gen;
                q_ru_lat = q_ru_lat + s.hwb_d(ihru).latq;
                q_ru_wyld = q_ru_wyld + s.hwb_d(ihru).wateryld;
            }
        }
    }
    if (q_ru_wyld == 0.0f) q_ru_wyld = 1e-6f;
    // hdsep1%flo_gwsw is only set by gwflow (refused)
    float q_gw = 0.0f / 86400.0f;
    if (!(q_gw < 10.0f)) q_gw = 10.0f;
    const float tw_local_prev = s.ch_out_d(ich).temp;
    float tw_local = 0.0f;
    if (ig <= 0) {
        tw_local = tw_def;
    } else {
        const climate_measured_data& m = s.tmp(ig);
        auto lagged = [&](int lag) {
            const int yrs_to_start = s.time.yrs - m.yrs_start;
            float mx = 0.0f, mn = 0.0f;
            if (jday <= lag && yrs_to_start > 1) {
                mx = (sum_ts(m.ts, 1, jday, yrs_to_start) + sum_ts(m.ts, jday + 365 - lag + 1, 365, yrs_to_start - 1)) /
                     static_cast<float>(lag);
                mn = (sum_ts(m.ts2, 1, jday, yrs_to_start) +
                      sum_ts(m.ts2, jday + 365 - lag + 1, 365, yrs_to_start - 1)) /
                     static_cast<float>(lag);
            } else {
                mx = sum_ts(m.ts, std::max(1, jday - lag + 1), jday, yrs_to_start) / static_cast<float>(lag);
                mn = sum_ts(m.ts2, std::max(1, jday - lag + 1), jday, yrs_to_start) / static_cast<float>(lag);
            }
            return (mx + mn) / 2.0f;
        };
        if (s.time.yrs == 1) surf_lag = std::min(jday, surf_lag);
        const float t_surf = fmax2(0.01f, lagged(surf_lag));
        if (s.time.yrs == 1) lat_lag = std::min(jday, lat_lag);
        const float t_lat = fmax2(0.01f, lagged(lat_lag));
        if (s.time.yrs == 1) gw_lag = std::min(jday, gw_lag);
        const float t_gw = fmax2(0.01f, lagged(gw_lag));
        if (s.time.yrs == 1) sno_lag = std::min(jday, sno_lag);
        const float t_sno = fmin2(2.5f, fmax2(0.01f, lagged(sno_lag)));
        const float sno_contr = sno_coef * (surf_lag_coef * t_sno) * q_ru_sno;
        const float gw_contr = gw_coef * (gw_lag_coef * t_gw) * q_gw;
        const float lat_contr = sur_lat_coef * (lat_lag_coef * t_lat) * q_ru_lat;
        const float surf_contr = sur_lat_coef * (surf_lag_coef * t_surf) * q_ru_surf;
        tw_local = (sno_contr + gw_contr + lat_contr + surf_contr) / q_ru_wyld;
        if (std::abs(tw_local - tw_local_prev) > 5.0f) tw_local = 5.0f + 0.75f * w.tave;
    }
    const float tw_up = ht1.temp;
    float tw_init = 0.0f;
    if (ht1.flo / 86400.0f > 1e-6f) {
        tw_init = (tw_up * (ht1.flo / 86400.0f) + (tw_local * q_ru_wyld)) / ((ht1.flo / 86400.0f) + q_ru_wyld);
    } else {
        tw_init = tw_local;
    }
    const float tdx_cal = w.dewpt * wt.hex_coef1;
    const float tdx = fmin2(tdx_cal, t_air);
    const float t_md = (tw_init + tdx) / 2.0f;
    const float ke_beta = 0.35f + 0.015f * t_md + 0.0012f * fpowi(t_md, 2);
    const float f_wind = 9.2f + 0.46f * fpowi(w.windsp, 2);
    const float k_e = 4.48f + 0.05f * tw_init + (ke_beta + 0.47f) * f_wind;
    if (wt.sf_on == 1) throw std::runtime_error("SWAT+ shade factors (shade_factor.shf) are not supported");
    const float ssff = wt.ssff;
    const float h_sr = 0.97f * (w.solrad * 11.57f) * (1.0f - ssff);
    const float e_s = 6.1275f * std::exp(17.62f * w.tave / (237.3f + w.tave));
    const float e_a = e_s * w.rhum;
    float cloud = 0.0f;
    if (w.solradmx < 1.e-4f) {
        cloud = 0.0f;
    } else {
        cloud = 0.9f * (w.solrad / w.solradmx) + 0.1f;
    }
    const float e_atm = 0.74f + 0.0065f * e_a * (1.0f + 0.17f * fpowi(cloud, 2));
    const float h_atm = 0.96f * e_atm * 5.67e-8f * fpowi(w.tave + 273.15f, 4);
    const float numerator = h_atm - 305.5f - 4.48f * tdx;
    const float t_equil = tdx + h_sr / k_e + numerator / k_e;
    const float vc = fmax2(0.01f, s.sd_ch_vel(ich).vel);
    float wid_chan = s.sd_chd(ich).chw;  // the reference indexes the database by channel
    const float dep_chan = s.sd_chd(ich).chd;
    const float wid_flow =
        std::sqrt((((s.ht2.flo / 86400.0f) / vc) * 2.0f) / ((dep_chan / wt.hex_coef2) / wid_chan));
    if (wid_chan < 0.1f) wid_chan = s.sd_chd(ich).chw * 0.25f;
    float dep_flow = fmax2(0.1f, (((dep_chan / wt.hex_coef2) / wid_chan)) * wid_flow);
    if (dep_flow > s.sd_chd(ich).chd * 2.0f) dep_flow = s.sd_chd(ich).chd * 2.0f;
    const float rttime = s.sd_ch_vel(ich).rttime / 24.0f;
    const float k_factor = 1000.0f * 4186.0f * dep_flow / 86400.0f;
    const float t_heat_exch = k_e * (t_equil - tw_init) / k_factor * rttime;
    float tw_final = tw_init + t_heat_exch;
    if (tw_final < 0.001f) tw_final = 0.001f;
    s.ht2.temp = tw_final;
    s.ch_stor(ich).temp = tw_final;
    s.ch_out_d(ich).temp = tw_final;
    s.wtemp = 5.0f + 0.75f * s.wst(s.iwst).weat.tave;
}

}  // namespace

// sd_channel_control3.f90
void sd_channel_control3(Sim& s) {
    s.ich = s.isdch;
    const int ich = s.ich;
    object_connectivity& o = s.ob(s.icmd);
    s.iwst = o.wst;
    hyd_output& ht1 = s.ht1;
    hyd_output& ht2 = s.ht2;
    ht1 = o.hin;
    ht2 = s.hz;
    s.ch_sed_bud(ich) = s.ch_sed_budz;
    if (o.trans.flo > 1.e-6f) {
        ht1 = ht1 + o.trans;
        o.trans = s.hz;
    }
    s.chsd_d(ich).flo_in = ht1.flo / 86400.0f;
    s.ch_in_d(ich) = ht1;
    s.ch_in_d(ich).flo = ht1.flo / 86400.0f;
    s.w = s.wst(s.iwst).weat;
    if (s.bsn_cc.lapse == 1) cli_lapse(s);
    s.wst(s.iwst).weat = s.w;
    ht1.temp = 5.0f + 0.75f * s.wst(s.iwst).weat.tave;
    s.wtemp = 5.0f + 0.75f * s.wst(s.iwst).weat.tave;
    if (s.sd_ch(ich).msk.nsteps == 1) o.tsin(1) = ht1.flo;
    if (s.sd_ch(ich).aqu_link > 0) throw std::runtime_error("SWAT+ geomorphic baseflow (aqu_cha.lin) is not supported");
    s.chsd_d(ich).flo_in = ht1.flo / 86400.0f;
    s.chsd_d(ich).flo_in_mm = ht1.flo / (10.0f * o.area_ha);
    s.ch_in_d(ich) = ht1;
    s.ch_in_d(ich).flo = ht1.flo / 86400.0f;
    ht2 = s.hz;
    sd_channel_sediment3(s);
    ch_rtmusk(s);
    if (ht1.flo > 1.e-6f) {
        s.jnut = s.sd_dat(ich).nut;
        s.ben_area = s.sd_ch(ich).chw * s.sd_ch(ich).chl;
        const float inflo_rate = s.ht3.flo / 86400.0f;
        rcurv_interp_flo(s, s.jrch, inflo_rate);
        ch_watqual4(s);
        if (s.bsn_cc.qual2e == 1) {
            float conc_chng = 1.0f - std::exp(-s.sd_ch(ich).n_sol_part * s.rcurv.ttime);
            s.ch_trans.orgn = conc_chng * ht1.orgn;
            s.ch_trans.orgn = fmin2(ht1.no3, s.ch_trans.orgn);
            s.ch_trans.no3 = s.ch_trans.orgn;
            ht2.orgn = ht1.orgn + s.ch_trans.orgn;
            ht2.no3 = ht1.no3 - s.ch_trans.no3;
            conc_chng = 1.0f - std::exp(-s.sd_ch(ich).p_sol_part * s.rcurv.ttime);
            s.ch_trans.sedp = conc_chng * ht1.sedp;
            s.ch_trans.sedp = fmin2(ht1.solp, s.ch_trans.sedp);
            s.ch_trans.solp = s.ch_trans.sedp;
            ht2.sedp = ht1.sedp + s.ch_trans.sedp;
            ht2.solp = ht1.solp - s.ch_trans.solp;
        }
        o.hout_tot = o.hout_tot + ht2;
    }
    s.ich = s.isdch;
    o.hd(1) = ht2;
    ch_temp(s);
    const int isdch = s.isdch;
    s.ch_out_d(isdch) = o.hd(1);
    s.ch_out_d(isdch).flo = o.hd(1).flo / 86400.0f;
    channel_sediment_budget_output& b = s.ch_sed_bud(ich);
    b.in_sed = s.ch_in_d(ich).sed;
    b.out_sed = ht2.sed;
    b.fp_dep = s.fp_dep.sed;
    b.ch_dep = s.ch_dep.sed;
    b.bank_ero = s.bank_ero.sed;
    b.in_no3 = s.ch_in_d(ich).no3;
    b.in_orgn = s.ch_in_d(ich).orgn;
    b.out_no3 = ht2.no3;
    b.out_orgn = ht2.orgn;
    b.fp_no3 = s.fp_dep.no3;
    b.bank_no3 = s.bank_ero.no3;
    b.bed_no3 = s.bed_ero.no3;
    b.fp_orgn = s.fp_dep.orgn;
    b.ch_orgn = s.ch_dep.orgn;
    b.bank_orgn = s.bank_ero.orgn;
    b.bed_orgn = s.bed_ero.orgn;
    b.in_solp = s.ch_in_d(ich).solp;
    b.in_orgp = s.ch_in_d(ich).sedp;
    b.out_solp = ht2.solp;
    b.out_orgp = ht2.sedp;
    b.fp_solp = s.fp_dep.solp;
    b.bank_solp = s.bank_ero.solp;
    b.bed_solp = s.bed_ero.solp;
    b.fp_orgp = s.fp_dep.sedp;
    b.ch_orgp = s.ch_dep.sedp;
    b.bank_orgp = s.bank_ero.sedp;
    b.bed_orgp = s.bed_ero.sedp;
    b.no3_orgn = s.ch_trans.no3;
    b.solp_orgp = s.ch_trans.solp;
    b.bed_ero = s.bed_ero.sed;
    sd_ch_output& d = s.chsd_d(isdch);
    d.flo = o.hd(1).flo / 86400.0f;
    d.flo_mm = o.hd(1).flo / (10.0f * o.area_ha);
    d.peakr = s.peakrate;
    d.sed_in = o.hin.sed;
    d.sed_out = o.hd(1).sed;
    d.sed_stor = s.ch_stor(isdch).sed;
    d.washld = o.hd(1).sed;
    d.bedld = s.ch_dep.sed;
    d.dep = s.fp_dep.sed;
    d.deg_btm = s.bed_ero.sed;
    d.deg_bank = s.bank_ero.sed;
    d.hc_sed = 0.0f;  // hc_sed, ebtm_m, ebank_m: locals the reference never sets
    d.width = s.sd_ch(isdch).chw;
    d.depth = s.sd_ch(isdch).chd;
    d.slope = s.sd_ch(isdch).chs;
    d.deg_btm_m = 0.0f;
    d.deg_bank_m = 0.0f;
    d.n_tot = o.hd(1).orgn + o.hd(1).no3 + o.hd(1).nh3 + o.hd(1).no2;
    d.p_tot = o.hd(1).sedp + o.hd(1).solp;
    s.chsd_d(ich).dep_bf = s.sd_ch_vel(ich).dep_bf;
    s.chsd_d(ich).velav_bf = s.sd_ch_vel(ich).vel_bf;
    s.ch_out_d(ich) = o.hd(1);
    s.ch_out_d(ich).flo = o.hd(1).flo / 86400.0f;
    s.chsd_d(ich).flo = o.hd(1).flo / 86400.0f;
    s.chsd_d(ich).flo_mm = o.hd(1).flo / (10.0f * o.area_ha);
    s.chsd_d(ich).peakr = s.peakrate;
}

}  // namespace swat
