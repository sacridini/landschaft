// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameter changes before the run (the "hard" calibration of calibration.cal):
// proc_cal.f90's cal_parm_read, cal_parmchg_read and cal_conditions, cal_parm_select.f90
// (the parameters of hrus, soils, plants and the basin), chg_par.f90, soil_awc_init.f90,
// soil_text_init.f90. Soft calibration is outside the port.

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

float chg_par(float val_cur, const std::string& chg_typ, float chg_val, float absmin, float absmax) {
    float r = 0.0f;
    if (chg_typ == "absval") r = chg_val;
    else if (chg_typ == "abschg") r = val_cur + chg_val;
    else if (chg_typ == "pctchg") r = (1.0f + chg_val / 100.0f) * val_cur;
    else if (chg_typ == "relchg") r = (1.0f + chg_val) * val_cur;
    else throw std::runtime_error("SWAT+ calibration change type " + chg_typ + " is unknown");
    r = fmax2(r, absmin);
    r = fmin2(r, absmax);
    return r;
}

// soil_awc_init.f90
void soil_awc_init(Sim& s, int isol) {
    soil_profile& so = s.soil(isol);
    const int nly = so.nly;
    for (int ly = 1; ly <= nly; ++ly) {
        soil_physical_properties& ph = so.phys(ly);
        if (ph.awc <= 1.e-6f) ph.awc = 0.005f;
        if (ph.awc >= 0.8f) ph.awc = 0.8f;
        ph.wp = 0.4f * ph.clay * ph.bd / 100.0f;
        if (ph.wp <= 0.0f) ph.wp = 0.005f;
        ph.up = ph.wp + ph.awc;
        ph.por = 1.0f - ph.bd / 2.65f;
        if (ph.up >= ph.por) {
            ph.up = ph.por - 0.05f;
            ph.wp = ph.up - ph.awc;
            if (ph.wp <= 0.0f) {
                ph.up = ph.por * 0.75f;
                ph.wp = ph.por * 0.25f;
            }
        }
        const float drpor = ph.por - ph.up;
        so.ly(ly).vwt = (437.13f * fpowi(drpor, 2)) - (95.08f * drpor) + 8.257f;
    }
    float depth_prev = 0.0f;
    float sumpor = 0.0f;
    so.sumfc = 0.0f;
    so.sumul = 0.0f;
    so.sw = 0.0f;
    so.sumwp = 0.0f;
    for (int ly = 1; ly <= nly; ++ly) {
        soil_physical_properties& ph = so.phys(ly);
        ph.thick = ph.d - depth_prev;
        const float pormm = ph.por * ph.thick;
        sumpor = sumpor + pormm;
        ph.ul = (ph.por - ph.wp) * ph.thick;
        so.sumul = so.sumul + ph.ul;
        ph.fc = ph.thick * (ph.up - ph.wp);
        so.sumfc = so.sumfc + ph.fc;
        ph.st = ph.fc * so.ffc;
        ph.hk = (ph.ul - ph.fc) / ph.k;
        if (ph.hk < 1.0f) ph.hk = 1.0f;
        so.sw = so.sw + ph.st;
        ph.wpmm = ph.wp * ph.thick;
        so.sumwp = so.sumwp + ph.wpmm;
        ph.crdep = so.crk * 0.916f * std::exp(-0.0012f * ph.d) * ph.thick;
        so.ly(ly).volcr = ph.crdep * (ph.fc - ph.st) / (ph.fc);
        depth_prev = ph.d;
    }
    so.swpwt = so.sw;
    if (so.ffc > 1.0f) {
        so.wat_tbl = (so.det_lag - so.ffc * so.sumfc) / so.phys(nly).d;
    } else {
        so.wat_tbl = 0.0f;
    }
    so.avpor = sumpor / so.phys(nly).d;
    so.avbd = 2.65f * (1.0f - so.avpor);
}

// soil_text_init.f90
void soil_text_init(Sim& s, int isol) {
    soil_profile& so = s.soil(isol);
    const float sa = so.phys(1).sand / 100.0f;
    const float cl = so.phys(1).clay / 100.0f;
    const float si = so.phys(1).silt / 100.0f;
    so.det_san = sa * fpow(1.0f - cl, 2.49f);
    so.det_sil = 0.13f * si;
    so.det_cla = 0.20f * cl;
    if (cl < 0.25f) {
        so.det_sag = 2.0f * cl;
    } else if (cl > 0.5f) {
        so.det_sag = 0.57f;
    } else {
        so.det_sag = 0.28f * (cl - 0.25f) + 0.5f;
    }
    so.det_lag = 1.0f - so.det_san - so.det_sil - so.det_cla - so.det_sag;
    if (so.det_lag < 0.0f) {
        so.det_san = so.det_san / (1.0f - so.det_lag);
        so.det_sil = so.det_sil / (1.0f - so.det_lag);
        so.det_cla = so.det_cla / (1.0f - so.det_lag);
        so.det_sag = so.det_sag / (1.0f - so.det_lag);
        so.det_lag = 0.0f;
    }
}

// cal_parm_select.f90 (the parameters of the port's objects)
void cal_parm_select(Sim& s, int ielem, int ly, const std::string& chg_parm_in, const std::string& chg_typ,
                     float chg_val, float absmin, float absmax) {
    const std::string chg_parm = chg_parm_in.substr(0, 16);
    auto C = [&](float& v) { v = chg_par(v, chg_typ, chg_val, absmin, absmax); };
    auto usle_mult = [&] {
        const float rock = std::exp(-0.053f * s.soil(ielem).phys(1).rock);
        hydrologic_response_unit& h = s.hru(ielem);
        h.lumv.usle_mult = rock * s.soil(ielem).ly(1).usle_k * h.lumv.usle_p * h.lumv.usle_ls * 11.8f;
    };
    const std::string& p = chg_parm;
    if (p == "cn2") {
        C(s.cn2(ielem));
        curno(s, s.cn2(ielem), ielem);
        return;
    }
    if (p == "lab_p" || p == "hum_c_n" || p == "hum_c_p") {
        // the reference applies the change once per soil layer, all to layer ly
        for (int jj = 1; jj <= s.soil(ielem).nly; ++jj) {
            if (p == "lab_p") C(s.soil1(ielem).mp(ly).lab);
            else if (p == "hum_c_n") C(s.soil1(ielem).hact(ly).n);
            else C(s.soil1(ielem).hact(ly).p);
        }
        return;
    }
    if (p == "z" || p == "bd" || p == "awc" || p == "k" || p == "cbn" || p == "clay" || p == "silt" || p == "sand" ||
        p == "rock" || p == "alb" || p == "usle_k" || p == "ec" || p == "cal" || p == "ph") {
        soil_physical_properties& ph = s.soil(ielem).phys(ly);
        soilayer& sl = s.soil(ielem).ly(ly);
        if (p == "z") {
            C(ph.d);
            soil_awc_init(s, ielem);
            curno(s, s.cn2(ielem), ielem);
        } else if (p == "bd") {
            C(ph.bd);
            soil_awc_init(s, ielem);
            curno(s, s.cn2(ielem), ielem);
        } else if (p == "awc") {
            C(ph.awc);
            soil_awc_init(s, ielem);
            curno(s, s.cn2(ielem), ielem);
        } else if (p == "k") {
            C(ph.k);
            ph.hk = (ph.ul - ph.fc) / ph.k;
            if (ph.hk < 1.0f) ph.hk = 1.0f;
        } else if (p == "cbn") {
            C(s.soil1(ielem).tot(ly).c);
        } else if (p == "clay") {
            C(ph.clay);
            soil_awc_init(s, ielem);
            soil_text_init(s, ielem);
            curno(s, s.cn2(ielem), ielem);
        } else if (p == "silt") {
            C(ph.silt);
            soil_text_init(s, ielem);
        } else if (p == "sand") {
            C(ph.sand);
            soil_text_init(s, ielem);
        } else if (p == "rock") {
            C(ph.rock);
            if (ly == 1) usle_mult();
        } else if (p == "alb") {
            C(sl.alb);
        } else if (p == "usle_k") {
            C(sl.usle_k);
            usle_mult();
        } else if (p == "ec") {
            C(sl.ec);
        } else if (p == "cal") {
            C(sl.cal);
        } else if (p == "ph") {
            C(sl.ph);
        }
        return;
    }
    if (p == "anion_excl") return C(s.soil(ielem).anion_excl);
    if (p == "crk") return C(s.soil(ielem).crk);
    if (p == "usle_c") return C(s.pldb(ielem).usle_c);
    if (p == "bio_e") return C(s.pldb(ielem).bio_e);
    if (p == "hvsti") return C(s.pldb(ielem).hvsti);
    if (p == "rdmx") return C(s.pldb(ielem).rdmx);
    if (p == "t_opt") return C(s.pldb(ielem).t_opt);
    if (p == "t_base") return C(s.pldb(ielem).t_base);
    if (s.hru.ubound() >= ielem && ielem >= 1) {
        hydrologic_response_unit& h = s.hru(ielem);
        if (p == "biomix") return C(h.hyd.biomix);
        if (p == "cn3_swf") {
            if (h.tiledrain == 0) {
                C(h.hyd.cn3_swf);
                curno(s, s.cn2(ielem), ielem);
            }
            return;
        }
        if (p == "usle_p") return C(h.lumv.usle_p);
        if (p == "ovn") return C(h.luse.ovn);
        if (p == "elev") return C(h.topo.elev);
        if (p == "slope") return C(h.topo.slope);
        if (p == "slope_len") return C(h.topo.slope_len);
        if (p == "lat_ttime") {
            C(h.hyd.lat_ttime);
            h.hyd.lat_ttime = 1.0f - std::exp(-1.0f / h.hyd.lat_ttime);
            return;
        }
        if (p == "lat_sed") return C(h.hyd.lat_sed);
        if (p == "lat_len") return C(h.topo.lat_len);
        if (p == "latq_co") return C(h.hyd.latq_co);
        if (p == "canmx") return C(h.hyd.canmx);
        if (p == "esco") return C(h.hyd.esco);
        if (p == "epco") {
            C(h.hyd.epco);
            for (int ipl = 1; ipl <= s.pcom(ielem).npl; ++ipl) s.pcom(ielem).plcur(ipl).epco = h.hyd.epco;
            return;
        }
        if (p == "erorgn") return C(h.hyd.erorgn);
        if (p == "erorgp") return C(h.hyd.erorgp);
        if (p == "dis_stream") return C(h.topo.dis_stream);
        if (p == "perco") {
            if (h.tiledrain == 0) {
                C(h.hyd.perco);
                if (h.hyd.perco > 1.e-6f) {
                    const float perc_ln_func = 1.0052f * std::log(-std::log(h.hyd.perco - 1.e-6f)) + 5.6862f;
                    h.hyd.perco_lim = std::exp(-perc_ln_func);
                    h.hyd.perco_lim = fmin2(1.0f, h.hyd.perco_lim);
                } else {
                    h.hyd.perco_lim = 0.0f;
                }
            }
            return;
        }
        if (p == "petco") return C(h.hyd.pet_co);
        if (p == "lat_orgn") return C(h.hyd.lat_orgn);
        if (p == "lat_orgp") return C(h.hyd.lat_orgp);
        if (p == "field_len") return C(h.field.length);
        if (p == "field_wid") return C(h.field.wid);
        if (p == "field_ang") return C(h.field.ang);
        if (p == "snofall_tmp") return C(h.sno.falltmp);
        if (p == "snomelt_tmp") return C(h.sno.melttmp);
        if (p == "snomelt_max") return C(h.sno.meltmx);
        if (p == "snomelt_min") return C(h.sno.meltmn);
        if (p == "snomelt_lag") return C(h.sno.timp);
        if (p == "cmn_h") return C(h.nut.cmn);
        if (p == "nperco_h") return C(h.nut.nperco);
        if (p == "pperco_h") return C(h.nut.pperco);
        if (p == "phoskd_h") return C(h.nut.phoskd);
        if (p == "psp_h") return C(h.nut.psp);
        if (p == "nperco_lchtile_h") return C(h.nut.nperco_lchtile);
    }
    basin_parms& b = s.bsn_prm;
    if (p == "plaps") return C(b.plaps);
    if (p == "tlaps") return C(b.tlaps);
    if (p == "surlag") {
        C(b.surlag);
        for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) s.brt(ihru) = 1.0f - std::exp(-b.surlag / s.tconc(ihru));
        return;
    }
    if (p == "adj_pkr") return C(b.adj_pkr);
    if (p == "prf") return C(b.prf);
    if (p == "evrch") return C(b.evrch);
    if (p == "evlai") return C(b.evlai);
    if (p == "ffcb") return C(b.ffcb);
    if (p == "cmn") return C(b.cmn);
    if (p == "nperco") return C(b.nperco);
    if (p == "pperco") return C(b.pperco);
    if (p == "phoskd") return C(b.phoskd);
    if (p == "psp") return C(b.psp);
    if (p == "rsdco") return C(b.rsdco);
    if (p == "percop") return C(b.percop);
    if (p == "msk_co1") return C(b.msk_co1);
    if (p == "msk_co2") return C(b.msk_co2);
    if (p == "msk_x") return C(b.msk_x);
    if (p == "nperco_lchtile") return C(b.nperco_lchtile);
    if (p == "cdn") return C(b.cdn);
    if (p == "tb_adj") return C(b.tb_adj);
    if (p == "sdnco") return C(b.sdnco);
    if (p == "n_updis") return C(b.n_updis);
    if (p == "rsd_covco") return C(b.rsd_covco);
    if (p == "dorm_hr") return C(b.dorm_hr);
    throw std::runtime_error("SWAT+ calibration of parameter " + chg_parm + " is not supported");
}

}  // namespace

// proc_cal.f90 (cal_parm_read, cal_parmchg_read, cal_conditions)
void proc_cal(Sim& s) {
    std::string titldum, header;
    // cal_parm_read.f90
    {
        const std::string& file = s.in_chg.cal_parms;
        int mchg_par = 0;
        if (!s.files.exists(file) || file == "null") {
            s.cal_parms.alloc(0, 0, calibration_parameters());
        } else {
            fio::File f = s.files.open(file);
            for (;;) {
                if (f.read(chars(titldum, 80)) < 0) break;
                if (f.read(mchg_par) < 0) break;
                s.cal_parms.alloc(0, mchg_par, calibration_parameters());
                if (f.read(chars(header, 80)) < 0) break;
                for (int i = 1; i <= mchg_par; ++i)
                    if (read_items(f, s.cal_parms(i)) < 0) break;
                break;
            }
        }
        s.db_mx.cal_parms = mchg_par;
    }
    // cal_parmchg_read.f90
    int mcal = 0;
    {
        const std::string& file = s.in_chg.cal_upd;
        if (!s.files.exists(file) || file == "null") {
            s.cal_upd.alloc(0, 0, update_parameters());
        } else {
            fio::File f = s.files.open(file);
            for (;;) {
                if (f.read(chars(titldum, 80)) < 0) break;
                const int eof = f.read(mcal);
                s.cal_upd.alloc(1, mcal, update_parameters());
                if (eof < 0) break;
                if (f.read(chars(header, 80)) < 0) break;
                for (int i = 1; i <= mcal; ++i) {
                    update_parameters& u = s.cal_upd(i);
                    int nspu = 0;
                    if (f.read(chars(u.name, u.name_len), chars(u.chg_typ, u.chg_typ_len), u.val, u.conds, u.lyr1,
                               u.lyr2, u.year1, u.year2, u.day1, u.day2, nspu) < 0)
                        break;
                    if (nspu > 0) {
                        f.backspace();
                        s.elem_cnt.alloc(1, nspu, 0);
                        fio::ItemList l;
                        fio::add(l, chars(u.name, u.name_len));
                        fio::add(l, chars(u.chg_typ, u.chg_typ_len));
                        fio::add(l, u.val);
                        fio::add(l, u.conds);
                        fio::add(l, u.lyr1);
                        fio::add(l, u.lyr2);
                        fio::add(l, u.year1);
                        fio::add(l, u.year2);
                        fio::add(l, u.day1);
                        fio::add(l, u.day2);
                        fio::add(l, u.num_tot);
                        for (int isp = 1; isp <= nspu; ++isp) fio::add(l, s.elem_cnt(isp));
                        if (f.read(l) < 0) break;
                    }
                    for (int ical = 1; ical <= s.db_mx.cal_parms; ++ical) {
                        if (fstrcmp(u.name, s.cal_parms(ical).name) == 0) {
                            u.num_db = ical;
                            break;
                        }
                    }
                    if (u.conds > 0) throw std::runtime_error("SWAT+ calibration conditions are not supported");
                    if (u.num_tot == 0) {
                        const std::string& t = s.cal_parms(u.num_db).ob_typ;
                        if (t == "hru" || t == "lyr" || t == "sol") u.num_elem = s.sp_ob.hru;
                        else if (t == "plt") u.num_elem = s.db_mx.plantparm;
                        else if (t == "ru") u.num_elem = s.sp_ob.ru;
                        else if (t == "aqu") u.num_elem = s.sp_ob.aqu;
                        else if (t == "sdc" || t == "rte") u.num_elem = s.sp_ob.chandeg;
                        else if (t == "bsn") u.num_elem = 1;
                        else throw std::runtime_error("SWAT+ calibration of " + t + " parameters is not supported");
                        u.num.alloc(1, u.num_elem, 0);
                        for (int ie = 1; ie <= u.num_elem; ++ie) u.num(ie) = ie;
                    } else {
                        int ielem1 = 0;
                        define_unit_elements(s, nspu, ielem1);
                        u.num = s.defunit_num;
                        u.num_elem = ielem1;
                        s.defunit_num = Vec<int>();
                        s.elem_cnt = Vec<int>();
                    }
                }
                break;
            }
        }
        s.db_mx.cal_upd = mcal;
    }
    // cal_conditions.f90
    for (int ichg_par = 1; ichg_par <= s.db_mx.cal_upd; ++ichg_par) {
        const update_parameters& u = s.cal_upd(ichg_par);
        for (int ispu = 1; ispu <= u.num_elem; ++ispu) {
            const int ielem = u.num(ispu);
            const calibration_parameters& cp = s.cal_parms(u.num_db);
            const std::string& t = cp.ob_typ;
            if (t != "sol" && t != "cli" && t != "plt" && t != "rdt")
                cal_parm_select(s, ielem, 0, u.name, u.chg_typ, u.val, cp.absmin, cp.absmax);
            if (t == "sol") {
                int cal_lyr1 = std::max(u.lyr1, 1);
                int cal_lyr2 = u.lyr2;
                if (cal_lyr2 <= 0) cal_lyr2 = s.soil(ielem).nly;
                cal_lyr2 = std::min(cal_lyr2, s.soil(ielem).nly);
                for (int lyr = cal_lyr1; lyr <= cal_lyr2; ++lyr)
                    cal_parm_select(s, ielem, lyr, u.name, u.chg_typ, u.val, cp.absmin, cp.absmax);
            } else if (t == "plt" || t == "cli" || t == "rdt") {
                throw std::runtime_error("SWAT+ calibration of " + t + " parameters (" + u.name + ") is not supported");
            }
        }
    }
}

}  // namespace swat
