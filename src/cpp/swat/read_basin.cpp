// SPDX-License-Identifier: GPL-3.0-or-later
//
// proc_bsn: file.cio, codes.bsn, object.cnt, time.sim, parameters.bsn, print.prt, CO2
// (readcio_read.f90, basin_read_cc.f90, basin_read_objs.f90, time_read.f90,
// basin_read_prm.f90, basin_prm_default.f90, basin_print_codes_read.f90, co2_read.f90).

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

// character(len=25) items of a file.cio line
template <class... S>
int read_names(fio::File& f, std::string& label, S&... s) {
    return f.read(chars(label, 15), chars(s, 25)...);
}

}  // namespace

void readcio_read(Sim& s) {
    InputFiles& in = s.in;
    std::string titldum, name;
    if (s.files.exists("file.cio")) {
        fio::File f = s.files.open("file.cio");
        f.read(chars(titldum, 80));
        for (int i = 1; i <= 31; ++i) {
            if (read_names(f, name, in.time, in.prt, in.object_prt, in.object_cnt, in.cs_db) < 0) break;
            if (read_names(f, name, in.codes_bas, in.parms_bas, in.carbon_bsn) < 0) break;
            if (read_names(f, name, in.weat_sta, in.weat_wgn, in.pet_cli, in.pcp_cli, in.tmp_cli, in.slr_cli,
                           in.hmd_cli, in.wnd_cli, in.atmo_cli) < 0)
                break;
            if (read_names(f, name, in.hru_con, in.hruez_con, in.ru_con, in.gwflow_con, in.aqu_con, in.aqu2d_con,
                           in.chan_con, in.res_con, in.rec_con, in.exco_con, in.delr_con, in.out_con,
                           in.chandeg_con) < 0)
                break;
            if (read_names(f, name, in.cha_init, in.cha_dat, in.cha_hyd, in.cha_sed, in.cha_nut, in.chan_ez, in.hyd_sed,
                           in.cha_temp) < 0)
                break;
            if (read_names(f, name, in.init_res, in.res, in.hyd_res, in.sed_res, in.nut_res, in.weir_res, in.wet,
                           in.hyd_wet) < 0)
                break;
            if (read_names(f, name, in.ru_def, in.ru_ele, in.ru, in.ru_dr) < 0) break;
            if (read_names(f, name, in.hru_data, in.hru_ez) < 0) break;
            if (read_names(f, name, in.exco, in.exco_om, in.exco_pest, in.exco_path, in.exco_hmet, in.exco_salt) < 0)
                break;
            if (read_names(f, name, in.recall_rec) < 0) break;
            if (read_names(f, name, in.del_ratio, in.dr_om, in.dr_pest, in.dr_path, in.dr_hmet, in.dr_salt) < 0) break;
            if (read_names(f, name, in.aqu_init, in.aqu) < 0) break;
            if (read_names(f, name, in.animal, in.herd, in.ranch) < 0) break;
            if (read_names(f, name, in.pou, in.pod) < 0) break;
            if (read_names(f, name, in.chan_surf, in.aqu_cha) < 0) break;
            if (read_names(f, name, in.hydrol_hyd, in.topogr_hyd, in.field_fld) < 0) break;
            if (read_names(f, name, in.tiledrain_str, in.septic_str, in.fstrip_str, in.grassww_str, in.bmpuser_str) < 0)
                break;
            if (read_names(f, name, in.plants_plt, in.fert_frt, in.till_til, in.pest, in.pathcom_db, in.hmetcom_db,
                           in.saltcom_db, in.urban_urb, in.septic_sep, in.snow) < 0)
                break;
            if (read_names(f, name, in.harv_ops, in.graze_ops, in.irr_ops, in.chem_ops, in.fire_ops, in.sweep_ops) < 0)
                break;
            if (read_names(f, name, in.landuse_lum, in.management_sch, in.cntable_lum, in.cons_prac_lum, in.ovn_lum) <
                0)
                break;
            if (read_names(f, name, in.cal_parms, in.cal_upd, in.codes_sft, in.wb_parms_sft, in.water_balance_sft,
                           in.ch_sed_budget_sft, in.ch_sed_parms_sft, in.plant_parms_sft, in.plant_gro_sft) < 0)
                break;
            if (read_names(f, name, in.plant_ini, in.soil_plant_ini, in.om_water, in.pest_soil, in.pest_water,
                           in.path_soil, in.path_water, in.hmet_soil, in.hmet_water, in.salt_soil, in.salt_water) < 0)
                break;
            if (read_names(f, name, in.soils_sol, in.nut_sol, in.lte_sol) < 0) break;
            if (read_names(f, name, in.dtbl_lum, in.dtbl_res, in.dtbl_scen, in.dtbl_flo) < 0) break;
            if (read_names(f, name, in.ele_lsu, in.def_lsu, in.ele_reg, in.def_reg, in.cal_lcu, in.ele_cha, in.def_cha,
                           in.def_cha_reg, in.ele_aqu, in.def_aqu, in.def_aqu_reg, in.ele_res, in.def_res,
                           in.def_res_reg, in.ele_psc, in.def_psc, in.def_psc_reg) < 0)
                break;
            if (f.read(chars(name, 15), chars(in.path_pcp, 80)) < 0) break;
            if (f.read(chars(name, 15), chars(in.path_tmp, 80)) < 0) break;
            if (f.read(chars(name, 15), chars(in.path_slr, 80)) < 0) break;
            if (f.read(chars(name, 15), chars(in.path_hmd, 80)) < 0) break;
            if (f.read(chars(name, 15), chars(in.path_wnd, 80)) < 0) break;
            std::string line;
            if (f.read_line(line) >= 0) {
                // the label, then the rest of the line
                std::size_t a = line.find_first_not_of(' ');
                if (a != std::string::npos) {
                    std::size_t b = line.find(' ', a);
                    if (b != std::string::npos) {
                        std::size_t c = line.find_first_not_of(' ', b);
                        in.out_path = c == std::string::npos ? "" : line.substr(c);
                    }
                }
            }
            break;
        }
    }
}

void basin_read_cc(Sim& s) {
    BasinCodes& c = s.bsn_cc;
    std::string titldum, header;
    if (s.files.exists(s.in.codes_bas) || s.in.codes_bas != "null") {
        fio::File f = s.files.open(s.in.codes_bas);
        if (!f.open()) throw std::runtime_error("cannot open " + s.in.codes_bas);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            f.read(chars(c.petfile, 16), chars(c.wwqfile, 16), c.pet, c.nam1, c.crk, c.swift_out, c.sed_det, c.rte,
                   c.deg, c.wq, c.nostress, c.cn, c.cfac, c.cswat, c.lapse, c.uhyd, c.sed_ch, c.tdrn, c.wtdn,
                   c.sol_p_model, c.gampt, chars(c.atmo, 1), c.smax, c.qual2e, c.gwflow, c.idc_till);
            break;
        }
    }
    if (c.pet == 3) throw std::runtime_error("SWAT+ pet = 3 (pet.cli) is not supported");
}

void basin_read_objs(Sim& s) {
    std::string titldum, header;
    if (!s.files.exists(s.in.object_cnt) || s.in.object_cnt == "null")
        throw std::runtime_error("Cannot find object.cnt input file");
    fio::File f = s.files.open(s.in.object_cnt);
    // the reference loops until the end of the file, re-reading the counts: the last line wins
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        SpatialObjects& o = s.sp_ob;
        if (f.read(chars(s.bsn.name, 25), s.bsn.area_ls_ha, s.bsn.area_tot_ha, o.objs, o.hru, o.hru_lte, o.ru, o.gwflow,
                   o.aqu, o.chan, o.res, o.recall, o.exco, o.dr, o.canal, o.pump, o.outlet, o.chandeg, o.aqu2d, o.herd,
                   o.wro) < 0)
            break;
    }
    if (s.bsn_cc.gwflow == 1) throw std::runtime_error("SWAT+ gwflow is not supported");
}

void time_read(Sim& s) {
    std::string titldum, header;
    Time& t = s.time;
    if (s.files.exists(s.in.time) || s.in.time != "null") {
        fio::File f = s.files.open(s.in.time);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 500)) < 0) break;
            if (f.read(t.day_start, t.yrc_start, t.day_end, t.yrc_end, t.step) < 0) break;
            if (t.step <= 0) t.step = 1;
            if (t.day_start <= 0) t.day_start = 1;
            t.nbyr = t.yrc_end - t.yrc_start + 1;
            int mo = 0, day_mo = 0;
            xmon(s, t.day_start, mo, day_mo);
            t.mo = mo;
            t.day_mo = day_mo;
            t.mo_start = mo;
            break;
        }
    }
    t.yrc = t.yrc_start;
    if (t.step != 1) throw std::runtime_error("SWAT+ sub-daily time steps are not supported");
}

void basin_read_prm(Sim& s) {
    std::string titldum, header;
    BasinParms& p = s.bsn_prm;
    if (s.files.exists(s.in.parms_bas) || s.in.parms_bas != "null") {
        fio::File f = s.files.open(s.in.parms_bas);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            f.read(p.evlai, p.ffcb, p.surlag, p.adj_pkr, p.prf, p.pestgwfact, p.temp_decay, p.cmn, p.n_updis,
                   p.p_updis, p.nperco, p.pperco, p.phoskd, p.psp, p.rsdco, p.percop, p.msk_co1, p.msk_co2, p.msk_x,
                   p.nperco_lchtile, p.evrch, p.scoef, p.cdn, p.sdnco, p.bact_swf, p.tb_adj, p.cn_froz, p.dorm_hr,
                   p.plaps, p.tlaps, p.nfixmx, p.decr_min, p.rsd_covco, p.urb_init_abst, p.petco_pmpt, p.uhalpha,
                   p.eros_spl, p.rill_mult, p.eros_expo, p.c_factor, p.ch_d50, p.co2, p.day_lag_mx, p.igen);
            break;
        }
    }
}

void basin_prm_default(Sim& s) {
    BasinParms& p = s.bsn_prm;
    if (p.evlai < 1.e-6f) p.evlai = 3.0f;
    if (p.ffcb < 1.e-6f) p.ffcb = 0.0f;
    if (p.surlag < 1.e-6f) p.surlag = 4.0f;
    if (p.adj_pkr < 1.e-6f) p.adj_pkr = 1.0f;
    if (p.prf < 1.e-6f) p.prf = 1.0f;
    if (p.cmn < 1.e-6f) p.cmn = 0.0003f;
    if (p.n_updis < 1.e-6f) p.n_updis = 20.0f;
    if (p.p_updis < 1.e-6f) p.p_updis = 20.0f;
    if (p.nperco < 1.e-6f) p.nperco = 0.1f;
    if (p.pperco < 1.e-6f) p.pperco = 10.0f;
    if (p.phoskd < 1.e-6f) p.phoskd = 175.0f;
    if (p.psp < 1.e-6f) p.psp = 0.40f;
    if (p.rsdco < 1.e-6f) p.rsdco = 0.05f;
    if (p.percop < 1.e-6f) p.percop = 0.5f;
    if (p.msk_co1 < 1.e-6f) p.msk_co1 = 0.75f;
    if (p.msk_co2 < 1.e-6f) p.msk_co2 = 0.25f;
    if (p.msk_x < 1.e-6f) p.msk_x = 0.20f;
    if (p.nperco_lchtile < 1.e-6f) p.nperco_lchtile = 0.50f;
    if (p.evrch < 1.e-6f) p.evrch = 0.60f;
    if (p.cdn < 1.e-6f) p.cdn = 1.40f;
    if (p.sdnco < 1.e-6f) p.sdnco = 1.30f;
    if (p.bact_swf < 1.e-6f) p.bact_swf = 0.15f;
    if (p.tb_adj < 1.e-6f) p.tb_adj = 0.0f;
    if (p.cn_froz < 1.e-6f) p.cn_froz = 0.000862f;
    if (p.nfixmx < 1.e-6f) p.nfixmx = 20.0f;
    if (p.decr_min < 1.e-6f) p.decr_min = 0.01f;
    if (p.rsd_covco < 1.e-6f) p.rsd_covco = 0.75f;
    if (p.urb_init_abst < 1.e-6f) p.urb_init_abst = 0.0f;
    if (p.petco_pmpt < 0.5f && p.petco_pmpt > 0.0f) p.petco_pmpt = 0.0f;
    p.petco_pmpt = (100.0f + p.petco_pmpt) / 100.0f;
    if (p.uhalpha < 1.e-6f) p.uhalpha = 1.0f;
    if (p.eros_spl < 1.e-6f) p.eros_spl = 0.0f;
    if (p.rill_mult < 1.e-6f) p.rill_mult = 0.0f;
    if (p.eros_expo < 1.e-6f) p.eros_expo = 0.0f;
    if (p.c_factor < 1.e-6f) p.c_factor = 0.0f;
    if (p.ch_d50 < 1.e-6f) p.ch_d50 = 0.0f;
    if (p.co2 < 100.0f) p.co2 = 400.0f;
    s.uptake.water_dis = 10.0f;
    s.uptake.water_norm = 1.0f - std::exp(-s.uptake.water_dis);
    s.uptake.n_norm = 1.0f - std::exp(-p.n_updis);
    s.uptake.p_norm = 1.0f - exp_w(-p.p_updis);
    p.day_lag_mx = 2;
}

void basin_print_codes_read(Sim& s) {
    // only the first line changes the simulation (the years skipped); the rest selects outputs
    std::string titldum, header;
    if (s.files.exists(s.in.prt) || s.in.prt != "null") {
        fio::File f = s.files.open(s.in.prt);
        if (f.read(chars(titldum, 80)) < 0) return;
        if (f.read(chars(header, 500)) < 0) return;
        PrintCodes& p = s.pco;
        f.read(p.nyskip, p.day_start, p.yrc_start, p.day_end, p.yrc_end, p.int_day);
    }
}

void co2_read(Sim& s) {
    if (s.files.exists("co2_yr.dat")) throw std::runtime_error("SWAT+ co2_yr.dat is not supported");
    s.co2y.alloc(1, s.time.nbyr, 0.0f);
    for (int iyr = 1; iyr <= s.time.nbyr; ++iyr) s.co2y(iyr) = s.bsn_prm.co2;
}

void proc_bsn(Sim& s) {
    readcio_read(s);
    basin_read_cc(s);
    basin_read_objs(s);
    time_read(s);
    s.time.dtm = 1440.0f / static_cast<float>(s.time.step);
    basin_read_prm(s);
    basin_prm_default(s);
    basin_print_codes_read(s);
    co2_read(s);
}

void xmon(const Sim& s, int jd, int& mo, int& day_mo) {
    for (int i_mo = 1; i_mo <= 12; ++i_mo) {
        const int nda = s.ndays[static_cast<std::size_t>(i_mo)];
        if (jd <= nda) {
            mo = i_mo;
            day_mo = jd - s.ndays[static_cast<std::size_t>(i_mo - 1)];
            return;
        }
    }
}

}  // namespace swat
