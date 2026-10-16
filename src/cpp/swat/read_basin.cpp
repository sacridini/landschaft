// SPDX-License-Identifier: GPL-3.0-or-later
//
// proc_bsn: file.cio, codes.bsn, object.cnt, time.sim, parameters.bsn, print.prt, CO2
// (readcio_read.f90, basin_read_cc.f90, basin_read_objs.f90, time_read.f90,
// basin_read_prm.f90, basin_prm_default.f90, basin_print_codes_read.f90, co2_read.f90).

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

void readcio_read(Sim& s) {
    std::string titldum, name, line;
    if (!s.files.exists("file.cio")) return;
    fio::File f = s.files.open("file.cio");
    f.read(chars(titldum, 80));
    for (int i = 1; i <= 31; ++i) {
        if (f.read(chars(name, 15), items(s.in_sim)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_basin)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_cli)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_con)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_cha)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_res)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_ru)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_hru)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_exco)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_rec)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_delr)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_aqu)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_herd)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_wallo)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_link)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_hyd)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_str)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_parmdb)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_ops)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_lum)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_chg)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_init)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_sol)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_cond)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_regs)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_path_pcp)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_path_tmp)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_path_slr)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_path_hmd)) < 0) break;
        if (f.read(chars(name, 15), items(s.in_path_wnd)) < 0) break;
        // the output path: the rest of the line after its label (not used by the port)
        f.read_line(line);
        break;
    }
}

void basin_read_cc(Sim& s) {
    basin_control_codes& c = s.bsn_cc;
    std::string titldum, header;
    const std::string& name = s.in_basin.codes_bas;
    if (s.files.exists(name) || name != "null") {
        fio::File f = s.files.open(name);
        if (!f.open()) throw std::runtime_error("cannot open " + name);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            read_items(f, c);
            break;
        }
    }
    if (c.pet == 3) throw std::runtime_error("SWAT+ pet = 3 (pet.cli) is not supported");
}

void basin_read_objs(Sim& s) {
    std::string titldum, header;
    if (!s.files.exists(s.in_sim.object_cnt) || s.in_sim.object_cnt == "null")
        throw std::runtime_error("Cannot find object.cnt input file");
    fio::File f = s.files.open(s.in_sim.object_cnt);
    // the reference loops until the end of the file, re-reading the counts: the last line wins
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        if (f.read(items(s.bsn), items(s.sp_ob)) < 0) break;
    }
    if (s.bsn_cc.gwflow == 1) throw std::runtime_error("SWAT+ gwflow is not supported");
    s.ob.alloc(1, s.sp_ob.objs, object_connectivity());
}

void time_read(Sim& s) {
    std::string titldum, header;
    time_current& t = s.time;
    if (s.files.exists(s.in_sim.time) || s.in_sim.time != "null") {
        fio::File f = s.files.open(s.in_sim.time);
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
    if (s.files.exists(s.in_basin.parms_bas) || s.in_basin.parms_bas != "null") {
        fio::File f = s.files.open(s.in_basin.parms_bas);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            read_items(f, s.bsn_prm);
            break;
        }
    }
}

void basin_prm_default(Sim& s) {
    basin_parms& p = s.bsn_prm;
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
    std::string titldum, header, name;
    basin_print_codes& p = s.pco;
    // the print.prt objects in the order of a file without labels (use_obj_labels = n)
    const std::vector<print_interval*> fixed = {
        &p.wb_bsn, &p.nb_bsn, &p.ls_bsn, &p.pw_bsn, &p.aqu_bsn, &p.res_bsn, &p.chan_bsn, &p.sd_chan_bsn,
        &p.recall_bsn, &p.wb_reg, &p.nb_reg, &p.ls_reg, &p.pw_reg, &p.aqu_reg, &p.res_reg, &p.sd_chan_reg,
        &p.recall_reg, &p.water_allo, &p.wb_lsu, &p.nb_lsu, &p.ls_lsu, &p.pw_lsu, &p.wb_hru, &p.nb_hru,
        &p.ls_hru, &p.pw_hru, &p.wb_sd, &p.nb_sd, &p.ls_sd, &p.pw_sd, &p.chan, &p.sd_chan,
        &p.aqu, &p.res, &p.recall, &p.hyd, &p.ru, &p.pest, &p.salt_basin, &p.salt_hru,
        &p.salt_ru, &p.salt_aqu, &p.salt_chn, &p.salt_res, &p.salt_wet, &p.cs_basin, &p.cs_hru, &p.cs_ru,
        &p.cs_aqu, &p.cs_chn, &p.cs_res, &p.cs_wet};
    // the labels of a file read by label
    const std::vector<std::pair<const char*, print_interval*>> labels = {
        {"basin_wb", &p.wb_bsn}, {"basin_nb", &p.nb_bsn}, {"basin_ls", &p.ls_bsn}, {"basin_pw", &p.pw_bsn},
        {"basin_aqu", &p.aqu_bsn}, {"basin_res", &p.res_bsn}, {"basin_cha", &p.chan_bsn}, {"basin_sd_cha", &p.sd_chan_bsn},
        {"basin_psc", &p.recall_bsn}, {"region_wb", &p.wb_reg}, {"region_nb", &p.nb_reg}, {"region_ls", &p.ls_reg},
        {"region_pw", &p.pw_reg}, {"region_aqu", &p.aqu_reg}, {"region_res", &p.res_reg}, {"region_sd_cha", &p.sd_chan_reg},
        {"region_psc", &p.recall_reg}, {"water_allo", &p.water_allo}, {"lsunit_wb", &p.wb_lsu}, {"lsunit_nb", &p.nb_lsu},
        {"lsunit_ls", &p.ls_lsu}, {"lsunit_pw", &p.pw_lsu}, {"lsu_cb_gl", &p.cb_gl_lsu}, {"lsu_cb_trf", &p.cb_trf_lsu},
        {"lsu_cb_plt", &p.cb_plt_lsu}, {"hru_wb", &p.wb_hru}, {"hru_nb", &p.nb_hru}, {"hru_ls", &p.ls_hru},
        {"hru_pw", &p.pw_hru}, {"hru_cb", &p.cb_hru}, {"hru_cb_vars", &p.cb_vars_hru}, {"hru_cb_gl", &p.cb_gl_hru},
        {"hru_cb_trf", &p.cb_trf_hru}, {"hru_cb_lyr", &p.cb_lyr_hru}, {"hru_cb_cpool", &p.cb_cpool_hru}, {"hru_cb_npool", &p.cb_npool_hru},
        {"hru_cb_plt", &p.cb_plt_hru}, {"hru_cb_flux", &p.cb_flux_hru}, {"hru_cb_drv", &p.cb_drv_hru}, {"hru_cb_dyn", &p.cb_dyn_hru},
        {"hru_cb_snap", &p.cb_snap_hru}, {"hru-lte_wb", &p.wb_sd}, {"hru-lte_nb", &p.nb_sd}, {"hru-lte_ls", &p.ls_sd},
        {"hru-lte_pw", &p.pw_sd}, {"channel", &p.chan}, {"channel_sd", &p.sd_chan}, {"aquifer", &p.aqu},
        {"reservoir", &p.res}, {"recall", &p.recall}, {"hyd", &p.hyd}, {"ru", &p.ru},
        {"pest", &p.pest}, {"basin_salt", &p.salt_basin}, {"hru_salt", &p.salt_hru}, {"ru_salt", &p.salt_ru},
        {"aqu_salt", &p.salt_aqu}, {"channel_salt", &p.salt_chn}, {"res_salt", &p.salt_res}, {"wetland_salt", &p.salt_wet},
        {"basin_cs", &p.cs_basin}, {"hru_cs", &p.cs_hru}, {"ru_cs", &p.cs_ru}, {"aqu_cs", &p.cs_aqu},
        {"channel_cs", &p.cs_chn}, {"res_cs", &p.cs_res}, {"wetland_cs", &p.cs_wet}, {"gwflow_wb", &p.gwflow_wb},
        {"gwflow_flux", &p.gwflow_flux}, {"gwflow_heat", &p.gwflow_heat}, {"gwflow_solute", &p.gwflow_solute}, {"gwflow_obs", &p.gwflow_obs},
        {"gwflow_pump", &p.gwflow_pump}};
    if (s.files.exists(s.in_sim.prt) || s.in_sim.prt != "null") {
        fio::File f = s.files.open(s.in_sim.prt);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 500)) < 0) break;
            if (f.read(p.nyskip, p.day_start, p.yrc_start, p.day_end, p.yrc_end, p.int_day) < 0) break;
            if (f.read(chars(header, 500)) < 0) break;
            f.read(p.aa_numint);
            if (p.aa_numint > 0) {
                p.aa_yrs.alloc(1, p.aa_numint, 0);
                f.backspace();
                fio::File::Statement st = f.reader();
                st.item(p.aa_numint);
                for (int ii = 1; ii <= p.aa_numint; ++ii) st.item(p.aa_yrs(ii));
                if (st.finish() < 0) break;
            } else {
                p.aa_yrs.alloc(1, 1, 0);
            }
            if (f.read(chars(header, 500)) < 0) break;
            if (f.read(chars(p.csvout, p.csvout_len), chars(p.use_obj_labels, p.use_obj_labels_len),
                       chars(p.cdfout, p.cdfout_len)) < 0)
                break;
            if (f.read(chars(header, 500)) < 0) break;
            if (f.read(chars(p.crop_yld, p.crop_yld_len), chars(p.mgtout, p.mgtout_len), chars(p.hydcon, p.hydcon_len),
                       chars(p.fdcout, p.fdcout_len)) < 0)
                break;
            if (f.read(chars(header, 500)) < 0) break;
            auto read_interval = [&](print_interval& pi) {
                return f.read(chars(name, 16), chars(pi.d, 1), chars(pi.m, 1), chars(pi.y, 1), chars(pi.a, 1));
            };
            if (p.use_obj_labels == "n") {
                for (print_interval* pi : fixed)
                    if (read_interval(*pi) < 0) break;
            } else {
                std::set<print_interval*> already_read_in;
                int eof = 0;
                while (eof >= 0) {
                    name.clear();
                    eof = f.read(chars(name, 16));
                    if (name.empty()) continue;
                    for (const auto& [label, pi] : labels) {
                        if (name == label) {
                            if (already_read_in.count(pi)) {
                                s.warn("print.prt: " + name + " given more than once");
                            } else {
                                f.backspace();
                                eof = read_interval(*pi);
                                already_read_in.insert(pi);
                            }
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    if (p.day_start == 0) p.day_start = 1;
    if (p.day_end == 0) p.day_end = 366;
    if (p.yrc_start == 0) p.yrc_start = s.time.yrc;
    if (p.yrc_end == 0) p.yrc_end = s.time.yrc + s.time.nbyr;
    if (p.int_day <= 0) p.int_day = 1;
    p.int_day_cur = p.int_day;
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
        const int nda = s.ndays(i_mo + 1);
        if (jd <= nda) {
            mo = i_mo;
            day_mo = jd - s.ndays(i_mo);
            return;
        }
    }
}

}  // namespace swat
