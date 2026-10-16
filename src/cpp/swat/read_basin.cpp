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
    // only the first line changes the simulation (the years skipped); the rest selects outputs
    std::string titldum, header;
    if (s.files.exists(s.in_sim.prt) || s.in_sim.prt != "null") {
        fio::File f = s.files.open(s.in_sim.prt);
        if (f.read(chars(titldum, 80)) < 0) return;
        if (f.read(chars(header, 500)) < 0) return;
        basin_print_codes& p = s.pco;
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
        const int nda = s.ndays(i_mo + 1);
        if (jd <= nda) {
            mo = i_mo;
            day_mo = jd - s.ndays(i_mo);
            return;
        }
    }
}

}  // namespace swat
