// SPDX-License-Identifier: GPL-3.0-or-later
//
// The simulation: main.f90's sequence of reads and initialisations, and the yearly and
// daily loops of time_control.f90.

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

namespace {

void dump_weather(Sim& s) {
    if (!s.dump) return;
    for (int j = 1; j <= s.db_mx.wst; ++j) {
        const weather_daily& w = s.wst(j).weat;
        s.dump_record(1, j,
                      {w.precip, w.precip_next, w.tmax, w.tmin, w.tave, w.solrad, w.solradmx, w.rhum, w.dewpt, w.windsp,
                       w.pet, w.wndir, w.phubase0, w.ppet, w.daylength, w.precip_half_hr});
    }
}

// ls_dump_init of the reference: soil layers (tag 3, id = 100 hru + layer), hru values
// (tag 4) and plants (tag 5, id = 100 hru + plant) at the start of the daily loop
void dump_init(Sim& s) {
    if (!s.dump) return;
    for (int j = 1; j <= s.sp_ob.hru; ++j) {
        const soil_profile& so = s.soil(j);
        for (int ly = 1; ly <= so.nly; ++ly) {
            const soil_physical_properties& p = so.phys(ly);
            s.dump_record(3, 100 * j + ly,
                          {p.d, p.thick, p.bd, p.k, p.cbn, p.clay, p.silt, p.sand, p.rock, p.conv_wt, p.crdep, p.awc,
                           p.fc, p.hk, p.por, p.st, p.tmp, p.ul, p.up, p.wp, p.wpmm, p.tot_sw});
        }
        const hydrologic_response_unit& h = s.hru(j);
        const plant_community& pc = s.pcom(j);
        const plant_community_mass& pm = s.pl_mass(j);
        s.dump_record(4, j,
                      {so.sw, so.sumfc, so.sumul, so.avpor, so.avbd, so.sumwp, s.cn2(j), h.lumv.usle_ls, h.lumv.usle_p,
                       s.wfsh(j), h.topo.slope, h.hyd.perco_lim, s.cvm_com(j), pc.rsd_covfac,
                       static_cast<float>(so.nly), static_cast<float>(pc.npl), h.sno_mm, h.snocov1, h.snocov2,
                       h.hyd.pet_co, h.topo.lat_len, pm.rsd_tot.m, pm.rsd_tot.n, pc.laimx_sum});
        for (int ipl = 1; ipl <= pc.npl; ++ipl) {
            const plant_status& c = pc.plcur(ipl);
            const plant_growth& g = pc.plg(ipl);
            s.dump_record(5, 100 * j + ipl,
                          {c.phumat, c.phuacc, static_cast<float>(c.curyr_mat), c.phumat_p, c.phuacc_p, g.lai, g.cht,
                           g.laimxfr, g.root_dep, pm.tot(ipl).m, pm.tot(ipl).n, pm.tot(ipl).p, pm.root(ipl).m,
                           pm.ab_gr(ipl).m, pm.leaf(ipl).m, pm.stem(ipl).m, pm.seed(ipl).m, pc.plm(ipl).n_fr,
                           pc.plm(ipl).p_fr, c.lai_pot});
        }
    }
    // delivery ratios of the routing units' elements (tag 6)
    for (int i = s.ru_elem.lbound(); i <= s.ru_elem.ubound(); ++i)
        if (i >= 1) s.dump_record(6, i, values(s.ru_elem(i).dr));
}

// phubase, yr_skip and a pending skip operation of a HRU at the end of its year
void reset_year_hru(Sim& s, int j) {
    s.phubase(j) = 0.0f;
    s.yr_skip(j) = 0;
    const int isched = s.hru(j).mgt_ops;
    if (isched > 0 && s.sched(isched).num_ops > 0) {
        hydrologic_response_unit& h = s.hru(j);
        if (s.sched(isched).mgt_ops(h.cur_op).op == "skip") h.cur_op = h.cur_op + 1;
        if (h.cur_op > s.sched(isched).num_ops) h.cur_op = 1;
    }
}

}  // namespace

void time_control(Sim& s) {
    time_current& t = s.time;
    t.yrc = t.yrc_start;
    dump_init(s);
    // generate precip for the first day (precip_next)
    t.day_end_yr = (t.yrc % 4 == 0) ? s.ndays_leap(13) : s.ndays_noleap(13);
    t.yrs = 1;
    t.day = t.day_start;
    int mo = 0, day_mo = 0;
    xmon(s, t.day, mo, day_mo);
    t.mo = mo;
    t.day_mo = day_mo;
    cli_precip_control(s, 0);

    for (int curyr = 1; curyr <= t.nbyr; ++curyr) {
        t.yrs = curyr;
        if (t.yrc % 4 == 0) {
            if (t.yrc % 100 == 0) {
                if (t.yrc % 400 == 0) {
                    s.ndays = s.ndays_leap;
                    t.num_leap = t.num_leap + 1;
                } else {
                    s.ndays = s.ndays_noleap;
                }
            } else {
                s.ndays = s.ndays_leap;
                t.num_leap = t.num_leap + 1;
            }
        } else {
            s.ndays = s.ndays_noleap;
        }
        if (t.yrs > 1 || t.day_start == 0) t.day_start = 1;
        t.day_end_yr = s.ndays(13);
        if (t.yrs == t.nbyr && t.day_end > 0) {
            t.day_end_yr = t.day_end;
            t.day_end_yr = std::min(t.day_end_yr, s.ndays(13));
        }
        if (t.yrs > s.pco.nyskip) {
            t.yrs_prt = t.yrs_prt + static_cast<float>(t.day_end_yr - t.day_start + 1);
            t.days_prt = t.days_prt + static_cast<float>(t.day_end_yr - t.day_start + 1);
        } else {
            t.yrs_prt = static_cast<float>(t.nbyr);
        }

        for (int julian_day = t.day_start; julian_day <= t.day_end_yr; ++julian_day) {
            t.day = julian_day;
            xmon(s, t.day, mo, day_mo);
            t.mo = mo;
            t.day_mo = day_mo;
            t.yrc_tot = t.yrc_end - t.yrc_start + 1;
            t.end_mo = 0;
            t.end_yr = 0;
            t.end_sim = 0;
            if (t.day == s.ndays(t.mo + 1)) t.end_mo = 1;
            if (t.day == t.day_end_yr) {
                t.end_yr = 1;
                if (t.yrs == t.nbyr) {
                    t.end_sim = 1;
                    t.yrs_prt = t.yrs_prt / (365.0f + static_cast<float>(t.num_leap / t.nbyr));
                }
            }
            if (s.sp_ob.hru > 0) sim_initday(s);
            if (t.yrs > s.pco.nyskip) s.ndmo(t.mo) += 1;

            climate_control(s);
            dump_weather(s);
            // irrigation of the day (zeroed here for water allocation) and the sediment budget
            for (int j = s.irrig.lbound(); j <= s.irrig.ubound(); ++j) {
                s.irrig(j).runoff = 0.0f;
                s.irrig(j).applied = 0.0f;
                s.irrig(j).eff = 0.0f;
                s.irrig(j).demand = 0.0f;
            }
            s.bsn_sedbud = s.bsn_sedbudz;
            for (int i = s.ch_morph.lbound(); i <= s.ch_morph.ubound(); ++i) s.ch_morph(i) = s.ch_morphz;
            s.ch_morph_ord.fill(s.ch_morphz);
            command(s);
            // southern hemisphere: heat units and skipped years reset near the winter solstice
            if (t.day == 181) {
                for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru)
                    if (s.ob(s.sp_ob1.hru + ihru - 1).lat < 0.0f) reset_year_hru(s, ihru);
            }
        }
        // end of the year
        for (int j = 1; j <= s.sp_ob.hru; ++j) {
            if (s.bsn_cc.cswat == 0 && s.hru(j).hyd.biomix > 1.e-6f) mgt_newtillmix_cswat0(s, j, s.hru(j).hyd.biomix, 0);
            plant_community& pc = s.pcom(j);
            for (int ipl = 1; ipl <= pc.npl; ++ipl) {
                const int idp = pc.plcur(ipl).idplt;
                if (idp > 0 && s.pldb(idp).typ == "perennial") {
                    pc.plcur(ipl).curyr_mat = pc.plcur(ipl).curyr_mat + 1;
                    pc.plcur(ipl).curyr_mat = std::min(pc.plcur(ipl).curyr_mat, s.pldb(idp).mat_yrs);
                }
            }
            if (s.ob(s.sp_ob1.hru + j - 1).lat >= 0.0f) reset_year_hru(s, j);
        }
        t.yrc = t.yrc + 1;
    }
}

void run(Sim& s, const RunOptions& opt) {
    s.dump = opt.dump;
    proc_bsn(s);
    proc_date_time(s);
    proc_db(s);
    proc_read(s);
    hyd_connect(s);
    // main.f90's order around the HRU set-up
    cli_lapse(s);
    om_water_init(s);
    proc_hru(s);
    proc_cha(s);  // calls time_conc_init
    proc_aqu(s);
    dtbl_lum_read(s);
    proc_cond(s);
    hru_dtbl_actions_init(s);
    if (s.bsn_cc.cswat != 0) throw std::runtime_error("SWAT+ carbon models other than the static one (carbon = 0) are not supported");
    proc_cal(s);
    for (s.ihru = 1; s.ihru <= s.sp_ob.hru; ++s.ihru) soil_nutcarb_init(s);
    dr_ru(s);
    for (const auto& [i, v] : s.inject_dr) std::memcpy(&s.ru_elem(i).dr, v.data(), sizeof(hyd_output));
    s.time_init = s.time;
    time_control(s);
}

}  // namespace swat
