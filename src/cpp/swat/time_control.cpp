// SPDX-License-Identifier: GPL-3.0-or-later
//
// The simulation: main.f90's sequence of reads and initialisations, and the yearly and
// daily loops of time_control.f90.

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
            if (t.yrs > s.pco.nyskip) s.ndmo(t.mo) += 1;

            climate_control(s);
            dump_weather(s);
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
    proc_hru(s);
    time_conc_init(s);  // in proc_cha
    if (s.bsn_cc.cswat != 0) throw std::runtime_error("SWAT+ carbon models other than the static one (carbon = 0) are not supported");
    proc_cal(s);
    for (s.ihru = 1; s.ihru <= s.sp_ob.hru; ++s.ihru) soil_nutcarb_init(s);
    s.time_init = s.time;
    time_control(s);
}

}  // namespace swat
