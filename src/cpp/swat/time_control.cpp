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
        const WeatherDaily& w = s.wst(j).weat;
        s.dump_record(1, j,
                      {w.precip, w.precip_next, w.tmax, w.tmin, w.tave, w.solrad, w.solradmx, w.rhum, w.dewpt, w.windsp,
                       w.pet, w.wndir, w.phubase0, w.ppet, w.daylength, w.precip_half_hr});
    }
}

}  // namespace

void time_control(Sim& s) {
    Time& t = s.time;
    t.yrc = t.yrc_start;
    // generate precip for the first day (precip_next)
    t.day_end_yr = (t.yrc % 4 == 0) ? ndays_leap[12] : ndays_noleap[12];
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
                    s.ndays = ndays_leap;
                    t.num_leap = t.num_leap + 1;
                } else {
                    s.ndays = ndays_noleap;
                }
            } else {
                s.ndays = ndays_leap;
                t.num_leap = t.num_leap + 1;
            }
        } else {
            s.ndays = ndays_noleap;
        }
        if (t.yrs > 1 || t.day_start == 0) t.day_start = 1;
        t.day_end_yr = s.ndays[12];
        if (t.yrs == t.nbyr && t.day_end > 0) {
            t.day_end_yr = t.day_end;
            t.day_end_yr = std::min(t.day_end_yr, s.ndays[12]);
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
            if (t.day == s.ndays[static_cast<std::size_t>(t.mo)]) t.end_mo = 1;
            if (t.day == t.day_end_yr) {
                t.end_yr = 1;
                if (t.yrs == t.nbyr) {
                    t.end_sim = 1;
                    t.yrs_prt = t.yrs_prt / (365.0f + static_cast<float>(t.num_leap / t.nbyr));
                }
            }
            if (t.yrs > s.pco.nyskip) s.ndmo[static_cast<std::size_t>(t.mo)] += 1;

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
    cli_staread(s);
    s.time_init = s.time;
    time_control(s);
}

}  // namespace swat
