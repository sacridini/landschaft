// SPDX-License-Identifier: GPL-3.0-or-later
//
// SWAT+ 62.0.1 ported to C++ (the subset of ls.hydro.swat): the Fortran subroutines
// under their own names, each taking the simulation state.

#pragma once

#include <string>

#include "swat/state.hpp"

namespace swat {

// What run() does besides simulating.
struct RunOptions {
    bool dump = true;  // keep the daily records of the reference's ls_dump_module
};

// Run the project in s.files (a TxtInOut directory read into memory).
void run(Sim& s, const RunOptions& opt = {});

// proc_bsn and its readers
void proc_bsn(Sim& s);
void readcio_read(Sim& s);
void basin_read_cc(Sim& s);
void basin_read_objs(Sim& s);
void time_read(Sim& s);
void basin_read_prm(Sim& s);
void basin_prm_default(Sim& s);
void basin_print_codes_read(Sim& s);
void co2_read(Sim& s);

// climate
void proc_date_time(Sim& s);
void cli_staread(Sim& s);
void cli_precip_control(Sim& s, int istart);
void climate_control(Sim& s);

// time
void time_control(Sim& s);
void xmon(const Sim& s, int jd, int& mo, int& day_mo);

// utilities
int fstrcmp(const std::string& a, const std::string& b);
void search(const Vec<std::string>& sch, int max, const std::string& cfind, int& iseq);

}  // namespace swat
