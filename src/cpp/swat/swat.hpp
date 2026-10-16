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

// databases, files and connections
void proc_db(Sim& s);
void proc_read(Sim& s);
void hyd_connect(Sim& s);
void define_unit_elements(Sim& s, int num_elem, int& ielem);
void proc_cal(Sim& s);

// climate
void proc_date_time(Sim& s);
void cli_staread(Sim& s);
void cli_precip_control(Sim& s, int istart);
void climate_control(Sim& s);

// objects, routing units, aquifers and channels (command.cpp, aquifer.cpp, channel.cpp)
void om_water_init(Sim& s);
void cli_lapse(Sim& s);
void proc_cha(Sim& s);
void proc_aqu(Sim& s);
void dr_ru(Sim& s);
void command(Sim& s);
void ru_control(Sim& s);
void aqu_1d_control(Sim& s);
void sd_channel_control3(Sim& s);
void rcurv_interp_dep(Sim& s, int icha, float flow_dep);
void rcurv_interp_flo(Sim& s, int icha, float flo_rate);

// HRU (hru_daily.cpp, mgt.cpp)
void hru_control(Sim& s);
void sim_initday(Sim& s);
void mgt_newtillmix_cswat0(Sim& s, int jj, float bmix, int idtill);
void dtbl_lum_read(Sim& s);
void proc_cond(Sim& s);
void hru_dtbl_actions_init(Sim& s);
void mgt_operatn(Sim& s);
void hru_dtbl_control(Sim& s);

// time
void time_control(Sim& s);
void xmon(const Sim& s, int jd, int& mo, int& day_mo);

// hru
void proc_hru(Sim& s);
void time_conc_init(Sim& s);
void soil_nutcarb_init(Sim& s);
void curno(Sim& s, float cnn, int h);
void pl_rootfr(Sim& s, int j);
void pl_root_gro(Sim& s, int j);
void pl_seed_gro(Sim& s, int j);
void pl_partition(Sim& s, int j, int init);

// utilities
void ascrv(float x1, float x2, float x3, float x4, float& x5, float& x6);
int fstrcmp(const std::string& a, const std::string& b);
void search(const Vec<std::string>& sch, int max, const std::string& cfind, int& iseq);

}  // namespace swat
