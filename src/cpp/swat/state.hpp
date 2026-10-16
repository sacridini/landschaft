// SPDX-License-Identifier: GPL-3.0-or-later
//
// The state of a SWAT+ simulation: the variables of the SWAT+ 62.0.1 Fortran modules
// the port covers, under the same names (time_module -> Time, climate_module -> the
// climate members, ...). Defaults are the Fortran initialisers.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "swat/farray.hpp"
#include "swat/fio.hpp"

namespace swat {

// ---- input_file_module ----------------------------------------------------------
struct InputFiles {
    // simulation
    std::string time = "time.sim", prt = "print.prt", object_prt = "object.prt", object_cnt = "object.cnt",
                cs_db = "constituents.cs";
    // basin
    std::string codes_bas = "codes.bsn", parms_bas = "parameters.bsn", carbon_bsn = "carbon.bsn";
    // climate
    std::string weat_sta = "weather-sta.cli", weat_wgn = "weather-wgn.cli", pet_cli = "pet.cli", pcp_cli = "pcp.cli",
                tmp_cli = "tmp.cli", slr_cli = "slr.cli", hmd_cli = "hmd.cli", wnd_cli = "wnd.cli",
                atmo_cli = "atmodep.cli";
    // connect
    std::string hru_con = "hru.con", hruez_con = "hru-lte.con", ru_con = "rout_unit.con", gwflow_con = "gwflow.con",
                aqu_con = "aquifer.con", aqu2d_con = "aquifer2d.con", chan_con = "channel.con",
                res_con = "reservoir.con", rec_con = "recall.con", exco_con = "exco.con", delr_con = "delratio.con",
                out_con = "outlet.con", chandeg_con = "chandeg.con";
    // channel
    std::string cha_init = "initial.cha", cha_dat = "channel.cha", cha_hyd = "hydrology.cha",
                cha_sed = "sediment.cha", cha_nut = "nutrients.cha", chan_ez = "channel-lte.cha",
                hyd_sed = "hyd-sed-lte.cha", cha_temp = "temperature.cha";
    // reservoir
    std::string init_res = "initial.res", res = "reservoir.res", hyd_res = "hydrology.res", sed_res = "sediment.res",
                nut_res = "nutrients.res", weir_res = "weir.res", wet = "wetland.wet", hyd_wet = "hydrology.wet";
    // routing unit
    std::string ru_def = "rout_unit.def", ru_ele = "rout_unit.ele", ru = "rout_unit.rtu", ru_dr = "rout_unit.dr";
    // hru
    std::string hru_data = "hru-data.hru", hru_ez = "hru-lte.hru";
    // exco
    std::string exco = "exco.exc", exco_om = "exco_om.exc", exco_pest = "exco_pest.exc", exco_path = "exco_path.exc",
                exco_hmet = "exco_hmet.exc", exco_salt = "exco_salt.exc";
    // recall
    std::string recall_rec = "recall.rec";
    // delivery ratio
    std::string del_ratio = "delratio.del", dr_om = "dr_om.del", dr_pest = "dr_pest.del", dr_path = "dr_path.del",
                dr_hmet = "dr_hmet.del", dr_salt = "dr_salt.del";
    // aquifer
    std::string aqu_init = "initial.aqu", aqu = "aquifer.aqu";
    // herd
    std::string animal = "animal.hrd", herd = "herd.hrd", ranch = "ranch.hrd";
    // water allocation
    std::string pou = "place_of_use.wro", pod = "point_of_diver.wro";
    // link
    std::string chan_surf = "chan-surf.lin", aqu_cha = "aqu_cha.lin";
    // hydrology
    std::string hydrol_hyd = "hydrology.hyd", topogr_hyd = "topography.hyd", field_fld = "field.fld";
    // structural
    std::string tiledrain_str = "tiledrain.str", septic_str = "septic.str", fstrip_str = "filterstrip.str",
                grassww_str = "grassedww.str", bmpuser_str = "bmpuser.str";
    // parameter databases
    std::string plants_plt = "plants.plt", fert_frt = "fertilizer.frt", till_til = "tillage.til",
                pest = "pesticide.pes", pathcom_db = "pathogens.pth", hmetcom_db = "metals.mtl",
                saltcom_db = "salt.slt", urban_urb = "urban.urb", septic_sep = "septic.sep", snow = "snow.sno";
    // operations
    std::string harv_ops = "harv.ops", graze_ops = "graze.ops", irr_ops = "irr.ops", chem_ops = "chem_app.ops",
                fire_ops = "fire.ops", sweep_ops = "sweep.ops";
    // land use management
    std::string landuse_lum = "landuse.lum", management_sch = "management.sch", cntable_lum = "cntable.lum",
                cons_prac_lum = "cons_practice.lum", ovn_lum = "ovn_table.lum";
    // calibration
    std::string cal_parms = "cal_parms.cal", cal_upd = "calibration.cal", codes_sft = "codes.sft",
                wb_parms_sft = "wb_parms.sft", water_balance_sft = "water_balance.sft",
                ch_sed_budget_sft = "ch_sed_budget.sft", ch_sed_parms_sft = "ch_sed_parms.sft",
                plant_parms_sft = "plant_parms.sft", plant_gro_sft = "plant_gro.sft";
    // initialisation
    std::string plant_ini = "plant.ini", soil_plant_ini = "soil_plant.ini", om_water = "om_water.ini",
                pest_soil = "pest_hru.ini", pest_water = "pest_water.ini", path_soil = "path_hru.ini",
                path_water = "path_water.ini", hmet_soil = "hmet_hru.ini", hmet_water = "hmet_water.ini",
                salt_soil = "salt_hru.ini", salt_water = "salt_water.ini";
    // soils
    std::string soils_sol = "soils.sol", nut_sol = "nutrients.sol", lte_sol = "soils_lte.sol";
    // decision tables
    std::string dtbl_lum = "lum.dtl", dtbl_res = "res_rel.dtl", dtbl_scen = "scen_lu.dtl", dtbl_flo = "flo_con.dtl";
    // regions
    std::string ele_lsu = "ls_unit.ele", def_lsu = "ls_unit.def", ele_reg = "ls_reg.ele", def_reg = "ls_reg.def",
                cal_lcu = "ls_cal.reg", ele_cha = "ch_catunit.ele", def_cha = "ch_catunit.def",
                def_cha_reg = "ch_reg.def", ele_aqu = "aqu_catunit.ele", def_aqu = "aqu_catunit.def",
                def_aqu_reg = "aqu_reg.def", ele_res = "res_catunit.ele", def_res = "res_catunit.def",
                def_res_reg = "res_reg.def", ele_psc = "rec_catunit.ele", def_psc = "rec_catunit.def",
                def_psc_reg = "rec_reg.def";
    // weather paths (len=80, default blank)
    std::string path_pcp, path_tmp, path_slr, path_hmd, path_wnd, path_pet;
    std::string out_path;
};

// ---- time_module ------------------------------------------------------------------
struct Time {
    std::string day_print = "n";
    int day = 0, mo = 0, mo_start = 0, yrc = 0, yrc_start = 0, yrc_end = 0, yrs = 0, day_mo = 0, end_mo = 0,
        end_yr = 0, end_sim = 0, end_aa_prt = 0, day_start = 0, day_end_yr = 0, day_end = 0, nbyr = 3, step = 0;
    float dtm = 0.0f, days_prt = 0.0f, yrs_prt = 0.0f, yrs_prt_int = 0.0f;
    int num_leap = 0, prt_int_cur = 1, yrc_tot = 0;
};

inline constexpr std::array<int, 13> ndays_leap = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
inline constexpr std::array<int, 13> ndays_noleap = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};

// ---- basin_module -------------------------------------------------------------------
struct BasinCodes {
    std::string petfile = "pet.cli", wwqfile;
    int pet = 0, nam1 = 0, crk = 0, swift_out = 0, sed_det = 0, rte = 0, deg = 0, wq = 0, nostress = 0, cn = 0,
        cfac = 0, cswat = 0, lapse = 0, uhyd = 1, sed_ch = 0, tdrn = 0, wtdn = 0, sol_p_model = 0, gampt = 0;
    std::string atmo = "a";
    int smax = 0, qual2e = 0, gwflow = 0, idc_till = 3;
};

struct BasinParms {
    float evlai = 3.0f, ffcb = 0.0f, surlag = 4.0f, adj_pkr = 1.0f, prf = 484.0f, pestgwfact = 0.0f,
          temp_decay = 0.0f, cmn = 0.003f, n_updis = 20.0f, p_updis = 20.0f, nperco = 0.10f, pperco = 10.0f,
          phoskd = 175.0f, psp = 0.40f, rsdco = 0.05f, percop = 0.5f, msk_co1 = 0.75f, msk_co2 = 0.25f,
          msk_x = 0.20f, nperco_lchtile = 0.5f, evrch = 0.60f, scoef = 1.0f, cdn = 1.40f, sdnco = 1.30f,
          bact_swf = 0.15f, tb_adj = 0.0f, cn_froz = 0.000862f, dorm_hr = -1.0f, plaps = 0.0f, tlaps = 6.5f,
          nfixmx = 20.0f, decr_min = 0.01f, rsd_covco = 0.75f, urb_init_abst = 1.0f, petco_pmpt = 100.0f,
          uhalpha = 1.0f, eros_spl = 0.0f, rill_mult = 0.0f, eros_expo = 0.0f, c_factor = 0.0f, ch_d50 = 0.0f,
          co2 = 400.0f;
    int day_lag_mx = 0, igen = 5;
};

struct Basin {
    std::string name;
    float area_ls_ha = 0.0f, area_tot_ha = 0.0f;
};

struct SpatialObjects {
    int objs = 0, hru = 0, hru_lte = 0, ru = 0, gwflow = 0, aqu = 0, chan = 0, res = 0, recall = 0, exco = 0, dr = 0,
        canal = 0, pump = 0, outlet = 0, chandeg = 0, aqu2d = 0, herd = 0, wro = 0;
};

struct PrintCodes {
    int nyskip = 0, day_start = 0, yrc_start = 0, day_end = 0, yrc_end = 0, int_day = 1;
};

// hru_module uptake
struct Uptake {
    float water_dis = 10.0f, water_norm = 0.0f, n_norm = 0.0f, p_norm = 0.0f;
};

// ---- climate_module -------------------------------------------------------------------
struct WeatherGeneratorDb {
    float lat = 0.0f, long_ = 0.0f, elev = 0.0f, rain_yrs = 10.0f;
    std::array<float, 13> tmpmx{}, tmpmn{}, tmpstdmx{}, tmpstdmn{}, pcpmm{}, pcpstd{}, pcpskw{}, pr_wd{}, pr_ww{},
        pcpd{}, rainhmx{}, solarav{}, dewpt{}, windav{};  // index 1..12
};

struct WgnParms {
    std::array<float, 13> pr_wdays{}, pcpmean{};
    float daylmn = 0.0f, daylth = 0.0f, latsin = 0.0f, latcos = 0.0f, phutot = 0.0f, pcpdays = 0.0f, tmp_an = 0.0f,
          pcp_an = 0.0f, ppet_an = 0.0f, precip_sum = 0.0f, pet_sum = 0.0f, p_pet_rto = 0.0f;
    std::array<float, 13> pcf{}, amp_r{}, pet{};
    Vec<int> mne_ppet;
    Vec<float> precip_mce, pet_mce;
    int ireg = 1, idewpt = 0;
};

struct WeatherDaily {
    float precip = 0.0f, precip_next = 0.0f, tmax = 0.0f, tmin = 0.0f, tave = 0.0f, solrad = 0.0f, solradmx = 0.0f,
          rhum = 0.0f, dewpt = 0.0f, windsp = 0.0f, pet = 0.0f, wndir = 0.0f, phubase0 = 0.0f, ppet = 0.0f,
          daylength = 0.0f, precip_half_hr = 0.0f;
    std::string precip_prior_day = "dry";
    Vec<float> ts, ts_next;
};

struct WeatherStation {
    std::string name = "Farmer Branch IL";
    float lat = 0.0f;
    // wco_c
    std::string c_wgn, c_pgage, c_tgage, c_sgage, c_hgage, c_wgage, c_petgage, c_atmodep;
    // wco
    int wgn = 1, pgage = 0, tgage = 0, sgage = 0, hgage = 0, wgage = 0, petgage = 0, atmodep = 0;
    WeatherDaily weat;
    float precip_aa = 0.0f, pet_aa = 0.0f;
    int pcp_ts = 0;
    std::array<float, 13> rfinc{}, tmpinc{}, radinc{}, huminc{};
    Vec<float> tlag;
    float airlag_temp = 0.0f;
    int tlag_mne = 1;
};

struct ClimateMeasured {
    std::string filename;
    float lat = 0.0f, long_ = 0.0f, elev = 0.0f;
    int nbyr = 0, tstep = 0, days_gen = 0, yrs_start = 1, start_day = 0, start_yr = 0, end_day = 0, end_yr = 0;
    std::array<float, 13> mean_mon{}, max_mon{}, min_mon{};
    Mat<float> ts, ts2;
    Cube<float> tss;
};

// maximum_data_module (the counts the port uses)
struct DbMax {
    int wst = 0, wgnsta = 0, pcpfiles = 0, tmpfiles = 0, slrfiles = 0, rhfiles = 0, wndfiles = 0, petfiles = 0,
        atmodep = 0;
};

// One daily dump record, the same as the reference's ls_dump_module writes.
struct DumpRecord {
    int tag, yrc, day, id;
    std::vector<float> values;
};

struct Sim {
    fio::FileSet files;
    InputFiles in;
    Time time, time_init;
    std::array<int, 13> ndays = ndays_leap;
    std::array<int, 13> ndmo{};
    BasinCodes bsn_cc;
    BasinParms bsn_prm;
    Basin bsn;
    SpatialObjects sp_ob;
    PrintCodes pco;
    Uptake uptake;
    DbMax db_mx;

    // climate_module
    int ppet_ndays = 30, ppet_mce = 0;
    Mat<float> frad, wgncur, wgnold;
    Vec<int> idg;
    Mat<int> rndseed;
    Vec<float> rnd2, rnd3, rnd8, rnd9;
    int rndseed_cond = 748932582;
    Vec<float> co2y;
    Vec<WeatherGeneratorDb> wgn;
    Vec<WgnParms> wgn_pms;
    Vec<WeatherStation> wst;
    Vec<ClimateMeasured> pcp, tmp, slr, hmd, wnd, petm;
    Vec<std::string> wst_n, wgn_n, pcp_n, tmp_n, slr_n, hmd_n, wnd_n, atmo_n, petm_n;
    int iwst = 0;  // hydrograph_module: current weather station

    // what the reference writes to diagnostics.out
    std::vector<std::string> warnings;
    void warn(const std::string& m) { warnings.push_back(m); }

    // daily dump
    bool dump = false;
    std::vector<DumpRecord> dumps;
    void dump_record(int tag, int id, std::vector<float> v) {
        if (dump) dumps.push_back({tag, time.yrc, time.day, id, std::move(v)});
    }
};

}  // namespace swat
