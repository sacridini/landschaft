// SPDX-License-Identifier: GPL-3.0-or-later
//
// The HRUs before the first day: proc_hru.f90 and what it calls (allocate_parms,
// hru_read, hrudb_init, hru_lum_init_all, topohyd_init, soils_init, soil_phys_init,
// structure_init, plant_all_init, plant_init, cn2_init_all, curno, hydro_init) and
// time_conc_init.f90 (called from proc_cha).

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/om_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

// allocate_parms.f90 (the arrays of the port; zero0 sets the random roughness to 20)
void allocate_parms(Sim& s) {
    const int mhru = s.sp_ob.hru;
    const int step = s.time.step;
    s.mhyd = 1;
    s.wnan.alloc(1, 10, 0.0f);
    s.ranrns_hru.alloc(1, mhru, 20.0f);
    const int mpc = 20;
    s.uno3d.alloc(1, mpc, 0.0f);
    s.uapd.alloc(1, mpc, 0.0f);
    s.un2.alloc(1, mpc, 0.0f);
    s.up2.alloc(1, mpc, 0.0f);
    s.translt.alloc(1, mpc, 0.0f);
    s.par.alloc(1, mpc, 0.0f);
    s.htfac.alloc(1, mpc, 0.0f);
    s.epmax.alloc(1, mpc, 0.0f);
    auto vf = [&](Vec<float>& v) { v.alloc(1, mhru, 0.0f); };
    auto vi = [&](Vec<int>& v) { v.alloc(1, mhru, 0); };
    vf(s.cvm_com); vf(s.percn); vi(s.i_sep); vi(s.sep_tsincefail); vf(s.qstemm); vf(s.bio_bod); vf(s.biom);
    vf(s.rbiom); vf(s.fcoli); vf(s.bz_perc); vf(s.plqm); vi(s.itb);
    s.hhqday.alloc(1, mhru, 1, step, 0.0f);
    vf(s.sol_sumno3); vf(s.sol_sumsolp); vi(s.iseptic); vi(s.grz_days); vf(s.brt); vf(s.canstor); vf(s.cbodu);
    vf(s.chl_a); vf(s.cklsp); vf(s.cn2); vf(s.cnday); vf(s.cumei); vf(s.cumeira); vf(s.cumrt); vf(s.cumrai);
    vf(s.dormhr); vf(s.doxq); vf(s.filterw); vi(s.igrz); vi(s.yr_skip); vi(s.isweep); vf(s.phusw); vf(s.latno3);
    vf(s.latq); vi(s.ndeat); vf(s.nplnt); vf(s.orgn_con); vf(s.orgp_con); vf(s.ovrlnd); vf(s.phubase); vf(s.pplnt);
    vf(s.qdr); vf(s.gwsoilq); vf(s.satexq); vf(s.gwsoiln); vf(s.gwsoilp); vf(s.satexn); vf(s.sstmaxd);
    vf(s.sedminpa); vf(s.sedminps); vf(s.sedorgn); vf(s.sedorgp); vf(s.sedyld); vf(s.sanyld); vf(s.silyld);
    vf(s.clayld); vf(s.sagyld); vf(s.lagyld); vf(s.grayld); vf(s.sed_con); vf(s.sepbtm); vf(s.smx); vf(s.soln_con);
    vf(s.solp_con); vf(s.stmaxd); vi(s.itill); vf(s.surfq); vf(s.surqno3); vf(s.surqsolp); vi(s.swtrg);
    vf(s.rateinf_prev); vf(s.urb_abstinit); vf(s.t_ov); vf(s.tconc); vf(s.tc_gwat); vf(s.tileno3); vf(s.twash);
    vf(s.usle_cfac); vf(s.usle_eifac); vf(s.wfsh);
    s.bss.alloc(1, 40, 1, mhru, 0.0f);
    s.bss_ex.alloc(1, 10, 1, mhru, 0.0f);
    s.wrt.alloc(1, 2, 1, mhru, 0.0f);
    s.surf_bs.alloc(1, 55, 1, mhru, 0.0f);
    s.ubnrunoff.alloc(1, step, 0.0f);
    s.ubntss.alloc(1, step, 0.0f);
    s.hhsedy.alloc(1, mhru, 1, step, 0.0f);
    s.ovrlnd_dt.alloc(1, mhru, 1, step, 0.0f);
    vf(s.init_abstrc);
    s.hhsurfq.alloc(1, mhru, 1, step, 0.0f);
    vi(s.tillage_switch); vf(s.tillage_depth); vi(s.tillage_days); vf(s.tillage_factor);
}

// hru_allo.f90 (the arrays of the port)
void hru_allo(Sim& s) {
    const int imax = s.sp_ob.hru;
    s.hru.alloc(0, imax, hydrologic_response_unit());
    s.soil.alloc(0, imax, soil_profile());
    s.soil1.alloc(0, imax, soil_profile_mass());
    s.soil1_init.alloc(0, imax, soil_profile_mass());
    s.pcom.alloc(0, imax, plant_community());
    s.pl_mass.alloc(0, imax, plant_community_mass());
    s.irrig.alloc(0, imax, irrigation_water_transfer());
    s.wet.alloc(0, imax, hyd_output());
}

// hru_read.f90
void hru_read(Sim& s) {
    allocate_parms(s);
    const std::string& file = s.in_hru.hru_data;
    std::string titldum, header;
    if (!s.files.exists(file) || file == "null") {
        s.hru_db.alloc(0, 0, hydrologic_response_unit_db());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        int imax = 0, eof = 0;
        while (eof == 0) {
            int i = 0;
            eof = f.read(i);
            if (eof < 0) break;
            imax = std::max(imax, i);
        }
        s.hru_db.alloc(0, imax, hydrologic_response_unit_db());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (s.ihru = 1; s.ihru <= imax; ++s.ihru) {
            int i = 0, k = 0;
            if (f.read(i) < 0) break;
            f.backspace();
            hydrologic_response_unit_db& h = s.hru_db(i);
            if (f.read(k, items(h.dbsc)) < 0) break;
            for (int ilum = 1; ilum <= s.db_mx.landuse; ++ilum) {
                if (fstrcmp(h.dbsc.land_use_mgt, s.lum(ilum).name) == 0) {
                    h.dbs.land_use_mgt = ilum;
                    break;
                }
            }
            if (h.dbs.land_use_mgt == 0) s.warn(h.dbsc.land_use_mgt + "not found (landuse.lum)");
            for (int isp_ini = 1; isp_ini <= s.db_mx.sol_plt_ini; ++isp_ini) {
                soil_plant_initialize& p = s.sol_plt_ini(isp_ini);
                if (fstrcmp(h.dbsc.soil_plant_init, p.name) == 0) {
                    h.dbs.soil_plant_init = isp_ini;
                    for (int ics = 1; ics <= s.db_mx.soiltest; ++ics) {
                        if (fstrcmp(p.nutc, s.solt_db(ics).name) == 0) {
                            p.nut = ics;
                            break;
                        }
                    }
                    break;
                }
            }
            for (int ith = 1; ith <= s.db_mx.topo; ++ith) {
                if (fstrcmp(h.dbsc.topo, s.topo_db(ith).name) == 0) {
                    h.dbs.topo = ith;
                    break;
                }
            }
            if (h.dbs.topo == 0) s.warn(h.dbsc.topo + "not found (topography.hyd)");
            for (int ithyd = 1; ithyd <= s.db_mx.hyd; ++ithyd) {
                if (fstrcmp(h.dbsc.hyd, s.hyd_db(ithyd).name) == 0) {
                    h.dbs.hyd = ithyd;
                    break;
                }
            }
            if (h.dbs.hyd == 0) s.warn(h.dbsc.hyd + "not found (hydrograph.hyd)");
            for (int isol = 1; isol <= s.db_mx.soil; ++isol) {
                if (fstrcmp(h.dbsc.soil, s.soildb(isol).s.snam) == 0) {
                    h.dbs.soil = isol;
                    break;
                }
            }
            if (h.dbs.soil == 0) s.warn(h.dbsc.soil + "not found (soils.sol)");
            for (int isno = 1; isno <= s.db_mx.sno; ++isno) {
                if (fstrcmp(h.dbsc.snow, s.snodb(isno).name) == 0) {
                    h.dbs.snow = isno;
                    break;
                }
            }
            if (h.dbs.snow == 0 && h.dbsc.snow != "null") s.warn(h.dbsc.snow + "not found (snow.sno)");
            for (int ifld = 1; ifld <= s.db_mx.field; ++ifld) {
                if (fstrcmp(h.dbsc.field, s.field_db(ifld).name) == 0) {
                    h.dbs.field = ifld;
                    break;
                }
            }
            if (h.dbs.field == 0 && h.dbsc.field != "null") s.warn(h.dbsc.field + "not found (field.fld)");
            if (h.dbsc.surf_stor != "null" && !h.dbsc.surf_stor.empty())
                throw std::runtime_error("SWAT+ wetlands (surf_stor of hru-data.hru) are not supported");
        }
        break;
    }
}

// hrudb_init.f90
void hrudb_init(Sim& s) {
    for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
        const int iob = s.sp_ob1.hru + ihru - 1;
        const int ihru_db = s.ob(iob).props;
        hydrologic_response_unit& h = s.hru(ihru);
        h.dbs = s.hru_db(ihru_db).dbs;
        h.dbsc = s.hru_db(ihru_db).dbsc;
        h.obj_no = s.sp_ob1.hru + ihru - 1;
        h.area_ha = s.ob(iob).area_ha;
        h.km = s.ob(iob).area_ha / 100.0f;
        h.land_use_mgt_c = s.hru_db(ihru_db).dbsc.land_use_mgt;
        const int ilu = h.dbs.land_use_mgt;
        h.cal_group = s.lum(ilu).cal_group;
        h.nut.phoskd = s.bsn_prm.phoskd;
        h.nut.pperco = s.bsn_prm.pperco;
        h.nut.psp = s.bsn_prm.psp;
        h.nut.nperco = s.bsn_prm.nperco;
        h.nut.cmn = s.bsn_prm.cmn;
        h.nut.nperco_lchtile = s.bsn_prm.nperco_lchtile;
    }
}

// hru_lum_init.f90
void hru_lum_init(Sim& s, int iihru) {
    hydrologic_response_unit& h = s.hru(iihru);
    const int ilu = h.land_use_mgt;
    s.pcom(iihru).name = s.lum(ilu).plant_cov;
    h.plant_cov = s.lum_str(ilu).plant_cov;
    h.lum_group_c = s.lum(ilu).cal_group;
    for (int ilug = 1; ilug <= s.lum_grp.num; ++ilug)
        if (fstrcmp(h.lum_group_c, s.lum_grp.name(ilu)) == 0) h.lum_group = ilug;
    h.mgt_ops = s.lum_str(ilu).mgt_ops;
    h.tiledrain = s.lum_str(ilu).tiledrain;
    h.septic = s.lum_str(ilu).septic;
    h.fstrip = s.lum_str(ilu).fstrip;
    h.grassww = s.lum_str(ilu).grassww;
    h.bmpuser = s.lum_str(ilu).bmpuser;
    h.luse.cn_lu = s.lum_str(ilu).cn_lu;
    h.luse.cons_prac = s.lum_str(ilu).cons_prac;
}

// topohyd_init.f90
void topohyd_init(Sim& s) {
    for (s.ihru = 1; s.ihru <= s.sp_ob.hru; ++s.ihru) {
        hydrologic_response_unit& h = s.hru(s.ihru);
        const int iob = s.sp_ob1.hru + s.ihru - 1;
        const int itopo_db = h.dbs.topo;
        const int ihyd_db = h.dbs.hyd;
        const int itopohd_db = h.dbs.topo;
        const int ifield_db = h.dbs.field;
        const topography_db& t = s.topo_db(itopohd_db);
        const hydrology_db& y = s.hyd_db(ihyd_db);
        h.topo.name = s.topo_db(itopo_db).name;
        h.topo.elev = s.ob(iob).elev;
        h.topo.slope = t.slope;
        h.topo.slope_len = t.slope_len;
        h.hyd.name = y.name;
        h.hyd.lat_ttime = y.lat_ttime;
        h.hyd.lat_sed = y.lat_sed;
        h.topo.lat_len = t.lat_len;
        if (h.topo.lat_len < 1.e-6f) h.topo.lat_len = 50.0f;
        h.hyd.canmx = y.canmx;
        h.hyd.esco = y.esco;
        h.hyd.epco = y.epco;
        h.hyd.erorgn = y.erorgn;
        h.hyd.erorgp = y.erorgp;
        h.hyd.cn3_swf = y.cn3_swf;
        h.hyd.perco = y.perco;
        const int isno_db = h.dbs.snow;
        h.sno = s.snodb(isno_db);
        ascrv(0.5f, 0.95f, h.sno.cov50, 0.95f, h.snocov1, h.snocov2);
        if (h.tiledrain > 0) {
            h.hyd.cn3_swf = 0.95f;
            h.hyd.perco = 0.1f;
        }
        if (h.hyd.perco > 1.e-9f) {
            const float perc_ln_func = 1.0052f * std::log(-std::log(h.hyd.perco - 1.e-6f)) + 5.6862f;
            h.hyd.perco_lim = std::exp(-perc_ln_func);
            h.hyd.perco_lim = fmin2(1.0f, h.hyd.perco_lim);
        } else {
            h.hyd.perco_lim = 0.0f;
        }
        h.topo.dis_stream = t.dis_stream;
        h.hyd.biomix = y.biomix;
        h.hyd.lat_orgn = y.lat_orgn;
        h.hyd.lat_orgp = y.lat_orgp;
        h.hyd.latq_co = y.latq_co;
        h.hyd.pet_co = y.pet_co;
        if (h.hyd.pet_co < 1.e-6f) {
            h.hyd.pet_co = 1.0f;
        } else if (h.hyd.pet_co < 0.01f) {
            h.hyd.pet_co = h.hyd.pet_co / 0.0023f;
        }
        h.field.length = s.field_db(ifield_db).length;
        h.field.wid = s.field_db(ifield_db).wid;
        h.field.ang = s.field_db(ifield_db).ang;
        h.topo.dep_co = t.dep_co;
        const int isno = h.dbs.snow;
        h.sno_mm = s.snodb(isno).init_mm;
        h.hydcal = h.hyd;
    }
}

// soil_phys_init.f90
void soil_phys_init(Sim& s, int isol) {
    soil_profile& p = s.sol(isol).s;
    soil_hru_database& d = s.sol(isol);
    if (p.alb < 0.1f) p.alb = 0.1f;
    if (p.anion_excl <= 1.e-6f) p.anion_excl = 0.5f;
    if (p.anion_excl >= 1.0f) p.anion_excl = 0.99f;
    int nly = p.nly;
    for (int j = 1; j <= nly; ++j) {
        soil_physical_properties& ph = d.phys(j);
        const float a = 50.0f, b = 20.0f, c = 5.0f, dd = 2.0f, nota = 10.0f;
        if (ph.k <= 0.0f) {
            if (p.hydgrp == "A") ph.k = a;
            else if (p.hydgrp == "B") ph.k = b;
            else if (p.hydgrp == "C") ph.k = c;
            else if (p.hydgrp == "D") ph.k = dd;
            else ph.k = nota;
        }
        if (ph.bd <= 1.e-6f) ph.bd = 1.3f;
        if (ph.bd > 2.0f) ph.bd = 2.0f;
        if (ph.awc <= 1.e-6f) ph.awc = 0.005f;
        if (ph.awc >= 0.8f) ph.awc = 0.8f;
        if (ph.rock > 98.0f) ph.rock = 98.0f;
        if (d.ly(j).cal <= 1.e-6f) d.ly(j).cal = 2.8f;
        if (d.ly(j).ph <= 1.e-6f) d.ly(j).ph = 6.5f;
    }
    nly = p.nly;
    for (int j = 1; j <= nly; ++j) {
        soil_physical_properties& ph = d.phys(j);
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
        d.ly(j).vwt = (437.13f * fpowi(drpor, 2)) - (95.08f * drpor) + 8.257f;
    }
    const float sa = d.phys(1).sand / 100.0f;
    const float cl = d.phys(1).clay / 100.0f;
    const float si = d.phys(1).silt / 100.0f;
    p.det_san = sa * fpow(1.0f - cl, 2.49f);
    p.det_sil = 0.13f * si;
    p.det_cla = 0.20f * cl;
    if (cl < 0.25f) {
        p.det_sag = 2.0f * cl;
    } else if (cl > 0.5f) {
        p.det_sag = 0.57f;
    } else {
        p.det_sag = 0.28f * (cl - 0.25f) + 0.5f;
    }
    p.det_lag = 1.0f - p.det_san - p.det_sil - p.det_cla - p.det_sag;
    if (p.det_lag < 0.0f) {
        p.det_san = p.det_san / (1.0f - p.det_lag);
        p.det_sil = p.det_sil / (1.0f - p.det_lag);
        p.det_cla = p.det_cla / (1.0f - p.det_lag);
        p.det_sag = p.det_sag / (1.0f - p.det_lag);
        p.det_lag = 0.0f;
    }
    float depth_prev = 0.0f;
    float sumpor = 0.0f;
    for (int j = 1; j <= nly; ++j) {
        soil_physical_properties& ph = d.phys(j);
        ph.thick = ph.d - depth_prev;
        const float pormm = ph.por * ph.thick;
        sumpor = sumpor + pormm;
        ph.ul = (ph.por - ph.wp) * ph.thick;
        p.sumul = p.sumul + ph.ul;
        ph.fc = ph.thick * (ph.up - ph.wp);
        p.sumfc = p.sumfc + ph.fc;
        ph.st = ph.fc * p.ffc;
        ph.hk = (ph.ul - ph.fc) / ph.k;
        if (ph.hk < 1.0f) ph.hk = 1.0f;
        p.sw = p.sw + ph.st;
        ph.wpmm = ph.wp * ph.thick;
        p.sumwp = p.sumwp + ph.wpmm;
        ph.crdep = p.crk * 0.916f * std::exp(-0.0012f * ph.d) * ph.thick;
        d.ly(j).volcr = ph.crdep * (ph.fc - ph.st) / (ph.fc);
        depth_prev = ph.d;
    }
    p.swpwt = p.sw;
    if (p.ffc > 1.0f) {
        p.wat_tbl = (p.det_lag - p.ffc * p.sumfc) / d.phys(nly).d;
    } else {
        p.wat_tbl = 0.0f;
    }
    p.avpor = sumpor / d.phys(nly).d;
    p.avbd = 2.65f * (1.0f - p.avpor);
    if (s.time.step > 1) d.phys(1).sand = 100.0f - d.phys(1).clay - d.phys(1).silt;
}

// soils_init.f90 (septic layers outside the port)
void soils_init(Sim& s) {
    const int msoils = std::max(0, s.db_mx.soil);
    s.sol.alloc(0, msoils, soil_hru_database());
    const bool custom = s.files.exists("soil_lyr_depths.sol");
    for (int isol = 1; isol <= msoils; ++isol) {
        soil_database& db = s.soildb(isol);
        soil_hru_database& so = s.sol(isol);
        so.s.snam = db.s.snam;
        so.s.hydgrp = db.s.hydgrp;
        so.s.zmx = db.s.zmx;
        so.s.anion_excl = db.s.anion_excl;
        so.s.crk = db.s.crk;
        so.s.texture = db.s.texture;
        int mlyr = 0;
        auto copy = [&](int j, const soilayer_db& l) {
            so.phys(j).bd = l.bd;
            so.phys(j).awc = l.awc;
            so.phys(j).k = l.k;
            so.phys(j).cbn = l.cbn;
            so.phys(j).clay = l.clay;
            so.phys(j).silt = l.silt;
            so.phys(j).sand = l.sand;
            so.phys(j).rock = l.rock;
            so.ly(j).alb = l.alb;
            so.ly(j).usle_k = l.usle_k;
            so.ly(j).ec = l.ec;
            so.ly(j).cal = l.cal;
            so.ly(j).ph = l.ph;
        };
        if (!custom) {
            so.s.nly = db.ly(1).z > 19.5f ? db.s.nly + 1 : db.s.nly;
            mlyr = so.s.nly;
            so.ly.alloc(1, mlyr, soilayer());
            so.phys.alloc(1, mlyr, soil_physical_properties());
            so.phys(1).d = 10.0f;
            copy(1, db.ly(1));
            if (db.ly(1).z > 19.5f) {
                for (int j = 2; j <= mlyr; ++j) {
                    so.phys(j).d = db.ly(j - 1).z;
                    copy(j, db.ly(j - 1));
                }
            } else {
                for (int j = 2; j <= mlyr; ++j) {
                    so.phys(j).d = db.ly(j).z;
                    copy(j, db.ly(j));
                }
            }
        } else {
            // custom depths: 1 mm layers averaged into the layers of soil_lyr_depths.sol
            int tot_soil_depth = static_cast<int>(db.ly(db.s.nly).z);
            Vec<soilayer_db> mm;
            mm.alloc(1, tot_soil_depth, soilayer_db());
            int prev_depth = 0;
            for (int j = 1; j <= db.s.nly; ++j) {
                for (int i = prev_depth + 1; i <= tot_soil_depth; ++i) {
                    if (static_cast<float>(i) <= db.ly(j).z && i > prev_depth) {
                        soilayer_db l = db.ly(j);
                        l.z = static_cast<float>(i);
                        mm(i) = l;
                    } else {
                        break;
                    }
                }
                prev_depth = static_cast<int>(db.ly(j).z);
            }
            fio::File f = s.files.open("soil_lyr_depths.sol");
            std::string titldum, header, units;
            if (f.read(chars(titldum, 80)) < 0) throw std::runtime_error("soil_lyr_depths.sol: no title");
            if (f.read(chars(header, 500)) < 0) throw std::runtime_error("soil_lyr_depths.sol: no header");
            if (f.read(chars(units, 16)) < 0) throw std::runtime_error("soil_lyr_depths.sol: no units");
            mlyr = 0;
            int csld = 0;
            for (;;) {
                if (f.read(csld) < 0) break;
                if (csld <= tot_soil_depth) {
                    if (csld > 10) {
                        ++mlyr;
                        if (csld == tot_soil_depth) break;
                    } else {
                        continue;
                    }
                } else {
                    csld = tot_soil_depth;
                    ++mlyr;
                    break;
                }
            }
            mlyr = mlyr + 1;
            tot_soil_depth = std::min(tot_soil_depth, csld);
            so.s.nly = mlyr;
            so.ly.alloc(1, mlyr, soilayer());
            so.phys.alloc(1, mlyr, soil_physical_properties());
            f.rewind();
            f.read(chars(titldum, 80));
            f.read(chars(header, 500));
            f.read(chars(units, 16));
            bool first_layer_flag = true;
            int pcd = 1;
            for (int i = 1; i <= mlyr; ++i) {
                for (;;) {
                    f.read(csld);
                    if (csld > 10) break;
                }
                if (first_layer_flag) {
                    so.phys(i).d = 10.0f;
                    csld = 10;
                    first_layer_flag = false;
                    f.backspace();
                } else {
                    if (csld > tot_soil_depth) csld = tot_soil_depth;
                    so.phys(i).d = static_cast<float>(csld);
                }
                int n = 0;
                float sum_bd = 0.0f, sum_awc = 0.0f, sum_cbn = 0.0f, sum_k = 0.0f, sum_clay = 0.0f, sum_silt = 0.0f,
                      sum_sand = 0.0f, sum_rock = 0.0f, sum_alb = 0.0f, sum_usle_k = 0.0f, sum_ec = 0.0f,
                      sum_cal = 0.0f, sum_ph = 0.0f;
                for (int j = pcd; j <= csld; ++j) {
                    const soilayer_db& l = mm(j);
                    sum_bd = sum_bd + l.bd;
                    sum_awc = sum_awc + l.awc;
                    sum_k = sum_k + l.k;
                    sum_cbn = sum_cbn + l.cbn;
                    sum_clay = sum_clay + l.clay;
                    sum_silt = sum_silt + l.silt;
                    sum_sand = sum_sand + l.sand;
                    sum_rock = sum_rock + l.rock;
                    sum_alb = sum_alb + l.alb;
                    sum_usle_k = sum_usle_k + l.usle_k;
                    sum_ec = sum_ec + l.ec;
                    sum_cal = sum_cal + l.cal;
                    sum_ph = sum_ph + l.ph;
                    ++n;
                }
                const float fn = static_cast<float>(n);
                so.phys(i).bd = sum_bd / fn;
                so.phys(i).awc = sum_awc / fn;
                so.phys(i).k = sum_k / fn;
                so.phys(i).cbn = sum_cbn / fn;
                so.phys(i).clay = sum_clay / fn;
                so.phys(i).silt = sum_silt / fn;
                so.phys(i).sand = sum_sand / fn;
                so.phys(i).rock = sum_rock / fn;
                so.ly(i).alb = sum_alb / fn;
                so.ly(i).usle_k = sum_usle_k / fn;
                so.ly(i).ec = sum_ec / fn;
                so.ly(i).cal = sum_cal / fn;
                so.ly(i).ph = sum_ph / fn;
                pcd = csld + 1;
            }
        }
        if (s.sol_test.allocated()) throw std::runtime_error("SWAT+ soil test adjustments are not supported");
    }
    for (int isol = 1; isol <= msoils; ++isol) soil_phys_init(s, isol);
    for (s.ihru = 1; s.ihru <= s.sp_ob.hru; ++s.ihru) {
        const int j = s.ihru;
        const int isol = s.hru(j).dbs.soil;
        const soil_physical_properties& p1 = s.sol(isol).phys(1);
        s.wfsh(j) = 10.0f * std::exp(6.5309f - 7.32561f * p1.por + 3.809479f * fpowi(p1.por, 2) +
                                     0.001583f * fpowi(p1.clay, 2) + 0.000344f * p1.sand * p1.clay -
                                     0.049837f * p1.por * p1.sand + 0.001608f * fpowi(p1.por, 2) * fpowi(p1.sand, 2) +
                                     0.001602f * fpowi(p1.por, 2) * fpowi(p1.clay, 2) -
                                     0.0000136f * fpowi(p1.sand, 2) * p1.clay -
                                     0.003479f * fpowi(p1.clay, 2) * p1.por - 0.000799f * fpowi(p1.sand, 2) * p1.por);
        soil_profile& sp = s.soil(j);
        sp = s.sol(isol).s;
        const int nly = sp.nly;
        sp.ly.alloc(1, nly, soilayer());
        sp.phys.alloc(1, nly, soil_physical_properties());
        for (int ly = 1; ly <= nly; ++ly) {
            sp.phys(ly) = s.sol(isol).phys(ly);
            sp.ly(ly) = s.sol(isol).ly(ly);
        }
        s.isep = s.iseptic(j);
        if (s.sep(s.isep).opt != 0) throw std::runtime_error("SWAT+ septic systems are not supported");
        soil_profile_mass& m = s.soil1(j);
        m.sw.alloc(1, nly, 0.0f);
        m.cbn.alloc(1, nly, 0.0f);
        m.sed.alloc(1, nly, sediment());
        m.rsd_tot.alloc(1, nly, organic_mass());
        m.emix.alloc(1, nly, 0.0f);
        m.root_tot.alloc(1, nly, organic_mass());
        m.mn.alloc(1, nly, mineral_nitrogen());
        m.mp.alloc(1, nly, mineral_phosphorus());
        m.tot.alloc(1, nly, organic_mass());
        m.seq.alloc(1, nly, organic_mass());
        m.hact.alloc(1, nly, organic_mass());
        m.hsta.alloc(1, nly, organic_mass());
        m.str.alloc(1, nly, organic_mass());
        m.lig.alloc(1, nly, organic_mass());
        m.nonlig.alloc(1, nly, organic_mass());
        m.meta.alloc(1, nly, organic_mass());
        m.hs.alloc(1, nly, organic_mass());
        m.hp.alloc(1, nly, organic_mass());
        m.microb.alloc(1, nly, organic_mass());
        m.man.alloc(1, nly, organic_mass());
        m.water.alloc(1, nly, organic_mass());
        soil_profile_mass& mi = s.soil1_init(j);
        mi = m;
    }
}

// structure_init.f90: the port has no structures
void structure_init(Sim& s) {
    for (int j = 1; j <= s.sp_ob.hru; ++j) {
        const land_use_management& l = s.lum(s.hru(j).land_use_mgt);
        if (l.tiledrain != "null" || l.fstrip != "null" || l.grassww != "null" || l.bmpuser != "null")
            throw std::runtime_error("SWAT+ structures (tile drains, filter strips, grassed waterways, user BMPs) of "
                                     "land use " + l.name + " are not supported");
        if (l.septic != "null") throw std::runtime_error("SWAT+ septic systems are not supported");
    }
}

}  // namespace

// pl_rootfr.f90
void pl_rootfr(Sim& s, int j) {
    plant_growth& g = s.pcom(j).plg(s.ipl);
    const soil_profile& so = s.soil(j);
    g.rtfr.fill(0.0f);
    if (g.root_dep < 1.e-6f) {
        g.rtfr(1) = 1.0f;
        return;
    }
    const float a = 1.15f, b = 11.7f, c = 0.022f, d = 0.12029f;
    int k = 0;
    float cum_rf = 0.0f;
    for (int ly = 1; ly <= so.nly; ++ly) {
        const float cum_rd = so.phys(ly).d >= g.root_dep ? g.root_dep : so.phys(ly).d;
        const float x1 = (cum_rd - so.phys(ly).thick) / g.root_dep;
        const float x2 = cum_rd / g.root_dep;
        float xx1 = -b * x1;
        if (xx1 > 20.0f) xx1 = 20.0f;
        float xx2 = -b * x2;
        if (xx2 > 20.0f) xx2 = 20.0f;
        g.rtfr(ly) = (a / b * (std::exp(xx1) - std::exp(xx2)) + c * (x2 - x1)) / d;
        const float xx = cum_rf;
        cum_rf = cum_rf + g.rtfr(ly);
        if (cum_rf > 1.0f) {
            g.rtfr(ly) = 1.0f - xx;
            cum_rf = 1.0f;
        }
        k = ly;
        if (cum_rd >= g.root_dep) break;
    }
    for (int ly = 1; ly <= so.nly; ++ly) {
        g.rtfr(ly) = g.rtfr(ly) / cum_rf;
        if (ly == k) break;
    }
}

// pl_root_gro.f90
void pl_root_gro(Sim& s, int j) {
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    plant_growth& g = pc.plg(ipl);
    if (db.typ == "warm_annual" || db.typ == "cold_annual" || db.typ == "warm_annual_tuber" ||
        db.typ == "cold_annual_tuber") {
        g.root_dep = 2.5f * pc.plcur(ipl).phuacc * 1000.0f * db.rdmx;
    } else {
        g.root_dep = 2.5f * pc.plcur(ipl).phuacc_p * 1000.0f * db.rdmx;
    }
    if (g.root_dep > s.soil(j).zmx) g.root_dep = s.soil(j).zmx;
    if (g.root_dep < 25.4f) g.root_dep = 25.4f;
    if (db.typ == "perennial") {
        if (db.mat_yrs > 0) {
            const float rto = static_cast<float>(pc.plcur(ipl).curyr_mat) / static_cast<float>(db.mat_yrs);
            if (rto < 0.2f) {
                g.root_frac = db.rsr1 - (db.rsr1 - db.rsr2) * rto / 0.2f;
            } else {
                g.root_frac = db.rsr2;
            }
        } else {
            g.root_frac = db.rsr2;
        }
    } else {
        const float phumax = fmin2(1.0f, pc.plcur(ipl).phuacc);
        g.root_frac = db.rsr1 - (db.rsr1 - db.rsr2) * phumax;
    }
    pl_rootfr(s, j);
}

// pl_seed_gro.f90
void pl_seed_gro(Sim& s, int j) {
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    plant_growth& g = pc.plg(ipl);
    s.iwst = s.ob(j).wst;
    const float phuacc = pc.plcur(ipl).phuacc;
    float ajhi = pc.plcur(ipl).harv_idx * 100.0f * phuacc / (100.0f * phuacc + std::exp(11.1f - 10.0f * phuacc));
    if (phuacc > 0.5f) {
        g.plet = g.plet + s.ep_day;
        g.plpet = g.plpet + s.pet_day;
        float etr = 0.0f;
        if (g.plpet > 1.e-6f) {
            etr = 100.0f * g.plet / g.plpet;
            const float ajhi_min = ajhi / 2.0f;
            ajhi = (ajhi - ajhi_min) * (etr / (etr + std::exp(6.13f - 0.0883f * etr))) + ajhi_min;
            if (ajhi > db.hvsti) ajhi = db.hvsti;
        } else {
            etr = 1.0f;
        }
        const float dhi = ajhi - g.hi_prev;
        // the water and temperature adjustments are computed but not applied (as the reference)
        g.hi_adj = g.hi_adj + dhi;
        g.hi_adj = fmin2(db.hvsti, g.hi_adj);
        g.hi_adj = fmax2(0.0f, g.hi_adj);
        g.hi_prev = ajhi;
    } else {
        g.hi_prev = 0.0f;
    }
}

// pl_partition.f90
void pl_partition(Sim& s, int j, int init) {
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    const plant_growth& g = pc.plg(ipl);
    organic_mass& tot = pm.tot(ipl);
    organic_mass& root = pm.root(ipl);
    organic_mass& seed = pm.seed(ipl);
    organic_mass& leaf = pm.leaf(ipl);
    organic_mass& stem = pm.stem(ipl);
    organic_mass& ab_gr = pm.ab_gr(ipl);
    tot.m = tot.m + s.pl_mass_up.m;
    tot.c = tot.c + s.pl_mass_up.c;
    const float leaf_frac_veg = db.typ == "perennial" ? 0.02f : 0.30f;
    const float leaf_mass_frac_veg = leaf_frac_veg * g.lai / pc.plcur(ipl).lai_pot;
    float root_frac = 0.0f, ab_gr_frac = 0.0f, seed_mass_frac = 0.0f, leaf_mass_frac = 0.0f, stem_mass_frac = 0.0f;
    if (db.typ == "warm_annual_tuber" || db.typ == "cold_annual_tuber") {
        root_frac = g.root_frac;
        ab_gr_frac = (1.0f - root_frac) / (1.0f + g.hi_adj);
        seed_mass_frac = 1.0f - root_frac - ab_gr_frac;
        leaf_mass_frac = leaf_mass_frac_veg * ab_gr_frac;
        stem_mass_frac = (1.0f - leaf_mass_frac_veg) * ab_gr_frac;
    } else {
        root_frac = g.root_frac;
        ab_gr_frac = 1.0f - root_frac;
        seed_mass_frac = g.hi_adj;
        leaf_mass_frac = leaf_mass_frac_veg * (1.0f - seed_mass_frac);
        stem_mass_frac = 1.0f - (leaf_mass_frac_veg + seed_mass_frac);
    }
    if (init == 0) {
        float mass_left = s.pl_mass_up.m;
        auto grow = [&](organic_mass& part, float frac) {
            const float mass_act = part.m;
            const float mass_opt = frac * tot.m;
            if (mass_act < mass_opt) {
                float mass_add = mass_opt - mass_act;
                mass_add = fmin2(mass_add, mass_left);
                part.m = part.m + mass_add;
                mass_left = mass_left - mass_add;
            }
        };
        grow(root, root_frac);
        grow(seed, seed_mass_frac);
        grow(leaf, leaf_mass_frac);
        stem.m = stem.m + mass_left;
        ab_gr.m = stem.m + leaf.m + seed.m;
    } else {
        ab_gr.m = ab_gr_frac * tot.m;
        root.m = root_frac * tot.m;
        leaf.m = leaf_mass_frac * ab_gr.m;
        seed.m = seed_mass_frac * ab_gr.m;
        stem.m = stem_mass_frac * ab_gr.m;
    }
    leaf.c = s.c_frac.leaf * leaf.m;
    stem.c = s.c_frac.stem * stem.m;
    seed.c = s.c_frac.seed * seed.m;
    root.c = s.c_frac.root * root.m;
    ab_gr.c = leaf.c + stem.c + seed.c;
    tot.c = ab_gr.c + root.c;
    if (db.typ == "perennial") {
        const float mass_left = leaf.m + stem.m + root.m;
        if (mass_left > 1.e-9f) {
            seed.n = db.cnyld * seed.m;
            float n_left = tot.n - seed.n;
            if (n_left < 0.0f) {
                seed.n = 0.0f;
                n_left = seed.n + n_left;
            }
            leaf.n = n_left * leaf.m / mass_left;
            stem.n = n_left * stem.m / mass_left;
            root.n = n_left * root.m / mass_left;
            ab_gr.n = seed.n + leaf.n + stem.n;
            seed.p = db.cpyld * seed.m;
            float p_left = tot.p - seed.p;
            if (p_left < 0.0f) {
                seed.p = 0.0f;
                p_left = seed.p + p_left;
            }
            leaf.p = p_left * leaf.m / mass_left;
            stem.p = p_left * stem.m / mass_left;
            root.p = p_left * root.m / mass_left;
            ab_gr.p = seed.p + leaf.p + stem.p;
        }
    } else {
        seed.n = db.cnyld * seed.m;
        float n_left = tot.n - seed.n;
        if (n_left < 0.0f) {
            seed.n = 0.9f * tot.n;
            n_left = 0.1f * tot.n;
        }
        const float n_frac = tot.m - seed.m > 1.e-6f ? n_left / (tot.m - seed.m) : 0.0f;
        leaf.n = n_frac * leaf.m;
        stem.n = n_frac * stem.m;
        root.n = n_frac * root.m;
        ab_gr.n = seed.n + leaf.n + stem.n;
        seed.p = db.cpyld * seed.m;
        float p_left = tot.p - seed.p;
        if (p_left < 0.0f) {
            seed.p = 0.9f * tot.p;
            p_left = 0.1f * tot.p;
        }
        const float p_frac = tot.m - seed.m > 1.e-6f ? p_left / (tot.m - seed.m) : 0.0f;
        leaf.p = p_frac * leaf.m;
        stem.p = p_frac * stem.m;
        root.p = p_frac * root.m;
        ab_gr.p = seed.p + leaf.p + stem.p;
    }
}

namespace {

// plant_init.f90 (init = 0: the start of the simulation)
void plant_init(Sim& s, int init, int iihru) {
    if (init != 0) throw std::runtime_error("SWAT+ plant community changes during the run are not supported");
    const int j = iihru;
    hydrologic_response_unit& h = s.hru(j);
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const int icom = h.plant_cov;
    if (icom == 0) {
        pc.npl = 0;
    } else {
        pc.npl = s.pcomdb(icom).plants_com;
        const int n = pc.npl;
        pc.pl.alloc(1, n, "");
        pc.plg.alloc(1, n, plant_growth());
        pc.plm.alloc(1, n, plant_mass());
        pm.tot.alloc(1, n, organic_mass());
        pm.ab_gr.alloc(1, n, organic_mass());
        pm.leaf.alloc(1, n, organic_mass());
        pm.stem.alloc(1, n, organic_mass());
        pm.seed.alloc(1, n, organic_mass());
        pm.root.alloc(1, n, organic_mass());
        pm.yield_tot.alloc(1, n, organic_mass());
        pm.yield_yr.alloc(1, n, organic_mass());
        pc.plstr.alloc(1, n, plant_stress());
        pc.plcur.alloc(1, n, plant_status());
        pm.rsd.alloc(1, n, organic_mass());
        s.soil1(j).pl.alloc(1, n, plant_residue());
        for (int ipl = 1; ipl <= n; ++ipl) {
            pc.plcur(ipl).uptake.alloc(1, s.soil(j).nly, 0.0f);
            pc.plg(ipl).rtfr.alloc(1, s.soil(j).nly, 0.0f);
        }
        for (int ipl = 1; ipl <= n; ++ipl) s.soil1(j).pl(ipl).rsd.alloc(1, s.soil(j).nly, organic_mass());
        pc.rsd_covfac = 0.0f;
        s.cvm_com(j) = 0.0f;
        pc.pcomdb = icom;
        pc.rot_yr = 1;
        pc.laimx_sum = 0.0f;
        pm.rsd.fill(s.plt_mass_z);
        pm.rsd_tot = s.plt_mass_z;
        const plant_community_db& cdb = s.pcomdb(icom);
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            const int ipl = s.ipl;
            const plant_init_db& pin = cdb.pl(ipl);
            plant_status& cur = pc.plcur(ipl);
            pc.pl(ipl) = pin.cpnm;
            cur.gro = pin.igro;
            cur.idorm = "y";
            const int idp = pin.db_num;
            const plant_db& db = s.pldb(idp);
            pm.rsd(ipl).m = pm.rsd(ipl).m + pin.rsdin;
            pm.rsd(ipl).c = pm.rsd(ipl).c + 0.42f * pin.rsdin;
            pm.rsd(ipl).n = pm.rsd(ipl).n + 0.42f * pin.rsdin / 10.0f;
            pm.rsd(ipl).p = pm.rsd(ipl).p + 0.42f * pin.rsdin / 100.0f;
            pm.rsd_tot = pm.rsd_tot + pm.rsd(ipl);
            const int iob = h.obj_no;
            s.iwst = s.ob(iob).wst;
            const int iwgn = s.wst(s.iwst).wco.wgn;
            const weather_generator_db& g = s.wgn(iwgn);
            int mo = 0, day_mo = 0;
            float phu0 = 0.0f;
            for (int iday = 1; iday <= 365; ++iday) {
                xmon(s, iday, mo, day_mo);
                const float tave = (g.tmpmx(mo) + g.tmpmn(mo)) / 2.0f;
                if (tave > 0.0f) phu0 = phu0 + tave;
            }
            int iday_sh = 181;
            int igrow = 0;
            if (static_cast<float>(db.days_mat) < 1.e-6f && static_cast<float>(db.days_mat) > -1.e-6f) {
                float phutot = 0.0f;
                for (int iday = 1; iday <= 365; ++iday) {
                    xmon(s, iday, mo, day_mo);
                    const float tave = (g.tmpmx(mo) + g.tmpmn(mo)) / 2.0f;
                    const float phuday = tave - db.t_base;
                    if (phuday > 0.0f) phutot = phutot + phuday;
                }
                cur.phumat = 0.95f * phutot;
            } else if (static_cast<float>(db.days_mat) < 2.e-6f) {
                cur.phumat = -cur.phumat;
            } else {
                if (db.typ == "warm_annual" || db.typ == "warm_annual_tuber" || db.typ == "cold_annual" ||
                    db.typ == "cold_annual_tuber") {
                    int iday_sum = 181;
                    float phutot = 0.0f;
                    phu0 = 0.15f * phu0;
                    for (int iday = 1; iday <= 365; ++iday) {
                        if (g.lat > 0.0f) {
                            xmon(s, iday, mo, day_mo);
                        } else {
                            iday_sum = iday_sum + 1;
                            iday_sh = iday_sum > 365 ? iday_sum - 365 : iday_sum;
                            xmon(s, iday_sh, mo, day_mo);
                        }
                        const float tave = (g.tmpmx(mo) + g.tmpmn(mo)) / 2.0f;
                        const float phuday = tave;
                        if (phuday > 0.0f) phutot = phutot + phuday;
                        if (phutot > phu0) {
                            igrow = g.lat > 0.0f ? iday : iday_sh;
                            break;
                        }
                    }
                }
                if (db.typ == "null" || db.typ == "null1") {
                    igrow = g.lat > 0.0f ? 1 : 181;
                    int iday = igrow;
                    const wgn_parms& wp = s.wgn_pms(iwgn);
                    for (; iday <= igrow + 180; ++iday) {
                        xmon(s, iday, mo, day_mo);
                        const float sd = std::asin(0.4f * std::sin((static_cast<float>(iday) - 82.0f) / 58.09f));
                        const float sdlat = -wp.latsin * std::tan(sd) / wp.latcos;
                        float hh = 0.0f;
                        if (sdlat > 1.0f) {
                            hh = 0.0f;
                        } else if (sdlat >= -1.0f) {
                            hh = std::acos(sdlat);
                        } else {
                            hh = 3.1416f;
                        }
                        const float daylength = 7.6394f * hh;
                        if (daylength - wp.daylth >= wp.daylmn) break;
                    }
                    igrow = iday;
                }
                float phutot = 0.0f;
                for (int iday = igrow; iday <= igrow + db.days_mat; ++iday) {
                    if (g.lat > 0.0f) {
                        xmon(s, iday, mo, day_mo);
                    } else {
                        iday_sh = iday > 365 ? iday - 365 : iday;
                        xmon(s, iday_sh, mo, day_mo);
                    }
                    const float tave = (g.tmpmx(mo) + g.tmpmn(mo)) / 2.0f;
                    const float phuday = tave - db.t_base;
                    if (phuday > 0.0f) phutot = phutot + phuday;
                }
                cur.phumat = phutot;
            }
            const int isched = h.mgt_ops;
            management_schedule& sc = s.sched(isched);
            if (sc.num_ops > 0) {
                if (sc.mgt_ops(1).jday > 0) {
                    int irot = 1;
                    int jday_prev = sc.mgt_ops(1).jday;
                    int iop = 1;
                    for (; iop <= sc.num_ops; ++iop) {
                        if (irot == cdb.rot_yr_ini && s.time.day_start <= sc.mgt_ops(iop).jday) {
                            break;
                        } else if (sc.mgt_ops(iop).op == "skip" || sc.mgt_ops(iop).jday < jday_prev) {
                            irot = irot + 1;
                        }
                        jday_prev = sc.mgt_ops(iop).jday;
                    }
                    sc.first_op = std::min(iop - 1, sc.num_ops);
                    sc.first_op = std::max(1, sc.first_op);
                } else {
                    sc.first_op = 1;
                }
            }
            h.cur_op = sc.first_op;
            pc.name = cdb.name;
            pc.rot_yr = cdb.rot_yr_ini;
            cur.phuacc = pin.phuacc;
            float matur_frac = 0.0f;
            const plant_cp& cp = s.plcp(idp);
            if (db.typ == "perennial") {
                matur_frac = pin.fr_yrmat;
                pc.plg(ipl).cht = matur_frac * db.chtmx;
            } else {
                matur_frac = pin.phuacc;
                const float f = cur.phuacc / (cur.phuacc + std::exp(cp.leaf1 - cp.leaf2 * cur.phuacc));
                pc.plg(ipl).cht = db.chtmx * std::sqrt(f);
            }
            pc.plg(ipl).laimxfr = matur_frac / (matur_frac + std::exp(cp.leaf1 - cp.leaf2 * matur_frac));
            pc.plg(ipl).lai = pin.igro == "y" ? pin.lai : 0.0f;
            pc.laimx_sum = pc.laimx_sum + db.blai;
            pm.tot(ipl).m = pin.bioms;
            cur.curyr_mat = static_cast<int>(pin.fr_yrmat * static_cast<float>(db.mat_yrs));
            cur.curyr_mat = std::max(1, cur.curyr_mat);
            cur.phumat_p = cur.phumat * static_cast<float>(db.mat_yrs);
            s.cvm_com(j) = cp.cvm + s.cvm_com(j);
            pc.rsd_covfac = pc.rsd_covfac + db.rsd_covfac;
            cur.idplt = pin.db_num;
            pc.plm(ipl).n_fr = (db.pltnfr1 - db.pltnfr3) *
                                   (1.0f - matur_frac / (matur_frac + std::exp(cp.nup1 - cp.nup2 * matur_frac))) +
                               db.pltnfr3;
            pm.tot(ipl).n = pc.plm(ipl).n_fr * pm.tot(ipl).m;
            pc.plm(ipl).p_fr = (db.pltpfr1 - db.pltpfr3) *
                                   (1.0f - matur_frac / (matur_frac + std::exp(cp.pup1 - cp.pup2 * matur_frac))) +
                               db.pltpfr3;
            pm.tot(ipl).p = pc.plm(ipl).p_fr * pm.tot(ipl).m;
            float laimx_pop = 0.0f;
            if (cur.pop_com < 1.e-6f) {
                laimx_pop = db.blai;
            } else {
                const float xx = cur.pop_com / 1001.0f;
                laimx_pop = db.blai * xx / (xx + std::exp(db.pop1 - db.pop2 * xx));
            }
            cur.harv_idx = db.hvsti;
            cur.lai_pot = laimx_pop;
            if (cur.gro == "y") {
                if (db.typ == "perennial") {
                    if (cur.phumat_p > 0.0f) {
                        cur.phuacc_p = pin.fr_yrmat + (cur.phumat * pin.phuacc) / cur.phumat_p;
                    } else {
                        cur.phuacc_p = pin.fr_yrmat;
                    }
                }
                pl_root_gro(s, j);
                pl_seed_gro(s, j);
                pl_partition(s, j, 1);
                pl_rootfr(s, j);
            }
        }
        if (pc.npl > 0) {
            pc.rsd_covfac = pc.rsd_covfac / static_cast<float>(pc.npl);
            s.cvm_com(j) = s.cvm_com(j) / static_cast<float>(pc.npl);
        } else {
            pc.rsd_covfac = 0.0f;
            s.cvm_com(j) = 0.0f;
        }
    }
    const int ilum = h.land_use_mgt;
    for (int ipl = 1; ipl <= pc.npl; ++ipl) pc.plcur(ipl).epco = h.hyd.epco;
    const int icp = s.lum_str(ilum).cons_prac;
    const float xm = 0.6f * (1.0f - std::exp(-35.835f * h.topo.slope));
    const float sin_sl = std::sin(std::atan(h.topo.slope));
    // sl_len = min(slope_len, cons_prac sl_len_mx) is computed and not used by the reference
    h.lumv.usle_ls = fpow(h.topo.slope_len / 22.128f, xm) * (65.41f * sin_sl * sin_sl + 4.56f * sin_sl + 0.065f);
    h.lumv.usle_p = s.cons_prac(icp).pfac;
    h.luse.urb_ro = s.lum(ilum).urb_ro;
    for (int idb = 1; idb <= s.db_mx.urban; ++idb) {
        if (fstrcmp(s.lum(ilum).urb_lu, s.urbdb(idb).urbnm) == 0) {
            h.luse.urb_lu = idb;
            break;
        }
    }
    if (h.luse.urb_lu > 0) throw std::runtime_error("SWAT+ urban land uses are not supported");
    for (int idb = 1; idb <= s.db_mx.ovn; ++idb) {
        if (fstrcmp(s.lum(ilum).ovn, s.overland_n(idb).name) == 0) {
            h.luse.ovn = s.overland_n(idb).ovn;
            break;
        }
    }
}

// plant_all_init.f90
void plant_all_init(Sim& s) {
    s.plts_bsn.alloc(1, s.db_mx.plantparm, "");
    s.bsn_crop_yld.alloc(1, s.db_mx.plantparm, basin_crop_yields());
    s.bsn_crop_yld_aa.alloc(1, s.db_mx.plantparm, basin_crop_yields());
    for (int iihru = 1; iihru <= s.sp_ob.hru; ++iihru) {
        s.ilu = s.hru(iihru).dbs.land_use_mgt;
        s.isol = s.hru(iihru).dbs.soil;
        plant_init(s, 0, iihru);
    }
    for (int iihru = 1; iihru <= s.sp_ob.hru; ++iihru) {
        for (int ipl = 1; ipl <= s.pcom(iihru).npl; ++ipl) {
            if (s.basin_plants == 0) {
                s.plts_bsn(1) = s.pcom(iihru).pl(ipl);
                s.basin_plants = 1;
            } else {
                const int num_plts_cur = s.basin_plants;
                for (int iplt = 1; iplt <= num_plts_cur; ++iplt) {
                    if (fstrcmp(s.pcom(iihru).pl(ipl), s.plts_bsn(iplt)) == 0) break;
                    if (iplt == num_plts_cur) {
                        s.plts_bsn(iplt + 1) = s.pcom(iihru).pl(ipl);
                        s.basin_plants = s.basin_plants + 1;
                    }
                }
            }
        }
    }
    for (int ipl_bsn = 1; ipl_bsn <= s.basin_plants; ++ipl_bsn) {
        s.bsn_crop_yld(ipl_bsn) = s.bsn_crop_yld_z;
        s.bsn_crop_yld_aa(ipl_bsn) = s.bsn_crop_yld_z;
    }
    for (int iihru = 1; iihru <= s.sp_ob.hru; ++iihru) {
        for (int ipl = 1; ipl <= s.pcom(iihru).npl; ++ipl) {
            for (int ipl_bsn = 1; ipl_bsn <= s.basin_plants; ++ipl_bsn) {
                if (fstrcmp(s.pcom(iihru).pl(ipl), s.plts_bsn(ipl_bsn)) == 0) {
                    s.pcom(iihru).plcur(ipl).bsn_num = ipl_bsn;
                    break;
                }
            }
        }
    }
}

}  // namespace

// curno.f90
void curno(Sim& s, float cnn, int h) {
    s.cn2(h) = cnn;
    const float c2 = 100.0f - cnn;
    float cn1 = cnn - 20.0f * c2 / (c2 + std::exp(2.533f - 0.0636f * c2));
    cn1 = fmax2(cn1, 0.4f * cnn);
    const float cn3 = cnn * std::exp(0.006729f * c2);
    s.smx(h) = 254.0f * (100.0f / cn1 - 1.0f);
    const float s3 = 254.0f * (100.0f / cn3 - 1.0f);
    const float rto3 = 1.0f - s3 / s.smx(h);
    const float rtos = 1.0f - 2.54f / s.smx(h);
    const float sumul = s.soil(h).sumul;
    float sumfc = s.soil(h).sumfc + s.hru(h).hyd.cn3_swf * (s.soil(h).sumul - s.soil(h).sumfc);
    if (sumfc < 0.0f) sumfc = 10.0f;
    sumfc = fmax2(0.05f, sumfc);
    sumfc = fmin2(sumul - 0.05f, sumfc);
    ascrv(rto3, rtos, sumfc, sumul, s.wrt(1, h), s.wrt(2, h));
}

namespace {

// cn2_init.f90
void cn2_init(Sim& s, int j) {
    const int ilum = s.hru(j).land_use_mgt;
    const int isol = s.hru(j).dbs.soil;
    const int icn = s.lum_str(ilum).cn_lu;
    const std::string& g = s.sol(isol).s.hydgrp;
    if (g == "A") s.cn2(j) = s.cn(icn).cn(1);
    else if (g == "B") s.cn2(j) = s.cn(icn).cn(2);
    else if (g == "C") s.cn2(j) = s.cn(icn).cn(3);
    else if (g == "D") s.cn2(j) = s.cn(icn).cn(4);
    curno(s, s.cn2(j), j);
}

// hydro_init.f90
void hydro_init(Sim& s) {
    for (int j = 1; j <= s.sp_ob.hru; ++j) {
        s.ihru = j;
        hydrologic_response_unit& h = s.hru(j);
        soil_profile& so = s.soil(j);
        const int iob = h.obj_no;
        const int iwst = s.ob(iob).wst;
        const int iwgn = s.wst(iwst).wco.wgn;
        const float rock = std::exp(-0.053f * so.phys(1).rock);
        h.lumv.usle_mult = rock * so.ly(1).usle_k * h.lumv.usle_p * h.lumv.usle_ls * 11.8f;
        // the reference averages tmpmx(12) with itself
        const float tsoil = (s.wgn(iwgn).tmpmx(12) + s.wgn(iwgn).tmpmx(12)) / 2.0f;
        float sffc = 0.0f;
        if (s.bsn_prm.ffcb <= 0.0f) {
            const float pcp_an = s.wgn_pms(iwgn).pcp_an;
            sffc = pcp_an / (pcp_an + std::exp(9.043f - 0.002135f * pcp_an));
        } else {
            sffc = s.bsn_prm.ffcb;
        }
        const int nly = so.nly;
        so.sw = 0.0f;
        so.ffc = sffc;
        for (int k = 1; k <= nly; ++k) {
            so.phys(k).tmp = tsoil;
            so.phys(k).st = sffc * so.phys(k).fc;
            so.sw = so.sw + so.phys(k).st;
        }
        s.dormhr(j) = s.wgn_pms(iwgn).daylth;
        const float day = static_cast<float>(s.time.day);
        const float sd = std::asin(0.4f * std::sin((day - 82.0f) / 58.09f));
        const float sdlat = -s.wgn_pms(iwgn).latsin * std::tan(sd) / s.wgn_pms(iwgn).latcos;
        float hlat = 0.0f;
        if (sdlat > 1.0f) {
            hlat = 0.0f;
        } else if (sdlat >= -1.0f) {
            hlat = std::acos(sdlat);
        } else {
            hlat = 3.1416f;
        }
        const float daylength = 7.6394f * hlat;
        for (int ipl = 1; ipl <= s.pcom(j).npl; ++ipl) {
            plant_status& cur = s.pcom(j).plcur(ipl);
            cur.idorm = (cur.gro == "y" && daylength - s.dormhr(j) < s.wgn_pms(iwgn).daylmn) ? "y" : "n";
        }
        so.zmx = so.phys(nly).d;
        if (h.hyd.lat_ttime <= 0.0f) {
            float scmx = 0.0f;
            for (int l = 1; l <= nly; ++l)
                if (so.phys(l).k > scmx) scmx = so.phys(l).k;
            float xx = 10.4f * h.topo.lat_len / scmx;
            if (xx < 1.0f) xx = 1.0f;
            h.hyd.lat_ttime = 1.0f - std::exp(-1.0f / xx);
        } else {
            h.hyd.lat_ttime = 1.0f - std::exp(-1.0f / h.hyd.lat_ttime);
        }
        h.lumv.tile_ttime = 0.0f;
        if (h.lumv.ldrain > 0) throw std::runtime_error("SWAT+ tile drains are not supported");
    }
}

}  // namespace

// time_conc_init.f90
void time_conc_init(Sim& s) {
    for (s.iru = 1; s.iru <= s.sp_ob.ru; ++s.iru) {
        const int iru = s.iru;
        s.ru_n(iru) = 0.0f;
        for (int ii = 1; ii <= s.ru_def(iru).num_tot; ++ii) {
            const int ielem = s.ru_def(iru).num(ii);
            if (s.ru_elem(ielem).obtyp == "hru") {
                s.ihru = s.ru_elem(ielem).obtypno;
                s.ru_n(iru) = s.ru_n(iru) + s.hru(s.ihru).luse.ovn * s.hru(s.ihru).km;
            } else {
                s.ru_n(iru) = 0.1f;
            }
        }
    }
    for (int iru = 1; iru <= s.sp_ob.ru; ++iru) {
        const int iob = s.sp_ob1.ru + iru - 1;
        s.ru(iru).da_km2 = s.ob(iob).area_ha / 100.0f;
        s.ru_n(iru) = s.ru_n(iru) / s.ru(iru).da_km2;
        const topography_db& t = s.topo_db(s.ru(iru).dbs.toposub_db);
        const float tov = 0.0556f * fpow(t.slope_len * s.ru_n(iru), 0.6f) / fpow(t.slope + 0.001f, 0.3f);
        const float ch_slope = 0.5f * (t.slope + 0.001f);
        const float ch_n = s.ru_n(iru);
        const float ch_l = s.ru(iru).field.length / 1000.0f;
        const float t_ch = 0.62f * ch_l * fpow(ch_n, 0.75f) / (fpow(s.ru(iru).da_km2, 0.125f) * fpow(ch_slope, 0.375f));
        s.ru_tc(iru) = tov + t_ch;
    }
    for (s.ihru = 1; s.ihru <= s.sp_ob.hru; ++s.ihru) {
        const int j = s.ihru;
        hydrologic_response_unit& h = s.hru(j);
        const topography_db& t = s.topo_db(h.dbs.topo);
        s.t_ov(j) = 0.0556f * fpow(h.topo.slope_len * h.luse.ovn, 0.6f) / fpow(h.topo.slope + 0.0001f, 0.3f);
        const float ch_slope = 0.5f * t.slope;
        const float ch_n = h.luse.ovn;
        const float ch_l = 0.5f * std::sqrt(h.area_ha / 2.0f);
        const float t_ch = 0.31f * ch_l * fpow(ch_n, 0.75f) / (fpow(h.km, 0.125f) * fpow(ch_slope + 0.001f, 0.375f));
        s.tconc(j) = s.t_ov(j) + t_ch;
        if (s.time.step > 1) {
            s.brt(j) = 1.0f - std::exp(-s.bsn_prm.surlag / (s.tconc(j) / (s.time.dtm / 60.0f)));
        } else {
            s.brt(j) = 1.0f - std::exp(-s.bsn_prm.surlag / s.tconc(j));
        }
    }
}

// soil_nutcarb_init.f90 (static carbon); main.f90 calls it for each hru (s.ihru)
void soil_nutcarb_init(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& m = s.soil1(j);
    const int nly = so.nly;
    const int isol_pl = s.hru(j).dbs.soil_plant_init;
    const int isolt = s.sol_plt_ini(isol_pl).nut;
    soiltest_db& t = s.solt_db(isolt);
    for (int ly = 1; ly <= nly; ++ly) m.cbn(ly) = fmax2(0.001f, so.phys(ly).cbn);
    for (int ly = 1; ly <= nly; ++ly) {
        soil_physical_properties& ph = so.phys(ly);
        ph.conv_wt = ph.bd * ph.thick / 100.0f;
        const float wt1 = ph.conv_wt;
        const float dep_frac = std::exp(-t.exp_co * ph.d);
        m.mn(ly).no3 = t.nitrate > 1.e-9f ? t.nitrate * dep_frac : 7.0f * dep_frac;
        m.mn(ly).no3 = m.mn(ly).no3 * wt1;
        m.mp(ly).lab = t.lab_p > 1.e-9f ? t.lab_p * dep_frac : 5.0f * dep_frac;
        m.mp(ly).lab = m.mp(ly).lab * wt1;
        float psp = 0.0f;
        if (s.bsn_cc.sol_p_model == 1) {
            const float solp = m.mp(ly).lab / wt1;
            if (ph.clay > 0.0f) {
                psp = -0.045f * std::log(ph.clay) + (0.001f * solp);
                psp = psp - (0.035f * m.cbn(ly)) + 0.43f;
            }
            if (psp < 0.10f) {
                psp = 0.10f;
            } else if (psp > 0.7f) {
                psp = 0.7f;
            }
        } else {
            psp = s.hru(j).nut.psp;
        }
        m.mp(ly).act = m.mp(ly).lab * (1.0f - psp) / psp;
        if (s.bsn_cc.sol_p_model == 1) {
            const float actp = m.mp(ly).act / wt1;
            const float solp = m.mp(ly).lab / wt1;
            float ssp = 25.044f * fpow(actp + solp, -0.3833f);
            if (ssp > 7.0f) ssp = 7.0f;
            if (ssp < 1.0f) ssp = 1.0f;
            m.mp(ly).sta = ssp * (m.mp(ly).act + m.mp(ly).lab);
        } else {
            m.mp(ly).sta = 4.0f * m.mp(ly).act;
        }
    }
    for (int ly = 1; ly <= nly; ++ly) {
        const soil_physical_properties& ph = so.phys(ly);
        const float tot_mass = 10000.0f * ph.thick * ph.bd;
        m.tot(ly).m = tot_mass * (m.cbn(ly) / 100.0f) / 0.58f;
        m.tot(ly).c = tot_mass * (m.cbn(ly) / 100.0f);
        m.tot(ly).n = m.tot(ly).c / 10.0f;
        m.tot(ly).p = m.tot(ly).c / 100.0f;
        if (t.fr_hum_act < 1.e-9f) t.fr_hum_act = 0.02f;
        const float frac_hum_active = t.fr_hum_act;
        m.hact(ly).m = frac_hum_active * m.tot(ly).m;
        m.hact(ly).c = frac_hum_active * m.tot(ly).c;
        m.hact(ly).n = m.hact(ly).c / t.hum_c_n;
        m.hact(ly).p = m.hact(ly).c / t.hum_c_p;
        m.hsta(ly).m = (1.0f - frac_hum_active) * m.tot(ly).m;
        m.hsta(ly).c = (1.0f - frac_hum_active) * m.tot(ly).c;
        m.hsta(ly).n = m.hsta(ly).c / t.hum_c_n;
        m.hsta(ly).p = m.hsta(ly).c / t.hum_c_p;
        if (s.bmix_depth >= ph.d) {
            so.ly(ly).init_bmix = s.bmix_eff;
        } else if (s.bmix_depth > so.phys(ly - 1).d && s.bmix_depth <= ph.d) {
            so.ly(ly).init_bmix = s.bmix_eff * (s.bmix_depth - so.phys(ly - 1).d) / ph.thick;
        } else {
            so.ly(ly).init_bmix = 0.0f;
        }
        // the sum of the CENTURY pools (zero with the static carbon model)
        m.tot(ly) = m.str(ly) + m.meta(ly) + m.hs(ly) + m.hp(ly) + m.microb(ly);
        m.seq(ly) = m.hs(ly) + m.hp(ly) + m.microb(ly);
    }
}

// proc_hru.f90
void proc_hru(Sim& s) {
    if (s.sp_ob.hru <= 0) return;
    hru_allo(s);
    hru_read(s);
    hrudb_init(s);
    for (int iihru = 1; iihru <= s.sp_ob.hru; ++iihru) {
        s.hru(iihru).land_use_mgt = s.hru(iihru).dbs.land_use_mgt;
        hru_lum_init(s, iihru);
    }
    topohyd_init(s);
    // hru_output_allo (the daily records; the monthly, yearly and average ones are not kept)
    const int mhru = s.sp_ob.hru;
    s.hwb_d.alloc(1, mhru, output_waterbal());
    s.hnb_d.alloc(1, mhru, output_nutbal());
    s.hls_d.alloc(1, mhru, output_losses());
    s.hpw_d.alloc(1, mhru, output_plantweather());
    s.hsc_d.alloc(1, mhru, s.hscz);
    s.hrc_d.alloc(1, mhru, s.hrcz);
    s.hpc_d.alloc(1, mhru, s.hpcz);
    s.hscf_d.alloc(1, mhru, s.hscfz);
    soils_init(s);
    structure_init(s);
    plant_all_init(s);
    for (int j = 1; j <= s.sp_ob.hru; ++j) cn2_init(s, j);
    hydro_init(s);
}

}  // namespace swat
