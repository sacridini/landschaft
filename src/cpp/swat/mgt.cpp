// SPDX-License-Identifier: GPL-3.0-or-later
//
// Management of a HRU: mgt_newtillmix_cswat0.f90 (tillage and biological mixing), the
// scheduled operations (mgt_operatn, mgt_sched and the operations of the port) and the
// land use decision tables (dtbl_lum_read, proc_cond, hru_dtbl_actions_init, conditions,
// actions).

#include <algorithm>
#include <stdexcept>
#include <string>

#include "swat/fmath.hpp"
#include "swat/om_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;
using fio::items;

// mgt_newtillmix_cswat0.f90 (bmix > 0: biological mixing; else the tillage idtill)
void mgt_newtillmix_cswat0(Sim& s, int jj, float bmix, int idtill) {
    soil_profile& so = s.soil(jj);
    soil_profile_mass& sm = s.soil1(jj);
    plant_community_mass& pm = s.pl_mass(jj);
    const int npl = s.pcom(jj).npl;
    const int nly = so.nly;
    organic_mixing_mass& mo = s.mix_org;
    s.mix_mn = s.mnz;
    s.mix_mp = s.mpz;
    for (organic_mass* o : {&mo.tot, &mo.hact, &mo.hsta, &mo.hs, &mo.hp, &mo.microb, &mo.str, &mo.lig, &mo.meta,
                            &mo.man, &mo.water})
        *o = s.orgz;
    float mix_clay = 0.0f, mix_silt = 0.0f, mix_sand = 0.0f, mix_rock = 0.0f, mix_sw = 0.0f, mix_bd = 0.0f;
    float emix = 0.0f, dtil = 0.0f;
    Vec<float> sol_mass, sol_msm, sol_msn, frac_dep;
    sol_mass.alloc(1, nly, 0.0f);
    sol_msm.alloc(1, nly, 0.0f);
    sol_msn.alloc(1, nly, 0.0f);
    frac_dep.alloc(1, nly, 0.0f);
    mo.rsd.alloc(1, npl, s.orgz);
    bool bio_mix_event = false;
    if (bmix > 1.e-6f) {
        emix = bmix;
        dtil = fmin2(so.phys(nly).d, 50.0f);
        bio_mix_event = true;
    } else {
        emix = s.tilldb(idtill).effmix;
        dtil = s.tilldb(idtill).deptil;
        s.tillage_days(jj) = 0;
        s.tillage_depth(jj) = dtil;
        s.tillage_switch(jj) = 1;
    }
    if (dtil <= 0.0f) return;
    for (int l = 1; l <= nly; ++l)
        sol_mass(l) = (so.phys(l).thick / 1000.0f) * 10000.0f * so.phys(1).bd * 1000.0f * (1.0f - so.phys(l).rock / 100.0f);
    if (dtil < 10.0f) dtil = 11.0f;
    for (int l = 1; l <= nly; ++l) {
        soil_physical_properties& ph = so.phys(l);
        if (ph.d <= dtil) {
            sol_msm(l) = emix * sol_mass(l);
            sol_msn(l) = sol_mass(l) - sol_msm(l);
            frac_dep(l) = ph.thick / dtil;
        } else if (ph.d > dtil && l > 1 && so.phys(l - 1).d < dtil) {
            sol_msm(l) = emix * sol_mass(l) * (dtil - so.phys(l - 1).d) / ph.thick;
            sol_msn(l) = sol_mass(l) - sol_msm(l);
            frac_dep(l) = (dtil - so.phys(l - 1).d) / dtil;
        } else {
            sol_msm(l) = 0.0f;
            sol_msn(l) = sol_mass(l);
            frac_dep(l) = 0.0f;
        }
        const float fm = sol_msm(l) / sol_mass(l);
        mix_sw = mix_sw + fm * ph.st;
        mix_bd = mix_bd + fm * ph.bd;
        if (!bio_mix_event) mix_rock = mix_rock + fm * ph.rock * sol_mass(l) / 100.0f;
        mix_sand = mix_sand + fm * ph.sand * sol_mass(l) / 100.0f;
        mix_silt = mix_silt + fm * ph.silt * sol_mass(l) / 100.0f;
        mix_clay = mix_clay + fm * ph.clay * sol_mass(l) / 100.0f;
        s.mix_mn = s.mix_mn + fm * sm.mn(l);
        s.mix_mp = s.mix_mp + fm * sm.mp(l);
        mo.tot = mo.tot + fm * sm.tot(l);
        for (int ipl = 1; ipl <= npl; ++ipl) mo.rsd(ipl) = mo.rsd(ipl) + fm * pm.rsd(ipl);
        mo.hact = mo.hact + fm * sm.hact(l);
        mo.hsta = mo.hsta + fm * sm.hsta(l);
        mo.hs = mo.hs + fm * sm.hs(l);
        mo.hp = mo.hp + fm * sm.hp(l);
        mo.microb = mo.microb + fm * sm.microb(l);
        mo.str = mo.str + fm * sm.str(l);
        mo.lig = mo.lig + fm * sm.lig(l);
        mo.meta = mo.meta + fm * sm.meta(l);
        mo.man = mo.man + fm * sm.man(l);
        mo.water = mo.water + fm * sm.water(l);
    }
    (void)mix_bd;
    for (int l = 1; l <= nly; ++l) {
        soil_physical_properties& ph = so.phys(l);
        const float fn = sol_msn(l) / sol_mass(l);
        const float fd = frac_dep(l);
        sm.mn(l) = fn * sm.mn(l) + fd * s.mix_mn;
        sm.mp(l) = fn * sm.mp(l) + fd * s.mix_mp;
        sm.tot(l) = fn * sm.tot(l) + fd * mo.tot;
        for (int ipl = 1; ipl <= npl; ++ipl) {
            organic_mass& r = sm.pl(ipl).rsd(l);
            r = fn * r + fd * mo.rsd(ipl);
            mo.surf_rsd = emix * fd * pm.rsd(ipl);
            r = r + mo.surf_rsd;
            pm.rsd(ipl) = pm.rsd(ipl) - mo.surf_rsd;
            pm.rsd_tot = pm.rsd_tot - mo.surf_rsd;
        }
        sm.hact(l) = fn * sm.hact(l) + fd * mo.hact;
        sm.hsta(l) = fn * sm.hsta(l) + fd * mo.hsta;
        sm.hs(l) = fn * sm.hs(l) + fd * mo.hs;
        sm.hp(l) = fn * sm.hp(l) + fd * mo.hp;
        sm.microb(l) = fn * sm.microb(l) + fd * mo.microb;
        sm.str(l) = fn * sm.str(l) + fd * mo.str;
        sm.lig(l) = fn * sm.lig(l) + fd * mo.lig;
        sm.meta(l) = fn * sm.meta(l) + fd * mo.meta;
        sm.man(l) = fn * sm.man(l) + fd * mo.man;
        sm.water(l) = fn * sm.water(l) + fd * mo.water;
        ph.clay = 100.0f / sol_mass(l) * (fn * ph.clay * sol_mass(l) / 100.0f + fd * mix_clay);
        ph.silt = 100.0f / sol_mass(l) * (fn * ph.silt * sol_mass(l) / 100.0f + fd * mix_silt);
        ph.sand = 100.0f / sol_mass(l) * (fn * ph.sand * sol_mass(l) / 100.0f + fd * mix_sand);
        if (!bio_mix_event) ph.rock = 100.0f / sol_mass(l) * (fn * ph.rock * sol_mass(l) / 100.0f + fd * mix_rock);
        ph.st = fn * ph.st + fd * mix_sw;
    }
}

namespace {

[[noreturn]] void refuse(const std::string& what) {
    throw std::runtime_error("SWAT+ " + what + " is not supported yet");
}

// plant_module's plg_zero
void plg_zero(plant_growth& g) {
    g.cht = 0.0f;
    g.lai = 0.0f;
    g.plet = 0.0f;
    g.plpet = 0.0f;
    g.laimxfr = 0.0f;
    g.laimxfr_p = 0.0f;
    g.hi_adj = 0.0f;
    g.hi_prev = 0.0f;
    g.olai = 0.0f;
    g.dphu = 0.0f;
    g.d_senes = 0.0f;
    g.leaf_frac = 0.0f;
    g.root_dep = 0.0f;
    g.root_frac = 0.0f;
    g.rtfr.fill(0.0f);
}

// mgt_plantop.f90 (plant s.ipl of HRU s.ihru)
void mgt_plantop(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    pc.plcur(ipl).gro = "y";
    pc.plcur(ipl).idorm = "n";
    pc.plcur(ipl).phuacc = 0.0f;
    pm.tot(ipl) = s.orgz;
    pm.ab_gr(ipl) = s.orgz;
    pm.leaf(ipl) = s.orgz;
    pm.stem(ipl) = s.orgz;
    pm.seed(ipl) = s.orgz;
    pm.root(ipl) = s.orgz;
    pc.plg(ipl).plet = 0.0f;
    pc.plg(ipl).plpet = 0.0f;
    pc.plg(ipl).laimxfr = 0.0f;
    pc.plg(ipl).hi_adj = 0.0f;
    pc.plg(ipl).olai = 0.0f;
    pc.plg(ipl).dphu = 0.0f;
    pm.root(ipl).m = 0.0f;
    pc.plstr(ipl) = s.plstrz;
    const float plt_zmx = 1000.0f * s.pldb(pc.plcur(ipl).idplt).rdmx;
    s.soil(j).zmx = fmin2(s.soil(j).zmx, plt_zmx);
}

// mgt_harvgrain.f90 (no pesticides)
void mgt_harvgrain(Sim& s, int jj, int iplant, int iharvop) {
    const int j = jj;
    s.ipl = iplant;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const int idp = pc.plcur(ipl).idplt;
    const float harveff = s.harvop_db(iharvop).eff;
    pc.plcur(ipl).harv_idx = fmax2(pc.plcur(ipl).harv_idx, s.pldb(idp).wsyf);
    pm.tot(ipl) = pm.tot(ipl) - pm.seed(ipl);
    pm.ab_gr(ipl) = pm.ab_gr(ipl) - pm.seed(ipl);
    s.pl_yield = harveff * pm.seed(ipl);
    s.pl_yield = (1.0f - pc.plcur(ipl).pest_stress) * s.pl_yield;
    s.hpc_d(j).harv_abgr_c = s.hpc_d(j).harv_abgr_c + s.pl_yield.c;
    const float harveff1 = 1.0f - harveff;
    s.soil1(j).hact(1) = harveff1 * pm.seed(ipl) + s.soil1(j).hact(1);
    pm.seed(ipl) = s.plt_mass_z;
}

// mgt_harvresidue.f90
void mgt_harvresidue(Sim& s, int jj, float harveff, int iharvop) {
    const int j = jj;
    plant_community_mass& pm = s.pl_mass(j);
    const float eff = harveff < .00001f ? s.harvop_db(iharvop).eff : harveff;
    const float harv_idx = s.harvop_db(iharvop).hi_ovr;
    const float bm_min = s.harvop_db(iharvop).bm_min;
    const float net_eff = eff * harv_idx;
    s.hrc_d(j).harv_stov_c = 0.0f;
    organic_mass rsd_removed = s.orgz;
    for (int ipl = 1; ipl <= s.pcom(j).npl; ++ipl) {
        rsd_removed = net_eff * pm.rsd(ipl);
        if ((pm.rsd(ipl).m - rsd_removed.m) < bm_min) {
            const float reduction_frac = (pm.rsd(ipl).m - bm_min) / pm.rsd(ipl).m;
            rsd_removed = reduction_frac * pm.rsd(ipl);
        }
        pm.rsd(ipl) = pm.rsd(ipl) - rsd_removed;
        pm.rsd_tot = pm.rsd_tot - rsd_removed;
        if (pm.rsd_tot.m < 1.e-6f) pm.rsd_tot = s.orgz;
        s.hrc_d(j).harv_stov_c = rsd_removed.c;
    }
}

// mgt_killop.f90 (no pesticides)
void mgt_killop(Sim& s, int jj, int iplant) {
    const int j = jj;
    s.ipl = iplant;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    pl_rootfr(s, j);
    pm.rsd(ipl) = pm.rsd(ipl) + pm.ab_gr(ipl);
    pm.rsd_tot = s.orgz;
    for (int npl = 1; npl <= pc.npl; ++npl) pm.rsd_tot = pm.rsd_tot + pm.rsd(npl);
    for (int ly = 1; ly <= s.soil(j).nly; ++ly)
        s.soil1(j).pl(ipl).rsd(ly) = s.soil1(j).pl(ipl).rsd(ly) + pc.plg(ipl).rtfr(ly) * pm.root(ipl);
    pm.tot_com = pm.tot_com - pm.tot(ipl);
    pm.ab_gr_com = pm.ab_gr_com - pm.ab_gr(ipl);
    pm.leaf_com = pm.leaf_com - pm.leaf(ipl);
    pm.stem_com = pm.stem_com - pm.stem(ipl);
    pm.seed_com = pm.seed_com - pm.seed(ipl);
    pm.root_com = pm.root_com - pm.root(ipl);
    s.hrc_d(j).plant_surf_c = s.hrc_d(j).plant_surf_c + pm.ab_gr(ipl).c;
    s.hrc_d(j).plant_root_c = s.hrc_d(j).plant_root_c + pm.root(ipl).c;
    s.hpc_d(j).drop_c = s.hpc_d(j).drop_c + pm.ab_gr(ipl).c + pm.root(ipl).c;
    pm.tot(ipl) = s.plt_mass_z;
    pm.ab_gr(ipl) = s.plt_mass_z;
    pm.leaf(ipl) = s.plt_mass_z;
    pm.stem(ipl) = s.plt_mass_z;
    pm.seed(ipl) = s.plt_mass_z;
    pm.root(ipl) = s.plt_mass_z;
    plg_zero(pc.plg(ipl));
    pc.plm(ipl) = s.plmz;
    pc.plstr(ipl) = s.plstrz;
    pc.plcur(ipl).gro = "n";
    pc.plcur(ipl).idorm = "n";
    pc.plcur(ipl).phuacc = 0.0f;
    pc.plcur(ipl).curyr_mat = 1;
}

// pl_fert.f90 (static carbon; the humus P line reads hsta, as the reference: see IDEAS.md)
void pl_fert(Sim& s, int ifrt, float frt_kg, int fertop) {
    const int j = s.ihru;
    soil_profile_mass& sm = s.soil1(j);
    const fertilizer_db& fd = s.fertdb(ifrt);
    const float rtof = s.man_coef.rtof;
    for (int l = 1; l <= 2; ++l) {
        const float fr_ly = l == 1 ? s.chemapp_db(fertop).surf_frac : 1.0f - s.chemapp_db(fertop).surf_frac;
        sm.mn(l).no3 = sm.mn(l).no3 + fr_ly * frt_kg * (1.0f - fd.fnh3n) * fd.fminn;
        sm.mn(l).nh4 = sm.mn(l).nh4 + fr_ly * frt_kg * fd.fnh3n * fd.fminn;
        sm.mp(l).lab = sm.mp(l).lab + fr_ly * frt_kg * fd.fminp;
        sm.tot(l).n = sm.tot(l).n + rtof * fr_ly * frt_kg * fd.forgn;
        sm.tot(l).p = sm.tot(l).p + rtof * fr_ly * frt_kg * fd.forgp;
        sm.pl(1).rsd(l).n = sm.pl(1).rsd(l).n + rtof * fr_ly * frt_kg * fd.forgn;
        sm.pl(1).rsd(l).p = sm.pl(1).rsd(l).p + rtof * fr_ly * frt_kg * fd.forgp;
        sm.hact(l).n = sm.hact(l).n + (1.0f - rtof) * fr_ly * frt_kg * fd.forgn;
        sm.hact(l).p = sm.hsta(l).p + (1.0f - rtof) * fr_ly * frt_kg * fd.forgp;
    }
    s.fertno3 = frt_kg * fd.fminn * (1.0f - fd.fnh3n);
    s.fertnh3 = frt_kg * (fd.fminn * fd.fnh3n);
    s.fertorgn = frt_kg * fd.forgn;
    s.fertsolp = frt_kg * fd.fminp;
    s.fertorgp = frt_kg * fd.forgp;
    s.fertn = s.fertn + frt_kg * (fd.fminn + fd.forgn);
    s.fertp = s.fertp + frt_kg * (fd.fminp + fd.forgp);
}

// the yields of a harvest (mgt_sched's harv and hvkl)
void sum_yield(Sim& s, int j, int ipl) {
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    pm.yield_tot(ipl) = pm.yield_tot(ipl) + s.pl_yield;
    pm.yield_yr(ipl) = pm.yield_yr(ipl) + s.pl_yield;
    pc.plcur(ipl).harv_num = pc.plcur(ipl).harv_num + 1;
    pc.plcur(ipl).harv_num_yr = pc.plcur(ipl).harv_num_yr + 1;
    const int iplt_bsn = pc.plcur(ipl).bsn_num;
    s.bsn_crop_yld(iplt_bsn).area_ha = s.bsn_crop_yld(iplt_bsn).area_ha + s.hru(j).area_ha;
    s.bsn_crop_yld(iplt_bsn).yield = s.bsn_crop_yld(iplt_bsn).yield + s.pl_yield.m * s.hru(j).area_ha / 1000.0f;
}

void harvest(Sim& s, int j, int iharvop, bool kill) {
    const management_ops& mgt = s.mgt;
    const std::string& typ = s.harvop_db(iharvop).typ;
    const int ipl = s.ipl;
    if (typ == "grain" || typ == "picker" || (kill && typ == "stripper")) {
        mgt_harvgrain(s, j, ipl, iharvop);
    } else if (typ == "residue") {
        mgt_harvresidue(s, j, mgt.op3, iharvop);
    } else if (kill && typ == "tree") {
    } else {
        refuse("harvest type " + typ);
    }
}

// mgt_sched.f90 (the operations of the port; the others are refused when met)
void mgt_sched(Sim& s, int isched) {
    const int j = s.ihru;
    management_ops& mgt = s.mgt;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    int& icom = s.saved.mgt_sched_icom;
    if (mgt.op != "fert") {
        mgt.op2 = 0;
        icom = pc.pcomdb;
        if (icom > 0 && pc.npl > 1) {
            for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
                if (mgt.op_char == s.pcomdb(icom).pl(s.ipl).cpnm) {
                    mgt.op2 = s.ipl;
                    break;
                }
            }
        }
    }
    const std::string& op = mgt.op;
    if (op == "plnt") {
        icom = pc.pcomdb;
        pc.days_plant = 1;
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            const int ipl = s.ipl;
            if (mgt.op_char == s.pcomdb(icom).pl(ipl).cpnm && pc.plcur(ipl).gro == "n") {
                pc.plcur(ipl).gro = "y";
                pc.plcur(ipl).idorm = "n";
                mgt_plantop(s);
                if (mgt.op3 > 1.e-6f)
                    for (int idb = 1; idb <= s.db_mx.transplant; ++idb)
                        if (mgt.op_plant == s.transpl(idb).name) refuse("transplanting");
            }
        }
    } else if (op == "harv") {
        const int iharvop = mgt.op1;
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            const int ipl = s.ipl;
            const bool match = mgt.op_char == s.pcomdb(icom).pl(ipl).cpnm || mgt.op_char == "all";
            if (pc.plcur(ipl).gro == "y") {
                const float biomass = pm.tot(ipl).m;
                if (match) {
                    pc.days_harv = 1;
                    if (biomass > s.harvop_db(iharvop).bm_min) {
                        const std::string& typ = s.harvop_db(iharvop).typ;
                        if (typ == "grain" || typ == "picker") {
                            mgt_harvgrain(s, j, ipl, iharvop);
                        } else if (typ == "residue") {
                            mgt_harvresidue(s, j, mgt.op3, iharvop);
                        } else {
                            refuse("harvest type " + typ);
                        }
                    }
                    if (s.cal_codes.plt == "y") refuse("plant calibration");
                    sum_yield(s, j, ipl);
                }
            } else if (match && s.harvop_db(iharvop).typ == "residue") {
                mgt_harvresidue(s, j, mgt.op3, iharvop);
            }
        }
    } else if (op == "kill") {
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            const int ipl = s.ipl;
            if (pc.plcur(ipl).gro == "y" &&
                (mgt.op_char == s.pcomdb(icom).pl(ipl).cpnm || mgt.op_char == "all")) {
                pc.days_kill = 1;
                mgt_killop(s, j, ipl);
            }
            pc.plcur(ipl).phuacc = 0.0f;
        }
    } else if (op == "hvkl") {
        const int iharvop = mgt.op1;
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            const int ipl = s.ipl;
            if (pc.plcur(ipl).gro == "y") {
                const float biomass = pm.tot(ipl).m;
                if (mgt.op_char == s.pcomdb(icom).pl(ipl).cpnm || mgt.op_char == "all") {
                    pc.days_harv = 1;
                    pc.days_kill = 1;
                    if (biomass > s.harvop_db(iharvop).bm_min) harvest(s, j, iharvop, true);
                    mgt_killop(s, j, ipl);
                    if (s.cal_codes.plt == "y") refuse("plant calibration");
                    sum_yield(s, j, ipl);
                }
                pc.plstr(ipl) = s.plstrz;
                pc.plcur(ipl).phuacc = 0.0f;
                s.phubase(j) = 0.0f;
            } else if (s.harvop_db(iharvop).typ == "residue") {
                mgt_harvresidue(s, j, mgt.op3, iharvop);
            }
        }
    } else if (op == "till") {
        const int idtill = mgt.op1;
        s.ipl = std::max(1, mgt.op2);
        mgt_newtillmix_cswat0(s, j, 0.0f, idtill);
    } else if (op == "irrm") {
        s.ipl = 1;
        const irrigation_operation& io = s.irrop_db(mgt.op1);
        s.irrig(j).applied = io.amt_mm * io.eff * (1.0f - io.surq);
        s.irrig(j).runoff = io.amt_mm * io.surq;
        pc.days_irr = 1;
        s.hru(j).irr_yr = s.hru(j).irr_yr + s.irrig(j).applied;
    } else if (op == "fert") {
        s.ipl = 1;
        if (s.wet(j).flo > 0.0f) refuse("fertilizing standing water");
        pl_fert(s, mgt.op1, mgt.op3, mgt.op4);
    } else if (op == "skip") {
        s.yr_skip(j) = 1;
    } else {
        refuse("management operation " + op);
    }
    hydrologic_response_unit& h = s.hru(j);
    if (mgt.op != "skip") h.cur_op = h.cur_op + 1;
    if (h.cur_op > s.sched(isched).num_ops) h.cur_op = 1;
    mgt = s.sched(isched).mgt_ops(h.cur_op);
}

void cond_real(decision_table& d, int ic, float var_cur, float var_tbl) {
    for (int ialt = 1; ialt <= d.alts; ++ialt) {
        const std::string& a = d.alt(ic, ialt);
        if (a != "-" && d.act_hit(ialt) == "y") {
            if (a == "<" && var_cur >= var_tbl) d.act_hit(ialt) = "n";
            if (a == ">" && var_cur <= var_tbl) d.act_hit(ialt) = "n";
            if (a == "<=" && var_cur > var_tbl) d.act_hit(ialt) = "n";
            if (a == ">=" && var_cur < var_tbl) d.act_hit(ialt) = "n";
            if (a == "=" && var_cur != var_tbl) d.act_hit(ialt) = "n";
            if (a == "/=" && var_cur == var_tbl) d.act_hit(ialt) = "n";
        }
    }
}

void cond_integer(decision_table& d, int ic, int var_cur, int var_tbl) {
    for (int ialt = 1; ialt <= d.alts; ++ialt) {
        const std::string& a = d.alt(ic, ialt);
        if (a != "-" && d.act_hit(ialt) == "y") {
            if (a == "<" && var_cur >= var_tbl) d.act_hit(ialt) = "n";
            if (a == ">" && var_cur <= var_tbl) d.act_hit(ialt) = "n";
            if (a == "<=" && var_cur > var_tbl) d.act_hit(ialt) = "n";
            if (a == ">=" && var_cur < var_tbl) d.act_hit(ialt) = "n";
            if (a == "=" && var_cur != var_tbl) d.act_hit(ialt) = "n";
            if (a == "/=" && var_cur == var_tbl) d.act_hit(ialt) = "n";
        }
    }
}

// conditions.f90 (the HRU conditions of the port)
void conditions(Sim& s, int ob_cur, int idtbl) {
    decision_table& d = *s.d_tbl;
    d.act_hit.fill("y");
    int& ipl_saved = s.saved.conditions_ipl;
    for (int ic = 1; ic <= d.conds; ++ic) {
        const conditions_var& c = d.cond(ic);
        const std::string& v = c.var;
        const int ob_num = c.ob_num == 0 ? ob_cur : c.ob_num;
        auto mean_stress = [&](auto&& field, bool none_is_one) {
            const plant_community& pc = s.pcom(ob_num);
            int pl_sum = 0;
            float strs_sum = 0.0f;
            for (int ipl = 1; ipl <= pc.npl; ++ipl) {
                if (pc.plcur(ipl).gro == "y") {
                    pl_sum = pl_sum + 1;
                    strs_sum = strs_sum + field(pc.plstr(ipl));
                }
            }
            if (pl_sum > 0) {
                strs_sum = strs_sum / static_cast<float>(pl_sum);
            } else if (none_is_one) {
                strs_sum = 1.0f;
            }
            return strs_sum;
        };
        if (v == "w_stress") {
            cond_real(d, ic, mean_stress([](const plant_stress& p) { return p.strsw; }, true), c.lim_const);
        } else if (v == "n_stress") {
            cond_real(d, ic, mean_stress([](const plant_stress& p) { return p.strsn; }, false), c.lim_const);
        } else if (v == "p_stress") {
            cond_real(d, ic, mean_stress([](const plant_stress& p) { return p.strsp; }, false), c.lim_const);
        } else if (v == "phu_plant") {
            const plant_community& pc = s.pcom(ob_num);
            for (int ialt = 1; ialt <= d.alts; ++ialt) {
                if (pc.npl == 0) {
                    d.act_hit(ialt) = "n";
                } else {
                    ipl_saved = 1;
                    for (int iipl = 1; iipl <= pc.npl; ++iipl) {
                        if (pc.plcur(iipl).phuacc > 1.e-6f) {
                            ipl_saved = iipl;
                            break;
                        }
                    }
                }
            }
            cond_real(d, ic, pc.plcur(ipl_saved).phuacc, c.lim_const);
        } else if (v == "plant_gro") {
            ipl_saved = std::max(static_cast<int>(c.lim_const), 1);
            for (int ialt = 1; ialt <= d.alts; ++ialt)
                if (d.alt(ic, ialt) == "=" && s.pcom(ob_num).plcur(ipl_saved).gro == "n") d.act_hit(ialt) = "n";
        } else if (v == "days_plant") {
            cond_integer(d, ic, s.pcom(ob_num).days_plant, static_cast<int>(c.lim_const));
        } else if (v == "days_harv") {
            cond_integer(d, ic, s.pcom(ob_num).days_harv, static_cast<int>(c.lim_const));
        } else if (v == "days_irr") {
            cond_integer(d, ic, s.pcom(ob_num).days_irr, static_cast<int>(c.lim_const));
        } else if (v == "days_act") {
            const int iac = d.con_act(ic);
            cond_integer(d, ic, s.pcom(ob_num).dtbl(idtbl).days_act(iac), static_cast<int>(c.lim_const) + 2);
        } else if (v == "irr_year") {
            cond_real(d, ic, s.hru(ob_num).irr_yr, c.lim_const);
        } else if (v == "slope") {
            cond_real(d, ic, s.hru(ob_num).topo.slope, c.lim_const);
        } else if (v == "soil_water") {
            float targ_val = 0.0f;
            if (c.lim_var == "fc") targ_val = s.soil(ob_num).sumfc;
            if (c.lim_var == "ul") targ_val = s.soil(ob_num).sumul;
            float& targ = s.saved.conditions_targ;
            if (c.lim_op == "*") targ = targ_val * c.lim_const;
            if (c.lim_op == "+") targ = targ_val + c.lim_const;
            if (c.lim_op == "-") targ = targ_val - c.lim_const;
            if (c.lim_op == "/") targ = targ_val / c.lim_const;
            cond_real(d, ic, s.soil(ob_num).sw, targ);
        } else if (v == "jday") {
            cond_integer(d, ic, s.time.day, static_cast<int>(c.lim_const));
        } else if (v == "month") {
            cond_integer(d, ic, s.time.mo, static_cast<int>(c.lim_const));
        } else if (v == "year_rot") {
            cond_integer(d, ic, s.pcom(ob_num).rot_yr, static_cast<int>(c.lim_const));
        } else if (v == "year_gro" || v == "cur_yrs_mat") {
            cond_integer(d, ic, s.pcom(ob_num).plcur(1).curyr_mat, static_cast<int>(c.lim_const));
        } else if (v == "year_cal") {
            cond_integer(d, ic, s.time.yrc, static_cast<int>(c.lim_const));
        } else if (v == "year_seq") {
            cond_integer(d, ic, s.time.yrs, static_cast<int>(c.lim_const));
        } else if (v == "biomass" || v == "ground_cov") {
            cond_real(d, ic, s.pl_mass(ob_num).ab_gr_com.m, c.lim_const);
        } else if (v == "leaf_area") {
            cond_real(d, ic, s.pcom(ob_num).lai_sum, c.lim_const);
        } else if (v == "soil_temp2") {
            cond_real(d, ic, s.soil(ob_num).phys(2).tmp, c.lim_const);
        } else if (v == "land_use") {
            for (int ialt = 1; ialt <= d.alts; ++ialt)
                if (d.alt(ic, ialt) == "=" && s.hru(ob_num).land_use_mgt_c != c.lim_var) d.act_hit(ialt) = "n";
        } else {
            refuse("decision table condition " + v);
        }
    }
}

// actions.f90 (irrigation from an unlimited source: the action of the port)
void actions(Sim& s, int ob_cur, int ob_num, int idtbl) {
    (void)ob_num;
    decision_table& d = *s.d_tbl;
    for (int iac = 1; iac <= d.acts; ++iac) {
        bool action = false;
        for (int ial = 1; ial <= d.alts; ++ial) {
            if (d.act_hit(ial) == "y" && d.act_outcomes(iac, ial) == "y") {
                action = true;
                break;
            }
        }
        if (!action) continue;
        const actions_var& a = d.act(iac);
        if (a.typ == "irrigate") {
            s.ipl = 1;
            const int j = ob_cur;
            auto_operations& at = s.pcom(j).dtbl(idtbl);
            if (at.num_actions(iac) <= static_cast<int>(a.const2)) {
                const irrigation_operation& io = s.irrop_db(d.act_typ(iac));
                if (a.name == "ponding") refuse("paddy irrigation");
                s.irrig(j).applied = a.const_ * io.eff * (1.0f - io.surq);
                s.irrig(j).runoff = a.const_ * io.surq;
                s.irrig(j).demand = a.const_ * s.hru(j).area_ha * 10.0f;
                if (a.ob == "aqu" || a.ob == "cha" || a.ob == "sdc" || a.ob == "res")
                    refuse("irrigation from " + a.ob + " objects");
                s.hru(j).irr_yr = s.hru(j).irr_yr + s.irrig(j).applied;
                at.num_actions(iac) = at.num_actions(iac) + 1;
                at.days_act(iac) = 1;
                s.pcom(j).days_irr = 1;
                if (iac > 1) at.days_act(iac - 1) = 0;
            }
        } else {
            refuse("decision table action " + a.typ);
        }
    }
}

}  // namespace

// dtbl_lum_read.f90
void dtbl_lum_read(Sim& s) {
    const std::string& file = s.in_cond.dtbl_lum;
    if (!s.files.exists(file) || file == "null") {
        s.dtbl_lum.alloc(0, 0, decision_table());
        return;
    }
    fio::File f = s.files.open(file);
    std::string titldum, header;
    int mdtbl = 0;
    if (f.read(chars(titldum, 80)) < 0) return;
    if (f.read(mdtbl) < 0) return;
    if (f.read() < 0) return;
    s.dtbl_lum.alloc(1, mdtbl, decision_table());
    for (int i = 1; i <= mdtbl; ++i) {
        decision_table& d = s.dtbl_lum(i);
        if (f.read(chars(header, 80)) < 0) break;
        if (f.read(chars(d.name, d.name_len), d.conds, d.alts, d.acts) < 0) break;
        d.cond.alloc(1, d.conds, conditions_var());
        d.con_act.alloc(1, d.conds, 0);
        d.alt.alloc(1, d.conds, 1, d.alts, "");
        d.act.alloc(1, d.acts, actions_var());
        d.act_hit.alloc(1, d.alts, "");
        d.act_typ.alloc(1, d.acts, 0);
        d.act_app.alloc(1, d.acts, 0);
        d.lu_chg_mx.alloc(1, d.acts, 0);
        d.snow_chg_mx.alloc(1, d.acts, 0);
        d.act_outcomes.alloc(1, d.acts, 1, d.alts, "");
        if (f.read(chars(header, 80)) < 0) break;
        for (int ic = 1; ic <= d.conds; ++ic) {
            fio::File::Statement st = f.reader();
            st.item(items(d.cond(ic)));
            for (int ial = 1; ial <= d.alts; ++ial) st.item(chars(d.alt(ic, ial), 25));
            if (st.finish() < 0) break;
            if (d.cond(ic).var == "prob_unif") {
                f.backspace();
                if (f.read(chars(d.cond(ic).var, d.cond(ic).var_len), d.frac_app) < 0) break;
            }
        }
        d.hru_lu = 0;
        d.ha_lu = 0.0f;
        for (int ic = 1; ic <= d.conds; ++ic) {
            if (d.cond(ic).var == "land_use") {
                for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
                    if (d.cond(ic).lim_var == s.hru(ihru).land_use_mgt_c) {
                        d.hru_lu = d.hru_lu + 1;
                        d.ha_lu = d.ha_lu + s.hru(ihru).area_ha;
                    }
                }
            }
        }
        if (f.read(chars(header, 80)) < 0) break;
        for (int iac = 1; iac <= d.acts; ++iac) {
            fio::File::Statement st = f.reader();
            st.item(items(d.act(iac)));
            for (int ial = 1; ial <= d.alts; ++ial) st.item(chars(d.act_outcomes(iac, ial), 1));
            const int eof = st.finish();
            d.act(iac).const2 = fmax2(1.0f, d.act(iac).const2);
            if (eof < 0) break;
        }
        for (int iac = 1; iac <= d.acts; ++iac) {
            const actions_var& a = d.act(iac);
            auto find = [](const auto& db, int n, const std::string& name, auto&& key) {
                for (int idb = 1; idb <= n; ++idb)
                    if (key(db(idb)) == name) return idb;
                return 0;
            };
            auto nm = [](const auto& x) -> const std::string& { return x.name; };
            if (a.typ == "plant") {
                if (int k = find(s.transpl, s.db_mx.transplant, a.file_pointer, nm)) d.act_app(iac) = k;
            } else if (a.typ == "harvest" || a.typ == "harvest_kill") {
                if (int k = find(s.harvop_db, s.db_mx.harvop_db, a.file_pointer, nm)) d.act_typ(iac) = k;
            } else if (a.typ == "till") {
                if (int k = find(s.tilldb, s.db_mx.tillparm, a.option,
                                 [](const tillage_db& x) -> const std::string& { return x.tillnm; }))
                    d.act_typ(iac) = k;
            } else if (a.typ == "irr_demand" || a.typ == "irr_wallo" || a.typ == "irrigate") {
                if (int k = find(s.irrop_db, s.db_mx.irrop_db, a.option, nm)) d.act_typ(iac) = k;
            } else if (a.typ == "fertilize" || a.typ == "fert_future") {
                if (int k = find(s.fertdb, s.db_mx.fertparm, a.option,
                                 [](const fertilizer_db& x) -> const std::string& { return x.fertnm; }))
                    d.act_typ(iac) = k;
                if (int k = find(s.chemapp_db, s.db_mx.chemapp_db, a.file_pointer, nm)) d.act_app(iac) = k;
            }
            for (int ic = 1; ic <= d.conds; ++ic)
                if (d.cond(ic).lim_var == a.name) d.con_act(ic) = iac;
        }
    }
    s.db_mx.dtbl_lum = mdtbl;
}

// proc_cond.f90 (the land use and management decision tables of the schedules)
void proc_cond(Sim& s) {
    for (int ihru = 1; ihru <= s.sp_ob.hru; ++ihru) {
        management_schedule& sc = s.sched(s.hru(ihru).mgt_ops);
        if (sc.num_autos > 0) {
            sc.irr = 1;
            for (int iauto = 1; iauto <= sc.num_autos; ++iauto)
                for (int ictl = 1; ictl <= s.db_mx.dtbl_lum; ++ictl)
                    if (sc.auto_name(iauto) == s.dtbl_lum(ictl).name) sc.num_db(iauto) = ictl;
        }
    }
}

// hru_dtbl_actions_init.f90 (the HRU is indexed by its object number, as the reference)
void hru_dtbl_actions_init(Sim& s) {
    for (int iihru = 1; iihru <= s.sp_ob.hru; ++iihru) {
        const int ihru = s.sp_ob1.hru + iihru - 1;
        hydrologic_response_unit& h = s.hru(ihru);
        const int isched = h.mgt_ops;
        int m_autos = s.sched(isched).num_autos;
        if (h.man_trn_dtbl > 0) refuse("manure transfer decision tables");
        if (m_autos <= 0) continue;
        plant_community& pc = s.pcom(ihru);
        pc.dtbl.alloc(1, m_autos, auto_operations());
        int id = 0;
        for (int iauto = 1; iauto <= m_autos; ++iauto) {
            if (iauto != h.irr_trn_iauto && iauto != h.man_trn_iauto) id = s.sched(isched).num_db(iauto);
            const decision_table& d = s.dtbl_lum(id);
            pc.dtbl(iauto).num_actions.alloc(1, d.acts, 1);
            pc.dtbl(iauto).days_act.alloc(1, d.acts, 0);
            for (int iac = 1; iac <= d.acts; ++iac)
                if (d.act(iac).typ == "fert_future") refuse("decision table action fert_future");
        }
    }
}

// mgt_operatn.f90
void mgt_operatn(Sim& s) {
    const int j = s.ihru;
    const int isched = s.hru(j).mgt_ops;
    management_schedule& sc = s.sched(isched);
    if (sc.num_ops < 1) return;
    management_ops& mgt = s.mgt;
    mgt = sc.mgt_ops(s.hru(j).cur_op);
    int iter_guard = 0;
    while (mgt.mon == s.time.mo && mgt.day == s.time.day_mo) {
        mgt_sched(s, isched);
        if (sc.num_ops == 1) break;
        if (s.yr_skip(j) == 1) break;
        iter_guard = iter_guard + 1;
        if (iter_guard >= sc.num_ops) break;
    }
    plant_community& pc = s.pcom(j);
    s.ipl = std::max(mgt.op2, 1);
    float aphu = pc.plcur(s.ipl).gro == "n" ? s.phubase(j) : pc.plcur(s.ipl).phuacc;
    while (mgt.husc > 0.0f && aphu > mgt.husc) {
        mgt_sched(s, isched);
        if (sc.num_ops == 1) break;
        s.ipl = std::max(mgt.op2, 1);
        aphu = pc.plcur(s.ipl).gro == "n" ? s.phubase(j) : pc.plcur(s.ipl).phuacc;
        if (mgt.op == "skip") mgt_sched(s, isched);
        if (s.yr_skip(j) == 1) break;
    }
}

// hru_control's decision tables of the HRU's schedule (before its operations)
void hru_dtbl_control(Sim& s) {
    const int j = s.ihru;
    const int isched = s.hru(j).mgt_ops;
    management_schedule& sc = s.sched(isched);
    if (sc.num_autos <= 0) return;
    plant_community& pc = s.pcom(j);
    const int iob = s.hru(j).obj_no;
    for (int iauto = 1; iauto <= sc.num_autos; ++iauto) {
        const int id = sc.num_db(iauto);
        s.d_tbl = &s.dtbl_lum(id);
        conditions(s, j, iauto);
        actions(s, j, iob, iauto);
        if (s.time.end_yr == 1)
            for (int iac = 1; iac <= s.d_tbl->acts; ++iac) pc.dtbl(iauto).num_actions(iac) = 1;
    }
    if (s.time.end_yr == 1) pc.rot_yr = pc.rot_yr + 1;
    for (int iauto = 1; iauto <= sc.num_autos; ++iauto) {
        const int id = sc.num_db(iauto);
        for (int iac = 1; iac <= s.dtbl_lum(id).acts; ++iac)
            if (pc.dtbl(iauto).days_act(iac) > 0) pc.dtbl(iauto).days_act(iac) = pc.dtbl(iauto).days_act(iac) + 1;
    }
}

}  // namespace swat
