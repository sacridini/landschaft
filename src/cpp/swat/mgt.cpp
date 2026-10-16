// SPDX-License-Identifier: GPL-3.0-or-later
//
// Management of a HRU: mgt_newtillmix_cswat0.f90 (tillage and biological mixing).

#include "swat/fmath.hpp"
#include "swat/om_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

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

}  // namespace swat
