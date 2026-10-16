// SPDX-License-Identifier: GPL-3.0-or-later
//
// One day of a HRU: hru_control.f90 and its subroutines in the port (sim_initday,
// varinit, albedo, stmp_solt, sq_canopyint, sq_snom, et_pot, et_act, surface, sq_dailycn,
// sq_daycn, ero_pkq, ero_eiusle, ero_cfactor, ero_ysed, swr_percmain, swr_percmicro,
// swr_satexcess, rsd_decomp, nut_nminrl, nut_nitvol, nut_pminrl, pl_community,
// pl_waterup, pl_grow, pl_nut_demand, pl_nupd, pl_pupd, pl_dormant, pl_biomass_gro,
// pl_tstr, pl_nup, pl_nfix, pl_pup, nuts, pl_leaf_gro, pl_leaf_senes, pl_mortality,
// pest_enrsb, nut_orgn, nut_psed, nut_nrain, nut_nlch, nut_solp, swr_latsed,
// stor_surfstor, swr_substor, sq_surfst, swr_subwq, hru_hyds).

#include <cmath>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/hyd_ops.hpp"
#include "swat/om_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

namespace {

float fmin_list(std::initializer_list<float> v) {
    auto it = v.begin();
    float r = *it++;
    for (; it != v.end(); ++it) r = fmin2(r, *it);
    return r;
}

// varinit.f90
void varinit(Sim& s) {
    const int j = s.ihru;
    for (int ly = 1; ly <= s.soil(j).nly; ++ly) {
        s.soil(j).ly(ly).prk = 0.0f;
        s.soil(j).ly(ly).flat = 0.0f;
    }
    s.albday = 0.0f;
    s.bioday = 0.0f;
    s.bsprev = 0.0f;
    s.canev = 0.0f;
    s.ep_day = 0.0f;
    s.ep_max = 0.0f;
    s.es_day = 0.0f;
    s.fertn = 0.0f;
    s.fertp = 0.0f;
    s.fixn = 0.0f;
    s.grazn = 0.0f;
    s.grazp = 0.0f;
    s.inflpcp = 0.0f;
    s.lyrtile = 0.0f;
    s.qp_cms = 0.0f;
    s.pet_day = 0.0f;
    s.qday = 0.0f;
    s.qtile = 0.0f;
    s.ls_overq = 0.0f;
    s.latqrunon = 0.0f;
    s.sepday = 0.0f;
    s.snoev = 0.0f;
    s.snofall = 0.0f;
    s.snomlt = 0.0f;
    s.sw_excess = 0.0f;
    s.uno3d.fill(0.0f);
    s.usle = 0.0f;
    s.usle_ei = 0.0f;
    s.vpd = 0.0f;
    s.voltot = 0.0f;
    s.ubnrunoff.fill(0.0f);
    for (int k = 1; k <= s.time.step; ++k) s.hhsedy(j, k) = 0.0f;
    s.ubntss.fill(0.0f);
}

// albedo.f90
void albedo(Sim& s) {
    const int j = s.ihru;
    const float cej = -5.e-5f;
    const float cover = s.pl_mass(j).ab_gr_com.m + s.pl_mass(j).rsd_tot.m;
    const float eaj = std::exp(cej * (cover + 0.1f));
    if (s.hru(j).sno_mm <= 0.5f) {
        s.albday = s.soil(j).ly(1).alb;
        if (s.pcom(j).lai_sum > 0.0f) s.albday = 0.23f * (1.0f - eaj) + s.soil(j).ly(1).alb * eaj;
    } else {
        s.albday = 0.8f;
    }
}

// stmp_solt.f90
void stmp_solt(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    const weather_daily& w = s.w;
    const float tlag = 0.8f;
    float f = so.avbd / (so.avbd + 686.0f * std::exp(-5.63f * so.avbd));
    const float dp = 1000.0f + 2500.0f * f;
    const float ww = 0.356f - 0.144f * so.avbd;
    const float wc = so.sw / (ww * so.phys(so.nly).d);
    const float b = std::log(500.0f / dp);
    f = std::exp(b * fpowi((1.0f - wc) / (1.0f + wc), 2));
    const float dd = f * dp;
    const float cover = s.pl_mass(j).ab_gr_com.m + s.pl_mass(j).rsd_tot.m;
    float bcv = cover / (cover + std::exp(7.563f - 1.297e-4f * cover));
    const float sno = s.hru(j).sno_mm;
    if (sno != 0.0f) {
        float xx = 0.0f;
        if (sno <= 120.0f) {
            xx = sno / (sno + std::exp(6.055f - 0.3002f * sno));
        } else {
            xx = 1.0f;
        }
        bcv = fmax2(xx, bcv);
    }
    so.tmp_srf = 0.0f;
    const float st0 = (w.solrad * (1.0f - s.albday) - 14.0f) / 20.0f;
    const float tbare = w.tave + 0.5f * (w.tmax - w.tmin) * st0;
    const float tcov = bcv * so.phys(2).tmp + (1.0f - bcv) * tbare;
    so.tmp_srf = 0.5f * (tbare + tcov);
    float xx = 0.0f;
    for (int k = 1; k <= so.nly; ++k) {
        float zd = (xx + so.phys(k).d) / 2.0f;
        zd = zd / dd;
        const float df = zd / (zd + std::exp(-0.8669f - 2.0775f * zd));
        so.phys(k).tmp =
            tlag * so.phys(k).tmp + (1.0f - tlag) * (df * (s.wgn_pms(s.iwgen).tmp_an - so.tmp_srf) + so.tmp_srf);
        xx = so.phys(k).d;
    }
}

// sq_canopyint.f90 (daily step)
void sq_canopyint(Sim& s) {
    const int j = s.ihru;
    const plant_community& pc = s.pcom(j);
    if (pc.lai_sum < 0.001f || pc.laimx_sum < 0.001f) return;
    const float canmxl = s.hru(j).hyd.canmx * pc.lai_sum / pc.laimx_sum;
    if (s.precip_eff < canmxl - s.canstor(j)) {
        s.canstor(j) = s.canstor(j) + s.precip_eff;
        s.precip_eff = 0.0f;
    } else {
        s.precip_eff = s.precip_eff - (canmxl - s.canstor(j));
        s.canstor(j) = canmxl;
    }
}

// sq_snom.f90 (the pack temperature starts from 0 every day, as the reference)
void sq_snom(Sim& s) {
    const int j = s.ihru;
    hydrologic_response_unit& h = s.hru(j);
    const weather_daily& w = s.w;
    float snotmp = 0.0f;
    snotmp = snotmp * (1.0f - h.sno.timp) + w.tave * h.sno.timp;
    if (w.tave <= h.sno.falltmp) {
        h.sno_mm = h.sno_mm + s.precip_eff;
        s.snofall = s.precip_eff;
        s.precip_eff = 0.0f;
    }
    if (w.tmax > h.sno.melttmp && h.sno_mm > 0.0f) {
        const float smfac = (h.sno.meltmx + h.sno.meltmn) / 2.0f +
                            std::sin(static_cast<float>(s.time.day - 81) / 58.09f) * (h.sno.meltmx - h.sno.meltmn) / 2.0f;
        s.snomlt = smfac * (((snotmp + w.tmax) / 2.0f) - h.sno.melttmp);
        float snocov = 0.0f;
        if (h.sno_mm < h.sno.covmx) {
            const float rto_sno = h.sno_mm / h.sno.covmx;
            snocov = rto_sno / (rto_sno + std::exp(h.snocov1 - h.snocov2 * rto_sno));
        } else {
            snocov = 1.0f;
        }
        s.snomlt = s.snomlt * snocov;
        if (s.snomlt < 0.0f) s.snomlt = 0.0f;
        if (s.snomlt > h.sno_mm) s.snomlt = h.sno_mm;
        h.sno_mm = h.sno_mm - s.snomlt;
        s.precip_eff = s.precip_eff + s.snomlt;
        if (s.precip_eff < 0.0f) s.precip_eff = 0.0f;
    } else {
        s.snomlt = 0.0f;
    }
}

float ee(float tk) {
    float e = 0.0f;
    if (tk + 237.3f != 0.0f) {
        e = (16.78f * tk - 116.9f) / (tk + 237.3f);
        e = std::exp(e);
    }
    return e;
}

// et_pot.f90
void et_pot(Sim& s) {
    const int j = s.ihru;
    hydrologic_response_unit& h = s.hru(j);
    weather_daily& w = s.w;
    const plant_community& pc = s.pcom(j);
    const float tk = w.tave + 273.15f;
    const float pb = 101.3f - h.topo.elev * (0.01152f - 0.544e-6f * h.topo.elev);
    const float xl = 2.501f - 2.361e-3f * w.tave;
    const float gma = 1.013e-3f * pb / (0.622f * xl);
    const float ea = ee(w.tave);
    const float ed = ea * w.rhum;
    s.vpd = ea - ed;
    const float dlt = 4098.0f * ea / fpowi(w.tave + 237.3f, 2);
    switch (s.bsn_cc.pet) {
        case 0: {
            const float ralb = h.sno_mm <= 0.5f ? w.solrad * (1.0f - 0.23f) : w.solrad * (1.0f - 0.8f);
            const float rbo = -(0.34f - 0.139f * std::sqrt(ed));
            const float rto = w.solradmx < 1.e-4f ? 0.0f : 0.9f * (w.solrad / w.solradmx) + 0.1f;
            const float rout = rbo * rto * 4.9e-9f * fpowi(tk, 4);
            const float rn_pet = ralb + rout;
            const float pet_alpha = 1.28f;
            s.pet_day = pet_alpha * (dlt / (dlt + gma)) * rn_pet / xl;
            s.pet_day = fmax2(0.0f, s.pet_day);
            break;
        }
        case 1: {
            const float ralb = h.sno_mm <= 0.5f ? w.solrad * (1.0f - 0.23f) : w.solrad * (1.0f - 0.8f);
            const float ralb1 = w.solrad * (1.0f - s.albday);
            const float rbo = -(0.34f - 0.139f * std::sqrt(ed));
            float rto = w.solradmx < 1.e-4f ? 0.0f : 0.9f * (w.solrad / w.solradmx) + 0.1f;
            const float rout = rbo * rto * 4.9e-9f * fpowi(tk, 4);
            const float rn = ralb1 + rout;
            const float rn_pet = ralb + rout;
            const float rho = 1710.0f - 6.85f * w.tave;
            if (w.windsp < 0.01f) w.windsp = 0.01f;
            float rv = 114.0f / (w.windsp * fpow(170.0f / 1000.0f, 0.2f));
            float rc = 49.0f / (1.4f - 0.4f * s.co2y(s.time.yrs) / 330.0f);
            s.pet_day = (dlt * rn_pet + gma * rho * s.vpd / rv) / (xl * (dlt + gma * (1.0f + rc / rv)));
            s.pet_day = fmax2(0.0f, s.pet_day);
            int igrocom = 0;
            for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl)
                if (pc.plcur(s.ipl).gro == "y") igrocom = 1;
            if (igrocom <= 0) {
                s.ep_max = 0.0f;
            } else {
                const float zz = pc.cht_mx <= 1.0f ? 170.0f : pc.cht_mx * 100.0f + 100.0f;
                const float uzz = w.windsp * fpow(zz / 1000.0f, 0.2f);
                const float chz = pc.cht_mx < 0.01f ? 1.0f : pc.cht_mx * 100.0f;
                const float zom = chz <= 200.0f ? 0.123f * chz : 0.058f * fpow(chz, 1.19f);
                const float zov = 0.1f * zom;
                const float d = 0.667f * chz;
                rv = std::log((zz - d) / zom) * std::log((zz - d) / zov);
                rv = rv / (fpowi(0.41f, 2) * uzz);
                float gsi_wav = 0.0f;
                float gsi_adj = 0.0f;
                for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
                    const int idp = pc.plcur(s.ipl).idplt;
                    rto = pc.plg(s.ipl).lai / (pc.lai_sum + 0.01f);
                    const float xx = s.vpd - 1.0f;
                    const float fvpd = xx > 0.0f ? fmax2(0.1f, 1.0f - s.plcp(idp).vpd2 * xx) : 1.0f;
                    gsi_adj = s.pldb(idp).gsi * fvpd;
                    gsi_wav = gsi_wav + gsi_adj * rto;
                }
                // the reference uses the last plant's gsi_adj, not the weighted gsi_wav
                rc = 1.0f / (gsi_adj + 0.01f);
                rc = rc / (0.5f * (pc.lai_sum + 0.01f) * (1.4f - 0.4f * s.co2y(s.time.yrs) / 330.0f));
                s.ep_max = (dlt * rn + gma * rho * s.vpd / rv) / (xl * (dlt + gma * (1.0f + rc / rv)));
                if (s.ep_max < 0.0f) s.ep_max = 0.0f;
                s.ep_max = fmin2(s.ep_max, s.pet_day);
            }
            break;
        }
        case 2: {
            const float ramm = w.solradmx * 37.59f / 30.0f;
            if (w.tmax > w.tmin) {
                s.pet_day = 0.0023f * (ramm / xl) * (w.tave + 17.8f) * fpow(w.tmax - w.tmin, 0.5f);
                s.pet_day = fmax2(0.0f, s.pet_day);
            } else {
                s.pet_day = 0.0f;
            }
            break;
        }
        default:
            throw std::runtime_error("SWAT+ pet method " + std::to_string(s.bsn_cc.pet) + " is not supported");
    }
    s.pet_day = h.hyd.pet_co * s.pet_day;
}

// et_act.f90 (no wetlands)
void et_act(Sim& s) {
    const int j = s.ihru;
    hydrologic_response_unit& h = s.hru(j);
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    float pet = s.pet_day;
    const float esd = 500.0f;
    const float etco = 0.80f;
    const float effnup = 0.05f;
    float es_max = 0.0f;
    s.canev = 0.0f;
    pet = pet - s.canstor(j);
    if (pet < 0.0f) {
        s.canstor(j) = -pet;
        s.canev = s.pet_day;
        pet = 0.0f;
        s.ep_max = 0.0f;
        es_max = 0.0f;
    } else {
        s.canev = s.canstor(j);
        s.canstor(j) = 0.0f;
    }
    if (pet > 1.0e-6f) {
        if (s.pcom(j).lai_sum <= 3.0f) {
            s.ep_max = s.pcom(j).lai_sum * pet / 3.0f;
        } else {
            s.ep_max = pet;
        }
        if (s.ep_max < 0.0f) s.ep_max = 0.0f;
        const float cej = -5.e-5f;
        float eaj = 0.0f;
        const float cover = s.pl_mass(j).ab_gr_com.m + s.pl_mass(j).rsd_tot.m;
        if (h.sno_mm >= 0.5f) {
            eaj = 0.5f;
        } else {
            eaj = std::exp(cej * (cover + 0.1f));
        }
        es_max = pet * eaj;
        float eos1 = pet / (es_max + s.ep_max + 1.e-10f);
        eos1 = es_max * eos1;
        es_max = fmin2(es_max, eos1);
        es_max = fmax2(es_max, 0.0f);
        if (s.pet_day < es_max + s.ep_max) {
            es_max = s.pet_day - s.ep_max;
            if (pet < es_max + s.ep_max) {
                es_max = pet * es_max / (es_max + s.ep_max);
                s.ep_max = pet * s.ep_max / (es_max + s.ep_max);
            }
        }
        float esleft = es_max;
        if (s.w.tave > 0.0f) {
            if (h.sno_mm >= esleft) {
                h.sno_mm = h.sno_mm - esleft;
                s.snoev = s.snoev + esleft;
                esleft = 0.0f;
            } else {
                esleft = esleft - h.sno_mm;
                s.snoev = s.snoev + h.sno_mm;
                h.sno_mm = 0.0f;
            }
        }
        float evzp = 0.0f;
        const float eosl = esleft;
        float sev = s.saved.et_act_sev;
        for (int ly = 1; ly <= so.nly; ++ly) {
            soil_physical_properties& ph = so.phys(ly);
            const float dep = ly == 1 ? so.phys(1).d : so.phys(ly - 1).d;
            if (dep < esd) {
                const float evz = eosl * ph.d / (ph.d + std::exp(2.374f - 0.00713f * ph.d));
                sev = evz - evzp * (1.0f - h.hyd.esco);
                evzp = evz;
                if (ph.st < ph.fc) {
                    const float xx = 2.5f * (ph.st - ph.fc) / ph.fc;
                    sev = sev * std::exp(xx);
                }
                sev = fmin2(sev, ph.st * etco);
                if (sev < 0.0f) sev = 0.0f;
                if (sev > esleft) sev = esleft;
                if (ph.st > sev) {
                    esleft = fmax2(0.0f, esleft - sev);
                    ph.st = fmax2(1.e-6f, ph.st - sev);
                } else {
                    esleft = fmax2(0.0f, esleft - ph.st);
                    sev = ph.st;
                    ph.st = 0.0f;
                }
            }
            if (ly == 2) {
                float sev_st = so.phys(2).st > 1.e-3f ? sev / (so.phys(2).st) : 0.0f;
                sev_st = fmin2(1.0f, sev_st);
                float no3up = effnup * sev_st * sm.mn(2).no3;
                no3up = fmin2(no3up, sm.mn(2).no3);
                sm.mn(2).no3 = fmax2(0.0001f, sm.mn(2).no3 - no3up);
                sm.mn(1).no3 = fmax2(0.0001f, sm.mn(1).no3 + no3up);
            }
        }
        s.saved.et_act_sev = sev;
        s.es_day = fmax2(0.0f, es_max - esleft);
    }
    so.sw = 0.0f;
    for (int ly = 1; ly <= so.nly; ++ly) so.sw = so.sw + so.phys(ly).st;
}

// sq_dailycn.f90
void sq_dailycn(Sim& s) {
    const int j = s.ihru;
    const soil_profile& so = s.soil(j);
    float sw_fac = s.wrt(1, j) - s.wrt(2, j) * so.sw;
    if (sw_fac < -20.0f) sw_fac = -20.0f;
    if (sw_fac > 20.0f) sw_fac = 20.0f;
    float r2 = 0.0f;
    if ((so.sw + std::exp(sw_fac)) > 0.001f) {
        r2 = s.smx(j) * (1.0f - so.sw / (so.sw + std::exp(sw_fac)));
    } else {
        r2 = s.smx(j);
    }
    if (so.phys(2).tmp <= 0.0f) r2 = s.smx(j) * (1.0f - std::exp(-s.bsn_prm.cn_froz * r2));
    r2 = fmax2(3.0f, r2);
    s.cnday(j) = 25400.0f / (r2 + 254.0f);
}

// sq_daycn.f90 (urban HRUs: the connected impervious fraction with CN 98)
void sq_daycn(Sim& s) {
    const int j = s.ihru;
    float r2 = 25400.0f / s.cnday(j) - 254.0f;
    float bb = 0.2f * r2;
    float pb = s.precip_eff - bb;
    s.surfq(j) = 0.0f;
    if (pb > 0.0f) s.surfq(j) = pb * pb / (s.precip_eff + 0.8f * r2);
    if (s.hru(j).luse.urb_lu > 0) {
        float surfqimp = 0.0f;
        const float cnimp = 98.0f;
        r2 = 25400.0f / cnimp - 254.0f;
        bb = 0.2f * r2;
        pb = s.precip_eff - bb;
        if (pb > 0.0f) surfqimp = pb * pb / (s.precip_eff + 0.8f * r2);
        const urban_db& u = s.urbdb(s.hru(j).luse.urb_lu);
        s.surfq(j) = s.surfq(j) * (1.0f - u.fcimp) + surfqimp * u.fcimp;
    }
}

// regres.f90: the USGS regression loads of urban runoff (kg)
float regres(const Sim& s, int k) {
    static const float bcod[15] = {407.0f, 0.626f, 0.710f, 0.379f, 1.518f, 151.0f, 0.823f, 0.726f,
                                   0.564f, 1.451f, 102.0f, 0.851f, 0.601f, 0.528f, 1.978f};
    static const float bsus[15] = {1778.0f, 0.867f, 0.728f, 0.157f, 2.367f, 812.0f, 1.236f, 0.436f,
                                   0.202f,  1.938f, 97.7f,  1.002f, 1.009f, 0.837f, 2.818f};
    static const float btn[15] = {20.2f, 0.825f, 1.070f, 0.479f, 1.258f, 4.04f, 0.936f, 0.937f,
                                  0.692f, 1.373f, 1.66f,  0.703f, 0.465f, 0.521f, 1.845f};
    static const float btp[15] = {1.725f, 0.884f, 0.826f, 0.467f, 2.130f, 0.697f, 1.008f, 0.628f,
                                  0.469f, 1.790f, 1.618f, 0.954f, 0.789f, 0.289f, 2.247f};
    const int j = s.ihru;
    const int ii = s.wgn_pms(s.iwgen).ireg;
    const float* beta = k == 1 ? bcod : k == 2 ? bsus : k == 3 ? btn : btp;
    auto b = [&](int i) { return beta[(ii - 1) * 5 + (i - 1)]; };  // beta(i, ii), column major
    const float fimp = s.urbdb(s.hru(j).luse.urb_lu).fimp;
    float r = b(1) * fpow(s.w.precip / 25.4f, b(2)) * fpow(s.hru(j).km * fimp / 2.589f, b(3)) *
              fpow(fimp * 100.0f + 1.0f, b(4)) * b(5);
    r = r / 2.205f;
    return r;
}

// hru_urban.f90 (daily step)
void hru_urban(Sim& s) {
    const int j = s.ihru;
    const hydrologic_response_unit& h = s.hru(j);
    const urban_db& u = s.urbdb(h.luse.urb_lu);
    s.iwst = s.ob(h.obj_no).wst;
    const float f = u.fimp;
    if (h.luse.urb_ro == "usgs_reg") {
        if (s.w.precip > 0.1f && s.surfq(j) > 0.1f) {
            const float cod = regres(s, 1);
            (void)cod;
            const float sus_sol = regres(s, 2);
            const float tn = regres(s, 3);
            const float tp = regres(s, 4);
            s.sedyld(j) = (0.001f * sus_sol) * f + s.sedyld(j) * (1.0f - f);
            s.silyld(j) = (0.001f * sus_sol) * f + s.silyld(j) * (1.0f - f);
            s.sanyld(j) = s.sanyld(j) * (1.0f - f);
            s.clayld(j) = s.clayld(j) * (1.0f - f);
            s.sagyld(j) = s.sagyld(j) * (1.0f - f);
            s.lagyld(j) = s.lagyld(j) * (1.0f - f);
            s.sedorgn(j) = (0.7f * tn / (h.km * 100.0f)) * f + s.sedorgn(j) * (1.0f - f);
            s.surqno3(j) = (0.3f * tn / (h.km * 100.0f)) * f + s.surqno3(j) * (1.0f - f);
            s.sedorgp(j) = (0.75f * tp / (h.km * 100.0f)) * f + s.sedorgp(j) * (1.0f - f);
            s.surqsolp(j) = 0.25f * tp / (h.km * 100.0f) * f + s.surqsolp(j) * (1.0f - f);
        }
    } else if (h.luse.urb_ro == "buildup_washoff") {
        if (s.surfq(j) > 0.1f) {
            float dirt = u.dirtmx * s.twash(j) / (u.thalf + s.twash(j));
            const float dirto = dirt;
            const float urbk = u.urbcoef * (s.qp_cms * 3.6f / h.km);
            const float rp1 = -2.0f * std::log(1.0f - s.wst(s.iwst).weat.precip_half_hr);
            const float durf = 4.605f / rp1;
            float turo = durf + s.tconc(j);
            if (turo > 24.0f) turo = 24.0f;
            float xx = urbk * turo;
            if (xx > 24.0f) xx = 24.0f;
            dirt = dirt * std::exp(-xx);
            if (dirt < 1.e-6f) dirt = 0.0f;
            s.twash(j) = u.thalf * dirt / (u.dirtmx - dirt);
            const float sus_sol = fmax2(0.0f, (dirto - dirt) * u.curbden);
            const float tn = u.tnconc * sus_sol / 1.e6f;
            const float tp = u.tpconc * sus_sol / 1.e6f;
            const float tno3 = u.tno3conc * sus_sol / 1.e6f;
            s.sedyld(j) = (0.001f * sus_sol) * f + s.sedyld(j) * (1.0f - f);
            s.silyld(j) = (0.001f * sus_sol * h.area_ha) * f + s.silyld(j) * (1.0f - f);
            s.sanyld(j) = s.sanyld(j) * (1.0f - f);
            s.clayld(j) = s.clayld(j) * (1.0f - f);
            s.sagyld(j) = s.sagyld(j) * (1.0f - f);
            s.lagyld(j) = s.lagyld(j) * (1.0f - f);
            s.surqno3(j) = tno3 * f + s.surqno3(j) * (1.0f - f);
            s.sedorgn(j) = (tn - tno3) * f + s.sedorgn(j) * (1.0f - f);
            s.sedorgp(j) = 0.75f * tp * f + s.sedorgp(j) * (1.0f - f);
            s.surqsolp(j) = 0.25f * tp * f + s.surqsolp(j) * (1.0f - f);
        } else {
            s.twash(j) = s.twash(j) + 1.0f;
        }
    }
}

// ero_pkq.f90
void ero_pkq(Sim& s) {
    const int j = s.ihru;
    const hydrologic_response_unit& h = s.hru(j);
    s.iwst = s.ob(h.obj_no).wst;
    if (s.bsn_cc.sed_det == 1) {
        const float xx = (2.0f * s.tconc(j) * std::log(1.0f - s.wst(s.iwst).weat.precip_half_hr));
        const float altc = 1.0f - std::exp(xx);
        s.qp_cms = altc * s.qday / s.tconc(j);
        s.qp_cms = s.qp_cms * h.km / 3.6f;
    } else {
        const float qp_cfs = s.bsn_prm.prf / 6578.6f * h.area_ha * s.qday / s.tconc(j);
        s.qp_cms = qp_cfs / 35.3f;
    }
}

// ero_eiusle.f90
void ero_eiusle(Sim& s) {
    const int j = s.ihru;
    s.iwst = s.ob(s.hru(j).obj_no).wst;
    const weather_daily& w = s.w;
    if (w.precip > 1.e-4f) {
        const float phh = s.wst(s.iwst).weat.precip_half_hr;
        const float xb = -2.0f * std::log(1.0f - phh);
        const float pkrf30 = 2.0f * w.precip * phh;
        const float pkrf = xb * w.precip;
        s.usle_ei = w.precip * (12.1f + 8.9f * (std::log10(pkrf) - 0.4343f)) * pkrf30 / 1000.0f;
        if (s.usle_ei < 1.e-4f) s.usle_ei = 0.0f;
        s.usle_eifac(j) = s.usle_ei;
    }
}

// ero_cfactor.f90 (the reference always uses the residue and cover method)
void ero_cfactor(Sim& s) {
    const int j = s.ihru;
    // SWAT+ 62 always takes the residue and biomass cover of APEX (codes.bsn c_fact is not
    // used); RunOptions::c_factor_plant gives its old method, the plant's minimum C
    s.bsn_cc.cfac = s.c_factor_plant ? 0 : 1;
    const plant_community_mass& pm = s.pl_mass(j);
    if (s.bsn_cc.cfac == 0) {
        const float cover = pm.ab_gr_com.m + pm.rsd_tot.m;
        float c = 0.0f;
        if (s.pcom(j).npl > 0) {
            c = exp_w((-0.2231f - s.cvm_com(j)) * std::exp(-0.00115f * cover) + s.cvm_com(j));
        } else if (cover > 1.e-4f) {
            c = exp_w(-0.2231f * exp_w(-0.00115f * cover));
        } else {
            c = 0.8f;
        }
        s.usle_cfac(s.ihru) = c;
        return;
    }
    const float rsd_sumfac = (pm.rsd_tot.m + 1.0f) / 1000.0f;
    const float rsd_covfact = exp_w(-s.bsn_prm.rsd_covco * rsd_sumfac);
    const float ab_gr_t = pm.ab_gr_com.m / 1000.0f;
    const float grcov_frac = ab_gr_t / (ab_gr_t + exp_w(1.175f - 1.748f * ab_gr_t));
    float bio_covfact = 1.0f - grcov_frac * exp_w(-0.328f * s.pcom(j).cht_mx);
    bio_covfact = fmax2(1.e-10f, bio_covfact);
    bio_covfact = fmin2(1.0f, bio_covfact);
    const float c = fmax2(1.e-10f, rsd_covfact * bio_covfact);
    s.usle_cfac(s.ihru) = c;
}

// ero_ysed.f90 (the snow test of the reference never reduces the yield: see IDEAS.md)
void ero_ysed(Sim& s) {
    const int j = s.ihru;
    const hydrologic_response_unit& h = s.hru(j);
    s.cklsp(j) = s.usle_cfac(j) * h.lumv.usle_mult;
    s.sedyld(j) = fpow(10.0f * s.surfq(j) * s.qp_cms * h.area_ha, 0.56f) * s.cklsp(j);
    if (s.sedyld(j) < 0.0f) s.sedyld(j) = 0.0f;
    if (h.sno_mm > 0.0f) {
        if (s.sedyld(j) < 1.e-6f) s.sedyld(j) = 0.0f;
    } else if (h.sno_mm > 100.0f) {
        s.sedyld(j) = 0.0f;
    } else {
        s.sedyld(j) = s.sedyld(j) / std::exp(h.sno_mm * 3.0f / 25.4f);
    }
    s.usle = 1.292f * s.usle_ei * s.cklsp(j) / 11.8f;
}

// surface.f90
void surface(Sim& s) {
    const int j = s.ihru;
    sq_dailycn(s);
    if (s.precip_eff > 0.1f) {
        if (s.bsn_cc.gampt != 0) throw std::runtime_error("SWAT+ Green-Ampt infiltration is not supported");
        sq_daycn(s);
    }
    s.surfq(j) = s.surfq(j) + s.irrig(j).runoff;
    s.qday = s.surfq(j);
    if (s.qday > 1.e-6f) ero_pkq(s);
    if (s.qday > 1.e-6f && s.qp_cms > 1.e-6f) ero_eiusle(s);  // ero_ovrsed: sub-daily only
    if (s.surfq(j) > 1.e-6f && s.qp_cms > 1.e-6f) {
        ero_cfactor(s);
        ero_ysed(s);
    }
    if (s.qday < 0.0f) s.qday = 0.0f;
}

// swr_percmicro.f90 (no septic systems)
void swr_percmicro(Sim& s, int ly1) {
    const int j = s.ihru;
    soil_physical_properties& ph = s.soil(j).phys(ly1);
    const hydrologic_response_unit& h = s.hru(j);
    if (ph.tmp <= 0.0f) {
        s.sepday = 0.0f;
        return;
    }
    float ho = 0.0f;
    if (ph.ul - ph.fc <= 0.0f) {
        ho = 0.0f;
    } else {
        ho = 2.0f * s.sw_excess / ((ph.ul - ph.fc) / ph.thick);
    }
    if (ly1 == 1) {
        s.latlyr = 0.0f;
    } else {
        s.latlyr = h.hyd.latq_co * ho * ph.k * h.topo.slope / h.topo.lat_len * 0.024f;
    }
    if (s.latlyr < 0.0f) s.latlyr = 0.0f;
    if (s.latlyr > s.sw_excess) s.latlyr = s.sw_excess;
    ph.hk = (ph.ul - ph.fc) / ph.k;
    ph.hk = fmax2(2.0f, ph.hk);
    s.sepday = (ph.st - ph.fc) * (1.0f - std::exp(-24.0f / ph.hk));
    s.sepday = fmax2(0.0f, s.sepday);
    if (ly1 == s.soil(j).nly) s.sepday = s.sepday * h.hyd.perco_lim;
    if (s.sepday + s.latlyr > s.sw_excess) {
        const float ratio = s.sepday / (s.latlyr + s.sepday);
        s.sepday = s.sw_excess * ratio;
        s.latlyr = s.sw_excess * (1.0f - ratio);
    }
    if (s.sepday + s.lyrtile > s.sw_excess) s.sepday = s.sw_excess - s.lyrtile;
}

// swr_satexcess.f90 (no wetlands)
void swr_satexcess(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    const int nn = so.nly;
    for (int ly = nn; ly >= 1; --ly) {
        float ul_excess = 0.0f;
        if (ly > 1) {
            if (so.phys(ly).st > so.phys(ly).ul) {
                ul_excess = so.phys(ly).st - so.phys(ly).ul;
                so.phys(ly).st = so.phys(ly).ul;
                so.phys(ly - 1).st = so.phys(ly - 1).st + ul_excess;
            } else {
                ul_excess = 0.0f;
            }
        } else {
            ul_excess = so.phys(1).st - so.phys(1).ul;
            if (ul_excess > 0.0f) {
                so.phys(1).st = so.phys(1).ul;
                for (int ly1 = 2; ly1 <= nn; ++ly1) {
                    so.phys(ly1).st = so.phys(ly1).st + ul_excess;
                    if (so.phys(ly1).st > so.phys(ly1).ul) {
                        ul_excess = so.phys(ly1).st - so.phys(ly1).ul;
                        so.phys(ly1).st = so.phys(ly1).ul;
                    } else {
                        ul_excess = 0.0f;
                        break;
                    }
                }
            }
            if (ul_excess > 0.0f) s.surfq(j) = s.surfq(j) + ul_excess;
        }
    }
}

// swr_percmain.f90 (no cracks, septic systems, tiles or gwflow)
void swr_percmain(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    hydrologic_response_unit& h = s.hru(j);
    s.sepday = s.inflpcp + s.irrig(j).applied + s.ht1.flo / (h.area_ha * 10.0f);
    const float slug = 1000.0f;
    float sep_left = s.sepday;
    for (;;) {
        s.sepday = fmin2(sep_left, slug);
        sep_left = sep_left - s.sepday;
        sep_left = fmax2(0.0f, sep_left);
        for (int j1 = 1; j1 <= so.nly; ++j1) {
            so.phys(j1).st = so.phys(j1).st + s.sepday;
            s.sw_excess = so.phys(j1).st - so.phys(j1).fc;
            s.sepday = 0.0f;
            s.latlyr = 0.0f;
            s.lyrtile = 0.0f;
            if (s.sw_excess > 1.e-5f) {
                swr_percmicro(s, j1);
                so.phys(j1).st = so.phys(j1).st - s.sepday - s.latlyr - s.lyrtile;
                so.phys(j1).st = fmax2(1.e-6f, so.phys(j1).st);
            }
            if (j1 == so.nly) s.sepbtm(j) = s.sepbtm(j) + s.sepday;
            s.latq(j) = s.latq(j) + s.latlyr;
            s.qtile = s.qtile + s.lyrtile;
            so.ly(j1).flat = s.latlyr + s.lyrtile;
            so.ly(j1).prk = so.ly(j1).prk + s.sepday;
            if (s.latq(j) < 1.e-6f) s.latq(j) = 0.0f;
            if (s.qtile < 1.e-6f) s.qtile = 0.0f;
            if (so.ly(j1).flat < 1.e-6f) so.ly(j1).flat = 0.0f;
        }
        if (sep_left <= 0.0f) break;
    }
    swr_satexcess(s);
    so.sw = 0.0f;
    for (int j1 = 1; j1 <= so.nly; ++j1) so.sw = so.sw + so.phys(j1).st;
    s.qtile = 0.0f;
    s.wt_shall = so.zmx;
    if (so.phys(2).tmp > 0.0f) {
        const float por_air = 0.9f;
        if (s.bsn_cc.wtdn == 0) {
            if (so.sw > so.sumfc) {
                float yy = so.sumul * por_air;
                if (yy < 1.1f * so.sumfc) yy = 1.1f * so.sumfc;
                float xx = (so.sw - so.sumfc) / (yy - so.sumfc);
                if (xx > 1.0f) xx = 1.0f;
                s.wt_shall = xx * so.zmx;
            }
        } else {
            for (int j1 = 1; j1 <= so.nly; ++j1) {
                if (so.wat_tbl < so.phys(j1).d) {
                    const float sw_del = so.swpwt - so.sw;
                    const float wt_del = sw_del * so.ly(j1).vwt;
                    so.wat_tbl = so.wat_tbl + wt_del;
                    if (so.wat_tbl > so.zmx) so.wat_tbl = so.zmx;
                    s.wt_shall = so.zmx - so.wat_tbl;
                    so.swpwt = so.sw;
                    break;
                }
            }
        }
    }
    so.sw = 0.0f;
    for (int j1 = 1; j1 <= so.nly; ++j1) so.sw = so.sw + so.phys(j1).st;
}

// rsd_decomp.f90
void rsd_decomp(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    plant_community_mass& pm = s.pl_mass(j);
    s.hnb_d(j).rsd_nitorg_n = 0.0f;
    s.hnb_d(j).rsd_laborg_p = 0.0f;
    for (s.ipl = 1; s.ipl <= s.pcom(j).npl; ++s.ipl) {
        const int ipl = s.ipl;
        if (so.phys(1).tmp > 0.0f) {
            organic_mass& rsd = pm.rsd(ipl);
            float cnrf = 1.0f, cprf = 1.0f;
            if (rsd.n > 1.e-4f) {
                float cnr = rsd.c / rsd.n;
                if (cnr > 500.0f) cnr = 500.0f;
                cnrf = std::exp(-0.693f * (cnr - 25.0f) / 25.0f);
            }
            if (rsd.p > 1.e-4f) {
                float cpr = rsd.c / rsd.p;
                if (cpr > 5000.0f) cpr = 5000.0f;
                cprf = std::exp(-0.693f * (cpr - 200.0f) / 200.0f);
            }
            if (so.phys(1).st < 0.0f) so.phys(1).st = 0.0000001f;
            float sut = 0.1f + 0.9f * std::sqrt(so.phys(1).st / so.phys(1).fc);
            sut = fmax2(0.05f, sut);
            float xx = so.phys(1).tmp;
            float cdg = 0.9f * xx / (xx + std::exp(9.93f - 0.312f * xx)) + 0.1f;
            cdg = fmax2(0.1f, cdg);
            xx = cdg * sut;
            if (xx < 0.0f) xx = 0.0f;
            if (xx > 1.e6f) xx = 1.e6f;
            const float csf = std::sqrt(xx);
            const float ca = fmin3(cnrf, cprf, 1.0f);
            const int idp = s.pcom(j).plcur(ipl).idplt;
            float decr = s.pldb(idp).rsdco_pl * ca * csf;
            decr = fmax2(s.bsn_prm.decr_min, decr);
            decr = fmin2(decr, 1.0f);
            s.decomp = decr * rsd;
            rsd = rsd - s.decomp;
            sm.mn(1).no3 = sm.mn(1).no3 + 0.8f * s.decomp.n;
            sm.hact(1).n = sm.hact(1).n + 0.2f * s.decomp.n;
            sm.mp(1).lab = sm.mp(1).lab + 0.8f * s.decomp.p;
            sm.hact(1).p = sm.hact(1).p + 0.2f * s.decomp.p;
            s.hrc_d(j).rsd_surfdecay_c = s.hrc_d(j).rsd_surfdecay_c + 0.42f * s.decomp.c;
            s.hnb_d(j).rsd_nitorg_n = s.hnb_d(j).rsd_nitorg_n + 0.8f * s.decomp.n;
            s.hnb_d(j).rsd_laborg_p = s.hnb_d(j).rsd_laborg_p + 0.8f * s.decomp.p;
        }
    }
    pm.rsd_tot = s.plt_mass_z;
    for (s.ipl = 1; s.ipl <= s.pcom(j).npl; ++s.ipl) pm.rsd_tot = pm.rsd_tot + pm.rsd(s.ipl);
}

// nut_nminrl.f90 (no septic systems)
void nut_nminrl(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    output_nutbal& nb = s.hnb_d(j);
    nb.act_nit_n = 0.0f;
    nb.org_lab_p = 0.0f;
    nb.act_sta_n = 0.0f;
    nb.denit = 0.0f;
    nb.rsd_nitorg_n = 0.0f;
    nb.rsd_laborg_p = 0.0f;
    for (int k = 1; k <= so.nly; ++k) {
        const int kk = k == 1 ? 2 : k;
        if (so.phys(kk).tmp > 0.0f) {
            float sut = 0.1f + 0.9f * std::sqrt(so.phys(kk).st / so.phys(kk).fc);
            sut = fmax2(0.05f, sut);
            float xx = so.phys(kk).tmp;
            float cdg = 0.9f * xx / (xx + std::exp(9.93f - 0.312f * xx)) + 0.1f;
            cdg = fmax2(0.1f, cdg);
            xx = cdg * sut;
            if (xx < 0.0f) xx = 0.0f;
            if (xx > 1.e6f) xx = 1.e6f;
            const float csf = std::sqrt(xx);
            float rwn = 0.1e-4f * ((sm.hact(k).n * (1.0f / s.n_act_frac - 1.0f) - sm.hsta(k).n));
            if (rwn > 0.0f) {
                rwn = fmin2(rwn, sm.hact(k).n);
            } else {
                rwn = -(fmin2(std::abs(rwn), sm.hsta(k).n));
            }
            sm.hsta(k).n = fmax2(1.e-6f, sm.hsta(k).n + rwn);
            sm.hact(k).n = fmax2(1.e-6f, sm.hact(k).n - rwn);
            nb.act_sta_n = nb.act_sta_n + rwn;
            float hmn = s.hru(j).nut.cmn * csf * sm.hact(k).n;
            hmn = fmin2(hmn, sm.hact(k).n);
            xx = sm.hsta(k).n + sm.hact(k).n;
            float hmp = xx > 1.e-6f ? 1.4f * hmn * sm.hsta(k).p / xx : 0.0f;
            hmp = fmin2(hmp, sm.hsta(k).p);
            sm.hact(k).n = fmax2(1.e-6f, sm.hact(k).n - hmn);
            sm.mn(k).no3 = sm.mn(k).no3 + hmn;
            sm.hsta(k).p = sm.hsta(k).p - hmp;
            sm.mp(k).lab = sm.mp(k).lab + hmp;
            nb.act_nit_n = nb.act_nit_n + hmn;
            nb.org_lab_p = nb.org_lab_p + hmp;
            for (s.ipl = 1; s.ipl <= s.pcom(j).npl; ++s.ipl) {
                organic_mass& r = sm.pl(s.ipl).rsd(k);
                float cnrf = 1.0f, cprf = 1.0f;
                if (r.n > 1.e-4f) {
                    float cnr = r.c / r.n;
                    if (cnr > 500.0f) cnr = 500.0f;
                    cnrf = std::exp(-0.693f * (cnr - 25.0f) / 25.0f);
                }
                if (r.p > 1.e-4f) {
                    float cpr = r.c / r.p;
                    if (cpr > 5000.0f) cpr = 5000.0f;
                    cprf = std::exp(-0.693f * (cpr - 200.0f) / 200.0f);
                }
                const float ca = fmin3(cnrf, cprf, 1.0f);
                const int idp = s.pcom(j).plcur(s.ipl).idplt;
                float decr = s.pldb(idp).rsdco_pl * ca * csf;
                decr = fmax2(s.bsn_prm.decr_min, decr);
                decr = fmin2(decr, 1.0f);
                s.decomp = decr * r;
                r = r - s.decomp;
                if (r.m < 1.e-10f) r.m = 0.0f;
                if (r.c < 1.e-10f) r.c = 0.0f;
                if (r.n < 1.e-10f) r.n = 0.0f;
                if (r.p < 1.e-10f) r.p = 0.0f;
                sm.mn(k).no3 = sm.mn(k).no3 + 0.8f * s.decomp.n;
                sm.hact(k).n = sm.hact(k).n + 0.2f * s.decomp.n;
                sm.mp(k).lab = sm.mp(k).lab + 0.8f * s.decomp.p;
                sm.hsta(k).p = sm.hsta(k).p + 0.2f * s.decomp.p;
            }
            float wdn = 0.0f;
            if (sut >= s.bsn_prm.sdnco) {
                wdn = sm.mn(k).no3 * (1.0f - std::exp(-s.bsn_prm.cdn * cdg * sm.cbn(k) / 100.0f));
            }
            sm.mn(k).no3 = fmax2(0.0001f, sm.mn(k).no3 - wdn);
            nb.denit = nb.denit + wdn;
        }
    }
}

// nut_nitvol.f90 (no septic systems)
void nut_nitvol(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    const float cecf = 0.15f;
    for (int k = 1; k <= so.nly; ++k) {
        const soil_physical_properties& ph = so.phys(k);
        const float tf = 0.41f * (ph.tmp - 5.0f) / 10.0f;
        if (sm.mn(k).nh4 > 0.0f && tf >= 0.001f) {
            const float sw25 = ph.wpmm + 0.25f * ph.fc;
            const float swwp = ph.wpmm + ph.st;
            float swf = 0.0f;
            if (swwp < sw25) {
                swf = (swwp - ph.wpmm) / (sw25 - ph.wpmm);
            } else {
                swf = 1.0f;
            }
            const float xx = k == 1 ? 0.0f : so.phys(k - 1).d;
            const float dmidl = (ph.d + xx) / 2.0f;
            const float dpf = 1.0f - dmidl / (dmidl + std::exp(4.706f - 0.0305f * dmidl));
            const float akn = tf * swf;
            const float akv = tf * dpf * cecf;
            const float rnv = sm.mn(k).nh4 * (1.0f - std::exp(-akn - akv));
            float rnit = 1.0f - std::exp(-akn);
            float rvol = 1.0f - std::exp(-akv);
            if (rvol + rnit > 1.e-6f) {
                rvol = rnv * rvol / (rvol + rnit);
                rnit = rnv - rvol;
                if (rnit < 0.0f) rnit = 0.0f;
                sm.mn(k).nh4 = fmax2(1.e-6f, sm.mn(k).nh4 - rnit);
            }
            if (sm.mn(k).nh4 < 0.0f) {
                rnit = rnit + sm.mn(k).nh4;
                sm.mn(k).nh4 = 0.0f;
            }
            sm.mn(k).no3 = sm.mn(k).no3 + rnit;
            sm.mn(k).nh4 = fmax2(1.e-6f, sm.mn(k).nh4 - rvol);
            if (sm.mn(k).nh4 < 0.0f) {
                rvol = rvol + sm.mn(k).nh4;
                sm.mn(k).nh4 = 0.0f;
            }
        }
    }
}

// nut_pminrl.f90
void nut_pminrl(Sim& s) {
    const int j = s.ihru;
    soil_profile_mass& sm = s.soil1(j);
    output_nutbal& nb = s.hnb_d(j);
    const float bk = 0.01f;
    nb.lab_min_p = 0.0f;
    nb.act_sta_p = 0.0f;
    const float rto = s.hru(j).nut.psp / (1.0f - s.hru(j).nut.psp);
    for (int l = 1; l <= s.soil(j).nly; ++l) {
        mineral_phosphorus& mp = sm.mp(l);
        float rmp1 = (mp.lab - mp.act * rto);
        if (rmp1 > 0.0f) rmp1 = rmp1 * 0.1f;
        if (rmp1 < 0.0f) rmp1 = rmp1 * 0.6f;
        rmp1 = fmin2(rmp1, mp.lab);
        float roc = bk * (4.0f * mp.act - mp.sta);
        if (roc < 0.0f) roc = roc * 0.1f;
        roc = fmin2(roc, mp.act);
        mp.sta = mp.sta + roc;
        if (mp.sta < 0.0f) mp.sta = 0.0f;
        mp.act = mp.act - roc + rmp1;
        if (mp.act < 0.0f) mp.act = 0.0f;
        mp.lab = mp.lab - rmp1;
        if (mp.lab < 0.0f) mp.lab = 0.0f;
        nb.lab_min_p = nb.lab_min_p + rmp1;
        nb.act_sta_p = nb.act_sta_p + roc;
    }
}

// pl_waterup.f90 (no salts)
void pl_waterup(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    soil_profile& so = s.soil(j);
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    if (so.sw > so.sumfc) {
        const float satco = (so.sw - so.sumfc) / (so.sumul - so.sumfc);
        const float scparm = 100.0f * (satco - s.pldb(idp).aeration) / (1.0001f - s.pldb(idp).aeration);
        if (scparm > 0.0f) {
            pc.plstr(ipl).strsa = 1.0f - (scparm / (scparm + std::exp(2.9014f - 0.03867f * scparm)));
        } else {
            pc.plstr(ipl).strsa = 1.0f;
        }
    }
    if (s.epmax(ipl) <= 1.e-6f) {
        pc.plstr(ipl).strsw = 1.0f;
        return;
    }
    float gx = 0.0f;
    int ir = 0;
    float sump = 0.0f, wuse = 0.0f, sum_wuse = 0.0f, sum_wusep = 0.0f;
    pc.plcur(ipl).uptake.fill(0.0f);
    for (int k = 1; k <= so.nly; ++k) {
        if (ir > 0) break;
        if (pc.plg(ipl).root_dep <= so.phys(k).d) {
            gx = pc.plg(ipl).root_dep;
            ir = k;
        } else {
            gx = so.phys(k).d;
        }
        float sum = 0.0f;
        if (pc.plg(ipl).root_dep <= 0.01f) {
            sum = s.epmax(ipl) / s.uptake.water_norm;
        } else {
            sum = s.epmax(ipl) * (1.0f - std::exp(-s.uptake.water_dis * gx / pc.plg(ipl).root_dep)) /
                  s.uptake.water_norm;
        }
        if (k == 2) {
            wuse = sum;
        } else {
            wuse = sum - sump * (1.0f - pc.plcur(ipl).epco);
        }
        wuse = fmin2(wuse, so.phys(k).st);
        sum_wuse = sum_wuse + wuse;
        if (sum_wuse > s.epmax(ipl)) {
            wuse = s.epmax(ipl) - sum_wusep;
            sum_wuse = s.epmax(ipl);
        }
        sump = sum;
        sum_wusep = sum_wuse;
        if (so.phys(k).st < wuse) wuse = so.phys(k).st;
        so.phys(k).st = fmax2(1.e-6f, so.phys(k).st - wuse);
        pc.plcur(ipl).uptake(k) = wuse;
    }
    so.sw = 0.0f;
    for (int k = 1; k <= so.nly; ++k) so.sw = so.sw + so.phys(k).st;
    pc.plstr(ipl).strss = 1.0f;
    pc.plstr(ipl).strsw = sum_wuse / s.epmax(ipl);
    s.ep_day = s.ep_day + sum_wuse;
}

// pl_community.f90 (no competition for light)
void pl_community(Sim& s) {
    const int j = s.ihru;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    s.par.fill(0.0f);
    pc.lai_sum = 0.0f;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) pc.lai_sum = pc.lai_sum + pc.plg(s.ipl).lai;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        if (pc.lai_sum > 1.e-6f) {
            s.epmax(s.ipl) = s.ep_max * pc.plg(s.ipl).lai / pc.lai_sum;
        } else {
            s.epmax(s.ipl) = 0.0f;
        }
    }
    pm.tot_com = s.orgz;
    pm.ab_gr_com = s.orgz;
    pm.leaf_com = s.orgz;
    pm.stem_com = s.orgz;
    pm.seed_com = s.orgz;
    pm.root_com = s.orgz;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        const int ipl = s.ipl;
        pm.tot_com = pm.tot_com + pm.tot(ipl);
        pm.ab_gr_com = pm.ab_gr_com + pm.ab_gr(ipl);
        pm.leaf_com = pm.leaf_com + pm.leaf(ipl);
        pm.stem_com = pm.stem_com + pm.stem(ipl);
        pm.seed_com = pm.seed_com + pm.seed(ipl);
        pm.root_com = pm.root_com + pm.root(ipl);
    }
    pc.cht_mx = 0.0f;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) pc.cht_mx = fmax2(pc.cht_mx, pc.plg(s.ipl).cht);
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl)
        if (pc.plcur(s.ipl).gro == "y") pl_waterup(s);
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        const int ipl = s.ipl;
        if (pc.plcur(ipl).idorm == "n" && pc.plcur(ipl).gro == "y") {
            const int idp = pc.plcur(ipl).idplt;
            const float ext = s.pldb(idp).ext_coef;
            s.par(ipl) = (1.0f - std::exp(-ext * (pc.plg(ipl).lai + 0.05f)));
            s.par(ipl) = 0.5f * s.w.solrad * (1.0f - std::exp(-ext * (pc.plg(ipl).lai + 0.05f)));
        }
    }
}

// pl_nupd.f90
void pl_nupd(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    const plant_cp& cp = s.plcp(idp);
    const float matur_frac = db.typ == "perennial"
                                 ? static_cast<float>(pc.plcur(ipl).curyr_mat) / static_cast<float>(db.mat_yrs)
                                 : pc.plcur(ipl).phuacc;
    pc.plm(ipl).n_fr =
        (db.pltnfr1 - db.pltnfr3) * (1.0f - matur_frac / (matur_frac + std::exp(cp.nup1 - cp.nup2 * matur_frac))) +
        db.pltnfr3;
    s.un2(ipl) = pc.plm(ipl).n_fr * s.pl_mass(j).tot(ipl).m;
    if (s.un2(ipl) < s.pl_mass(j).tot(ipl).n) s.un2(ipl) = s.pl_mass(j).tot(ipl).n;
    s.uno3d(ipl) = s.un2(ipl) - s.pl_mass(j).tot(ipl).n;
}

// pl_pupd.f90
void pl_pupd(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    const plant_cp& cp = s.plcp(idp);
    const float matur_frac = db.typ == "perennial"
                                 ? static_cast<float>(pc.plcur(ipl).curyr_mat) / static_cast<float>(db.mat_yrs)
                                 : pc.plcur(ipl).phuacc;
    pc.plm(ipl).p_fr =
        (db.pltpfr1 - db.pltpfr3) * (1.0f - matur_frac / (matur_frac + std::exp(cp.pup1 - cp.pup2 * matur_frac))) +
        db.pltpfr3;
    s.up2(ipl) = pc.plm(ipl).p_fr * s.pl_mass(j).ab_gr(ipl).m;
    if (s.up2(ipl) < s.pl_mass(j).ab_gr(ipl).p) s.up2(ipl) = s.pl_mass(j).ab_gr(ipl).p;
    s.uapd(ipl) = s.up2(ipl) - s.pl_mass(j).ab_gr(ipl).p;
    s.uapd(ipl) = 1.5f * s.uapd(ipl);
}

// pl_nut_demand.f90 (delg_p keeps its value from an earlier call when phumat <= 0.1, as
// the reference's SAVEd local)
void pl_nut_demand(Sim& s) {
    const int j = s.ihru;
    plant_community& pc = s.pcom(j);
    s.uno3d(s.ipl) = 0.0f;
    s.uno3d_tot = 0.0f;
    s.uapd(s.ipl) = 0.0f;
    s.uapd_tot = 0.0f;
    float& delg_p = s.saved.pl_nut_demand_delg_p;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        const int ipl = s.ipl;
        plant_status& cur = pc.plcur(ipl);
        const int idp = cur.idplt;
        if (cur.idorm == "n" && cur.gro == "y") {
            float delg = 0.0f;
            if (cur.phumat > 0.1f) {
                delg = (s.w.tave - s.pldb(idp).t_base) / cur.phumat;
                delg_p = (s.w.tave - s.pldb(idp).t_base) / cur.phumat_p;
            }
            if (delg < 0.0f) delg = 0.0f;
            if (delg_p < 0.0f) delg_p = 0.0f;
            cur.phuacc = cur.phuacc + delg;
            cur.phuacc_p = cur.phuacc_p + delg_p;
            pl_nupd(s);
            pl_pupd(s);
            s.uno3d_tot = s.uno3d_tot + s.uno3d(ipl);
            s.uapd_tot = s.uapd_tot + s.uapd(ipl);
        }
    }
    s.sum_no3 = 0.0f;
    s.sum_solp = 0.0f;
    for (int nly = 1; nly <= s.soil(j).nly; ++nly) {
        s.sum_no3 = s.sum_no3 + s.soil1(j).mn(nly).no3;
        s.sum_solp = s.sum_solp + s.soil1(j).mp(nly).lab;
    }
}

// pl_dormant.f90
void pl_dormant(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    plant_status& cur = pc.plcur(ipl);
    const int idp = cur.idplt;
    const plant_db& db = s.pldb(idp);
    s.iwst = s.ob(s.hru(j).obj_no).wst;
    const int iwgn = s.wst(s.iwst).wco.wgn;
    const float daylength = s.wst(s.iwst).weat.daylength;
    if (cur.idorm == "n" && daylength - s.dormhr(j) < s.wgn_pms(iwgn).daylmn) {
        if (db.typ == "perennial") {
            cur.idorm = "y";
            const float rto = 0.0f;
            s.stem_drop = rto * pm.stem(ipl);
            const float lai_init = pc.plg(ipl).lai;
            pc.plg(ipl).lai = rto * lai_init;
            s.leaf_drop.m = rto * pm.leaf(ipl).m;
            s.leaf_drop.n = s.leaf_drop.m * pc.plm(ipl).n_fr;
            s.leaf_drop.n = fmax2(0.0f, s.leaf_drop.n);
            s.leaf_drop.p = s.leaf_drop.m * pc.plm(ipl).p_fr;
            s.leaf_drop.p = fmax2(0.0f, s.leaf_drop.p);
            s.seed_drop = pm.seed(ipl);
            s.abgr_drop = s.stem_drop + s.leaf_drop + s.seed_drop;
            pm.tot(ipl) = pm.tot(ipl) - s.abgr_drop;
            if (pm.tot_com.m < 0.0f) pm.tot_com.m = 0.0f;
            pm.ab_gr(ipl) = pm.ab_gr(ipl) - s.abgr_drop;
            pm.stem(ipl) = pm.stem(ipl) - s.stem_drop;
            pm.leaf(ipl) = pm.leaf(ipl) - s.leaf_drop;
            pm.seed(ipl) = s.plt_mass_z;
            pm.rsd(ipl) = pm.rsd(ipl) + s.abgr_drop;
            pm.rsd_tot = pm.rsd_tot + s.abgr_drop;
            cur.idorm = "y";
            cur.phuacc = 0.0f;
            pc.plstr(ipl).strsw = 1.0f;
        }
        if (db.typ == "cold_annual") {
            if (cur.phuacc < 0.75f) {
                cur.idorm = "y";
                pc.plstr(ipl).strsw = 1.0f;
            }
        }
    }
    if (cur.idorm == "y" && daylength - s.dormhr(j) >= s.wgn_pms(iwgn).daylmn) {
        if (db.typ == "perennial") {
            cur.idorm = "n";
            cur.phuacc = 0.0f;
            pc.plg(ipl).d_senes = 0;
        }
        if (db.typ == "cold_annual") {
            cur.idorm = "n";
            cur.phuacc = 0.0f;
        }
    }
}

// pl_tstr.f90 (the reference's last assignment, sin(1.5707 rto), always wins)
void pl_tstr(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const plant_db& db = s.pldb(pc.plcur(ipl).idplt);
    const weather_daily& w = s.w;
    float tgx = w.tave - db.t_base;
    float& strst = pc.plstr(ipl).strst;
    if (tgx <= 0.0f) {
        strst = 0.0f;
    } else {
        if (w.tave > db.t_opt) tgx = 2.0f * db.t_opt - db.t_base - w.tave;
        const float rto = fpowi((db.t_opt - w.tave) / (tgx + 1.e-6f), 2);
        if (rto <= 200.0f && tgx > 0.0f) {
            strst = std::exp(-0.1054f * rto);
        } else {
            strst = 0.0f;
        }
        if (w.tmin <= s.wgn_pms(s.iwgen).tmp_an - 15.0f) strst = 0.0f;
    }
    const float rto = (w.tave - db.t_base) / (db.t_opt - db.t_base);
    if (rto > 0.0f || rto < 2.0f) {
        strst = std::sin(1.5707f * rto);
    } else {
        strst = 0.0f;
    }
    strst = fmax2(0.0f, strst);
    strst = fmin2(1.0f, strst);
}

// nuts.f90
void nuts(float u1, float u2, float& uu) {
    uu = 200.0f * (u1 / (u2 + 0.0001f) - 0.5f);
    if (uu <= 0.0f) {
        uu = 0.0f;
    } else if (uu < 99.0f) {
        uu = uu / (uu + std::exp(3.535f - 0.02597f * uu));
    } else {
        uu = 1.0f;
    }
    if (u2 <= 1.e-6f) uu = 1.0f;
}

// pl_nfix.f90
void pl_nfix(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    const plant_status& cur = s.pcom(j).plcur(ipl);
    const plant_db& db = s.pldb(cur.idplt);
    float uno3l = 0.0f;
    if (s.uno3d(ipl) > s.nplnt(j)) {
        uno3l = s.uno3d(ipl) - s.nplnt(j);
    } else {
        s.fixn = 0.0f;
        return;
    }
    const soil_profile& so = s.soil(j);
    const float fxw = so.sw / (0.85f * so.sumfc);
    float sumn = 0.0f, fxn = 0.0f;
    for (int l = 1; l <= so.nly; ++l) sumn = sumn + s.soil1(j).mn(l).no3;
    if (sumn > 300.0f) fxn = 0.0f;
    if (sumn > 100.0f && sumn <= 300.0f) fxn = 1.5f - 0.005f * sumn;
    if (sumn <= 100.0f) fxn = 1.0f;
    float fxg = 0.0f;
    const float phu = cur.phuacc;
    if (phu > 0.15f && phu <= 0.30f) fxg = 6.67f * phu - 1.0f;
    if (phu > 0.30f && phu <= 0.55f) fxg = 1.0f;
    if (phu > 0.55f && phu <= 0.75f) fxg = 3.75f - 5.0f * phu;
    float fxr = fmin3(1.0f, fxw, fxn) * fxg;
    fxr = fmax2(0.0f, fxr);
    s.fixn = fmin2(6.0f, fxr * uno3l);
    s.fixn = db.nfix_co * s.fixn + (1.0f - db.nfix_co) * uno3l;
    s.fixn = fmin2(s.fixn, uno3l);
    s.fixn = fmin2(s.bsn_prm.nfixmx, s.fixn);
}

// pl_nup.f90
void pl_nup(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const soil_profile& so = s.soil(j);
    const int idp = pc.plcur(ipl).idplt;
    pc.plstr(ipl).strsn = 1.0f;
    s.hnb_d(j).nuptake = 0.0f;
    if (s.uno3d(ipl) < 1.e-6f) return;
    float root_depth = fmax2(10.1f, pc.plg(ipl).root_dep);
    for (int l = 1; l <= so.nly; ++l) {
        const float soil_depth = so.phys(l).d;
        if (root_depth < soil_depth) {
            root_depth = so.phys(l).d;
            break;
        }
    }
    for (int l = 1; l <= so.nly; ++l) {
        const float soil_depth = so.phys(l).d;
        if (root_depth < soil_depth) break;
        const float unmx = s.uno3d(ipl) * s.rto_no3 * (1.0f - std::exp(-s.bsn_prm.n_updis * soil_depth / root_depth)) /
                           s.uptake.n_norm;
        const float uno3l = fmin2(unmx - s.nplnt(j), s.soil1(j).mn(l).no3);
        s.nplnt(j) = s.nplnt(j) + uno3l;
        s.soil1(j).mn(l).no3 = s.soil1(j).mn(l).no3 - uno3l;
    }
    if (s.nplnt(j) < 0.0f) s.nplnt(j) = 0.0f;
    if (s.pldb(idp).nfix_co > 1.e-6f) pl_nfix(s);
    s.nplnt(j) = s.nplnt(j) + s.fixn;
    pm.tot(ipl).n = pm.tot(ipl).n + s.nplnt(j);
    pm.ab_gr(ipl).n = pm.ab_gr(ipl).n + s.nplnt(j) * (1.0f - pc.plg(ipl).root_frac);
    pm.root(ipl).n = pm.root(ipl).n + s.nplnt(j) * pc.plg(ipl).root_frac;
    s.pl_mass_up.n = s.nplnt(j);
    s.hnb_d(j).nuptake = s.hnb_d(j).nuptake + s.nplnt(j);
    if (s.pldb(idp).nfix_co > 1.e-6f) {
        pc.plstr(ipl).strsn = 1.0f;
    } else {
        nuts(pm.tot(ipl).n, s.un2(ipl), pc.plstr(ipl).strsn);
        const float xx = s.uno3d(ipl) > 1.e-5f ? s.nplnt(j) / s.uno3d(ipl) : 1.0f;
        pc.plstr(ipl).strsn = fmax2(pc.plstr(ipl).strsn, xx);
        pc.plstr(ipl).strsn = fmin2(pc.plstr(ipl).strsn, 1.0f);
    }
}

// pl_pup.f90
void pl_pup(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const soil_profile& so = s.soil(j);
    pc.plstr(ipl).strsp = 1.0f;
    s.hnb_d(j).puptake = 0.0f;
    if (s.uapd(ipl) < 1.e-6f) return;
    float root_depth = fmax2(10.1f, pc.plg(ipl).root_dep);
    for (int l = 1; l <= so.nly; ++l) {
        const float soil_depth = so.phys(l).d;
        if (root_depth < soil_depth) {
            root_depth = so.phys(l).d;
            break;
        }
    }
    for (int l = 1; l <= so.nly; ++l) {
        const float soil_depth = so.phys(l).d;
        if (root_depth < soil_depth) break;
        const float upmx = s.uapd(ipl) * s.rto_solp * (1.0f - exp_w(-s.bsn_prm.p_updis * soil_depth / root_depth)) /
                           s.uptake.p_norm;
        const float uapl = fmin2(upmx - s.pplnt(j), s.soil1(j).mp(l).lab);
        s.pplnt(j) = s.pplnt(j) + uapl;
        s.soil1(j).mp(l).lab = s.soil1(j).mp(l).lab - uapl;
    }
    if (s.pplnt(j) < 0.0f) s.pplnt(j) = 0.0f;
    pm.tot(ipl).p = pm.tot(ipl).p + s.pplnt(j);
    pm.ab_gr(ipl).p = pm.ab_gr(ipl).p + s.pplnt(j) * (1.0f - pc.plg(ipl).root_frac);
    pm.root(ipl).p = pm.root(ipl).p + s.pplnt(j) * pc.plg(ipl).root_frac;
    s.pl_mass_up.p = s.pplnt(j);
    s.hnb_d(j).puptake = s.hnb_d(j).puptake + s.pplnt(j);
    nuts(pm.ab_gr(ipl).p, s.up2(ipl), pc.plstr(ipl).strsp);
}

// pl_biomass_gro.f90 (no salts or constituents)
void pl_biomass_gro(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    const int idp = pc.plcur(ipl).idplt;
    const plant_db& db = s.pldb(idp);
    s.iwst = s.ob(j).wst;
    if (pc.plcur(ipl).phuacc <= 1.0f) {
        float beadj = 0.0f;
        const float co2 = s.co2y(s.time.yrs);
        if (s.bsn_prm.co2 > 350.0f) {
            beadj = 100.0f * co2 / (co2 + std::exp(s.plcp(idp).ruc1 - co2 * s.plcp(idp).ruc2));
        } else {
            beadj = db.bio_e;
        }
        if (s.vpd > 1.0f) {
            const float ruedecl = s.vpd - 1.0f;
            beadj = beadj - db.wavp * ruedecl;
            beadj = fmax2(beadj, 0.27f * db.bio_e);
        }
        s.iwst = s.ob(s.hru(j).obj_no).wst;
        s.bioday = beadj * s.par(ipl);
        if (s.bioday < 0.0f) s.bioday = 0.0f;
        pl_tstr(s);
        s.rto_no3 = s.uno3d_tot > s.sum_no3 ? s.uno3d(ipl) / s.uno3d_tot : 1.0f;
        s.rto_solp = s.uapd_tot > s.sum_solp ? s.uapd(ipl) / s.uapd_tot : 1.0f;
        s.uno3d(ipl) = fmin2(4.0f * db.pltnfr3 * s.bioday, s.uno3d(ipl));
        if (s.uapd(ipl) > 10.0f) s.uapd(ipl) = fmin2(4.0f * db.pltpfr3 * s.bioday, s.uapd(ipl));
        pl_nup(s);
        pl_pup(s);
        plant_stress& st = pc.plstr(ipl);
        if (s.bsn_cc.nostress == 1) {
            st.strsw = 1.0f;
            st.strst = 1.0f;
            st.strsn = 1.0f;
            st.strsp = 1.0f;
            st.strsa = 1.0f;
            st.strss = 1.0f;
        }
        if (s.bsn_cc.nostress == 2) {
            st.strsn = 1.0f;
            st.strsp = 1.0f;
        }
        st.reg = fmin_list({st.strsw, st.strst, st.strsn, st.strsp, st.strsa, st.strss});
        if (st.reg < 0.0f) st.reg = 0.0f;
        if (st.reg > 1.0f) st.reg = 1.0f;
        s.pl_mass_up.m = s.bioday * st.reg;
        s.pl_mass_up.c = 0.42f * s.pl_mass_up.m;
        s.hpw_d(j).bm_grow = s.hpw_d(j).bm_grow + s.pl_mass_up.m;
        s.hpw_d(j).c_gro = s.hpw_d(j).c_gro + s.pl_mass_up.c;
        st.sum_w = st.sum_w + (1.0f - st.strsw);
        st.sum_tmp = st.sum_tmp + (1.0f - st.strst);
        st.sum_n = st.sum_n + (1.0f - st.strsn);
        st.sum_p = st.sum_p + (1.0f - st.strsp);
        st.sum_a = st.sum_a + (1.0f - st.strsa);
    }
}

// pl_leaf_gro.f90
void pl_leaf_gro(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_status& cur = pc.plcur(ipl);
    plant_growth& g = pc.plg(ipl);
    const int idp = cur.idplt;
    const plant_db& db = s.pldb(idp);
    const plant_cp& cp = s.plcp(idp);
    const float f = cur.phuacc / (cur.phuacc + std::exp(cp.leaf1 - cp.leaf2 * cur.phuacc));
    g.laimxfr = fmin2(f, g.laimxfr);
    const float ff = f - g.laimxfr;
    g.laimxfr = f;
    float exponent = cp.leaf1 - cp.leaf2 * cur.phuacc_p;
    if (exponent < -16.0f) exponent = -16.0f;
    const float f_p = cur.phuacc_p / (cur.phuacc_p + std::exp(exponent));
    if (cur.phuacc < db.dlai) {
        float laimax = 0.0f;
        if (db.typ == "perennial") {
            if (cur.curyr_mat < 1) cur.curyr_mat = 1;
            const float rto_lin = static_cast<float>(cur.curyr_mat) / static_cast<float>(db.mat_yrs);
            const float rto = std::log10(rto_lin);
            const float lai_exp = rto * db.laixco_tree;
            laimax = cur.lai_pot * fpow(10.0f, lai_exp);
            laimax = fmin2(laimax, cur.lai_pot);
        } else {
            laimax = cur.lai_pot;
        }
        if (db.typ == "perennial") {
            g.cht = db.chtmx * std::sqrt(f_p);
        } else {
            g.cht = db.chtmx * std::sqrt(f);
        }
        if (db.typ == "perennial") g.leaf_frac = 0.03f;
        if (g.lai > laimax) g.lai = laimax;
        float deltalai = ff * laimax * (1.0f - std::exp(5.0f * (g.lai - laimax))) * std::sqrt(pc.plstr(ipl).strsw);
        float sumlaiht = 0.0f;
        for (int jpl = 1; jpl <= pc.npl; ++jpl) sumlaiht = sumlaiht + g.lai * pc.plg(jpl).cht;
        const float rto = sumlaiht > 1.e-6f ? (g.lai * g.cht) / sumlaiht : 1.0f;
        deltalai = deltalai * rto;
        g.lai = g.lai + deltalai;
        if (g.lai > laimax) g.lai = laimax;
        g.olai = g.lai;
        g.dphu = cur.phuacc;
    }
}

// pl_leaf_senes.f90
void pl_leaf_senes(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    plant_status& cur = pc.plcur(ipl);
    plant_growth& g = pc.plg(ipl);
    const int idp = cur.idplt;
    const plant_db& db = s.pldb(idp);
    s.leaf_drop.m = 0.0f;
    if (db.typ == "warm_annual" || db.typ == "cold_annual" || db.typ == "warm_annual_tuber" ||
        db.typ == "cold_annual_tuber") {
        if (cur.phuacc > db.dlai && cur.phuacc < 1.0f) {
            const float rto = (1.0f - cur.phuacc) / (1.0f - g.dphu);
            g.lai = g.olai * fpow(rto, db.dlai_rate);
        }
    }
    if (db.typ == "perennial" && db.trig == "temp_gro") {
        if (cur.phuacc > db.dlai && g.d_senes < 15.0f) {
            s.iwst = s.ob(s.hru(j).obj_no).wst;
            g.d_senes = g.d_senes + 1.0f;
            const float rto = 1.0f - (g.d_senes / 15.0f);
            g.lai = (g.olai - db.alai_min) * rto + db.alai_min;
            g.lai = fmax2(g.lai, db.alai_min);
            s.leaf_drop.m = cur.leaf_tov * pm.leaf(ipl).m;
            s.leaf_drop.n = 0.68f * s.leaf_drop.m * pc.plm(ipl).n_fr;
            s.leaf_drop.n = fmax2(0.0f, s.leaf_drop.n);
            s.leaf_drop.p = 0.68f * s.leaf_drop.m * pc.plm(ipl).p_fr;
            s.leaf_drop.p = fmax2(0.0f, s.leaf_drop.p);
        }
    }
    if (db.typ == "perennial" && db.trig == "moisture_gro") {
        s.iwst = s.ob(s.hru(j).obj_no).wst;
        const int iwgn = s.wst(s.iwst).wco.wgn;
        const float ppet = s.wgn_pms(iwgn).precip_sum / s.wgn_pms(iwgn).pet_sum;
        float leaf_tov_mon = 0.0f;
        if (ppet < 0.5f) {
            const float coef = 1.0f;
            const float exp_co = -10.0f * ppet + 6.0f;
            leaf_tov_mon = coef * std::exp(-exp_co) * (db.leaf_tov_min - db.leaf_tov_max) + db.leaf_tov_max;
        } else {
            leaf_tov_mon = db.leaf_tov_min;
        }
        leaf_tov_mon = fmin2(leaf_tov_mon, db.leaf_tov_min);
        leaf_tov_mon = fmax2(leaf_tov_mon, db.leaf_tov_max);
        cur.leaf_tov = (1.0f / (30.0f * leaf_tov_mon));
        g.lai = g.lai - cur.leaf_tov;
        g.lai = fmax2(g.lai, db.alai_min);
        s.leaf_drop.m = cur.leaf_tov * pm.leaf(ipl).m;
        s.leaf_drop.n = 0.68f * s.leaf_drop.m * pc.plm(ipl).n_fr;
        s.leaf_drop.n = fmax2(0.0f, s.leaf_drop.n);
        s.leaf_drop.p = 0.68f * s.leaf_drop.m * pc.plm(ipl).p_fr;
        s.leaf_drop.p = fmax2(0.0f, s.leaf_drop.p);
    }
    if (s.leaf_drop.m > 0.0f) {
        pm.leaf(ipl) = pm.leaf(ipl) - s.leaf_drop;
        pm.tot(ipl) = pm.tot(ipl) - s.leaf_drop;
        pm.ab_gr(ipl) = pm.ab_gr(ipl) - s.leaf_drop;
        pm.rsd(ipl) = pm.rsd(ipl) + s.leaf_drop;
        pm.rsd_tot = pm.rsd_tot + s.leaf_drop;
        pm.tot_com = pm.tot_com - s.leaf_drop;
        pm.ab_gr_com = pm.ab_gr_com - s.leaf_drop;
        pm.leaf_com = pm.leaf_com - s.leaf_drop;
        s.hrc_d(j).plant_surf_c = s.hrc_d(j).plant_surf_c + s.leaf_drop.c;
        s.hpc_d(j).drop_c = s.hpc_d(j).drop_c + s.leaf_drop.c;
    }
}

// pl_mortality.f90
void pl_mortality(Sim& s) {
    const int j = s.ihru;
    const int ipl = s.ipl;
    plant_community& pc = s.pcom(j);
    plant_community_mass& pm = s.pl_mass(j);
    const plant_db& db = s.pldb(pc.plcur(ipl).idplt);
    const float bm_dieoff = (1.0f + db.bm_dieoff) * (pm.ab_gr(ipl).m - (db.bmx_peren * 1000.0f));
    if (bm_dieoff > 1.e-6f && pm.ab_gr(ipl).m > bm_dieoff) {
        float rto = bm_dieoff / pm.ab_gr(ipl).m;
        rto = fmin2(1.0f, rto);
        float rto1 = 1.0f - rto;
        rto1 = fmax2(0.0f, rto1);
        pm.rsd(ipl) = pm.rsd(ipl) + rto * pm.ab_gr(ipl);
        pm.rsd_tot = pm.rsd_tot + rto * pm.ab_gr(ipl);
        for (int ly = 1; ly <= s.soil(j).nly; ++ly)
            s.soil1(j).pl(ipl).rsd(ly) = s.soil1(j).pl(ipl).rsd(ly) + rto * pc.plg(ipl).rtfr(ly) * pm.root(ipl);
        pm.tot(ipl) = rto1 * pm.tot(ipl);
        pm.ab_gr(ipl) = rto1 * pm.ab_gr(ipl);
        pm.leaf(ipl) = rto1 * pm.leaf(ipl);
        pm.stem(ipl) = rto1 * pm.stem(ipl);
        pm.seed(ipl) = rto1 * pm.seed(ipl);
        pm.root(ipl) = rto1 * pm.root(ipl);
    }
}

// pl_grow.f90
void pl_grow(Sim& s) {
    const int j = s.ihru;
    plant_community& pc = s.pcom(j);
    pl_nut_demand(s);
    s.hpw_d(j).bm_grow = 0.0f;
    s.hpw_d(j).c_gro = 0.0f;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        const int ipl = s.ipl;
        s.pl_mass_up = s.plt_mass_z;
        if (pc.plcur(ipl).gro == "y") {
            const int idp = pc.plcur(ipl).idplt;
            if (s.pldb(idp).trig == "temp_gro") pl_dormant(s);
        }
        if (pc.plcur(ipl).idorm == "n" && pc.plcur(ipl).gro == "y") {
            pl_biomass_gro(s);
            pl_root_gro(s, j);
            pl_leaf_gro(s);
            pl_leaf_senes(s);
            pl_seed_gro(s, j);
            pl_partition(s, j, 0);
        }
        if (s.time.end_yr == 1) pl_mortality(s);
    }
}

// pest_enrsb.f90: the enrichment ratio of the sediment
void pest_enrsb(Sim& s) {
    const int j = s.ihru;
    if (s.sedyld(j) < 1.e-4f) {
        s.sedyld(j) = 0.0f;
        s.sanyld(j) = 0.0f;
        s.silyld(j) = 0.0f;
        s.clayld(j) = 0.0f;
        s.sagyld(j) = 0.0f;
        s.lagyld(j) = 0.0f;
    }
    const float cy = 0.1f * s.sedyld(j) / (s.hru(j).area_ha * s.surfq(j) + 1.e-6f);
    if (cy > 1.e-6f) {
        s.enratio = 0.78f * fpow(cy, -0.2468f);
    } else {
        s.enratio = 0.0f;
    }
    if (s.enratio > 3.0f) s.enratio = 3.0f;
}

// nut_orgn.f90
void nut_orgn(Sim& s) {
    const int j = s.ihru;
    soil_profile_mass& sm = s.soil1(j);
    const float orgn_kgha = sm.hsta(1).n + sm.hact(1).n;
    const float wt1 = 10000.0f * s.soil(j).phys(1).bd * s.soil(j).phys(1).d;
    const float er = s.hru(j).hyd.erorgn > 0.001f ? s.hru(j).hyd.erorgn : s.enratio;
    const float frac = orgn_kgha * er / wt1;
    s.sedorgn(j) = 1000.0f * frac * s.sedyld(j) / s.hru(j).area_ha;
    if (orgn_kgha > 1.e-6f) {
        sm.hact(1).n = sm.hact(1).n - s.sedorgn(j) * (sm.hact(1).n / orgn_kgha);
        sm.hsta(1).n = sm.hsta(1).n - s.sedorgn(j) * (sm.hsta(1).n / orgn_kgha);
        if (sm.hact(1).n < 0.0f) {
            s.sedorgn(j) = s.sedorgn(j) + sm.hact(1).n;
            sm.hact(1).n = 0.0f;
        }
        if (sm.hsta(1).n < 0.0f) {
            s.sedorgn(j) = s.sedorgn(j) + sm.hsta(1).n;
            sm.hsta(1).n = 0.0f;
        }
    }
}

// nut_psed.f90 (the reference's fractions, kept from an earlier call when the layer
// has no phosphorus: see IDEAS.md)
void nut_psed(Sim& s) {
    const int j = s.ihru;
    soil_profile_mass& sm = s.soil1(j);
    float& fr_orgp = s.saved.nut_psed_fr_orgp;
    float& fr_actmin = s.saved.nut_psed_fr_actmin;
    float& fr_stamin = s.saved.nut_psed_fr_stamin;
    const float sedp_attach = sm.hsta(1).p + sm.man(1).p + sm.man(1).p + sm.mp(1).sta + sm.mp(1).act;
    if (sedp_attach > 1.e-9f) {
        fr_orgp = (sm.hsta(1).p + sm.man(1).p + sm.man(1).p) / sedp_attach;
        fr_actmin = sm.mp(1).sta / sedp_attach;
        fr_stamin = sm.mp(1).act / sedp_attach;
    }
    const float wt1 = 10000.0f * s.soil(j).phys(1).bd * s.soil(j).phys(1).d;
    const float er = s.hru(j).hyd.erorgp > 0.001f ? s.hru(j).hyd.erorgp : s.enratio;
    const float frac = sedp_attach * er / wt1;
    const float sedp = 1000.0f * frac * s.sedyld(j) / s.hru(j).area_ha;
    if (sedp > 1.e-9f) {
        s.sedorgp(j) = sedp * fr_orgp;
        s.sedminpa(j) = sedp * fr_actmin;
        s.sedminpa(j) = fmin2(s.sedminpa(j), sm.mp(1).act);
        sm.mp(1).act = sm.mp(1).act - s.sedminpa(j);
        s.sedminps(j) = sedp * fr_stamin;
        s.sedminps(j) = fmin2(s.sedminps(j), sm.mp(1).sta);
        sm.mp(1).sta = sm.mp(1).sta - s.sedminps(j);
        const float sed_orgp = sm.hsta(1).p + sm.man(1).p;
        if (sed_orgp > 1.e-6f) {
            float sed_hump = s.sedorgp(j) * (sm.hsta(1).p / sed_orgp);
            sed_hump = fmin2(sed_hump, sm.hsta(1).p);
            sm.hsta(1).p = sm.hsta(1).p - sed_hump;
            float sed_manp = s.sedorgp(j) * (sm.man(1).p / sed_orgp);
            sed_manp = fmin2(sed_manp, sm.man(1).p);
            sm.man(1).p = sm.man(1).p - sed_manp;
            float sed_rsd_manp = s.sedorgp(j) * (sm.man(1).p / sed_orgp);
            sed_rsd_manp = fmin2(sed_rsd_manp, sm.man(1).p);
            sm.man(1).p = sm.man(1).p - sed_rsd_manp;
        }
    } else {
        s.sedorgp(j) = 0.0f;
        s.sedminpa(j) = 0.0f;
        s.sedminps(j) = 0.0f;
    }
}

// nut_nlch.f90 (no gwflow)
void nut_nlch(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    const hydrologic_response_unit& h = s.hru(j);
    float percnlyr = 0.0f;
    for (int jj = 1; jj <= so.nly; ++jj) {
        sm.mn(jj).no3 = sm.mn(jj).no3 + percnlyr;
        if (sm.mn(jj).no3 < 1.e-6f) sm.mn(jj).no3 = 0.0f;
        const float sro = jj == 1 ? s.surfq(j) : 0.0f;
        float vv = so.ly(jj).prk + sro + so.ly(jj).flat + 1.e-10f;
        if (h.lumv.ldrain == jj) vv = vv + s.qtile;
        float ww = -vv / ((1.0f - so.anion_excl) * so.phys(jj).ul);
        if (ww < -80.0f) ww = -80.0f;
        const float vno3 = sm.mn(jj).no3 * (1.0f - std::exp(ww));
        const float co = fmax2(vno3 / vv, 0.0f);
        if (jj == 1) {
            s.surqno3(j) = s.surfq(j) * h.nut.nperco * co;
            s.surqno3(j) = fmin2(s.surqno3(j), sm.mn(jj).no3);
            sm.mn(jj).no3 = fmax2(0.0001f, sm.mn(jj).no3 - s.surqno3(j));
        }
        float ssfnlyr = co * so.ly(jj).flat;
        ssfnlyr = fmin2(ssfnlyr, sm.mn(jj).no3);
        s.latno3(j) = s.latno3(j) + ssfnlyr;
        sm.mn(jj).no3 = fmax2(0.0001f, sm.mn(jj).no3 - ssfnlyr);
        percnlyr = co * so.ly(jj).prk;
        percnlyr = fmin2(percnlyr, sm.mn(jj).no3);
        sm.mn(jj).no3 = fmax2(0.0001f, sm.mn(jj).no3 - percnlyr);
        if (jj == so.nly) s.percn(j) = percnlyr;
    }
}

// nut_solp.f90 (no gwflow)
void nut_solp(Sim& s) {
    const int j = s.ihru;
    soil_profile& so = s.soil(j);
    soil_profile_mass& sm = s.soil1(j);
    const hydrologic_response_unit& h = s.hru(j);
    output_losses& ls = s.hls_d(j);
    ls.surqsolp = 0.0f;
    ls.lchlabp = 0.0f;
    ls.tilelabp = 0.0f;
    sm.mp(1).lab = sm.mp(1).lab + s.ht1.solp;
    const float xx = so.phys(1).bd * so.phys(1).d * h.nut.phoskd;
    s.surqsolp(j) = sm.mp(1).lab * s.surfq(j) / (xx + 1.0f);
    s.surqsolp(j) = fmin2(s.surqsolp(j), sm.mp(1).lab);
    s.surqsolp(j) = fmax2(s.surqsolp(j), 0.0f);
    ls.surqsolp = s.surqsolp(j);
    sm.mp(1).lab = sm.mp(1).lab - s.surqsolp(j);
    for (int ly = 1; ly <= so.nly; ++ly) {
        if (ly != s.i_sep(j)) {
            float vap = -so.ly(ly).prk / (0.01f * so.phys(ly).st + 0.1f * h.nut.pperco * so.phys(ly).bd);
            float plch = 0.001f * sm.mp(ly).lab * (1.0f - exp_w(vap));
            plch = fmin2(plch, sm.mp(ly).lab);
            sm.mp(ly).lab = sm.mp(ly).lab - plch;
            if (ly == so.nly) {
                ls.lchlabp = plch;
            } else {
                sm.mp(ly + 1).lab = sm.mp(ly + 1).lab + plch;
            }
            if (ly == h.lumv.ldrain) {
                vap = -s.qtile / (0.01f * so.phys(ly).st + 0.1f * h.nut.pperco * so.phys(ly).bd);
                plch = 0.001f * sm.mp(ly).lab * (1.0f - exp_w(vap));
                plch = fmin2(plch, sm.mp(ly).lab);
                sm.mp(ly).lab = sm.mp(ly).lab - plch;
                ls.tilelabp = plch;
            }
        }
    }
}

// swr_latsed.f90 (the lateral sediment is added twice, as the reference: see IDEAS.md)
void swr_latsed(Sim& s) {
    const int j = s.ihru;
    const hydrologic_response_unit& h = s.hru(j);
    const soil_profile& so = s.soil(j);
    s.sedyld(j) = s.sedyld(j) + h.hyd.lat_sed * h.area_ha * (s.latq(j) + s.qtile) / 100000.0f;
    s.sedyld(j) = s.sedyld(j) + (s.latq(j) + s.qtile) * h.hyd.lat_sed / 100000.0f;
    s.sanyld(j) = s.sanyld(j) + s.latq(j) * h.km * h.hyd.lat_sed * so.det_san;
    s.silyld(j) = s.silyld(j) + s.latq(j) * h.km * h.hyd.lat_sed * so.det_sil;
    s.clayld(j) = s.clayld(j) + s.latq(j) * h.km * h.hyd.lat_sed * so.det_cla;
    s.sagyld(j) = s.sagyld(j) + s.latq(j) * h.km * h.hyd.lat_sed * so.det_sag;
    s.lagyld(j) = s.lagyld(j) + s.latq(j) * h.km * h.hyd.lat_sed * so.det_lag;
    s.sedorgn(j) = s.sedorgn(j) + s.latq(j) * h.hyd.lat_orgn / 10000.0f;
    s.sedorgp(j) = s.sedorgp(j) + s.latq(j) * h.hyd.lat_orgp / 10000.0f;
    for (Vec<float>* v : {&s.sedyld, &s.sanyld, &s.silyld, &s.clayld, &s.sagyld, &s.lagyld, &s.sedorgn, &s.sedorgp})
        if ((*v)(j) < 0.0f) (*v)(j) = 0.0f;
}

// stor_surfstor.f90 (daily step, no salts or constituents)
void stor_surfstor(Sim& s) {
    const int j = s.ihru;
    Mat<float>& bs = s.surf_bs;
    const float brt = s.brt(j);
    bs(2, j) = fmax2(1.e-9f, bs(2, j) + s.sedyld(j));
    s.sedyld(j) = bs(2, j) * brt;
    bs(2, j) = bs(2, j) - s.sedyld(j);
    bs(13, j) = fmax2(1.e-6f, bs(13, j) + s.sanyld(j));
    bs(14, j) = fmax2(1.e-6f, bs(14, j) + s.silyld(j));
    bs(15, j) = fmax2(1.e-6f, bs(15, j) + s.clayld(j));
    bs(16, j) = fmax2(1.e-6f, bs(16, j) + s.sagyld(j));
    bs(17, j) = fmax2(1.e-6f, bs(17, j) + s.lagyld(j));
    bs(3, j) = fmax2(1.e-9f, bs(3, j) + s.sedorgn(j));
    bs(4, j) = fmax2(1.e-9f, bs(4, j) + s.sedorgp(j));
    bs(5, j) = fmax2(1.e-9f, bs(5, j) + s.surqno3(j));
    bs(6, j) = fmax2(1.e-9f, bs(6, j) + s.surqsolp(j));
    bs(7, j) = fmax2(1.e-9f, bs(7, j) + s.sedminpa(j));
    bs(8, j) = fmax2(1.e-9f, bs(8, j) + s.sedminps(j));
    s.sanyld(j) = bs(13, j) * brt;
    s.silyld(j) = bs(14, j) * brt;
    s.clayld(j) = bs(15, j) * brt;
    s.sagyld(j) = bs(16, j) * brt;
    s.lagyld(j) = bs(17, j) * brt;
    s.sedorgn(j) = bs(3, j) * brt;
    s.sedorgp(j) = bs(4, j) * brt;
    s.surqno3(j) = bs(5, j) * brt;
    s.surqsolp(j) = bs(6, j) * brt;
    s.sedminpa(j) = bs(7, j) * brt;
    s.sedminps(j) = bs(8, j) * brt;
    bs(2, j) = bs(2, j) - s.sedyld(j);
    bs(13, j) = bs(13, j) - s.sanyld(j);
    bs(14, j) = bs(14, j) - s.silyld(j);
    bs(15, j) = bs(15, j) - s.clayld(j);
    bs(16, j) = bs(16, j) - s.sagyld(j);
    bs(17, j) = bs(17, j) - s.lagyld(j);
    bs(3, j) = bs(3, j) - s.sedorgn(j);
    bs(4, j) = bs(4, j) - s.sedorgp(j);
    bs(5, j) = bs(5, j) - s.surqno3(j);
    bs(6, j) = bs(6, j) - s.surqsolp(j);
    bs(7, j) = bs(7, j) - s.sedminpa(j);
    bs(8, j) = bs(8, j) - s.sedminps(j);
}

// swr_substor.f90
void swr_substor(Sim& s) {
    const int j = s.ihru;
    Mat<float>& b = s.bss;
    const hydrologic_response_unit& h = s.hru(j);
    b(1, j) = b(1, j) + s.latq(j);
    b(2, j) = b(2, j) + s.latno3(j);
    b(3, j) = b(3, j) + s.qtile;
    b(4, j) = b(4, j) + s.tileno3(j);
    for (int k = 1; k <= 4; ++k)
        if (b(k, j) < 1.e-6f) b(k, j) = 0.0f;
    s.latq(j) = b(1, j) * h.hyd.lat_ttime;
    s.latno3(j) = b(2, j) * h.hyd.lat_ttime;
    s.qtile = b(3, j) * h.lumv.tile_ttime;
    s.tileno3(j) = b(4, j) * h.lumv.tile_ttime;
    if (s.latq(j) < 1.e-6f) s.latq(j) = 0.0f;
    if (s.latno3(j) < 1.e-6f) s.latno3(j) = 0.0f;
    if (s.qtile < 1.e-6f) s.qtile = 0.0f;
    if (s.tileno3(j) < 1.e-6f) s.tileno3(j) = 0.0f;
    b(1, j) = b(1, j) - s.latq(j);
    b(2, j) = b(2, j) - s.latno3(j);
    b(3, j) = b(3, j) - s.qtile;
    b(4, j) = b(4, j) - s.tileno3(j);
}

// sq_surfst.f90 (daily step)
void sq_surfst(Sim& s) {
    const int j = s.ihru;
    s.bsprev = s.surf_bs(1, j);
    s.surf_bs(1, j) = fmax2(1.e-6f, s.surf_bs(1, j) + s.surfq(j));
    s.qday = s.surf_bs(1, j) * s.brt(j);
    s.surf_bs(1, j) = s.surf_bs(1, j) - s.qday;
}

// swr_subwq.f90 (static carbon)
void swr_subwq(Sim& s) {
    const int j = s.ihru;
    float wtmp = 5.0f + 0.75f * s.w.tave;
    if (wtmp <= 0.1f) wtmp = 0.1f;
    wtmp = wtmp + 273.15f;
    if (s.qdr(j) > 1.e-4f) {
        const float tp = 100.0f * (s.sedorgn(j) + s.surqno3(j)) / s.qdr(j);
        s.chl_a(j) = 0.1f * tp;
        const float org_c = (s.soil1(j).cbn(1) / 100.0f) * s.enratio * s.sedyld(j) * 1000.0f;
        s.cbodu(j) = 2.7f * org_c / (s.qdr(j) * s.hru(j).km) / 10000.0f;
        const float ww = -139.34410f + (1.575701E05f / wtmp);
        const float xx = 6.642308E07f / fpowi(wtmp, 2);
        const float yy = 1.243800E10f / fpowi(wtmp, 3);
        const float zz = 8.621949E11f / fpowi(wtmp, 4);
        float soxy = std::exp(ww - xx + yy - zz);
        if (soxy < 0.0f) soxy = 0.0f;
        s.doxq(j) = soxy * std::exp(-0.1f * s.cbodu(j));
        if (s.doxq(j) < 0.0f) s.doxq(j) = 0.0f;
        if (s.doxq(j) > soxy) s.doxq(j) = soxy;
    } else {
        s.chl_a(j) = 0.0f;
        s.cbodu(j) = 0.0f;
        s.doxq(j) = 0.0f;
    }
}

// hru_hyds.f90 (daily step, no constituents)
void hru_hyds(Sim& s) {
    const int j = s.ihru;
    const hydrologic_response_unit& h = s.hru(j);
    object_connectivity& o = s.ob(s.icmd);
    const float cnv_m3 = h.area_ha * 10.0f;
    const float cnv_kg = h.area_ha;
    for (int k = 1; k <= 5; ++k) o.hd(k) = s.hz;
    o.hdsep.flo_surq = s.qday * cnv_m3;
    o.hdsep.flo_satexsw = s.satexq_chan * cnv_m3;
    o.peakrate = s.qp_cms;
    hyd_output& sur = o.hd(3);
    sur.temp = 5.0f + 0.75f * s.w.tave;
    sur.flo = s.qday * cnv_m3;
    sur.sed = s.sedyld(j);
    sur.orgn = s.sedorgn(j) * cnv_kg;
    sur.sedp = (s.sedorgp(j) + s.sedminpa(j) + s.sedminps(j)) * cnv_kg;
    sur.no3 = s.surqno3(j) * cnv_kg;
    sur.solp = (s.surqsolp(j) + s.hls_d(j).tilelabp) * cnv_kg;
    sur.chla = s.chl_a(j) * cnv_kg;
    sur.nh3 = 0.0f;
    sur.no2 = 0.0f;
    const float cnv_ppm = sur.flo > 0.01f ? 1.0f / (1000.0f * sur.flo) : 0.0f;
    sur.cbod = s.cbodu(j) * cnv_ppm;
    sur.dox = s.doxq(j) * cnv_ppm;
    sur.san = s.sanyld(j);
    sur.sil = s.silyld(j);
    sur.cla = s.clayld(j);
    sur.sag = s.sagyld(j);
    sur.lag = s.lagyld(j);
    o.hd(2).flo = s.sepbtm(j) * cnv_m3;
    o.hd(2).no3 = s.percn(j) * cnv_kg;
    o.hd(4).flo = s.latq(j) * cnv_m3;
    o.hdsep.flo_latq = s.latq(j) * cnv_m3;
    o.hd(4).no3 = s.latno3(j) * cnv_kg;
    o.hd(5).flo = s.qtile * cnv_m3;
    o.hd(5).no3 = s.tileno3(j) * cnv_kg;
    for (int ihyd = 3; ihyd <= 5; ++ihyd) o.hd(1) = o.hd(1) + o.hd(ihyd);
    o.hyd_flo(o.day_cur, 1) = o.hd(1).flo;
}

}  // namespace

// sim_initday.f90
void sim_initday(Sim& s) {
    for (Vec<float>* v : {&s.cbodu, &s.chl_a, &s.cnday, &s.doxq, &s.latno3, &s.latq, &s.nplnt, &s.percn, &s.pplnt,
                          &s.qdr, &s.sedminpa, &s.sedminps, &s.sedorgn, &s.sedorgp, &s.sedyld, &s.sanyld, &s.silyld,
                          &s.clayld, &s.sagyld, &s.lagyld, &s.sepbtm, &s.surfq, &s.surqno3, &s.surqsolp, &s.tileno3,
                          &s.gwsoilq, &s.gwsoiln, &s.gwsoilp, &s.satexq, &s.satexn, &s.ubnrunoff, &s.ubntss,
                          &s.sol_sumno3, &s.sol_sumsolp})
        v->fill(0.0f);
    s.hhsurfq.fill(0.0f);
    for (int j = 1; j <= s.sp_ob.hru; ++j) {
        for (int ly = 1; ly <= s.soil(j).nly; ++ly) {
            s.sol_sumno3(j) = s.sol_sumno3(j) + s.soil1(j).mn(ly).no3 + s.soil1(j).mn(ly).nh4;
            s.sol_sumsolp(j) = s.sol_sumsolp(j) + s.soil1(j).mp(ly).lab;
        }
    }
}

// hru_control.f90 (the parts of the port; refused features stop at set-up)
void hru_control(Sim& s) {
    const int j = s.ihru;
    hydrologic_response_unit& h = s.hru(j);
    soil_profile& so = s.soil(j);
    plant_community& pc = s.pcom(j);
    float sw_volume_begin = 0.0f;
    for (int j1 = 1; j1 <= so.nly; ++j1) sw_volume_begin = sw_volume_begin + so.phys(j1).st;
    for (int ly = 1; ly <= so.nly; ++ly) s.soil1(j).emix(ly) = 0.0f;
    const int iob = h.obj_no;
    s.iwst = s.ob(iob).wst;
    s.iwgen = s.wst(s.iwst).wco.wgn;
    const int iwgn = s.wst(s.iwst).wco.wgn;
    const int isched = h.mgt_ops;
    s.w = s.wst(s.iwst).weat;
    if (s.bsn_cc.lapse == 1) {
        s.w.precip = s.w.precip + s.ob(iob).plaps;
        s.w.precip = fmax2(0.0f, s.w.precip);
        s.w.tmax = s.w.tmax + s.ob(iob).tlaps;
        s.w.tmin = s.w.tmin + s.ob(iob).tlaps;
        s.w.tave = s.w.tave + s.ob(iob).tlaps;
    }
    s.precip_eff = s.w.precip;
    s.hsc_d(j) = s.hscz;
    s.hrc_d(j) = s.hrcz;
    s.hpc_d(j) = s.hpcz;
    s.hscf_d(j) = s.hscfz;
    h.water_seep = 0.0f;
    h.water_evap = 0.0f;
    s.hnb_d(j).nuptake = 0.0f;
    s.hnb_d(j).puptake = 0.0f;
    s.hwb_d(j).wet_out = 0.0f;
    s.hnb_d(j).denit = 0.0f;
    varinit(s);
    s.nd_30 = s.nd_30 + 1;
    if (s.nd_30 > 30) s.nd_30 = 1;
    s.ht1 = s.hz;
    s.ht2 = s.hz;
    hru_dtbl_control(s);
    pc.days_plant = pc.days_plant + 1;
    pc.days_harv = pc.days_harv + 1;
    pc.days_kill = pc.days_kill + 1;
    pc.days_irr = pc.days_irr + 1;
    if (s.w.tave > 0.0f && s.wgn_pms(iwgn).phutot > 0.01f)
        s.phubase(j) = s.phubase(j) + s.w.tave / s.wgn_pms(iwgn).phutot;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        plant_stress& st = pc.plstr(s.ipl);
        st.strsw = 1.0f;
        st.strst = 1.0f;
        st.strsn = 1.0f;
        st.strsp = 1.0f;
        st.strsa = 1.0f;
        st.strss = 1.0f;
    }
    albedo(s);
    stmp_solt(s);
    sq_canopyint(s);
    sq_snom(s);
    const object_connectivity& o = s.ob(s.icmd);
    if (o.hin_sur.flo > 1.e-6f || o.hin_lat.flo > 0.0f || o.hin_aqu.flo > 0.0f)
        throw std::runtime_error("SWAT+ inflows to HRUs (run-on, lateral or aquifer) are not supported yet");
    et_pot(s);
    et_act(s);
    if (s.yr_skip(j) == 0) mgt_operatn(s);
    surface(s);
    s.sedyld(j) = s.sedyld(j) + s.ht2.sed;
    s.ht2 = s.hz;
    float wet_outflow = 0.0f;
    s.ht2.flo = s.wet(j).flo * h.area_ha * 10.0f;
    s.wet(j).flo = 0.0f;
    s.inflpcp = s.precip_eff - s.surfq(j);
    s.inflpcp = fmax2(0.0f, s.inflpcp);
    swr_percmain(s);
    s.bss_ex(1, j) = s.bss_ex(1, j) + s.satexq(j);
    s.satexq_chan = s.bss_ex(1, j) * s.brt(j);
    s.bss_ex(1, j) = s.bss_ex(1, j) - s.satexq_chan;
    rsd_decomp(s);
    nut_nminrl(s);
    nut_nitvol(s);
    if (s.bsn_cc.sol_p_model == 1) throw std::runtime_error("SWAT+ soil P model 1 (Vadas and White) is not supported");
    nut_pminrl(s);
    pl_community(s);
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        plant_status& cur = pc.plcur(s.ipl);
        if (cur.mseas == "y") {
            const int idp = cur.idplt;
            if (s.pldb(idp).trig == "moisture_gro" && s.wgn_pms(iwgn).p_pet_rto > 0.5f) {
                cur.phuacc = 0.0f;
                cur.gro = "y";
                cur.idorm = "n";
                cur.mseas = "n";
            }
        }
    }
    pl_grow(s);
    if (s.time.end_yr == 1) {
        for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
            s.pl_mass(j).yield_yr(s.ipl) = s.plt_mass_z;
            pc.plcur(s.ipl).harv_num_yr = 0;
        }
    }
    float strsw_av = 0.0f, strsa_av = 0.0f, strsn_av = 0.0f, strsp_av = 0.0f, strstmp_av = 0.0f, strss_av = 0.0f;
    int npl_gro = 0;
    for (s.ipl = 1; s.ipl <= pc.npl; ++s.ipl) {
        const plant_status& cur = pc.plcur(s.ipl);
        if (cur.gro == "y" && cur.idorm == "n" && cur.phuacc <= 1.0f) {
            const plant_stress& st = pc.plstr(s.ipl);
            ++npl_gro;
            strsw_av = strsw_av + (1.0f - st.strsw);
            strsa_av = strsa_av + (1.0f - st.strsa);
            strsn_av = strsn_av + (1.0f - st.strsn);
            strsp_av = strsp_av + (1.0f - st.strsp);
            strss_av = strss_av + (1.0f - st.strss);
            strstmp_av = strstmp_av + (1.0f - st.strst);
        }
    }
    if (npl_gro > 0) {
        const float n = static_cast<float>(npl_gro);
        strsw_av = strsw_av / n;
        strsa_av = strsa_av / n;
        strsn_av = strsn_av / n;
        strsp_av = strsp_av / n;
        strstmp_av = strstmp_av / n;
    }
    so.sw_300 = 0.0f;
    for (int ly = 1; ly <= so.nly; ++ly) {
        const float dep = ly == 1 ? 0.0f : so.phys(ly - 1).d;
        if (so.phys(ly).d >= 300.0f) {
            so.sw_300 = so.sw_300 + so.phys(ly).st * (300.0f - dep) / so.phys(ly).thick;
            break;
        }
        so.sw_300 = so.sw_300 + so.phys(ly).st;
    }
    for (int ly = 1; ly <= so.nly; ++ly) so.phys(ly).tot_sw = (so.phys(ly).st / so.phys(ly).thick) + so.phys(ly).wp;
    s.etday = s.ep_day + s.es_day + s.canev;
    if (s.surfq(j) > 0.0f && s.qp_cms > 1.e-6f) {
        if (s.precip_eff > 0.0f) {
            pest_enrsb(s);
            nut_orgn(s);
            nut_psed(s);
        }
    }
    // nut_nrain: atmospheric deposition is refused at set-up
    nut_nlch(s);
    nut_solp(s);
    if (h.luse.urb_lu > 0) hru_urban(s);
    swr_latsed(s);
    stor_surfstor(s);
    swr_substor(s);
    if (s.ht2.flo > 0.0f) {
        wet_outflow = s.ht2.flo / h.area_ha / 10.0f;
        s.qday = s.qday + wet_outflow;
        s.qdr(j) = s.qdr(j) + wet_outflow;
        s.ht2.flo = 0.0f;
    }
    sq_surfst(s);
    s.qdr(j) = s.qday + s.latq(j) + s.qtile;
    if (s.qdr(j) < 0.0f) s.qdr(j) = 0.0f;
    swr_subwq(s);
    int iob_out = iob;
    if (s.ob(iob).ru_tot > 0) iob_out = s.sp_ob1.ru + s.ob(iob).ru(1) - 1;
    output_waterbal& wb = s.hwb_d(j);
    wb.surq_cha = 0.0f;
    wb.latq_cha = 0.0f;
    wb.surq_res = 0.0f;
    wb.latq_res = 0.0f;
    wb.surq_ls = 0.0f;
    wb.latq_ls = 0.0f;
    const object_connectivity& oo = s.ob(iob_out);
    for (int iout = 1; iout <= oo.src_tot; ++iout) {
        const std::string& t = oo.obtyp_out(iout);
        const std::string& ht = oo.htyp_out(iout);
        const float fr = oo.frac_out(iout);
        const bool sur = ht == "sur" || ht == "tot";
        const bool lat = ht == "lat" || ht == "tot";
        if (t == "cha" || t == "sdc") {
            if (sur) wb.surq_cha = wb.surq_cha + s.surfq(j) * fr;
            if (lat) wb.latq_cha = wb.latq_cha + s.latq(j) * fr;
        } else if (t == "res") {
            if (sur) wb.surq_res = wb.surq_res + s.surfq(j) * fr;
            if (lat) wb.latq_res = wb.latq_res + s.latq(j) * fr;
        } else if (t == "hru" || t == "ru" || t == "hlt") {
            if (sur) wb.surq_ls = wb.surq_ls + s.surfq(j) * fr;
            if (lat) wb.latq_ls = wb.latq_ls + s.latq(j) * fr;
        }
    }
    wb.precip = s.w.precip;
    wb.snofall = s.snofall;
    wb.snomlt = s.snomlt;
    wb.surq_gen = s.qday;
    wb.latq = s.latq(j);
    wb.wateryld = s.qdr(j);
    wb.perc = s.sepbtm(j);
    wb.et = s.etday;
    wb.ecanopy = s.canev;
    wb.eplant = s.ep_day;
    wb.esoil = s.es_day;
    wb.wet_evap = h.water_evap;
    wb.wet_out = wet_outflow;
    wb.wet_stor = s.wet(j).flo / (10.0f * h.area_ha);
    wb.surq_cont = s.surfq(j);
    wb.cn = s.cnday(j);
    wb.sw = so.sw;
    wb.sw_final = so.sw;
    wb.sw_300 = so.sw_300;
    wb.snopack = h.sno_mm;
    wb.pet = s.pet_day;
    wb.qtile = s.qtile;
    wb.irr = s.irrig(j).applied;
    wb.surq_runon = s.ls_overq;
    wb.latq_runon = s.latqrunon;
    wb.overbank = h.wet_obank_in;
    h.wet_obank_in = 0.0f;
    wb.gwsoil = s.gwsoilq(j);
    wb.satex = s.satexq(j);
    wb.satex_chan = s.satexq_chan;
    wb.delsw = so.sw - sw_volume_begin;
    wb.lagsurf = s.surf_bs(1, j);
    wb.laglatq = s.bss(1, j);
    wb.lagsatex = s.bss_ex(1, j);
    s.satexq(j) = 0.0f;
    output_nutbal& nb = s.hnb_d(j);
    nb.grazn = s.grazn;
    nb.grazp = s.grazp;
    nb.fertn = s.fertn;
    nb.fertp = s.fertp;
    nb.fixn = s.fixn;
    nb.gwsoiln = s.gwsoiln(j);
    nb.gwsoilp = s.gwsoilp(j);
    output_plantweather& pw = s.hpw_d(j);
    plant_community_mass& pm = s.pl_mass(j);
    pw.lai = pc.lai_sum;
    pw.bioms = pm.tot_com.m;
    if (pm.tot_com.m < 0.0f) pm.tot_com.m = 0.0f;
    pw.residue = pm.rsd_tot.m;
    pw.yield = s.pl_yield.m;
    s.pl_yield = s.plt_mass_z;
    pw.sol_tmp = so.phys(2).tmp;
    pw.strsw = strsw_av;
    pw.strsa = strsa_av;
    pw.strstmp = strstmp_av;
    pw.strsn = strsn_av;
    pw.strsp = strsp_av;
    pw.strss = strss_av;
    pw.nplnt = pm.tot_com.n;
    pw.percn = s.percn(j);
    pw.pplnt = pm.tot_com.p;
    pw.tmx = s.w.tmax;
    pw.tmn = s.w.tmin;
    pw.tmpav = s.w.tave;
    pw.solrad = s.w.solrad;
    pw.wndspd = s.w.windsp;
    pw.rhum = s.w.rhum;
    pw.phubase0 = s.phubase(j);
    if (s.time.yrs > s.pco.nyskip) s.bsn_sedbud.upland_t = s.bsn_sedbud.upland_t + s.sedyld(j);
    output_losses& ls = s.hls_d(j);
    ls.sedyld = s.sedyld(j) / h.area_ha;
    ls.sedorgn = s.sedorgn(j);
    ls.sedorgp = s.sedorgp(j);
    ls.surqno3 = s.surqno3(j);
    ls.latno3 = s.latno3(j);
    ls.surqsolp = s.surqsolp(j);
    ls.usle = s.usle;
    ls.sedminp = s.sedminpa(j) + s.sedminps(j);
    ls.tileno3 = s.tileno3(j);
    hru_hyds(s);
}

}  // namespace swat
