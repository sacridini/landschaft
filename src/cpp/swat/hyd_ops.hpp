// SPDX-License-Identifier: GPL-3.0-or-later
//
// The hydrograph arithmetic of hydrograph_module.f90 and sd_channel_module.f90 (their
// `contains` sections): Fortran's operator(+) = hydout_add, (-) = hydout_subtract,
// (**) = hydout_mult (component by component), (*) = hydout_mult_const, (/) =
// hydout_div_const, (//) = hydout_div_conv, .add. = hydout_add_const; chrc_mult and
// chrc_interp for the rating curves. Results start from the types' default values, as
// a Fortran function result of a type with default initialisation does.

#pragma once

#include "swat/fmath.hpp"
#include "swat/modules.hpp"

namespace swat {

inline hyd_output operator+(const hyd_output& a, const hyd_output& b) {
    hyd_output c;
    c.flo = a.flo + b.flo;
    c.sed = a.sed + b.sed;
    c.orgn = a.orgn + b.orgn;
    c.sedp = a.sedp + b.sedp;
    c.no3 = a.no3 + b.no3;
    c.solp = a.solp + b.solp;
    c.chla = a.chla + b.chla;
    c.nh3 = a.nh3 + b.nh3;
    c.no2 = a.no2 + b.no2;
    c.cbod = a.cbod + b.cbod;
    c.dox = a.dox + b.dox;
    c.san = a.san + b.san;
    c.sil = a.sil + b.sil;
    c.cla = a.cla + b.cla;
    c.sag = a.sag + b.sag;
    c.lag = a.lag + b.lag;
    c.grv = a.grv + b.grv;
    if (a.flo + b.flo > 1.e-6f) {
        c.temp = (a.flo * a.temp + b.flo * b.temp) / (a.flo + b.flo);
    } else {
        c.temp = 0.0f;
    }
    return c;
}

inline hyd_output operator-(const hyd_output& a, const hyd_output& b) {
    hyd_output c;
    c.flo = a.flo - b.flo;
    c.sed = a.sed - b.sed;
    c.orgn = a.orgn - b.orgn;
    c.sedp = a.sedp - b.sedp;
    c.no3 = a.no3 - b.no3;
    c.solp = a.solp - b.solp;
    c.chla = a.chla - b.chla;
    c.nh3 = a.nh3 - b.nh3;
    c.no2 = a.no2 - b.no2;
    c.cbod = a.cbod - b.cbod;
    c.dox = a.dox - b.dox;
    c.san = a.san - b.san;
    c.sil = a.sil - b.sil;
    c.cla = a.cla - b.cla;
    c.sag = a.sag - b.sag;
    c.lag = a.lag - b.lag;
    c.grv = a.grv - b.grv;
    c.temp = a.temp;
    return c;
}

// Fortran's hyd1 ** hyd2 (hydout_mult)
inline hyd_output hyd_mult(const hyd_output& a, const hyd_output& b) {
    hyd_output c;
    c.flo = a.flo * b.flo;
    c.sed = a.sed * b.sed;
    c.orgn = a.orgn * b.orgn;
    c.sedp = a.sedp * b.sedp;
    c.no3 = a.no3 * b.no3;
    c.solp = a.solp * b.solp;
    c.chla = a.chla * b.chla;
    c.nh3 = a.nh3 * b.nh3;
    c.no2 = a.no2 * b.no2;
    c.cbod = a.cbod * b.cbod;
    c.dox = a.dox * b.dox;
    c.san = a.san * b.san;
    c.sil = a.sil * b.sil;
    c.cla = a.cla * b.cla;
    c.sag = a.sag * b.sag;
    c.lag = a.lag * b.lag;
    c.grv = a.grv * b.grv;
    c.temp = a.temp;
    return c;
}

// const * hyd (hydout_mult_const)
inline hyd_output operator*(float k, const hyd_output& a) {
    hyd_output c;
    c.temp = a.temp;
    c.flo = k * a.flo;
    c.sed = k * a.sed;
    c.orgn = k * a.orgn;
    c.sedp = k * a.sedp;
    c.no3 = k * a.no3;
    c.solp = k * a.solp;
    c.chla = k * a.chla;
    c.nh3 = k * a.nh3;
    c.no2 = k * a.no2;
    c.cbod = k * a.cbod;
    c.dox = k * a.dox;
    c.san = k * a.san;
    c.sil = k * a.sil;
    c.cla = k * a.cla;
    c.sag = k * a.sag;
    c.lag = k * a.lag;
    c.grv = k * a.grv;
    c.temp = a.temp;
    return c;
}

// hyd / const (hydout_div_const)
inline hyd_output operator/(const hyd_output& a, float k) {
    hyd_output c;
    c.flo = a.flo / k;
    c.sed = a.sed / k;
    c.orgn = a.orgn / k;
    c.sedp = a.sedp / k;
    c.no3 = a.no3 / k;
    c.solp = a.solp / k;
    c.chla = a.chla / k;
    c.nh3 = a.nh3 / k;
    c.no2 = a.no2 / k;
    c.cbod = a.cbod / k;
    c.dox = a.dox / k;
    c.san = a.san / k;
    c.sil = a.sil / k;
    c.cla = a.cla / k;
    c.sag = a.sag / k;
    c.lag = a.lag / k;
    c.grv = a.grv / k;
    c.temp = a.temp;
    return c;
}

// const .add. hyd (hydout_add_const)
inline hyd_output add_const(float k, const hyd_output& a) {
    hyd_output c;
    c.flo = k + a.flo;
    c.sed = k + a.sed;
    c.orgn = k + a.orgn;
    c.sedp = k + a.sedp;
    c.no3 = k + a.no3;
    c.solp = k + a.solp;
    c.chla = k + a.chla;
    c.nh3 = k + a.nh3;
    c.no2 = k + a.no2;
    c.cbod = k + a.cbod;
    c.dox = k + a.dox;
    c.san = k + a.san;
    c.sil = k + a.sil;
    c.cla = k + a.cla;
    c.sag = k + a.sag;
    c.lag = k + a.lag;
    c.grv = k + a.grv;
    c.temp = a.temp;
    return c;
}

// hyd_convert_conc_to_mass
inline void hyd_convert_conc_to_mass(hyd_output& h) {
    h.sed = h.sed * h.flo / 1000000.0f;
    h.orgn = h.orgn * h.flo / 1000.0f;
    h.sedp = h.sedp * h.flo / 1000.0f;
    h.no3 = h.no3 * h.flo / 1000.0f;
    h.solp = h.solp * h.flo / 1000.0f;
    h.chla = h.chla * h.flo / 1000.0f;
    h.nh3 = h.nh3 * h.flo / 1000.0f;
    h.no2 = h.no2 * h.flo / 1000.0f;
    h.cbod = h.cbod * h.flo / 1000.0f;
    h.dox = h.dox * h.flo / 1000.0f;
    h.san = h.san * h.flo / 1000000.0f;
    h.sil = h.sil * h.flo / 1000000.0f;
    h.cla = h.cla * h.flo / 1000000.0f;
    h.sag = h.sag * h.flo / 1000000.0f;
    h.lag = h.lag * h.flo / 1000000.0f;
    h.grv = h.grv * h.flo / 1000000.0f;
}

// rating curve * const (chrc_mult)
inline channel_rating_curve_parameters operator*(const channel_rating_curve_parameters& a, float k) {
    channel_rating_curve_parameters c;
    c.xsec_area = a.xsec_area * k;
    c.surf_area = a.surf_area * k;
    c.flo_rate = a.flo_rate * k;
    c.dep = a.dep * k;
    c.top_wid = a.top_wid * k;
    c.vol = a.vol * k;
    c.vol_ch = a.vol_ch * k;
    c.vol_fp = a.vol_fp * k;
    c.wet_perim = a.wet_perim * k;
    c.vel = a.vel * k;
    c.ttime = a.ttime * k;
    return c;
}

// chrc_interp (rci is intent(out): default values first)
inline void chrc_interp(const channel_rating_curve_parameters& a, const channel_rating_curve_parameters& b, float k,
                        channel_rating_curve_parameters& r) {
    r = channel_rating_curve_parameters();
    r.xsec_area = a.xsec_area + k * (b.xsec_area - a.xsec_area);
    r.surf_area = a.surf_area + k * (b.surf_area - a.surf_area);
    r.flo_rate = a.flo_rate + k * (b.flo_rate - a.flo_rate);
    r.dep = a.dep + k * (b.dep - a.dep);
    r.top_wid = a.top_wid + k * (b.top_wid - a.top_wid);
    r.vol = a.vol + k * (b.vol - a.vol);
    r.vol_ch = a.vol_ch + k * (b.vol_ch - a.vol_ch);
    r.vol_fp = a.vol_fp + k * (b.vol_fp - a.vol_fp);
    r.wet_perim = a.wet_perim + k * (b.wet_perim - a.wet_perim);
    r.vel = a.vel + k * (b.vel - a.vel);
    r.ttime = a.ttime + k * (b.ttime - a.ttime);
}

}  // namespace swat
