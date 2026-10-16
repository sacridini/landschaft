// SPDX-License-Identifier: GPL-3.0-or-later
//
// The operators of organic_mineral_mass_module.f90 (om_add1, om_subtract, om_mult_const,
// om_divide, nmin_add, nmin_mult_const, pmin_add, pmin_mult_const).

#pragma once

#include "swat/modules.hpp"

namespace swat {

inline organic_mass operator+(const organic_mass& a, const organic_mass& b) {
    organic_mass r;
    r.m = a.m + b.m;
    r.c = a.c + b.c;
    r.n = a.n + b.n;
    r.p = a.p + b.p;
    return r;
}

inline organic_mass operator-(const organic_mass& a, const organic_mass& b) {
    organic_mass r;
    r.m = a.m - b.m;
    r.c = a.c - b.c;
    r.n = a.n - b.n;
    r.p = a.p - b.p;
    return r;
}

inline organic_mass operator*(float k, const organic_mass& a) {
    organic_mass r;
    r.m = k * a.m;
    r.c = k * a.c;
    r.n = k * a.n;
    r.p = k * a.p;
    return r;
}

inline organic_mass operator/(const organic_mass& a, float k) {
    organic_mass r;
    r.m = a.m / k;
    r.c = a.c / k;
    r.n = a.n / k;
    r.p = a.p / k;
    return r;
}

inline mineral_nitrogen operator+(const mineral_nitrogen& a, const mineral_nitrogen& b) {
    mineral_nitrogen r;
    r.no3 = a.no3 + b.no3;
    r.nh4 = a.nh4 + b.nh4;
    return r;
}

inline mineral_nitrogen operator*(float k, const mineral_nitrogen& a) {
    mineral_nitrogen r;
    r.no3 = k * a.no3;
    r.nh4 = k * a.nh4;
    return r;
}

inline mineral_phosphorus operator+(const mineral_phosphorus& a, const mineral_phosphorus& b) {
    mineral_phosphorus r;
    r.wsol = a.wsol + b.wsol;
    r.lab = a.lab + b.lab;
    r.act = a.act + b.act;
    r.sta = a.sta + b.sta;
    return r;
}

inline mineral_phosphorus operator*(float k, const mineral_phosphorus& a) {
    mineral_phosphorus r;
    r.wsol = k * a.wsol;
    r.lab = k * a.lab;
    r.act = k * a.act;
    r.sta = k * a.sta;
    return r;
}

}  // namespace swat
