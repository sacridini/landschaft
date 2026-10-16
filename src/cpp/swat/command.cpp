// SPDX-License-Identifier: GPL-3.0-or-later
//
// The daily command loop over the spatial objects (command.f90), routing units
// (ru_control.f90) and their delivery ratios (dr_ru.f90). The end-of-day output
// writers, constituents, the hydrograph separation (hdsep) and sub-daily routing are
// outside the port.

#include <cstring>
#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/hyd_ops.hpp"
#include "swat/swat.hpp"

namespace swat {

namespace {

hyd_output from_floats(const float* v) {
    static_assert(sizeof(hyd_output) == 18 * sizeof(float), "hyd_output is 18 floats");
    hyd_output h;
    std::memcpy(&h, v, sizeof(hyd_output));
    return h;
}

std::vector<float> to_floats(const hyd_output* h, int n) {
    std::vector<float> v(static_cast<std::size_t>(18 * n));
    if (n > 0) std::memcpy(v.data(), h, sizeof(hyd_output) * static_cast<std::size_t>(n));
    return v;
}

template <class T>
std::vector<float> struct_floats(const T& x) {
    static_assert(sizeof(T) % sizeof(float) == 0, "a struct of floats");
    std::vector<float> v(sizeof(T) / sizeof(float));
    std::memcpy(v.data(), &x, sizeof(T));
    return v;
}

// The reference's outputs of a HRU object (tests of the routing without hru_control):
// what hru_hyds leaves in ob(icmd) for the objects downstream, and hwb_d for ch_temp.
void inject_hru(Sim& s, int ihru) {
    object_connectivity& o = s.ob(s.icmd);
    const InjectKey key{s.time.yrc, s.time.day, s.icmd};
    const auto it = s.inject_hd.find(key);
    if (it == s.inject_hd.end()) throw std::runtime_error("no injected hydrograph for a HRU object");
    for (int ihyd = 1; ihyd <= o.nhyds; ++ihyd)
        o.hd(ihyd) = from_floats(it->second.data() + 18 * (ihyd - 1));
    o.hyd_flo(o.day_cur, 1) = o.hd(1).flo;
    const auto wb = s.inject_hwb.find(InjectKey{s.time.yrc, s.time.day, ihru});
    if (s.hwb_d.size() < s.sp_ob.hru) s.hwb_d.alloc(1, s.sp_ob.hru, output_waterbal());
    if (wb != s.inject_hwb.end()) {
        static_assert(sizeof(output_waterbal) % sizeof(float) == 0, "output_waterbal is floats");
        std::memcpy(&s.hwb_d(ihru), wb->second.data(),
                    std::min(sizeof(output_waterbal), wb->second.size() * sizeof(float)));
    }
}

}  // namespace

// dr_ru.f90
void dr_ru(Sim& s) {
    for (s.iru = 1; s.iru <= s.sp_ob.ru; ++s.iru) {
        const int iru = s.iru;
        for (int ii = 1; ii <= s.ru_def(iru).num_tot; ++ii) {
            const int ielem = s.ru_def(iru).num(ii);
            routing_unit_elements& e = s.ru_elem(ielem);
            s.ihru = e.obtypno;
            float rto = 0.0f;
            if (e.dr_name == "calc" || e.dr_name == "0") {
                if (e.obtyp == "hru") {
                    rto = s.tconc(s.ihru) / s.ru_tc(iru);
                } else if (e.obtyp == "sdc" || e.obtyp == "ru") {
                    rto = 1.0f;
                } else if (e.obtyp == "hlt") {
                    throw std::runtime_error("SWAT+ hru_lte objects are not supported");
                }
                rto = fmin2(1.0f, fpow(rto, 0.5f));
                e.dr = add_const(rto, s.hz);
                e.dr.flo = 1.0f;
            }
            if (e.dr_name == "full" || e.dr_name == "1") e.dr = add_const(1.0f, s.hz);
        }
    }
}

// ru_control.f90 (daily step)
void ru_control(Sim& s) {
    const int iru = s.iru;
    object_connectivity& o = s.ob(s.icmd);
    s.ru_d(iru) = s.hz;
    for (int k = 1; k <= 5; ++k) o.hd(k) = s.hz;
    const int day_cur = o.day_cur;
    for (int ielem = 1; ielem <= s.ru_def(iru).num_tot; ++ielem) {
        const int ise = s.ru_def(iru).num(ielem);
        const routing_unit_elements& e = s.ru_elem(ise);
        const int iob = e.obj;
        s.ihru = e.obtypno;
        s.ht1 = s.hz;
        s.ht2 = s.hz;
        s.ht3 = s.hz;
        s.ht4 = s.hz;
        s.ht5 = s.hz;
        s.delrto = s.hz;
        s.delrto = e.dr;
        if (e.obtyp == "exc") throw std::runtime_error("SWAT+ export coefficients are not supported");
        float ef = e.frac;
        const object_connectivity& oe = s.ob(iob);
        if (oe.typ == "hru" || oe.typ == "hru_lte") {
            if (ef < 0.99999f) ef = ef * s.ru(iru).da_km2 / (oe.area_ha / 100.0f);
        }
        for (int ihtypno = 1; ihtypno <= oe.nhyds; ++ihtypno) {
            if (ihtypno != 2) {
                s.ht1 = hyd_mult(oe.hd(ihtypno), s.delrto);
            } else {
                s.ht1 = oe.hd(ihtypno);
            }
            s.ht1 = ef * s.ht1;
            o.hd(ihtypno) = o.hd(ihtypno) + s.ht1;
            s.ru_d(iru) = s.ru_d(iru) + s.ht1;
        }
    }
    if (s.time.step == 1) o.hyd_flo(day_cur, 1) = o.hd(1).flo;
}

// command.f90
void command(Sim& s) {
    s.icmd = s.sp_ob1.objs;
    while (s.icmd != 0) {
        object_connectivity& o = s.ob(s.icmd);
        if (o.typ == "hru" || o.typ == "ru") {
            o.day_cur = o.day_cur + 1;
            if (o.day_cur > o.day_max) o.day_cur = 1;
        } else {
            o.day_cur = 1;
            o.day_cur = o.day_cur + 1;
            if (o.day_cur > o.day_max) o.day_cur = 1;
        }
        o.hin = s.hz;
        o.hin_sur = s.hz;
        o.hin_lat = s.hz;
        o.hin_til = s.hz;
        s.ht1 = s.hz;
        o.tsin.fill(0.0f);
        o.peakrate = 0.0f;
        std::vector<float> hyd_flo(static_cast<std::size_t>(s.time.step), 0.0f);
        const bool landscape = o.typ == "hru" || o.typ == "ru" || o.typ == "hru_lte";
        if (o.rcv_tot > 0) {
            for (int in = 1; in <= o.rcv_tot; ++in) {
                const int iob = o.obj_in(in);
                const int ihyd = o.ihtyp_in(in);
                const float frac_in = o.frac_in(in);
                const object_connectivity& oi = s.ob(iob);
                o.peakrate = oi.peakrate;
                if (landscape) {
                    const std::string& t = o.obtyp_in(in);
                    if (t == "hru" || t == "ru" || t == "hru_lte") {
                        if (o.htyp_in(in) == "tot") {
                            o.hin_sur = o.hin_sur + frac_in * oi.hd(3);
                            o.hin_til = o.hin_til + frac_in * oi.hd(5);
                            o.hin_lat = o.hin_lat + frac_in * oi.hd(4);
                        } else {
                            const std::string& h = o.htyp_in(in);
                            if (h == "tot" || h == "sur") {
                                o.hin_sur = o.hin_sur + frac_in * oi.hd(ihyd);
                            } else if (h == "lat") {
                                o.hin_lat = o.hin_lat + frac_in * oi.hd(ihyd);
                            } else if (h == "til") {
                                o.hin_til = o.hin_til + frac_in * oi.hd(ihyd);
                            } else if (h == "aqu") {
                                o.hin_aqu = o.hin_aqu + frac_in * oi.hd(ihyd);
                            }
                        }
                    } else {
                        o.hin_sur = o.hin_sur + frac_in * oi.hd(1);
                    }
                } else {
                    s.ht1 = frac_in * oi.hd(ihyd);
                    o.hin = o.hin + s.ht1;
                    o.hin_d(in) = s.ht1;
                }
                // sub-daily inflow hydrographs
                const int iday = oi.day_cur;
                if (oi.typ == "hru" || oi.typ == "ru") {
                    const std::string& h = o.htyp_in(in);
                    const float step = static_cast<float>(s.time.step);
                    for (int k = 1; k <= s.time.step; ++k) {
                        float& v = hyd_flo[static_cast<std::size_t>(k - 1)];
                        if (h == "tot") {
                            v = oi.hyd_flo(iday, k) + (oi.hd(4).flo + oi.hd(5).flo) / step;
                        } else if (h == "sur") {
                            v = oi.hyd_flo(iday, k);
                        } else if (h == "rhg") {
                            v = oi.hd(2).flo / step;
                        } else if (h == "lat") {
                            v = oi.hd(4).flo / step;
                        } else if (h == "til") {
                            v = oi.hd(5).flo / step;
                        }
                    }
                }
                {
                    const std::string& t = oi.typ;
                    const float step = static_cast<float>(s.time.step);
                    for (int k = 1; k <= s.time.step; ++k) {
                        float& v = hyd_flo[static_cast<std::size_t>(k - 1)];
                        if (t == "aqu" || t == "res" || t == "outlet") {
                            v = oi.hd(ihyd).flo / step;
                        } else if (t == "chandeg") {
                            v = oi.hyd_flo(1, k);
                        }
                    }
                }
                for (int k = 1; k <= s.time.step; ++k) {
                    float& v = hyd_flo[static_cast<std::size_t>(k - 1)];
                    v = frac_in * v;
                    o.tsin(k) = o.tsin(k) + v;
                }
            }
            if (o.typ == "hru" || o.typ == "ru") {
                const float conv = o.area_ha;
                o.hin_sur = o.hin_sur / conv;
                o.hin_sur.flo = o.hin_sur.flo / 10.0f;
                o.hin_lat = o.hin_lat / conv;
                o.hin_lat.flo = o.hin_lat.flo / 10.0f;
                o.hin_til = o.hin_til / conv;
                o.hin_til.flo = o.hin_til.flo / 10.0f;
            }
        }
        if (s.dump) {
            std::vector<float> v;
            for (const hyd_output* h : {&o.hin, &o.hin_sur, &o.hin_lat, &o.hin_til, &o.hin_aqu}) {
                const std::vector<float> w = to_floats(h, 1);
                v.insert(v.end(), w.begin(), w.end());
            }
            s.dump_record(41, s.icmd, std::move(v));
        }
        if (o.typ == "hru") {
            s.ihru = o.num;
            if (s.inject_hd.empty()) {
                hru_control(s);
                if (s.dump) {
                    s.dump_record(10, s.ihru, values(s.hwb_d(s.ihru)));
                    s.dump_record(11, s.ihru, values(s.hnb_d(s.ihru)));
                    s.dump_record(12, s.ihru, values(s.hls_d(s.ihru)));
                    s.dump_record(13, s.ihru, values(s.hpw_d(s.ihru)));
                }
            } else {
                inject_hru(s, s.ihru);
            }
        } else if (o.typ == "ru") {
            s.iru = o.num;
            ru_control(s);
        } else if (o.typ == "aqu") {
            if (o.dfn_tot == 0) {
                aqu_1d_control(s);
                if (s.dump) s.dump_record(20, o.num, struct_floats(s.aqu_d(o.num)));
            }
        } else if (o.typ == "outlet") {
            o.hd(1) = o.hin;
        } else if (o.typ == "chandeg") {
            s.isdch = o.num;
            s.isd_chsur = o.props2;
            const int isdch = s.isdch;
            if (s.sd_ch(isdch).chl > 1.e-3f) {
                sd_channel_control3(s);
                if (s.dump) s.dump_record(30, isdch, struct_floats(s.chsd_d(isdch)));
            } else {
                // artificial channel (length 0): no transformations
                o.hd(1) = o.hin;
                s.ch_in_d(isdch) = s.ht1;
                s.chsd_d(isdch).flo_in = s.ht1.flo / 86400.0f;
                s.ch_in_d(isdch).flo = s.ht1.flo / 86400.0f;
                s.ch_out_d(isdch) = s.ht1;
                s.ch_out_d(isdch).flo = s.ht1.flo / 86400.0f;
                sd_ch_output& d = s.chsd_d(isdch);
                d.flo = s.ht1.flo / 86400.0f;
                d.peakr = 0.0f;
                d.sed_in = o.hin.sed;
                d.sed_out = o.hin.sed;
                d.washld = 0.0f;
                d.bedld = 0.0f;
                d.dep = 0.0f;
                d.deg_btm = 0.0f;
                d.deg_bank = 0.0f;
                d.hc_sed = 0.0f;
                d.width = s.sd_ch(isdch).chw;
                d.depth = s.sd_ch(isdch).chd;
                d.slope = s.sd_ch(isdch).chs;
                d.deg_btm_m = 0.0f;
                d.deg_bank_m = 0.0f;
                d.hc_m = 0.0f;
            }
        }
        if (s.dump) s.dump_record(40, s.icmd, to_floats(o.hd.data(), o.hd.size()));
        s.icmd = o.cmd_next;
    }
}

}  // namespace swat
