// SPDX-License-Identifier: GPL-3.0-or-later
//
// The spatial objects and their connections: hyd_connect.f90, hyd_read_connect.f90,
// ru_read.f90, ru_read_elements.f90, define_unit_elements.f90, aqu2d_read.f90.

#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

// hyd_read_connect.f90 (the constituent hydrographs left out)
void hyd_read_connect(Sim& s, const std::string& con_file, const std::string& obtyp, int nspu1, int nspu, int nhyds,
                      int ndsave) {
    std::string titldum, header;
    int ob1 = 0, ob2 = -1;
    if (s.files.exists(con_file)) {
        fio::File f = s.files.open(con_file);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            if (nspu > 0) {
                ob1 = nspu1;
                ob2 = nspu1 + nspu - 1;
                int isp_ob = 0;
                bool stop = false;
                for (int i = ob1; i <= ob2 && !stop; ++i) {
                    object_connectivity& o = s.ob(i);
                    o.typ = obtyp;
                    o.nhyds = nhyds;
                    ++isp_ob;
                    o.sp_ob_no = isp_ob;
                    o.hd.alloc(1, nhyds, hyd_output());
                    o.hd_aa.alloc(1, nhyds, hyd_output());
                    o.trans = s.hz;
                    o.hin_tot = s.hz;
                    o.hout_tot = s.hz;
                    o.day_max = ndsave;
                    o.ts.alloc(1, o.day_max, 1, s.time.step, hyd_output());
                    o.tsin.alloc(1, s.time.step, 0.0f);
                    o.uh.alloc(1, o.day_max, 1, s.time.step, 0.0f);
                    o.hyd_flo.alloc(1, o.day_max, 1, s.time.step, 0.0f);
                    int eof = f.read(o.num, chars(o.name, o.name_len), o.gis_id, o.area_ha, o.lat, o.long_, o.elev,
                                     o.props, chars(o.wst_c, o.wst_c_len), o.constit, o.props2,
                                     chars(o.ruleset, o.ruleset_len), o.src_tot);
                    o.area_ha_calc = (o.typ == "hru" || o.typ == "ru" || o.typ == "recall") ? o.area_ha : 0.0f;
                    if (eof < 0) break;
                    if (o.src_tot > 0) {
                        const int nout = o.src_tot;
                        o.obj_out.alloc(1, nout, 0);
                        o.obtyp_out.alloc(1, nout, "");
                        o.obtypno_out.alloc(1, nout, 0);
                        o.htyp_out.alloc(1, nout, "");
                        o.ihtyp_out.alloc(1, nout, 0);
                        o.frac_out.alloc(1, nout, 0.0f);
                        o.rcvob_inhyd.alloc(1, nout, 0);
                        o.hout_m.alloc(1, nout, hyd_output());
                        o.hout_y.alloc(1, nout, hyd_output());
                        o.hout_a.alloc(1, nout, hyd_output());
                        f.backspace();
                        fio::ItemList l;
                        fio::add(l, o.num);
                        fio::add(l, chars(o.name, o.name_len));
                        fio::add(l, o.gis_id);
                        fio::add(l, o.area_ha);
                        fio::add(l, o.lat);
                        fio::add(l, o.long_);
                        fio::add(l, o.elev);
                        fio::add(l, o.props);
                        fio::add(l, chars(o.wst_c, o.wst_c_len));
                        fio::add(l, o.constit);
                        fio::add(l, o.props2);
                        fio::add(l, chars(o.ruleset, o.ruleset_len));
                        fio::add(l, o.src_tot);
                        for (int isp = 1; isp <= nout; ++isp) {
                            fio::add(l, chars(o.obtyp_out(isp), 3));
                            fio::add(l, o.obtypno_out(isp));
                            fio::add(l, chars(o.htyp_out(isp), 3));
                            fio::add(l, o.frac_out(isp));
                        }
                        eof = f.read(l);
                        if (eof < 0) stop = true;
                    } else {
                        o.obtypno_out.alloc(1, 1, 0);
                    }
                }
            }
            break;
        }
    }
    for (int i = ob1; i <= ob2; ++i) search(s.wst_n, s.db_mx.wst, s.ob(i).wst_c, s.ob(i).wst);
}

// define_unit_elements.f90: element lists such as "1 -5 8" (1 to 5, then 8)
void define_unit_elements(Sim& s, int num_elem, int& ielem) {
    const Vec<int>& ec = s.elem_cnt;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) s.defunit_num.alloc(1, ielem, 0);
        ielem = 0;
        int ii = 1;
        while (ii <= num_elem) {
            if (ii == num_elem) {
                if (ii == 1) {
                    if (pass == 1) s.defunit_num(1) = ec(1);
                    ielem = 1;
                    break;
                }
                const int ie1 = ec(ii - 1);
                int ie2 = ec(ii);
                if (ie2 > 0) {
                    ++ielem;
                    if (pass == 1) s.defunit_num(ielem) = ie2;
                } else {
                    ie2 = std::abs(ie2);
                    for (int ie = ie1; ie <= ie2; ++ie) {
                        ++ielem;
                        if (pass == 1) s.defunit_num(ielem) = ie;
                    }
                }
                ++ii;
            } else {
                const int ie1 = ec(ii);
                int ie2 = ec(ii + 1);
                if (ie2 > 0) {
                    ++ielem;
                    if (pass == 1) s.defunit_num(ielem) = ie1;
                    ++ii;
                } else {
                    ie2 = std::abs(ie2);
                    for (int ie = ie1; ie <= ie2; ++ie) {
                        ++ielem;
                        if (pass == 1) s.defunit_num(ielem) = ie;
                    }
                    ii += 2;
                }
            }
        }
    }
}

// ru_read.f90
void ru_read(Sim& s) {
    const std::string& file = s.in_ru.ru;
    std::string titldum, header;
    s.mru_db = 0;
    if (!s.files.exists(file) || file == "null") {
        s.ru.alloc(0, 0, ru_parameters());
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
            if (eof < 0 || i == 0) break;
            imax = std::max(imax, i);
            s.mru_db = s.mru_db + 1;
        }
        s.ru.alloc(0, s.sp_ob.ru, ru_parameters());
        s.ru_d.alloc(1, s.sp_ob.ru, hyd_output());
        s.ru_m.alloc(1, s.sp_ob.ru, hyd_output());
        s.ru_y.alloc(1, s.sp_ob.ru, hyd_output());
        s.ru_a.alloc(1, s.sp_ob.ru, hyd_output());
        s.ru_tc.alloc(0, s.sp_ob.ru, 0.0f);
        s.ru_n.alloc(0, s.sp_ob.ru, 0.0f);
        s.itsb.alloc(1, s.sp_ob.ru, 0);
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int iru = 1; iru <= s.mru_db; ++iru) {
            int i = 0, k = 0;
            if (f.read(i) < 0) break;
            f.backspace();
            ru_parameters& r = s.ru(i);
            if (f.read(k, chars(r.name, r.name_len), items(r.dbsc)) < 0) break;
            for (int ith = 1; ith <= s.db_mx.topo; ++ith) {
                if (fstrcmp(r.dbsc.toposub_db, s.topo_db(ith).name) == 0) {
                    r.dbs.toposub_db = ith;
                    break;
                }
            }
            for (int ith = 1; ith <= s.db_mx.field; ++ith) {
                if (fstrcmp(r.dbsc.field_db, s.field_db(ith).name) == 0) {
                    r.dbs.field_db = ith;
                    r.field.length = s.field_db(ith).length;
                    r.field.wid = s.field_db(ith).wid;
                    r.field.ang = s.field_db(ith).ang;
                    break;
                }
            }
        }
        break;
    }
}

// ru_read_elements.f90
void ru_read_elements(Sim& s) {
    std::string titldum, header, namedum;
    const std::string& efile = s.in_ru.ru_ele;
    if (s.files.exists(efile) || efile != "null") {
        fio::File f = s.files.open(efile);
        if (!f.open()) throw std::runtime_error("cannot open " + efile);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            int imax = 0, eof = 0;
            while (eof == 0) {
                int i = 0;
                eof = f.read(i);
                if (eof < 0) break;
                imax = std::max(i, imax);
            }
            s.db_mx.ru_elem = imax;
            s.ru_def.alloc(1, imax, routing_unit_data());
            s.ru_elem.alloc(1, imax, routing_unit_elements());
            s.ielem_ru.alloc(1, imax, 0);
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int isp = 1; isp <= imax; ++isp) {
                int i = 0, k = 0;
                if (f.read(i) < 0) break;
                f.backspace();
                routing_unit_elements& e = s.ru_elem(i);
                if (f.read(k, chars(e.name, e.name_len), chars(e.obtyp, e.obtyp_len), e.obtypno, e.frac,
                           chars(e.dr_name, e.dr_name_len)) < 0)
                    break;
                // the delivery ratio is looked up in delratio.del (dr_db), outside the port
                if (s.db_mx.dr_om > 0) throw std::runtime_error("SWAT+ delivery ratios (dr_om.del) are not supported");
            }
            break;
        }
    }
    const std::string& dfile = s.in_ru.ru_def;
    if (s.files.exists(dfile) || dfile != "null") {
        fio::File f = s.files.open(dfile);
        if (!f.open()) throw std::runtime_error("cannot open " + dfile);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            (void)0;
            {
                int eof = 0;
                while (eof == 0) {
                    eof = f.read(chars(titldum, 80));
                    if (eof < 0) break;
                }
            }
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            bool stop = false;
            for (int iru = 1; iru <= s.sp_ob.ru && !stop; ++iru) {
                int iob = s.sp_ob1.ru + iru - 1;
                s.ob(iob).ru_tot = 0;
                int numb = 0, nspu = 0;
                if (f.read(numb, chars(namedum, 16), nspu) < 0) break;
                routing_unit_data& d = s.ru_def(iru);
                d.num_tot = 0;
                if (nspu > 0) {
                    f.backspace();
                    s.elem_cnt.alloc(1, nspu, 0);
                    fio::ItemList l;
                    fio::add(l, numb);
                    fio::add(l, chars(d.name, d.name_len));
                    fio::add(l, nspu);
                    for (int isp = 1; isp <= nspu; ++isp) fio::add(l, s.elem_cnt(isp));
                    if (f.read(l) < 0) {
                        stop = true;
                        break;
                    }
                    int ielem1 = 0;
                    define_unit_elements(s, nspu, ielem1);
                    d.num = s.defunit_num;
                    d.num_tot = ielem1;
                    s.defunit_num = Vec<int>();
                    s.elem_cnt = Vec<int>();
                    iob = s.sp_ob1.ru + iru - 1;
                    s.ob(iob).dfn_tot = d.num_tot;
                    for (int ie = 1; ie <= d.num_tot; ++ie) {
                        const int ii = d.num(ie);
                        routing_unit_elements& e = s.ru_elem(ii);
                        const std::string& t = e.obtyp;
                        if (t == "hru") e.obj = s.sp_ob1.hru + e.obtypno - 1;
                        else if (t == "hlt") e.obj = s.sp_ob1.hru_lte + e.obtypno - 1;
                        else if (t == "ru") e.obj = s.sp_ob1.ru + e.obtypno - 1;
                        else if (t == "cha") e.obj = s.sp_ob1.chan + e.obtypno - 1;
                        else if (t == "res") e.obj = s.sp_ob1.res + e.obtypno - 1;
                        else if (t == "exc") e.obj = s.sp_ob1.exco + e.obtypno - 1;
                        else if (t == "dr") e.obj = s.sp_ob1.dr + e.obtypno - 1;
                        else if (t == "out") e.obj = s.sp_ob1.outlet + e.obtypno - 1;
                        else if (t == "sdc") e.obj = s.sp_ob1.chandeg + e.obtypno - 1;
                        const int k = e.obj;
                        s.ob(k).ru_tot = s.ob(k).ru_tot + 1;
                    }
                }
            }
            break;
        }
    }
    for (int iru = 1; iru <= s.sp_ob.ru; ++iru) {
        const routing_unit_data& d = s.ru_def(iru);
        for (int ie1 = 1; ie1 <= d.num_tot; ++ie1) {
            const int ie = d.num(ie1);
            const int iob = s.ru_elem(ie).obj;
            s.ob(iob).ru.alloc(1, 1, 0);
        }
        for (int ie1 = 1; ie1 <= d.num_tot; ++ie1) {
            const int ie = d.num(ie1);
            s.ielem_ru(ie) = s.ielem_ru(ie) + 1;
            const int iob = s.ru_elem(ie).obj;
            s.ob(iob).ru(s.ielem_ru(ie)) = iru;
            s.ob(iob).elem = ie1;
        }
    }
}

}  // namespace

// hyd_connect.f90 (hru, routing units, aquifers, swat-deg channels and outlets)
void hyd_connect(Sim& s) {
    const spatial_objects& n = s.sp_ob;
    spatial_objects& o1 = s.sp_ob1;
    if (n.hru_lte > 0 || n.gwflow > 0 || n.chan > 0 || n.res > 0 || n.recall > 0 || n.exco > 0 || n.dr > 0 ||
        n.canal > 0 || n.pump > 0 || n.aqu2d > 0 || n.herd > 0 || n.wro > 0)
        throw std::runtime_error(
            "SWAT+ objects other than hru, rout_unit, aquifer, chandeg and outlet (object.cnt) are not supported");
    s.rcv_sum.alloc(1, n.objs, 0);
    s.dfn_sum.alloc(1, n.objs, 0);
    s.ru_seq.alloc(1, n.objs, 0);
    int nspu = 1;
    auto first = [&](int count, int& slot) {
        if (count > 0) {
            slot = nspu;
            nspu = count + nspu;
        }
    };
    first(n.hru, o1.hru);
    first(n.hru_lte, o1.hru_lte);
    first(n.ru, o1.ru);
    first(n.gwflow, o1.gwflow);
    first(n.aqu, o1.aqu);
    first(n.chan, o1.chan);
    first(n.res, o1.res);
    first(n.recall, o1.recall);
    first(n.exco, o1.exco);
    first(n.dr, o1.dr);
    first(n.canal, o1.canal);
    first(n.pump, o1.pump);
    first(n.outlet, o1.outlet);
    first(n.chandeg, o1.chandeg);

    if (n.hru > 0) hyd_read_connect(s, s.in_con.hru_con, "hru", o1.hru, n.hru, s.hd_tot.hru, s.bsn_prm.day_lag_mx);
    if (n.ru > 0) {
        hyd_read_connect(s, s.in_con.ru_con, "ru", o1.ru, n.ru, s.hd_tot.ru, s.bsn_prm.day_lag_mx);
        ru_read(s);
        ru_read_elements(s);
    }
    if (n.aqu > 0) {
        hyd_read_connect(s, s.in_con.aqu_con, "aqu", o1.aqu, n.aqu, s.hd_tot.aqu, 1);
        if (s.files.exists(s.in_link.aqu_cha) && s.in_link.aqu_cha != "null")
            throw std::runtime_error("SWAT+ aquifer-channel links (aqu_cha.lin) are not supported");
    }
    if (n.outlet > 0)
        hyd_read_connect(s, s.in_con.out_con, "outlet", o1.outlet, n.outlet, s.hd_tot.outlet, s.bsn_prm.day_lag_mx);
    if (n.chandeg > 0)
        hyd_read_connect(s, s.in_con.chandeg_con, "chandeg", o1.chandeg, n.chandeg, s.hd_tot.chandeg,
                         s.bsn_prm.day_lag_mx);

    for (int i = 1; i <= n.objs; ++i) {
        const int k = s.ob(i).ru_tot;
        if (k > 0) s.ob(i).obj_subs.alloc(1, k, 0);
    }
    for (int iru = 1; iru <= n.ru; ++iru) {
        for (int ii = 1; ii <= s.ru_def(iru).num_tot; ++ii) {
            const int ielem = s.ru_def(iru).num(ii);
            const int k = s.ru_elem(ielem).obj;
            const int iob = o1.ru + iru - 1;
            s.ru_seq(k) = s.ru_seq(k) + 1;
            const int kk = s.ru_seq(k);
            s.ob(k).obj_subs(kk) = iob;
            s.dfn_sum(iob) = s.dfn_sum(iob) + 1;
        }
    }

    for (int i = 1; i <= n.objs; ++i) {
        object_connectivity& o = s.ob(i);
        for (int ii = 1; ii <= o.src_tot; ++ii) {
            const std::string& t = o.obtyp_out(ii);
            int base = 0;
            if (t == "hru") base = o1.hru;
            else if (t == "hlt") base = o1.hru_lte;
            else if (t == "ru") base = o1.ru;
            else if (t == "aqu") base = o1.aqu;
            else if (t == "cha") base = o1.chan;
            else if (t == "res") base = o1.res;
            else if (t == "exc") base = o1.exco;
            else if (t == "dr") base = o1.dr;
            else if (t == "out") base = o1.outlet;
            else if (t == "sdc") base = o1.chandeg;
            else throw std::runtime_error("SWAT+ outflow to object type " + t + " is not supported");
            o.obj_out(ii) = base + o.obtypno_out(ii) - 1;
            const int j = o.obj_out(ii);
            s.ob(j).rcv_tot = s.ob(j).rcv_tot + 1;
            if (t == "ru") {
                const int iru = o.obtypno_out(ii);
                for (int kk = 1; kk <= s.ru_def(iru).num_tot; ++kk) {
                    const int ielem = s.ru_def(iru).num(kk);
                    const int iob = s.ru_elem(ielem).obj;
                    s.ob(iob).rcv_tot = s.ob(iob).rcv_tot + 1;
                }
            }
            const std::string& h = o.htyp_out(ii);
            if (h == "tot") o.ihtyp_out(ii) = 1;
            else if (h == "rhg") o.ihtyp_out(ii) = 2;
            else if (h == "sur") o.ihtyp_out(ii) = 3;
            else if (h == "lat") o.ihtyp_out(ii) = 4;
            else if (h == "til") o.ihtyp_out(ii) = 5;
        }
    }

    for (int i = 1; i <= n.objs; ++i) {
        object_connectivity& o = s.ob(i);
        if (o.rcv_tot > 0) {
            const int m = o.rcv_tot;
            o.obj_in.alloc(1, m, 0);
            o.obtyp_in.alloc(1, m, "");
            o.obtypno_in.alloc(1, m, 0);
            o.htyp_in.alloc(1, m, "");
            o.ihtyp_in.alloc(1, m, 0);
            o.frac_in.alloc(1, m, 0.0f);
            o.hin_uh.alloc(1, m, inflow_unit_hyds());
            for (int ii = 1; ii <= m; ++ii) {
                o.hin_uh(ii).uh.alloc(1, s.bsn_prm.day_lag_mx, 1, s.time.step, 0.0f);
                o.hin_uh(ii).hyd_flo.alloc(1, s.bsn_prm.day_lag_mx, 1, s.time.step, 0.0f);
            }
            o.hin_d.alloc(1, m, hyd_output());
            o.hin_m.alloc(1, m, hyd_output());
            o.hin_y.alloc(1, m, hyd_output());
            o.hin_a.alloc(1, m, hyd_output());
        }
    }

    for (int i = 1; i <= n.objs; ++i) {
        object_connectivity& oi = s.ob(i);
        for (int ii = 1; ii <= oi.src_tot; ++ii) {
            int kk = oi.obj_out(ii);
            s.rcv_sum(kk) = s.rcv_sum(kk) + 1;
            int jj = s.rcv_sum(kk);
            object_connectivity& ok = s.ob(kk);
            ok.obj_in(jj) = i;
            ok.obtyp_in(jj) = oi.typ;
            ok.obtypno_in(jj) = oi.num;
            ok.htyp_in(jj) = oi.htyp_out(ii);
            ok.ihtyp_in(jj) = oi.ihtyp_out(ii);
            ok.frac_in(jj) = oi.frac_out(ii);
            oi.rcvob_inhyd(ii) = jj;
            if (ok.typ == "ru") {
                const int iru = ok.num;
                for (int ielem = 1; ielem <= s.ru_def(iru).num_tot; ++ielem) {
                    const int ielem_db = s.ru_def(iru).num(ielem);
                    kk = s.ru_elem(ielem_db).obj;
                    s.rcv_sum(kk) = s.rcv_sum(kk) + 1;
                    jj = s.rcv_sum(kk);
                    object_connectivity& oe = s.ob(kk);
                    oe.obj_in(jj) = i;
                    oe.obtyp_in(jj) = oi.typ;
                    oe.obtypno_in(jj) = oi.props;
                    oe.htyp_in(jj) = oi.htyp_out(ii);
                    oe.ihtyp_in(jj) = oi.ihtyp_out(ii);
                    oe.frac_in(jj) = oi.frac_out(ii) * s.ru_elem(ielem_db).frac;
                    oi.rcvob_inhyd(ii) = jj;
                }
            }
        }
    }

    // the command sequence: an object fires once all it receives (and all its
    // routing unit's elements) have fired
    int iobj_tot = 0;
    s.rcv_sum.fill(0);
    int iord = 1;
    s.dfn_sum.fill(0);
    int cmd_prev = 0;
    bool idone = n.objs == 0;
    while (!idone) {
        for (int i = 1; i <= n.objs; ++i) {
            if (iord > 5000) throw std::runtime_error("SWAT+: an infinite loop is detected in the connect file(s)");
            object_connectivity& o = s.ob(i);
            if (s.rcv_sum(i) == o.rcv_tot && s.dfn_sum(i) == o.dfn_tot) {
                if (o.fired == 0) {
                    o.fired = 1;
                    for (int ircv = 1; ircv <= o.rcv_tot; ++ircv) {
                        const std::string& t = o.obtyp_in(ircv);
                        if (t == "hru" || t == "ru" || t == "chandeg" || t == "recall" || t == "res" || t == "outlet") {
                            const int ob1 = o.obj_in(ircv);
                            o.area_ha_calc = o.area_ha_calc + s.ob(ob1).area_ha_calc * o.frac_in(ircv);
                        }
                    }
                    ++iobj_tot;
                    if (iobj_tot == n.objs) idone = true;
                    for (int k = 1; k <= o.ru_tot; ++k) s.dfn_sum(o.obj_subs(k)) = s.dfn_sum(o.obj_subs(k)) + 1;
                    const int isrc_tot = std::max(o.src_tot, 1);
                    for (int ii = 1; ii <= isrc_tot; ++ii) {
                        if (o.src_tot > 0 && o.obtypno_out(ii) > 0) {
                            const int k = o.obj_out(ii);
                            s.rcv_sum(k) = s.rcv_sum(k) + 1;
                            if (s.ob(k).typ == "ru") {
                                const int iru = s.ob(k).props;
                                for (int jj = 1; jj <= s.ru_def(iru).num_tot; ++jj) {
                                    const int ielem = s.ru_def(iru).num(jj);
                                    const int iob = s.ru_elem(ielem).obj;
                                    s.rcv_sum(iob) = s.rcv_sum(iob) + 1;
                                }
                            }
                        }
                        if (o.rcv_tot == 0 && o.typ != "ru") {
                            o.cmd_order = 1;
                        } else {
                            int iorder = 0;
                            for (int ircv = 1; ircv <= o.rcv_tot; ++ircv)
                                iorder = std::max(iorder, s.ob(o.obj_in(ircv)).cmd_order);
                            if (o.typ == "ru" && o.rcv_tot == 0) iorder = 1;
                            o.cmd_order = iorder + 1;
                        }
                        o.cmd_prev = cmd_prev;
                        if (cmd_prev > 0) {
                            s.ob(cmd_prev).cmd_next = i;
                        } else {
                            o1.objs = i;
                        }
                        cmd_prev = i;
                        o.cmd_order = iord;
                    }
                }
            }
        }
        ++iord;
    }
}

}  // namespace swat
