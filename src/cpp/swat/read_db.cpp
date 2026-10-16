// SPDX-License-Identifier: GPL-3.0-or-later
//
// proc_db and proc_read: the parameter databases, management schedules, plant
// communities, land use, soils, topography and hydrology (plant_parm_read.f90,
// plantparm_init.f90, plant_transplant_read.f90, till_parm_read.f90, fert_parm_read.f90,
// mgt_read_*.f90, read_mgtops.f90, readpcom.f90, cntbl_read.f90, cons_prac_read.f90,
// overland_n_read.f90, landuse_read.f90, soil_plant_init.f90, solt_db_read.f90,
// topo_read.f90, field_read.f90, hydrol_read.f90, snowdb_read.f90, soil_db_read.f90).
// Databases of features the port does not simulate are read for their names only: a
// HRU that uses them is refused when it is set up.

#include <stdexcept>

#include "swat/fmath.hpp"
#include "swat/swat.hpp"

namespace swat {

using fio::chars;

namespace {

int count_after_header(fio::File& f) {
    std::string titldum;
    int imax = 0;
    int eof = 0;
    while (eof == 0) {
        eof = f.read(chars(titldum, 80));
        if (eof < 0) break;
        ++imax;
    }
    return imax;
}

// The usual database file: title, header, one record per entry, read as the whole type
// ("read (107,*,iostat=eof) db(i)"). each(i) runs after entry i is read.
template <class T, class Each>
void read_db(Sim& s, const std::string& file, Vec<T>& db, int& count, Each&& each) {
    std::string titldum, header;
    int imax = 0;
    if (!s.files.exists(file) || file == "null") {
        db.alloc(0, 0, T());
    } else {
        fio::File f = s.files.open(file);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            imax = count_after_header(f);
            db.alloc(0, imax, T());
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int i = 1; i <= imax; ++i) {
                const int eof = read_items(f, db(i));
                each(i);
                if (eof < 0) break;
            }
            break;
        }
    }
    count = imax;
}

template <class T>
void read_db(Sim& s, const std::string& file, Vec<T>& db, int& count) {
    read_db(s, file, db, count, [](int) {});
}

// A database of a feature outside the port: the first item (the name) of each entry.
template <class T, class Name>
void read_db_names(Sim& s, const std::string& file, Vec<T>& db, int& count, Name&& name_of) {
    std::string titldum, header;
    int imax = 0;
    if (!s.files.exists(file) || file == "null") {
        db.alloc(0, 0, T());
    } else {
        fio::File f = s.files.open(file);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            imax = count_after_header(f);
            db.alloc(0, imax, T());
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int i = 1; i <= imax; ++i)
                if (f.read(name_of(db(i))) < 0) break;
            break;
        }
    }
    count = imax;
}

template <class T>
void read_db_names(Sim& s, const std::string& file, Vec<T>& db, int& count) {
    read_db_names(s, file, db, count, [](T& x) { return chars(x.name, T::name_len); });
}

// plant_parm_read.f90
void plant_parm_read(Sim& s) {
    read_db(s, s.in_parmdb.plants_plt, s.pldb, s.db_mx.plantparm, [&](int ic) {
        plant_db& p = s.pldb(ic);
        p.mat_yrs = std::max(1, p.mat_yrs);
        p.res_part_fracs.meta_frac = 0.85f;
        p.res_part_fracs.str_frac = 0.15f;
        p.res_part_fracs.lig_frac = 0.12f;
    });
    if (s.bsn_cc.nam1 != 0) throw std::runtime_error("SWAT+ plant classes (codes.bsn nam1 != 0) are not supported");
    s.plcp.alloc(0, s.db_mx.plantparm, plant_cp());
    s.pl_class.alloc(0, s.db_mx.plantparm, "");
}

// ascrv.f90: the shape coefficients of an S curve through (x1, y1) and (x2, y2)
void ascrv(float x1, float x2, float x3, float x4, float& x5, float& x6) {
    float xx = 0.0f;
    xx = std::log(x3 / x1 - x3);
    x6 = (xx - std::log(x4 / x2 - x4)) / (x4 - x3);
    x5 = xx + (x3 * x6);
}

// plantparm_init.f90
void plantparm_init(Sim& s) {
    for (int ic = 1; ic <= s.db_mx.plantparm; ++ic) {
        plant_db& p = s.pldb(ic);
        plant_cp& c = s.plcp(ic);
        if (p.bm_dieoff <= 1.e-6f) p.bm_dieoff = 1.00f;
        if (p.ext_coef < 1.e-6f) p.ext_coef = 0.65f;
        if (p.rsdco_pl < 1.e-6f) p.rsdco_pl = s.bsn_prm.rsdco;
        if (p.usle_c <= 0.0f) p.usle_c = 0.001f;
        if (p.usle_c >= 1.0f) p.usle_c = 1.0f;
        if (p.blai <= 0.0f) p.blai = 0.0f;
        if (p.blai >= 10.0f) p.blai = 10.0f;
        if (p.rsr1 <= 0.0f) p.rsr1 = 0.4f;
        if (p.rsr2 <= 0.0f) p.rsr2 = 0.2f;
        if (p.aeration <= 0.0f) p.aeration = 0.2f;
        if (p.rsd_pctcov <= 0.0f) p.rsd_pctcov = 0.4f;
        if (p.rsd_covfac <= 0.0f) p.rsd_covfac = 0.04f;
        if (p.typ == "warm_annual_tuber" || p.typ == "cold_annual_tuber") p.rsr2 = 0.7f;
        if (p.hvsti > 0.7f) p.hvsti = 0.02f;
        if (p.bio_e > 0.0f && p.plantnm != "WATR") {
            if (p.pop1 + p.pop2 > 1.e-6f) {
                p.pop1 = p.pop1 / 1001.0f;
                p.pop2 = p.pop2 / 1001.0f;
                ascrv(p.frlai1, p.frlai2, p.pop1, p.pop2, c.popsc1, c.popsc2);
            }
            ascrv(p.laimx1, p.laimx2, p.frgrw1, p.frgrw2, c.leaf1, c.leaf2);
            const float c1 = 330.0f;
            if (p.co2hi == 330.0f) p.co2hi = 660.0f;
            float b1 = p.bio_e * 0.01f;
            float b2 = p.bioehi * 0.01f;
            ascrv(b1, b2, c1, p.co2hi, c.ruc1, c.ruc2);
            c.cvm = std::log(p.usle_c);
            if (p.pltnfr1 - p.pltnfr2 < 0.0001f) p.pltnfr2 = p.pltnfr1 - 0.0001f;
            if (p.pltnfr2 - p.pltnfr3 < 0.0001f) p.pltnfr3 = 0.75f * p.pltnfr3;
            b1 = p.pltnfr1 - p.pltnfr3;
            b2 = 1.0f - (p.pltnfr2 - p.pltnfr3) / b1;
            float b3 = 1.0f - 0.00001f / b1;
            ascrv(b2, b3, 0.5f, 1.0f, c.nup1, c.nup2);
            if (p.pltpfr1 - p.pltpfr2 < 0.0001f) p.pltpfr2 = p.pltpfr1 - 0.0001f;
            if (p.pltpfr2 - p.pltpfr3 < 0.0001f) p.pltpfr3 = 0.75f * p.pltpfr3;
            b1 = p.pltpfr1 - p.pltpfr3;
            b2 = 1.0f - (p.pltpfr2 - p.pltpfr3) / b1;
            b3 = 1.0f - 0.00001f / b1;
            ascrv(b2, b3, 0.5f, 1.0f, c.pup1, c.pup2);
            c.vpd2 = (1.0f - p.gmaxfr) / (p.vpdfr - 1.0f);
        }
    }
}

// till_parm_read.f90
void till_parm_read(Sim& s) {
    s.bmix_idtill = 0;
    const std::string& file = s.in_parmdb.till_til;
    const bool present = s.files.exists(file) && file != "null";
    read_db(s, file, s.tilldb, s.db_mx.tillparm, [&](int itl) {
        if (s.tilldb(itl).tillnm == "biomix") {
            s.bmix_idtill = itl;
            s.bmix_eff = s.tilldb(itl).effmix;
            s.bmix_depth = s.tilldb(itl).deptil;
        }
    });
    if (present && s.bmix_idtill == 0) {
        s.bmix_eff = 0.2f;
        s.bmix_depth = 50.0f;
    }
}

int jdt(const Fix<int, 1, 13>& numdays, int i, int m) {
    int j = 0;
    if (m != 0) j = m <= 2 ? numdays(m) + i : numdays(m) - 1 + i;
    return j;
}

// read_mgtops.f90
void read_mgtops(Sim& s, fio::File& f, int isched) {
    management_schedule& sc = s.sched(isched);
    int iyear = 1;
    for (int iop = 1; iop <= sc.num_ops; ++iop) {
        management_ops& op = sc.mgt_ops(iop);
        f.read(chars(op.op, op.op_len), op.mon, op.day, op.husc, chars(op.op_char, op.op_char_len),
               chars(op.op_plant, op.op_plant_len), op.op3);
        op.jday = jdt(s.ndays, op.day, op.mon);
        op.year = iyear;
        if (op.op == "skip") iyear = iyear + 1;
        auto find = [](const auto& db, int n, const std::string& name, auto&& key) {
            for (int idb = 1; idb <= n; ++idb)
                if (fstrcmp(key(db(idb)), name) == 0) return idb;
            return 0;
        };
        if (op.op == "pcom") {
            if (int k = find(s.pcomdb, s.db_mx.plantcom, op.op_char, [](const auto& d) -> const std::string& { return d.name; })) op.op1 = k;
        } else if (op.op == "plnt") {
            if (int k = find(s.pldb, s.db_mx.plantparm, op.op_char, [](const auto& d) -> const std::string& { return d.plantnm; })) op.op1 = k;
            if (op.op1 == 0) s.warn(" mgt schedule " + std::to_string(isched) + " op numb " + std::to_string(iop) + " " +
                                    op.op_char + " not found in plants.plt database");
            if (int k = find(s.transpl, s.db_mx.transplant, op.op_plant, [](const auto& d) -> const std::string& { return d.name; })) op.op4 = k;
        } else if (op.op == "harv" || op.op == "hvkl") {
            if (int k = find(s.harvop_db, s.db_mx.harvop_db, op.op_plant, [](const auto& d) -> const std::string& { return d.name; })) op.op1 = k;
        } else if (op.op == "till") {
            if (int k = find(s.tilldb, s.db_mx.tillparm, op.op_char, [](const auto& d) -> const std::string& { return d.tillnm; })) op.op1 = k;
        } else if (op.op == "irrm" || op.op == "irrp") {
            sc.irr = 1;
            if (int k = find(s.irrop_db, s.db_mx.irrop_db, op.op_char, [](const auto& d) -> const std::string& { return d.name; })) op.op1 = k;
        } else if (op.op == "fert") {
            if (int k = find(s.fertdb, s.db_mx.fertparm, op.op_char, [](const auto& d) -> const std::string& { return d.fertnm; })) op.op1 = k;
            if (int k = find(s.chemapp_db, s.db_mx.chemapp_db, op.op_plant, [](const auto& d) -> const std::string& { return d.name; })) op.op4 = k;
        } else if (op.op == "manu" || op.op == "pest" || op.op == "graz" || op.op == "burn" || op.op == "swep") {
            // features outside the port: refused if a HRU follows the schedule (mgt setup)
        }
    }
}

// mgt_read_mgtops.f90
void mgt_read_mgtops(Sim& s) {
    const std::string& file = s.in_lum.management_sch;
    std::string titldum, header;
    int imax = 0;
    if (!s.files.exists(file) || file == "null") {
        s.sched.alloc(0, 0, management_schedule());
    } else {
        fio::File f = s.files.open(file);
        for (;;) {
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            int eof = 0;
            while (eof == 0) {
                int nops = 0, nauto = 0;
                eof = f.read(chars(titldum, 80), nops, nauto);
                if (eof < 0) break;
                for (int k = 1; k <= nauto; ++k) {
                    eof = f.read(chars(titldum, 80));
                    if (eof < 0) break;
                }
                for (int k = 1; k <= nops; ++k) {
                    eof = f.read(chars(titldum, 80));
                    if (eof < 0) break;
                }
                ++imax;
            }
            s.sched.alloc(0, imax, management_schedule());
            f.rewind();
            if (f.read(chars(titldum, 80)) < 0) break;
            if (f.read(chars(header, 80)) < 0) break;
            for (int isched = 1; isched <= imax; ++isched) {
                management_schedule& sc = s.sched(isched);
                eof = f.read(chars(sc.name, sc.name_len), sc.num_ops, sc.num_autos);
                if (eof < 0) break;
                const int m_autos = sc.num_autos;
                if (m_autos > 0) {
                    sc.auto_name.alloc(1, m_autos, "");
                    sc.num_db.alloc(1, m_autos, 0);
                    for (int iauto = 1; iauto <= m_autos; ++iauto) {
                        eof = f.read(chars(sc.auto_name(iauto), sc.auto_name_len));
                        const std::string& an = sc.auto_name(iauto);
                        if (an == "pl_hv_summer1" || an == "pl_hv_winter1" || an == "pl_hv_summer2")
                            throw std::runtime_error("SWAT+ automatic crop schedules (" + an + ") are not supported");
                        if (eof < 0) break;
                    }
                }
                sc.mgt_ops.alloc(1, sc.num_ops, management_ops());
                read_mgtops(s, f, isched);
            }
            break;
        }
    }
    s.db_mx.mgt_ops = imax;
}

// readpcom.f90
void readpcom(Sim& s) {
    const std::string& file = s.in_init.plant;
    std::string titldum, header, name;
    int imax = 0;
    if (!s.files.exists(file) || file == "null") {
        s.pcomdb.alloc(0, 0, plant_community_db());
        s.pcomdb(0).pl.alloc(0, 0, plant_init_db());
        s.db_mx.plantcom = 0;  // set to 1 here, then to imax (0) after the if
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        int eof = 0;
        while (eof == 0) {
            int numb = 0;
            eof = f.read(chars(name, 13), numb);
            if (eof < 0) break;
            for (int ii = 1; ii <= numb; ++ii) {
                eof = f.read(chars(name, 13));
                if (eof < 0) break;
            }
            if (eof < 0) break;
            ++imax;
        }
        s.pcomdb.alloc(0, imax, plant_community_db());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int icom = 1; icom <= imax; ++icom) {
            plant_community_db& c = s.pcomdb(icom);
            eof = f.read(chars(c.name, c.name_len), c.plants_com, c.rot_yr_ini);
            if (eof < 0) break;
            c.pl.alloc(1, c.plants_com, plant_init_db());
            for (int iplt = 1; iplt <= c.plants_com; ++iplt) {
                plant_init_db& p = c.pl(iplt);
                eof = f.read(chars(p.cpnm, p.cpnm_len), chars(p.igro, p.igro_len), p.lai, p.bioms, p.phuacc, p.pop, p.fr_yrmat, p.rsdin);
                if (eof < 0) break;
                for (int ipldb = 1; ipldb <= s.db_mx.plantparm; ++ipldb) {
                    if (fstrcmp(p.cpnm, s.pldb(ipldb).plantnm) == 0) {
                        p.db_num = ipldb;
                        break;
                    }
                }
                if (eof < 0) break;
            }
        }
        break;
    }
    s.db_mx.plantcom = imax;
}

// landuse_read.f90
void landuse_read(Sim& s) {
    read_db(s, s.in_lum.landuse_lum, s.lum, s.db_mx.landuse);
    const int imax = s.db_mx.landuse;
    s.lum_str.alloc(0, imax, land_use_structures());
    auto find = [](const auto& db, int n, const std::string& name) {
        for (int k = 1; k <= n; ++k)
            if (fstrcmp(db(k).name, name) == 0) return k;
        return 0;
    };
    for (int ilu = 1; ilu <= imax; ++ilu) {
        land_use_management& l = s.lum(ilu);
        land_use_structures& ls = s.lum_str(ilu);
        if (l.plant_cov != "null") {
            if (int k = find(s.pcomdb, s.db_mx.plantcom, l.plant_cov)) ls.plant_cov = k;
        }
        if (l.mgt_ops != "null") {
            if (int k = find(s.sched, s.db_mx.mgt_ops, l.mgt_ops)) ls.mgt_ops = k;
        }
        if (l.cn_lu != "null") {
            if (int k = find(s.cn, s.db_mx.cn_lu, l.cn_lu)) ls.cn_lu = k;
        }
        if (l.cons_prac != "null") {
            if (int k = find(s.cons_prac, s.db_mx.cons_prac, l.cons_prac)) ls.cons_prac = k;
        }
        if (l.tiledrain != "null") {
            if (int k = find(s.hru_module::sdr, s.db_mx.sdr, l.tiledrain)) ls.tiledrain = k;
        }
        if (l.septic != "null") {
            if (int k = find(s.sep, s.db_mx.septic, l.septic)) ls.septic = k;
        }
        if (l.fstrip != "null") {
            if (int k = find(s.filtstrip_db, s.db_mx.filtop_db, l.fstrip)) ls.fstrip = k;
        }
        if (l.grassww != "null") {
            if (int k = find(s.grwaterway_db, s.db_mx.grassop_db, l.grassww)) ls.grassww = k;
        }
        if (l.bmpuser != "null") {
            if (int k = find(s.bmpuser_db, s.db_mx.bmpuserop_db, l.bmpuser)) ls.bmpuser = k;
        }
        if (ls.plant_cov == 0 && l.plant_cov != "null") s.warn(l.name + " " + l.plant_cov + " not found in plant.ini");
        if (ls.mgt_ops == 0 && l.mgt_ops != "null") s.warn(l.name + " " + l.mgt_ops + " not found in management.sch");
        if (ls.cn_lu == 0 && l.cn_lu != "null") s.warn(l.name + " " + l.cn_lu + " not found in cntable.lum");
        if (ls.cons_prac == 0 && l.cons_prac != "null") s.warn(l.name + " " + l.cons_prac + " not found (landuse-consprac)");
    }
}

}  // namespace

void proc_db(Sim& s) {
    plant_parm_read(s);
    plantparm_init(s);
    read_db(s, "transplant.ops", s.transpl, s.db_mx.transplant);
    till_parm_read(s);
    read_db_names(s, s.in_parmdb.pest, s.pestdb, s.db_mx.pestparm);
    read_db(s, s.in_parmdb.fert_frt, s.fertdb, s.db_mx.fertparm);
    read_db_names(s, "manure_om.frt", s.manure_om, s.db_mx.manure_om);
    read_db_names(s, "manure_db.frt", s.manure_db, s.db_mx.manureparm);
    read_db_names(s, s.in_parmdb.urban_urb, s.urbdb, s.db_mx.urban, [](urban_db& x) { return chars(x.urbnm, x.urbnm_len); });
    read_db_names(s, s.in_parmdb.pathcom_db, s.path_db, s.db_mx.path, [](pathogen_db& x) { return chars(x.pathnm, x.pathnm_len); });
    read_db_names(s, s.in_parmdb.septic_sep, s.sepdb, s.db_mx.sep, [](septic_db& x) { return chars(x.sepnm, x.sepnm_len); });
    read_db(s, s.in_ops.irr_ops, s.irrop_db, s.db_mx.irrop_db);
    read_db(s, s.in_ops.chem_ops, s.chemapp_db, s.db_mx.chemapp_db);
    read_db(s, s.in_ops.harv_ops, s.harvop_db, s.db_mx.harvop_db);
    read_db_names(s, s.in_ops.graze_ops, s.grazeop_db, s.db_mx.grazeop_db);
    read_db_names(s, s.in_ops.sweep_ops, s.sweepop_db, s.db_mx.sweepop_db);
    read_db_names(s, s.in_ops.fire_ops, s.fire_db, s.db_mx.fireop_db);
    mgt_read_mgtops(s);
    read_db_names(s, "puddle.ops", s.pudl_db, s.db_mx.pudl_db);
    read_db_names(s, s.in_str.tiledrain_str, s.hru_module::sdr, s.db_mx.sdr);
    read_db_names(s, s.in_str.septic_str, s.sep, s.db_mx.septic);
    read_db_names(s, s.in_str.grassww_str, s.grwaterway_db, s.db_mx.grassop_db);
    read_db_names(s, s.in_str.fstrip_str, s.filtstrip_db, s.db_mx.filtop_db);
    read_db_names(s, s.in_str.bmpuser_str, s.bmpuser_db, s.db_mx.bmpuserop_db);
    read_db_names(s, "satbuffer.str", s.satbuff_db, s.db_mx.sat_buff);
    readpcom(s);
    read_db(s, s.in_lum.cntable_lum, s.cn, s.db_mx.cn_lu);
    read_db(s, s.in_lum.cons_prac_lum, s.cons_prac, s.db_mx.cons_prac);
    read_db(s, s.in_lum.ovn_lum, s.overland_n, s.db_mx.ovn);
    landuse_read(s);
}

namespace {

// soil_plant_init.f90
void soil_plant_init(Sim& s) {
    const std::string& file = s.in_init.soil_plant_ini;
    std::string titldum, header;
    if (!(s.files.exists(file) || file != "null")) return;
    fio::File f = s.files.open(file);
    if (!f.open()) throw std::runtime_error("cannot open " + file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        const int imax = count_after_header(f);
        s.db_mx.sol_plt_ini = imax;
        s.sol_plt_ini.alloc(1, imax, soil_plant_initialize());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int ii = 1; ii <= imax; ++ii) {
            soil_plant_initialize& p = s.sol_plt_ini(ii);
            const int eof = f.read(chars(p.name, p.name_len), p.sw_frac, chars(p.nutc, p.nutc_len),
                                   chars(p.pestc, p.pestc_len), chars(p.pathc, p.pathc_len), chars(p.saltc, p.saltc_len),
                                   chars(p.hmetc, p.hmetc_len));
            if (eof < 0) break;
        }
        break;
    }
}

// soil_db_read.f90
void soil_db_read(Sim& s) {
    const std::string& file = s.in_sol.soils_sol;
    std::string titldum, header;
    int imax = 0;
    if (!s.files.exists(file) || file == "null") {
        s.soildb.alloc(0, 0, soil_database());
        s.soildb(0).ly.alloc(0, 0, soilayer_db());
        return;
    }
    fio::File f = s.files.open(file);
    for (;;) {
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        int eof = 0;
        while (eof == 0) {
            int nlyr = 0;
            eof = f.read(chars(titldum, 80), nlyr);
            if (eof < 0) break;
            for (int lyr = 1; lyr <= nlyr; ++lyr) {
                eof = f.read(chars(titldum, 80));
                if (eof < 0) break;
            }
            ++imax;
        }
        s.db_mx.soil = imax;
        s.soildb.alloc(0, imax, soil_database());
        f.rewind();
        if (f.read(chars(titldum, 80)) < 0) break;
        if (f.read(chars(header, 80)) < 0) break;
        for (int isol = 1; isol <= s.db_mx.soil; ++isol) {
            soil_database& d = s.soildb(isol);
            eof = f.read(chars(d.s.snam, d.s.snam_len), d.s.nly);
            if (eof < 0) break;
            const int mlyr = d.s.nly;
            d.ly.alloc(1, mlyr, soilayer_db());
            f.backspace();
            eof = f.read(chars(d.s.snam, d.s.snam_len), d.s.nly, chars(d.s.hydgrp, d.s.hydgrp_len), d.s.zmx,
                         d.s.anion_excl, d.s.crk, chars(d.s.texture, d.s.texture_len));
            if (eof < 0) break;
            for (int j = 1; j <= mlyr; ++j) {
                soilayer_db& l = d.ly(j);
                eof = f.read(l.z, l.bd, l.awc, l.k, l.cbn, l.clay, l.silt, l.sand, l.rock, l.alb, l.usle_k, l.ec, l.cal,
                             l.ph);
                if (eof < 0) break;
            }
            if (d.ly(1).z < 20.0f) {
                if (d.s.nly == 1 || d.ly(2).z > 20.0f) d.ly(1).z = 20.0f;
            }
        }
        break;
    }
}

}  // namespace

void proc_read(Sim& s) {
    cli_staread(s);
    soil_plant_init(s);
    read_db(s, s.in_sol.nut_sol, s.solt_db, s.db_mx.soiltest, [&](int i) {
        if (s.solt_db(i).exp_co > 0.005f) s.solt_db(i).exp_co = 0.001f;
    });
    read_db(s, s.in_hyd.topogr_hyd, s.topo_db, s.db_mx.topo);
    {
        // field_read.f90 (reads each entry after a backspace: the same records)
        read_db(s, s.in_hyd.field_fld, s.field_db, s.db_mx.field);
    }
    read_db(s, s.in_hyd.hydrol_hyd, s.hyd_db, s.db_mx.hyd);
    read_db(s, s.in_parmdb.snow, s.snodb, s.db_mx.sno);
    soil_db_read(s);
}

}  // namespace swat
