// SPDX-License-Identifier: GPL-3.0-or-later
//
// Evaluation of Biodiverse's spatial conditions (see conditions.hpp).

#include "bio/conditions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "bio/perlmath.hpp"
#include "lsm/parallel.hpp"

namespace bio {

double round_precision(double x, double precision) { return posix_round(x * precision) / precision; }

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Distances between the processing group and a candidate, as
// Biodiverse::SpatialConditions::get_distances computes them.
struct Dists {
    std::vector<double> d, D, c, C;
    std::vector<std::uint8_t> have;  // axis computed
    double Dv = 0, Cv = 0, Dsqr = 0, Csqr = 0;
    bool computed = false;
};

class Evaluator {
  public:
    Evaluator(const Groups& g, const Program& p) : g_(g), p_(p) {
        const auto nd = static_cast<std::size_t>(g.ndim);
        dist_.d.assign(nd, 0);
        dist_.D.assign(nd, 0);
        dist_.c.assign(nd, 0);
        dist_.C.assign(nd, 0);
        dist_.have.assign(nd, 0);
        locals_.assign(static_cast<std::size_t>(std::max(p.n_locals, 1)), 0.0);
        if (p.use_euc || p.use_cell) {
            for (i32 k = 0; k < g.ndim; ++k) iters_.push_back(k);
        } else {
            iters_ = p.iter_axes;
        }
    }

    bool test(i64 proc, i64 nbr, bool def_query) {
        proc_ = proc;
        nbr_ = nbr;
        def_query_ = def_query;
        dist_.computed = false;
        return truth(eval(p_.root));
    }

  private:
    const Groups& g_;
    const Program& p_;
    Dists dist_;
    std::vector<double> locals_;
    std::vector<i32> iters_;
    i64 proc_ = 0, nbr_ = 0;
    bool def_query_ = false;

    static bool truth(double v) { return v != 0 && !std::isnan(v); }

    double coord(i64 g, i32 k) const {
        if (k < 0 || k >= g_.ndim) return 0;  // undef
        return g_.coords[static_cast<std::size_t>(g * g_.ndim + k)];
    }

    const Dists& dists() {
        if (dist_.computed) return dist_;
        if (def_query_) throw std::runtime_error("distances are not available in a definition query");
        const bool cells = p_.use_cell || p_.cell_lists;
        double sum_d = 0, sum_c = 0;
        std::fill(dist_.have.begin(), dist_.have.end(), 0);
        for (i32 k : iters_) {
            if (k < 0 || k >= g_.ndim) continue;
            if (!g_.numeric[static_cast<std::size_t>(k)])
                throw std::runtime_error("distances on a text axis");
            const double dv = coord(nbr_, k) - coord(proc_, k);
            sum_d += dv * dv;
            const double dr = round_precision(dv);
            dist_.d[k] = dr;
            dist_.D[k] = std::fabs(dr);
            dist_.have[k] = 1;
            if (cells) {
                const double cs = g_.cell_sizes[static_cast<std::size_t>(k)];
                if (cs <= 0) throw std::runtime_error("cell distances need positive cell sizes");
                const double cv = dr / cs;
                sum_c += cv * cv;
                const double cr = round_precision(cv);
                dist_.c[k] = cr;
                dist_.C[k] = std::fabs(cr);
            }
        }
        dist_.Dsqr = sum_d;
        dist_.Csqr = sum_c;
        dist_.Dv = p_.use_euc ? round_precision(perl_sqrt(sum_d)) : 0;
        dist_.Cv = p_.use_cell ? round_precision(perl_sqrt(sum_c)) : 0;
        dist_.computed = true;
        return dist_;
    }

    // string id of a string node (-1: undef or a number)
    i32 eval_str(i32 id, double* num) {
        const Node& n = p_.nodes[static_cast<std::size_t>(id)];
        switch (n.op) {
            case Op::StrConst:
                return n.a;
            case Op::CoordId:
                return g_.name_str[static_cast<std::size_t>(n.a == 0 ? proc_ : nbr_)];
            case Op::AxisStr: {
                const i64 grp = (n.a == 0 || def_query_) ? proc_ : nbr_;
                const i32 k = static_cast<i32>(eval(n.b));
                if (k < 0 || k >= g_.ndim) return -1;
                const i32 s = g_.axis_str[static_cast<std::size_t>(grp * g_.ndim + k)];
                if (s < 0 && num) *num = coord(grp, k);
                return s;
            }
            default:
                if (num) *num = eval(id);
                return -2;  // a number
        }
    }

    bool str_equal(i32 a, i32 b) {
        double na = 0, nb = 0;
        const i32 sa = eval_str(a, &na), sb = eval_str(b, &nb);
        if (sa >= 0 && sb >= 0) return sa == sb;
        if (sa < 0 && sb < 0) return na == nb;
        // one string, one number: compare the text
        const std::string& s = g_.strings[static_cast<std::size_t>(sa >= 0 ? sa : sb)];
        const double v = sa >= 0 ? nb : na;
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.15g", v);
        return s == buf;
    }

    double eval(i32 id) {
        const Node& n = p_.nodes[static_cast<std::size_t>(id)];
        switch (n.op) {
            case Op::Num:
                return n.v;
            case Op::Undef:
                return 0;
            case Op::Var:
                return variable(static_cast<VarId>(n.a));
            case Op::IdxVar:
                return array_value(static_cast<ArrId>(n.a), static_cast<i32>(eval(n.b)));
            case Op::Neg:
                return -eval(n.a);
            case Op::Not:
                return truth(eval(n.a)) ? 0.0 : 1.0;
            case Op::Add:
                return eval(n.a) + eval(n.b);
            case Op::Sub:
                return eval(n.a) - eval(n.b);
            case Op::Mul:
                return eval(n.a) * eval(n.b);
            case Op::Div: {
                const double den = eval(n.b);
                if (den == 0) throw std::runtime_error("Illegal division by zero");
                return eval(n.a) / den;
            }
            case Op::Mod: {
                // Perl's % works on integers, sign of the right operand
                const double a = std::trunc(eval(n.a)), b = std::trunc(eval(n.b));
                if (b == 0) throw std::runtime_error("Illegal modulus zero");
                double r = std::fmod(a, b);
                if (r != 0 && ((r < 0) != (b < 0))) r += b;
                return r;
            }
            case Op::Pow:
                return perl_pow(eval(n.a), eval(n.b));
            case Op::Lt:
                return eval(n.a) < eval(n.b) ? 1 : 0;
            case Op::Le:
                return eval(n.a) <= eval(n.b) ? 1 : 0;
            case Op::Gt:
                return eval(n.a) > eval(n.b) ? 1 : 0;
            case Op::Ge:
                return eval(n.a) >= eval(n.b) ? 1 : 0;
            case Op::Eq:
                return eval(n.a) == eval(n.b) ? 1 : 0;
            case Op::Ne:
                return eval(n.a) != eval(n.b) ? 1 : 0;
            case Op::Cmp: {
                const double a = eval(n.a), b = eval(n.b);
                return a < b ? -1 : (a > b ? 1 : 0);
            }
            case Op::And: {
                const double a = eval(n.a);
                return truth(a) ? eval(n.b) : a;
            }
            case Op::Or: {
                const double a = eval(n.a);
                return truth(a) ? a : eval(n.b);
            }
            case Op::Dor: {
                // no undef tracking: a defined left operand is returned
                return eval(n.a);
            }
            case Op::Tern:
                return truth(eval(n.a)) ? eval(n.b) : eval(n.c);
            case Op::Func:
                return func(static_cast<FuncId>(n.a), n.b, n.c);
            case Op::StrEq:
                return str_equal(n.a, n.b) ? 1 : 0;
            case Op::StrNe:
                return str_equal(n.a, n.b) ? 0 : 1;
            case Op::Call:
                return call(p_.calls[static_cast<std::size_t>(n.a)]);
            case Op::Local:
                return locals_[static_cast<std::size_t>(n.a)];
            case Op::SetLocal:
                return locals_[static_cast<std::size_t>(n.a)] = eval(n.b);
            case Op::Seq: {
                double v = 0;
                for (i32 k = 0; k < n.b; ++k) v = eval(p_.seq[static_cast<std::size_t>(n.a + k)]);
                return v;
            }
            case Op::StrConst:
            case Op::CoordId:
            case Op::AxisStr: {
                double num = 0;
                const i32 s = eval_str(id, &num);
                if (s >= 0) return truth_of_string(s);
                return num;
            }
        }
        return 0;
    }

    double truth_of_string(i32 s) const {
        // numeric value of a string (used in arithmetic); "" and "0" are false
        const std::string& str = g_.strings[static_cast<std::size_t>(s)];
        char* end = nullptr;
        const double v = std::strtod(str.c_str(), &end);
        if (end != str.c_str()) return v;
        return str.empty() || str == "0" ? 0.0 : 0.0;
    }

    double variable(VarId v) {
        switch (v) {
            case VarId::D:
                return dists().Dv;
            case VarId::C:
                return dists().Cv;
            case VarId::Dsqr:
                return dists().Dsqr;
            case VarId::Csqr:
                return dists().Csqr;
            case VarId::X:
                return coord(proc_, 0);
            case VarId::Y:
                return coord(proc_, 1);
            case VarId::Z:
                return coord(proc_, 2);
            case VarId::NbrX:
                return def_query_ ? 0 : coord(nbr_, 0);
            case VarId::NbrY:
                return def_query_ ? 0 : coord(nbr_, 1);
            case VarId::NbrZ:
                return def_query_ ? 0 : coord(nbr_, 2);
        }
        return 0;
    }

    double array_value(ArrId a, i32 k) {
        if (k < 0) k += g_.ndim;  // Perl's negative subscripts
        switch (a) {
            case ArrId::coord:
                return coord(proc_, k);
            case ArrId::nbrcoord:
                return def_query_ ? 0 : coord(nbr_, k);
            default:
                break;
        }
        const Dists& d = dists();
        if (k < 0 || k >= g_.ndim || !d.have[static_cast<std::size_t>(k)]) return 0;
        switch (a) {
            case ArrId::d:
                return d.d[k];
            case ArrId::D:
                return d.D[k];
            case ArrId::c:
                return d.c[k];
            case ArrId::C:
                return d.C[k];
            default:
                return 0;
        }
    }

    double func(FuncId f, i32 a, i32 b) {
        const double x = eval(a);
        switch (f) {
            case FuncId::Abs:
                return std::fabs(x);
            case FuncId::Sqrt:
                return perl_sqrt(x);
            case FuncId::Sin:
                return perl_sin(x);
            case FuncId::Cos:
                return perl_cos(x);
            case FuncId::Log:
                return perl_log(x);
            case FuncId::Exp:
                return perl_exp(x);
            case FuncId::Int:
                return std::trunc(x);
            case FuncId::Atan2:
                return perl_atan2(x, eval(b));
        }
        return 0;
    }

    double call(const Call& c) {
        switch (c.kind) {
            case CallKind::SelfOnly:
                return proc_ == nbr_ ? 1 : 0;
            case CallKind::SelectAll:
                return 1;
            case CallKind::Const:
                return c.p[0];
            case CallKind::Mask: {
                const i64 grp = (c.which == 1 || def_query_) ? proc_ : nbr_;
                return p_.masks[static_cast<std::size_t>(c.index)][static_cast<std::size_t>(grp)] ? 1 : 0;
            }
            case CallKind::ElementIs: {
                const i64 grp = (c.which == 1 || def_query_) ? proc_ : nbr_;
                return static_cast<i64>(c.p[0]) == grp ? 1 : 0;
            }
            case CallKind::TextEq: {
                double num = 0;
                const i32 t = eval_str(c.text, &num);
                const i64 grp = (c.which == 1 || def_query_) ? proc_ : nbr_;
                i32 s;
                double cnum = 0;
                if (c.axis < 0) {
                    s = g_.name_str[static_cast<std::size_t>(grp)];
                } else {
                    s = g_.axis_str[static_cast<std::size_t>(grp * g_.ndim + c.axis)];
                    if (s < 0) cnum = coord(grp, c.axis);
                }
                if (t >= 0 && s >= 0) return t == s ? 1 : 0;
                if (t < 0 && s < 0) return num == cnum ? 1 : 0;
                const std::string& str = g_.strings[static_cast<std::size_t>(t >= 0 ? t : s)];
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.15g", t >= 0 ? cnum : num);
                return str == buf ? 1 : 0;
            }
            case CallKind::ExplicitSet: {
                const Csr& s = p_.sets[static_cast<std::size_t>(c.index)];
                const auto b = s.indices.begin() + s.indptr[static_cast<std::size_t>(proc_)];
                const auto e = s.indices.begin() + s.indptr[static_cast<std::size_t>(proc_) + 1];
                return std::binary_search(b, e, static_cast<i32>(nbr_)) ? 1 : 0;
            }
            case CallKind::SameFeature: {
                if (proc_ == nbr_) return 1;
                const Csr& s = p_.sets[static_cast<std::size_t>(c.index)];
                auto a0 = s.indices.begin() + s.indptr[static_cast<std::size_t>(proc_)];
                auto a1 = s.indices.begin() + s.indptr[static_cast<std::size_t>(proc_) + 1];
                auto b0 = s.indices.begin() + s.indptr[static_cast<std::size_t>(nbr_)];
                auto b1 = s.indices.begin() + s.indptr[static_cast<std::size_t>(nbr_) + 1];
                if (a0 == a1 && b0 == b1) return 1;
                while (a0 != a1 && b0 != b1) {
                    if (*a0 == *b0) return 1;
                    if (*a0 < *b0)
                        ++a0;
                    else
                        ++b0;
                }
                return 0;
            }
            case CallKind::Side: {
                const std::vector<double>& rot = p_.rotations[static_cast<std::size_t>(c.index)];
                const i32 a0 = c.axes[0], a1 = c.axes[1];
                double side;
                if (coord(nbr_, a1) == coord(proc_, a1) && coord(nbr_, a0) == coord(proc_, a0)) {
                    side = 0;
                } else {
                    const double rn = rot[static_cast<std::size_t>(nbr_)], rp = rot[static_cast<std::size_t>(proc_)];
                    side = rn < rp ? -1 : (rn > rp ? 1 : 0);
                }
                return side == c.p[0] ? 1 : 0;
            }
            case CallKind::Block: {
                for (std::size_t k = 0; k < c.sizes.size() && static_cast<i32>(k) < g_.ndim; ++k) {
                    const double sz = c.sizes[k];
                    if (std::isnan(sz)) continue;
                    const double o = k < c.origin.size() ? c.origin[k] : 0.0;
                    const double cv = std::floor((coord(proc_, static_cast<i32>(k)) - o) / sz);
                    const double nv = std::floor((coord(nbr_, static_cast<i32>(k)) - o) / sz);
                    if (cv != nv) return 0;
                }
                return 1;
            }
            default:
                break;
        }
        const Dists& d = dists();
        switch (c.kind) {
            case CallKind::Circle: {
                const double r = c.p[0];
                double dist;
                if (!c.axes.empty()) {
                    double s = 0;
                    for (i32 k : c.axes) {
                        const double v = (k >= 0 && k < g_.ndim) ? d.D[k] : 0;
                        if (v > r) return 0;
                        s += v * v;
                    }
                    dist = perl_sqrt(s);
                } else {
                    dist = d.Dv;
                }
                return dist <= r ? 1 : 0;
            }
            case CallKind::CircleCell: {
                double dist;
                if (!c.axes.empty()) {
                    double s = 0;
                    for (i32 k : c.axes) {
                        const double v = (k >= 0 && k < g_.ndim) ? d.C[k] : 0;
                        s += v * v;
                    }
                    dist = perl_sqrt(s);
                } else {
                    dist = d.Cv;
                }
                return dist <= c.p[0] ? 1 : 0;
            }
            case CallKind::Rectangle: {
                for (std::size_t i = 0; i < c.axes.size(); ++i) {
                    const i32 k = c.axes[i];
                    const double v = (k >= 0 && k < g_.ndim) ? d.D[k] : 0;
                    if (v > c.sizes[i]) return 0;
                    if (round_precision(v) > c.sizes[i] / 2) return 0;
                }
                return 1;
            }
            case CallKind::Annulus: {
                double dist;
                if (!c.axes.empty()) {
                    double s = 0;
                    for (i32 k : c.axes) {
                        const double v = (k >= 0 && k < g_.ndim) ? d.D[k] : 0;
                        if (v > c.p[1]) return 0;
                        s += v * v;
                    }
                    dist = perl_sqrt(s);
                } else {
                    dist = d.Dv;
                }
                return (dist >= c.p[0] && dist <= c.p[1]) ? 1 : 0;
            }
            case CallKind::Square: {
                const double half = c.p[0] / 2;
                for (i32 k = 0; k < g_.ndim; ++k)
                    if (d.have[k] && d.D[k] > half) return 0;
                return 1;
            }
            case CallKind::SquareCell: {
                const double half = c.p[0] / 2;
                for (i32 k = 0; k < g_.ndim; ++k)
                    if (d.have[k] && d.C[k] > half) return 0;
                return 1;
            }
            case CallKind::Ellipse: {
                const i32 a0 = c.axes[0], a1 = c.axes[1];
                const double d0 = d.d[a0], d1 = d.d[a1];
                const double D = perl_sqrt(d0 * d0 + d1 * d1);
                const double bearing = perl_atan2(d0, d1) + c.p[2];
                const double rx = perl_sin(bearing) * D;
                const double ry = perl_cos(bearing) * D;
                double a = (ry * ry) / (c.p[0] * c.p[0]);
                double b = (rx * rx) / (c.p[1] * c.p[1]);
                const double prec = 1.4 * 1e10;
                a = round_precision(a, prec);
                b = round_precision(b, prec);
                return 1 >= (a + b) ? 1 : 0;
            }
            default:
                break;
        }
        return 0;
    }
};

// A uniform grid over the first (up to) two axes for candidate searches.
struct Grid {
    i32 axes = 0;                 // 0: no pruning
    i32 ax[2] = {0, 1};
    double lo[2] = {0, 0}, size[2] = {1, 1}, reach[2] = {0, 0};
    i64 dims[2] = {1, 1};
    std::vector<i64> start;       // cell -> first index into `members`
    std::vector<i32> members;     // group indices, by cell

    i64 cell_of(const Groups& g, i64 grp, i32 k) const {
        const double v = g.coords[static_cast<std::size_t>(grp * g.ndim + ax[k])];
        i64 c = static_cast<i64>(std::floor((v - lo[k]) / size[k]));
        return std::clamp<i64>(c, 0, dims[k] - 1);
    }
};

Grid build_grid(const Groups& g, const Program& p) {
    Grid grid;
    std::vector<i32> use;
    for (i32 k = 0; k < std::min<i32>(g.ndim, 2); ++k) {
        if (k < static_cast<i32>(p.bounds.size()) && std::isfinite(p.bounds[k]) && g.numeric[k]) use.push_back(k);
    }
    if (use.empty() || g.n < 64) return grid;
    grid.axes = static_cast<i32>(use.size());
    i64 total = 1;
    for (i32 k = 0; k < grid.axes; ++k) {
        const i32 a = use[k];
        grid.ax[k] = a;
        double mn = kInf, mx = -kInf;
        for (i64 i = 0; i < g.n; ++i) {
            const double v = g.coords[static_cast<std::size_t>(i * g.ndim + a)];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        // the condition compares rounded distances: widen the reach a little
        const double reach = p.bounds[a] * (1 + 1e-9) + 1e-9 * std::max(std::fabs(mn), std::fabs(mx)) + 1e-9;
        const double cs = (a < static_cast<i32>(g.cell_sizes.size()) && g.cell_sizes[a] > 0) ? g.cell_sizes[a] : 0;
        double size = std::max(reach, cs);
        if (!(size > 0)) size = std::max(mx - mn, 1.0) / 1024;
        i64 dim = static_cast<i64>(std::floor((mx - mn) / size)) + 1;
        if (dim > 4096) {
            dim = 4096;
            size = (mx - mn) / 4095.0;
        }
        grid.lo[k] = mn;
        grid.size[k] = size;
        grid.reach[k] = reach;
        grid.dims[k] = dim;
        total *= dim;
    }
    std::vector<i64> count(static_cast<std::size_t>(total) + 1, 0);
    std::vector<i64> cell(static_cast<std::size_t>(g.n));
    for (i64 i = 0; i < g.n; ++i) {
        i64 c = grid.cell_of(g, i, 0);
        if (grid.axes == 2) c = c * grid.dims[1] + grid.cell_of(g, i, 1);
        cell[static_cast<std::size_t>(i)] = c;
        ++count[static_cast<std::size_t>(c) + 1];
    }
    for (std::size_t c = 1; c < count.size(); ++c) count[c] += count[c - 1];
    grid.start = count;
    grid.members.resize(static_cast<std::size_t>(g.n));
    std::vector<i64> fill(count.begin(), count.end() - 1);
    for (i64 i = 0; i < g.n; ++i) grid.members[static_cast<std::size_t>(fill[static_cast<std::size_t>(cell[i])]++)] = static_cast<i32>(i);
    return grid;
}

void candidates(const Groups& g, const Grid& grid, i64 proc, std::vector<i32>& out) {
    out.clear();
    if (grid.axes == 0) {
        out.resize(static_cast<std::size_t>(g.n));
        for (i64 i = 0; i < g.n; ++i) out[static_cast<std::size_t>(i)] = static_cast<i32>(i);
        return;
    }
    i64 lo[2], hi[2];
    for (i32 k = 0; k < grid.axes; ++k) {
        const double v = g.coords[static_cast<std::size_t>(proc * g.ndim + grid.ax[k])];
        lo[k] = std::max<i64>(0, static_cast<i64>(std::floor((v - grid.reach[k] - grid.lo[k]) / grid.size[k])));
        hi[k] = std::min<i64>(grid.dims[k] - 1, static_cast<i64>(std::floor((v + grid.reach[k] - grid.lo[k]) / grid.size[k])));
    }
    if (grid.axes == 1) {
        for (i64 c = lo[0]; c <= hi[0]; ++c)
            for (i64 m = grid.start[c]; m < grid.start[c + 1]; ++m) out.push_back(grid.members[static_cast<std::size_t>(m)]);
    } else {
        for (i64 c0 = lo[0]; c0 <= hi[0]; ++c0)
            for (i64 c1 = lo[1]; c1 <= hi[1]; ++c1) {
                const i64 c = c0 * grid.dims[1] + c1;
                for (i64 m = grid.start[c]; m < grid.start[c + 1]; ++m) out.push_back(grid.members[static_cast<std::size_t>(m)]);
            }
    }
    std::sort(out.begin(), out.end());
}

}  // namespace

Csr find_neighbours(const Groups& groups, const Program& prog, const std::vector<i32>& procs, const Csr* exclude,
                    bool parallel) {
    const i64 np = static_cast<i64>(procs.size());
    std::vector<std::vector<i32>> rows(static_cast<std::size_t>(np));
    if (prog.always_false) {
        // nothing
    } else if (prog.self_only || prog.always_true) {
        for (i64 r = 0; r < np; ++r) {
            auto& row = rows[static_cast<std::size_t>(r)];
            if (prog.self_only) {
                row.push_back(procs[static_cast<std::size_t>(r)]);
            } else {
                row.resize(static_cast<std::size_t>(groups.n));
                for (i64 i = 0; i < groups.n; ++i) row[static_cast<std::size_t>(i)] = static_cast<i32>(i);
            }
        }
    } else {
        const Grid grid = build_grid(groups, prog);
        std::string error;
        bool failed = false;
        lsm::parallel_for_dynamic(parallel && np > 1, np, [&](i64 r) {
            if (failed) return;
            try {
                thread_local std::vector<i32> cand;
                Evaluator ev(groups, prog);
                const i64 proc = procs[static_cast<std::size_t>(r)];
                candidates(groups, grid, proc, cand);
                auto& row = rows[static_cast<std::size_t>(r)];
                for (i32 j : cand)
                    if (ev.test(proc, j, false)) row.push_back(j);
            } catch (const std::exception& e) {
#pragma omp critical(bio_conditions_error)
                {
                    failed = true;
                    error = e.what();
                }
            }
        });
        if (failed) throw std::runtime_error(error);
    }
    if (exclude) {
        for (i64 r = 0; r < np; ++r) {
            auto& row = rows[static_cast<std::size_t>(r)];
            const auto b = exclude->indices.begin() + exclude->indptr[static_cast<std::size_t>(r)];
            const auto e = exclude->indices.begin() + exclude->indptr[static_cast<std::size_t>(r) + 1];
            row.erase(std::remove_if(row.begin(), row.end(), [&](i32 j) { return std::binary_search(b, e, j); }),
                      row.end());
        }
    }
    Csr out;
    out.indptr.resize(static_cast<std::size_t>(np) + 1, 0);
    for (i64 r = 0; r < np; ++r) out.indptr[static_cast<std::size_t>(r) + 1] = out.indptr[static_cast<std::size_t>(r)] + static_cast<i64>(rows[static_cast<std::size_t>(r)].size());
    out.indices.resize(static_cast<std::size_t>(out.indptr.back()));
    for (i64 r = 0; r < np; ++r)
        std::copy(rows[static_cast<std::size_t>(r)].begin(), rows[static_cast<std::size_t>(r)].end(),
                  out.indices.begin() + out.indptr[static_cast<std::size_t>(r)]);
    return out;
}

std::vector<std::uint8_t> definition_query(const Groups& groups, const Program& prog, bool parallel) {
    std::vector<std::uint8_t> pass(static_cast<std::size_t>(groups.n), 0);
    std::string error;
    bool failed = false;
    lsm::parallel_for_dynamic(parallel && groups.n > 256, groups.n, [&](i64 i) {
        if (failed) return;
        try {
            Evaluator ev(groups, prog);
            pass[static_cast<std::size_t>(i)] = ev.test(i, i, true) ? 1 : 0;
        } catch (const std::exception& e) {
#pragma omp critical(bio_defq_error)
            {
                failed = true;
                error = e.what();
            }
        }
    });
    if (failed) throw std::runtime_error(error);
    return pass;
}

std::vector<double> side_rotation(const Groups& g, i32 a0, i32 a1, double angle) {
    std::vector<double> mn(2, kInf), mx(2, -kInf);
    const i32 ax[2] = {a0, a1};
    for (i64 i = 0; i < g.n; ++i)
        for (int k = 0; k < 2; ++k) {
            const double v = g.coords[static_cast<std::size_t>(i * g.ndim + ax[k])];
            mn[k] = std::min(mn[k], v);
            mx[k] = std::max(mx[k], v);
        }
    const double range0 = mx[0] - mn[0], range1 = mx[1] - mn[1];
    const double ca = perl_cos(angle), sa = perl_sin(angle);
    std::vector<double> out(static_cast<std::size_t>(g.n));
    for (i64 i = 0; i < g.n; ++i) {
        const double x = (g.coords[static_cast<std::size_t>(i * g.ndim + a0)] - mn[0]) / range0;
        const double y = (g.coords[static_cast<std::size_t>(i * g.ndim + a1)] - mn[1]) / range1;
        out[static_cast<std::size_t>(i)] = 0 + round_precision(x * ca - y * sa);
    }
    return out;
}

}  // namespace bio
