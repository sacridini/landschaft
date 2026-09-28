// SPDX-License-Identifier: GPL-3.0-or-later
//
// Biodiverse's spatial conditions: neighbourhoods of groups.
//
// A condition arrives compiled from its Perl text (landschaft/bio/_conditions.py)
// as a small expression tree evaluated for each (processing group, candidate
// neighbour) pair with Biodiverse's semantics: its distances (d, D, c, C
// rounded to 1e-10), its sp_* functions, Perl's truth and mingw's x87 maths
// (lsm/rmath.hpp, the library Strawberry Perl uses). Candidates come from a
// uniform grid when the condition bounds the distance on the first axes;
// processing groups run in parallel.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bio {

using i64 = std::int64_t;
using i32 = std::int32_t;

enum class Op : i32 {
    Num = 0,     // v
    Var,         // a: variable id (VarId)
    IdxVar,      // a: array id (ArrId), b: index node
    Neg,         // a
    Not,         // a
    Add, Sub, Mul, Div, Mod, Pow,
    Lt, Le, Gt, Ge, Eq, Ne, Cmp,
    And, Or, Dor,  // Perl's && || // (value of the deciding operand)
    Tern,        // a ? b : c
    Func,        // a: FuncId, b: arg node, c: second arg node (atan2) or -1
    StrEq, StrNe,  // a, b: string nodes
    Call,        // a: call index
    Local,       // a: slot
    SetLocal,    // a: slot, b: value node
    Seq,         // a: first child index into `seq`, b: count
    // string nodes (only below string comparisons / as text arguments)
    StrConst,    // a: string id
    CoordId,     // a: 0 processing group, 1 neighbour
    AxisStr,     // a: 0 processing, 1 neighbour; b: axis node
    Undef,
};

enum class VarId : i32 { D = 0, C, Dsqr, Csqr, X, Y, Z, NbrX, NbrY, NbrZ };
enum class ArrId : i32 { d = 0, D, c, C, coord, nbrcoord };
enum class FuncId : i32 { Abs = 0, Sqrt, Sin, Cos, Log, Exp, Int, Atan2 };

enum class CallKind : i32 {
    Circle = 0,     // p0 radius; axes (empty: all)
    CircleCell,     // p0 radius; axes
    Rectangle,      // sizes; axes
    Annulus,        // p0 inner, p1 outer; axes
    Square,         // p0 size
    SquareCell,     // p0 size
    Ellipse,        // p0 major, p1 minor, p2 rotate angle; axes (2)
    Block,          // sizes (NaN: axis ignored), origin
    SelfOnly,
    SelectAll,
    Side,           // p0 sign wanted (-1 left, 1 right, 0 in line); rot = rotated x per group
    Mask,           // mask index; which (0 neighbour, 1 processing group)
    Const,          // p0 value
    SameFeature,    // set index: features per group (CSR); true if both in none or share one
    ExplicitSet,    // set index: neighbours per processing group (CSR over groups)
    ElementIs,      // p0 group index (-1: none); which
    TextEq,         // text node `text`; comparator axis (-1: group name) and which
};

struct Node {
    Op op;
    i32 a = -1, b = -1, c = -1;
    double v = 0;
};

struct Call {
    CallKind kind;
    std::vector<double> p;
    std::vector<i32> axes;
    std::vector<double> sizes;
    std::vector<double> origin;
    i32 which = 0;
    i32 index = -1;   // mask / set / rotation index
    i32 text = -1;    // text node (TextEq)
    i32 axis = -1;    // comparator axis (TextEq)
};

struct Csr {
    std::vector<i64> indptr;
    std::vector<i32> indices;  // sorted within each row
};

struct Program {
    std::vector<Node> nodes;
    std::vector<i32> seq;         // children of Seq nodes
    i32 root = -1;
    i32 n_locals = 0;
    std::vector<Call> calls;
    std::vector<std::vector<std::uint8_t>> masks;
    std::vector<Csr> sets;
    std::vector<std::vector<double>> rotations;  // rotated x per group (Side)
    // distances Biodiverse computes (get_distances)
    bool use_euc = false;         // $D, or a call needing all axes
    bool use_cell = false;        // $C
    bool cell_lists = false;      // $c[i] / $C[i] used
    std::vector<i32> iter_axes;   // axes computed when neither $D nor $C is used
    // half-widths of the box holding every neighbour, per axis (inf: unbounded)
    std::vector<double> bounds;
    bool self_only = false, always_true = false, always_false = false;
};

struct Groups {
    i64 n = 0;
    i32 ndim = 0;
    std::vector<double> coords;        // n * ndim (text axes: their rank)
    std::vector<std::uint8_t> numeric; // per axis
    std::vector<i32> axis_str;         // n * ndim string ids (-1 for numeric values)
    std::vector<i32> name_str;         // string id of each group name
    std::vector<double> cell_sizes;
    std::vector<std::string> strings;  // string table
};

// Neighbours of each processing group: rows follow `procs`, columns are group
// indices in ascending order. `exclude` (optional, rows as `procs`) removes
// groups, e.g. the first neighbour set from the second.
Csr find_neighbours(const Groups& groups, const Program& prog, const std::vector<i32>& procs, const Csr* exclude,
                    bool parallel);

// Groups passing a definition query (evaluated with the group as both the
// processing group and the neighbour, as Biodiverse does).
std::vector<std::uint8_t> definition_query(const Groups& groups, const Program& prog, bool parallel);

// Rotated, scaled x coordinate of every group for the sidedness conditions.
std::vector<double> side_rotation(const Groups& groups, i32 axis0, i32 axis1, double angle);

// Perl's round_to_precision_aa: POSIX round(x * p) / p.
double round_precision(double x, double precision = 1e10);

}  // namespace bio
