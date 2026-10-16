# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate src/cpp/swat/modules.hpp from SWAT+'s Fortran modules.

Usage::

    python tools/swat/f2cpp.py <swatplus>/src

Each derived type becomes a struct with the same component names, types, initial
values and order (the order is what a list-directed READ of the whole type follows:
a ``read_items(T&)`` lists them); each module's variables become a struct the
simulation state (swat::Sim) derives from. REAL is float, INTEGER int, CHARACTER
std::string (the declared length in ``len_of``), fixed arrays Fix<T, N...> (Fortran
bounds), allocatable ones Vec / Mat / Cube, pointers raw pointers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "cpp" / "swat" / "modules.hpp"

# the modules of the port, in an order where each type is defined before its use
MODULES = [
    "input_file_module", "time_module", "basin_module", "climate_module", "maximum_data_module", "hydrograph_module",
    "plant_data_module", "landuse_data_module", "mgt_operations_module", "tillage_data_module",
    "fertilizer_data_module", "pesticide_data_module", "pathogen_data_module", "urban_data_module", "septic_data_module", "soil_data_module", "soil_module",
    "plant_module", "carbon_module", "organic_mineral_mass_module", "hru_module", "hydrology_data_module",
    "topography_data_module", "output_landscape_module", "aquifer_module", "sd_channel_module",
    "channel_velocity_module", "channel_data_module", "channel_module", "ru_module", "calibration_data_module",
    "conditional_module", "water_body_module", "erosion_module", "constituent_mass_module", "tiles_data_module",
    "reservoir_data_module", "reservoir_module", "exco_module", "dr_module", "recall_module",
]

KEYWORDS = {"int", "long", "float", "double", "char", "class", "new", "delete", "default", "register", "union",
            "this", "template", "typename", "operator", "private", "public", "protected", "friend", "virtual",
            "return", "switch", "case", "do", "for", "while", "if", "else", "auto", "const", "static", "extern",
            "signed", "unsigned", "short", "void", "bool", "true", "false", "struct", "enum", "goto", "break",
            "continue", "inline", "volatile", "namespace", "using", "try", "catch", "throw", "sizeof", "and",
            "or", "not", "xor", "export", "import", "module"}


def cname(n: str) -> str:
    n = n.lower()
    return n + "_" if n in KEYWORDS else n


def strip_comment(line: str) -> str:
    out, q = [], None
    for c in line:
        if q:
            out.append(c)
            if c == q:
                q = None
        elif c in "'\"":
            q = c
            out.append(c)
        elif c == "!":
            break
        else:
            out.append(c)
    return "".join(out).rstrip()


def logical_lines(text: str):
    """Fortran free-form statements: comments dropped, '&' continuations joined."""
    buf = ""
    for raw in text.split("\n"):
        line = strip_comment(raw)
        if not line.strip():
            continue
        s = line.strip()
        if buf:
            if s.startswith("&"):
                s = s[1:]
            buf += " " + s
        else:
            buf = s
        if buf.endswith("&"):
            buf = buf[:-1]
            continue
        yield buf
        buf = ""
    if buf:
        yield buf


def split_top(s: str, sep: str = ","):
    """Split at commas outside parentheses, brackets and quotes."""
    parts, depth, q, cur = [], 0, None, ""
    for c in s:
        if q:
            cur += c
            if c == q:
                q = None
            continue
        if c in "'\"":
            q = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def fortran_value(v: str, base: str) -> str:
    """A Fortran initial value as C++ (float literals for REAL)."""
    v = v.strip()
    if v.startswith(("'", '"')):
        inner = v[1:-1].replace('\\', '\\\\').replace('"', '\\"')
        return f'"{inner.rstrip()}"'
    lv = v.lower()
    if lv in (".true.", ".false."):
        return "true" if lv == ".true." else "false"
    if base in ("int", "long long") and re.fullmatch(r"[-+]?\d*\.\d*(?:[edED][-+]?\d+)?", v):
        return str(int(float(v.lower().replace("d", "e"))))
    if base == "float" or base == "double":
        m = re.fullmatch(r"([-+]?)(\d*)\.?(\d*)(?:[edED]([-+]?\d+))?", v)
        if not m:
            return v  # an expression; left for the compiler to reject
        sign, a, b, e = m.groups()
        num = f"{a or '0'}.{b or '0'}" + (f"e{e}" if e else "")
        return f"{sign}{num}" + ("f" if base == "float" and "d" not in lv else "")
    return v


class Decl:
    def __init__(self, base, length, dims, alloc, pointer, name, init, typ=None):
        self.base, self.length, self.dims, self.alloc, self.pointer = base, length, dims, alloc, pointer
        self.name, self.init, self.typ = name, init, typ


def parse_decl(stmt: str):
    """A type declaration statement -> list of Decl, or None."""
    m = re.match(r"(?i)^(real\s*\*\s*8|double\s+precision|real(?:\s*\(\s*(?:kind\s*=\s*)?\w+\s*\))?|integer(?:\s*\(\s*(?:kind\s*=\s*)?\w+\s*\))?|"
                 r"logical|character\s*(?:\*\s*\d+|\(\s*(?:len\s*=\s*)?[\w*]+\s*\))?|type\s*\(\s*\w+\s*\))\s*(.*)$", stmt)
    if not m:
        return None
    head, rest = m.group(1), m.group(2)
    if "::" in rest:
        attrs, names = rest.split("::", 1)
    else:
        if rest.lower().startswith(("function", "subroutine")):
            return None
        attrs, names = "", rest
    hl = head.lower().replace(" ", "")
    typ = None
    length = None
    if hl.startswith("real*8") or hl.startswith("doubleprecision") or "kind=8" in hl or hl == "real(8)":
        base = "double"
    elif hl.startswith("real"):
        base = "float"
    elif hl.startswith("integer"):
        base = "long long" if ("8" in hl) else "int"
    elif hl.startswith("logical"):
        base = "bool"
    elif hl.startswith("character"):
        base = "std::string"
        lm = re.search(r"(\d+)", hl)
        length = int(lm.group(1)) if lm else 1
    else:
        base = "type"
        typ = re.search(r"\(\s*(\w+)\s*\)", head).group(1).lower()
    alist = [a.strip().lower() for a in split_top(attrs)] if attrs.strip() else []
    dims = None
    alloc = "allocatable" in alist
    pointer = "pointer" in alist
    if "parameter" in alist:
        param = True
    else:
        param = False
    for a in alist:
        dm = re.match(r"dimension\s*\((.*)\)$", a)
        if dm:
            dims = [d.strip() for d in split_top(dm.group(1))]
    out = []
    for item in split_top(names):
        im = re.match(r"(\w+)\s*(\((.*?)\))?\s*(?:\*\s*(\d+))?\s*(?:=>?\s*(.*))?$", item, re.S)
        if not im:
            continue
        name, _, idims, clen, init = im.groups()
        d = dims
        if idims and not item.lstrip().startswith(name + "=") and "(" in item.split("=")[0]:
            d = [x.strip() for x in split_top(idims)]
        ln = int(clen) if clen else length
        dec = Decl(base, ln, d, alloc or (d is not None and all(x == ":" for x in d)), pointer, name, init, typ)
        dec.param = param
        out.append(dec)
    return out


def cpp_decl(d: Decl) -> str:
    base = d.typ if d.base == "type" else d.base
    name = cname(d.name)
    if d.pointer and not d.dims:
        return f"{base}* {name} = nullptr;"
    if d.dims:
        if d.alloc or any(x == ":" for x in d.dims):
            n = len(d.dims)
            cont = {1: "Vec", 2: "Mat", 3: "Cube"}[n]
            return f"{cont}<{base}> {name};"
        # fixed bounds: Fix<T, lo, hi> / Fix2
        bounds = []
        for x in d.dims:
            if ":" in x:
                lo, hi = x.split(":")
            else:
                lo, hi = "1", x
            bounds.append((lo.strip(), hi.strip()))
        if len(bounds) == 1:
            t = f"Fix<{base}, {bounds[0][0]}, {bounds[0][1]}>"
        else:
            t = f"Fix2<{base}, {bounds[0][0]}, {bounds[0][1]}, {bounds[1][0]}, {bounds[1][1]}>"
        init = ""
        if d.init is not None:
            v = d.init.strip()
            if v.startswith(("(/", "[")):
                vals = [fortran_value(x, d.base) for x in split_top(v.strip("(/)[] "))]
                init = "{" + ", ".join(vals) + "}"
            else:
                init = "{" + fortran_value(v, d.base) + "}"
                return f"{t} {name} = {t}::filled({fortran_value(v, d.base)});"
        return f"{t} {name}{init};"
    init = ""
    if d.init is not None and d.base != "type":
        init = " = " + fortran_value(d.init, d.base)
    elif d.base in ("float", "double", "int", "long long"):
        init = "{}"
    elif d.base == "bool":
        init = " = false"
    return f"{base} {name}{init};"


def parse_module(path: Path):
    text = path.read_text(errors="ignore")
    types, variables, order = {}, [], []
    cur = None
    in_contains = False
    for stmt in logical_lines(text):
        low = stmt.lower()
        if re.match(r"^contains\b", low):
            in_contains = True
        if in_contains:
            continue
        m = re.match(r"^type\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)\s*$", low)
        if m and not low.startswith("type ("):
            cur = m.group(1)
            types[cur] = []
            order.append(("type", cur))
            continue
        if re.match(r"^end\s*type", low):
            cur = None
            continue
        if low.startswith(("module ", "implicit", "use ", "end module", "private", "public", "sequence", "save")):
            continue
        decls = parse_decl(stmt)
        if decls is None:
            continue
        if cur:
            types[cur].extend(decls)
        else:
            variables.extend(decls)
    return types, variables, order


def main(src: str) -> None:
    srcdir = Path(src)
    out = ["// SPDX-License-Identifier: GPL-3.0-or-later",
           "//",
           "// GENERATED by tools/swat/f2cpp.py from SWAT+ 62.0.1's Fortran modules: the derived",
           "// types and module variables of the port, with the Fortran names, initial values and",
           "// component order. Do not edit by hand.",
           "",
           "#pragma once",
           "",
           "#include <string>",
           "",
           "#include \"swat/farray.hpp\"",
           "#include \"swat/fio.hpp\"",
           "",
           "namespace swat {",
           ""]
    all_types = {}
    for mod in MODULES:
        p = srcdir / f"{mod}.f90"
        types, variables, order = parse_module(p)
        # a type name another module already defines: this module's copy is <module>_<name>
        rename = {t: f"{mod}_{t}" for t in types if t in all_types}
        if rename:
            for comps in list(types.values()) + [variables]:
                for c in comps:
                    if c.base == "type" and c.typ in rename:
                        c.typ = rename[c.typ]
            types = {rename.get(k, k): v for k, v in types.items()}
            order = [(a, rename.get(b, b)) for a, b in order]
        out.append(f"// ---- {mod} " + "-" * max(0, 80 - len(mod) - 9))
        for _, tname in order:
            comps = types[tname]
            all_types[tname] = comps
            out.append(f"struct {tname} {{")
            for d in comps:
                out.append("    " + cpp_decl(d))
                if d.base == "std::string" and d.length:
                    out.append(f"    static constexpr std::size_t {cname(d.name)}_len = {d.length};")
            out.append("};")
            # list-directed read of the whole type (allocatable components cannot be read)
            if all(not (c.alloc or c.pointer) for c in comps):
                items = []
                def expand(prefix, comps_):
                    for c in comps_:
                        nm = prefix + cname(c.name)
                        if c.pointer or c.alloc:
                            return False
                        if c.base == "type":
                            if c.dims:
                                return False
                            sub = all_types.get(c.typ)
                            if sub is None or expand(nm + ".", sub) is False:
                                return False
                        elif c.base == "std::string":
                            if c.dims:
                                return False
                            items.append(f"fio::chars(x.{nm}, {c.length})")
                        elif c.dims:
                            items.append(f"x.{nm}.items()")
                        else:
                            items.append(f"x.{nm}")
                    return True
                if expand("", comps) is not False and items:
                    out.append(f"inline fio::ItemList items({tname}& x) {{")
                    out.append("    fio::ItemList l;")
                    for it in items:
                        out.append(f"    fio::add(l, {it});")
                    out.append("    return l;")
                    out.append("}")
                    out.append(f"inline int read_items(fio::File& f, {tname}& x) {{ return f.read(items(x)); }}")
            out.append("")
        vars_ = [v for v in variables if not getattr(v, "param", False)]
        params = [v for v in variables if getattr(v, "param", False)]
        for v in params:
            base = v.typ if v.base == "type" else v.base
            if v.dims:
                continue
            out.append(f"inline constexpr {'const char*' if base == 'std::string' else base} {cname(v.name)} = "
                       f"{fortran_value(v.init or '0', v.base)};")
        out.append(f"struct {mod} {{")
        for v in vars_:
            out.append("    " + cpp_decl(v))
            if v.base == "std::string" and v.length and not v.dims:
                out.append(f"    static constexpr std::size_t {cname(v.name)}_len = {v.length};")
        out.append("};")
        out.append("")
    out.append("}  // namespace swat")
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}: {sum(len(c) for c in all_types.values())} components of {len(all_types)} types")


if __name__ == "__main__":
    main(sys.argv[1])
