# SPDX-License-Identifier: GPL-3.0-or-later
"""Add dump records to the reference SWAT+ at the printed outputs: before each formatted
WRITE of hru_output, basin_output, aquifer_output and sd_channel_output, a
``call ls_push`` of each printed value and a ``call ls_flush (unit, id)`` (the derived types and reals
after the object's name), so the port's results can be compared bit for bit with
what SWAT+ prints. The dump tag is the Fortran unit (2000-2083, 2500-2523).

    python tools/swat/patch_output_dumps.py <swatplus>/src

Idempotent: a file that already has the calls is left as it is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = {
    "hru_output.f90": range(2000, 2044),
    "basin_output.f90": range(2050, 2084),
    "aquifer_output.f90": range(2520, 2524),
    "sd_channel_output.f90": range(2500, 2504),
}
MARK = "!ls_dump_output"


def split_top(s: str):
    out, depth, cur, q = [], 0, "", None
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def patch(path: Path, units) -> int:
    lines = path.read_text().splitlines()
    if any(MARK in l for l in lines):
        return 0
    out, n, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"(\s*)write\s*\((\d+)\s*,\s*\d+\s*\)(.*)", line, re.I)
        if m and int(m.group(2)) in units:
            stmt, j = m.group(3), i
            while stmt.split("!")[0].rstrip().endswith("&"):
                j += 1
                stmt = stmt.split("!")[0].rstrip()[:-1] + " " + lines[j].strip().lstrip("&")
            items = split_top(stmt.split("!")[0])
            ident = items[4]
            if ident.startswith(('"', "'")):
                ident = "1"
            name = next(k for k, it in enumerate(items) if it.endswith("%name"))
            vals = [it for it in items[name + 1:] if not it.startswith(("lum(", "'", '"')) and it not in ("cal_sim", "cal_adj")]
            if int(m.group(2)) == 2053:
                vals = vals[:1]
            ind = m.group(1)
            for v in vals:
                out.append(f"{ind}call ls_push ({v})  {MARK}")
            out.append(f"{ind}call ls_flush ({m.group(2)}, {ident})  {MARK}")
            n += 1
        out.append(line)
        i += 1
    text = "\n".join(out) + "\n"
    if "use ls_dump_module" not in text:
        text = re.sub(r"^(\s*subroutine \w+[^\n]*\n)", r"\1      use ls_dump_module, only : ls_push, ls_flush\n", text, count=1,
                      flags=re.M)
    path.write_text(text, newline="\n")
    return n


def main(src: str) -> None:
    for f, units in FILES.items():
        print(f, patch(Path(src) / f, units), "dump calls")


if __name__ == "__main__":
    main(sys.argv[1])
