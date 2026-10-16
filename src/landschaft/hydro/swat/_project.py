# SPDX-License-Identifier: GPL-3.0-or-later
"""A SWAT+ project: the files of a TxtInOut folder, held in memory."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SwatProject", "load_txtinout"]

# what SWAT+ writes (outputs of earlier runs found in a TxtInOut folder)
_OUTPUT = re.compile(r"(\.txt|\.out|\.fin|\.csv|\.bin|\.log)$|^fort\.\d+$", re.I)


def _normalize(b: bytes) -> bytes:
    """LF line ends and a final newline (the Fortran runtime reads records)."""
    b = b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if b and not b.endswith(b"\n"):
        b += b"\n"
    return b


@dataclass
class SwatProject:
    """The input files of a SWAT+ model (the TxtInOut folder), by name.

    Made by :func:`load_txtinout` (or, later, from rasters); :func:`run` simulates
    it and :meth:`write` writes the folder for SWAT+ itself. The files are SWAT+
    62.0.1's formats; names are matched without regard to case, as on Windows.

    Attributes
    ----------
    files : dict
        ``{file name: contents}`` (bytes, LF line ends).
    """

    files: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.files = {str(k): _normalize(v if isinstance(v, bytes) else str(v).encode()) for k, v in self.files.items()}

    def __repr__(self) -> str:
        return f"SwatProject({len(self.files)} files)"

    def __contains__(self, name: str) -> bool:
        return self._key(name) is not None

    def _key(self, name: str) -> str | None:
        low = name.lower()
        for k in self.files:
            if k.lower() == low:
                return k
        return None

    def text(self, name: str) -> str:
        """The contents of a file."""
        k = self._key(name)
        if k is None:
            raise KeyError(f"no file {name!r} in the project")
        return self.files[k].decode()

    def set_text(self, name: str, text: str) -> None:
        """Replace (or add) a file."""
        k = self._key(name) or name
        self.files[k] = _normalize(text.encode())

    def copy(self) -> SwatProject:
        return SwatProject(dict(self.files))

    def print_file(self) -> str:
        """The name of the print settings file (file.cio's simulation line, else print.prt)."""
        if "file.cio" in self:
            for line in self.text("file.cio").split("\n"):
                p = line.split()
                if len(p) >= 3 and p[0] == "simulation":
                    return p[2]
        return "print.prt"

    def set_print(self, tables=None, *, nyskip: int | None = None) -> None:
        """Choose what SWAT+ prints (its print.prt): the years it skips before printing
        and, for each table, the steps printed.

        Parameters
        ----------
        tables : dict or list, optional
            ``{table: steps}``, with steps from ``"day"``, ``"month"``, ``"year"``,
            ``"aa"`` (average annual), ``"all"`` or ``[]`` (none); a list of tables
            means all steps. Tables: ``hru_wb``, ``hru_nb``, ``hru_ls``, ``hru_pw``,
            ``basin_wb``, ``basin_nb``, ``basin_ls``, ``basin_pw``, ``channel_sd``,
            ``aquifer`` (and the other rows of print.prt by their SWAT+ names).
        nyskip : int, optional
            Years simulated before printing (spin-up).
        """
        name = self.print_file()
        lines = self.text(name).split("\n")
        if nyskip is not None:
            vals = lines[2].split()
            vals[0] = str(int(nyskip))
            lines[2] = " ".join(f"{v:<10}" for v in vals)
        if isinstance(tables, (list, tuple, set)):
            tables = {t: "all" for t in tables}
        for table, steps in (tables or {}).items():
            if isinstance(steps, str):
                steps = ["day", "month", "year", "aa"] if steps == "all" else [steps]
            bad = set(steps) - {"day", "month", "year", "aa"}
            if bad:
                raise ValueError(f"unknown print steps {sorted(bad)} (day, month, year, aa)")
            row = f"{table:<24}" + "".join(f"{'y' if s in steps else 'n':>14}" for s in ("day", "month", "year", "aa"))
            for k, line in enumerate(lines):
                p = line.split()
                if len(p) == 5 and p[0] == table:
                    lines[k] = row
                    break
            else:
                raise ValueError(f"{name} has no row {table!r}")
        self.set_text(name, "\n".join(lines))

    def write(self, path) -> Path:
        """Write the files into a folder (made if needed), for SWAT+ to run."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for name, b in self.files.items():
            (out / name).write_bytes(b)
        return out


def load_txtinout(path) -> SwatProject:
    """Read a SWAT+ TxtInOut folder (the outputs of earlier runs in it are left out).

    Examples
    --------
    >>> import landschaft as ls
    >>> project = ls.hydro.swat.load_txtinout(ls.example_data("swat_hubbard_brook"))  # doctest: +SKIP
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {os.fspath(path)}")
    files = {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file() and not _OUTPUT.search(p.name)}
    if "file.cio" not in {k.lower() for k in files}:
        raise ValueError(f"no file.cio in {os.fspath(path)}: not a SWAT+ TxtInOut folder")
    return SwatProject(files)
