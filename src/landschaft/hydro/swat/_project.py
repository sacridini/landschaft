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
