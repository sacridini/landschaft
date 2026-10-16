# SPDX-License-Identifier: GPL-3.0-or-later
"""The work handed out: jobs made of tasks, each a registered operation of landschaft
with JSON parameters and the input files it reads, and the helpers to move those files
and a task's outputs around.

A task never carries code: a helper runs only the operations of ``_ops`` (each checks
its own parameters), in a child process of its own (``_slots``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PARTIAL = ".landschaft-partial"
_PROBE = 1 << 20
_FULL_HASH = 256 << 20    # files up to this size are named by the hash of all their bytes


def probe(path, size: int | None = None) -> str:
    """A cheap content check: SHA-256 of the size and the first and last MiB."""
    size = os.path.getsize(path) if size is None else size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_PROBE))
        if size > _PROBE:
            f.seek(max(_PROBE, size - _PROBE))
            h.update(f.read(_PROBE))
    return h.hexdigest()


def _content_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def network_path(path: Path) -> str:
    """The path as other computers may see it: a mapped Windows drive as its UNC path."""
    text = str(Path(path).resolve())
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    if sys.platform == "win32" and len(text) > 1 and text[1] == ":":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            n = ctypes.c_ulong(len(buf))
            if ctypes.windll.mpr.WNetGetConnectionW(text[:2], buf, ctypes.byref(n)) == 0:
                text = buf.value.rstrip("\\") + text[2:]
        except Exception:  # noqa: BLE001, S110
            pass
    return text


def check_name(name: str) -> str:
    """A file name inside a task folder: relative, without ``..``, drive or backslash."""
    text = str(name)
    parts = PurePosixPath(text).parts
    if (not text or text.startswith("/") or "\\" in text or ":" in text or not parts
            or any(p in ("..", ".") for p in parts)):
        raise ValueError(f"not a file name inside a task: {name!r}")
    return text


@dataclass
class InputFile:
    """A file a task reads, at ``name`` (a relative POSIX path) inside its input folder."""

    name: str
    path: Path
    size: int = 0
    mtime_ns: int = 0
    _key: str = ""
    _probe: str = ""
    _shared: str = ""

    @classmethod
    def of(cls, path, name: str | None = None) -> "InputFile":
        path = Path(path)
        st = path.stat()
        return cls(check_name(name if name is not None else path.name), path, st.st_size, st.st_mtime_ns)

    @property
    def probe(self) -> str:
        if not self._probe:
            self._probe = probe(self.path, self.size)
        return self._probe

    @property
    def key(self) -> str:
        """Names the file in a helper's cache: the hash of its bytes (of its size, date and
        first and last MiB for large files), so the same data is sent once."""
        if not self._key:
            text = (_content_hash(self.path) if self.size <= _FULL_HASH
                    else f"{self.size}|{self.mtime_ns}|{self.probe}")
            self._key = hashlib.sha256(text.encode()).hexdigest()[:32]
        return self._key

    @property
    def shared_path(self) -> str:
        if not self._shared:
            self._shared = network_path(self.path)
        return self._shared


@dataclass
class Task:
    """One unit of work: ``op(params)`` on the files, in a folder where each file is at its
    name (on this computer: ``root``, where they already are)."""

    name: str
    op: str
    params: dict
    files: list = field(default_factory=list)      # InputFile
    root: Path | None = None
    weight: float = 0.0                            # for the order (heaviest first) and the estimates
    id: int = -1

    def __post_init__(self):
        if not self.weight:
            self.weight = float(sum(f.size for f in self.files)) or 1.0


@dataclass
class Job:
    """Tasks run together; ``collect(task, folder, stats)`` takes each finished task's
    outputs on this computer (the folder is removed after it returns)."""

    label: str
    tasks: list
    collect: Callable | None = None
    threads: int | None = None            # per task; None: each machine splits its cores among the tasks
    results_dir: Path | None = None       # helpers that see this folder (a NAS) write their outputs into it
    id: int = 0

    def threads_on(self, cores: int) -> int:
        """Threads per task on a machine with ``cores`` cores."""
        if self.threads:
            return max(1, min(int(self.threads), cores))
        return max(1, cores // max(1, min(cores, len(self.tasks))))


def check_placed(folder: Path) -> bool:
    """Outputs a helper wrote into the results folder: regular files only."""
    if not folder.is_dir() or folder.is_symlink():
        return False
    return all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in folder.rglob("*"))


def pack(folder, archive) -> int:
    """The files of an output folder as an (uncompressed: GeoTIFFs already are) tar."""
    folder = Path(folder)
    with tarfile.open(archive, "w") as tar:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    return os.path.getsize(archive)


def unpack(archive, dest) -> None:
    """Extract a helper's outputs: regular files only, never outside ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(fileobj=archive, mode="r|") if hasattr(archive, "read") else tarfile.open(archive) as tar:
        for member in tar:
            name = member.name
            if not member.isfile():
                raise ValueError(f"refusing the archive member {name!r}")
            try:
                check_name(name)
            except ValueError:
                raise ValueError(f"refusing the archive member {name!r}") from None
            target = (dest / name).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing the archive member {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            with open(target, "wb") as f:
                while chunk := src.read(1 << 20):
                    f.write(chunk)


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} TB"


def format_seconds(s: float) -> str:
    if s < 9.95:
        return f"{s:.1f} s"
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s" if s < 600 else f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"
