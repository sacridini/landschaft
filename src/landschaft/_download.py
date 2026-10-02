# SPDX-License-Identifier: GPL-3.0-or-later
"""Data downloaded on first use and kept in a cache folder.

Reference data too large to ship with the package (coastlines, country
centroids, institutions...) are downloaded the first time a tool needs them
and read from the cache afterwards. Every file is pinned by its SHA-256, so a
result never changes because a remote file did. The cache is
``LANDSCHAFT_CACHE`` when set, else the platform's cache folder
(``%LOCALAPPDATA%\\landschaft\\cache`` on Windows, ``~/Library/Caches/landschaft``
on macOS, ``$XDG_CACHE_HOME/landschaft`` or ``~/.cache/landschaft`` elsewhere).
Without internet access, copy the files into that folder by hand.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

__all__ = ["cache_dir", "fetch"]


def cache_dir() -> Path:
    """The folder of the downloaded data (created if missing)."""
    env = os.environ.get("LANDSCHAFT_CACHE")
    if env:
        base = Path(env)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "landschaft" / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "landschaft"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "landschaft"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, urls: str | list[str], sha256: str | None = None, *, refresh: bool = False) -> Path:
    """Path of a cached file, downloading it first if needed.

    Parameters
    ----------
    name
        File name in the cache (sub-folders allowed, e.g. ``"sdm/land.gpkg"``).
    urls
        Where to download it from: the mirrors are tried in order.
    sha256
        Expected SHA-256 of the file; a file that does not match is downloaded
        again, and a download that does not match is an error.
    refresh
        Download again even if the file is in the cache.
    """
    path = cache_dir() / name
    if path.exists() and not refresh and (sha256 is None or _sha256(path) == sha256):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in [urls] if isinstance(urls, str) else urls:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        os.close(fd)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "landschaft"})
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
            got = _sha256(Path(tmp))
            if sha256 is not None and got != sha256:
                raise OSError(f"checksum mismatch (got {got}, expected {sha256})")
            os.replace(tmp, path)
            return path
        except Exception as err:  # try the next mirror
            errors.append(f"{url}: {err}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise OSError(
        f"could not download {name!r}:\n  " + "\n  ".join(errors)
        + f"\nWithout internet access, download it by hand and save it as {path}"
    )
