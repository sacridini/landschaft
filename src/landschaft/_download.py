# SPDX-License-Identifier: GPL-3.0-or-later
"""Data downloaded on first use and kept in a cache folder.

Reference data too large to ship with the package (coastlines, country
centroids, institutions...) are downloaded the first time a tool needs them
and read from the cache afterwards. Every file is pinned by its SHA-256, so a
result never changes because a remote file did. The cache is
``LANDSCHAFT_CACHE`` when set, else the platform's cache folder
(``%LOCALAPPDATA%\\landschaft\\cache`` on Windows, ``~/Library/Caches/landschaft``
on macOS, ``$XDG_CACHE_HOME/landschaft`` or ``~/.cache/landschaft`` elsewhere).
On a computer without internet access (a cluster's compute node, often),
``landschaft fetch`` downloads them beforehand on one that has it (the login
node), into a cache both see.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path

__all__ = ["cache_dir", "fetch", "prefetch"]

#: What ``landschaft fetch`` downloads: the first folder of a cached file's name -> the command's argument.
FETCH = {"sdm": "sdm", "simulate": "examples"}

# The connection itself failed: no name resolution, no route, refused, timed out.
_ERRNOS = {errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT, errno.ECONNRESET,
           errno.ENETDOWN, errno.EHOSTDOWN}

OFFLINE_HINT = ("This computer seems to have no internet access (common on the compute nodes of a cluster): "
                "download what is needed beforehand on a computer that has it, such as the cluster's login node")


def connection_failed(error: BaseException | None) -> bool:
    """Whether ``error`` (or what caused it) is a failure to reach the server at all, and not an answer
    from it (an HTTP error)."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (socket.gaierror, socket.timeout, TimeoutError, ConnectionRefusedError,
                              ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(error, OSError) and error.errno in _ERRNOS:
            return True
        reason = getattr(error, "reason", None)  # urllib.error.URLError
        if isinstance(reason, BaseException):
            error = reason
            continue
        error = error.__cause__ or error.__context__
    return False


def offline_message(what: str, fetch: str | None = None) -> str:
    """The message for ``what`` that could not be reached; ``fetch``: the command that downloads it
    beforehand."""
    return f"could not reach {what}. {OFFLINE_HINT}" + (f", with: {fetch}" if fetch else "") + "."


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
    offline = True
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
            offline = offline and connection_failed(err)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if offline:
        command = FETCH.get(name.split("/", 1)[0])
        raise OSError(offline_message(f"the servers of {name!r} ({', '.join(e.split(': ')[0] for e in errors)})",
                                      f"landschaft fetch {command}" if command else None)
                      + f" Or save the file by hand as {path}.")
    raise OSError(
        f"could not download {name!r}:\n  " + "\n  ".join(errors)
        + f"\nWithout internet access, download it by hand and save it as {path}"
    )


def prefetch(what: str = "all", out=print) -> list[str]:
    """Download into the cache the data that landschaft downloads on first use (``landschaft fetch``):
    ``"sdm"``, the reference data of :func:`landschaft.sdm.reference_data` (CoordinateCleaner's
    tables and Natural Earth's layers, ~40 MB); ``"examples"``, the example data downloaded by
    ``example_data`` (the Sao Lourenco basin of :mod:`landschaft.simulate`); ``"all"``, both.
    Returns what was fetched."""
    if what not in ("all", *FETCH.values()):
        raise ValueError(f"landschaft fetch takes {', '.join(['all', *FETCH.values()])}, got {what!r}")
    done = []
    if what in ("all", "sdm"):
        from .sdm._refdata import _DATA, _NATURAL_EARTH, reference_data

        for name in [*_DATA, *_NATURAL_EARTH]:
            out(f"sdm reference data: {name}")
            reference_data(name)
            done.append(f"sdm/{name}")
    if what in ("all", "examples"):
        from .simulate._examples import example_data

        out("example data: simulate sao_lourenco")
        example_data("sao_lourenco")
        done.append("simulate/sao_lourenco")
    out(f"In {cache_dir()}")
    return done
