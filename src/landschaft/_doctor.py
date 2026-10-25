# SPDX-License-Identifier: GPL-3.0-or-later
"""``landschaft doctor``: what landschaft finds on this computer, the first thing to look at on a
cluster and to paste in an issue: the threads and where their number comes from, the batch job, the
caches and whether they can be written, the network, the display and the versions."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["doctor", "report"]

# What landschaft downloads from, by what needs it.
SERVICES = (
    ("GBIF (sdm.download_gbif)", "https://api.gbif.org/v1"),
    ("CRAN (sdm.reference_data)", "https://cloud.r-project.org"),
    ("Natural Earth (sdm.reference_data)", "https://naciscdn.org"),
    ("Zenodo (example data)", "https://zenodo.org"),
    ("OpenStreetMap (plot basemaps)", "https://tile.openstreetmap.org"),
)

VARIABLES = ("LANDSCHAFT_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "NCPUS",
             "LSB_DJOB_NUMPROC", "LANDSCHAFT_CACHE", "LANDSCHAFT_CONFIG_DIR", "XDG_CACHE_HOME")


def probe(url: str, timeout: float = 5.0) -> str | None:
    """None when ``url`` answers (with any HTTP status), else why not, in a few words."""
    import socket
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "landschaft"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return None
    except urllib.error.HTTPError:
        return None
    except Exception as e:  # noqa: BLE001 - reported, not raised
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.gaierror):
            return "the name does not resolve (no DNS)"
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return f"no answer in {timeout:g} s"
        if isinstance(reason, ConnectionRefusedError):
            return "connection refused"
        return str(reason) or type(reason).__name__


def writable(folder: Path) -> bool:
    """Whether a file can be created in ``folder`` (created if missing)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".landschaft-doctor-"):
            pass
        return True
    except OSError:
        return False


def _versions() -> list[str]:
    from ._dist._group import version

    libc = platform.libc_ver()
    system = f"{platform.system()} {platform.machine()}" + (f", {libc[0]} {libc[1]}" if libc[0] else "")
    lines = [f"landschaft {version()} (Python {platform.python_version()}, {system})",
             f"  installed in {Path(__file__).resolve().parent}"]
    try:
        from . import _core

        avx2 = _core.cpu_supports_avx2_fma()
        lines.append(f"  compiled core: {Path(_core.__file__).name}, OpenMP {'yes' if _core.openmp else 'NO'}, "
                     f"AVX2+FMA {'yes' if avx2 else 'no'}")
        if not _core.openmp:
            lines.append("  ! built without OpenMP: every tool runs on one thread")
    except ImportError as e:  # pragma: no cover - the import of landschaft would have failed first
        lines.append(f"  ! the compiled core does not load: {e}")
    return lines


def _threads() -> list[str]:
    from ._threads import detect

    found = detect()
    lines = [f"Threads (the default of n_threads): {found.describe()}"]
    given = [f"{v}={os.environ[v]}" for v in VARIABLES[:6] if os.environ.get(v)]
    if given:
        lines.append("  " + ", ".join(given))
    if found.idle():
        lines.append(f"  ! OMP_NUM_THREADS={found.all} leaves {found.idle()} of this job's processors idle (cluster "
                     f"modules often set it for MPI programs): set LANDSCHAFT_NUM_THREADS={found.job_processors}, "
                     "which landschaft reads first")
    return lines


def _job() -> list[str]:
    from ._jobs import array_task, in_job

    scheduler = in_job()
    if scheduler is None:
        return ["Job: none (not inside Slurm, PBS or LSF)"]
    env = os.environ
    job = {"Slurm": env.get("SLURM_JOB_ID"), "PBS": env.get("PBS_JOBID"), "LSF": env.get("LSB_JOBID")}[scheduler]
    found = [f"{scheduler} job {job}" + (f" on {env['SLURMD_NODENAME']}" if env.get("SLURMD_NODENAME") else "")]
    if env.get("SLURM_JOB_NUM_NODES", "1") not in ("", "1"):
        found.append(f"{env['SLURM_JOB_NUM_NODES']} nodes ({env.get('SLURM_JOB_NODELIST', '?')})")
    try:
        task = array_task()
    except ValueError as e:
        found.append(f"! {e}")
    else:
        if task is not None:
            step = f" step {task.step}" if task.step != 1 else ""
            found.append(f"array task {task.task} of {task.count} ({task.first}-{task.last}{step})")
    return ["Job: " + "; ".join(found)]


def _caches() -> list[str]:
    from ._dist._group import config_dir, load_secret
    from ._dist._worker import default_cache_dir
    from ._download import cache_dir

    lines = []
    try:
        root = cache_dir()
    except OSError as e:
        root = None
        lines.append(f"Cache: cannot be created ({e})")
    if root is not None:
        origin = "LANDSCHAFT_CACHE" if os.environ.get("LANDSCHAFT_CACHE") else (
            "XDG_CACHE_HOME" if os.environ.get("XDG_CACHE_HOME") and sys.platform not in ("win32", "darwin")
            else "default")
        lines.append(f"Cache: {root} ({origin}), {'writable' if writable(root) else 'NOT WRITABLE'}")
        if not writable(root) and not os.environ.get("LANDSCHAFT_CACHE"):
            lines.append("  ! set LANDSCHAFT_CACHE to a folder that can be written (on a cluster: the scratch space)")
    try:
        helper = default_cache_dir()
        lines.append(f"  helper files: {helper}")
    except OSError:
        pass
    group = "in a group" if load_secret() is not None else "in no group"
    lines.append(f"Configuration: {config_dir()} ({group} of helpers)")
    return lines


def _network(timeout: float) -> list[str]:
    from ._download import OFFLINE_HINT

    with ThreadPoolExecutor(len(SERVICES)) as pool:
        answers = list(pool.map(lambda s: probe(s[1], timeout), SERVICES))
    lines = ["Network:"]
    for (name, url), problem in zip(SERVICES, answers):
        lines.append(f"  {name:<38} {'ok' if problem is None else 'NO: ' + problem}  ({url.split('/')[2]})")
    if all(a is not None for a in answers):
        lines.append(f"  ! {OFFLINE_HINT}: `landschaft fetch` downloads the data landschaft downloads on first use.")
    return lines


def _display() -> list[str]:
    from ._plotting._env import headless

    why = headless()
    return [f"Display: {'none, ' + why + ': plots are not shown, .save() writes them' if why else 'yes'}"]


def report(network: bool = True, timeout: float = 5.0) -> str:
    """What ``landschaft doctor`` prints."""
    lines = _versions() + _threads() + _job() + _caches()
    if network:
        lines += _network(timeout)
    lines += _display()
    return "\n".join(lines)


def doctor(network: bool = True) -> None:
    """Print what landschaft finds on this computer.

    The threads it uses by default and where that number comes from, the
    batch job (Slurm, PBS, LSF) and job array task it runs in, its cache and
    whether it can be written, whether the servers it downloads from answer,
    whether plots can open a window, and the versions. Also ``landschaft
    doctor`` on the command line: the first thing to run on a cluster, and to
    paste in an issue.

    Parameters
    ----------
    network
        Check the servers too (a few seconds at most).

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.doctor(network=False)        # doctest: +ELLIPSIS
    landschaft ...
    Threads (the default of n_threads): ...
    """
    print(report(network=network), flush=True)
