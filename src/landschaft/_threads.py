# SPDX-License-Identifier: GPL-3.0-or-later
"""Threads of landschaft's parallel code.

Every function that runs in parallel takes ``n_threads`` (and the ones that
run several rasters at once ``n_workers``); ``None`` or 0 means the default,
from the first of these that applies:

1. the count given to :func:`set_default_threads`;
2. ``LANDSCHAFT_NUM_THREADS``;
3. the threads of the dask.distributed worker running the call (its nanny sets
   ``OMP_NUM_THREADS=1`` for the BLAS libraries, which would leave the worker
   idle);
4. ``OMP_NUM_THREADS``;
5. the processors of the batch job (Slurm, PBS, LSF) where the process may run
   on more (a cluster that does not confine its jobs): ``SLURM_CPUS_PER_TASK``,
   else ``SLURM_CPUS_ON_NODE``; ``NCPUS``; ``LSB_DJOB_NUMPROC``;
6. the processors the process may run on (its affinity and, on Linux, the CPU
   quota of its cgroup): all but one on a desktop, so that the machine stays
   responsive, and all of them inside a job or when the process is confined to
   fewer than the machine has.

The compiled functions go through a guard that sets the OpenMP thread count
of the calling thread before each call (the count requested with
:func:`limit`, else the default), so parallel code without a thread argument
of its own follows the same rule, in any Python thread.
"""

from __future__ import annotations

import functools
import math
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NamedTuple

__all__ = ["available_cores", "default_threads", "limit", "resolve", "set_default_threads"]

_default: int | None = None
_local = threading.local()
_warned = False

#: The variable that names the job of each batch scheduler.
JOB_VARIABLES = (("Slurm", "SLURM_JOB_ID"), ("PBS", "PBS_JOBID"), ("LSF", "LSB_JOBID"))
#: The variables of each scheduler that give the processors of the job on this node, in the order read.
JOB_CPU_VARIABLES = {"Slurm": ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"), "PBS": ("NCPUS",),
                     "LSF": ("LSB_DJOB_NUMPROC",)}


class Threads(NamedTuple):
    """What decides the default thread count."""

    threads: int                # the default
    all: int                    # without the processor kept free
    source: str | None          # what gave the count (a variable, "set_default_threads", "dask worker"), or None
    available: int              # processors this process may run on (affinity and cgroup quota)
    machine: int                # processors of the machine
    scheduler: str | None       # Slurm, PBS, LSF, or None outside a job
    job: str | None
    job_processors: int         # the job's processors here (``available`` outside a job)
    reserved: bool              # one processor kept free (a desktop)

    def describe(self) -> str:
        """``"16 threads (16 of 128 processors available to this process; Slurm job 123456)"``."""
        where = []
        if self.source is not None:
            where.append(f"{self.source}={self.all}" if self.source.isupper() else f"{self.source}: {self.all}")
        if self.available < self.machine:
            where.append(f"{self.available} of {self.machine} processors available to this process")
        else:
            where.append(f"{self.machine} processors")
        if self.job is not None:
            where.append(f"{self.scheduler} job {self.job}")
        elif self.reserved:
            where.append("one kept free")
        return f"{self.threads} thread{'s' if self.threads != 1 else ''} ({'; '.join(where)})"

    def idle(self) -> int:
        """The job's processors that an ``OMP_NUM_THREADS`` lower than them leaves idle (0 outside a
        job, or when something else gave the count)."""
        if self.source != "OMP_NUM_THREADS" or self.job is None:
            return 0
        return max(0, self.job_processors - self.all)


def _count(value: str | None) -> int | None:
    """A positive count from a variable ("8"; "8,4" for OpenMP's nested levels: the first; "8(x2)" as
    Slurm writes some), else None."""
    if not value:
        return None
    head = value.split(",")[0].split("(")[0].strip()
    try:
        n = int(head)
    except ValueError:
        return None
    return n if n > 0 else None


def decide(env: Callable[[str], str | None], available: int, machine: int, *, fixed: int | None = None,
           worker: int | None = None) -> Threads:
    """The rule of the module on the variables ``env`` gives (name -> value or None) and the
    processors: pure, for the tests. ``fixed``: the count of :func:`set_default_threads`; ``worker``:
    the threads of the dask.distributed worker running the call."""
    available = max(1, int(available))
    machine = max(available, int(machine))
    scheduler = job = None
    for name, variable in JOB_VARIABLES:
        value = (env(variable) or "").strip()
        if value:
            scheduler, job = name, value
            break
    job_processors, job_variable = available, None
    if scheduler is not None:
        for variable in JOB_CPU_VARIABLES[scheduler]:
            n = _count(env(variable))
            if n is not None:
                if n < available:
                    job_processors, job_variable = n, variable
                break
    source = None
    if fixed is not None:
        count, source = int(fixed), "set_default_threads"
    elif _count(env("LANDSCHAFT_NUM_THREADS")) is not None:
        count, source = _count(env("LANDSCHAFT_NUM_THREADS")), "LANDSCHAFT_NUM_THREADS"
    elif worker:
        count, source = int(worker), "dask worker"
    elif _count(env("OMP_NUM_THREADS")) is not None:
        count, source = _count(env("OMP_NUM_THREADS")), "OMP_NUM_THREADS"
    elif job_variable is not None:
        count, source = job_processors, job_variable
    else:
        count = available
    reserved = source is None and job is None and available >= machine
    threads = max(1, count - 1) if reserved else count
    return Threads(threads, count, source, available, machine, scheduler, job, job_processors, reserved)


def _cgroup_quota(root: Path = Path("/")) -> int | None:
    """The processors the CPU quota of this process's cgroup allows (Linux; cgroup v2 ``cpu.max`` of the
    group and its parents, else v1 ``cpu.cfs_quota_us``), rounded up; None without a quota."""
    try:
        lines = (root / "proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    best = None

    def take(quota: float, period: float) -> None:
        nonlocal best
        if quota > 0 and period > 0:
            n = max(1, math.ceil(quota / period))
            best = n if best is None else min(best, n)

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        path = path.strip().lstrip("/")
        if controllers == "":  # v2: the group and every parent, up to the mount point
            base = root / "sys/fs/cgroup"
            here = base / path if path else base
            while True:
                try:
                    quota, period = (here / "cpu.max").read_text().split()[:2]
                    if quota != "max":
                        take(float(quota), float(period))
                except (OSError, ValueError):
                    pass
                if here == base or base not in here.parents:
                    break
                here = here.parent
        elif "cpu" in controllers.split(","):  # v1
            for mount in ("cpu,cpuacct", "cpu", "cpuacct,cpu"):
                for here in (root / "sys/fs/cgroup" / mount / path, root / "sys/fs/cgroup" / mount):
                    try:
                        quota = float((here / "cpu.cfs_quota_us").read_text())
                        period = float((here / "cpu.cfs_period_us").read_text())
                    except (OSError, ValueError):
                        continue
                    take(quota, period)
                    break
    return best


def _windows_affinity() -> int | None:
    """The processors of this process's affinity mask (Windows; None where it cannot tell, as with
    more than one processor group)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                ctypes.POINTER(ctypes.c_size_t)]
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(process),
                                           ctypes.byref(system)) or not process.value:
        return None
    if bin(system.value).count("1") < (os.cpu_count() or 1):  # several processor groups: the mask is one's
        return None
    return bin(process.value).count("1")


@functools.lru_cache(maxsize=None)
def processors() -> tuple[int, int]:
    """(processors this process may run on, processors of the machine), read once."""
    machine = os.cpu_count() or 1
    n = None
    count = getattr(os, "process_cpu_count", None)
    if count is not None:
        n = count()
    else:
        try:
            n = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            if sys.platform == "win32":
                try:
                    n = _windows_affinity()
                except (OSError, AttributeError):
                    n = None
    n = int(n or machine)
    if sys.platform.startswith("linux"):
        quota = _cgroup_quota()
        if quota is not None:
            n = min(n, quota)
    n = max(1, n)
    return n, max(n, machine)


def available_cores() -> int:
    """Cores this process may run on (its CPU affinity and, on Linux, its cgroup's CPU quota)."""
    return processors()[0]


def _worker_threads() -> int | None:
    """The threads of the dask.distributed worker running this, or None outside one."""
    distributed = sys.modules.get("distributed")
    if distributed is None:
        return None
    try:
        return int(distributed.get_worker().state.nthreads) or None
    except Exception:  # noqa: BLE001 - not in a worker
        return None


_NAMES = ("LANDSCHAFT_NUM_THREADS", "OMP_NUM_THREADS", *(v for _, v in JOB_VARIABLES),
          *(v for vs in JOB_CPU_VARIABLES.values() for v in vs))
_ENCODED: tuple | None = None
_memo: tuple = (None, None)


def _environment() -> tuple:
    """The values of the variables the rule reads, cheaply: every compiled call asks (the encoded keys
    of ``os.environ``'s own dictionary, where CPython has it)."""
    global _ENCODED
    data, encode = getattr(os.environ, "_data", None), getattr(os.environ, "encodekey", None)
    if data is None or encode is None:
        return tuple(map(os.environ.get, _NAMES))
    if _ENCODED is None:
        _ENCODED = tuple(encode(n) for n in _NAMES)
    return tuple(map(data.get, _ENCODED))


def detect() -> Threads:
    """The default thread count now, and where it comes from."""
    global _memo
    key = (_environment(), _default, _worker_threads())
    seen, found = _memo
    if seen == key:
        return found
    available, machine = processors()
    found = decide(os.environ.get, available, machine, fixed=_default, worker=key[2])
    _memo = (key, found)
    return found


def _warn_idle(found: Threads) -> None:
    """Once: a job's processors left idle by an OMP_NUM_THREADS its environment set (cluster modules
    often export OMP_NUM_THREADS=1 for MPI programs)."""
    global _warned
    if _warned or not found.idle():
        return
    _warned = True
    warnings.warn(f"landschaft uses {found.all} thread{'s' if found.all != 1 else ''} because OMP_NUM_THREADS="
                  f"{found.all}, while this {found.scheduler} job may use {found.job_processors} processors. To "
                  f"use them, set LANDSCHAFT_NUM_THREADS={found.job_processors} (read before OMP_NUM_THREADS), "
                  "call ls.set_default_threads() or pass n_threads.", RuntimeWarning, stacklevel=4)


def default_threads() -> int:
    """The number of threads a parallel function uses when not given one.

    The count given to :func:`set_default_threads`, else
    ``LANDSCHAFT_NUM_THREADS``, else ``OMP_NUM_THREADS``, else the processors
    this process may use: all but one on a desktop, so that the computer stays
    responsive, and all of them inside a batch job (Slurm, PBS, LSF), whose
    processors it counts. ``landschaft doctor`` says where the number came
    from.

    Returns
    -------
    int

    Examples
    --------
    >>> import landschaft as ls
    >>> n = ls.default_threads()
    >>> n >= 1
    True
    """
    if _default is not None:
        return _default
    found = detect()
    if found.source == "OMP_NUM_THREADS":
        _warn_idle(found)
    return found.threads


def set_default_threads(n: int | None = None) -> int:
    """Set the default thread count of every parallel function for this process.

    Parameters
    ----------
    n
        Threads to use when a function is not given ``n_threads``; ``None``
        goes back to the automatic count (see :func:`default_threads`).

    Returns
    -------
    int
        The new default.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.set_default_threads(2)
    2
    >>> result = ls.fad(ls.load_raster(ls.example_data("input")))   # 2 threads
    >>> _ = ls.set_default_threads(None)                            # the automatic count again
    """
    global _default
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"the default thread count must be >= 1, got {n}")
    _default = n
    return default_threads()


def resolve(n_threads: int | None) -> int:
    """A thread argument as a count: ``None`` or 0 give :func:`default_threads`."""
    if n_threads is None or n_threads == 0:
        return default_threads()
    n = int(n_threads)
    if n < 0:
        raise ValueError(f"n_threads must be >= 0, got {n_threads}")
    return n


def current() -> int:
    """The count for compiled calls from this thread (see :func:`limit`)."""
    n = getattr(_local, "n", None)
    return n if n is not None else default_threads()


@contextmanager
def limit(n_threads: int | None):
    """Compiled calls from this thread use ``n_threads`` threads inside the block."""
    previous = getattr(_local, "n", None)
    _local.n = resolve(n_threads)
    try:
        yield _local.n
    finally:
        _local.n = previous


def threaded(fn):
    """Give ``fn`` a keyword ``n_threads`` (None / 0: :func:`default_threads`)
    under which every compiled call it makes runs (see :func:`limit`)."""
    import inspect

    @functools.wraps(fn)
    def call(*args, n_threads: int | None = None, **kwargs):
        with limit(n_threads):
            return fn(*args, **kwargs)

    sig = inspect.signature(fn)
    if "n_threads" not in sig.parameters:
        params = list(sig.parameters.values())
        new = inspect.Parameter("n_threads", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="int | None")
        at = next((i for i, p in enumerate(params) if p.kind is inspect.Parameter.VAR_KEYWORD), len(params))
        params.insert(at, new)
        call.__signature__ = sig.replace(parameters=params)
    return call


_SKIP = {"set_num_threads", "num_procs", "max_threads", "cpu_supports_avx2_fma"}


def install(module, set_num_threads) -> None:
    """Guard the functions of a compiled module (once)."""
    if getattr(module, "_landschaft_threads", False):
        return

    def guard(f):
        @functools.wraps(f)
        def call(*args, **kwargs):
            set_num_threads(current())
            return f(*args, **kwargs)

        return call

    for name, obj in list(vars(module).items()):
        if name.startswith("_") or name in _SKIP or isinstance(obj, type) or not callable(obj):
            continue
        setattr(module, name, guard(obj))
    module._landschaft_threads = True
