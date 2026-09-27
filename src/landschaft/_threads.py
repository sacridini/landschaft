# SPDX-License-Identifier: GPL-3.0-or-later
"""Threads of landschaft's parallel code.

Every function that runs in parallel takes ``n_threads`` (and the ones that
run several rasters at once ``n_workers``); ``None`` or 0 means the default:
all the cores the process may use but one, so that the machine stays
responsive. :func:`set_default_threads` changes that default for the whole
process.

The compiled functions go through a guard that sets the OpenMP thread count
of the calling thread before each call (the count requested with
:func:`limit`, else the default), so parallel code without a thread argument
of its own follows the same rule, in any Python thread.
"""

from __future__ import annotations

import functools
import os
import threading
from contextlib import contextmanager

__all__ = ["available_cores", "default_threads", "limit", "resolve", "set_default_threads"]

_default: int | None = None
_local = threading.local()


def available_cores() -> int:
    """Cores this process may run on (its CPU affinity where the system tells it)."""
    count = getattr(os, "process_cpu_count", None)
    if count is not None:
        n = count()
    else:
        try:
            n = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n = os.cpu_count()
    return max(1, int(n or 1))


def default_threads() -> int:
    """The thread count used when a function is not given one: all cores but
    one (at least 1), or the value of :func:`set_default_threads`."""
    return _default if _default is not None else max(1, available_cores() - 1)


def set_default_threads(n: int | None = None) -> int:
    """Set the default thread count of every parallel function (``None``:
    back to all cores but one). Returns the new default."""
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
